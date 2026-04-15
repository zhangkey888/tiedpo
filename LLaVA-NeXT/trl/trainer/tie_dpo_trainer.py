# DPO + Tie Regularization Trainer
# Based on DPO Authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn 2023
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

from ..import_utils import is_peft_available, is_wandb_available
from ..models import PreTrainedModelWrapper, create_reference_model
from .utils import (
    DPODataCollatorWithPadding,
    disable_dropout_in_model,
    pad_to_length,
    peft_module_casting_to_bf16,
    trl_sanitze_kwargs_for_tagging,
)


if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed

from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled


class TieDPODataCollatorWithPadding:
    r"""
    Data collator for TieDPO that supports three-way labels: "better_a", "better_b", "tie".

    Input format per sample:
        {
            "prompt": ...,
            "chosen": ...,         # response_a (or better one for strict pairs)
            "rejected": ...,       # response_b (or worse one for strict pairs)
            "is_tie": True/False,  # True means this is a tie pair
        }

    For tie pairs, "chosen" and "rejected" are just placeholders (response_a and response_b),
    without semantic ordering.
    """

    tokenizer: PreTrainedTokenizerBase
    pad_token_id: int = 0
    label_pad_token_id: int = -100
    is_encoder_decoder: Optional[bool] = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        padded_batch = {}
        for k in features[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                if self.is_encoder_decoder:
                    to_pad = [torch.LongTensor(ex[k]) for ex in features]
                    if (k.startswith("prompt")) and (k.endswith("input_ids")):
                        padding_value = self.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif (k.startswith("chosen")) or (k.startswith("rejected")) or ("decoder" in k):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    from torch.nn.utils.rnn import pad_sequence
                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                else:
                    from torch.nn.utils.rnn import pad_sequence
                    if "prompt" in k:
                        to_pad = [torch.LongTensor(ex[k][::-1]) for ex in features]
                    else:
                        to_pad = [torch.LongTensor(ex[k]) for ex in features]
                    if k.endswith("_input_ids"):
                        padding_value = self.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                    if "prompt" in k:
                        padded_batch[k] = padded_batch[k].flip(dims=[1])
            elif k.endswith("_logps"):
                padded_batch[k] = torch.tensor([ex[k] for ex in features])
            elif k == "is_tie":
                padded_batch[k] = torch.tensor([ex[k] for ex in features], dtype=torch.bool)
            else:
                padded_batch[k] = [ex[k] for ex in features]

        return padded_batch


class TieDPOTrainer(Trainer):
    r"""
    TieDPO Trainer implementing DPO + Tie Regularization.

    Loss formulation:
        For strict pairs (better_a / better_b):
            L_strict = -log sigma(beta * delta)   or   -log sigma(-beta * delta)
        For tie pairs:
            L_tie = max(0, |beta * delta| - tie_margin)^2

        where delta = (log pi(a|x) - log pi(b|x)) - (log pi_ref(a|x) - log pi_ref(b|x))

    Args:
        model: The model to train.
        ref_model: Reference model for implicit reward.
        beta: DPO temperature parameter.
        dpo_alpha: Scaling factor for DPO loss.
        gamma: SFT auxiliary loss weight.
        lambda_tie: Weight for tie regularization loss.
        tie_margin: Margin m for tie loss: max(0, |beta*delta| - m)^2. Set 0 to use plain squared loss.
        loss_type: "sigmoid" only (tie variant only supports sigmoid base).
        label_smoothing: Label smoothing for strict pairs.
    """

    _tag_names = ["trl", "tie_dpo"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        dpo_alpha: float = 1.0,
        beta: float = 0.1,
        gamma: float = 0.1,
        lambda_tie: float = 1.0,
        tie_margin: float = 0.0,
        label_smoothing: float = 0,
        loss_type: Literal["sigmoid"] = "sigmoid",
        args: Optional[TrainingArguments] = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: Optional[int] = None,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        precompute_ref_log_probs: bool = False,
        dataset_num_proc: Optional[int] = None,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
        model_adapter_name: Optional[str] = None,
        ref_adapter_name: Optional[str] = None,
        reference_free: bool = False,
    ):
        if model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs to the TieDPOTrainer. But your model is already instantiated.")

        if ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError("You passed ref_model_kwargs to the TieDPOTrainer. But your ref_model is already instantiated.")

        if isinstance(model, str):
            warnings.warn("You passed a model_id to the TieDPOTrainer. This will automatically create an `AutoModelForCausalLM` for you.")
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if isinstance(ref_model, str):
            warnings.warn("You passed a ref model_id to the TieDPOTrainer. This will automatically create an `AutoModelForCausalLM`.")
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_init_kwargs)

        self._peft_has_been_casted_to_bf16 = False

        if generate_during_eval and not is_wandb_available():
            raise ValueError("`generate_during_eval=True` requires Weights and Biases to be installed.")

        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif is_encoder_decoder is None:
            raise ValueError("When no model is provided, you need to pass the parameter is_encoder_decoder.")
        else:
            self.is_encoder_decoder = is_encoder_decoder

        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        self.model_adapter_name = model_adapter_name
        self.ref_adapter_name = ref_adapter_name
        self.reference_free = reference_free

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model or precompute_ref_log_probs:
            self.ref_model = None
        else:
            if is_deepspeed_zero3_enabled():
                self.ref_model = AutoModelForCausalLM.from_pretrained(model)
            else:
                self.ref_model = create_reference_model(model)

        if tokenizer is None:
            raise ValueError("tokenizer must be specified to tokenize a TieDPO dataset.")
        if max_length is None:
            warnings.warn("`max_length` is not set in the TieDPOTrainer's init, it will default to `512`.", UserWarning)
            max_length = 512
        if max_prompt_length is None:
            warnings.warn("`max_prompt_length` is not set in the TieDPOTrainer's init, it will default to `128`.", UserWarning)
            max_prompt_length = 128

        if max_target_length is None and self.is_encoder_decoder:
            warnings.warn("When using an encoder decoder architecture, you should set `max_target_length`, it will default to `128`.", UserWarning)
            max_target_length = 128

        if data_collator is None:
            data_collator = TieDPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                warnings.warn(
                    "When using TieDPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments, we have set it for you.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        if disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.max_length = max_length
        self.generate_during_eval = generate_during_eval
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value if padding_value is not None else tokenizer.pad_token_id
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = truncation_mode
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self.precompute_ref_log_probs = precompute_ref_log_probs
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False

        self.dpo_alpha = dpo_alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_tie = lambda_tie
        self.tie_margin = tie_margin
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type

        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self.dataset_num_proc = dataset_num_proc

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        if not hasattr(self, "accelerator"):
            raise AttributeError("Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`.")

        if self.is_deepspeed_enabled:
            if self.accelerator.state.deepspeed_plugin.zero_stage == 3 and self.precompute_ref_log_probs:
                raise ValueError("You cannot use `precompute_ref_log_probs=True` with Deepspeed ZeRO-3.")

        if self.ref_model is None:
            if not (self.is_peft_model or self.precompute_ref_log_probs):
                raise ValueError("No reference model and model is not a Peft model. Try setting `precompute_ref_log_probs=True`")
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = max(model.config.hidden_sizes) if getattr(model.config, "hidden_sizes", None) else getattr(model.config, "hidden_size", None)
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    @contextmanager
    def null_ref_context(self):
        if self.is_peft_model and hasattr(self.model, "disable_adapter"):
            with self.model.disable_adapter():
                yield
        else:
            yield

    def build_tokenized_answer(self, prompt, answer):
        full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
        prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids):]
        answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids):]

        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])
        full_input_ids = np.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError("Prompt input ids and answer input ids should have the same length.")

        response_token_ids_start_idx = len(prompt_input_ids)
        if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError("Prompt input ids and attention mask should have the same length.")

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

        return dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
        )

    def tokenize_row(self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> Dict:
        """Tokenize a single row. Supports is_tie field for tie-regularization."""
        batch = {}
        prompt = feature["prompt"]
        chosen = feature["chosen"]
        rejected = feature["rejected"]
        is_tie = feature.get("is_tie", False)

        if not self.is_encoder_decoder:
            if not isinstance(prompt, str):
                raise ValueError(f"prompt should be an str but got {type(prompt)}")
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
            prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

            if not isinstance(chosen, str):
                raise ValueError(f"chosen should be an str but got {type(chosen)}")
            chosen_tokens = self.build_tokenized_answer(prompt, chosen)

            if not isinstance(rejected, str):
                raise ValueError(f"rejected should be an str but got {type(rejected)}")
            rejected_tokens = self.build_tokenized_answer(prompt, rejected)

            prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])
            chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
            rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
            prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

            for k, v in prompt_tokens.items():
                prompt_tokens[k] = v[:prompt_len_input_ids]

            # Truncation
            longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

            for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
                if len(answer_tokens["prompt_input_ids" if "prompt_input_ids" in answer_tokens else "input_ids"]) + longer_response_length > self.max_length:
                    if self.truncation_mode == "keep_start":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            if k in answer_tokens:
                                answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                    elif self.truncation_mode == "keep_end":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            if k in answer_tokens:
                                answer_tokens[k] = answer_tokens[k][-self.max_prompt_length:]
                    else:
                        raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

            for answer_tokens in [chosen_tokens, rejected_tokens]:
                if len(prompt_tokens["prompt_input_ids"]) + len(answer_tokens["input_ids"]) > self.max_length:
                    answer_tokens["input_ids"] = answer_tokens["input_ids"][: self.max_length - self.max_prompt_length]
                    answer_tokens["attention_mask"] = answer_tokens["attention_mask"][: self.max_length - self.max_prompt_length]

            chosen_sequence_tokens = {k: prompt_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]}
            rejected_sequence_tokens = {k: prompt_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]}
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
            chosen_sequence_tokens["labels"][: len(prompt_tokens["prompt_input_ids"])] = [self.label_pad_token_id] * len(prompt_tokens["prompt_input_ids"])
            rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
            rejected_sequence_tokens["labels"][: len(prompt_tokens["prompt_input_ids"])] = [self.label_pad_token_id] * len(prompt_tokens["prompt_input_ids"])

            for k, toks in {"chosen": chosen_sequence_tokens, "rejected": rejected_sequence_tokens, "": prompt_tokens}.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}_{type_key}" if k else type_key] = tokens

        else:
            chosen_tokens = self.tokenizer(chosen, truncation=True, max_length=self.max_target_length, add_special_tokens=True)
            rejected_tokens = self.tokenizer(rejected, truncation=True, max_length=self.max_target_length, add_special_tokens=True)
            prompt_tokens = self.tokenizer(prompt, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True)

            batch["chosen_labels"] = chosen_tokens["input_ids"]
            batch["rejected_labels"] = rejected_tokens["input_ids"]
            batch["prompt_input_ids"] = prompt_tokens["input_ids"]
            batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

            if model is not None and hasattr(model, "prepare_decoder_input_ids_from_labels"):
                batch["rejected_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(labels=torch.tensor(batch["rejected_labels"]))
                batch["chosen_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(labels=torch.tensor(batch["chosen_labels"]))

        batch["is_tie"] = is_tie
        return batch

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        concatenated_batch = {}

        if is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value).to(device=device)

        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            concatenated_batch["concatenated_attention_mask"] = batch["prompt_attention_mask"].repeat(2, 1).to(device=device)

        concatenated_batch["concatenated_images"] = batch["images"] * 2
        concatenated_batch["image_sizes"] = batch["image_sizes"] * 2
        concatenated_batch["modalities"] = batch["modalities"] * 2
        return concatenated_batch

    def tie_dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        is_tie: torch.BoolTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute TieDPO loss.

        delta = (log pi(a|x) - log pi(b|x)) - (log pi_ref(a|x) - log pi_ref(b|x))

        For strict pairs (is_tie=False):
            - "chosen" is the better response (better_a):  loss = -log sigma(beta * delta)
            - "rejected" is the worse response (better_b): same formula, as chosen > rejected by convention

        For tie pairs (is_tie=True):
            - loss = max(0, |beta * delta| - tie_margin)^2

        Total: L = dpo_alpha * L_strict + lambda_tie * L_tie
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if self.reference_free:
            ref_logratios = torch.zeros_like(pi_logratios)
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps

        pi_logratios = pi_logratios.to(self.accelerator.device)
        ref_logratios = ref_logratios.to(self.accelerator.device)
        is_tie = is_tie.to(self.accelerator.device)

        delta = pi_logratios - ref_logratios
        scaled_delta = self.beta * delta

        strict_mask = ~is_tie
        tie_mask = is_tie

        losses = torch.zeros_like(delta)

        if strict_mask.any():
            strict_losses = (
                -F.logsigmoid(scaled_delta[strict_mask]) * (1 - self.label_smoothing)
                - F.logsigmoid(-scaled_delta[strict_mask]) * self.label_smoothing
            )
            losses[strict_mask] = strict_losses

        if tie_mask.any():
            abs_scaled = scaled_delta[tie_mask].abs()
            if self.tie_margin > 0:
                tie_losses = torch.clamp(abs_scaled - self.tie_margin, min=0.0) ** 2
            else:
                tie_losses = abs_scaled ** 2
            losses[tie_mask] = tie_losses

        chosen_rewards = self.beta * (policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device)).detach()
        rejected_rewards = self.beta * (policy_rejected_logps.to(self.accelerator.device) - reference_rejected_logps.to(self.accelerator.device)).detach()

        strict_loss = losses[strict_mask].mean() if strict_mask.any() else torch.tensor(0.0, device=self.accelerator.device)
        tie_loss = losses[tie_mask].mean() if tie_mask.any() else torch.tensor(0.0, device=self.accelerator.device)

        return strict_loss, tie_loss, chosen_rewards, rejected_rewards

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        labels[labels == label_pad_token_id] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def get_sft_loss(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        return loss

    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        all_logits, new_labels = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            labels=concatenated_batch["concatenated_labels"],
            images=concatenated_batch["concatenated_images"],
            image_sizes=concatenated_batch["image_sizes"],
            modalities=concatenated_batch["modalities"],
            use_cache=False,
            dpo_forward=True,
        )
        all_logits = all_logits.to(torch.float32)
        all_logps = self.get_batch_logps(
            all_logits,
            new_labels,
            average_log_prob=False,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]
        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]
        chosen_labels = new_labels[:len_chosen]
        rejected_labels = new_labels[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_labels, rejected_labels)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute TieDPO loss and metrics for a batch."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            chosen_labels,
            rejected_labels,
        ) = self.concatenated_forward(model, batch)

        if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                        ) = self.concatenated_forward(self.model, batch)[:2]
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                    ) = self.concatenated_forward(self.ref_model, batch)[:2]

        is_tie = batch.get("is_tie", torch.zeros(policy_chosen_logps.shape[0], dtype=torch.bool))
        if isinstance(is_tie, list):
            is_tie = torch.tensor(is_tie, dtype=torch.bool)

        strict_loss, tie_loss, chosen_rewards, rejected_rewards = self.tie_dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            is_tie,
        )

        dpo_loss = self.dpo_alpha * strict_loss
        tie_reg_loss = self.lambda_tie * tie_loss
        sft_loss = self.gamma * self.get_sft_loss(policy_chosen_logits, chosen_labels)

        losses = dpo_loss + tie_reg_loss + sft_loss

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        def all_gather_tensor(tensor):
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                tensor = tensor.detach()
                gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(gathered_tensor, tensor)
                tensor = torch.cat(gathered_tensor, dim=0)
            return tensor

        chosen_rewards = all_gather_tensor(chosen_rewards)
        rejected_rewards = all_gather_tensor(rejected_rewards)
        reward_accuracies = all_gather_tensor(reward_accuracies)
        policy_chosen_logps = all_gather_tensor(policy_chosen_logps)
        policy_rejected_logps = all_gather_tensor(policy_rejected_logps)
        reference_chosen_logps = all_gather_tensor(reference_chosen_logps)
        reference_rejected_logps = all_gather_tensor(reference_rejected_logps)

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}losses/strict_dpo"] = strict_loss.cpu()
        metrics[f"{prefix}losses/tie_reg"] = tie_loss.cpu()
        metrics[f"{prefix}losses/sft"] = sft_loss.detach().cpu()
        metrics[f"{prefix}losses/total"] = losses.detach().cpu()
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}ref_logps/chosen"] = reference_chosen_logps.mean().cpu()
        metrics[f"{prefix}ref_logps/rejected"] = reference_rejected_logps.mean().cpu()

        n_tie = is_tie.sum().item()
        n_strict = (~is_tie).sum().item()
        metrics[f"{prefix}data/n_tie"] = float(n_tie)
        metrics[f"{prefix}data/n_strict"] = float(n_strict)

        return losses, metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with torch.no_grad(), prediction_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        return (loss.detach(), None, None)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        initial_output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
        return initial_output

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        train_eval = "train" if "loss" in logs else "eval"
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        if start_time is not None:
            return super().log(logs, start_time)
        return super().log(logs)

    @wraps(Trainer.push_to_hub)
    def push_to_hub(self, commit_message: Optional[str] = "End of training", blocking: bool = True, **kwargs) -> str:
        kwargs = trl_sanitze_kwargs_for_tagging(model=self.model, tag_names=self._tag_names, kwargs=kwargs)
        return super().push_to_hub(commit_message=commit_message, blocking=blocking, **kwargs)
