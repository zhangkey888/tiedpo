from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import Trainer
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from .loss import compute_tie_dpo_loss, get_batch_logps, get_sft_loss_per_sample

try:
    import deepspeed
except ImportError:
    deepspeed = None


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: int) -> torch.Tensor:
    if tensor.size(1) >= length:
        return tensor
    pad_size = list(tensor.shape)
    pad_size[1] = length - tensor.size(1)
    pad_tensor = torch.full(pad_size, pad_value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad_tensor], dim=1)


class Qwen3VLTieDPOTrainer(Trainer):
    def __init__(
        self,
        *args,
        processor,
        ref_model=None,
        dpo_alpha: float = 1.0,
        beta: float = 0.1,
        gamma: float = 0.03,
        sft_loss_mode: Literal["strict_only", "tie_symmetric"] = "tie_symmetric",
        lambda_tie: float = 1.0,
        tie_margin: float = 0.0,
        label_smoothing: float = 0.0,
        reference_free: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.processor.tokenizer.padding_side = "right"
        self.ref_model = ref_model
        self.dpo_alpha = dpo_alpha
        self.beta = beta
        self.gamma = gamma
        self.sft_loss_mode = sft_loss_mode
        self.lambda_tie = lambda_tie
        self.tie_margin = tie_margin
        self.label_smoothing = label_smoothing
        self.reference_free = reference_free
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        if not hasattr(self, "accelerator"):
            raise AttributeError("Trainer does not have an accelerator object.")

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    def _prepare_deepspeed(self, model: nn.Module):
        if deepspeed is None:
            raise ImportError("DeepSpeed is required but not available in the current environment.")

        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None and hasattr(model, "config"):
            hidden_sizes = getattr(model.config, "hidden_sizes", None)
            hidden_size = max(hidden_sizes) if hidden_sizes else getattr(model.config, "hidden_size", None)
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

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Any],
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        concatenated_batch: Dict[str, Any] = {}
        max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for key in ("chosen_input_ids", "chosen_labels", "chosen_attention_mask"):
            tensor = batch[key]
            if key.endswith("labels"):
                pad_value = label_pad_token_id
            elif key.endswith("attention_mask"):
                pad_value = 0
            else:
                pad_value = padding_value
            concatenated_key = key.replace("chosen_", "concatenated_")
            concatenated_batch[concatenated_key] = pad_to_length(tensor, max_length, pad_value).to(device=device)

        for key in ("rejected_input_ids", "rejected_labels", "rejected_attention_mask"):
            tensor = batch[key]
            if key.endswith("labels"):
                pad_value = label_pad_token_id
            elif key.endswith("attention_mask"):
                pad_value = 0
            else:
                pad_value = padding_value
            concatenated_key = key.replace("rejected_", "concatenated_")
            concatenated_batch[concatenated_key] = torch.cat(
                [
                    concatenated_batch[concatenated_key],
                    pad_to_length(tensor, max_length, pad_value),
                ],
                dim=0,
            ).to(device=device)

        for key, value in batch.items():
            if key.startswith(("chosen_", "rejected_", "concatenated_")):
                continue
            if key in {"is_tie", "is_ep_tie", "idx"}:
                concatenated_batch[key] = value.to(device=device) if torch.is_tensor(value) else value
                continue
            if torch.is_tensor(value):
                concatenated_batch[f"concatenated_{key}"] = torch.cat([value, value], dim=0).to(device=device)
            else:
                concatenated_batch[f"concatenated_{key}"] = value + value
        return concatenated_batch

    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Any]):
        concatenated_batch = self.concatenated_inputs(
            batch,
            label_pad_token_id=-100,
            padding_value=self.processor.tokenizer.pad_token_id,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_inputs = {
            "input_ids": concatenated_batch["concatenated_input_ids"],
            "attention_mask": concatenated_batch["concatenated_attention_mask"],
            "labels": concatenated_batch["concatenated_labels"],
        }
        for key, value in concatenated_batch.items():
            if not key.startswith("concatenated_"):
                continue
            raw_key = key.replace("concatenated_", "")
            if raw_key in {"input_ids", "attention_mask", "labels"}:
                continue
            model_inputs[raw_key] = value

        outputs = model(**model_inputs)
        all_logits = outputs.logits.to(torch.float32)
        all_labels = concatenated_batch["concatenated_labels"]
        all_logps = get_batch_logps(all_logits, all_labels)

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]
        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]
        chosen_labels = all_labels[:len_chosen]
        rejected_labels = all_labels[len_chosen:]
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_labels, rejected_labels

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, List[Any]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        device = self.accelerator.device
        (
            chosen_policy_logps,
            rejected_policy_logps,
            chosen_policy_logits,
            rejected_policy_logits,
            chosen_labels,
            rejected_labels,
        ) = self.concatenated_forward(model, batch)

        if self.reference_free:
            chosen_ref_logps = torch.zeros_like(chosen_policy_logps)
            rejected_ref_logps = torch.zeros_like(rejected_policy_logps)
        elif "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
            chosen_ref_logps = batch["reference_chosen_logps"].to(device)
            rejected_ref_logps = batch["reference_rejected_logps"].to(device)
        else:
            ref_device = next(self.ref_model.parameters()).device
            if ref_device != device:
                self.ref_model.to(device)
            with torch.no_grad():
                chosen_ref_logps, rejected_ref_logps = self.concatenated_forward(self.ref_model, batch)[:2]

        is_tie_tensor = batch["is_tie"].to(device).bool() if torch.is_tensor(batch["is_tie"]) else torch.tensor(batch["is_tie"], device=device, dtype=torch.bool)

        loss_dict = compute_tie_dpo_loss(
            policy_chosen_logps=chosen_policy_logps,
            policy_rejected_logps=rejected_policy_logps,
            reference_chosen_logps=chosen_ref_logps,
            reference_rejected_logps=rejected_ref_logps,
            is_tie=is_tie_tensor,
            beta=self.beta,
            label_smoothing=self.label_smoothing,
            tie_margin=self.tie_margin,
            reference_free=self.reference_free,
        )

        batch_size = chosen_policy_logps.shape[0]
        strict_mask = ~is_tie_tensor
        tie_mask = is_tie_tensor
        is_ep_tie = batch["is_ep_tie"].to(device).bool() if torch.is_tensor(batch["is_ep_tie"]) else torch.tensor(batch["is_ep_tie"], device=device, dtype=torch.bool)

        dpo_loss = self.dpo_alpha * loss_dict["strict_loss"] / batch_size
        tie_reg_loss = self.lambda_tie * loss_dict["tie_loss"] / batch_size

        chosen_sft_per_sample = get_sft_loss_per_sample(chosen_policy_logits, chosen_labels)
        rejected_sft_per_sample = get_sft_loss_per_sample(rejected_policy_logits, rejected_labels)

        sft_strict = torch.tensor(0.0, device=device)
        sft_tie = torch.tensor(0.0, device=device)
        if strict_mask.any():
            sft_strict = chosen_sft_per_sample[strict_mask].mean()
        if tie_mask.any():
            if self.sft_loss_mode == "tie_symmetric":
                sft_tie_chosen = chosen_sft_per_sample[tie_mask].mean()
                sft_tie_rejected = rejected_sft_per_sample[tie_mask].mean()
                sft_tie = 0.5 * (sft_tie_chosen + sft_tie_rejected)
            elif self.sft_loss_mode == "strict_only":
                sft_tie = torch.tensor(0.0, device=device)
            else:
                raise ValueError(f"Unsupported sft_loss_mode: {self.sft_loss_mode}")

        sft_loss = self.gamma * (sft_strict + sft_tie)
        loss = dpo_loss + tie_reg_loss + sft_loss

        chosen_rewards = self.beta * (chosen_policy_logps - chosen_ref_logps).detach()
        rejected_rewards = self.beta * (rejected_policy_logps - rejected_ref_logps).detach()
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        margins = loss_dict["margins"]
        abs_margins = margins.abs()
        tie_threshold = max(float(self.tie_margin), 1e-6)

        def to_float(value):
            if isinstance(value, torch.Tensor):
                return float(value.detach().to(dtype=torch.float32).item())
            return float(value)

        def safe_mean(numerator, denominator):
            denominator = max(float(denominator), 1.0)
            return to_float(numerator) / denominator

        prefix = "eval_" if train_eval == "eval" else ""
        total_count = float(chosen_policy_logps.shape[0])
        n_strict = float(strict_mask.sum().item())
        n_tie = float(tie_mask.sum().item())
        n_ep_tie = float(is_ep_tie.sum().item())

        strict_acc_num = (margins[strict_mask] > 0).float().sum() if strict_mask.any() else 0.0
        strict_margin_num = margins[strict_mask].sum() if strict_mask.any() else 0.0
        tie_margin_num = abs_margins[tie_mask].sum() if tie_mask.any() else 0.0
        ep_tie_acc_num = (abs_margins[is_ep_tie] <= tie_threshold).float().sum() if is_ep_tie.any() else 0.0
        ep_tie_margin_num = abs_margins[is_ep_tie].sum() if is_ep_tie.any() else 0.0
        tie_reward_l1_num = abs_margins[tie_mask].sum() if tie_mask.any() else 0.0
        tie_chosen_num = chosen_rewards[tie_mask].sum() if tie_mask.any() else 0.0
        tie_rejected_num = rejected_rewards[tie_mask].sum() if tie_mask.any() else 0.0
        strict_margin_global = safe_mean(strict_margin_num, n_strict)
        tie_margin_global = safe_mean(tie_margin_num, n_tie)

        pred_tie_mask = abs_margins <= tie_threshold
        pred_strict_mask = ~pred_tie_mask
        strict_tp = (pred_strict_mask & strict_mask).float().sum()
        strict_fp = (pred_strict_mask & tie_mask).float().sum()
        strict_fn = (pred_tie_mask & strict_mask).float().sum()
        tie_tp = (pred_tie_mask & tie_mask).float().sum()
        tie_fp = (pred_tie_mask & strict_mask).float().sum()
        tie_fn = (pred_strict_mask & tie_mask).float().sum()

        metrics = {
            f"{prefix}losses/strict_dpo": safe_mean(loss_dict["strict_loss"], n_strict),
            f"{prefix}losses/tie_reg": safe_mean(loss_dict["tie_loss"], n_tie),
            f"{prefix}losses/sft": safe_mean(sft_loss.detach(), 1.0),
            f"{prefix}losses/sft_strict": safe_mean((self.gamma * sft_strict).detach(), 1.0),
            f"{prefix}losses/sft_tie": safe_mean((self.gamma * sft_tie).detach(), 1.0),
            f"{prefix}losses/total": safe_mean(loss.detach(), 1.0),
            f"{prefix}rewards/chosen": safe_mean(chosen_rewards.sum(), total_count),
            f"{prefix}rewards/rejected": safe_mean(rejected_rewards.sum(), total_count),
            f"{prefix}rewards/accuracies": safe_mean(reward_accuracies.sum(), total_count),
            f"{prefix}rewards/pooled_accuracy": safe_mean(reward_accuracies.sum(), total_count),
            f"{prefix}rewards/margins": safe_mean(margins.sum(), total_count),
            f"{prefix}logps/chosen": safe_mean(chosen_policy_logps.sum(), total_count),
            f"{prefix}logps/rejected": safe_mean(rejected_policy_logps.sum(), total_count),
            f"{prefix}ref_logps/chosen": safe_mean(chosen_ref_logps.sum(), total_count),
            f"{prefix}ref_logps/rejected": safe_mean(rejected_ref_logps.sum(), total_count),
            f"{prefix}rewards/strict_accuracy": safe_mean(strict_acc_num, n_strict),
            f"{prefix}rewards/strict_margin": safe_mean(strict_margin_num, n_strict),
            f"{prefix}rewards/tie_reward_l1": safe_mean(tie_reward_l1_num, n_tie),
            f"{prefix}rewards/tie_chosen": safe_mean(tie_chosen_num, n_tie),
            f"{prefix}rewards/tie_rejected": safe_mean(tie_rejected_num, n_tie),
            f"{prefix}tiebench/StrictAcc": safe_mean(strict_acc_num, n_strict),
            f"{prefix}tiebench/StrictPrecision": safe_mean(strict_tp, strict_tp + strict_fp),
            f"{prefix}tiebench/StrictRecall": safe_mean(strict_tp, strict_tp + strict_fn),
            f"{prefix}tiebench/StrictF1": safe_mean(2 * strict_tp, 2 * strict_tp + strict_fp + strict_fn),
            f"{prefix}tiebench/TieAcc": safe_mean((abs_margins[tie_mask] <= tie_threshold).float().sum() if tie_mask.any() else 0.0, n_tie),
            f"{prefix}tiebench/TiePrecision": safe_mean(tie_tp, tie_tp + tie_fp),
            f"{prefix}tiebench/TieRecall": safe_mean(tie_tp, tie_tp + tie_fn),
            f"{prefix}tiebench/TieF1": safe_mean(2 * tie_tp, 2 * tie_tp + tie_fp + tie_fn),
            f"{prefix}tiebench/TieMargin": tie_margin_global,
            f"{prefix}tiebench/StrictMargin": strict_margin_global,
            f"{prefix}tiebench/CalibGap": strict_margin_global - tie_margin_global,
            f"{prefix}tiebench/EPTieAcc": safe_mean(ep_tie_acc_num, n_ep_tie),
            f"{prefix}tiebench/EPTieMargin": safe_mean(ep_tie_margin_num, n_ep_tie),
            f"{prefix}data/n_tie": n_tie,
            f"{prefix}data/n_strict": n_strict,
            f"{prefix}data/n_ep_tie": n_ep_tie,
        }
        return loss, metrics

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(float(value))

    def _sync_scalar(self, value: float, op: dist.ReduceOp = dist.ReduceOp.SUM) -> float:
        tensor = torch.tensor(float(value), device=self.accelerator.device, dtype=torch.float32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=op)
        return float(tensor.item())

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        train_eval = "train" if "loss" in logs else "eval"
        stored_metrics = self._stored_metrics.get(train_eval, {})
        for key, values in stored_metrics.items():
            if not values:
                continue
            local_total = float(sum(values))
            local_count = float(len(values))
            global_total = self._sync_scalar(local_total)
            global_count = self._sync_scalar(local_count)
            logs[key] = global_total / max(global_count, 1.0)
        if train_eval in self._stored_metrics:
            del self._stored_metrics[train_eval]
        if start_time is not None:
            return super().log(logs, start_time)
        return super().log(logs)

    def compute_loss(
        self,
        model,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        context = nullcontext
        with context():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")
        self.store_metrics(metrics, train_eval="train")
        if return_outputs:
            return loss, metrics
        return loss

    def prediction_step(
        self,
        model,
        inputs: Dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        with torch.no_grad():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")
        self.store_metrics(metrics, train_eval="eval")
        return loss.detach(), None, None
