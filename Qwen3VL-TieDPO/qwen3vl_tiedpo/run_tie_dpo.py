#!/usr/bin/env python3
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    HfArgumentParser,
    TrainerCallback,
)

from .data import DataCollatorForQwen3VLTieDPODataset, Qwen3VLTieDPODataset
from .trainer import Qwen3VLTieDPOTrainer


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen3-VL-8B-Instruct")
    ref_model_name_or_path: Optional[str] = field(default=None)
    lora_weight_path: Optional[str] = field(default=None)
    attn_implementation: str = field(default="sdpa")
    torch_dtype: str = field(default="bfloat16")
    max_pixels: Optional[int] = field(default=602112)
    min_pixels: Optional[int] = field(default=12544)
    use_lora: bool = field(default=True)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.0)
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    )


@dataclass
class DataArguments:
    train_data_path: str = field(default="")
    eval_data_path: Optional[str] = field(default=None)


@dataclass
class TieDPOArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = field(default=False)
    report_to: str = field(default="none")
    model_max_length: int = field(default=32768)
    max_prompt_length: int = field(default=4096)
    dpo_alpha: float = field(default=1.0)
    beta: float = field(default=0.1)
    gamma: float = field(default=0.03)
    sft_loss_mode: str = field(default="tie_symmetric")
    lambda_tie: float = field(default=2.0)
    tie_margin: float = field(default=0.1)
    label_smoothing: float = field(default=0.0)
    reference_free: bool = field(default=False)


class MetricLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.is_world_process_zero:
            keys = [k for k in logs if "losses/" in k or "tiebench/" in k]
            if keys:
                summary = {k: logs[k] for k in sorted(keys)}
                print(summary, flush=True)


def resolve_dtype(dtype_name: str):
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = dtype_name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {dtype_name}")
    return mapping[key]


def maybe_apply_lora(model, model_args: ModelArguments):
    if not model_args.use_lora:
        return model
    target_modules = [name.strip() for name in model_args.lora_target_modules.split(",") if name.strip()]
    peft_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def load_model(model_path: str, model_args: ModelArguments):
    torch_dtype = resolve_dtype(model_args.torch_dtype)
    return AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
    )


def maybe_load_lora_weights(model, lora_weight_path: Optional[str]):
    if not lora_weight_path:
        return model
    return PeftModel.from_pretrained(model, lora_weight_path, is_trainable=False)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TieDPOArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.do_train and not data_args.train_data_path:
        raise ValueError("--train_data_path is required")
    if training_args.do_eval and not data_args.eval_data_path:
        raise ValueError("--eval_data_path is required when --do_eval True")

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    os.makedirs(training_args.output_dir, exist_ok=True)

    processor_kwargs = {}
    if model_args.max_pixels is not None:
        processor_kwargs["max_pixels"] = model_args.max_pixels
    if model_args.min_pixels is not None:
        processor_kwargs["min_pixels"] = model_args.min_pixels
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, **processor_kwargs)
    processor.tokenizer.model_max_length = training_args.model_max_length
    policy_model = load_model(model_args.model_name_or_path, model_args)
    if model_args.lora_weight_path:
        policy_model = maybe_load_lora_weights(policy_model, model_args.lora_weight_path)
    elif training_args.do_train:
        policy_model = maybe_apply_lora(policy_model, model_args)
        if training_args.gradient_checkpointing:
            policy_model.config.use_cache = False
            if hasattr(policy_model, "gradient_checkpointing_enable"):
                policy_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs
                )
            if hasattr(policy_model, "enable_input_require_grads"):
                policy_model.enable_input_require_grads()

    ref_model = None
    if not training_args.reference_free:
        ref_path = model_args.ref_model_name_or_path or model_args.model_name_or_path
        ref_model = load_model(ref_path, model_args)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False

    train_dataset = (
        Qwen3VLTieDPODataset(
            data_args.train_data_path,
            processor=processor,
            model_max_length=training_args.model_max_length,
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        Qwen3VLTieDPODataset(
            data_args.eval_data_path,
            processor=processor,
            model_max_length=training_args.model_max_length,
        )
        if data_args.eval_data_path
        else None
    )

    trainer = Qwen3VLTieDPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        processor=processor,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForQwen3VLTieDPODataset(tokenizer=processor.tokenizer),
        dpo_alpha=training_args.dpo_alpha,
        beta=training_args.beta,
        gamma=training_args.gamma,
        sft_loss_mode=training_args.sft_loss_mode,
        lambda_tie=training_args.lambda_tie,
        tie_margin=training_args.tie_margin,
        label_smoothing=training_args.label_smoothing,
        reference_free=training_args.reference_free,
        callbacks=[MetricLogCallback()],
    )

    if training_args.do_train:
        trainer.train()
        trainer.save_state()
        trainer.save_model(training_args.output_dir)
        processor.save_pretrained(training_args.output_dir)

    if training_args.do_eval:
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
