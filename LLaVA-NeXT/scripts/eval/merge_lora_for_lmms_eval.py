#!/usr/bin/env python3
import argparse
import os
import shutil

import torch
from peft import PeftModel
from transformers import AutoProcessor, AutoTokenizer

from llava.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM


def load_non_lora_trainables(model, ckpt_path):
    non_lora_path = os.path.join(ckpt_path, "non_lora_trainables.bin")
    if not os.path.exists(non_lora_path):
        return

    non_lora = torch.load(non_lora_path, map_location="cpu")
    non_lora = {(k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora.items()}
    if any(k.startswith("model.model.") for k in non_lora):
        non_lora = {(k[6:] if k.startswith("model.") else k): v for k, v in non_lora.items()}
    model.load_state_dict(non_lora, strict=False)


def maybe_copy_extra_files(src_dir, dst_dir):
    extra_files = [
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "chat_template.json",
    ]
    for name in extra_files:
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Merge a LLaVA-OneVision LoRA checkpoint into a full model for lmms-eval.")
    parser.add_argument("--model-base", required=True, help="Base model path used during LoRA training.")
    parser.add_argument("--model-path", required=True, help="LoRA checkpoint directory.")
    parser.add_argument("--save-path", required=True, help="Directory to save merged full model.")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.torch_dtype]

    os.makedirs(args.save_path, exist_ok=True)

    print(f"[1/5] Loading configs from base/checkpoint")
    checkpoint_config = LlavaQwenConfig.from_pretrained(args.model_path)
    config = LlavaQwenConfig.from_pretrained(args.model_base)

    print(f"[2/5] Loading base model from {args.model_base}")
    model = LlavaQwenForCausalLM.from_pretrained(
        args.model_base,
        low_cpu_mem_usage=True,
        config=config,
        torch_dtype=torch_dtype,
    )

    print("[3/5] Loading non-LoRA trainables if present")
    load_non_lora_trainables(model, args.model_path)
    _emit_debug_event(
        "C",
        "merge_lora_for_lmms_eval.py:non_lora_loaded",
        "[DEBUG] loaded non-lora trainables",
        {
            "has_non_lora_trainables": os.path.exists(os.path.join(args.model_path, "non_lora_trainables.bin")),
        },
    )

    print(f"[4/5] Loading and merging LoRA weights from {args.model_path}")
    model = PeftModel.from_pretrained(model, args.model_path)
    model = model.merge_and_unload()
    _emit_debug_event(
        "D",
        "merge_lora_for_lmms_eval.py:merged",
        "[DEBUG] merged lora into model",
        {
            "merged_image_aspect_ratio": getattr(model.config, "image_aspect_ratio", None),
            "merged_image_grid_pinpoints": getattr(model.config, "image_grid_pinpoints", None),
            "merged_mm_vision_tower": getattr(model.config, "mm_vision_tower", None),
            "merged_model_type": getattr(model.config, "model_type", None),
        },
    )

    print(f"[5/5] Saving merged model to {args.save_path}")
    model.save_pretrained(args.save_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_base, use_fast=False)
    tokenizer.save_pretrained(args.save_path)

    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.save_pretrained(args.save_path)

    try:
        processor = AutoProcessor.from_pretrained(args.model_base)
        processor.save_pretrained(args.save_path)
    except Exception as exc:
        print(f"[WARN] AutoProcessor save skipped: {exc}")

    maybe_copy_extra_files(args.model_base, args.save_path)
    maybe_copy_extra_files(args.model_path, args.save_path)
    _emit_debug_event(
        "E",
        "merge_lora_for_lmms_eval.py:saved",
        "[DEBUG] saved merged model artifacts",
        {
            "save_path": args.save_path,
            "saved_files": sorted(os.listdir(args.save_path)),
        },
    )

    print(f"Merged model saved to: {args.save_path}")


if __name__ == "__main__":
    main()
