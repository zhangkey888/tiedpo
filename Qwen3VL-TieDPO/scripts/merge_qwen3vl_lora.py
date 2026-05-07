#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Merge a Qwen3-VL LoRA checkpoint into the base model.")
    parser.add_argument("--model-base", required=True)
    parser.add_argument("--lora-ckpt", required=True)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--torch-dtype", default="bfloat16")
    return parser.parse_args()


def main():
    args = parse_args()
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map.get(args.torch_dtype.lower(), torch.bfloat16)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_base,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
    )
    model = PeftModel.from_pretrained(model, args.lora_ckpt)
    model = model.merge_and_unload()
    model.save_pretrained(save_path)

    processor = AutoProcessor.from_pretrained(args.model_base)
    processor.save_pretrained(save_path)
    print(f"Merged model saved to: {save_path}")


if __name__ == "__main__":
    main()
