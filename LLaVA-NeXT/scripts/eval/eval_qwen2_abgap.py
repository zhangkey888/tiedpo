#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


def parse_args():
    script_path = Path(__file__).resolve()
    legacy_root_default = script_path.parents[2]
    tiedpo_root_default = script_path.parents[3]
    parser = argparse.ArgumentParser(description="Evaluate Qwen2/LLaVA models with policy-only A/B gap metrics.")
    parser.add_argument("--legacy-root", default=str(legacy_root_default))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--eval-data-path", required=True)
    parser.add_argument("--image-folder", default=str(tiedpo_root_default / "data"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--thresholds", default="0.1,0.2,0.5,0.8")
    parser.add_argument("--prompt-version", default="qwen_1_5")
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def safe_div(numerator, denominator):
    denominator = float(denominator)
    if denominator <= 0:
        return 0.0
    return float(numerator) / denominator


def get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, label_pad_token_id: int = -100):
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits and labels must have the same batch/sequence shape.")
    shifted_labels = labels[:, 1:].clone()
    shifted_logits = logits[:, :-1, :]
    loss_mask = shifted_labels != label_pad_token_id
    shifted_labels[shifted_labels == label_pad_token_id] = 0
    per_token_logps = torch.gather(
        shifted_logits.log_softmax(-1),
        dim=2,
        index=shifted_labels.unsqueeze(2),
    ).squeeze(2)
    token_counts = loss_mask.sum(-1).clamp_min(1)
    seq_logps = (per_token_logps * loss_mask).sum(-1)
    return seq_logps, token_counts


def to_python(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def move_images(images, device, dtype):
    moved = []
    for image in images:
        if torch.is_tensor(image):
            moved.append(image.to(device=device, dtype=dtype))
        else:
            moved.append(image)
    return moved


def import_llava_modules(legacy_root: Path):
    sys.path.insert(0, str(legacy_root))
    old_offline = os.environ.get("HF_HUB_OFFLINE")
    old_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")
    old_endpoint = os.environ.get("HF_ENDPOINT")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    try:
        import llava.conversation as conversation_lib
    finally:
        if old_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = old_offline
        if old_transformers_offline is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = old_transformers_offline
        if old_endpoint is None:
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = old_endpoint

    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model
    from llava.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM
    from llava.train.train_tie_dpo import DataCollatorForTieDPODataset, TieDPODataset
    from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

    return (
        conversation_lib,
        get_model_name_from_path,
        load_pretrained_model,
        LlavaQwenConfig,
        LlavaQwenForCausalLM,
        DataCollatorForTieDPODataset,
        TieDPODataset,
        DEFAULT_IMAGE_PATCH_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
    )


def load_non_lora_trainables(model, ckpt_path: str):
    non_lora_path = os.path.join(ckpt_path, "non_lora_trainables.bin")
    if not os.path.exists(non_lora_path):
        return
    non_lora = torch.load(non_lora_path, map_location="cpu")
    non_lora = {(k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora.items()}
    if any(k.startswith("model.model.") for k in non_lora):
        non_lora = {(k[6:] if k.startswith("model.") else k): v for k, v in non_lora.items()}
    model.load_state_dict(non_lora, strict=False)


def load_qwen2_lora_model(
    args,
    tokenizer_cls,
    peft_model_cls,
    llava_qwen_config_cls,
    llava_qwen_model_cls,
    image_patch_token: str,
    im_start_token: str,
    im_end_token: str,
):
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.torch_dtype]
    _ = llava_qwen_config_cls.from_pretrained(args.model_path)
    base_config = llava_qwen_config_cls.from_pretrained(args.model_base)
    tokenizer = tokenizer_cls.from_pretrained(args.model_base, use_fast=False)
    model = llava_qwen_model_cls.from_pretrained(
        args.model_base,
        low_cpu_mem_usage=True,
        config=base_config,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
        device_map="auto",
    )
    load_non_lora_trainables(model, args.model_path)
    model = peft_model_cls.from_pretrained(model, args.model_path)
    model = model.merge_and_unload()

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([image_patch_token], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([im_start_token, im_end_token], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map="auto")
    image_processor = vision_tower.image_processor
    return tokenizer, model, image_processor


def build_metrics(signed_margins, is_tie, is_ep_tie, thresholds):
    signed_margins = np.asarray(signed_margins, dtype=np.float64)
    abs_margins = np.abs(signed_margins)
    is_tie = np.asarray(is_tie, dtype=bool)
    is_ep_tie = np.asarray(is_ep_tie, dtype=bool)
    strict_mask = ~is_tie

    n_total = len(signed_margins)
    n_tie = int(is_tie.sum())
    n_strict = int(strict_mask.sum())
    n_ep_tie = int(is_ep_tie.sum())

    metrics = {
        "margin/mean_abs": float(abs_margins.mean()) if n_total else 0.0,
        "margin/tie_mean_abs": float(abs_margins[is_tie].mean()) if n_tie else 0.0,
        "margin/strict_mean_abs": float(abs_margins[strict_mask].mean()) if n_strict else 0.0,
        "margin/strict_mean_signed": float(signed_margins[strict_mask].mean()) if n_strict else 0.0,
        "data/n_samples": n_total,
        "data/n_tie": safe_div(n_tie, n_total),
        "data/n_strict": safe_div(n_strict, n_total),
        "data/n_ep_tie": safe_div(n_ep_tie, n_total),
    }

    for threshold in thresholds:
        threshold_str = f"{threshold:.1f}"
        pred_tie = abs_margins <= threshold
        pred_strict = signed_margins > threshold

        tie_tp = int((pred_tie & is_tie).sum())
        tie_fp = int((pred_tie & strict_mask).sum())
        tie_fn = int((~pred_tie & is_tie).sum())

        strict_tp = int((pred_strict & strict_mask).sum())
        strict_fp = int((pred_strict & is_tie).sum())
        strict_fn = int((~pred_strict & strict_mask).sum())

        overall_correct = tie_tp + strict_tp
        ep_tie_tp = int((pred_tie & is_ep_tie).sum())

        metrics.update(
            {
                f"thr/{threshold_str}/OverallAcc": safe_div(overall_correct, n_total),
                f"thr/{threshold_str}/TieAcc": safe_div(tie_tp, n_tie),
                f"thr/{threshold_str}/TiePrecision": safe_div(tie_tp, tie_tp + tie_fp),
                f"thr/{threshold_str}/TieRecall": safe_div(tie_tp, tie_tp + tie_fn),
                f"thr/{threshold_str}/StrictAcc": safe_div(strict_tp, n_strict),
                f"thr/{threshold_str}/StrictPrecision": safe_div(strict_tp, strict_tp + strict_fp),
                f"thr/{threshold_str}/StrictRecall": safe_div(strict_tp, strict_tp + strict_fn),
                f"thr/{threshold_str}/EPTieAcc": safe_div(ep_tie_tp, n_ep_tie),
                f"thr/{threshold_str}/TiePredRate": safe_div(int(pred_tie.sum()), n_total),
                f"thr/{threshold_str}/StrictPredRate": safe_div(int(pred_strict.sum()), n_total),
            }
        )
    return metrics


def get_dataset_records(dataset):
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        indices = list(dataset.indices)
    else:
        base_dataset = dataset
        indices = list(range(len(dataset)))

    list_data_dict = getattr(base_dataset, "list_data_dict", None)
    if list_data_dict is None:
        return [{"dataset_index": index} for index in indices]

    records = []
    for index in indices:
        raw = list_data_dict[index]
        records.append(
            {
                "dataset_index": index,
                "sample_id": raw.get("record_id", raw.get("id", raw.get("sample_id", index))),
                "question_id": raw.get("question_id"),
                "label": raw.get("label", raw.get("preference")),
                "image_path": raw.get("image", raw.get("image_path")),
            }
        )
    return records


def main():
    args = parse_args()
    legacy_root = Path(args.legacy_root).resolve()
    (
        conversation_lib,
        get_model_name_from_path,
        load_pretrained_model,
        LlavaQwenConfig,
        LlavaQwenForCausalLM,
        DataCollatorForTieDPODataset,
        TieDPODataset,
        DEFAULT_IMAGE_PATCH_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
    ) = import_llava_modules(legacy_root)
    from peft import PeftModel
    from transformers import AutoConfig, AutoTokenizer

    conversation_lib.default_conversation = conversation_lib.conv_templates[args.prompt_version]

    checkpoint_config = AutoConfig.from_pretrained(args.model_path)
    is_qwen_lora = bool(
        args.model_base
        and (
            getattr(checkpoint_config, "model_type", "") == "llava_qwen"
            or "qwen" in getattr(checkpoint_config, "model_type", "").lower()
            or any("Qwen" in arch for arch in getattr(checkpoint_config, "architectures", []))
        )
    )

    if is_qwen_lora:
        tokenizer, model, image_processor = load_qwen2_lora_model(
            args=args,
            tokenizer_cls=AutoTokenizer,
            peft_model_cls=PeftModel,
            llava_qwen_config_cls=LlavaQwenConfig,
            llava_qwen_model_cls=LlavaQwenForCausalLM,
            image_patch_token=DEFAULT_IMAGE_PATCH_TOKEN,
            im_start_token=DEFAULT_IM_START_TOKEN,
            im_end_token=DEFAULT_IM_END_TOKEN,
        )
    elif args.model_base:
        base_model_name = get_model_name_from_path(args.model_base)
        model_name = f"{base_model_name}-lora"
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.model_path,
            args.model_base,
            model_name,
            device_map=args.device,
            torch_dtype=args.torch_dtype,
            multimodal=True,
            attn_implementation=args.attn_implementation,
        )
    else:
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.model_path,
            args.model_base,
            model_name,
            device_map=args.device,
            torch_dtype=args.torch_dtype,
            multimodal=True,
            attn_implementation=args.attn_implementation,
        )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    data_args = SimpleNamespace(
        image_folder=args.image_folder,
        image_processor=image_processor,
        image_aspect_ratio=getattr(model.config, "image_aspect_ratio", "pad"),
        image_grid_pinpoints=getattr(model.config, "image_grid_pinpoints", None),
        is_multimodal=True,
        video_folder="",
        video_fps=1,
        frames_upbound=0,
    )
    dataset = TieDPODataset(
        data_path=args.eval_data_path,
        tokenizer=tokenizer,
        data_args=data_args,
    )
    if args.max_samples > 0:
        dataset = Subset(dataset, range(min(len(dataset), args.max_samples)))
    collator = DataCollatorForTieDPODataset(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    pad_token_id = tokenizer.pad_token_id
    model_dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    thresholds = [float(item.strip()) for item in args.thresholds.split(",") if item.strip()]

    all_signed_margins = []
    all_raw_signed_margins = []
    all_chosen_avg_logps = []
    all_rejected_avg_logps = []
    all_chosen_token_counts = []
    all_rejected_token_counts = []
    all_is_tie = []
    all_is_ep_tie = []
    per_sample_rows = []
    dataset_records = get_dataset_records(dataset)
    sample_offset = 0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        chosen_input_ids = batch["chosen_input_ids"].to(device)
        rejected_input_ids = batch["rejected_input_ids"].to(device)
        chosen_labels = batch["chosen_labels"].to(device)
        rejected_labels = batch["rejected_labels"].to(device)
        chosen_attention_mask = batch["chosen_attention_mask"].to(device)
        rejected_attention_mask = batch["rejected_attention_mask"].to(device)

        max_length = max(chosen_input_ids.shape[1], rejected_input_ids.shape[1])

        def pad_to_length(tensor, length, pad_value):
            if tensor.shape[1] >= length:
                return tensor
            pad_shape = (tensor.shape[0], length - tensor.shape[1])
            pad = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, pad], dim=1)

        concatenated_input_ids = torch.cat(
            [
                pad_to_length(chosen_input_ids, max_length, pad_token_id),
                pad_to_length(rejected_input_ids, max_length, pad_token_id),
            ],
            dim=0,
        )
        concatenated_labels = torch.cat(
            [
                pad_to_length(chosen_labels, max_length, -100),
                pad_to_length(rejected_labels, max_length, -100),
            ],
            dim=0,
        )
        concatenated_attention_mask = torch.cat(
            [
                pad_to_length(chosen_attention_mask, max_length, 0),
                pad_to_length(rejected_attention_mask, max_length, 0),
            ],
            dim=0,
        )

        concatenated_images = move_images(batch["images"] * 2, device=device, dtype=model_dtype)
        image_sizes = batch["image_sizes"] * 2
        modalities = batch["modalities"] * 2

        with torch.no_grad():
            all_logits, new_labels = model(
                concatenated_input_ids,
                attention_mask=concatenated_attention_mask,
                labels=concatenated_labels,
                images=concatenated_images,
                image_sizes=image_sizes,
                modalities=modalities,
                use_cache=False,
                dpo_forward=True,
            )
            all_logits = all_logits.to(torch.float32)
            all_logps, all_token_counts = get_batch_logps(all_logits, new_labels)

        batch_size = chosen_input_ids.shape[0]
        chosen_logps = all_logps[:batch_size]
        rejected_logps = all_logps[batch_size:]
        chosen_token_counts = all_token_counts[:batch_size].to(torch.float32)
        rejected_token_counts = all_token_counts[batch_size:].to(torch.float32)
        chosen_avg_logps = chosen_logps / chosen_token_counts
        rejected_avg_logps = rejected_logps / rejected_token_counts

        signed_margins = (chosen_avg_logps - rejected_avg_logps).detach().cpu().tolist()
        raw_signed_margins = (chosen_logps - rejected_logps).detach().cpu().tolist()

        all_signed_margins.extend(signed_margins)
        all_raw_signed_margins.extend(raw_signed_margins)
        all_chosen_avg_logps.extend(chosen_avg_logps.detach().cpu().tolist())
        all_rejected_avg_logps.extend(rejected_avg_logps.detach().cpu().tolist())
        all_chosen_token_counts.extend(chosen_token_counts.detach().cpu().tolist())
        all_rejected_token_counts.extend(rejected_token_counts.detach().cpu().tolist())
        all_is_tie.extend(to_python(batch["is_tie"]))
        all_is_ep_tie.extend(to_python(batch["is_ep_tie"]))

        batch_is_tie = to_python(batch["is_tie"])
        batch_is_ep_tie = to_python(batch["is_ep_tie"])
        for local_idx in range(batch_size):
            record = dataset_records[sample_offset + local_idx] if sample_offset + local_idx < len(dataset_records) else {}
            signed_gap = float(signed_margins[local_idx])
            raw_signed_gap = float(raw_signed_margins[local_idx])
            per_sample_rows.append(
                {
                    "sample_offset": sample_offset + local_idx,
                    "dataset_index": record.get("dataset_index"),
                    "sample_id": record.get("sample_id"),
                    "question_id": record.get("question_id"),
                    "label": record.get("label"),
                    "image_path": record.get("image_path"),
                    "is_tie": bool(batch_is_tie[local_idx]),
                    "is_ep_tie": bool(batch_is_ep_tie[local_idx]),
                    "policy_gap_signed": signed_gap,
                    "policy_gap_abs": abs(signed_gap),
                    "policy_gap_raw_signed": raw_signed_gap,
                    "policy_gap_raw_abs": abs(raw_signed_gap),
                    "chosen_avg_logp": float(chosen_avg_logps[local_idx].detach().cpu().item()),
                    "rejected_avg_logp": float(rejected_avg_logps[local_idx].detach().cpu().item()),
                    "chosen_token_count": float(chosen_token_counts[local_idx].detach().cpu().item()),
                    "rejected_token_count": float(rejected_token_counts[local_idx].detach().cpu().item()),
                }
            )
        sample_offset += batch_size

    metrics = build_metrics(all_signed_margins, all_is_tie, all_is_ep_tie, thresholds)
    raw_abs_margins = np.abs(np.asarray(all_raw_signed_margins, dtype=np.float64))
    tie_distances = [row["policy_gap_abs"] for row in per_sample_rows if row["is_tie"]]
    strict_distances = [row["policy_gap_abs"] for row in per_sample_rows if not row["is_tie"]]
    metrics.update(
        {
            "debug/raw_margin_mean_abs": float(raw_abs_margins.mean()) if len(raw_abs_margins) else 0.0,
            "debug/logp/chosen_mean": float(np.mean(all_chosen_avg_logps)) if all_chosen_avg_logps else 0.0,
            "debug/logp/rejected_mean": float(np.mean(all_rejected_avg_logps)) if all_rejected_avg_logps else 0.0,
            "debug/tokens/chosen_mean": float(np.mean(all_chosen_token_counts)) if all_chosen_token_counts else 0.0,
            "debug/tokens/rejected_mean": float(np.mean(all_rejected_token_counts)) if all_rejected_token_counts else 0.0,
            "debug/model_load/is_qwen_lora": float(is_qwen_lora),
            "distance/tie_mean_abs": float(np.mean(tie_distances)) if tie_distances else 0.0,
            "distance/strict_mean_abs": float(np.mean(strict_distances)) if strict_distances else 0.0,
        }
    )
    output = {
        "model_path": args.model_path,
        "model_base": args.model_base,
        "eval_data_path": args.eval_data_path,
        "thresholds": thresholds,
        "max_samples": args.max_samples,
        "per_sample_path": str(Path(args.output_dir) / "per_sample_distances.jsonl"),
        "metrics": metrics,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = output_dir / "per_sample_distances.jsonl"
    with per_sample_path.open("w", encoding="utf-8") as f:
        for row in per_sample_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    output_path = output_dir / "eval_results.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"eval_results={output_path}")


if __name__ == "__main__":
    main()
