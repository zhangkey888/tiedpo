#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from qwen3vl_tiedpo.data import DataCollatorForQwen3VLTieDPODataset, Qwen3VLTieDPODataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-VL models with policy-only A/B gap metrics.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--lora-weight-path", default=None)
    parser.add_argument("--eval-data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--model-max-length", type=int, default=32768)
    parser.add_argument("--max-pixels", type=int, default=602112)
    parser.add_argument("--min-pixels", type=int, default=12544)
    parser.add_argument("--thresholds", default="0.1,0.2,0.5,0.8")
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def safe_div(numerator, denominator):
    denominator = float(denominator)
    if denominator <= 0:
        return 0.0
    return float(numerator) / denominator


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


def get_batch_logps_and_counts(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    label_pad_token_id: int = -100,
):
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits and labels must have the same batch/sequence shape.")
    shifted_labels = labels[:, 1:].clone()
    shifted_logits = logits[:, :-1, :]
    loss_mask = shifted_labels != label_pad_token_id
    shifted_labels[shifted_labels == label_pad_token_id] = 0
    per_token_logps = torch.gather(
        shifted_logits.log_softmax(-1), dim=2, index=shifted_labels.unsqueeze(2)
    ).squeeze(2)
    token_counts = loss_mask.sum(-1).clamp_min(1)
    seq_logps = (per_token_logps * loss_mask).sum(-1)
    return seq_logps, token_counts


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


def load_model(model_path, torch_dtype, attn_implementation, lora_weight_path):
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    if lora_weight_path:
        model = PeftModel.from_pretrained(model, lora_weight_path, is_trainable=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model


def build_concatenated_batch(batch, device):
    concatenated_batch = {
        "concatenated_input_ids": batch["concatenated_input_ids"].to(device),
        "concatenated_labels": batch["concatenated_labels"].to(device),
        "concatenated_attention_mask": batch["concatenated_attention_mask"].to(device),
    }
    for key, value in batch.items():
        if key.startswith(("chosen_", "rejected_", "concatenated_")):
            continue
        if key in {"is_tie", "is_ep_tie", "idx"}:
            concatenated_batch[key] = value
            continue
        if torch.is_tensor(value):
            concatenated_batch[f"concatenated_{key}"] = torch.cat([value, value], dim=0).to(device)
        else:
            concatenated_batch[f"concatenated_{key}"] = value + value
    return concatenated_batch


def get_dataset_records(dataset):
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        indices = list(dataset.indices)
    else:
        base_dataset = dataset
        indices = list(range(len(dataset)))

    records = getattr(base_dataset, "records", None)
    if records is None:
        return [{"dataset_index": index, "sample_id": index} for index in indices]

    rows = []
    for index in indices:
        raw = records[index]
        rows.append(
            {
                "dataset_index": index,
                "sample_id": raw.get("record_id", raw.get("id", raw.get("sample_id", index))),
                "question_id": raw.get("question_id"),
                "label": raw.get("label", raw.get("preference")),
                "image_path": raw.get("image", raw.get("image_path")),
            }
        )
    return rows


def main():
    args = parse_args()
    torch_dtype = resolve_dtype(args.torch_dtype)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
    )
    processor.tokenizer.model_max_length = args.model_max_length
    if processor.tokenizer.pad_token_id is None and processor.tokenizer.eos_token_id is not None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = load_model(
        model_path=args.model_path,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
        lora_weight_path=args.lora_weight_path,
    )
    device = next(model.parameters()).device

    dataset = Qwen3VLTieDPODataset(
        args.eval_data_path,
        processor=processor,
        model_max_length=args.model_max_length,
    )
    if args.max_samples > 0:
        dataset = Subset(dataset, range(min(len(dataset), args.max_samples)))
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=DataCollatorForQwen3VLTieDPODataset(tokenizer=processor.tokenizer),
    )

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
        concatenated_batch = build_concatenated_batch(batch, device)
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

        with torch.no_grad():
            outputs = model(**model_inputs)
            all_logits = outputs.logits.to(torch.float32)
            all_labels = concatenated_batch["concatenated_labels"]
            all_logps, all_token_counts = get_batch_logps_and_counts(all_logits, all_labels)

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]
        chosen_token_counts = all_token_counts[:len_chosen].to(torch.float32)
        rejected_token_counts = all_token_counts[len_chosen:].to(torch.float32)
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
        all_is_tie.extend(batch["is_tie"].detach().cpu().tolist())
        all_is_ep_tie.extend(batch["is_ep_tie"].detach().cpu().tolist())

        batch_is_tie = batch["is_tie"].detach().cpu().tolist()
        batch_is_ep_tie = batch["is_ep_tie"].detach().cpu().tolist()
        batch_idx = batch["idx"]
        for local_idx in range(len_chosen):
            record = dataset_records[sample_offset + local_idx] if sample_offset + local_idx < len(dataset_records) else {}
            sample_id = batch_idx[local_idx] if isinstance(batch_idx, list) else record.get("sample_id")
            signed_gap = float(signed_margins[local_idx])
            raw_signed_gap = float(raw_signed_margins[local_idx])
            per_sample_rows.append(
                {
                    "sample_offset": sample_offset + local_idx,
                    "dataset_index": record.get("dataset_index"),
                    "sample_id": sample_id,
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
        sample_offset += len_chosen

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
            "distance/tie_mean_abs": float(np.mean(tie_distances)) if tie_distances else 0.0,
            "distance/strict_mean_abs": float(np.mean(strict_distances)) if strict_distances else 0.0,
        }
    )

    output = {
        "model_path": args.model_path,
        "lora_weight_path": args.lora_weight_path,
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
