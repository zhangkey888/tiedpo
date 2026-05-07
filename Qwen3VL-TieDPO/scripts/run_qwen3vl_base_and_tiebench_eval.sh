#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BASE_MODEL="${BASE_MODEL:-${PROJECT_ROOT}/models/Qwen3-VL-8B-Instruct}"
TIEDPO_CKPT="${TIEDPO_CKPT:-}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/workspace/final_eval}"
GROUP_NAME="${GROUP_NAME:-base_vs_tie_symmetric_$(date +%Y%m%d-%H%M%S)}"
GROUP_DIR="${RESULTS_ROOT}/${GROUP_NAME}"
SUMMARY_JSON="${GROUP_DIR}/summary.json"
SUMMARY_MD="${GROUP_DIR}/summary.md"
DATASET_VARIANT="${DATASET_VARIANT:-plus200}"
THRESHOLDS="${THRESHOLDS:-0.1,0.2,0.5,0.8}"

mkdir -p "${GROUP_DIR}"

echo "[1/3] Eval base model on AB-gap test set"
RUN_NAME="base_model" \
MODEL_PATH="${BASE_MODEL}" \
OUTPUT_DIR="${GROUP_DIR}/base_model" \
DATASET_VARIANT="${DATASET_VARIANT}" \
THRESHOLDS="${THRESHOLDS}" \
bash "${SCRIPT_DIR}/run_qwen3vl_base_tiebench_eval.sh"

if [[ -z "${TIEDPO_CKPT}" ]]; then
  echo "TIEDPO_CKPT is required"
  exit 1
fi

echo "[2/3] Eval tie_symmetric checkpoint on AB-gap test set"
RUN_NAME="tie_symmetric" \
MODEL_PATH="${BASE_MODEL}" \
LORA_WEIGHT_PATH="${TIEDPO_CKPT}" \
OUTPUT_DIR="${GROUP_DIR}/tie_symmetric" \
DATASET_VARIANT="${DATASET_VARIANT}" \
THRESHOLDS="${THRESHOLDS}" \
bash "${SCRIPT_DIR}/run_qwen3vl_base_tiebench_eval.sh"

echo "[3/3] Summarize metrics"
GROUP_DIR="${GROUP_DIR}" SUMMARY_JSON="${SUMMARY_JSON}" SUMMARY_MD="${SUMMARY_MD}" python - <<'PY'
import json
import os
from pathlib import Path

group_dir = Path(os.environ["GROUP_DIR"])
summary_json = Path(os.environ["SUMMARY_JSON"])
summary_md = Path(os.environ["SUMMARY_MD"])
model_order = ["base_model", "tie_symmetric"]
metric_order = [
    "distance/tie_mean_abs",
    "distance/strict_mean_abs",
    "margin/mean_abs",
    "margin/tie_mean_abs",
    "margin/strict_mean_abs",
    "thr/0.1/TieAcc",
    "thr/0.1/StrictAcc",
    "thr/0.2/TieAcc",
    "thr/0.2/StrictAcc",
    "thr/0.5/TieAcc",
    "thr/0.5/StrictAcc",
    "thr/0.8/TieAcc",
    "thr/0.8/StrictAcc",
    "data/n_tie",
    "data/n_strict",
    "data/n_ep_tie",
]

all_metrics = {}
for model_name in model_order:
    metrics_path = group_dir / model_name / "eval_results.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as f:
        all_metrics[model_name] = json.load(f)

summary = {
    "group_dir": str(group_dir),
    "models": model_order,
    "metrics": {
        metric: {model: all_metrics[model].get("metrics", {}).get(metric) for model in model_order}
        for metric in metric_order
    },
    "per_sample_paths": {
        model: all_metrics[model].get("per_sample_path")
        for model in model_order
    },
}

summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

lines = [
    "# qwen3vl abgap summary",
    "",
    f"- group_dir: `{group_dir}`",
    f"- per_sample_base: `{all_metrics['base_model'].get('per_sample_path')}`",
    f"- per_sample_tie: `{all_metrics['tie_symmetric'].get('per_sample_path')}`",
    "",
    "| Metric | Base | TieDPO-tie_symmetric |",
    "|---|---|---|",
]
for metric in metric_order:
    base_v = all_metrics["base_model"].get("metrics", {}).get(metric)
    tie_v = all_metrics["tie_symmetric"].get("metrics", {}).get(metric)
    base_s = f"{base_v:.4f}" if isinstance(base_v, float) else str(base_v)
    tie_s = f"{tie_v:.4f}" if isinstance(tie_v, float) else str(tie_v)
    lines.append(f"| {metric} | {base_s} | {tie_s} |")

summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY

echo "Summary: ${SUMMARY_MD}"
