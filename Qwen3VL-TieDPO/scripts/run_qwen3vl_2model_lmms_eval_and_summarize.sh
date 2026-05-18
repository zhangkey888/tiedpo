#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/workspace/lmms_eval_results}"
GROUP_NAME="${GROUP_NAME:-2model_compare_$(date +%Y%m%d-%H%M%S)}"
GROUP_DIR="${RESULTS_ROOT}/${GROUP_NAME}"
SUMMARY_JSON="${GROUP_DIR}/summary.json"
SUMMARY_MD="${GROUP_DIR}/summary.md"

BASE_MODEL="${BASE_MODEL:-${PROJECT_ROOT}/models/Qwen3-VL-8B-Instruct}"
TIEDPO_CKPT="${TIEDPO_CKPT:-}"
TASKS="${TASKS:-xlrs-lite}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"

mkdir -p "${GROUP_DIR}"

echo "[1/3] Eval base model with lmms-eval"
RUN_NAME="base_model" \
MODEL_BASE="${BASE_MODEL}" \
PRETRAINED_PATH="${BASE_MODEL}" \
RESULTS_DIR="${GROUP_DIR}" \
TASKS="${TASKS}" \
NUM_PROCESSES="${NUM_PROCESSES}" \
BATCH_SIZE="${BATCH_SIZE}" \
bash "${SCRIPT_DIR}/run_qwen3vl_lmms_eval.sh"

if [[ -z "${TIEDPO_CKPT}" ]]; then
  echo "TIEDPO_CKPT is required"
  exit 1
fi

echo "[2/3] Eval tie_symmetric checkpoint with lmms-eval"
RUN_NAME="tie_symmetric" \
MODEL_BASE="${BASE_MODEL}" \
LORA_CKPT="${TIEDPO_CKPT}" \
RESULTS_DIR="${GROUP_DIR}" \
TASKS="${TASKS}" \
NUM_PROCESSES="${NUM_PROCESSES}" \
BATCH_SIZE="${BATCH_SIZE}" \
bash "${SCRIPT_DIR}/run_qwen3vl_lmms_eval.sh"

echo "[3/3] Summarize lmms-eval"
GROUP_DIR="${GROUP_DIR}" SUMMARY_JSON="${SUMMARY_JSON}" SUMMARY_MD="${SUMMARY_MD}" python - <<'PY'
import glob
import json
import os

group_dir = os.environ["GROUP_DIR"]
summary_json = os.environ["SUMMARY_JSON"]
summary_md = os.environ["SUMMARY_MD"]
model_order = ["base_model", "tie_symmetric"]

def load_model_results(model_dir):
    results = {}
    for path in glob.glob(os.path.join(model_dir, "**", "*_results.json"), recursive=True):
        task_name = os.path.basename(path).replace("_results.json", "")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        metrics = data.get("results", {}) or {}
        primary_metric, primary_value = (None, None) if not metrics else next(iter(metrics.items()))
        results[task_name] = {
            "primary_metric": primary_metric,
            "primary_value": primary_value,
            "all_metrics": metrics,
        }
    return results

all_results = {model_name: load_model_results(os.path.join(group_dir, model_name)) for model_name in model_order}
all_tasks = sorted({task for per_model in all_results.values() for task in per_model})
summary = {"group_dir": group_dir, "models": model_order, "tasks": {}}
for task in all_tasks:
    summary["tasks"][task] = {model_name: all_results[model_name].get(task, {}) for model_name in model_order}

with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

lines = [
    "# qwen3vl lmms-eval summary",
    "",
    f"- group_dir: `{group_dir}`",
    "",
    "| Task | Base | TieDPO-tie_symmetric |",
    "|---|---|---|",
]
for task in all_tasks:
    row = [task]
    for model_name in model_order:
        item = summary["tasks"][task].get(model_name, {})
        metric = item.get("primary_metric")
        value = item.get("primary_value")
        if metric is None or value is None:
            row.append("N/A")
        else:
            row.append(f"{metric}={value:.4f}" if isinstance(value, float) else f"{metric}={value}")
    lines.append("| " + " | ".join(row) + " |")

with open(summary_md, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
PY

echo "Summary: ${SUMMARY_MD}"
