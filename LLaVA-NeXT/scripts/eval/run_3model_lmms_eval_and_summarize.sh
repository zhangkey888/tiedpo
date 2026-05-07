#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
MODELS_ROOT="${MODELS_ROOT:-${REPO_ROOT}/models}"

LMMS_EVAL_DIR="${LMMS_EVAL_DIR:-}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/lmms_eval_results}"
GROUP_NAME="${GROUP_NAME:-3model_compare_$(date +%Y%m%d-%H%M%S)}"

BASE_MODEL="${BASE_MODEL:-${MODELS_ROOT}/llava-onevision-qwen2-7b-ov}"
DPO_STRICT_CKPT="${DPO_STRICT_CKPT:-${PROJECT_ROOT}/ckpt/c1-minimal-20260430-012912/c1-minimal-dpo_strict-20260430-012912}"
TIEDPO_CKPT="${TIEDPO_CKPT:-${PROJECT_ROOT}/ckpt/c1-minimal-20260430-012912/c1-minimal-tie_symmetric-20260430-012912}"

NUM_PROCESSES="${NUM_PROCESSES:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
TASKS="${TASKS:-xlrs-lite}"
MODEL_NAME="${MODEL_NAME:-llava-onevision-qwen2-7b-ov}"
CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_1_5}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
HF_HOME="${HF_HOME:-}"
HF_HUB_CACHE="${HF_HUB_CACHE:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_ENDPOINT
export HF_HUB_ENABLE_HF_TRANSFER
if [[ -n "${HF_TOKEN}" ]]; then
    export HF_TOKEN
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi
if [[ -n "${HF_HOME}" ]]; then
    export HF_HOME
fi
if [[ -n "${HF_HUB_CACHE}" ]]; then
    export HF_HUB_CACHE
fi

MODEL_ORDER=("tie_symmetric" "dpo_strict")

GROUP_DIR="${RESULTS_ROOT}/${GROUP_NAME}"
SUMMARY_JSON="${GROUP_DIR}/summary.json"
SUMMARY_MD="${GROUP_DIR}/summary.md"

mkdir -p "${GROUP_DIR}"

echo "LMMS_EVAL_DIR=${LMMS_EVAL_DIR}"
echo "GROUP_DIR=${GROUP_DIR}"
echo "TASKS=${TASKS}"
echo "HF_ENDPOINT=${HF_ENDPOINT}"
echo "HF_TOKEN_SET=$([[ -n "${HF_TOKEN}" ]] && echo yes || echo no)"
echo "MODEL_NAME=${MODEL_NAME}"
echo "CONV_TEMPLATE=${CONV_TEMPLATE}"

if [[ -z "${LMMS_EVAL_DIR}" || ! -d "${LMMS_EVAL_DIR}" ]]; then
    echo "lmms-eval repo not found: ${LMMS_EVAL_DIR}"
  echo "Set LMMS_EVAL_DIR to your local lmms-eval checkout."
    exit 1
fi

run_lora_model() {
    local run_name="$1"
    local lora_ckpt="$2"
    echo "[*] Evaluating ${run_name} from ${lora_ckpt}"
    LMMS_EVAL_DIR="${LMMS_EVAL_DIR}" \
    BASE_MODEL="${BASE_MODEL}" \
    LORA_CKPT="${lora_ckpt}" \
    RUN_NAME="${run_name}" \
    RESULTS_DIR="${GROUP_DIR}" \
    TORCH_DTYPE="${TORCH_DTYPE}" \
    NUM_PROCESSES="${NUM_PROCESSES}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    TASKS="${TASKS}" \
    MODEL_NAME="${MODEL_NAME}" \
    CONV_TEMPLATE="${CONV_TEMPLATE}" \
    HF_ENDPOINT="${HF_ENDPOINT}" \
    HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER}" \
    bash "${SCRIPT_DIR}/run_lmms_eval_10bench.sh"
}

summarize_results() {
    echo "[3/3] Summarizing results"
    SUMMARY_JSON="${SUMMARY_JSON}" SUMMARY_MD="${SUMMARY_MD}" GROUP_DIR="${GROUP_DIR}" MODEL_ORDER_CSV="$(IFS=,; echo "${MODEL_ORDER[*]}")" python - <<'PY'
import glob
import json
import os

group_dir = os.environ["GROUP_DIR"]
summary_json = os.environ["SUMMARY_JSON"]
summary_md = os.environ["SUMMARY_MD"]
model_order = [x for x in os.environ["MODEL_ORDER_CSV"].split(",") if x]


def load_model_results(model_dir):
    results = {}
    for path in glob.glob(os.path.join(model_dir, "**", "*_results.json"), recursive=True):
        task_name = os.path.basename(path).replace("_results.json", "")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        metrics = data.get("results", {}) or {}
        if not metrics:
            results[task_name] = {"primary_metric": None, "primary_value": None, "all_metrics": {}}
            continue
        primary_metric, primary_value = next(iter(metrics.items()))
        results[task_name] = {
            "primary_metric": primary_metric,
            "primary_value": primary_value,
            "all_metrics": metrics,
        }
    return results


all_results = {}
all_tasks = set()
for model_name in model_order:
    model_dir = os.path.join(group_dir, model_name)
    all_results[model_name] = load_model_results(model_dir)
    all_tasks.update(all_results[model_name].keys())

all_tasks = sorted(all_tasks)

summary = {
    "group_dir": group_dir,
    "models": model_order,
    "tasks": {},
}

for task in all_tasks:
    summary["tasks"][task] = {}
    for model_name in model_order:
        summary["tasks"][task][model_name] = all_results[model_name].get(task, {})

with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

lines = []
lines.append("# lmms-eval summary")
lines.append("")
lines.append(f"- group_dir: `{group_dir}`")
lines.append("")

header_map = {
    "base_model": "Base",
    "dpo_strict": "DPO-strict",
    "tie_symmetric": "TieDPO-tie_symmetric",
}
header = ["Task"] + [header_map.get(x, x) for x in model_order]
lines.append("| " + " | ".join(header) + " |")
lines.append("|" + "---|" * len(header))

for task in all_tasks:
    row = [task]
    for model_name in model_order:
        item = summary["tasks"][task].get(model_name, {})
        metric = item.get("primary_metric")
        value = item.get("primary_value")
        if metric is None or value is None:
            row.append("N/A")
        else:
            if isinstance(value, float):
                row.append(f"{metric}={value:.4f}")
            else:
                row.append(f"{metric}={value}")
    lines.append("| " + " | ".join(row) + " |")

with open(summary_md, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"Summary written to: {summary_json}")
print(f"Summary written to: {summary_md}")
print("\n".join(lines))
PY
}

echo "[1/3] Skipping base model"
echo "[2/3] Evaluating LoRA checkpoints"
run_lora_model "tie_symmetric" "${TIEDPO_CKPT}"
run_lora_model "dpo_strict" "${DPO_STRICT_CKPT}"
summarize_results

echo "All evaluations finished."
echo "Results: ${GROUP_DIR}"
echo "Summary: ${SUMMARY_MD}"
