#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_PY="${PROJECT_ROOT}/scripts/eval_qwen3_abgap.py"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"

CONDA_SH="${CONDA_SH:-}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-qwen3vl-tiedpo}"

MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen3-VL-8B-Instruct}"
LORA_WEIGHT_PATH="${LORA_WEIGHT_PATH:-}"
DATASET_VARIANT="${DATASET_VARIANT:-plus200}"
if [[ "${DATASET_VARIANT}" == "plus200" ]]; then
  EVAL_DATA_PATH_DEFAULT="${DATA_ROOT}/processed_splits_balanced_16k_with_evidence/normalized_for_training/all_evidence_balanced_ab/test_normalized_balanced_ab_plus200_evidence_tie.jsonl"
else
  EVAL_DATA_PATH_DEFAULT="${DATA_ROOT}/processed_splits_balanced_16k_with_evidence/normalized_for_training/all_evidence_balanced_ab/test_normalized_balanced_ab.jsonl"
fi
EVAL_DATA_PATH="${EVAL_DATA_PATH:-${EVAL_DATA_PATH_DEFAULT}}"

RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
RUN_NAME="${RUN_NAME:-qwen3vl-base-tiebench-${RUN_TS}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/workspace/qwen3_base_eval/${RUN_NAME}}"

PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-32768}"
MAX_PIXELS="${MAX_PIXELS:-602112}"
MIN_PIXELS="${MIN_PIXELS:-12544}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
THRESHOLDS="${THRESHOLDS:-0.1,0.2,0.5,0.8}"

if [[ -z "${CONDA_SH}" ]] && command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [[ -n "${CONDA_BASE}" ]]; then
    CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
  fi
fi

if [[ ! -f "${EVAL_PY}" ]]; then
  echo "eval script not found: ${EVAL_PY}"
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH not found: ${MODEL_PATH}"
  exit 1
fi

if [[ ! -f "${EVAL_DATA_PATH}" ]]; then
  echo "missing EVAL_DATA_PATH"
  echo "EVAL_DATA_PATH=${EVAL_DATA_PATH}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "conda.sh not found: ${CONDA_SH}"
  exit 1
fi

# shellcheck disable=SC1090
source "${CONDA_SH}"
if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
  echo "Conda env not found: ${CONDA_ENV_NAME}"
  exit 1
fi
conda activate "${CONDA_ENV_NAME}"
echo "Activated conda env: ${CONDA_ENV_NAME}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
echo "PYTHONNOUSERSITE=${PYTHONNOUSERSITE}"
echo "PYTHONPATH=${PYTHONPATH}"

CMD=(
  python "${EVAL_PY}"
  --model-path "${MODEL_PATH}"
  --eval-data-path "${EVAL_DATA_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --batch-size "${PER_DEVICE_EVAL_BATCH_SIZE}"
  --num-workers "${DATALOADER_NUM_WORKERS}"
  --model-max-length "${MODEL_MAX_LENGTH}"
  --max-pixels "${MAX_PIXELS}"
  --min-pixels "${MIN_PIXELS}"
  --attn-implementation "${ATTN_IMPLEMENTATION}"
  --torch-dtype "${TORCH_DTYPE}"
  --thresholds "${THRESHOLDS}"
)

if [[ -n "${LORA_WEIGHT_PATH}" ]]; then
  CMD+=(--lora-weight-path "${LORA_WEIGHT_PATH}")
fi

"${CMD[@]}"

echo "output_dir=${OUTPUT_DIR}"
