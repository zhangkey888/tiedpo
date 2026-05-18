#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EVAL_PY="${SCRIPT_DIR}/eval_qwen2_abgap.py"
SUMMARY_SCRIPT="${SCRIPT_DIR}/summarize_qwen2_abgap_eval.py"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
MODELS_ROOT="${MODELS_ROOT:-${REPO_ROOT}/models}"

CONDA_SH="${CONDA_SH:-}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-dpo}"
HF_HOME="${HF_HOME:-${REPO_ROOT}/hf_cache}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-${MODELS_ROOT}/llava-onevision-qwen2-7b-ov}"
DPO_ONLY_CKPT="${DPO_ONLY_CKPT:-}"
TIE_SYMMETRIC_CKPT="${TIE_SYMMETRIC_CKPT:-}"

DATASET_VARIANT="${DATASET_VARIANT:-plus200}"
if [[ "${DATASET_VARIANT}" == "plus200" ]]; then
  EVAL_DATA_PATH_DEFAULT="${DATA_ROOT}/processed_splits_balanced_16k_with_evidence/normalized_for_training/all_evidence_balanced_ab/test_normalized_balanced_ab_plus200_evidence_tie.jsonl"
else
  EVAL_DATA_PATH_DEFAULT="${DATA_ROOT}/processed_splits_balanced_16k_with_evidence/normalized_for_training/all_evidence_balanced_ab/test_normalized_balanced_ab.jsonl"
fi
EVAL_DATA_PATH="${EVAL_DATA_PATH:-${EVAL_DATA_PATH_DEFAULT}}"

RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
GROUP_NAME="${GROUP_NAME:-qwen2-7b-3model-abgap-${RUN_TS}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/eval_abgap/${GROUP_NAME}}"

PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
THRESHOLDS="${THRESHOLDS:-0.1,0.2,0.5,0.8}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${DATA_ROOT}}"
PROMPT_VERSION="${PROMPT_VERSION:-qwen_1_5}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}"

if [[ -z "${CONDA_SH}" ]] && command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [[ -n "${CONDA_BASE}" ]]; then
    CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
  fi
fi

if [[ ! -f "${EVAL_PY}" || ! -f "${SUMMARY_SCRIPT}" ]]; then
  echo "eval python or summary script missing"
  echo "EVAL_PY=${EVAL_PY}"
  echo "SUMMARY_SCRIPT=${SUMMARY_SCRIPT}"
  exit 1
fi

if [[ -z "${DPO_ONLY_CKPT}" || -z "${TIE_SYMMETRIC_CKPT}" ]]; then
  echo "Please set DPO_ONLY_CKPT and TIE_SYMMETRIC_CKPT."
  exit 1
fi

if [[ ! -d "${BASE_MODEL_PATH}" || ! -d "${DPO_ONLY_CKPT}" || ! -d "${TIE_SYMMETRIC_CKPT}" ]]; then
  echo "missing model path"
  echo "BASE_MODEL_PATH=${BASE_MODEL_PATH}"
  echo "DPO_ONLY_CKPT=${DPO_ONLY_CKPT}"
  echo "TIE_SYMMETRIC_CKPT=${TIE_SYMMETRIC_CKPT}"
  exit 1
fi

if [[ ! -f "${EVAL_DATA_PATH}" ]]; then
  echo "missing EVAL_DATA_PATH=${EVAL_DATA_PATH}"
  exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "conda.sh not found: ${CONDA_SH}"
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

# shellcheck disable=SC1090
source "${CONDA_SH}"
if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
  echo "Conda env not found: ${CONDA_ENV_NAME}"
  exit 1
fi
conda activate "${CONDA_ENV_NAME}"
echo "Activated conda env: ${CONDA_ENV_NAME}"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}"
export HF_HOME
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
if [[ -n "${HF_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT
fi
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

run_eval() {
  local model_tag="$1"
  local model_path="$2"
  local model_base="$3"
  local output_dir="${OUTPUT_ROOT}/${model_tag}"

  mkdir -p "${output_dir}"
  echo "[eval] model_tag=${model_tag}"
  CMD=(
    python "${EVAL_PY}"
    --legacy-root "${PROJECT_ROOT}"
    --model-path "${model_path}"
    --eval-data-path "${EVAL_DATA_PATH}"
    --image-folder "${IMAGE_FOLDER}"
    --output-dir "${output_dir}"
    --batch-size "${PER_DEVICE_EVAL_BATCH_SIZE}"
    --num-workers "${DATALOADER_NUM_WORKERS}"
    --torch-dtype "${TORCH_DTYPE}"
    --attn-implementation "${ATTN_IMPLEMENTATION}"
    --thresholds "${THRESHOLDS}"
    --prompt-version "${PROMPT_VERSION}"
  )
  if [[ -n "${model_base}" ]]; then
    CMD+=(--model-base "${model_base}")
  fi
  "${CMD[@]}"
}

run_eval "base_model" "${BASE_MODEL_PATH}" ""
run_eval "tie_symmetric" "${TIE_SYMMETRIC_CKPT}" "${BASE_MODEL_PATH}"
run_eval "dpo_only" "${DPO_ONLY_CKPT}" "${BASE_MODEL_PATH}"

python "${SUMMARY_SCRIPT}" "${OUTPUT_ROOT}"

echo "output_root=${OUTPUT_ROOT}"
echo "summary_json=${OUTPUT_ROOT}/summary.json"
echo "summary_md=${OUTPUT_ROOT}/summary.md"
