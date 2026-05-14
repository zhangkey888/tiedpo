#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

CONDA_SH="${CONDA_SH:-}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lmms-eval}"

BASE_MODEL="${BASE_MODEL:-${PROJECT_ROOT}/models/Qwen3-VL-8B-Instruct}"
TIEDPO_CKPT="${TIEDPO_CKPT:-}"

LMMS_EVAL_DIR="${LMMS_EVAL_DIR:-}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/workspace/lmms_eval_results}"
GROUP_NAME="${GROUP_NAME:-qwen3vl-mirb-base-vs-tie_$(date +%Y%m%d-%H%M%S)}"
TASKS="${TASKS:-mirb}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"

HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
HF_HOME="${HF_HOME:-${REPO_ROOT}/hf_cache}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

if [[ -z "${CONDA_SH}" ]] && command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [[ -n "${CONDA_BASE}" ]]; then
    CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
  fi
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "conda.sh not found: ${CONDA_SH}"
  echo "Set CONDA_SH or initialize conda before running this script."
  exit 1
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"

export PYTHONNOUSERSITE=1
export HF_HUB_ENABLE_HF_TRANSFER
export HF_HOME
export HUGGINGFACE_HUB_CACHE
export TRANSFORMERS_CACHE
if [[ -n "${HF_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT
fi

echo "CONDA_ENV_NAME=${CONDA_ENV_NAME}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "TIEDPO_CKPT=${TIEDPO_CKPT}"
echo "TASKS=${TASKS}"
echo "GROUP_NAME=${GROUP_NAME}"
echo "LMMS_EVAL_DIR=${LMMS_EVAL_DIR}"

if [[ -z "${TIEDPO_CKPT}" ]]; then
  echo "TIEDPO_CKPT is required"
  exit 1
fi

BASE_MODEL="${BASE_MODEL}" \
TIEDPO_CKPT="${TIEDPO_CKPT}" \
LMMS_EVAL_DIR="${LMMS_EVAL_DIR}" \
RESULTS_ROOT="${RESULTS_ROOT}" \
GROUP_NAME="${GROUP_NAME}" \
TASKS="${TASKS}" \
NUM_PROCESSES="${NUM_PROCESSES}" \
BATCH_SIZE="${BATCH_SIZE}" \
bash "${SCRIPT_DIR}/run_qwen3vl_2model_lmms_eval_and_summarize.sh"
