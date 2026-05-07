#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

LMMS_EVAL_DIR="${LMMS_EVAL_DIR:-}"
RUN_NAME="${RUN_NAME:-qwen3vl_lmms_eval}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/workspace/lmms_eval_results}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TASKS="${TASKS:-xlrs-lite}"
MODEL_NAME="${MODEL_NAME:-qwen3_vl_chat}"
PRETRAINED_PATH="${PRETRAINED_PATH:-}"
MODEL_BASE="${MODEL_BASE:-${PROJECT_ROOT}/models/Qwen3-VL-8B-Instruct}"
LORA_CKPT="${LORA_CKPT:-}"
MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-${PROJECT_ROOT}/workspace/merged/${RUN_NAME}}"
MERGE_PYTHON_BIN="${MERGE_PYTHON_BIN:-python}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
HF_HOME="${HF_HOME:-${REPO_ROOT}/hf_cache}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

export HF_ENDPOINT
export HF_HUB_ENABLE_HF_TRANSFER
export HF_HOME
export HUGGINGFACE_HUB_CACHE
export TRANSFORMERS_CACHE
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ -z "${PRETRAINED_PATH}" ]]; then
  if [[ -n "${LORA_CKPT}" ]]; then
    "${MERGE_PYTHON_BIN}" "${SCRIPT_DIR}/merge_qwen3vl_lora.py" \
      --model-base "${MODEL_BASE}" \
      --lora-ckpt "${LORA_CKPT}" \
      --save-path "${MERGED_MODEL_DIR}"
    PRETRAINED_PATH="${MERGED_MODEL_DIR}"
  else
    PRETRAINED_PATH="${MODEL_BASE}"
  fi
fi

if [[ -z "${LMMS_EVAL_DIR}" || ! -d "${LMMS_EVAL_DIR}" ]]; then
  echo "Please set LMMS_EVAL_DIR to your local lmms-eval checkout."
  exit 1
fi

mkdir -p "${RESULTS_DIR}"

cd "${LMMS_EVAL_DIR}"
accelerate launch --num_processes="${NUM_PROCESSES}" \
  -m lmms_eval \
  --model "${MODEL_NAME}" \
  --model_args "pretrained=${PRETRAINED_PATH}" \
  --tasks "${TASKS}" \
  --batch_size "${BATCH_SIZE}" \
  --log_samples \
  --output_path "${RESULTS_DIR}/${RUN_NAME}"
