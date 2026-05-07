#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
MODELS_ROOT="${MODELS_ROOT:-${REPO_ROOT}/models}"

LMMS_EVAL_DIR="${LMMS_EVAL_DIR:-}"
BASE_MODEL="${BASE_MODEL:-${MODELS_ROOT}/llava-onevision-qwen2-7b-ov}"
LORA_CKPT="${LORA_CKPT:-}"
MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-}"
RESULTS_DIR="${RESULTS_DIR:-${LMMS_EVAL_DIR}/results}"
RUN_NAME="${RUN_NAME:-}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TASKS="${TASKS:-xlrs-lite}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
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

if [[ -z "${LORA_CKPT}" ]]; then
    echo "Please set LORA_CKPT to your LoRA checkpoint directory."
    exit 1
fi

if [[ -z "${LMMS_EVAL_DIR}" || ! -d "${LMMS_EVAL_DIR}" ]]; then
    echo "Please set LMMS_EVAL_DIR to your local lmms-eval checkout."
    exit 1
fi

if [[ -z "${RUN_NAME}" ]]; then
    RUN_NAME="$(basename "${LORA_CKPT}")"
fi

if [[ -z "${MERGED_MODEL_DIR}" ]]; then
    MERGED_MODEL_DIR="${PROJECT_ROOT}/ckpt/merged/${RUN_NAME}-merged"
fi

mkdir -p "${RESULTS_DIR}"

echo "project_root=${PROJECT_ROOT}"
echo "lmms_eval_dir=${LMMS_EVAL_DIR}"
echo "base_model=${BASE_MODEL}"
echo "lora_ckpt=${LORA_CKPT}"
echo "merged_model_dir=${MERGED_MODEL_DIR}"
echo "results_dir=${RESULTS_DIR}/${RUN_NAME}"
echo "tasks=${TASKS}"
echo "hf_endpoint=${HF_ENDPOINT}"
echo "hf_token_set=$([[ -n "${HF_TOKEN}" ]] && echo yes || echo no)"
echo "model_name=${MODEL_NAME}"
echo "conv_template=${CONV_TEMPLATE}"

cd "${PROJECT_ROOT}"

python "${PROJECT_ROOT}/scripts/eval/merge_lora_for_lmms_eval.py" \
    --model-base "${BASE_MODEL}" \
    --model-path "${LORA_CKPT}" \
    --save-path "${MERGED_MODEL_DIR}" \
    --torch-dtype "${TORCH_DTYPE}"

cd "${LMMS_EVAL_DIR}"

accelerate launch --num_processes="${NUM_PROCESSES}" \
    -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained="${MERGED_MODEL_DIR}",model_name="${MODEL_NAME}",conv_template="${CONV_TEMPLATE}" \
    --tasks "${TASKS}" \
    --batch_size "${BATCH_SIZE}" \
    --log_samples \
    --output_path "${RESULTS_DIR}/${RUN_NAME}"

echo "lmms-eval results saved to: ${RESULTS_DIR}/${RUN_NAME}"
