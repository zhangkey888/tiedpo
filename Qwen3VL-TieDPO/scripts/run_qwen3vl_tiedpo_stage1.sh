#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_DIR="${PROJECT_ROOT}/configs"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ -n "${HF_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT
fi

MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen3-VL-8B-Instruct}"
REF_MODEL_PATH="${REF_MODEL_PATH:-${MODEL_PATH}}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${DATA_ROOT}/processed_splits_balanced_16k_with_evidence/normalized_for_training/all_evidence_balanced_ab/train_normalized_all_evidence_balanced_ab_minus200_evidence_tie.jsonl}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-${DATA_ROOT}/processed_splits_balanced_16k_with_evidence/normalized_for_training/all_evidence_balanced_ab/test_normalized_balanced_ab_plus200_evidence_tie.jsonl}"

RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
RUN_NAME="${RUN_NAME:-qwen3vl-tie_symmetric-${RUN_TS}}"
OUTPUT_PARENT_DIR="${OUTPUT_PARENT_DIR:-${PROJECT_ROOT}/workspace/outputs}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_PARENT_DIR}/${RUN_NAME}}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29629}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"
EVAL_STEPS="${EVAL_STEPS:-100}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
REPORT_TO="${REPORT_TO:-none}"
DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-False}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-32768}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_PIXELS="${MAX_PIXELS:-602112}"
MIN_PIXELS="${MIN_PIXELS:-12544}"
ACCELERATOR_CONFIG="${ACCELERATOR_CONFIG:-{\"dispatch_batches\": false, \"split_batches\": false, \"even_batches\": true, \"use_seedable_sampler\": true}}"
LORA_R="${LORA_R:-128}"
LORA_ALPHA="${LORA_ALPHA:-256}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"

DPO_ALPHA="${DPO_ALPHA:-1.0}"
BETA="${BETA:-0.1}"
GAMMA="${GAMMA:-0.03}"
LAMBDA_TIE="${LAMBDA_TIE:-2.0}"
TIE_MARGIN="${TIE_MARGIN:-0.1}"
SFT_LOSS_MODE="${SFT_LOSS_MODE:-tie_symmetric}"
DO_EVAL="${DO_EVAL:-True}"
EVAL_STRATEGY="${EVAL_STRATEGY:-steps}"
USE_FSDP="${USE_FSDP:-False}"
FSDP_MODE="${FSDP_MODE:-full_shard auto_wrap}"
FSDP_CONFIG_PATH="${FSDP_CONFIG_PATH:-${CONFIG_DIR}/fsdp_full_shard_auto_wrap.json}"
USE_DEEPSPEED="${USE_DEEPSPEED:-False}"
DEEPSPEED_CONFIG_PATH="${DEEPSPEED_CONFIG_PATH:-${CONFIG_DIR}/zero3.json}"

if [[ ! -f "${TRAIN_DATA_PATH}" || ! -f "${EVAL_DATA_PATH}" ]]; then
  echo "missing TRAIN_DATA_PATH or EVAL_DATA_PATH"
  echo "TRAIN_DATA_PATH=${TRAIN_DATA_PATH}"
  echo "EVAL_DATA_PATH=${EVAL_DATA_PATH}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${CONFIG_DIR}"

CMD=(
  torchrun
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_port "${MASTER_PORT}"
  -m qwen3vl_tiedpo.run_tie_dpo
  --model_name_or_path "${MODEL_PATH}"
  --ref_model_name_or_path "${REF_MODEL_PATH}"
  --train_data_path "${TRAIN_DATA_PATH}"
  --eval_data_path "${EVAL_DATA_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --run_name "${RUN_NAME}"
  --do_train True
  --do_eval "${DO_EVAL}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning_rate "${LEARNING_RATE}"
  --logging_steps "${LOGGING_STEPS}"
  --save_strategy "${SAVE_STRATEGY}"
  --eval_strategy "${EVAL_STRATEGY}"
  --bf16 True
  --weight_decay "${WEIGHT_DECAY}"
  --warmup_ratio "${WARMUP_RATIO}"
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}"
  --max_prompt_length "${MAX_PROMPT_LENGTH}"
  --model_max_length "${MODEL_MAX_LENGTH}"
  --max_pixels "${MAX_PIXELS}"
  --min_pixels "${MIN_PIXELS}"
  --lora_r "${LORA_R}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout "${LORA_DROPOUT}"
  --dpo_alpha "${DPO_ALPHA}"
  --beta "${BETA}"
  --gamma "${GAMMA}"
  --lambda_tie "${LAMBDA_TIE}"
  --tie_margin "${TIE_MARGIN}"
  --sft_loss_mode "${SFT_LOSS_MODE}"
  --report_to "${REPORT_TO}"
  --ddp_find_unused_parameters "${DDP_FIND_UNUSED_PARAMETERS}"
  --accelerator_config "${ACCELERATOR_CONFIG}"
)

if [[ "${DO_EVAL}" == "True" || "${DO_EVAL}" == "true" ]]; then
  CMD+=(--eval_steps "${EVAL_STEPS}")
fi

if [[ "${SAVE_STRATEGY}" == "steps" ]]; then
  CMD+=(--save_steps "${SAVE_STEPS}")
fi

if [[ "${USE_DEEPSPEED}" == "True" || "${USE_DEEPSPEED}" == "true" ]]; then
  if [[ ! -f "${DEEPSPEED_CONFIG_PATH}" ]]; then
    echo "DeepSpeed config not found: ${DEEPSPEED_CONFIG_PATH}"
    exit 1
  fi
  CMD+=(--deepspeed "${DEEPSPEED_CONFIG_PATH}")
elif [[ "${USE_FSDP}" == "True" || "${USE_FSDP}" == "true" ]]; then
  if [[ ! -f "${FSDP_CONFIG_PATH}" ]]; then
    echo "FSDP config not found: ${FSDP_CONFIG_PATH}"
    exit 1
  fi
  CMD+=(--fsdp "${FSDP_MODE}" --fsdp_config "${FSDP_CONFIG_PATH}")
fi

"${CMD[@]}"
