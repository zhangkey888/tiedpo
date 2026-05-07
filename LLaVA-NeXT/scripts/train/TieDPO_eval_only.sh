#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
MODELS_ROOT="${MODELS_ROOT:-${REPO_ROOT}/models}"
cd "${PROJECT_ROOT}"
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH"
echo "pythonpath="$PYTHONPATH
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export OMP_NUM_THREADS=8

VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
export WANDB_PROJECT=llava-TieDPO
PROMPT_VERSION="qwen_1_5"

TRAIN_MODE="${TRAIN_MODE:-lora}"   # lora | full
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${MODELS_ROOT}/llava-onevision-qwen2-7b-ov}"
LORA_WEIGHT_PATH="${LORA_WEIGHT_PATH:-}"
MODEL_PATH="${MODEL_PATH:-${BASE_MODEL_PATH}}"
DATA_PATH="${DATA_PATH:-${DATA_ROOT}/processed_splits_balanced_16k_with_evidence/normalized_for_training/all_evidence_balanced_ab/train_normalized_all_evidence_balanced_ab.jsonl}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-${DATA_ROOT}/processed_splits_balanced_16k_with_evidence/normalized_for_training/all_evidence_balanced_ab/test_normalized_balanced_ab.jsonl}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${DATA_ROOT}}"
TRAIN_ENTRY="${PROJECT_ROOT}/llava/train/train_tie_dpo.py"

if [[ "${TRAIN_MODE}" == "lora" && -z "${LORA_WEIGHT_PATH}" ]]; then
    echo "Please set LORA_WEIGHT_PATH to a saved checkpoint directory."
    exit 1
fi

if [[ "${TRAIN_MODE}" == "full" && -n "${LORA_WEIGHT_PATH}" ]]; then
    echo "TRAIN_MODE=full ignores LORA_WEIGHT_PATH; MODEL_PATH will be evaluated directly."
fi

dpo_alpha="${DPO_ALPHA:-1.0}"
beta="${BETA:-0.1}"
lambda_tie="${LAMBDA_TIE:-2.0}"
tie_margin="${TIE_MARGIN:-0.1}"
gamma="${GAMMA:-0.03}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
SFT_LOSS_MODE="${SFT_LOSS_MODE:-tie_symmetric}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
LORA_ENABLE="True"
LORA_R="${LORA_R:-128}"
LORA_ALPHA="${LORA_ALPHA:-256}"
FREEZE_MM_MLP_ADAPTER="${FREEZE_MM_MLP_ADAPTER:-True}"

if [[ "${TRAIN_MODE}" == "full" ]]; then
    LORA_ENABLE="False"
    FREEZE_MM_MLP_ADAPTER="False"
fi

RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"

eval_target_name="$(basename "${MODEL_PATH}")"
if [[ "${TRAIN_MODE}" == "lora" ]]; then
    eval_target_name="$(basename "${LORA_WEIGHT_PATH}")"
fi
RUN_NAME="${RUN_NAME:-eval-${eval_target_name}-${RUN_TS}}"
OUTPUT_DIR="${OUTPUT_DIR:-./eval/${RUN_NAME}}"
mkdir -p "${OUTPUT_DIR}"
export WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}"

echo "${RUN_NAME}"
echo "train_mode=${TRAIN_MODE}"
echo "model_path=${MODEL_PATH}"
echo "lora_weight_path=${LORA_WEIGHT_PATH}"
echo "eval_data=${EVAL_DATA_PATH}"
echo "wandb_mode=${WANDB_MODE}"
echo "sft_loss_mode=${SFT_LOSS_MODE}"

python "${TRAIN_ENTRY}" \
    --eval_only True \
    --lora_enable "${LORA_ENABLE}" --lora_r "${LORA_R}" --lora_alpha "${LORA_ALPHA}" \
    --lora_weight_path "${LORA_WEIGHT_PATH}" \
    --freeze_mm_mlp_adapter "${FREEZE_MM_MLP_ADAPTER}" \
    --model_name_or_path="${MODEL_PATH}" \
    --attn_implementation sdpa \
    --dpo_alpha=${dpo_alpha} \
    --beta=${beta} \
    --gamma=${gamma} \
    --sft_loss_mode=${SFT_LOSS_MODE} \
    --lambda_tie=${lambda_tie} \
    --tie_margin=${tie_margin} \
    --version $PROMPT_VERSION \
    --data_path="${DATA_PATH}" \
    --eval_data_path="${EVAL_DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --unfreeze_mm_vision_tower False \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio pad \
    --image_grid_pinpoints "(1x1),(2x2),(3x3),(4x4),(5x5),(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name "${RUN_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_prompt_length ${MAX_PROMPT_LENGTH} \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --eval_strategy "no" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --lazy_preprocess True \
    --report_to wandb
