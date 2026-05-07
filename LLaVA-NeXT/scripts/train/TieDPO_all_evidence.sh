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
export WANDB_API_KEY="${WANDB_API_KEY:-}"
echo "pythonpath="$PYTHONPATH
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\///_}"

export WANDB_PROJECT=llava-TieDPO

PROMPT_VERSION="qwen_1_5"

# -------------------------------------------------------
# Paths: only dataset path is changed relative to TieDPO.sh
TRAIN_MODE="${TRAIN_MODE:-lora}"   # lora | full
MODEL_PATH="${MODEL_PATH:-${MODELS_ROOT}/llava-onevision-qwen2-7b-ov}"
DATA_PATH="${DATA_PATH:-${DATA_ROOT}/processed_splits_balanced_16k_with_evidence/normalized_for_training/all_evidence_balanced_ab/train_normalized_all_evidence_balanced_ab.jsonl}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-${DATA_ROOT}/processed_splits_balanced_16k_with_evidence/normalized_for_training/all_evidence_balanced_ab/test_normalized_balanced_ab.jsonl}"
# The normalized jsonl already stores absolute image paths. image_folder is kept as
# a fallback for legacy relative-path samples.
IMAGE_FOLDER="${IMAGE_FOLDER:-${DATA_ROOT}}"
TRAIN_ENTRY="${PROJECT_ROOT}/llava/train/train_tie_dpo.py"
DEEPSPEED_CONFIG="${PROJECT_ROOT}/scripts/zero3.json"
# -------------------------------------------------------

EPOCH="${EPOCH:-1}"
dpo_alpha="${DPO_ALPHA:-1.0}"
beta="${BETA:-0.1}"
lambda_tie="${LAMBDA_TIE:-2.0}"
tie_margin="${TIE_MARGIN:-0.1}"
gamma="${GAMMA:-0.03}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
EVAL_STEPS="${EVAL_STEPS:-50}"
EVAL_STRATEGY="${EVAL_STRATEGY:-}"  # steps | no
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-31952}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
GROUP_BY_MODALITY_LENGTH="${GROUP_BY_MODALITY_LENGTH:-True}"
REPORT_TO="${REPORT_TO:-wandb}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_PARENT_DIR="${OUTPUT_PARENT_DIR:-./ckpt}"
SFT_LOSS_MODE="${SFT_LOSS_MODE:-tie_symmetric}"
SFT_LOSS_MODES="${SFT_LOSS_MODES:-}"
AUTO_EVAL_AFTER_TRAIN="${AUTO_EVAL_AFTER_TRAIN:-1}"
AUTO_EVAL_BASE_MODEL="${AUTO_EVAL_BASE_MODEL:-1}"
AUTO_EVAL_LORA_MODEL="${AUTO_EVAL_LORA_MODEL:-1}"
AUTO_EVAL_BATCH_SIZE="${AUTO_EVAL_BATCH_SIZE:-1}"
AUTO_EVAL_DATALOADER_NUM_WORKERS="${AUTO_EVAL_DATALOADER_NUM_WORKERS:-0}"
AUTO_EVAL_RESULTS_DIR="${AUTO_EVAL_RESULTS_DIR:-${OUTPUT_PARENT_DIR%/}/eval}"
LORA_ENABLE="True"
LORA_R="${LORA_R:-128}"
LORA_ALPHA="${LORA_ALPHA:-256}"
FREEZE_MM_MLP_ADAPTER="${FREEZE_MM_MLP_ADAPTER:-True}"
LORA_LEARNING_RATE="${LORA_LEARNING_RATE:-5e-5}"
FULL_LEARNING_RATE="${FULL_LEARNING_RATE:-1e-5}"

if [[ "${TRAIN_MODE}" == "full" ]]; then
    LORA_ENABLE="False"
    FREEZE_MM_MLP_ADAPTER="False"
    LEARNING_RATE="${LEARNING_RATE:-${FULL_LEARNING_RATE}}"
    EVAL_STRATEGY="${EVAL_STRATEGY:-steps}"
else
    LEARNING_RATE="${LEARNING_RATE:-${LORA_LEARNING_RATE}}"
    EVAL_STRATEGY="${EVAL_STRATEGY:-steps}"
fi

run_one_experiment() {
    local mode="$1"
    local run_name="${RUN_NAME:-llava-onevision-qwen2-7b-ov-beta${beta}-epoch${EPOCH}-lambda_tie${lambda_tie}-TieDPO-all-evidence-${mode}-${RUN_TS}-${TRAIN_MODE}-fix}"
    local clean_name="${run_name##*/}"
    local output_dir="${OUTPUT_PARENT_DIR%/}/${clean_name}"
    export WANDB_NAME="${WANDB_NAME:-$clean_name}"

    echo "${run_name}"
    echo "train_data=${DATA_PATH}"
    echo "eval_data=${EVAL_DATA_PATH}"
    echo "wandb_mode=${WANDB_MODE}"
    echo "sft_loss_mode=${mode}"
    echo "train_mode=${TRAIN_MODE}"
    echo "freeze_mm_mlp_adapter=${FREEZE_MM_MLP_ADAPTER}"
    echo "learning_rate=${LEARNING_RATE}"
    echo "eval_strategy=${EVAL_STRATEGY}"
    echo "output_dir=${output_dir}"

    ACCELERATE_CPU_AFFINITY=1 torchrun --nproc-per-node="${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" \
        "${TRAIN_ENTRY}" \
        --lora_enable "${LORA_ENABLE}" --lora_r "${LORA_R}" --lora_alpha "${LORA_ALPHA}" \
        --freeze_mm_mlp_adapter "${FREEZE_MM_MLP_ADAPTER}" \
        --deepspeed "${DEEPSPEED_CONFIG}" \
        --model_name_or_path="${MODEL_PATH}" \
        --attn_implementation sdpa \
        --dpo_alpha=${dpo_alpha} \
        --beta=${beta} \
        --gamma=${gamma} \
        --sft_loss_mode=${mode} \
        --lambda_tie=${lambda_tie} \
        --tie_margin=${tie_margin} \
        --version $PROMPT_VERSION \
        --data_path=$DATA_PATH \
        --eval_data_path=$EVAL_DATA_PATH \
        --image_folder "${IMAGE_FOLDER}" \
        --unfreeze_mm_vision_tower False \
        --vision_tower ${VISION_MODEL_VERSION} \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --group_by_modality_length "${GROUP_BY_MODALITY_LENGTH}" \
        --image_aspect_ratio pad \
        --image_grid_pinpoints "(1x1),(2x2),(3x3),(4x4),(5x5),(6x6)" \
        --mm_patch_merge_type spatial_unpad \
        --bf16 True \
        --run_name $clean_name \
        --output_dir $output_dir \
        --num_train_epochs $EPOCH \
        --max_prompt_length ${MAX_PROMPT_LENGTH} \
        --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
        --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
        --eval_strategy "${EVAL_STRATEGY}" \
        --eval_steps ${EVAL_STEPS} \
        --save_strategy "steps" \
        --save_steps "${SAVE_STEPS}" \
        --save_total_limit "${SAVE_TOTAL_LIMIT}" \
        --learning_rate "${LEARNING_RATE}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --warmup_ratio "${WARMUP_RATIO}" \
        --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 32768 \
        --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
        --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
        --lazy_preprocess True \
        --report_to "${REPORT_TO}" \
        --dataloader_drop_last True

    if [[ "${AUTO_EVAL_AFTER_TRAIN}" == "1" ]]; then
        local eval_script="${PROJECT_ROOT}/scripts/train/TieDPO_eval_only.sh"
        local eval_root="${AUTO_EVAL_RESULTS_DIR%/}/${clean_name}"
        local base_eval_dir="${eval_root}/base"
        local lora_eval_dir="${eval_root}/lora"
        mkdir -p "${eval_root}"

        if [[ "${AUTO_EVAL_BASE_MODEL}" == "1" ]]; then
            echo "[auto-eval] Evaluating base model on ${EVAL_DATA_PATH}"
            TRAIN_MODE="full" \
            BASE_MODEL_PATH="${MODEL_PATH}" \
            MODEL_PATH="${MODEL_PATH}" \
            LORA_WEIGHT_PATH="" \
            DATA_PATH="${DATA_PATH}" \
            EVAL_DATA_PATH="${EVAL_DATA_PATH}" \
            DPO_ALPHA="${dpo_alpha}" \
            BETA="${beta}" \
            LAMBDA_TIE="${lambda_tie}" \
            TIE_MARGIN="${tie_margin}" \
            GAMMA="${gamma}" \
            MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH}" \
            PER_DEVICE_EVAL_BATCH_SIZE="${AUTO_EVAL_BATCH_SIZE}" \
            DATALOADER_NUM_WORKERS="${AUTO_EVAL_DATALOADER_NUM_WORKERS}" \
            SFT_LOSS_MODE="${mode}" \
            RUN_NAME="${clean_name}-base-eval" \
            OUTPUT_DIR="${base_eval_dir}" \
            bash "${eval_script}"
        fi

        if [[ "${AUTO_EVAL_LORA_MODEL}" == "1" && "${TRAIN_MODE}" == "lora" ]]; then
            echo "[auto-eval] Evaluating trained LoRA checkpoint on ${EVAL_DATA_PATH}"
            TRAIN_MODE="lora" \
            BASE_MODEL_PATH="${MODEL_PATH}" \
            MODEL_PATH="${MODEL_PATH}" \
            LORA_WEIGHT_PATH="${output_dir}" \
            DATA_PATH="${DATA_PATH}" \
            EVAL_DATA_PATH="${EVAL_DATA_PATH}" \
            DPO_ALPHA="${dpo_alpha}" \
            BETA="${beta}" \
            LAMBDA_TIE="${lambda_tie}" \
            TIE_MARGIN="${tie_margin}" \
            GAMMA="${gamma}" \
            MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH}" \
            PER_DEVICE_EVAL_BATCH_SIZE="${AUTO_EVAL_BATCH_SIZE}" \
            DATALOADER_NUM_WORKERS="${AUTO_EVAL_DATALOADER_NUM_WORKERS}" \
            SFT_LOSS_MODE="${mode}" \
            RUN_NAME="${clean_name}-lora-eval" \
            OUTPUT_DIR="${lora_eval_dir}" \
            bash "${eval_script}"
        fi

        SUMMARY_JSON="${eval_root}/summary.json" EVAL_ROOT="${eval_root}" RUN_NAME_VALUE="${clean_name}" python - <<'PY'
import json
import os
from pathlib import Path

eval_root = Path(os.environ["EVAL_ROOT"])
summary_json = Path(os.environ["SUMMARY_JSON"])
run_name = os.environ["RUN_NAME_VALUE"]

result = {"run_name": run_name, "eval_root": str(eval_root), "models": {}}
for model_name, subdir in [("base", "base"), ("lora", "lora")]:
    result_path = eval_root / subdir / "eval_results.json"
    if not result_path.exists():
        continue
    with result_path.open("r", encoding="utf-8") as f:
        result["models"][model_name] = json.load(f)

summary_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[auto-eval] Summary written to {summary_json}")
print(json.dumps(result, ensure_ascii=False, indent=2))
PY
    fi
}

if [[ -n "${SFT_LOSS_MODES}" ]]; then
    IFS=',' read -r -a loss_modes <<< "${SFT_LOSS_MODES}"
    for mode in "${loss_modes[@]}"; do
        export RUN_NAME=""
        export WANDB_NAME=""
        run_one_experiment "${mode}"
    done
else
    run_one_experiment "${SFT_LOSS_MODE}"
fi
