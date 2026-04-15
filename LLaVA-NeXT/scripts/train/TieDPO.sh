#!/bin/bash
export PYTHONPATH=$PYTHONPATH:`realpath .`
echo "pythonpath="$PYTHONPATH

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\///_}"

export WANDB_PROJECT=llava-TieDPO
export WANDB_NAME=TieDPO_v1

PROMPT_VERSION="qwen_1_5"

# -------------------------------------------------------
# Paths: modify these to match your environment
SFT_MODEL="/home/yuanyuan06/MLLMImage/zhangkeyao/model/llava-onevision-qwen2-7b-ov"
DATA_PATH="/home/kuai-blm/zhangkeyao/mantis/low_token_tie_output/tie_dpo_train.json"
IMAGE_FOLDER="/home/kuai-blm/zhangkeyao/mantis/mantis_export/visual_story_telling/train/images/"
# -------------------------------------------------------

EPOCH=1
beta=0.1
lambda_tie=1.0
tie_margin=0.0

RUN_NAME="llava-onevision-qwen2-7b-ov-beta${beta}-epoch${EPOCH}-lambda_tie${lambda_tie}-TieDPO-415"
CLEAN_NAME="${RUN_NAME##*/}"
OUTPUT_DIR="./ckpt/${CLEAN_NAME}"

echo $RUN_NAME

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc-per-node=8 --master_port 31952 \
    llava/train/train_tie_dpo.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 \
    --freeze_mm_mlp_adapter True \
    --deepspeed scripts/zero3.json \
    --model_name_or_path=${SFT_MODEL} \
    --dpo_alpha=1.0 \
    --beta=${beta} \
    --gamma=0.1 \
    --lambda_tie=${lambda_tie} \
    --tie_margin=${tie_margin} \
    --version $PROMPT_VERSION \
    --data_path=$DATA_PATH \
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
    --run_name $CLEAN_NAME \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --dataloader_drop_last True
