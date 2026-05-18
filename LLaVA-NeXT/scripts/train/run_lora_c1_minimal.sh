#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/train/TieDPO_all_evidence.sh"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"

TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${DATA_ROOT}/processed_splits_balanced_16k_with_evidence/normalized_for_training/all_evidence_balanced_ab/train_normalized_all_evidence_balanced_ab.jsonl}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-${DATA_ROOT}/processed_splits_balanced_16k_with_evidence/normalized_for_training/all_evidence_balanced_ab/test_normalized_balanced_ab.jsonl}"
STRICT_ONLY_TRAIN_DATA_PATH="${STRICT_ONLY_TRAIN_DATA_PATH:-${DATA_ROOT}/processed_splits_balanced_16k_with_evidence/normalized_for_training/all_evidence_balanced_ab/train_normalized_all_evidence_balanced_ab.strict_only.jsonl}"

RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
EXPERIMENT_GROUP_DIR="${EXPERIMENT_GROUP_DIR:-${PROJECT_ROOT}/ckpt/c1-minimal-${RUN_TS}}"

export TRAIN_MODE="${TRAIN_MODE:-lora}"
export LORA_LEARNING_RATE="${LORA_LEARNING_RATE:-5e-5}"
export LAMBDA_TIE="${LAMBDA_TIE:-2.0}"
export TIE_MARGIN="${TIE_MARGIN:-0.1}"
export GAMMA="${GAMMA:-0.03}"
export EVAL_STEPS="${EVAL_STEPS:-50}"
mkdir -p "${EXPERIMENT_GROUP_DIR}"

echo "[1/4] Build strict-only train split: ${STRICT_ONLY_TRAIN_DATA_PATH}"
python - <<'PY' "${TRAIN_DATA_PATH}" "${STRICT_ONLY_TRAIN_DATA_PATH}"
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
dst.parent.mkdir(parents=True, exist_ok=True)

kept = 0
total = 0
with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        total += 1
        sample = json.loads(line)
        if bool(sample.get("is_tie", False)):
            continue
        fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
        kept += 1

print(f"strict_only_train_samples={kept} / total={total}")
PY

echo "[2/4] Run LoRA tie_symmetric"
RUN_NAME="c1-minimal-tie_symmetric-${RUN_TS}" \
WANDB_NAME="c1-minimal-tie_symmetric-${RUN_TS}" \
OUTPUT_PARENT_DIR="${EXPERIMENT_GROUP_DIR}" \
DATA_PATH="${TRAIN_DATA_PATH}" \
EVAL_DATA_PATH="${EVAL_DATA_PATH}" \
SFT_LOSS_MODES="" \
SFT_LOSS_MODE="tie_symmetric" \
bash "${BASE_TRAIN_SCRIPT}"

echo "[3/4] Run LoRA strict_only"
RUN_NAME="c1-minimal-strict_only-${RUN_TS}" \
WANDB_NAME="c1-minimal-strict_only-${RUN_TS}" \
OUTPUT_PARENT_DIR="${EXPERIMENT_GROUP_DIR}" \
DATA_PATH="${TRAIN_DATA_PATH}" \
EVAL_DATA_PATH="${EVAL_DATA_PATH}" \
SFT_LOSS_MODES="" \
SFT_LOSS_MODE="strict_only" \
bash "${BASE_TRAIN_SCRIPT}"

echo "[4/4] Run LoRA DPO-strict"
RUN_NAME="c1-minimal-dpo_strict-${RUN_TS}" \
WANDB_NAME="c1-minimal-dpo_strict-${RUN_TS}" \
OUTPUT_PARENT_DIR="${EXPERIMENT_GROUP_DIR}" \
DATA_PATH="${STRICT_ONLY_TRAIN_DATA_PATH}" \
EVAL_DATA_PATH="${EVAL_DATA_PATH}" \
SFT_LOSS_MODES="" \
SFT_LOSS_MODE="strict_only" \
LAMBDA_TIE="0.0" \
bash "${BASE_TRAIN_SCRIPT}"

echo "All 3 experiments finished."
echo "Grouped outputs: ${EXPERIMENT_GROUP_DIR}"
