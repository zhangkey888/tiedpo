#!/bin/bash
# Setup script for TieDPO training environment.
# Run this once after creating a new conda/venv environment.
# Usage: bash setup_env.sh [conda_env_name]
#
# After running this script, train with:
#   cd LLaVA-NeXT && bash scripts/train/TieDPO.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[1/3] Installing Python dependencies..."
pip install -r "${SCRIPT_DIR}/requirements.txt"

echo "[2/3] Patching system trl with custom trainers..."
TRL_TRAINER_DIR=$(python3 -c "import trl.trainer, os; print(os.path.dirname(trl.trainer.__file__))")
echo "trl trainer path: ${TRL_TRAINER_DIR}"

LOCAL_TRAINER_DIR="${SCRIPT_DIR}/trl/trainer"

for f in "${LOCAL_TRAINER_DIR}"/*.py; do
    fname=$(basename "$f")
    if [ "$fname" = "__init__.py" ]; then
        continue
    fi
    echo "  Copying ${fname} -> ${TRL_TRAINER_DIR}/${fname}"
    cp "$f" "${TRL_TRAINER_DIR}/${fname}"
done

echo "  Patching ${TRL_TRAINER_DIR}/__init__.py to export custom trainers..."
TRL_INIT="${TRL_TRAINER_DIR}/__init__.py"
grep -q "VDPOTrainer" "${TRL_INIT}" || cat >> "${TRL_INIT}" << 'EOF'

# Custom trainers added by TieDPO setup_env.sh
from .vdpo_trainer import VDPOTrainer
from .vdpo_trainer_svco_triples import SVCOTrainer
from .tie_dpo_trainer import TieDPOTrainer
EOF

echo "[3/3] Verifying installation..."
python3 -c "
from trl.trainer import DPOTrainer, TieDPOTrainer
from transformers.trainer_pt_utils import AcceleratorConfig
print('OK: all imports successful')
"

echo ""
echo "Setup complete. Run training with:"
echo "  cd ${SCRIPT_DIR} && bash scripts/train/TieDPO.sh"
