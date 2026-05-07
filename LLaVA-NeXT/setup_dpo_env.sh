#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
ENV_NAME="${1:-dpo}"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found in PATH."
    echo "Please load conda first, then rerun this script."
    exit 1
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "[0/5] Project root: ${PROJECT_ROOT}"
echo "[0/5] Target conda env: ${ENV_NAME}"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "[1/5] Conda env '${ENV_NAME}' already exists, reusing it."
else
    echo "[1/5] Creating conda env '${ENV_NAME}' with Python 3.11..."
    # Use conda-forge only to avoid Anaconda defaults channel ToS prompts.
    conda create -n "${ENV_NAME}" --override-channels -c conda-forge python=3.11 -y
fi

echo "[2/5] Activating env '${ENV_NAME}'..."
conda activate "${ENV_NAME}"

echo "[3/5] Upgrading pip tooling..."
python -m pip install --upgrade pip setuptools wheel

echo "[4/5] Installing build/runtime dependencies with pip..."
python -m pip install cmake ninja

echo "[5/5] Installing project requirements and patching local trl..."
bash "${PROJECT_ROOT}/setup_env.sh"

echo ""
echo "Done. To use this environment:"
echo "  source \"${CONDA_BASE}/etc/profile.d/conda.sh\""
echo "  conda activate ${ENV_NAME}"
echo "  cd ${PROJECT_ROOT}"
echo "  bash scripts/train/run_lora_c1_minimal.sh"
