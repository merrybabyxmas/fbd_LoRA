#!/usr/bin/env bash
# FBD-LoRA: DreamBench — SVDiff
# Status: EXTERNAL_REQUIRED
# Official repo: https://github.com/mkshing/svdiff-pytorch
# Expected local path: external_repos/SVDiff
#
# Usage: bash scripts/imagen/dreambench/train_svdiff.sh <GPU_IDS> <RUN_EVAL_AFTER_TRAIN>
set -euo pipefail

REPO_ROOT="/home/dongwoo39/papers/fbd_lora/forward-backward-decoupled-low-rank-adaptation"
SVDIFF_PATH="${REPO_ROOT}/external_repos/SVDiff"

# ---------------------------------------------------------------------------
# Check for external repo
# ---------------------------------------------------------------------------
if [[ ! -d "${SVDIFF_PATH}" ]]; then
    echo "[ERROR] SVDiff is external-only. Official repo not found."
    echo "[ERROR] Official repo required: https://github.com/mkshing/svdiff-pytorch"
    echo "[ERROR] Expected local path: ${SVDIFF_PATH}"
    echo "[ERROR] Clone the official repo there or disable this baseline:"
    echo "[ERROR]   git clone https://github.com/mkshing/svdiff-pytorch ${SVDIFF_PATH}"
    exit 1
fi

echo "[INFO] SVDiff repo found at: ${SVDIFF_PATH}"

GPU_IDS="${1:-0}"
RUN_EVAL="${2:-false}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export LD_LIBRARY_PATH="/home/dongwoo39/.venv/lib/python3.13/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
source /home/dongwoo39/.venv/bin/activate

CONFIG_PATH="${REPO_ROOT}/configs/imagen/dreambench/svdiff.yaml"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[DRY RUN] Would call SVDiff training script from: ${SVDIFF_PATH}"
    exit 0
fi

# Delegate to official SVDiff repo entry point
if [[ -f "${SVDIFF_PATH}/train_svdiff.py" ]]; then
    python "${SVDIFF_PATH}/train_svdiff.py" --config "${CONFIG_PATH}"
elif [[ -f "${SVDIFF_PATH}/train.py" ]]; then
    python "${SVDIFF_PATH}/train.py" --config "${CONFIG_PATH}"
else
    echo "[ERROR] Could not find SVDiff entry point in: ${SVDIFF_PATH}"
    echo "[ERROR] Check the official SVDiff repo for the correct entry point."
    exit 1
fi
