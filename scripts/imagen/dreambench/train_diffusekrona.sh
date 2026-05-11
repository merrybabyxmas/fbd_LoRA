#!/usr/bin/env bash
# FBD-LoRA: DreamBench — DiffuseKronA
# Status: EXTERNAL_REQUIRED
# Official IBM repo: https://github.com/IBM/DiffuseKronA
# Expected local path: external_repos/DiffuseKronA
#
# Usage: bash scripts/imagen/dreambench/train_diffusekrona.sh <GPU_IDS> <RUN_EVAL_AFTER_TRAIN>
set -euo pipefail

REPO_ROOT="/home/dongwoo39/papers/fbd_lora/forward-backward-decoupled-low-rank-adaptation"
DIFFUSEKRONA_PATH="${REPO_ROOT}/external_repos/DiffuseKronA"

# ---------------------------------------------------------------------------
# Check for external repo
# ---------------------------------------------------------------------------
if [[ ! -d "${DIFFUSEKRONA_PATH}" ]]; then
    echo "[ERROR] DiffuseKronA is external-only. Official repo not found."
    echo "[ERROR] Official repo required: https://github.com/IBM/DiffuseKronA"
    echo "[ERROR] Expected local path: ${DIFFUSEKRONA_PATH}"
    echo "[ERROR] Clone the official repo there or disable this baseline:"
    echo "[ERROR]   git clone https://github.com/IBM/DiffuseKronA ${DIFFUSEKRONA_PATH}"
    exit 1
fi

echo "[INFO] DiffuseKronA repo found at: ${DIFFUSEKRONA_PATH}"

GPU_IDS="${1:-0}"
RUN_EVAL="${2:-false}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export LD_LIBRARY_PATH="/home/dongwoo39/.venv/lib/python3.13/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
source /home/dongwoo39/.venv/bin/activate

CONFIG_PATH="${REPO_ROOT}/configs/imagen/dreambench/diffusekrona.yaml"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[DRY RUN] Would call DiffuseKronA training script from: ${DIFFUSEKRONA_PATH}"
    exit 0
fi

# Delegate to official DiffuseKronA repo entry point
if [[ -f "${DIFFUSEKRONA_PATH}/train_diffusekrona.py" ]]; then
    python "${DIFFUSEKRONA_PATH}/train_diffusekrona.py" --config "${CONFIG_PATH}"
elif [[ -f "${DIFFUSEKRONA_PATH}/train.py" ]]; then
    python "${DIFFUSEKRONA_PATH}/train.py" --config "${CONFIG_PATH}"
else
    echo "[ERROR] Could not find DiffuseKronA entry point in: ${DIFFUSEKRONA_PATH}"
    echo "[ERROR] Check the official DiffuseKronA repo for the correct entry point."
    exit 1
fi
