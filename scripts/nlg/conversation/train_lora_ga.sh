#!/usr/bin/env bash
# FBD-LoRA: Conversation — LoRA-GA
# Status: EXTERNAL_REQUIRED
# Official repo: https://github.com/Outsider565/LoRA-GA
# Expected local path: external_repos/LoRA-GA
#
# Usage: bash scripts/nlg/conversation/train_lora_ga.sh <GPU_IDS> <RUN_EVAL_AFTER_TRAIN>
set -euo pipefail

REPO_ROOT="/home/dongwoo39/papers/fbd_lora/forward-backward-decoupled-low-rank-adaptation"
LORA_GA_PATH="${REPO_ROOT}/external_repos/LoRA-GA"

# ---------------------------------------------------------------------------
# Check for external repo
# ---------------------------------------------------------------------------
if [[ ! -d "${LORA_GA_PATH}" ]]; then
    echo "[ERROR] LoRA-GA is external-only. Official repo not found."
    echo "[ERROR] Official repo required: https://github.com/Outsider565/LoRA-GA"
    echo "[ERROR] Expected local path: ${LORA_GA_PATH}"
    echo "[ERROR] Clone the official repo there or disable this baseline:"
    echo "[ERROR]   git clone https://github.com/Outsider565/LoRA-GA ${LORA_GA_PATH}"
    exit 1
fi

echo "[INFO] LoRA-GA repo found at: ${LORA_GA_PATH}"

GPU_IDS="${1:-0}"
RUN_EVAL="${2:-false}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export LD_LIBRARY_PATH="/home/dongwoo39/.venv/lib/python3.13/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
source /home/dongwoo39/.venv/bin/activate

CONFIG_PATH="${REPO_ROOT}/configs/nlg/conversation/lora_ga.yaml"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[DRY RUN] Would call LoRA-GA training script from: ${LORA_GA_PATH}"
    exit 0
fi

if [[ -f "${LORA_GA_PATH}/train.py" ]]; then
    python "${LORA_GA_PATH}/train.py" --config "${CONFIG_PATH}"
elif [[ -f "${LORA_GA_PATH}/run_lora_ga.py" ]]; then
    python "${LORA_GA_PATH}/run_lora_ga.py" --config "${CONFIG_PATH}"
else
    echo "[ERROR] Could not find LoRA-GA entry point in: ${LORA_GA_PATH}"
    echo "[ERROR] Check the official LoRA-GA repo for the correct entry point."
    exit 1
fi
