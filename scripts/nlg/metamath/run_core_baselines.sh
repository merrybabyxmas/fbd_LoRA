#!/usr/bin/env bash
# FBD-LoRA: MetaMath — run core baselines sequentially
# Usage: bash scripts/nlg/metamath/run_core_baselines.sh <GPU_IDS> <RUN_EVAL_AFTER_TRAIN>
set -euo pipefail

REPO_ROOT="/home/dongwoo39/papers/fbd_lora/forward-backward-decoupled-low-rank-adaptation"

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/nlg/metamath/run_core_baselines.sh <GPU_IDS> <RUN_EVAL_AFTER_TRAIN>"
  exit 1
fi

GPU_IDS="$1"
RUN_EVAL="$2"

echo "[aggregator] Running MetaMath core baselines on GPU(s): ${GPU_IDS}"

echo "[aggregator] (1/2) Running LoRA baseline..."
bash "${REPO_ROOT}/scripts/nlg/metamath/train_lora.sh" "${GPU_IDS}" "${RUN_EVAL}"

echo "[aggregator] (2/2) Running FBD-LoRA..."
bash "${REPO_ROOT}/scripts/nlg/metamath/train_fbd.sh" "${GPU_IDS}" "${RUN_EVAL}"

echo "[aggregator] MetaMath core baselines complete."
