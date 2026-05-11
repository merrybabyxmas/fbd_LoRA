#!/usr/bin/env bash
# FBD-LoRA: DreamBench — run core baselines sequentially
# Usage: bash scripts/imagen/dreambench/run_core_baselines.sh <GPU_IDS> <RUN_EVAL_AFTER_TRAIN>
set -euo pipefail

REPO_ROOT="/home/dongwoo39/papers/fbd_lora/forward-backward-decoupled-low-rank-adaptation"

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/imagen/dreambench/run_core_baselines.sh <GPU_IDS> <RUN_EVAL_AFTER_TRAIN>"
  exit 1
fi

GPU_IDS="$1"
RUN_EVAL="$2"

echo "[aggregator] Running DreamBench core baselines on GPU(s): ${GPU_IDS}"

echo "[aggregator] (1/2) Running LoRA DreamBooth baseline..."
bash "${REPO_ROOT}/scripts/imagen/dreambench/train_lora.sh" "${GPU_IDS}" "${RUN_EVAL}"

echo "[aggregator] (2/2) Running FBD-LoRA DreamBooth..."
bash "${REPO_ROOT}/scripts/imagen/dreambench/train_fbd.sh" "${GPU_IDS}" "${RUN_EVAL}"

echo "[aggregator] DreamBench core baselines complete."
