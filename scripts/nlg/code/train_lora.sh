#!/usr/bin/env bash
# FBD-LoRA: Code Generation — LoRA baseline (PLACEHOLDER — NOT IMPLEMENTED)
# Usage: bash scripts/nlg/code/train_lora.sh <GPU_IDS> <RUN_EVAL_AFTER_TRAIN>
set -euo pipefail

REPO_ROOT="/home/dongwoo39/papers/fbd_lora/forward-backward-decoupled-low-rank-adaptation"

export TASK="nlg_code"
export METHOD="lora"
export CONFIG_PATH="${REPO_ROOT}/configs/nlg/code/lora.yaml"
export TRAIN_MODULE="fbd_lora.nlg.train"
export EVAL_MODULE="fbd_lora.nlg.run_eval_gsm8k"

exec "${REPO_ROOT}/scripts/common/run_experiment.sh" "$@"
