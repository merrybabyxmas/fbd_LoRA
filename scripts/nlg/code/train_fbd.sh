#!/usr/bin/env bash
# FBD-LoRA: Code Generation — FBD-LoRA (PLACEHOLDER — NOT IMPLEMENTED)
# Usage: bash scripts/nlg/code/train_fbd.sh <GPU_IDS> <RUN_EVAL_AFTER_TRAIN>
set -euo pipefail

REPO_ROOT="/home/dongwoo39/papers/fbd_lora/forward-backward-decoupled-low-rank-adaptation"

export TASK="nlg_code"
export METHOD="fbd"
export CONFIG_PATH="${REPO_ROOT}/configs/nlg/code/fbd.yaml"
export TRAIN_MODULE="fbd_lora.nlg.train"
export EVAL_MODULE="fbd_lora.nlg.run_eval_gsm8k"

exec "${REPO_ROOT}/scripts/common/run_experiment.sh" "$@"
