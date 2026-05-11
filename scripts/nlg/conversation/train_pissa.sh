#!/usr/bin/env bash
# FBD-LoRA: Conversation — PiSSA (PLACEHOLDER — NOT IMPLEMENTED)
# Usage: bash scripts/nlg/conversation/train_pissa.sh <GPU_IDS> <RUN_EVAL_AFTER_TRAIN>
set -euo pipefail

REPO_ROOT="/home/dongwoo39/papers/fbd_lora/forward-backward-decoupled-low-rank-adaptation"

export TASK="nlg_conversation"
export METHOD="pissa"
export CONFIG_PATH="${REPO_ROOT}/configs/nlg/conversation/pissa.yaml"
export TRAIN_MODULE="fbd_lora.nlg.train"
export EVAL_MODULE="fbd_lora.nlg.run_eval_gsm8k"

exec "${REPO_ROOT}/scripts/common/run_experiment.sh" "$@"
