#!/usr/bin/env bash
# FBD-LoRA: Conversation — LoRA baseline real-data sanity run (Mistral-7B-v0.1, 20 steps)
# Usage: bash scripts/sanity/nlg/conversation/train_lora_mistral.sh <GPU_IDS> <RUN_EVAL_AFTER_TRAIN>
set -euo pipefail

REPO_ROOT="/home/dongwoo39/papers/fbd_lora/forward-backward-decoupled-low-rank-adaptation"

export TASK="sanity_nlg_conversation_mistral"
export METHOD="lora"
export CONFIG_PATH="${REPO_ROOT}/configs/sanity/nlg/conversation/lora_mistral.yaml"
export TRAIN_MODULE="fbd_lora.nlg.train"
export EVAL_MODULE="fbd_lora.nlg.run_eval_gsm8k"

exec "${REPO_ROOT}/scripts/common/run_experiment.sh" "$@"
