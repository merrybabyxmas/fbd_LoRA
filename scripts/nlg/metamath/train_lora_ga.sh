#!/usr/bin/env bash
# FBD-LoRA: MetaMath — LoRA-GA (PLACEHOLDER — NOT IMPLEMENTED)
# Usage: bash scripts/nlg/metamath/train_lora_ga.sh <GPU_IDS> <RUN_EVAL_AFTER_TRAIN>
set -euo pipefail

REPO_ROOT="/home/dongwoo39/papers/fbd_lora/forward-backward-decoupled-low-rank-adaptation"

export TASK="nlg_metamath"
export METHOD="lora_ga"
export CONFIG_PATH="${REPO_ROOT}/configs/nlg/metamath/lora_ga.yaml"
export TRAIN_MODULE="fbd_lora.nlg.train"
export EVAL_MODULE="fbd_lora.nlg.run_eval_gsm8k"

exec "${REPO_ROOT}/scripts/common/run_experiment.sh" "$@"
