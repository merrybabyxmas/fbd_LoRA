#!/usr/bin/env bash
# FBD-LoRA: MetaMath — PiSSA
# Status: REAL via PEFT LoraConfig(init_lora_weights="pissa_niter_16")
# Usage: bash scripts/nlg/metamath/train_pissa.sh <GPU_IDS> <RUN_EVAL_AFTER_TRAIN>
set -euo pipefail

REPO_ROOT="/home/dongwoo39/papers/fbd_lora/forward-backward-decoupled-low-rank-adaptation"

export TASK="nlg_metamath"
export METHOD="pissa"
export CONFIG_PATH="${REPO_ROOT}/configs/nlg/metamath/pissa.yaml"
export TRAIN_MODULE="fbd_lora.nlg.train"
export EVAL_MODULE="fbd_lora.nlg.run_eval_gsm8k"

exec "${REPO_ROOT}/scripts/common/run_experiment.sh" "$@"
