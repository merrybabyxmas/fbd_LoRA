#!/usr/bin/env bash
# FBD-LoRA: DreamBench — DiffuseKrona (PLACEHOLDER — NOT IMPLEMENTED)
# Usage: bash scripts/imagen/dreambench/train_diffusekrona.sh <GPU_IDS> <RUN_EVAL_AFTER_TRAIN>
set -euo pipefail

REPO_ROOT="/home/dongwoo39/papers/fbd_lora/forward-backward-decoupled-low-rank-adaptation"

export TASK="imagen_dreambench"
export METHOD="diffusekrona"
export CONFIG_PATH="${REPO_ROOT}/configs/imagen/dreambench/diffusekrona.yaml"
export TRAIN_MODULE="fbd_lora.imagen.train_dreambooth_lora"
export EVAL_MODULE="fbd_lora.imagen.evaluate_clip_dino"

exec "${REPO_ROOT}/scripts/common/run_experiment.sh" "$@"
