#!/usr/bin/env bash
# FBD-LoRA: DreamBench — DreamBooth full fine-tuning
# Status: REAL via Hugging Face Diffusers official training
# Implementation: fbd_lora.imagen.train_dreambooth_full
#
# Usage: bash scripts/imagen/dreambench/train_dreambooth.sh <GPU_IDS> <RUN_EVAL_AFTER_TRAIN>
set -euo pipefail

REPO_ROOT="/home/dongwoo39/papers/fbd_lora/forward-backward-decoupled-low-rank-adaptation"

export TASK="imagen_dreambench"
export METHOD="dreambooth"
export CONFIG_PATH="${REPO_ROOT}/configs/imagen/dreambench/dreambooth.yaml"
export TRAIN_MODULE="fbd_lora.imagen.train_dreambooth_full"
export EVAL_MODULE="fbd_lora.imagen.evaluate_clip_dino"

exec "${REPO_ROOT}/scripts/common/run_experiment.sh" "$@"
