#!/usr/bin/env bash
# FBD-LoRA: DreamBench — Custom Diffusion
# Status: REAL via Hugging Face Diffusers official Custom Diffusion training
# Implementation: fbd_lora.imagen.train_custom_diffusion
#
# Usage: bash scripts/imagen/dreambench/train_custom_diffusion.sh <GPU_IDS> <RUN_EVAL_AFTER_TRAIN>
set -euo pipefail

REPO_ROOT="/home/dongwoo39/papers/fbd_lora/forward-backward-decoupled-low-rank-adaptation"

export TASK="imagen_dreambench"
export METHOD="custom_diffusion"
export CONFIG_PATH="${REPO_ROOT}/configs/imagen/dreambench/custom_diffusion.yaml"
export TRAIN_MODULE="fbd_lora.imagen.train_custom_diffusion"
export EVAL_MODULE="fbd_lora.imagen.evaluate_clip_dino"

exec "${REPO_ROOT}/scripts/common/run_experiment.sh" "$@"
