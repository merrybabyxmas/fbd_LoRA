#!/usr/bin/env bash
# FBD-LoRA: DreamBench+ — FBD-LoRA real-data sanity run (SD1.5, 20 steps)
# Usage: bash scripts/sanity/imagen/dreambench/train_fbd_sd15_dreambench_plus.sh <GPU_IDS> <RUN_EVAL_AFTER_TRAIN>
set -euo pipefail

REPO_ROOT="/home/dongwoo39/papers/fbd_lora/forward-backward-decoupled-low-rank-adaptation"

export TASK="sanity_imagen_dreambench_plus"
export METHOD="fbd"
export CONFIG_PATH="${REPO_ROOT}/configs/sanity/imagen/dreambench/fbd_sd15_dreambench_plus.yaml"
export TRAIN_MODULE="fbd_lora.imagen.train_dreambooth_lora"
export EVAL_MODULE="fbd_lora.imagen.evaluate_clip_dino"

exec "${REPO_ROOT}/scripts/common/run_experiment.sh" "$@"
