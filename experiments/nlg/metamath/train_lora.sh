#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/nlg/baselines/lora.yaml}
SEED=${SEED:-42}

export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=${WANDB_PROJECT:-fbd-lora}
export WANDB_LOG_MODEL=${WANDB_LOG_MODEL:-false}

accelerate launch \
  --config_file configs/accelerate/multi_gpu_ddp.yaml \
  -m fbd_lora.nlg.train \
  --config "${CONFIG}" \
  --seed "${SEED}"
