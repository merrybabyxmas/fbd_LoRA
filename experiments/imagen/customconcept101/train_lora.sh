#!/usr/bin/env bash
# Train standard LoRA baseline on a CustomConcept101 concept.
# Usage: SEED=42 CONCEPT_NAME=cat_statue bash experiments/imagen/customconcept101/train_lora.sh [config]
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"
source /home/dongwoo39/.venv/bin/activate

CONFIG=${1:-configs/imagen/baselines/lora_dreambooth.yaml}
SEED=${SEED:-42}
CONCEPT_NAME=${CONCEPT_NAME:-cat_statue}

export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=${WANDB_PROJECT:-fbd-lora}
export WANDB_ENTITY=${WANDB_ENTITY:-mw990909-sogang-university}
export WANDB_LOG_MODEL=${WANDB_LOG_MODEL:-false}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}

echo "[customconcept101/train_lora.sh] CONFIG=${CONFIG} SEED=${SEED} CONCEPT=${CONCEPT_NAME}"

accelerate launch \
  --config_file configs/accelerate/multi_gpu_ddp.yaml \
  -m fbd_lora.imagen.train_dreambooth_lora \
  --config "${CONFIG}" \
  --seed "${SEED}" \
  --concept_name "${CONCEPT_NAME}"
