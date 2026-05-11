#!/usr/bin/env bash
set -euo pipefail
cd /home/dongwoo39/papers/fbd_lora/forward-backward-decoupled-low-rank-adaptation
source /home/dongwoo39/.venv/bin/activate

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

python -m fbd_lora.nlg.train \
  --config configs/nlg/metamath_fbd.yaml \
  --seed 42 \
  --smoke_test true \
  --model.backbone sshleifer/tiny-gpt2 \
  --training.max_steps 5 \
  --training.per_device_train_batch_size 2 \
  --training.gradient_accumulation_steps 1 \
  --adapter.rank 2 \
  --adapter.alpha 2 \
  --adapter.target_modules "[c_attn]" \
  --dataset.max_samples 32 \
  --dataset.max_seq_length 128 \
  --wandb.enabled false \
  --fbd.gradient_stats_interval 1 \
  --training.logging_steps 1 2>&1
