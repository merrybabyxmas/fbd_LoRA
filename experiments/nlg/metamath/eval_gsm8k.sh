#!/usr/bin/env bash
set -euo pipefail

# Usage: bash experiments/nlg/metamath/eval_gsm8k.sh <run_id_or_checkpoint_path>
#   e.g. bash experiments/nlg/metamath/eval_gsm8k.sh outputs/runs/20260510-170005_.../checkpoints/final

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROOT"

CHECKPOINT="${1:-}"
if [[ -z "$CHECKPOINT" ]]; then
    # Auto-detect latest mistral run final checkpoint
    CHECKPOINT=$(ls -td outputs/runs/*/checkpoints/final 2>/dev/null | head -1)
    echo "[eval_gsm8k] Auto-detected checkpoint: $CHECKPOINT"
fi

BASE_MODEL="${BASE_MODEL:-mistralai/Mistral-7B-v0.1}"
MAX_SAMPLES="${MAX_SAMPLES:-}"   # empty = full 1319 test examples
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

# Derive output dir from checkpoint path
RUN_DIR="$(dirname "$(dirname "$CHECKPOINT")")"
OUTPUT_DIR="${RUN_DIR}/eval/gsm8k"

source /home/dongwoo39/.venv/bin/activate

export HF_TOKEN="${HF_TOKEN:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

ARGS=(
    --checkpoint "$CHECKPOINT"
    --base_model "$BASE_MODEL"
    --output_dir "$OUTPUT_DIR"
    --batch_size "$BATCH_SIZE"
    --max_new_tokens "$MAX_NEW_TOKENS"
)
[[ -n "$MAX_SAMPLES" ]] && ARGS+=(--max_samples "$MAX_SAMPLES")

echo "[eval_gsm8k] Checkpoint : $CHECKPOINT"
echo "[eval_gsm8k] Output dir : $OUTPUT_DIR"
echo "[eval_gsm8k] GPU        : $CUDA_VISIBLE_DEVICES"

python -m fbd_lora.nlg.run_eval_gsm8k "${ARGS[@]}"
