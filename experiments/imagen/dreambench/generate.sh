#!/usr/bin/env bash
# Generate evaluation images from a trained DreamBench adapter.
# Usage: bash experiments/imagen/dreambench/generate.sh <checkpoint_dir> [prompt]
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"
source /home/dongwoo39/.venv/bin/activate

CHECKPOINT_DIR=${1:?Error: checkpoint_dir required}
PROMPT=${2:-"a photo of the subject"}
NUM_IMAGES=${NUM_IMAGES:-4}
SEED=${SEED:-42}

# Derive output dir from checkpoint dir
RUN_DIR="$(dirname "$(dirname "${CHECKPOINT_DIR}")")"
OUTPUT_DIR="${RUN_DIR}/eval/generated"

echo "[generate.sh] checkpoint=${CHECKPOINT_DIR}"
echo "[generate.sh] output_dir=${OUTPUT_DIR}"

python -m fbd_lora.imagen.generate \
  --checkpoint "${CHECKPOINT_DIR}" \
  --prompt "${PROMPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_images "${NUM_IMAGES}" \
  --seed "${SEED}" \
  --num_inference_steps 50 \
  --guidance_scale 7.5
