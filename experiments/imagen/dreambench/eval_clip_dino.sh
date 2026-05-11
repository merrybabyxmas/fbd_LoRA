#!/usr/bin/env bash
# Evaluate CLIP-I / DINO / CLIP-T for a DreamBench run.
# Usage: bash experiments/imagen/dreambench/eval_clip_dino.sh <run_dir> [concept_dir]
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"
source /home/dongwoo39/.venv/bin/activate

RUN_DIR=${1:?Error: run_dir required}
CONCEPT_DIR=${2:-data/dreambench/reference_images}
GENERATED_DIR="${RUN_DIR}/eval/generated"
OUTPUT_DIR="${RUN_DIR}/eval"

echo "[eval_clip_dino.sh] run_dir=${RUN_DIR}"
echo "[eval_clip_dino.sh] concept_dir=${CONCEPT_DIR}"
echo "[eval_clip_dino.sh] generated_dir=${GENERATED_DIR}"

python -m fbd_lora.imagen.evaluate_clip_dino \
  --concept_dir "${CONCEPT_DIR}" \
  --generated_dir "${GENERATED_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --device cuda
