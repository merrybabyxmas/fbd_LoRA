#!/usr/bin/env bash
# Evaluate CLIP-I / DINO / CLIP-T for a CustomConcept101 training run.
# Usage: CONCEPT_NAME=cat_statue bash experiments/imagen/customconcept101/eval_clip_dino.sh <run_dir>
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"
source /home/dongwoo39/.venv/bin/activate

RUN_DIR=${1:?Error: run_dir required}
CONCEPT_NAME=${CONCEPT_NAME:-concept}

CONCEPT_DIR="data/customconcept101/${CONCEPT_NAME}"
GENERATED_DIR="${RUN_DIR}/eval/generated"
OUTPUT_DIR="${RUN_DIR}/eval"
PROMPT="a photo of sks ${CONCEPT_NAME//_/ }"

echo "[customconcept101/eval_clip_dino.sh] run_dir=${RUN_DIR}"
echo "[customconcept101/eval_clip_dino.sh] concept=${CONCEPT_NAME}"

python -m fbd_lora.imagen.evaluate_clip_dino \
  --concept_dir "${CONCEPT_DIR}" \
  --generated_dir "${GENERATED_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --prompt "${PROMPT}" \
  --device cuda
