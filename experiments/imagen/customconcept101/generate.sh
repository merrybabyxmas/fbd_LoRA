#!/usr/bin/env bash
# Generate evaluation images for a trained CustomConcept101 adapter.
# Usage: CONCEPT_NAME=cat_statue bash experiments/imagen/customconcept101/generate.sh <checkpoint_dir>
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"
source /home/dongwoo39/.venv/bin/activate

CHECKPOINT_DIR=${1:?Error: checkpoint_dir required}
CONCEPT_NAME=${CONCEPT_NAME:-concept}
NUM_IMAGES=${NUM_IMAGES:-4}
SEED=${SEED:-42}

# Derive output dir from checkpoint dir
RUN_DIR="$(dirname "$(dirname "${CHECKPOINT_DIR}")")"
OUTPUT_DIR="${RUN_DIR}/eval/generated"

echo "[customconcept101/generate.sh] checkpoint=${CHECKPOINT_DIR}"
echo "[customconcept101/generate.sh] concept=${CONCEPT_NAME}"

# Generate for multiple evaluation prompts
PROMPTS=(
    "a photo of sks ${CONCEPT_NAME//_/ }"
    "a photo of sks ${CONCEPT_NAME//_/ } in the park"
    "a photo of sks ${CONCEPT_NAME//_/ } on a white table"
    "a painting of sks ${CONCEPT_NAME//_/ }"
    "a photo of sks ${CONCEPT_NAME//_/ } with a city in the background"
)

for PROMPT in "${PROMPTS[@]}"; do
    python -m fbd_lora.imagen.generate \
      --checkpoint "${CHECKPOINT_DIR}" \
      --prompt "${PROMPT}" \
      --output_dir "${OUTPUT_DIR}" \
      --num_images "${NUM_IMAGES}" \
      --seed "${SEED}" \
      --num_inference_steps 50 \
      --guidance_scale 7.5
done

echo "[customconcept101/generate.sh] Generation complete for ${CONCEPT_NAME}."
