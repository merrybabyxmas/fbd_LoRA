#!/usr/bin/env bash
# Full CustomConcept101 pipeline: train -> generate -> evaluate.
# Usage: CONCEPT_NAME=cat_statue SEED=42 bash experiments/imagen/customconcept101/run_all.sh [fbd|lora]
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"

ADAPTER=${1:-fbd}
CONCEPT_NAME=${CONCEPT_NAME:-cat_statue}
SEED=${SEED:-42}

echo "=== CustomConcept101 run_all: adapter=${ADAPTER}, concept=${CONCEPT_NAME}, seed=${SEED} ==="

# Step 1: Train
if [ "${ADAPTER}" = "fbd" ]; then
    SEED="${SEED}" CONCEPT_NAME="${CONCEPT_NAME}" \
        bash experiments/imagen/customconcept101/train_fbd.sh configs/imagen/customconcept101_fbd.yaml
else
    SEED="${SEED}" CONCEPT_NAME="${CONCEPT_NAME}" \
        bash experiments/imagen/customconcept101/train_lora.sh configs/imagen/baselines/lora_dreambooth.yaml
fi

# Find the latest run dir for this concept
RUN_DIR=$(ls -td outputs/runs/*_imagen_customconcept101_${CONCEPT_NAME}_* 2>/dev/null | head -1)
if [ -z "${RUN_DIR}" ]; then
    # Try without concept name filter
    RUN_DIR=$(ls -td outputs/runs/*_imagen_customconcept101_* 2>/dev/null | head -1)
fi
if [ -z "${RUN_DIR}" ]; then
    echo "ERROR: No run directory found." >&2
    exit 1
fi
echo "Run dir: ${RUN_DIR}"

# Step 2: Generate
CONCEPT_NAME="${CONCEPT_NAME}" \
    bash experiments/imagen/customconcept101/generate.sh "${RUN_DIR}/checkpoints/final"

# Step 3: Evaluate
CONCEPT_NAME="${CONCEPT_NAME}" \
    bash experiments/imagen/customconcept101/eval_clip_dino.sh "${RUN_DIR}"

echo "=== run_all complete for concept=${CONCEPT_NAME} ==="
