#!/usr/bin/env bash
# Run full DreamBench FBD pipeline: train -> generate -> evaluate.
# Usage: CONCEPT_NAME=dog SEED=42 bash experiments/imagen/dreambench/run_all.sh [fbd|lora]
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"

ADAPTER=${1:-fbd}
CONCEPT_NAME=${CONCEPT_NAME:-dog}
SEED=${SEED:-42}

echo "=== DreamBench run_all: adapter=${ADAPTER}, concept=${CONCEPT_NAME}, seed=${SEED} ==="

# Step 1: Train
if [ "${ADAPTER}" = "fbd" ]; then
    SEED="${SEED}" CONCEPT_NAME="${CONCEPT_NAME}" \
        bash experiments/imagen/dreambench/train_fbd.sh configs/imagen/dreambench_fbd.yaml
else
    SEED="${SEED}" CONCEPT_NAME="${CONCEPT_NAME}" \
        bash experiments/imagen/dreambench/train_lora.sh configs/imagen/baselines/lora_dreambooth.yaml
fi

# The latest run dir is the most recent outputs/runs/* directory
RUN_DIR=$(ls -td outputs/runs/*_imagen_dreambench_* 2>/dev/null | head -1)
if [ -z "${RUN_DIR}" ]; then
    echo "ERROR: No run directory found." >&2
    exit 1
fi
echo "Run dir: ${RUN_DIR}"

# Step 2: Generate
CHECKPOINT_DIR="${RUN_DIR}/checkpoints/final"
PROMPT="a photo of sks ${CONCEPT_NAME}"
bash experiments/imagen/dreambench/generate.sh "${CHECKPOINT_DIR}" "${PROMPT}"

# Step 3: Evaluate
CONCEPT_REF_DIR="data/dreambench/reference_images/${CONCEPT_NAME}"
bash experiments/imagen/dreambench/eval_clip_dino.sh "${RUN_DIR}" "${CONCEPT_REF_DIR}"

echo "=== run_all complete for ${CONCEPT_NAME} ==="
