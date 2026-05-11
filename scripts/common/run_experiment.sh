#!/usr/bin/env bash
# FBD-LoRA common experiment launcher.
# Called by wrapper scripts in scripts/nlg/ and scripts/imagen/.
# Wrapper scripts must export: TASK, METHOD, CONFIG_PATH, TRAIN_MODULE, EVAL_MODULE
set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
if [[ $# -lt 2 ]]; then
  echo "Usage:"
  echo "  bash <script_path> <GPU_IDS> <RUN_EVAL_AFTER_TRAIN>"
  echo ""
  echo "Examples:"
  echo "  bash scripts/nlg/metamath/train_fbd.sh 0 true"
  echo "  bash scripts/nlg/metamath/train_fbd.sh 0,1 false"
  exit 1
fi

GPU_IDS="$1"
RUN_EVAL_RAW="$2"

case "${RUN_EVAL_RAW,,}" in
  true|1|yes|y)
    RUN_EVAL_AFTER_TRAIN="true"
    ;;
  false|0|no|n)
    RUN_EVAL_AFTER_TRAIN="false"
    ;;
  *)
    echo "[ERROR] Invalid RUN_EVAL_AFTER_TRAIN value: ${RUN_EVAL_RAW}"
    echo "Allowed: true, false, 1, 0, yes, no, y, n"
    exit 1
    ;;
esac

# ---------------------------------------------------------------------------
# Paths and environment
# ---------------------------------------------------------------------------
REPO_ROOT="/home/dongwoo39/papers/fbd_lora/forward-backward-decoupled-low-rank-adaptation"
CONFIG_ROOT="${REPO_ROOT}/configs"

# Validate required wrapper exports
: "${TASK:?TASK must be set by the wrapper script}"
: "${METHOD:?METHOD must be set by the wrapper script}"
: "${CONFIG_PATH:?CONFIG_PATH must be set by the wrapper script}"
: "${TRAIN_MODULE:?TRAIN_MODULE must be set by the wrapper script}"
: "${EVAL_MODULE:?EVAL_MODULE must be set by the wrapper script}"

cd "${REPO_ROOT}"

# LD_LIBRARY_PATH fix for cuSPARSELt
export LD_LIBRARY_PATH="/home/dongwoo39/.venv/lib/python3.13/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"

# Activate venv
source /home/dongwoo39/.venv/bin/activate

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export TOKENIZERS_PARALLELISM=false

# NCCL fix — RTX 4000 series does not support P2P or IB; disable unconditionally.
# This is safe for all GPU generations and required for RTX 4xxx on single-GPU runs.
NUM_GPUS=$(echo "${GPU_IDS}" | awk -F',' '{print NF}')
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Load .env if present (never print secret values)
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/.env"
  set +a
fi

# ---------------------------------------------------------------------------
# Secret presence checks (masked)
# ---------------------------------------------------------------------------
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "[INFO] WANDB_API_KEY is set."
else
  echo "[INFO] WANDB_API_KEY is not set. W&B may require prior login."
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "[INFO] HF_TOKEN is set."
else
  echo "[INFO] HF_TOKEN is not set."
fi

# ---------------------------------------------------------------------------
# Config file validation
# ---------------------------------------------------------------------------
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] Config file not found: ${CONFIG_PATH}"
  exit 1
fi

# Check for placeholder / unimplemented configs
if grep -q "implemented: false" "${CONFIG_PATH}" 2>/dev/null; then
  echo "[ERROR] This method is listed in the experiment matrix but is not implemented yet."
  echo "[ERROR] Config file: ${CONFIG_PATH}"
  echo "[ERROR] Task=${TASK}, Method=${METHOD}"
  echo "[ERROR] Please implement the config before running this script."
  exit 1
fi

# Python config validator
if python -m fbd_lora.config --validate --config "${CONFIG_PATH}" --mode train 2>/dev/null; then
  echo "[INFO] Config validation passed."
else
  echo "[WARNING] fbd_lora.config --validate not available or returned non-zero. Skipping CLI validation."
fi

# ---------------------------------------------------------------------------
# Python module availability check
# ---------------------------------------------------------------------------
if ! python - <<PY
import importlib, sys
for mod in ["${TRAIN_MODULE}"]:
    try:
        importlib.import_module(mod)
    except Exception as e:
        print(f"[ERROR] Could not import training module {mod}: {e}")
        sys.exit(1)
if "${RUN_EVAL_AFTER_TRAIN}" == "true":
    for mod in ["${EVAL_MODULE}"]:
        try:
            importlib.import_module(mod)
        except Exception as e:
            print(f"[ERROR] Could not import evaluation module {mod}: {e}")
            sys.exit(1)
PY
then
  exit 1
fi

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SAFE_GPU_IDS="${GPU_IDS//,/}"
LOG_DIR="${REPO_ROOT}/logs/${TASK}/${METHOD}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${TIMESTAMP}_${TASK}_${METHOD}_gpu${SAFE_GPU_IDS}.log"

# ---------------------------------------------------------------------------
# Build commands
# ---------------------------------------------------------------------------
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  if command -v accelerate >/dev/null 2>&1; then
    TRAIN_CMD=(accelerate launch --num_processes "${NUM_GPUS}" -m "${TRAIN_MODULE}" --config "${CONFIG_PATH}")
  else
    echo "[ERROR] Multiple GPUs requested but accelerate is not installed or not in PATH."
    exit 1
  fi
else
  TRAIN_CMD=(python -m "${TRAIN_MODULE}" --config "${CONFIG_PATH}")
fi

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  EVAL_CMD=(accelerate launch --num_processes "${NUM_GPUS}" -m "${EVAL_MODULE}" --config "${CONFIG_PATH}")
else
  EVAL_CMD=(python -m "${EVAL_MODULE}" --config "${CONFIG_PATH}")
fi

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo "Experiment : ${TASK}/${METHOD}"
echo "Repo root  : ${REPO_ROOT}"
echo "Config     : ${CONFIG_PATH}"
echo "Train mod  : ${TRAIN_MODULE}"
echo "Eval mod   : ${EVAL_MODULE}"
echo "GPU IDs    : ${GPU_IDS}"
echo "Num GPUs   : ${NUM_GPUS}"
echo "Eval after : ${RUN_EVAL_AFTER_TRAIN}"
echo "Log file   : ${LOG_FILE}"
echo "Dry run    : ${DRY_RUN:-0}"
echo "============================================================"

echo "[INFO] Training command:"
printf ' %q' "${TRAIN_CMD[@]}"
echo ""

if [[ "${RUN_EVAL_AFTER_TRAIN}" == "true" ]]; then
  echo "[INFO] Evaluation command:"
  printf ' %q' "${EVAL_CMD[@]}"
  echo ""
fi

# ---------------------------------------------------------------------------
# Dry-run exit
# ---------------------------------------------------------------------------
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY RUN] Exiting without training."
  exit 0
fi

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
"${TRAIN_CMD[@]}" 2>&1 | tee "${LOG_FILE}"

# ---------------------------------------------------------------------------
# Optional evaluation
# ---------------------------------------------------------------------------
if [[ "${RUN_EVAL_AFTER_TRAIN}" == "true" ]]; then
  echo "[INFO] Starting evaluation..." | tee -a "${LOG_FILE}"
  "${EVAL_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
else
  echo "[INFO] Skipping evaluation (RUN_EVAL_AFTER_TRAIN=false)" | tee -a "${LOG_FILE}"
fi

echo "[INFO] Done." | tee -a "${LOG_FILE}"
