# FBD-LoRA Experiment Runner

This document describes the config-driven shell script infrastructure for running
Forward-Backward Decoupled Low-Rank Adaptation (FBD-LoRA) experiments.

---

## Quick Start

```bash
# Run FBD-LoRA on MetaMath (GPU 0, with eval after training)
bash scripts/nlg/metamath/train_fbd.sh 0 true

# Run LoRA baseline on MetaMath (GPU 0, no eval)
bash scripts/nlg/metamath/train_lora.sh 0 true

# Run FBD-LoRA on conversation task (placeholder — exits with error until implemented)
bash scripts/nlg/conversation/train_fbd.sh 0 true

# Run FBD-LoRA image personalization on DreamBench (GPU 0, with eval)
bash scripts/imagen/dreambench/train_fbd.sh 0 true

# Dry run (validate config, print commands, no GPU needed)
DRY_RUN=1 bash scripts/nlg/metamath/train_fbd.sh 0 true
DRY_RUN=1 bash scripts/imagen/dreambench/train_lora.sh 0 false
```

---

## Script Interface

Every training script accepts exactly two arguments:

```
bash <script_path> <GPU_IDS> <RUN_EVAL_AFTER_TRAIN>
```

**GPU_IDS**: Comma-separated GPU device IDs.
- `0` — single GPU
- `0,1` — two GPUs (uses `accelerate launch --num_processes 2`)
- `0,1,2,3` — four GPUs

**RUN_EVAL_AFTER_TRAIN**: Whether to run evaluation after training.
- Accepted values: `true`, `false`, `1`, `0`, `yes`, `no`, `y`, `n`

---

## All Script Paths

### NLG — MetaMath (IMPLEMENTED)

| Script | Method | Status |
|--------|--------|--------|
| `scripts/nlg/metamath/train_fbd.sh` | FBD-LoRA | Implemented |
| `scripts/nlg/metamath/train_lora.sh` | LoRA baseline | Implemented |
| `scripts/nlg/metamath/train_dora.sh` | DoRA | Placeholder |
| `scripts/nlg/metamath/train_pissa.sh` | PiSSA | Placeholder |
| `scripts/nlg/metamath/train_adalora.sh` | AdaLoRA | Placeholder |
| `scripts/nlg/metamath/train_lora_ga.sh` | LoRA-GA | Placeholder |
| `scripts/nlg/metamath/run_core_baselines.sh` | Aggregator (lora + fbd) | Implemented |

### NLG — Conversation (PLACEHOLDER)

| Script | Method | Status |
|--------|--------|--------|
| `scripts/nlg/conversation/train_fbd.sh` | FBD-LoRA | Placeholder |
| `scripts/nlg/conversation/train_lora.sh` | LoRA | Placeholder |
| `scripts/nlg/conversation/train_dora.sh` | DoRA | Placeholder |
| `scripts/nlg/conversation/train_pissa.sh` | PiSSA | Placeholder |
| `scripts/nlg/conversation/train_adalora.sh` | AdaLoRA | Placeholder |
| `scripts/nlg/conversation/train_lora_ga.sh` | LoRA-GA | Placeholder |
| `scripts/nlg/conversation/run_core_baselines.sh` | Aggregator | Placeholder |

### NLG — Code Generation (PLACEHOLDER)

| Script | Method | Status |
|--------|--------|--------|
| `scripts/nlg/code/train_fbd.sh` | FBD-LoRA | Placeholder |
| `scripts/nlg/code/train_lora.sh` | LoRA | Placeholder |
| `scripts/nlg/code/train_dora.sh` | DoRA | Placeholder |
| `scripts/nlg/code/train_pissa.sh` | PiSSA | Placeholder |
| `scripts/nlg/code/train_adalora.sh` | AdaLoRA | Placeholder |
| `scripts/nlg/code/train_lora_ga.sh` | LoRA-GA | Placeholder |

### Image — DreamBench

| Script | Method | Status |
|--------|--------|--------|
| `scripts/imagen/dreambench/train_fbd.sh` | FBD-LoRA DreamBooth | Implemented |
| `scripts/imagen/dreambench/train_lora.sh` | LoRA DreamBooth | Implemented |
| `scripts/imagen/dreambench/train_dreambooth.sh` | DreamBooth (full FT) | Placeholder |
| `scripts/imagen/dreambench/train_custom_diffusion.sh` | Custom Diffusion | Placeholder |
| `scripts/imagen/dreambench/train_svdiff.sh` | SVDiff | Placeholder |
| `scripts/imagen/dreambench/train_diffusekrona.sh` | DiffuseKrona | Placeholder |
| `scripts/imagen/dreambench/run_core_baselines.sh` | Aggregator (lora + fbd) | Implemented |

---

## Config Layout

```
configs/
  nlg/
    metamath/
      fbd.yaml       # FBD-LoRA on MetaMath (Mistral-7B-v0.1)
      lora.yaml      # LoRA baseline
      dora.yaml      # Placeholder
      pissa.yaml     # Placeholder
      adalora.yaml   # Placeholder
      lora_ga.yaml   # Placeholder
    conversation/    # All placeholders
    code/            # All placeholders
  imagen/
    dreambench/
      fbd.yaml       # FBD-LoRA DreamBooth (SD v1.5)
      lora.yaml      # LoRA DreamBooth baseline
      dreambooth.yaml      # Placeholder
      custom_diffusion.yaml # Placeholder
      svdiff.yaml          # Placeholder
      diffusekrona.yaml    # Placeholder
  accelerate/
    multi_gpu_ddp.yaml
    imagen_3gpu.yaml
```

---

## Training Modules

| Domain | Module | Description |
|--------|--------|-------------|
| NLG train | `fbd_lora.nlg.train` | Causal LM fine-tuning with FBD-LoRA / LoRA |
| NLG eval | `fbd_lora.nlg.run_eval_gsm8k` | GSM8K evaluation |
| Imagen train | `fbd_lora.imagen.train_dreambooth_lora` | DreamBooth LoRA training |
| Imagen eval | `fbd_lora.imagen.evaluate_clip_dino` | CLIP-I/T and DINO evaluation |

---

## Dry Run

Dry run validates config paths and module imports, prints commands, then exits without
training. Requires no GPU.

```bash
DRY_RUN=1 bash scripts/nlg/metamath/train_fbd.sh 0 true
DRY_RUN=1 bash scripts/nlg/metamath/train_lora.sh 0 false
DRY_RUN=1 bash scripts/nlg/conversation/train_fbd.sh 0 true   # exits with error (placeholder)
DRY_RUN=1 bash scripts/imagen/dreambench/train_fbd.sh 0 true
DRY_RUN=1 bash scripts/nlg/metamath/train_fbd.sh 0,1 true     # shows accelerate launch
```

---

## W&B Configuration

W&B is controlled entirely by YAML config. The run name is generated by
`fbd_lora.naming.make_run_name()` and set as both `TrainingArguments.run_name`
and `wandb.init(name=...)`.

Example run name format:
```
20260511-104814_seed42_nlg_metamath_mistral-7b-v0-1_fbd_r16_a16_bs4_ga8_lr2e-4_qkvo_pullback_metric_mixe_lam0p25_a1b2c3d4
```

W&B config fields (in YAML):
```yaml
wandb:
  enabled: true
  project: fbd-lora
  entity: null          # or your W&B entity
  mode: online          # or 'offline', 'disabled'
```

Set `WANDB_API_KEY` in `.env` or export it before running.

---

## Google Drive Checkpoint Sync

Checkpoint sync is handled by `fbd_lora.checkpointing.FBDCheckpointCallback`
via `rclone`. Configure in your YAML:

```yaml
run:
  upload_to_gdrive: true  # set to true to enable
```

Set environment variables:
```bash
GDRIVE_REMOTE=fbd_gdrive   # rclone remote name
GDRIVE_ROOT=FBD_LORA_EXPERIMENTS
```

**rclone is not currently installed** on this machine. Sync is silently skipped
when `upload_to_gdrive: false` or rclone is absent.

---

## Logs

Logs are saved to:
```
logs/<task>/<method>/<timestamp>_<task>_<method>_gpu<ids>.log
```

Examples:
```
logs/nlg_metamath/fbd/20260511_104814_nlg_metamath_fbd_gpu0.log
logs/imagen_dreambench/lora/20260511_110000_imagen_dreambench_lora_gpu01.log
```

Logs capture both stdout and stderr via `tee`.

---

## Outputs and Checkpoints

Training outputs are written to:
```
outputs/runs/<run_id>/
  config.yaml              # saved config
  checkpoints/
    step_000010pct/        # checkpoint at 10%
    step_000020pct/        # checkpoint at 20%
    ...
    final/                 # final adapter
  logs/
    train.log
    wandb_id.txt           # run ID / W&B name
  hf_trainer/             # HuggingFace trainer artifacts
```

---

## Run Name Generation

Run names are generated by `fbd_lora.naming.make_run_name()` and include:
timestamp, seed, modality, task, backbone, adapter, rank, alpha, batch size,
gradient accumulation steps, learning rate, target modules, routing type,
lambda, and an 8-character config hash.

You can preview the run name without training:
```bash
python -m fbd_lora.config --print-run-name --config configs/nlg/metamath/fbd.yaml
python -m fbd_lora.naming --config configs/nlg/metamath/fbd.yaml --print-run-name
```

---

## Implemented vs Placeholder Methods

**Implemented (can run immediately with real data):**
- NLG MetaMath: `fbd`, `lora`
- Imagen DreamBench: `fbd`, `lora`

**Placeholder (config exists, will exit with clear error until implemented):**
- NLG MetaMath: `dora`, `pissa`, `adalora`, `lora_ga`
- NLG Conversation: all methods
- NLG Code: all methods
- Imagen DreamBench: `dreambooth`, `custom_diffusion`, `svdiff`, `diffusekrona`

---

## Secrets

Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
# edit .env
```

`.env` is gitignored. Never commit secrets.

---

## Real-Data Sanity Runs

We do not use tiny-gpt2 as evidence for FBD-LoRA correctness. NLG real-data sanity uses
Mistral-7B-v0.1 and `fxmeng/pissa-dataset`. Image real-data sanity uses Stable Diffusion
v1.5 and DreamBench++ or MS-Bench.

These are **20-step smoke tests** that verify the full training pipeline end-to-end with
real models and real data, without requiring task-level evaluation (eval disabled by default).

### Policy

- Sanity configs live under `configs/sanity/` (separate from production configs in `configs/nlg/`, `configs/imagen/`)
- Sanity scripts live under `scripts/sanity/` (separate from production scripts)
- 20 steps, rank=8, alpha=16 for NLG; 20 steps, rank=4, alpha=4 for imagen
- W&B enabled; checkpoints saved at 50% and 100%
- Evaluation is disabled (pass `false` as the second argument, or set `evaluation.enabled: false` in config)

### Environment Variables

Set these in `.env` or export before running:

```bash
# Path to a local cache of Mistral-7B-v0.1 (optional — downloads from HF if not set)
export FBD_MISTRAL_MODEL_PATH=/path/to/mistral-7b-v0-1

# Path to a local cache of SD v1.5 (optional — downloads from HF if not set)
export FBD_SD15_MODEL_PATH=/path/to/stable-diffusion-v1-5

# W&B API key
export WANDB_API_KEY=your_key

# HuggingFace token (needed for gated models/datasets)
export HF_TOKEN=your_token
```

### NLG Sanity Commands

```bash
# MetaMath — FBD, LoRA, DoRA, PiSSA, AdaLoRA (20 steps each on Mistral-7B-v0.1)
bash scripts/sanity/nlg/metamath/train_fbd_mistral.sh 0 false
bash scripts/sanity/nlg/metamath/train_lora_mistral.sh 0 false
bash scripts/sanity/nlg/metamath/train_dora_mistral.sh 0 false
bash scripts/sanity/nlg/metamath/train_pissa_mistral.sh 0 false
bash scripts/sanity/nlg/metamath/train_adalora_mistral.sh 0 false

# Conversation task — FBD and LoRA (20 steps each, sub_task: conversation)
bash scripts/sanity/nlg/conversation/train_fbd_mistral.sh 0 false
bash scripts/sanity/nlg/conversation/train_lora_mistral.sh 0 false

# Code generation task — FBD and LoRA (20 steps each, sub_task: python)
bash scripts/sanity/nlg/code/train_fbd_mistral.sh 0 false
bash scripts/sanity/nlg/code/train_lora_mistral.sh 0 false
```

### Image Sanity Commands

```bash
# MS-Bench (fallback) — FBD and LoRA (20 steps each on SD v1.5)
bash scripts/sanity/imagen/dreambench/train_fbd_sd15_msbench.sh 0 false
bash scripts/sanity/imagen/dreambench/train_lora_sd15_msbench.sh 0 false

# DreamBench++ — FBD and LoRA (20 steps each on SD v1.5; may fail to load dataset)
bash scripts/sanity/imagen/dreambench/train_fbd_sd15_dreambench_plus.sh 0 false
bash scripts/sanity/imagen/dreambench/train_lora_sd15_dreambench_plus.sh 0 false
```

### Sanity Config Layout

```
configs/sanity/
  nlg/
    metamath/
      fbd_mistral.yaml        # FBD-LoRA, MetaMath, Mistral-7B-v0.1
      lora_mistral.yaml       # LoRA baseline
      dora_mistral.yaml       # DoRA
      pissa_mistral.yaml      # PiSSA
      adalora_mistral.yaml    # AdaLoRA
    conversation/
      fbd_mistral.yaml        # FBD-LoRA, conversation sub-task
      lora_mistral.yaml       # LoRA baseline
    code/
      fbd_mistral.yaml        # FBD-LoRA, python (code) sub-task
      lora_mistral.yaml       # LoRA baseline
  imagen/
    dreambench/
      fbd_sd15_msbench.yaml           # FBD-LoRA, MS-Bench (doge1516/MS-Bench)
      lora_sd15_msbench.yaml          # LoRA baseline, MS-Bench
      fbd_sd15_dreambench_plus.yaml   # FBD-LoRA, DreamBench++
      lora_sd15_dreambench_plus.yaml  # LoRA baseline, DreamBench++
```

### Dataset Notes

- **NLG**: `fxmeng/pissa-dataset` — 798K examples, filtered by `type` field
  (`metamath`, `python`, `conversation`). `max_samples: 256` limits loading to 256 rows.
- **MS-Bench** (`doge1516/MS-Bench`): 40 rows, 7 concepts (labels 0–6), PIL images.
  `concept_id: 0` downloads images for label 0 (typically 3–5 images).
- **DreamBench++** (`yuangpeng/dreambench_plus`): may fail with `DatasetGenerationError`
  on some environments. Use MS-Bench configs as the reliable fallback.

### Pass / Fail Criteria

A sanity run is considered **PASS** if:
1. Training completes all 20 steps without exception
2. Loss is finite (not NaN / inf) at the final step
3. At least one checkpoint is saved
4. W&B run is created (if `wandb.enabled: true`)

A sanity run is considered **FAIL** if any of the above conditions are not met.

---

## DreamBench++ Loading

DreamBench++ is loaded via the **official-file approach** using `huggingface_hub.snapshot_download`, which avoids the `datasets.load_dataset("yuangpeng/dreambench_plus")` call that fails in most environments.

### How it works

`fbd_lora.imagen.data.load_dreambench_plus(data_cfg)` implements the following loading priority:

1. **Local path**: If `data.local_data_root` is set (non-empty), load from that directory.
2. **HuggingFace snapshot**: If `data.allow_hf_snapshot_download=true`, call `snapshot_download(repo_id="yuangpeng/dreambench_plus", repo_type="dataset")` to download all files to a local cache.
3. **Error**: If neither applies, raise a clear `RuntimeError` explaining the options.

### Dataset directory layout

The loader expects one subdirectory per concept, each containing images:

```
dreambench_plus/
  dog_statue/
    image_000.jpg
    image_001.jpg
    metadata.json          # optional: prompts
  teapot/
    image_000.jpg
    eval_prompts.txt       # alternative: one prompt per line
  backpack/
    ...
```

### Supported metadata formats

| Format | File | Description |
|--------|------|-------------|
| JSON list | `metadata.json` | `[{"image": "img.jpg", "prompts": ["..."]}]` |
| JSON dict (items) | `metadata.json` | `{"items": [{"image_path": "img.jpg", "prompt": "..."}]}` |
| JSON mapping | `metadata.json` | `{"concept_001": {"image": "img.jpg", "prompts": [...]}}` |
| JSONL | `metadata.jsonl` | one `{"image": "...", "prompt": "..."}` per line |
| CSV/TSV | `metadata.csv` | columns: `image`/`image_path`/`filename`, `prompt`/`prompts`/`text` |
| Plain text | `eval_prompts.txt` | one prompt per line (fallback if no JSON/CSV) |

### Config fields

```yaml
dataset:
  name: dreambench_plus               # use new official-file loader
  hf_repo_id: yuangpeng/dreambench_plus
  local_data_root: ${FBD_DREAMBENCH_PLUS_ROOT:-}  # set to use local path
  allow_hf_snapshot_download: true    # auto-download via HF Hub
  allow_fallback: false               # never silently switch to MS-Bench
  allow_sanity_prompt_fallback: false # fail clearly if no prompts found
  max_concepts: 1                     # load only 1 concept (sanity)
  max_train_images_per_concept: 4
  max_eval_prompts_per_concept: 2
```

### Environment variables

```bash
# Point to a local copy of the dataset (priority over snapshot_download)
export FBD_DREAMBENCH_PLUS_ROOT=/path/to/dreambench_plus

# HF token (for private repos or rate-limited downloads)
export HF_TOKEN=your_token
```

### Fallback policy

The loader **never silently switches to MS-Bench**. If DreamBench++ loading fails:
- If `allow_fallback=true`: logs a clear `[WARNING]` and switches to MS-Bench; run name includes `msbench_fallback`.
- If `allow_fallback=false` (default): raises `RuntimeError` with a clear message.

### Sanity scripts

```bash
# DreamBench++ — FBD and LoRA (20 steps each on SD v1.5)
bash scripts/sanity/imagen/dreambench/train_fbd_sd15_dreambench_plus.sh 1 false
bash scripts/sanity/imagen/dreambench/train_lora_sd15_dreambench_plus.sh 1 false

# DreamBench++ — DreamBooth and Custom Diffusion (stubs, dry-run only)
DRY_RUN=1 bash scripts/sanity/imagen/dreambench/train_dreambooth_sd15_dreambench_plus.sh 1 false
DRY_RUN=1 bash scripts/sanity/imagen/dreambench/train_custom_diffusion_sd15_dreambench_plus.sh 1 false

# Dry-run validation (no GPU needed)
DRY_RUN=1 bash scripts/sanity/imagen/dreambench/train_fbd_sd15_dreambench_plus.sh 0 false
```

---

## Aggregators (run multiple methods sequentially)

```bash
# Run LoRA then FBD-LoRA on MetaMath
bash scripts/nlg/metamath/run_core_baselines.sh 0 true

# Run LoRA then FBD-LoRA on DreamBench
bash scripts/imagen/dreambench/run_core_baselines.sh 0 true
```
