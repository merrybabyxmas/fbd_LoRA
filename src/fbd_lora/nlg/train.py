"""FBD-LoRA NLG training entry point.

Usage:
    python -m fbd_lora.nlg.train --config configs/nlg/metamath/fbd.yaml --seed 42

Supports CLI overrides via OmegaConf dot-notation:
    --model.backbone sshleifer/tiny-gpt2
    --training.max_steps 5
    --wandb.enabled false

Adapter types supported (adapter.name / adapter.type):
  fbd      - FBD-LoRA (PEFT LoRA + FBD backward hooks)  [REAL: internal]
  lora     - Vanilla LoRA via PEFT LoraConfig            [REAL: PEFT]
  dora     - DoRA via PEFT LoraConfig(use_dora=True)     [REAL: PEFT >= 0.12]
  pissa    - PiSSA via LoraConfig(init_lora_weights=...) [REAL: PEFT >= 0.12]
  adalora  - AdaLoRA via PEFT AdaLoraConfig              [REAL: PEFT]
"""

import argparse
import dataclasses
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FBD-LoRA NLG Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--smoke_test", type=str, default="false", help="Run smoke test (true/false).")
    # Accept dot-notation overrides as unknown args
    args, overrides = parser.parse_known_args()
    args.overrides = overrides
    return args


def build_omega_config(config_path: str, overrides: list, seed: int):
    """Load YAML config via OmegaConf and apply CLI overrides."""
    from omegaconf import OmegaConf, DictConfig

    cfg = OmegaConf.load(config_path)

    # Parse --key value pairs into OmegaConf dotlist
    dot_overrides = []
    i = 0
    while i < len(overrides):
        arg = overrides[i]
        if arg.startswith("--"):
            key = arg[2:]
            if i + 1 < len(overrides) and not overrides[i + 1].startswith("--"):
                val = overrides[i + 1]
                dot_overrides.append(f"{key}={val}")
                i += 2
            else:
                dot_overrides.append(f"{key}=true")
                i += 1
        else:
            i += 1

    if dot_overrides:
        override_cfg = OmegaConf.from_dotlist(dot_overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    # Override seed
    OmegaConf.update(cfg, "run.seed", seed, merge=True)

    return cfg


def setup_wandb(cfg, run_id: str) -> Optional[object]:
    """Initialize W&B run if enabled."""
    from omegaconf import OmegaConf

    wandb_enabled = OmegaConf.select(cfg, "wandb.enabled", default=True)
    wandb_str = str(wandb_enabled).lower()
    if wandb_str in ("false", "0", "no", "disabled"):
        os.environ["WANDB_MODE"] = "disabled"
        logger.info("W&B disabled.")
        return None

    try:
        import wandb
        wandb_project = OmegaConf.select(cfg, "wandb.project", default=os.environ.get("WANDB_PROJECT", "fbd-lora"))
        wandb_entity = OmegaConf.select(cfg, "wandb.entity", default=os.environ.get("WANDB_ENTITY"))
        wandb_mode = OmegaConf.select(cfg, "wandb.mode", default=os.environ.get("WANDB_MODE", "online"))

        flat_config = dict(OmegaConf.to_container(cfg, resolve=True))
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_id,
            config=flat_config,
            mode=wandb_mode,
        )
        logger.info("W&B run initialized: %s", run_id)
        return run
    except Exception as e:
        logger.warning("W&B init failed: %s", e)
        os.environ["WANDB_MODE"] = "disabled"
        return None


def _resolve_adapter_type(cfg) -> str:
    """Resolve the adapter type from config.

    Accepts both adapter.type (new schema) and adapter.name (old schema).
    """
    from omegaconf import OmegaConf
    # New schema: adapter.type
    adapter_type = OmegaConf.select(cfg, "adapter.type", default=None)
    if adapter_type is None:
        # Fallback to old schema: adapter.name
        adapter_type = OmegaConf.select(cfg, "adapter.name", default="lora")
    return str(adapter_type).lower()


def _apply_peft_adapter(model, cfg, adapter_type: str, is_smoke: bool, total_steps: int = -1):
    """Apply the appropriate PEFT adapter based on adapter_type.

    Returns:
        model with PEFT adapter applied.

    Raises:
        RuntimeError if adapter_type not supported by installed PEFT version.
        ValueError if adapter_type is unknown.
    """
    from omegaconf import OmegaConf
    from peft import get_peft_model, TaskType

    rank = int(OmegaConf.select(cfg, "adapter.rank", default=16))
    alpha = int(OmegaConf.select(cfg, "adapter.alpha", default=16))
    dropout = float(OmegaConf.select(cfg, "adapter.dropout", default=0.05))
    bias = OmegaConf.select(cfg, "adapter.bias", default="none")
    target_mods_raw = OmegaConf.select(cfg, "adapter.target_modules", default=[])
    target_mods = list(target_mods_raw) if target_mods_raw else None

    # Smoke test override: use tiny rank for speed
    if is_smoke:
        rank = min(rank, 4)
        alpha = rank

    if adapter_type in ("fbd", "lora"):
        from peft import LoraConfig
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_mods,
            lora_dropout=dropout,
            bias=bias,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        logger.info("Applied LoraConfig (r=%d, alpha=%d, target=%s)", rank, alpha, target_mods)

    elif adapter_type == "dora":
        import inspect
        from peft import LoraConfig
        if "use_dora" not in inspect.signature(LoraConfig).parameters:
            raise RuntimeError(
                "[ERROR] DoRA requires PEFT with LoraConfig(use_dora=True). "
                "Upgrade peft: pip install -U peft"
            )
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_mods,
            lora_dropout=dropout,
            bias=bias,
            task_type=TaskType.CAUSAL_LM,
            use_dora=True,
        )
        model = get_peft_model(model, lora_config)
        logger.info("Applied LoraConfig(use_dora=True) (r=%d, alpha=%d)", rank, alpha)

    elif adapter_type == "pissa":
        from peft import LoraConfig
        init_lora_weights = OmegaConf.select(cfg, "adapter.init_lora_weights", default="pissa_niter_16")
        try:
            # Validate that installed PEFT supports this init_lora_weights value
            LoraConfig(r=rank, init_lora_weights=init_lora_weights)
        except Exception as e:
            raise RuntimeError(
                f"[ERROR] PiSSA not supported by installed PEFT: {e}. "
                "Upgrade peft: pip install -U peft"
            ) from e
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_mods,
            lora_dropout=dropout,
            bias=bias,
            task_type=TaskType.CAUSAL_LM,
            init_lora_weights=init_lora_weights,
        )
        model = get_peft_model(model, lora_config)
        logger.info("Applied LoraConfig(init_lora_weights=%s) (r=%d, alpha=%d)", init_lora_weights, rank, alpha)

    elif adapter_type == "adalora":
        try:
            from peft import AdaLoraConfig
        except ImportError as e:
            raise RuntimeError(
                f"[ERROR] AdaLoRA requires peft.AdaLoraConfig: {e}. "
                "Upgrade peft: pip install -U peft"
            ) from e
        init_r = int(OmegaConf.select(cfg, "adapter.init_r", default=12))
        target_r = int(OmegaConf.select(cfg, "adapter.target_r", default=8))
        if is_smoke:
            init_r = min(init_r, 4)
            target_r = min(target_r, 2)
        # AdaLoRA requires total_step to be set; compute from training config
        # total_steps > 0 means max_steps is set; otherwise use large default
        adalora_total_step = total_steps if total_steps > 0 else 1000
        adalora_config_kwargs = dict(
            init_r=init_r,
            target_r=target_r,
            lora_alpha=alpha,
            target_modules=target_mods,
            lora_dropout=dropout,
            bias=bias,
            task_type=TaskType.CAUSAL_LM,
        )
        # total_step is required by newer PEFT AdaLoraConfig
        import inspect
        from peft import AdaLoraConfig as _AdaLoraConfig
        if "total_step" in inspect.signature(_AdaLoraConfig).parameters:
            adalora_config_kwargs["total_step"] = adalora_total_step
        adalora_config = AdaLoraConfig(**adalora_config_kwargs)
        model = get_peft_model(model, adalora_config)
        logger.info("Applied AdaLoraConfig (init_r=%d, target_r=%d, alpha=%d, total_step=%d)",
                    init_r, target_r, alpha, adalora_total_step)

    else:
        raise ValueError(
            f"[ERROR] Unknown adapter type: '{adapter_type}'. "
            "Supported: fbd, lora, dora, pissa, adalora"
        )

    model.print_trainable_parameters()
    return model


def main() -> None:
    args = parse_args()
    is_smoke = str(args.smoke_test).lower() in ("true", "1", "yes")

    # Also check env variable (set by run_experiment.sh in SMOKE_TEST mode)
    if os.environ.get("SMOKE_TEST", "0") in ("1", "true", "yes"):
        is_smoke = True

    # Load secrets from .env
    from fbd_lora.secrets import load_env_file
    env_file = ".env"
    # Try project-root .env
    script_dir = Path(__file__).parent
    for candidate in [
        Path.cwd() / ".env",
        script_dir.parents[3] / ".env",
    ]:
        if candidate.exists():
            env_file = str(candidate)
            break
    load_env_file(env_file)

    from fbd_lora.logging_utils import setup_logging
    from fbd_lora.seed import seed_everything
    from fbd_lora.naming import make_run_name

    # Setup config
    cfg = build_omega_config(args.config, args.overrides, args.seed)
    from omegaconf import OmegaConf

    # Smoke test: override model and steps to tiny values if SMOKE_TEST is set
    if is_smoke:
        smoke_backbone = OmegaConf.select(cfg, "model.backbone", default=None)
        # Only override to tiny-gpt2 if not already a small model
        if smoke_backbone and "gpt2" not in str(smoke_backbone) and "tiny" not in str(smoke_backbone):
            logger.info("SMOKE_TEST: overriding model to sshleifer/tiny-gpt2")
            OmegaConf.update(cfg, "model.backbone", "sshleifer/tiny-gpt2", merge=True)
            # tiny-gpt2 (GPT-2 Conv1D) uses c_attn, c_proj, c_fc — override target modules
            from omegaconf import ListConfig
            OmegaConf.update(cfg, "adapter.target_modules", ["c_attn"], merge=True)
        # Override steps
        OmegaConf.update(cfg, "training.max_steps", 5, merge=True)
        OmegaConf.update(cfg, "training.logging_steps", 1, merge=True)
        OmegaConf.update(cfg, "training.gradient_accumulation_steps", 1, merge=True)
        OmegaConf.update(cfg, "training.per_device_train_batch_size", 2, merge=True)
        OmegaConf.update(cfg, "wandb.enabled", False, merge=True)
        # Override adapter for tiny model
        OmegaConf.update(cfg, "adapter.rank", 2, merge=True)
        OmegaConf.update(cfg, "adapter.alpha", 2, merge=True)
        # Use small dataset
        OmegaConf.update(cfg, "dataset.max_samples", 32, merge=True)
        OmegaConf.update(cfg, "dataset.max_seq_length", 128, merge=True)

    seed = int(OmegaConf.select(cfg, "run.seed", default=args.seed))
    seed_everything(seed)

    # Resolve adapter type
    adapter_type = _resolve_adapter_type(cfg)

    # Resolve paths
    output_root = OmegaConf.select(cfg, "paths.output_root", default="outputs/runs")
    modality = OmegaConf.select(cfg, "modality", default="nlg")
    task = OmegaConf.select(cfg, "task", default="metamath")
    backbone = OmegaConf.select(cfg, "model.backbone", default="gpt2")
    adapter_name = adapter_type
    rank = int(OmegaConf.select(cfg, "adapter.rank", default=16))
    alpha = int(OmegaConf.select(cfg, "adapter.alpha", default=16))
    bs = int(OmegaConf.select(cfg, "training.per_device_train_batch_size", default=4))
    ga = int(OmegaConf.select(cfg, "training.gradient_accumulation_steps", default=8))
    lr = float(OmegaConf.select(cfg, "training.learning_rate", default=2e-4))
    target_mods = list(OmegaConf.select(cfg, "adapter.target_modules", default=[]))
    routing = OmegaConf.select(cfg, "fbd.routing_type", default="none")
    lambda_r = float(OmegaConf.select(cfg, "fbd.lambda_route", default=0.25))

    # Generate run ID
    run_id = make_run_name(
        seed=seed, modality=modality, task=task, backbone=backbone,
        adapter=adapter_name, rank=rank, alpha=alpha,
        batch_size=bs, grad_accum=ga, lr=lr,
        target_modules=target_mods, routing=routing, lambda_route=lambda_r,
        full_config=dict(OmegaConf.to_container(cfg, resolve=True)),
    )

    # Setup output directory
    output_dir = Path(output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(log_dir=str(log_dir), rank=int(os.environ.get("LOCAL_RANK", "0")))
    logger.info("Run ID: %s", run_id)
    logger.info("Output dir: %s", output_dir)
    logger.info("Adapter type: %s", adapter_type)
    logger.info("Smoke test: %s", is_smoke)

    # Save config
    OmegaConf.save(cfg, str(output_dir / "config.yaml"))
    (output_dir / "logs" / "wandb_id.txt").write_text(run_id)

    # Setup W&B
    wandb_run = setup_wandb(cfg, run_id)

    # Load model and tokenizer
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
    from transformers import DataCollatorForLanguageModeling

    hf_token = os.environ.get("HF_TOKEN")
    torch_dtype_str = OmegaConf.select(cfg, "model.torch_dtype", default="bf16")
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(
        torch_dtype_str, torch.bfloat16
    )

    trust_remote = OmegaConf.select(cfg, "model.trust_remote_code", default=False)
    load_4bit = OmegaConf.select(cfg, "model.load_in_4bit", default=False)

    logger.info("Loading tokenizer: %s", backbone)
    tokenizer = AutoTokenizer.from_pretrained(
        backbone,
        token=hf_token,
        trust_remote_code=trust_remote,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info("Loading model: %s (dtype=%s)", backbone, torch_dtype_str)
    model_load_kwargs = {
        "token": hf_token,
        "trust_remote_code": trust_remote,
    }

    # Only use dtype for non-quantized loading
    # Use 'dtype' (new API in transformers 5.x) with fallback to 'torch_dtype'
    if not load_4bit:
        model_load_kwargs["dtype"] = torch_dtype

    # Try flash attention (graceful fallback - only if explicitly enabled and available)
    use_flash_attn = OmegaConf.select(cfg, "model.use_flash_attention_2", default=False)
    if use_flash_attn and not is_smoke:
        try:
            import flash_attn  # noqa: F401
            model_load_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            logger.info("flash_attn not installed; using default attention implementation.")

    model = AutoModelForCausalLM.from_pretrained(backbone, **model_load_kwargs)

    # Compute total training steps (needed by AdaLoRA)
    _max_steps_pre = int(OmegaConf.select(cfg, "training.max_steps", default=-1))
    _pre_total_steps = _max_steps_pre if _max_steps_pre > 0 else -1

    # Apply PEFT adapter (type-dispatched)
    # For FBD: apply LoRA first, then add FBD hooks below
    model = _apply_peft_adapter(model, cfg, adapter_type, is_smoke, total_steps=_pre_total_steps)

    # Apply FBD hooks if adapter type is "fbd"
    fbd_state = None
    if adapter_type == "fbd":
        fbd_enabled = OmegaConf.select(cfg, "fbd.enabled", default=True)
        if str(fbd_enabled).lower() not in ("false", "0", "no"):
            from fbd_lora.config import FBDConfig
            from fbd_lora.fbd.hooks import apply_fbd_to_peft_model

            fbd_dict = OmegaConf.to_container(OmegaConf.select(cfg, "fbd", default={}), resolve=True)
            fbd_cfg = FBDConfig.from_dict(fbd_dict)
            # In smoke test, use minimal rank
            if is_smoke:
                fbd_cfg.lambda_route = fbd_cfg.lambda_route  # keep as-is; model is tiny
            fbd_state = apply_fbd_to_peft_model(model, fbd_cfg)
            logger.info("FBD-LoRA hooks applied: %d hooks registered.", len(fbd_state.handles))
        else:
            logger.info("FBD hooks disabled by config (fbd.enabled=false).")
    else:
        logger.info("Adapter type '%s': no FBD hooks applied.", adapter_type)

    # Load dataset
    from fbd_lora.nlg.data import load_pissa_dataset, load_metamathqa, tokenize_dataset

    dataset_name = OmegaConf.select(cfg, "dataset.hf_path", default=None)
    # Also check new schema: data.dataset_name
    if dataset_name is None:
        dataset_name = OmegaConf.select(cfg, "data.dataset_name", default="fxmeng/pissa-dataset")

    dataset_sub_task = OmegaConf.select(cfg, "dataset.sub_task", default=None)
    if dataset_sub_task is None:
        dataset_sub_task = OmegaConf.select(cfg, "data.sub_task", default=None)

    dataset_subset = OmegaConf.select(cfg, "dataset.subset", default=None)
    dataset_split = OmegaConf.select(cfg, "dataset.split", default="train")
    max_samples = OmegaConf.select(cfg, "dataset.max_samples", default=None)
    if max_samples is None:
        max_samples = OmegaConf.select(cfg, "data.max_train_samples", default=None)
    max_seq_length = int(OmegaConf.select(cfg, "dataset.max_seq_length", default=2048))
    prompt_template = OmegaConf.select(cfg, "dataset.prompt_template", default="alpaca")

    if is_smoke:
        max_samples = 32
        max_seq_length = 128
        logger.info("Smoke test mode: max_samples=%d, max_seq_length=%d", max_samples, max_seq_length)

    # Dispatch dataset loading
    if dataset_name == "fxmeng/pissa-dataset":
        raw_dataset = load_pissa_dataset(
            sub_task=dataset_sub_task,
            split=dataset_split,
            max_samples=int(max_samples) if max_samples else None,
            hf_token=hf_token,
        )
    else:
        # Legacy path: load_metamathqa works for meta-math/MetaMathQA and similar
        raw_dataset = load_metamathqa(
            hf_path=dataset_name,
            subset=dataset_subset,
            split=dataset_split,
            max_samples=int(max_samples) if max_samples else None,
            hf_token=hf_token,
        )

    tokenized_dataset = tokenize_dataset(
        raw_dataset, tokenizer,
        max_seq_length=max_seq_length,
        prompt_template=prompt_template,
        num_proc=1,  # safe default
    )

    # Training arguments
    from transformers import TrainingArguments

    max_steps = int(OmegaConf.select(cfg, "training.max_steps", default=-1))
    num_epochs = float(OmegaConf.select(cfg, "training.num_train_epochs", default=1))
    grad_checkpointing = OmegaConf.select(cfg, "training.gradient_checkpointing", default=True)
    logging_steps = int(OmegaConf.select(cfg, "training.logging_steps", default=10))
    max_grad_norm = float(OmegaConf.select(cfg, "training.max_grad_norm", default=1.0))
    weight_decay = float(OmegaConf.select(cfg, "training.weight_decay", default=0.0))
    warmup_ratio = float(OmegaConf.select(cfg, "training.warmup_ratio", default=0.03))
    lr_sched = OmegaConf.select(cfg, "training.lr_scheduler_type", default="cosine")
    use_bf16 = OmegaConf.select(cfg, "training.bf16", default=True)
    use_fp16 = OmegaConf.select(cfg, "training.fp16", default=False)

    if is_smoke:
        max_steps = max(max_steps, 5)  # at least 5 steps
        num_epochs = 1
        logging_steps = 1
        grad_checkpointing = False  # disable for speed in tiny model
        use_bf16 = False  # tiny-gpt2 doesn't need bf16

    # W&B report mode
    report_to = []
    if wandb_run is not None:
        report_to = ["wandb"]

    # Compute warmup_steps from warmup_ratio
    # (warmup_ratio is deprecated in transformers 5.x)
    total_train_steps = max_steps if max_steps > 0 else int(len(tokenized_dataset) / (bs * ga) * num_epochs)
    warmup_steps_computed = max(1, int(warmup_ratio * total_train_steps))

    training_args = TrainingArguments(
        output_dir=str(output_dir / "hf_trainer"),
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=ga,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps_computed,
        lr_scheduler_type=lr_sched,
        logging_steps=logging_steps,
        save_strategy="no",  # FBDCheckpointCallback handles saving
        bf16=bool(use_bf16) and not is_smoke,
        fp16=bool(use_fp16),
        gradient_checkpointing=bool(grad_checkpointing),
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        report_to=report_to,
        run_name=run_id,
        seed=seed,
        data_seed=seed,
        max_grad_norm=max_grad_norm,
    )

    # Data collator - manually pad to same length within batch
    from fbd_lora.nlg.data import make_clm_collator
    data_collator = make_clm_collator(tokenizer)

    # Checkpoint callback
    from fbd_lora.checkpointing import FBDCheckpointCallback
    from fbd_lora.nlg.trainer import FBDTrainer

    gdrive_remote = os.environ.get("GDRIVE_REMOTE")
    gdrive_root = os.environ.get("GDRIVE_ROOT", "FBD_LORA_EXPERIMENTS")
    upload_gdrive = OmegaConf.select(cfg, "run.upload_to_gdrive", default=False)
    save_every_pct = int(OmegaConf.select(cfg, "training.save_every_percent", default=10))

    # Resolve fbd_cfg for checkpoint callback even in non-FBD modes
    _fbd_cfg_for_ckpt = None
    if fbd_state is not None:
        _fbd_cfg_for_ckpt = fbd_cfg  # noqa: F821 - only set when fbd_state is set

    ckpt_callback = FBDCheckpointCallback(
        run_id=run_id,
        output_dir=str(output_dir),
        fbd_config=_fbd_cfg_for_ckpt,
        save_every_percent=save_every_pct,
        gdrive_remote=gdrive_remote if upload_gdrive else None,
        gdrive_root=gdrive_root,
        dry_run_gdrive=False,
        wandb_run=wandb_run,
    )

    # Build trainer
    trainer = FBDTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        fbd_state=fbd_state,
        gradient_stats_interval=int(OmegaConf.select(cfg, "fbd.gradient_stats_interval", default=10)),
        wandb_run=wandb_run,
        callbacks=[ckpt_callback],
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()
    logger.info("Training complete. Metrics: %s", train_result.metrics)

    # Save final model
    final_dir = output_dir / "checkpoints" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info("Final model saved to: %s", final_dir)

    # Cleanup hooks
    if fbd_state is not None:
        from fbd_lora.fbd.hooks import remove_fbd_hooks
        remove_fbd_hooks(fbd_state)

    if wandb_run is not None:
        wandb_run.finish()

    logger.info("Run complete: %s", run_id)


if __name__ == "__main__":
    main()
