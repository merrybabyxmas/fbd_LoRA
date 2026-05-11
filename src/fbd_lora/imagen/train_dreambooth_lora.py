"""DreamBooth LoRA training with FBD-LoRA gradient routing.

Entry point for concept personalization training on SD v1.5.
Called via accelerate launch -m fbd_lora.imagen.train_dreambooth_lora.

Usage:
    accelerate launch --config_file configs/accelerate/multi_gpu_ddp.yaml \\
        -m fbd_lora.imagen.train_dreambooth_lora \\
        --config configs/imagen/customconcept101_fbd.yaml \\
        --seed 42 \\
        --concept_name cat_statue \\
        --concept_dir data/customconcept101/cat_statue

Mathematical formulation:
    Forward:  z_t = vae.encode(x_0) * scale_factor
              eps_pred = unet(z_t, t, c)  where c = text_encoder(prompt)
    Loss:     L = ||eps_pred - eps||^2  (standard denoising objective)
    FBD hook: grad(lora_A) replaced by pullback-metric surrogate during backward.
"""

import argparse
import dataclasses
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FBD-LoRA DreamBooth training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concept_name", type=str, default=None,
                        help="Concept name (subfolder under dataset.root).")
    parser.add_argument("--concept_dir", type=str, default=None,
                        help="Override path to concept image directory.")
    parser.add_argument("--instance_prompt", type=str, default=None,
                        help="Override instance prompt (default: 'a photo of sks <concept_name>').")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run smoke test: 2 steps, rank 2, tiny model.")
    args, overrides = parser.parse_known_args()
    args.overrides = overrides
    return args


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def build_config(config_path: str, overrides: list, seed: int):
    """Load YAML config and apply CLI overrides via OmegaConf."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(config_path)

    # Parse --key value pairs
    dot_overrides = []
    i = 0
    while i < len(overrides):
        arg = overrides[i]
        if arg.startswith("--"):
            key = arg[2:]
            if i + 1 < len(overrides) and not overrides[i + 1].startswith("--"):
                dot_overrides.append(f"{key}={overrides[i+1]}")
                i += 2
            else:
                dot_overrides.append(f"{key}=true")
                i += 1
        else:
            i += 1
    if dot_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dot_overrides))

    OmegaConf.update(cfg, "run.seed", seed, merge=True)
    return cfg


# ---------------------------------------------------------------------------
# Noise prediction helper
# ---------------------------------------------------------------------------

def compute_snr(noise_scheduler, timesteps: "torch.Tensor") -> "torch.Tensor":
    """Compute signal-to-noise ratio for timesteps."""
    import torch
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5
    alpha = sqrt_alphas_cumprod[timesteps].to(torch.float32)
    sigma = sqrt_one_minus_alphas_cumprod[timesteps].to(torch.float32)
    snr = (alpha / sigma) ** 2
    return snr


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Load secrets
    from fbd_lora.secrets import load_env_file
    for candidate in [Path.cwd() / ".env", Path(__file__).parents[4] / ".env"]:
        if candidate.exists():
            load_env_file(str(candidate))
            break

    from fbd_lora.logging_utils import setup_logging
    from fbd_lora.seed import seed_everything
    from fbd_lora.naming import make_run_name
    from omegaconf import OmegaConf

    cfg = build_config(args.config, args.overrides, args.seed)

    # Initialize Accelerate first (needed to set up logging correctly in DDP)
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration, set_seed

    seed = int(OmegaConf.select(cfg, "run.seed", default=args.seed))
    mixed_precision = OmegaConf.select(cfg, "training.mixed_precision", default="fp16")
    output_root = OmegaConf.select(cfg, "paths.output_root", default="outputs/runs")

    # Resolve concept name and directory
    # Support HF dataset loading for msbench_hf and dreambench_plus_hf
    dataset_name_cfg = OmegaConf.select(cfg, "dataset.name", default="local")
    concept_name = args.concept_name or OmegaConf.select(cfg, "dataset.concept_name", default="concept")
    dataset_root = OmegaConf.select(cfg, "dataset.root", default="data/customconcept101")
    concept_dir_override = args.concept_dir

    if concept_dir_override is None and dataset_name_cfg in (
        "msbench_hf", "dreambench_plus_hf", "dreambench_plus", "dreambench_plus_official"
    ):
        hf_token = os.environ.get("HF_TOKEN")
        concept_id = int(OmegaConf.select(cfg, "dataset.concept_id", default=0))
        max_train_images = int(OmegaConf.select(cfg, "dataset.max_train_images_per_concept", default=4))
        hf_split = OmegaConf.select(cfg, "dataset.hf_split", default="train")

        if dataset_name_cfg == "msbench_hf":
            from fbd_lora.imagen.data import load_msbench_concept
            logger.info("Loading MS-Bench concept %d from HuggingFace...", concept_id)
            concept_dir_hf = load_msbench_concept(
                output_dir=str(Path(dataset_root)),
                concept_id=concept_id,
                max_images=max_train_images,
                hf_token=hf_token,
            )
            concept_name = f"msbench_concept_{concept_id:02d}"
            concept_dir = concept_dir_hf
        elif dataset_name_cfg in ("dreambench_plus", "dreambench_plus_official"):
            # Use new official-file loader via snapshot_download
            from fbd_lora.imagen.data import load_dreambench_plus
            dataset_cfg_node = OmegaConf.select(cfg, "dataset", default={})
            logger.info("Loading DreamBench++ via official-file loader...")
            allow_fallback = OmegaConf.select(cfg, "dataset.allow_fallback", default=False)
            try:
                db_concepts = load_dreambench_plus(dataset_cfg_node)
            except Exception as db_exc:
                if allow_fallback:
                    logger.warning(
                        "[WARNING] DreamBench++ loading failed. Falling back to MS-Bench "
                        "because allow_fallback=true. Error: %s", db_exc
                    )
                    from fbd_lora.imagen.data import load_msbench_concept
                    concept_dir_hf = load_msbench_concept(
                        output_dir=str(Path(dataset_root) / "msbench"),
                        concept_id=concept_id,
                        max_images=max_train_images,
                        hf_token=hf_token,
                    )
                    concept_name = f"msbench_fallback_concept_{concept_id:02d}"
                    concept_dir = concept_dir_hf
                else:
                    raise
            else:
                # Use first concept (or concept at index concept_id)
                idx = min(concept_id, len(db_concepts) - 1)
                selected = db_concepts[idx]
                concept_name = f"dreambench_{selected['concept_id']}"
                concept_dir = str(Path(selected["train_images"][0]).parent)
        else:  # dreambench_plus_hf (legacy)
            from fbd_lora.imagen.data import load_dreambench_plus_concept
            logger.info("Loading DreamBench+ concept %d from HuggingFace...", concept_id)
            concept_dir_hf = load_dreambench_plus_concept(
                output_dir=str(Path(dataset_root)),
                concept_id=concept_id,
                max_images=max_train_images,
                hf_split=hf_split,
                hf_token=hf_token,
            )
            concept_name = f"dreambench_concept_{concept_id:02d}"
            concept_dir = concept_dir_hf
    else:
        concept_dir = concept_dir_override or str(Path(dataset_root) / concept_name)

    instance_prompt = args.instance_prompt or f"a photo of sks {concept_name.replace('_', ' ')}"

    # Build run ID
    modality = OmegaConf.select(cfg, "modality", default="imagen")
    task = OmegaConf.select(cfg, "task", default="customconcept101")
    backbone = OmegaConf.select(cfg, "model.backbone", default="runwayml/stable-diffusion-v1-5")
    adapter_name_cfg = OmegaConf.select(cfg, "adapter.name", default="fbd")
    rank = int(OmegaConf.select(cfg, "adapter.rank", default=8))
    alpha = int(OmegaConf.select(cfg, "adapter.alpha", default=8))
    bs = int(OmegaConf.select(cfg, "training.train_batch_size", default=1))
    ga = int(OmegaConf.select(cfg, "training.gradient_accumulation_steps", default=4))
    lr = float(OmegaConf.select(cfg, "training.learning_rate", default=1e-4))
    target_mods = list(OmegaConf.select(cfg, "adapter.target_modules", default=["to_q", "to_k", "to_v", "to_out.0"]))
    routing = OmegaConf.select(cfg, "fbd.routing_type", default="none")
    lambda_r = float(OmegaConf.select(cfg, "fbd.lambda_route", default=0.0))
    fbd_enabled = OmegaConf.select(cfg, "fbd.enabled", default=False)
    if str(fbd_enabled).lower() in ("false", "0", "no"):
        fbd_enabled = False

    run_id = make_run_name(
        seed=seed,
        modality=modality,
        task=f"{task}_{concept_name}",
        backbone=backbone,
        adapter=adapter_name_cfg,
        rank=rank,
        alpha=alpha,
        batch_size=bs,
        grad_accum=ga,
        lr=lr,
        target_modules=target_mods,
        routing=routing if fbd_enabled else "none",
        lambda_route=lambda_r if fbd_enabled else 0.0,
        full_config=dict(OmegaConf.to_container(cfg, resolve=True)),
    )

    output_dir = Path(output_root) / run_id
    import torch as _torch
    _has_cuda = _torch.cuda.is_available()
    # Use fp32 when no GPU is available (smoke test on CPU, or debugging)
    _effective_mixed_precision = mixed_precision if _has_cuda else "no"
    if not _has_cuda and mixed_precision != "no":
        logger.warning(
            "No CUDA device available; overriding mixed_precision='%s' -> 'no' (fp32).",
            mixed_precision
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=ga,
        mixed_precision=_effective_mixed_precision,
        project_config=ProjectConfiguration(project_dir=str(output_dir)),
        log_with=None,
    )

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "logs").mkdir(exist_ok=True)
        setup_logging(log_dir=str(output_dir / "logs"), rank=0)
    else:
        logging.basicConfig(level=logging.WARNING)

    logger.info("Run ID: %s", run_id)
    logger.info("Concept: '%s' from '%s'", concept_name, concept_dir)
    logger.info("Instance prompt: '%s'", instance_prompt)

    set_seed(seed)
    seed_everything(seed)

    if accelerator.is_main_process:
        OmegaConf.save(cfg, str(output_dir / "config.yaml"))
        (output_dir / "logs" / "wandb_id.txt").write_text(run_id)

    # Setup W&B
    import wandb as wandb_module
    wandb_run = None
    wandb_enabled_cfg = OmegaConf.select(cfg, "wandb.enabled", default=True)
    if accelerator.is_main_process and str(wandb_enabled_cfg).lower() not in ("false", "0", "no"):
        wandb_api_key = os.environ.get("WANDB_API_KEY")
        if wandb_api_key:
            try:
                wandb_run = wandb_module.init(
                    project=os.environ.get("WANDB_PROJECT", "fbd-lora"),
                    entity=os.environ.get("WANDB_ENTITY"),
                    name=run_id,
                    config=dict(OmegaConf.to_container(cfg, resolve=True)),
                    mode=os.environ.get("WANDB_MODE", "online"),
                )
                logger.info("W&B run initialized: %s", run_id)
            except Exception as e:
                logger.warning("W&B init failed: %s", e)

    # -----------------------------------------------------------------------
    # Load model components
    # -----------------------------------------------------------------------
    import torch
    import torch.nn.functional as F
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    hf_token = os.environ.get("HF_TOKEN")
    torch_dtype_str = OmegaConf.select(cfg, "model.torch_dtype", default="fp16")
    _base_weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(
        torch_dtype_str, torch.float16
    )
    # On CPU, force fp32 to avoid dtype mismatch errors
    weight_dtype = _base_weight_dtype if torch.cuda.is_available() else torch.float32

    # Handle smoke test: use tiny model
    if args.smoke_test:
        model_id = "hf-internal-testing/tiny-stable-diffusion-pipe"
        logger.info("SMOKE TEST: using tiny SD model '%s'", model_id)
    else:
        model_id = backbone

    logger.info("Loading pipeline components from '%s'", model_id)

    tokenizer = CLIPTokenizer.from_pretrained(
        model_id, subfolder="tokenizer",
        token=hf_token,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder",
        token=hf_token,
    )
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae",
        token=hf_token,
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet",
        token=hf_token,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        model_id, subfolder="scheduler",
        token=hf_token,
    )

    # Freeze VAE and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae = vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder = text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Gradient checkpointing for UNet
    if OmegaConf.select(cfg, "model.gradient_checkpointing", default=True):
        unet.enable_gradient_checkpointing()

    # -----------------------------------------------------------------------
    # Apply PEFT LoRA to UNet
    # -----------------------------------------------------------------------
    from peft import LoraConfig, get_peft_model

    dropout = float(OmegaConf.select(cfg, "adapter.dropout", default=0.0))

    # In smoke test, use rank 2
    if args.smoke_test:
        rank = 2
        alpha = 2

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_mods,
        lora_dropout=dropout,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    if accelerator.is_main_process:
        unet.print_trainable_parameters()

    # -----------------------------------------------------------------------
    # Apply FBD hooks
    # -----------------------------------------------------------------------
    fbd_state = None
    if fbd_enabled:
        from fbd_lora.config import FBDConfig
        from fbd_lora.fbd.hooks import apply_fbd_to_peft_model

        fbd_dict = OmegaConf.to_container(OmegaConf.select(cfg, "fbd", default={}), resolve=True)
        fbd_dict.pop("enabled", None)
        fbd_cfg = FBDConfig.from_dict(fbd_dict)
        fbd_state = apply_fbd_to_peft_model(unet, fbd_cfg)
        logger.info("FBD hooks registered: %d hooks", len(fbd_state.handles))
    else:
        fbd_cfg = None
        logger.info("FBD disabled; standard LoRA training.")

    # -----------------------------------------------------------------------
    # Dataset and DataLoader
    # -----------------------------------------------------------------------
    from fbd_lora.imagen.data import ConceptDataset, collate_fn
    from torch.utils.data import DataLoader

    resolution = int(OmegaConf.select(cfg, "dataset.resolution", default=512))
    center_crop = bool(OmegaConf.select(cfg, "dataset.center_crop", default=True))
    random_flip = bool(OmegaConf.select(cfg, "dataset.random_flip", default=True))

    if args.smoke_test:
        resolution = 64  # tiny for smoke test

    train_dataset = ConceptDataset(
        image_dir=concept_dir,
        instance_prompt=instance_prompt,
        tokenizer=tokenizer,
        size=resolution,
        center_crop=center_crop,
        random_flip=random_flip,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    # -----------------------------------------------------------------------
    # Optimizer and LR scheduler
    # -----------------------------------------------------------------------
    from torch.optim import AdamW

    adam_beta1 = float(OmegaConf.select(cfg, "training.adam_beta1", default=0.9))
    adam_beta2 = float(OmegaConf.select(cfg, "training.adam_beta2", default=0.999))
    adam_weight_decay = float(OmegaConf.select(cfg, "training.adam_weight_decay", default=0.01))
    adam_epsilon = float(OmegaConf.select(cfg, "training.adam_epsilon", default=1e-8))
    max_grad_norm = float(OmegaConf.select(cfg, "training.max_grad_norm", default=1.0))

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=lr,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    max_train_steps = int(OmegaConf.select(cfg, "training.max_train_steps", default=1000))
    if args.smoke_test:
        max_train_steps = 2

    lr_warmup_steps = int(OmegaConf.select(cfg, "training.lr_warmup_steps", default=0))
    lr_scheduler_type = OmegaConf.select(cfg, "training.lr_scheduler", default="constant")

    from diffusers.optimization import get_scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * ga,
        num_training_steps=max_train_steps * ga,
    )

    # -----------------------------------------------------------------------
    # Prepare with Accelerate
    # -----------------------------------------------------------------------
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # -----------------------------------------------------------------------
    # Checkpoint manager
    # -----------------------------------------------------------------------
    from fbd_lora.imagen.callbacks import FBDDiffusersCheckpointManager

    save_every_pct = int(OmegaConf.select(cfg, "training.save_every_percent", default=10))
    gdrive_remote = os.environ.get("GDRIVE_REMOTE") if OmegaConf.select(cfg, "run.upload_to_gdrive", default=False) else None

    ckpt_manager = FBDDiffusersCheckpointManager(
        unet=unet,
        accelerator=accelerator,
        output_dir=str(output_dir),
        total_steps=max_train_steps,
        save_every_percent=save_every_pct,
        run_id=run_id,
        fbd_config=fbd_cfg,
        gdrive_remote=gdrive_remote,
        gdrive_root=os.environ.get("GDRIVE_ROOT", "FBD_LORA_EXPERIMENTS"),
        wandb_run=wandb_run,
    )

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    global_step = 0
    num_steps_per_epoch = math.ceil(len(train_dataloader) / ga)
    num_epochs = math.ceil(max_train_steps / max(1, num_steps_per_epoch))

    logger.info(
        "Training: max_steps=%d, epochs=%d, steps_per_epoch=%d, "
        "batch_size=%d, grad_accum=%d, lr=%.2e",
        max_train_steps, num_epochs, num_steps_per_epoch, bs, ga, lr
    )

    unet.train()

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            if global_step >= max_train_steps:
                break

            with accelerator.accumulate(unet):
                # Encode images to latent space
                # pixel_values: [B, 3, H, W] in [-1, 1]
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                with torch.no_grad():
                    # latents shape: [B, 4, H/8, W/8]
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device
                ).long()

                # Add noise to latents (forward diffusion)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Encode text prompts
                # input_ids: [B, seq_len]
                input_ids = batch["input_ids"].to(accelerator.device)
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(input_ids)[0]

                # Predict noise: model_pred [B, 4, H/8, W/8]
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                # Standard denoising loss: MSE against ground-truth noise
                # L = ||eps_pred - eps||^2
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)

                # Gradient clipping (applied to unwrapped model's params)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, unet.parameters()),
                        max_grad_norm
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Count only completed optimizer steps
            if accelerator.sync_gradients:
                global_step += 1

                # Log metrics
                loss_val = loss.detach().float().item()
                lr_val = lr_scheduler.get_last_lr()[0]

                if global_step % 10 == 0 or global_step <= 5:
                    logger.info(
                        "Step %d/%d | loss=%.4f | lr=%.2e",
                        global_step, max_train_steps, loss_val, lr_val
                    )

                if wandb_run is not None and accelerator.is_main_process:
                    log_dict = {
                        "train/loss": loss_val,
                        "train/learning_rate": lr_val,
                        "train/step": global_step,
                        "train/epoch": epoch,
                    }
                    # Log FBD gradient stats if available
                    if fbd_state is not None:
                        fbd_stats = fbd_state.get_and_clear_stats()
                        if fbd_stats:
                            for k, v in fbd_stats.items():
                                log_dict[f"train/fbd_{k}"] = v
                    wandb_run.log(log_dict, step=global_step)

                # Save checkpoint at schedule points
                ckpt_manager.step(global_step, {"loss": loss_val})

                if global_step >= max_train_steps:
                    break

        if global_step >= max_train_steps:
            break

    # -----------------------------------------------------------------------
    # Save final checkpoint
    # -----------------------------------------------------------------------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        ckpt_manager.save_final(global_step, {"loss": loss_val})
        logger.info("Final checkpoint saved.")

        # Also save the full pipeline for easy generation
        final_dir = output_dir / "checkpoints" / "final"
        final_dir.mkdir(parents=True, exist_ok=True)

        # Save run metadata
        run_meta = {
            "run_id": run_id,
            "concept_name": concept_name,
            "instance_prompt": instance_prompt,
            "model_id": model_id,
            "total_steps": global_step,
            "rank": rank,
            "alpha": alpha,
            "fbd_enabled": fbd_enabled,
        }
        (output_dir / "metadata.json").write_text(json.dumps(run_meta, indent=2))

    # Cleanup FBD hooks
    if fbd_state is not None:
        from fbd_lora.fbd.hooks import remove_fbd_hooks
        remove_fbd_hooks(fbd_state)

    if wandb_run is not None:
        wandb_run.finish()

    accelerator.end_training()
    logger.info("Training complete. Run: %s", run_id)


if __name__ == "__main__":
    main()
