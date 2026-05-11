"""Image generation script for FBD-LoRA trained DreamBooth adapters.

Loads a LoRA adapter from a checkpoint directory, generates images for each
prompt (from CLI or file), and saves them under a structured output directory.

Usage:
    python -m fbd_lora.imagen.generate \\
        --checkpoint outputs/runs/<run_id>/checkpoints/final \\
        --prompt "a photo of sks dog in the park" \\
        --output_dir outputs/runs/<run_id>/eval/generated \\
        --num_images 4 \\
        --seed 42

Or with a prompts file (one prompt per line or JSON array):
    python -m fbd_lora.imagen.generate \\
        --checkpoint outputs/runs/<run_id>/checkpoints/final \\
        --prompts_file data/eval_prompts.txt \\
        --output_dir outputs/runs/<run_id>/eval/generated
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images with FBD-LoRA adapter")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory (containing adapter_model.safetensors).")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Base model ID.")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single generation prompt.")
    parser.add_argument("--prompts_file", type=str, default=None,
                        help="Path to prompts file (.txt one-per-line or .json list).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save generated images.")
    parser.add_argument("--num_images", type=int, default=4,
                        help="Number of images per prompt.")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def slugify(text: str, max_len: int = 50) -> str:
    """Convert prompt text to a safe directory name."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower())
    slug = slug.strip("_")[:max_len]
    return slug or "prompt"


def load_prompts(args: argparse.Namespace) -> List[str]:
    """Load prompts from CLI arg or file."""
    prompts = []
    if args.prompt:
        prompts.append(args.prompt)
    if args.prompts_file:
        p = Path(args.prompts_file)
        if not p.exists():
            logger.warning("Prompts file not found: %s", p)
        elif p.suffix.lower() == ".json":
            with open(p) as f:
                data = json.load(f)
            if isinstance(data, list):
                prompts.extend([str(x) for x in data])
            elif isinstance(data, dict):
                # Support {concept: [prompts]} format
                for v in data.values():
                    if isinstance(v, list):
                        prompts.extend([str(x) for x in v])
                    else:
                        prompts.append(str(v))
        else:
            # Plain text, one prompt per line
            prompts.extend([line.strip() for line in p.read_text().splitlines() if line.strip()])
    if not prompts:
        raise ValueError("No prompts provided. Use --prompt or --prompts_file.")
    logger.info("Loaded %d prompts.", len(prompts))
    return prompts


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import torch
    from diffusers import StableDiffusionPipeline
    from PIL import Image

    # Load secrets
    for candidate in [Path.cwd() / ".env", Path(__file__).parents[4] / ".env"]:
        if candidate.exists():
            from fbd_lora.secrets import load_env_file
            load_env_file(str(candidate))
            break

    hf_token = os.environ.get("HF_TOKEN")

    # Determine weight dtype
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    weight_dtype = dtype_map.get(args.mixed_precision, torch.float16)
    device = args.device if torch.cuda.is_available() else "cpu"

    checkpoint_dir = Path(args.checkpoint)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    logger.info("Loading base pipeline from '%s'", args.model_id)
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=weight_dtype,
        token=hf_token,
        safety_checker=None,
    )

    # Load LoRA adapter from checkpoint
    # Strategy: try multiple methods in order (diffusers 0.37.1 + PEFT 0.17.0 compatible)
    # The checkpoints are saved in PEFT format (adapter_model.safetensors + adapter_config.json)
    # with keys like 'base_model.model.down_blocks...lora_A.weight'.
    # PeftModel.from_pretrained is the canonical way to load these.
    logger.info("Loading LoRA adapter from '%s'", checkpoint_dir)

    adapter_loaded = False

    # Method 1: PeftModel.from_pretrained (primary method for PEFT-format checkpoints)
    # This handles the key format 'base_model.model...' correctly.
    if (checkpoint_dir / "adapter_model.safetensors").exists() and \
       (checkpoint_dir / "adapter_config.json").exists():
        try:
            from peft import PeftModel
            pipeline.unet = PeftModel.from_pretrained(pipeline.unet, str(checkpoint_dir))
            # Merge LoRA weights into base model for faster inference (no extra compute per layer)
            pipeline.unet = pipeline.unet.merge_and_unload()
            pipeline.unet.eval()
            adapter_loaded = True
            logger.info("LoRA adapter loaded and merged via PeftModel.from_pretrained + merge_and_unload.")
        except Exception as e1:
            logger.warning("PeftModel.from_pretrained failed (%s)", e1)

    # Method 2: pipeline.load_lora_weights (for diffusers-format checkpoints)
    # Only works if keys have 'unet.' prefix in the safetensors file.
    if not adapter_loaded and (
        (checkpoint_dir / "adapter_model.safetensors").exists() or
        (checkpoint_dir / "adapter_config.json").exists()
    ):
        try:
            import safetensors.torch as _st
            _ckpt_path = checkpoint_dir / "adapter_model.safetensors"
            if _ckpt_path.exists():
                _keys = list(_st.load_file(str(_ckpt_path)).keys())
                # Only use this method if keys have 'unet.' prefix (diffusers format)
                if any(k.startswith("unet.") for k in _keys):
                    pipeline.load_lora_weights(str(checkpoint_dir))
                    adapter_loaded = True
                    logger.info("LoRA adapter loaded via pipeline.load_lora_weights.")
                else:
                    logger.info(
                        "Skipping pipeline.load_lora_weights: keys are PEFT format, "
                        "not diffusers format (first key: %s)", _keys[0] if _keys else "none"
                    )
        except Exception as e2:
            logger.warning("pipeline.load_lora_weights failed (%s)", e2)

    # Method 3: load_attn_procs (older diffusers API, needs pytorch_lora_weights.bin)
    if not adapter_loaded and (checkpoint_dir / "pytorch_lora_weights.bin").exists():
        try:
            pipeline.unet.load_attn_procs(str(checkpoint_dir))
            adapter_loaded = True
            logger.info("LoRA adapter loaded via load_attn_procs.")
        except Exception as e3:
            logger.warning("load_attn_procs failed: %s", e3)

    if not adapter_loaded:
        logger.warning(
            "No adapter loaded from %s; generating with base model only.",
            checkpoint_dir
        )

    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    # Load prompts
    prompts = load_prompts(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating %d images per prompt, %d prompts total", args.num_images, len(prompts))

    all_generated_paths = []

    for prompt_idx, prompt in enumerate(prompts):
        prompt_slug = slugify(prompt)
        prompt_dir = output_dir / prompt_slug
        prompt_dir.mkdir(parents=True, exist_ok=True)

        logger.info("[%d/%d] Prompt: '%s'", prompt_idx + 1, len(prompts), prompt)

        for img_idx in range(args.num_images):
            # Deterministic but varied seeds per image
            img_seed = args.seed + prompt_idx * 1000 + img_idx
            generator = torch.Generator(device=device).manual_seed(img_seed)

            with torch.no_grad():
                result = pipeline(
                    prompt=prompt,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                    num_images_per_prompt=1,
                )

            image = result.images[0]
            out_path = prompt_dir / f"image_{img_idx:04d}.png"
            image.save(str(out_path))
            all_generated_paths.append(str(out_path))
            logger.debug("Saved: %s", out_path)

        logger.info("Generated %d images for prompt '%s'", args.num_images, prompt_slug)

    # Save manifest
    manifest = {
        "checkpoint": str(checkpoint_dir),
        "model_id": args.model_id,
        "num_images_per_prompt": args.num_images,
        "seed": args.seed,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "prompts": prompts,
        "total_images": len(all_generated_paths),
        "output_dir": str(output_dir),
    }
    (output_dir / "generation_manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Generation complete. %d images saved to '%s'", len(all_generated_paths), output_dir)


if __name__ == "__main__":
    main()
