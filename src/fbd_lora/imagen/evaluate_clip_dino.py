"""CLIP-I, DINO, and CLIP-T evaluation for image personalization.

Computes:
    CLIP-I:  CLIP image-image cosine similarity between generated and reference images.
    DINO:    DINO ViT-S/16 image-image cosine similarity.
    CLIP-T:  CLIP text-image cosine similarity between generated images and prompts.

Usage:
    python -m fbd_lora.imagen.evaluate_clip_dino \\
        --concept_dir data/customconcept101/cat_statue \\
        --generated_dir outputs/runs/<run_id>/eval/generated \\
        --prompts_file data/eval_prompts.txt \\
        --output_dir outputs/runs/<run_id>/eval

Results are saved to:
    eval/image_metrics_by_sample.csv
    eval/image_metrics_summary.json
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Image loading helpers
# -------------------------------------------------------------------------

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _load_image_paths(directory: str) -> List[Path]:
    """Return sorted list of image paths from a directory tree."""
    p = Path(directory)
    return sorted([
        f for f in p.rglob("*")
        if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS
    ])


def _load_pil_images(paths: List[Path]):
    """Load PIL images, skipping corrupt files."""
    from PIL import Image
    images = []
    for p in paths:
        try:
            images.append(Image.open(p).convert("RGB"))
        except Exception as e:
            logger.warning("Skipping corrupt image %s: %s", p, e)
    return images


# -------------------------------------------------------------------------
# CLIP model loader (open_clip)
# -------------------------------------------------------------------------

_clip_model_cache = {}


def _get_clip_model(device: str = "cuda"):
    """Load and cache open_clip ViT-B-32 model."""
    global _clip_model_cache
    if "model" not in _clip_model_cache:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        _clip_model_cache = {"model": model, "preprocess": preprocess, "tokenizer": tokenizer}
        logger.info("Loaded open_clip ViT-B-32 (openai) on %s", device)
    return (
        _clip_model_cache["model"],
        _clip_model_cache["preprocess"],
        _clip_model_cache["tokenizer"],
    )


def _embed_images_clip(
    images,
    device: str = "cuda",
    batch_size: int = 32,
) -> torch.Tensor:
    """Compute normalized CLIP image embeddings. Returns [N, D]."""
    model, preprocess, _ = _get_clip_model(device)
    all_features = []
    for i in range(0, len(images), batch_size):
        batch = [preprocess(img) for img in images[i:i+batch_size]]
        batch_tensor = torch.stack(batch).to(device)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda" if device != "cpu" else "cpu"):
            feats = model.encode_image(batch_tensor)
        feats = F.normalize(feats.float(), dim=-1)
        all_features.append(feats.cpu())
    return torch.cat(all_features, dim=0)  # [N, D]


def _embed_texts_clip(
    texts: List[str],
    device: str = "cuda",
    batch_size: int = 64,
) -> torch.Tensor:
    """Compute normalized CLIP text embeddings. Returns [N, D]."""
    model, _, tokenizer = _get_clip_model(device)
    all_features = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        tokens = tokenizer(batch_texts).to(device)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda" if device != "cpu" else "cpu"):
            feats = model.encode_text(tokens)
        feats = F.normalize(feats.float(), dim=-1)
        all_features.append(feats.cpu())
    return torch.cat(all_features, dim=0)  # [N, D]


# -------------------------------------------------------------------------
# DINO model loader
# -------------------------------------------------------------------------

_dino_model_cache = {}


def _get_dino_model(device: str = "cuda"):
    """Load and cache DINO ViT-S/16 model."""
    global _dino_model_cache
    if "model" not in _dino_model_cache:
        try:
            model = torch.hub.load("facebookresearch/dino:main", "dino_vits16", verbose=False)
        except Exception:
            # Fallback: use timm
            try:
                import timm
                model = timm.create_model("vit_small_patch16_224.dino", pretrained=True)
            except Exception as e:
                raise RuntimeError("Could not load DINO model: %s" % e) from e
        model = model.to(device).eval()
        _dino_model_cache["model"] = model
        logger.info("Loaded DINO ViT-S/16 on %s", device)

    return _dino_model_cache["model"]


def _dino_preprocess(images, size: int = 224):
    """Preprocess PIL images for DINO (resize, normalize)."""
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return torch.stack([transform(img) for img in images])


def _embed_images_dino(
    images,
    device: str = "cuda",
    batch_size: int = 32,
) -> torch.Tensor:
    """Compute normalized DINO image embeddings. Returns [N, D]."""
    model = _get_dino_model(device)
    all_features = []
    for i in range(0, len(images), batch_size):
        batch = _dino_preprocess(images[i:i+batch_size]).to(device)
        with torch.no_grad():
            feats = model(batch)
        feats = F.normalize(feats.float(), dim=-1)
        all_features.append(feats.cpu())
    return torch.cat(all_features, dim=0)  # [N, D]


# -------------------------------------------------------------------------
# Core metric functions
# -------------------------------------------------------------------------

def compute_clip_i(
    generated_images,
    reference_images,
    device: str = "cuda",
) -> float:
    """CLIP image-image similarity: average cosine sim between generated and reference.

    For each generated image, computes max similarity to any reference image,
    then averages over all generated images.

    Args:
        generated_images: List of PIL images (generated).
        reference_images: List of PIL images (reference/concept).
        device: Torch device.

    Returns:
        Mean CLIP-I score in [0, 1].
    """
    if not generated_images or not reference_images:
        logger.warning("Empty image list for CLIP-I computation.")
        return 0.0

    gen_feats = _embed_images_clip(generated_images, device=device)    # [N_gen, D]
    ref_feats = _embed_images_clip(reference_images, device=device)    # [N_ref, D]

    # Cosine similarity matrix: [N_gen, N_ref]
    sim_matrix = gen_feats @ ref_feats.T  # already normalized

    # For each generated image, take max sim over reference images, then average
    max_sims = sim_matrix.max(dim=1).values   # [N_gen]
    score = max_sims.mean().item()
    logger.info("CLIP-I: %.4f (avg max-sim over %d gen vs %d ref images)", score, len(generated_images), len(reference_images))
    return score


def compute_dino_score(
    generated_images,
    reference_images,
    device: str = "cuda",
) -> float:
    """DINO ViT-S/16 similarity between generated and reference images.

    Args:
        generated_images: List of PIL images (generated).
        reference_images: List of PIL images (reference/concept).
        device: Torch device.

    Returns:
        Mean DINO score in [0, 1].
    """
    if not generated_images or not reference_images:
        logger.warning("Empty image list for DINO computation.")
        return 0.0

    gen_feats = _embed_images_dino(generated_images, device=device)
    ref_feats = _embed_images_dino(reference_images, device=device)

    sim_matrix = gen_feats @ ref_feats.T
    max_sims = sim_matrix.max(dim=1).values
    score = max_sims.mean().item()
    logger.info("DINO: %.4f (avg max-sim over %d gen vs %d ref images)", score, len(generated_images), len(reference_images))
    return score


def compute_clip_t(
    generated_images,
    prompts: List[str],
    device: str = "cuda",
) -> float:
    """CLIP text-image similarity between generated images and their prompts.

    Args:
        generated_images: List of PIL images.
        prompts: List of text prompts (same length as generated_images, or one prompt
                 for all images).
        device: Torch device.

    Returns:
        Mean CLIP-T score in [0, 1].
    """
    if not generated_images or not prompts:
        logger.warning("Empty input for CLIP-T computation.")
        return 0.0

    # Broadcast single prompt to all images
    if len(prompts) == 1:
        prompts = prompts * len(generated_images)
    elif len(prompts) != len(generated_images):
        # Use first prompt repeated if mismatch
        logger.warning(
            "CLIP-T: prompts length (%d) != images length (%d). Using first prompt.",
            len(prompts), len(generated_images)
        )
        prompts = [prompts[0]] * len(generated_images)

    img_feats = _embed_images_clip(generated_images, device=device)   # [N, D]
    txt_feats = _embed_texts_clip(prompts, device=device)              # [N, D]

    # Per-sample cosine similarity (dot product of normalized vectors)
    similarities = (img_feats * txt_feats).sum(dim=-1)  # [N]
    score = similarities.mean().item()
    logger.info("CLIP-T: %.4f (over %d image-prompt pairs)", score, len(generated_images))
    return score


# -------------------------------------------------------------------------
# Concept-level evaluation
# -------------------------------------------------------------------------

def evaluate_concept(
    concept_dir: str,
    generated_dir: str,
    prompts: List[str],
    output_dir: str,
    device: str = "cuda",
    compute_dino: bool = True,
) -> Dict[str, float]:
    """Compute CLIP-I, DINO, CLIP-T for a single concept.

    Saves results to CSV and JSON in output_dir.

    Args:
        concept_dir: Directory with reference concept images.
        generated_dir: Directory tree with generated images (recursive search).
        prompts: List of text prompts used for generation.
        output_dir: Directory to save evaluation results.
        device: Torch device.
        compute_dino: Whether to compute DINO score (slow, requires internet first time).

    Returns:
        Dict with keys clip_i, dino (optional), clip_t.
    """
    import csv

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load images
    ref_paths = _load_image_paths(concept_dir)
    gen_paths = _load_image_paths(generated_dir)

    if not ref_paths:
        raise ValueError(f"No reference images found in: {concept_dir}")
    if not gen_paths:
        raise ValueError(f"No generated images found in: {generated_dir}")

    ref_images = _load_pil_images(ref_paths)
    gen_images = _load_pil_images(gen_paths)

    logger.info(
        "Evaluating: %d reference images, %d generated images",
        len(ref_images), len(gen_images)
    )

    # Compute metrics
    clip_i = compute_clip_i(gen_images, ref_images, device=device)
    clip_t = compute_clip_t(gen_images, prompts, device=device)

    metrics: Dict[str, float] = {
        "clip_i": clip_i,
        "clip_t": clip_t,
        "num_generated": len(gen_images),
        "num_reference": len(ref_images),
    }

    if compute_dino:
        try:
            dino = compute_dino_score(gen_images, ref_images, device=device)
            metrics["dino"] = dino
        except Exception as e:
            logger.warning("DINO computation failed: %s", e)
            metrics["dino"] = -1.0

    # Save JSON summary
    summary_path = output_path / "image_metrics_summary.json"
    summary_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics saved to %s", summary_path)

    # Save per-image CSV
    csv_path = output_path / "image_metrics_by_sample.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "prompt", "clip_t_score"])
        writer.writeheader()
        per_img_prompts = prompts if len(prompts) == len(gen_images) else [prompts[0]] * len(gen_images)
        for img_path, prompt in zip(gen_paths, per_img_prompts):
            writer.writerow({
                "image_path": str(img_path),
                "prompt": prompt,
                "clip_t_score": "",  # individual CLIP-T requires per-image compute
            })
    logger.info("Per-sample CSV saved to %s", csv_path)

    # Print summary
    logger.info("=" * 50)
    logger.info("Evaluation Summary:")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info("  %s: %.4f", k, v)
    logger.info("=" * 50)

    return metrics


# -------------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLIP-I / DINO / CLIP-T evaluation")
    parser.add_argument("--concept_dir", type=str, required=True,
                        help="Directory with reference concept images.")
    parser.add_argument("--generated_dir", type=str, required=True,
                        help="Directory (tree) with generated images.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save evaluation metrics.")
    parser.add_argument("--prompts_file", type=str, default=None,
                        help="Path to prompts file (txt or json).")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt (fallback if no prompts_file).")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device.")
    parser.add_argument("--no_dino", action="store_true",
                        help="Skip DINO computation.")
    return parser.parse_args()


def _load_prompts_from_args(args: argparse.Namespace) -> List[str]:
    """Load prompts from CLI args."""
    prompts = []
    if args.prompts_file:
        p = Path(args.prompts_file)
        if p.suffix.lower() == ".json":
            with open(p) as f:
                data = json.load(f)
            prompts = data if isinstance(data, list) else list(data.values())
        else:
            prompts = [l.strip() for l in p.read_text().splitlines() if l.strip()]
    if args.prompt:
        prompts.append(args.prompt)
    if not prompts:
        prompts = ["a photo of the concept"]
        logger.warning("No prompts provided; using default: '%s'", prompts[0])
    return prompts


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    import torch
    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    prompts = _load_prompts_from_args(args)

    metrics = evaluate_concept(
        concept_dir=args.concept_dir,
        generated_dir=args.generated_dir,
        prompts=prompts,
        output_dir=args.output_dir,
        device=device,
        compute_dino=not args.no_dino,
    )

    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
        else:
            print(f"  {k:20s}: {v}")


if __name__ == "__main__":
    main()
