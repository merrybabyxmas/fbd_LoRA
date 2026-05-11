"""Dataset loaders for DreamBooth / CustomConcept101 personalization training.

ConceptDataset: loads all images from a directory, tokenizes the instance prompt.
download_customconcept101_concept: downloads one concept from the HuggingFace dataset.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}


def _load_images_from_dir(image_dir: str) -> List[Path]:
    """Return sorted list of image paths from a directory."""
    p = Path(image_dir)
    if not p.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    paths = sorted([
        f for f in p.iterdir()
        if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS
    ])
    if not paths:
        raise ValueError(f"No images found in: {image_dir}")
    return paths


class ConceptDataset(Dataset):
    """Dataset of concept images for personalization training.

    Each item contains tokenized instance prompt and pixel_values in [-1, 1].

    Args:
        image_dir: Directory containing concept images.
        instance_prompt: Text prompt describing the concept (e.g. "a photo of sks dog").
        tokenizer: HuggingFace tokenizer for the text encoder.
        size: Target image resolution (default 512).
        center_crop: Whether to center-crop the image (default False).
        random_flip: Whether to apply random horizontal flip (default True).
    """

    def __init__(
        self,
        image_dir: str,
        instance_prompt: str,
        tokenizer,
        size: int = 512,
        center_crop: bool = False,
        random_flip: bool = True,
    ) -> None:
        self.image_paths = _load_images_from_dir(image_dir)
        self.instance_prompt = instance_prompt
        self.tokenizer = tokenizer
        self.size = size

        # Build transform pipeline
        transform_list = [T.Resize(size, interpolation=T.InterpolationMode.BILINEAR)]
        if center_crop:
            transform_list.append(T.CenterCrop(size))
        else:
            transform_list.append(T.RandomCrop(size))
        if random_flip:
            transform_list.append(T.RandomHorizontalFlip())
        transform_list += [
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),  # maps [0,1] -> [-1, 1]
        ]
        self.transform = T.Compose(transform_list)

        # Pre-tokenize the instance prompt (same for all images)
        self._input_ids = self._tokenize(instance_prompt)

        logger.info(
            "ConceptDataset: %d images from '%s', prompt='%s'",
            len(self.image_paths), image_dir, instance_prompt
        )

    def _tokenize(self, prompt: str) -> torch.Tensor:
        """Tokenize prompt and return input_ids [seq_len]."""
        tokens = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return tokens.input_ids[0]  # [seq_len]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        """Return dict with pixel_values [3, H, W] in [-1, 1] and input_ids [seq_len]."""
        img_path = self.image_paths[idx % len(self.image_paths)]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning("Failed to load image %s: %s", img_path, e)
            img = Image.new("RGB", (self.size, self.size), color=128)

        pixel_values = self.transform(img)  # [3, H, W], float in [-1, 1]
        assert pixel_values.shape == (3, self.size, self.size), \
            f"Unexpected pixel_values shape: {pixel_values.shape}"

        return {
            "pixel_values": pixel_values,
            "input_ids": self._input_ids.clone(),
        }


class ConceptDatasetWithPrior(Dataset):
    """ConceptDataset with optional prior-preservation class images.

    Returns interleaved instance and class samples for DreamBooth training
    with prior preservation loss.

    Args:
        instance_dataset: ConceptDataset for the concept images.
        class_image_dir: Directory containing class images (for prior preservation).
        class_prompt: Text prompt for class images.
        tokenizer: Tokenizer for class prompt.
        size: Target image resolution.
        center_crop: Whether to center-crop class images.
    """

    def __init__(
        self,
        instance_dataset: ConceptDataset,
        class_image_dir: str,
        class_prompt: str,
        tokenizer,
        size: int = 512,
        center_crop: bool = True,
    ) -> None:
        self.instance_dataset = instance_dataset
        self.class_dataset = ConceptDataset(
            image_dir=class_image_dir,
            instance_prompt=class_prompt,
            tokenizer=tokenizer,
            size=size,
            center_crop=center_crop,
            random_flip=False,
        )

    def __len__(self) -> int:
        return max(len(self.instance_dataset), len(self.class_dataset))

    def __getitem__(self, idx: int) -> dict:
        instance_item = self.instance_dataset[idx % len(self.instance_dataset)]
        class_item = self.class_dataset[idx % len(self.class_dataset)]
        return {
            "pixel_values": instance_item["pixel_values"],
            "input_ids": instance_item["input_ids"],
            "class_pixel_values": class_item["pixel_values"],
            "class_input_ids": class_item["input_ids"],
        }


def collate_fn(batch: list) -> dict:
    """Collate function for ConceptDataset batches.

    Stacks pixel_values and input_ids. Handles optional class fields.
    """
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    input_ids = torch.stack([b["input_ids"] for b in batch])
    result = {"pixel_values": pixel_values, "input_ids": input_ids}

    if "class_pixel_values" in batch[0]:
        result["class_pixel_values"] = torch.stack([b["class_pixel_values"] for b in batch])
        result["class_input_ids"] = torch.stack([b["class_input_ids"] for b in batch])

    return result


def download_customconcept101_concept(
    concept_name: str,
    output_dir: str,
    hf_dataset_id: str = "fazzie/CustomConcept101",
    hf_token: Optional[str] = None,
) -> str:
    """Download images for a single CustomConcept101 concept from HuggingFace.

    Images are saved to output_dir/concept_name/*.jpg.

    Args:
        concept_name: Name of the concept (e.g. "cat_statue").
        output_dir: Root directory to save images under.
        hf_dataset_id: HuggingFace dataset identifier.
        hf_token: Optional HuggingFace token for private datasets.

    Returns:
        Path to directory containing downloaded images.

    Raises:
        RuntimeError: If download fails and no fallback succeeds.
    """
    from datasets import load_dataset

    concept_dir = Path(output_dir) / concept_name
    concept_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing = [f for f in concept_dir.iterdir() if f.suffix.lower() in _IMAGE_EXTENSIONS]
    if existing:
        logger.info("Concept '%s' already has %d images in %s", concept_name, len(existing), concept_dir)
        return str(concept_dir)

    logger.info("Downloading concept '%s' from %s ...", concept_name, hf_dataset_id)

    try:
        kwargs = {"split": "train"}
        if hf_token:
            kwargs["token"] = hf_token

        ds = load_dataset(hf_dataset_id, **kwargs)

        # Filter for the requested concept
        # CustomConcept101 has fields: image (PIL), label (str), image_id (str)
        concept_rows = [row for row in ds if row.get("label", "").lower().replace(" ", "_") == concept_name.lower()]

        if not concept_rows:
            # Try partial match
            concept_rows = [
                row for row in ds
                if concept_name.lower() in row.get("label", "").lower().replace(" ", "_")
            ]

        if not concept_rows:
            available = list(set(row.get("label", "") for row in ds))[:20]
            logger.warning(
                "Concept '%s' not found. Available (first 20): %s",
                concept_name, available
            )
            # Return all images from dataset as fallback
            concept_rows = list(ds)[:15]

        logger.info("Found %d images for concept '%s'", len(concept_rows), concept_name)

        for i, row in enumerate(concept_rows):
            img = row.get("image")
            if img is None:
                continue
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img = img.convert("RGB")
            out_path = concept_dir / f"image_{i:03d}.jpg"
            img.save(str(out_path), quality=95)
            logger.debug("Saved: %s", out_path)

        saved = list(concept_dir.glob("*.jpg"))
        logger.info("Downloaded %d images for '%s' to %s", len(saved), concept_name, concept_dir)
        return str(concept_dir)

    except Exception as e:
        logger.error("Failed to download from %s: %s", hf_dataset_id, e)
        raise RuntimeError(f"Could not download concept '{concept_name}': {e}") from e


def load_msbench_concept(
    output_dir: str,
    concept_id: int = 0,
    max_images: int = 4,
    hf_token: Optional[str] = None,
) -> str:
    """Load images for a single concept from doge1516/MS-Bench HuggingFace dataset.

    MS-Bench has columns: image (PIL), label (int, 0-based concept index).

    Args:
        output_dir: Root directory to save extracted images.
        concept_id: Integer concept index (0-6 for the 7 concepts in MS-Bench).
        max_images: Maximum number of images to extract.
        hf_token: Optional HF token.

    Returns:
        Path to directory containing extracted images.

    Raises:
        RuntimeError: If loading fails.
    """
    from datasets import load_dataset

    concept_name = f"concept_{concept_id:02d}"
    concept_dir = Path(output_dir) / concept_name
    concept_dir.mkdir(parents=True, exist_ok=True)

    existing = [f for f in concept_dir.iterdir() if f.suffix.lower() in _IMAGE_EXTENSIONS]
    if len(existing) >= max_images:
        logger.info("MS-Bench concept %d already has %d images in %s", concept_id, len(existing), concept_dir)
        return str(concept_dir)

    logger.info("Loading MS-Bench concept %d from HuggingFace...", concept_id)
    try:
        kwargs = {}
        if hf_token:
            kwargs["token"] = hf_token
        ds = load_dataset("doge1516/MS-Bench", split="train", **kwargs)

        # Filter by concept_id
        concept_rows = [row for row in ds if row["label"] == concept_id]
        if not concept_rows:
            available = sorted(set(row["label"] for row in ds))
            raise RuntimeError(
                f"Concept ID {concept_id} not found in MS-Bench. "
                f"Available concept IDs: {available}"
            )

        # Limit images
        concept_rows = concept_rows[:max_images]
        logger.info("Found %d images for MS-Bench concept %d", len(concept_rows), concept_id)

        for i, row in enumerate(concept_rows):
            img = row["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img = img.convert("RGB")
            out_path = concept_dir / f"image_{i:03d}.jpg"
            img.save(str(out_path), quality=95)
            logger.debug("Saved: %s", out_path)

        saved = list(concept_dir.glob("*.jpg"))
        logger.info("Extracted %d images for MS-Bench concept %d to %s", len(saved), concept_id, concept_dir)
        return str(concept_dir)

    except Exception as e:
        logger.error("Failed to load MS-Bench: %s", e)
        raise RuntimeError(f"Could not load MS-Bench concept {concept_id}: {e}") from e


def load_dreambench_plus_concept(
    output_dir: str,
    concept_id: int = 0,
    max_images: int = 4,
    hf_split: str = "test",
    hf_token: Optional[str] = None,
) -> str:
    """Load images for a single concept from yuangpeng/dreambench_plus HuggingFace dataset.

    DreamBench++ schema (if loadable) has per-concept image groups.
    Falls back to loading all available images from the concept group.

    Args:
        output_dir: Root directory to save extracted images.
        concept_id: Integer concept group index.
        max_images: Maximum number of images to extract.
        hf_split: Dataset split to use ('test' or 'train').
        hf_token: Optional HF token.

    Returns:
        Path to directory containing extracted images.

    Raises:
        RuntimeError: If loading fails.
    """
    from datasets import load_dataset

    concept_name = f"concept_{concept_id:02d}"
    concept_dir = Path(output_dir) / concept_name
    concept_dir.mkdir(parents=True, exist_ok=True)

    existing = [f for f in concept_dir.iterdir() if f.suffix.lower() in _IMAGE_EXTENSIONS]
    if len(existing) >= max_images:
        logger.info("DreamBench+ concept %d already has %d images.", concept_id, len(existing))
        return str(concept_dir)

    logger.info("Loading DreamBench+ concept %d from HuggingFace (split=%s)...", concept_id, hf_split)
    try:
        kwargs = {}
        if hf_token:
            kwargs["token"] = hf_token
        ds = load_dataset("yuangpeng/dreambench_plus", split=hf_split, **kwargs)

        # Try to group by concept; dataset may have 'label', 'subject', or ordering
        image_col = None
        label_col = None
        for col in ds.column_names:
            if "image" in col.lower():
                image_col = col
            if "label" in col.lower() or "subject" in col.lower() or "class" in col.lower():
                label_col = col

        if label_col is not None:
            concept_rows = [row for row in ds if row.get(label_col) == concept_id]
        else:
            # Fall back: assume rows are grouped by concept in fixed-size blocks
            # or just take the first max_images rows
            concept_rows = list(ds)[concept_id * max_images: (concept_id + 1) * max_images]
            if not concept_rows:
                concept_rows = list(ds)[:max_images]

        concept_rows = concept_rows[:max_images]
        logger.info("Found %d images for DreamBench+ concept %d", len(concept_rows), concept_id)

        for i, row in enumerate(concept_rows):
            img = row.get(image_col) if image_col else list(row.values())[0]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img = img.convert("RGB")
            out_path = concept_dir / f"image_{i:03d}.jpg"
            img.save(str(out_path), quality=95)

        saved = list(concept_dir.glob("*.jpg"))
        logger.info("Extracted %d images for DreamBench+ concept %d to %s", len(saved), concept_id, concept_dir)
        return str(concept_dir)

    except Exception as e:
        logger.error("Failed to load DreamBench+: %s", e)
        raise RuntimeError(f"Could not load DreamBench+ concept {concept_id}: {e}") from e


def download_dreambooth_concept(
    concept_name: str,
    output_dir: str,
    hf_token: Optional[str] = None,
) -> str:
    """Download a DreamBooth benchmark concept from HuggingFace.

    Uses 'google/dreambooth' dataset or falls back to shi-labs dataset.

    Args:
        concept_name: Subject name (e.g. "dog", "backpack", "teapot").
        output_dir: Root directory to save images.
        hf_token: Optional HF token.

    Returns:
        Path to directory with images.
    """
    from datasets import load_dataset

    concept_dir = Path(output_dir) / concept_name
    concept_dir.mkdir(parents=True, exist_ok=True)

    existing = [f for f in concept_dir.iterdir() if f.suffix.lower() in _IMAGE_EXTENSIONS]
    if existing:
        logger.info("Concept '%s' already exists with %d images.", concept_name, len(existing))
        return str(concept_dir)

    dataset_ids = ["google/dreambooth", "shi-labs/dreambooth_dataset"]
    for ds_id in dataset_ids:
        try:
            kwargs = {"split": "train"}
            if hf_token:
                kwargs["token"] = hf_token
            ds = load_dataset(ds_id, **kwargs)
            rows = [r for r in ds if concept_name.lower() in str(r.get("label", r.get("subject", ""))).lower()]
            if not rows:
                rows = list(ds)[:15]
            for i, row in enumerate(rows[:15]):
                img = row.get("image") or row.get("img")
                if img is None:
                    continue
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                img = img.convert("RGB")
                (concept_dir / f"image_{i:03d}.jpg").open("wb").close()
                img.save(str(concept_dir / f"image_{i:03d}.jpg"), quality=95)
            logger.info("Downloaded from %s: %d images for '%s'", ds_id, len(rows), concept_name)
            return str(concept_dir)
        except Exception as e:
            logger.warning("Failed to load from %s: %s", ds_id, e)

    raise RuntimeError(f"Could not download DreamBooth concept '{concept_name}'")
