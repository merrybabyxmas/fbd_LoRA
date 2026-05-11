"""Dataset loaders for DreamBooth / CustomConcept101 personalization training.

ConceptDataset: loads all images from a directory, tokenizes the instance prompt.
download_customconcept101_concept: downloads one concept from the HuggingFace dataset.
load_dreambench_plus: official-file loader for yuangpeng/dreambench_plus via snapshot_download.
"""

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}

# Extensions recognized as concept images (spec subset, case-insensitive)
_DREAMBENCH_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Directories to exclude when scanning
_EXCLUDE_DIRS = {"outputs", "results", "samples", "generated", "__pycache__", ".git"}

# Flexible key recognition
_IMAGE_KEYS = {
    "image", "image_path", "image_file", "img", "img_path",
    "path", "file", "filename", "reference_image", "ref_image",
    "source_image",
}
_PROMPT_KEYS = {
    "prompt", "prompts", "text", "caption", "captions",
    "eval_prompt", "evaluation_prompt", "target_prompt",
}
_CONCEPT_KEYS = {
    "concept", "concept_id", "subject", "subject_id", "id", "name", "category",
}


# ---------------------------------------------------------------------------
# Directory tree printer (for error diagnostics)
# ---------------------------------------------------------------------------

def print_directory_tree(root: str, max_depth: int = 3, max_files: int = 200) -> str:
    """Return a human-readable directory tree string for diagnostics.

    Args:
        root: Root directory path.
        max_depth: Maximum recursion depth (default 3).
        max_files: Maximum total files to print before truncating (default 200).

    Returns:
        Multi-line string representation of the tree.
    """
    root_path = Path(root)
    if not root_path.exists():
        return f"[directory does not exist: {root}]"

    lines: List[str] = [str(root_path)]
    file_count = [0]

    def _recurse(path: Path, depth: int, prefix: str) -> None:
        if depth > max_depth or file_count[0] >= max_files:
            return
        try:
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        except PermissionError:
            lines.append(f"{prefix}[permission denied]")
            return

        for i, entry in enumerate(entries):
            is_last = (i == len(entries) - 1)
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{entry.name}")
            file_count[0] += 1
            if file_count[0] >= max_files:
                lines.append(f"{prefix}    ... (truncated at {max_files} entries)")
                return
            if entry.is_dir() and entry.name not in _EXCLUDE_DIRS:
                child_prefix = prefix + ("    " if is_last else "│   ")
                _recurse(entry, depth + 1, child_prefix)

    _recurse(root_path, 1, "")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DreamBench++ official-file loader
# ---------------------------------------------------------------------------

def resolve_dreambench_plus_root(data_cfg) -> Optional[str]:
    """Resolve the local data root from data config, expanding env vars and ~.

    Args:
        data_cfg: OmegaConf node or dict with a ``local_data_root`` field.

    Returns:
        Expanded path string if set and non-empty, else None.
    """
    # Support both OmegaConf node and plain dict
    if hasattr(data_cfg, "get"):
        raw = data_cfg.get("local_data_root", "") or ""
    else:
        try:
            from omegaconf import OmegaConf
            raw = OmegaConf.select(data_cfg, "local_data_root", default="") or ""
        except Exception:
            raw = getattr(data_cfg, "local_data_root", "") or ""

    raw = str(raw).strip()
    if not raw:
        return None
    # Expand ${VAR} style
    raw = os.path.expandvars(raw)
    # Expand ~
    raw = os.path.expanduser(raw)
    # If after expansion still has unresolved ${...}, treat as not set
    if raw.startswith("${") or not raw:
        return None
    return raw


def _find_metadata_file(root: Path) -> Optional[Path]:
    """Search for a metadata file in root directory.

    Priority: metadata.json > metadata.jsonl > prompts.json > prompts.jsonl >
              captions.json > *.json (first) > metadata.csv > prompts.csv > *.csv (first) >
              metadata.tsv > *.tsv (first)
    """
    candidates = [
        "metadata.json", "metadata.jsonl", "prompts.json", "prompts.jsonl",
        "captions.json", "labels.json", "annotations.json",
        "metadata.csv", "prompts.csv", "captions.csv",
        "metadata.tsv", "prompts.tsv",
    ]
    for name in candidates:
        p = root / name
        if p.exists():
            return p
    # Try any JSON file
    for p in sorted(root.glob("*.json")):
        return p
    # Try any JSONL file
    for p in sorted(root.glob("*.jsonl")):
        return p
    # Try any CSV
    for p in sorted(root.glob("*.csv")):
        return p
    # Try any TSV
    for p in sorted(root.glob("*.tsv")):
        return p
    return None


def _extract_value(record: dict, keys: set) -> Optional[Any]:
    """Extract the first value matching any key from keys set (case-insensitive)."""
    for k, v in record.items():
        if k.lower() in keys:
            return v
    return None


def _parse_prompts(raw) -> List[str]:
    """Normalize prompt field to list of non-empty strings."""
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = raw.strip()
        return [raw] if raw else []
    if isinstance(raw, list):
        result = []
        for item in raw:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    result.append(s)
        return result
    return []


def _parse_metadata_file(meta_path: Path, concept_dir: Path) -> Dict[str, Any]:
    """Parse a metadata file and return mapping of image_path -> list[prompt].

    Returns:
        Dict mapping relative-or-absolute image path str → list of prompt strings.
        Also returns raw parsed records under key ``_raw_records`` for dict-format datasets.
    """
    suffix = meta_path.suffix.lower()
    result: Dict[str, List[str]] = {}

    try:
        if suffix in (".json",):
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                # JSON list format: [{"concept_id":..., "image":..., "prompts":[...]}, ...]
                for rec in data:
                    if not isinstance(rec, dict):
                        continue
                    img_val = _extract_value(rec, _IMAGE_KEYS)
                    prompt_val = _extract_value(rec, _PROMPT_KEYS)
                    prompts = _parse_prompts(prompt_val)
                    if img_val:
                        result[str(img_val)] = prompts
                    else:
                        # Might be a concept-level record without image key
                        concept_val = _extract_value(rec, _CONCEPT_KEYS)
                        if concept_val is not None:
                            result[f"__concept__{concept_val}"] = prompts

            elif isinstance(data, dict):
                if "items" in data and isinstance(data["items"], list):
                    # JSON dict with items key
                    for rec in data["items"]:
                        if not isinstance(rec, dict):
                            continue
                        img_val = _extract_value(rec, _IMAGE_KEYS)
                        prompt_val = _extract_value(rec, _PROMPT_KEYS)
                        prompts = _parse_prompts(prompt_val)
                        if img_val:
                            result[str(img_val)] = prompts
                else:
                    # JSON mapping: {"concept_001": {"image":..., "prompts":[...]}}
                    for key, val in data.items():
                        if isinstance(val, dict):
                            img_val = _extract_value(val, _IMAGE_KEYS)
                            prompt_val = _extract_value(val, _PROMPT_KEYS)
                            prompts = _parse_prompts(prompt_val)
                            if img_val:
                                result[str(img_val)] = prompts
                            else:
                                result[f"__concept__{key}"] = prompts
                        elif isinstance(val, list):
                            # value is a list of prompts for key=image_path
                            prompts = _parse_prompts(val)
                            result[str(key)] = prompts
                        elif isinstance(val, str):
                            result[str(key)] = [val.strip()] if val.strip() else []

        elif suffix in (".jsonl",):
            with open(meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(rec, dict):
                        continue
                    img_val = _extract_value(rec, _IMAGE_KEYS)
                    prompt_val = _extract_value(rec, _PROMPT_KEYS)
                    prompts = _parse_prompts(prompt_val)
                    if img_val:
                        result[str(img_val)] = prompts

        elif suffix in (".csv", ".tsv"):
            delimiter = "\t" if suffix == ".tsv" else ","
            with open(meta_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                for rec in reader:
                    lower_rec = {k.lower().strip(): v for k, v in rec.items()}
                    img_val = None
                    for key in _IMAGE_KEYS:
                        if key in lower_rec:
                            img_val = lower_rec[key]
                            break
                    prompt_val = None
                    for key in _PROMPT_KEYS:
                        if key in lower_rec:
                            prompt_val = lower_rec[key]
                            break
                    prompts = _parse_prompts(prompt_val)
                    if img_val:
                        result[str(img_val)] = prompts

    except Exception as e:
        logger.warning("Failed to parse metadata file %s: %s", meta_path, e)

    return result


def _collect_images_in_dir(concept_dir: Path, max_images: Optional[int] = None) -> List[Path]:
    """Collect image files from concept directory, excluding excluded subdirs."""
    images = []
    for p in sorted(concept_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in _DREAMBENCH_IMAGE_EXTENSIONS:
            images.append(p)
        if max_images is not None and len(images) >= max_images:
            break
    return images


def _resolve_image_path(img_key: str, concept_dir: Path) -> Optional[Path]:
    """Try to resolve an image key (possibly relative) to an existing path."""
    # Try as-is
    p = Path(img_key)
    if p.is_absolute() and p.exists():
        return p
    # Relative to concept_dir
    candidate = concept_dir / img_key
    if candidate.exists():
        return candidate
    # Just the filename part
    candidate2 = concept_dir / Path(img_key).name
    if candidate2.exists():
        return candidate2
    return None


def _build_concept_from_dir(
    concept_dir: Path,
    concept_id: str,
    category: Optional[str],
    metadata: Dict[str, List[str]],
    max_train_images: Optional[int],
    max_eval_prompts: Optional[int],
    allow_sanity_prompt_fallback: bool,
) -> Optional[dict]:
    """Build a concept dict from a directory with optional metadata mapping.

    Args:
        concept_dir: Path to directory containing concept images.
        concept_id: String identifier for this concept.
        category: Category label or None.
        metadata: dict of image_path_str -> list[prompt], parsed from metadata file.
        max_train_images: Limit on train images.
        max_eval_prompts: Limit on eval prompts.
        allow_sanity_prompt_fallback: If True, use fallback prompts when none found.

    Returns:
        Concept dict or None if the directory has no images.
    """
    # Collect images
    all_images = _collect_images_in_dir(concept_dir, max_images=None)
    if not all_images:
        return None

    # Gather all prompts from metadata
    all_prompts: List[str] = []

    # Try to match images to metadata keys
    matched_prompts: Dict[str, List[str]] = {}
    for img_path in all_images:
        # Try different lookups
        for key in [img_path.name, str(img_path), img_path.stem]:
            if key in metadata:
                matched_prompts[str(img_path)] = metadata[key]
                break

    # Collect prompts from matched records
    for prompts in matched_prompts.values():
        for p in prompts:
            if p not in all_prompts:
                all_prompts.append(p)

    # If no image-level match, look for concept-level prompts
    if not all_prompts:
        for key, prompts in metadata.items():
            if key.startswith("__concept__") or not any(
                (concept_dir / key).exists() or (concept_dir / Path(key).name).exists()
                for _ in [1]  # just trigger the loop
            ):
                for p in prompts:
                    if p and p not in all_prompts:
                        all_prompts.append(p)

    # Fallback: look for .txt prompt files in directory
    if not all_prompts:
        for txt_name in ["eval_prompts.txt", "prompts.txt", "prompt.txt"]:
            txt_path = concept_dir / txt_name
            if txt_path.exists():
                lines = txt_path.read_text(encoding="utf-8").strip().splitlines()
                for line in lines:
                    line = line.strip()
                    if line:
                        all_prompts.append(line)
                if all_prompts:
                    break

    if not all_prompts:
        if allow_sanity_prompt_fallback:
            fallback = f"a photo of {concept_id.replace('_', ' ')}"
            all_prompts = [fallback]
            logger.warning(
                "[WARNING] DreamBench++ prompt metadata not found. "
                "Using sanity-only fallback prompts."
            )
            used_fallback = True
        else:
            return None  # Will be caught by caller
        used_fallback = True
    else:
        used_fallback = False

    # Apply limits
    train_images = all_images[:max_train_images] if max_train_images else all_images
    eval_prompts = all_prompts[:max_eval_prompts] if max_eval_prompts else all_prompts

    return {
        "concept_id": concept_id,
        "category": category,
        "train_images": [str(p) for p in train_images],
        "eval_prompts": eval_prompts,
        "reference_images": [str(p) for p in all_images],
        "metadata": {
            "num_train_images_original": len(all_images),
            "num_eval_prompts_original": len(all_prompts),
            "used_sanity_prompt_fallback": used_fallback,
            "concept_dir": str(concept_dir),
        },
    }


def load_dreambench_plus(data_cfg) -> List[dict]:
    """Load DreamBench++ dataset from local files or HuggingFace snapshot.

    Loading priority:
    1. If data_cfg.local_data_root is set → load from that local path.
    2. Elif data_cfg.allow_hf_snapshot_download=true → use snapshot_download.
    3. Else → raise clear error.

    Args:
        data_cfg: OmegaConf node or dict with fields:
            - local_data_root: str (optional, empty = not set)
            - allow_hf_snapshot_download: bool (default False)
            - allow_fallback: bool (default False)
            - allow_sanity_prompt_fallback: bool (default False)
            - max_concepts: int (optional)
            - max_train_images_per_concept: int (optional)
            - max_eval_prompts_per_concept: int (optional)
            - hf_repo_id: str (default "yuangpeng/dreambench_plus")
            - print_dataset_tree_on_error: bool (default True)
            - max_tree_depth: int (default 3)
            - max_tree_files: int (default 200)

    Returns:
        List of concept dicts, each with keys:
            concept_id, category, train_images, eval_prompts,
            reference_images, metadata.

    Raises:
        RuntimeError: If loading fails and allow_fallback=False.
        ValueError: If the loaded dataset fails validation.
    """
    # -----------------------------------------------------------------------
    # Read config fields
    # -----------------------------------------------------------------------
    def _get(key, default=None):
        if hasattr(data_cfg, "get"):
            return data_cfg.get(key, default)
        try:
            from omegaconf import OmegaConf
            val = OmegaConf.select(data_cfg, key, default=default)
            return val if val is not None else default
        except Exception:
            return getattr(data_cfg, key, default)

    allow_hf_snapshot = bool(_get("allow_hf_snapshot_download", False))
    allow_fallback = bool(_get("allow_fallback", False))
    allow_sanity_prompt_fallback = bool(_get("allow_sanity_prompt_fallback", False))
    max_concepts = _get("max_concepts", None)
    max_train_images = _get("max_train_images_per_concept", None)
    max_eval_prompts = _get("max_eval_prompts_per_concept", None)
    hf_repo_id = _get("hf_repo_id", "yuangpeng/dreambench_plus")
    print_tree_on_error = bool(_get("print_dataset_tree_on_error", True))
    max_tree_depth = int(_get("max_tree_depth", 3))
    max_tree_files = int(_get("max_tree_files", 200))

    if max_train_images is not None:
        max_train_images = int(max_train_images)
    if max_eval_prompts is not None:
        max_eval_prompts = int(max_eval_prompts)
    if max_concepts is not None:
        max_concepts = int(max_concepts)

    # -----------------------------------------------------------------------
    # Resolve data root
    # -----------------------------------------------------------------------
    local_root = resolve_dreambench_plus_root(data_cfg)

    if local_root is not None:
        root_path = Path(local_root)
        logger.info("DreamBench++: loading from local path: %s", root_path)
        if not root_path.exists():
            msg = f"DreamBench++ local_data_root does not exist: {root_path}"
            if print_tree_on_error:
                parent = root_path.parent
                if parent.exists():
                    logger.error("%s\nParent directory tree:\n%s", msg,
                                 print_directory_tree(str(parent), max_tree_depth, max_tree_files))
            raise FileNotFoundError(msg)
    elif allow_hf_snapshot:
        logger.info(
            "DreamBench++: no local_data_root set; using snapshot_download(repo_id='%s')",
            hf_repo_id,
        )
        try:
            from huggingface_hub import snapshot_download
            hf_token = os.environ.get("HF_TOKEN")
            kwargs: Dict[str, Any] = {"repo_id": hf_repo_id, "repo_type": "dataset"}
            if hf_token:
                kwargs["token"] = hf_token
            local_root = snapshot_download(**kwargs)
            root_path = Path(local_root)
            logger.info("DreamBench++: snapshot downloaded to %s", root_path)
        except Exception as e:
            msg = f"DreamBench++ snapshot_download failed: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e
    else:
        raise RuntimeError(
            "DreamBench++: local_data_root is not set and "
            "allow_hf_snapshot_download=false. "
            "Either set data.local_data_root to a local path, or set "
            "data.allow_hf_snapshot_download=true to auto-download via HuggingFace Hub."
        )

    # -----------------------------------------------------------------------
    # Handle zip-based layout (yuangpeng/dreambench_plus stores data.zip)
    # -----------------------------------------------------------------------
    root_path = _resolve_dreambench_plus_zip(root_path)

    # -----------------------------------------------------------------------
    # Scan directory for concepts
    # -----------------------------------------------------------------------
    concepts = _scan_dreambench_plus_root(
        root_path=root_path,
        max_concepts=max_concepts,
        max_train_images=max_train_images,
        max_eval_prompts=max_eval_prompts,
        allow_sanity_prompt_fallback=allow_sanity_prompt_fallback,
        print_tree_on_error=print_tree_on_error,
        max_tree_depth=max_tree_depth,
        max_tree_files=max_tree_files,
    )

    # -----------------------------------------------------------------------
    # Validate
    # -----------------------------------------------------------------------
    _validate_dreambench_plus(concepts, data_cfg)

    return concepts


def _resolve_dreambench_plus_zip(root_path: Path) -> Path:
    """Handle the yuangpeng/dreambench_plus zip-based layout.

    The official HuggingFace snapshot for yuangpeng/dreambench_plus contains:
      data.zip — archive with structure:
        data/images/{category}/{id}.jpg   — one image per concept
        data/captions/{category}/{id}.txt — first line: concept name; rest: prompts

    If root_path contains a data.zip (and no concept subdirectories or images),
    extract the zip to root_path/extracted/ and build per-concept directories
    with images and metadata.json.

    Returns:
        Path to a directory with per-concept subdirectories (may be newly created).
    """
    import zipfile

    zip_path = root_path / "data.zip"
    if not zip_path.exists():
        return root_path

    # Check if already extracted
    extracted_root = root_path / "extracted"
    concepts_dir = extracted_root / "concepts"
    if concepts_dir.exists() and any(concepts_dir.iterdir()):
        logger.info("DreamBench++: using already-extracted concepts at %s", concepts_dir)
        return concepts_dir

    logger.info("DreamBench++: extracting data.zip to %s ...", extracted_root)
    extracted_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(extracted_root))

    logger.info("DreamBench++: extracted data.zip successfully")

    # Build per-concept directory layout
    # Structure: data/images/{category}/{id}.jpg + data/captions/{category}/{id}.txt
    data_dir = extracted_root / "data"
    images_dir = data_dir / "images"
    captions_dir = data_dir / "captions"

    if not images_dir.exists():
        logger.warning(
            "DreamBench++: data.zip extracted but no data/images/ directory found at %s",
            extracted_root,
        )
        return extracted_root

    concepts_dir.mkdir(parents=True, exist_ok=True)

    # Collect all concepts: each category/{id}.jpg is a concept
    # id links captions/{category}/{id}.txt
    num_created = 0
    for category_dir in sorted(images_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        for img_file in sorted(category_dir.iterdir()):
            if img_file.suffix.lower() not in _DREAMBENCH_IMAGE_EXTENSIONS:
                continue
            concept_idx = img_file.stem  # e.g. "00", "01"
            concept_name = f"{category}_{concept_idx}"

            # Read captions if available
            cap_file = captions_dir / category / f"{concept_idx}.txt"
            prompts: List[str] = []
            concept_label: Optional[str] = None
            if cap_file.exists():
                lines = cap_file.read_text(encoding="utf-8").strip().splitlines()
                if lines:
                    concept_label = lines[0].strip()  # first line = concept name
                    prompts = [l.strip() for l in lines[1:] if l.strip()]

            # Create concept directory
            concept_dir = concepts_dir / concept_name
            concept_dir.mkdir(parents=True, exist_ok=True)

            # Symlink or copy image
            dest_img = concept_dir / img_file.name
            if not dest_img.exists():
                try:
                    dest_img.symlink_to(img_file.resolve())
                except Exception:
                    import shutil
                    shutil.copy2(str(img_file), str(dest_img))

            # Write metadata.json
            meta_path = concept_dir / "metadata.json"
            if not meta_path.exists():
                meta = [{
                    "image": img_file.name,
                    "prompts": prompts,
                    "concept_label": concept_label or concept_name,
                    "category": category,
                }]
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            num_created += 1

    logger.info(
        "DreamBench++: created %d concept directories in %s", num_created, concepts_dir
    )
    return concepts_dir


def _scan_dreambench_plus_root(
    root_path: Path,
    max_concepts: Optional[int],
    max_train_images: Optional[int],
    max_eval_prompts: Optional[int],
    allow_sanity_prompt_fallback: bool,
    print_tree_on_error: bool,
    max_tree_depth: int,
    max_tree_files: int,
) -> List[dict]:
    """Scan the DreamBench++ root directory and extract concept dicts.

    Handles two layouts:
    - Flat: root contains per-concept subdirectories (most common DreamBench++ layout).
    - Flat with top-level metadata: root contains a metadata JSON plus images.

    Returns:
        List of concept dicts.
    """
    concepts: List[dict] = []

    # Check if root itself contains images (single-concept flat layout)
    root_images = [
        p for p in sorted(root_path.iterdir())
        if p.is_file() and p.suffix.lower() in _DREAMBENCH_IMAGE_EXTENSIONS
    ]

    # Look for top-level metadata
    top_meta_file = _find_metadata_file(root_path)
    top_metadata: Dict[str, List[str]] = {}
    if top_meta_file:
        top_metadata = _parse_metadata_file(top_meta_file, root_path)
        logger.info("DreamBench++: found top-level metadata at %s", top_meta_file)

    # Get subdirectories (concept dirs)
    subdirs = sorted([
        d for d in root_path.iterdir()
        if d.is_dir() and d.name not in _EXCLUDE_DIRS
    ])

    if not subdirs and not root_images:
        msg = (
            f"DreamBench++: no concept subdirectories or images found in {root_path}. "
            "Expected one subdirectory per concept, each containing concept images."
        )
        if print_tree_on_error:
            logger.error("%s\nDirectory tree:\n%s", msg,
                         print_directory_tree(str(root_path), max_tree_depth, max_tree_files))
        raise RuntimeError(msg)

    if subdirs:
        # Standard layout: each subdir is a concept
        for concept_dir in subdirs:
            if max_concepts is not None and len(concepts) >= max_concepts:
                break

            concept_id = concept_dir.name
            # Look for concept-level metadata file
            local_meta = _find_metadata_file(concept_dir)
            local_metadata: Dict[str, List[str]] = {}
            if local_meta:
                local_metadata = _parse_metadata_file(local_meta, concept_dir)

            # Merge: local metadata takes priority over top-level
            merged_metadata: Dict[str, List[str]] = {**top_metadata, **local_metadata}

            # Try to find prompts for this concept in top-level metadata
            concept_meta_prompts: List[str] = []
            for key in [concept_id, f"__concept__{concept_id}"]:
                if key in merged_metadata:
                    concept_meta_prompts = merged_metadata[key]
                    break

            # Build metadata with concept-level prompts injected if found
            effective_metadata = dict(merged_metadata)
            if concept_meta_prompts:
                # Inject concept prompts so they're picked up
                effective_metadata[f"__concept__{concept_id}"] = concept_meta_prompts

            concept_dict = _build_concept_from_dir(
                concept_dir=concept_dir,
                concept_id=concept_id,
                category=None,
                metadata=effective_metadata,
                max_train_images=max_train_images,
                max_eval_prompts=max_eval_prompts,
                allow_sanity_prompt_fallback=allow_sanity_prompt_fallback,
            )
            if concept_dict is None:
                if allow_sanity_prompt_fallback:
                    logger.warning(
                        "DreamBench++: concept '%s' has no images or prompts — skipping.",
                        concept_id,
                    )
                else:
                    raise RuntimeError(
                        f"DreamBench++: concept '{concept_id}' has no eval prompts and "
                        "allow_sanity_prompt_fallback=false. "
                        "Please add a metadata file with prompts, or set "
                        "data.allow_sanity_prompt_fallback=true."
                    )
            else:
                concepts.append(concept_dict)
    else:
        # Flat layout: root dir is single concept
        concept_id = root_path.name
        concept_dict = _build_concept_from_dir(
            concept_dir=root_path,
            concept_id=concept_id,
            category=None,
            metadata=top_metadata,
            max_train_images=max_train_images,
            max_eval_prompts=max_eval_prompts,
            allow_sanity_prompt_fallback=allow_sanity_prompt_fallback,
        )
        if concept_dict is not None:
            concepts.append(concept_dict)

    return concepts


def _validate_dreambench_plus(concepts: List[dict], data_cfg) -> None:
    """Validate loaded DreamBench++ concepts.

    Checks:
    - At least one concept.
    - Each concept has >=1 train image.
    - Each concept has >=1 eval prompt (unless evaluation.enabled=False).
    - All image paths exist.
    - Images can be opened by PIL.
    - Prompts are non-empty strings.

    Raises:
        ValueError: On validation failure.
    """
    def _get(key, default=None):
        if hasattr(data_cfg, "get"):
            return data_cfg.get(key, default)
        try:
            from omegaconf import OmegaConf
            val = OmegaConf.select(data_cfg, key, default=default)
            return val if val is not None else default
        except Exception:
            return getattr(data_cfg, key, default)

    eval_enabled = _get("evaluation_enabled", True)

    if not concepts:
        raise ValueError(
            "DreamBench++: loaded 0 concepts. "
            "Check that your data root contains concept subdirectories with images."
        )

    for concept in concepts:
        cid = concept["concept_id"]

        # Check train images
        train_images = concept.get("train_images", [])
        if not train_images:
            raise ValueError(f"DreamBench++ concept '{cid}': no train images found.")

        # Validate image existence and openability
        for img_path in train_images:
            p = Path(img_path)
            if not p.exists():
                raise ValueError(
                    f"DreamBench++ concept '{cid}': image does not exist: {img_path}"
                )
            try:
                with Image.open(p) as img:
                    img.verify()
            except Exception as e:
                raise ValueError(
                    f"DreamBench++ concept '{cid}': cannot open image {img_path}: {e}"
                ) from e

        # Check eval prompts
        eval_prompts = concept.get("eval_prompts", [])
        if eval_enabled and not eval_prompts:
            raise ValueError(
                f"DreamBench++ concept '{cid}': no eval prompts found and "
                "evaluation is enabled. Add a metadata file with prompts, or set "
                "evaluation.enabled=false, or set data.allow_sanity_prompt_fallback=true."
            )

        # Validate prompts
        for prompt in eval_prompts:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"DreamBench++ concept '{cid}': invalid prompt (empty or not string): "
                    f"{repr(prompt)}"
                )


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
