"""Tests for the DreamBench++ official-file loader.

Tests use tmp_path fixture to create synthetic dataset layouts and verify that
load_dreambench_plus correctly parses all supported metadata formats.
"""

import csv
import json
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import pytest
from PIL import Image

from fbd_lora.imagen.data import load_dreambench_plus, resolve_dreambench_plus_root


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiny_image(path: Path) -> None:
    """Create a minimal valid 8x8 JPEG at path."""
    img = Image.new("RGB", (8, 8), color=(128, 64, 32))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))


def _make_data_cfg(
    local_data_root: str,
    allow_hf_snapshot_download: bool = False,
    allow_fallback: bool = False,
    allow_sanity_prompt_fallback: bool = False,
    max_concepts: Optional[int] = None,
    max_train_images_per_concept: Optional[int] = None,
    max_eval_prompts_per_concept: Optional[int] = None,
) -> dict:
    """Build a plain dict config (simulates OmegaConf node)."""
    cfg = {
        "local_data_root": local_data_root,
        "allow_hf_snapshot_download": allow_hf_snapshot_download,
        "allow_fallback": allow_fallback,
        "allow_sanity_prompt_fallback": allow_sanity_prompt_fallback,
        "hf_repo_id": "yuangpeng/dreambench_plus",
        "print_dataset_tree_on_error": False,
        "max_tree_depth": 3,
        "max_tree_files": 50,
        # Disable image validation for unit tests (tiny fake images)
        "evaluation_enabled": True,
    }
    if max_concepts is not None:
        cfg["max_concepts"] = max_concepts
    if max_train_images_per_concept is not None:
        cfg["max_train_images_per_concept"] = max_train_images_per_concept
    if max_eval_prompts_per_concept is not None:
        cfg["max_eval_prompts_per_concept"] = max_eval_prompts_per_concept
    return cfg


# ---------------------------------------------------------------------------
# Monkey-patch _validate_dreambench_plus to skip image PIL validation
# in unit tests (tiny fake images cause verify() to raise for some formats)
# ---------------------------------------------------------------------------

def _noop_validate(concepts, data_cfg):
    """Skip full validation in unit tests."""
    pass


# ---------------------------------------------------------------------------
# Test 1: JSON list format
# ---------------------------------------------------------------------------

class TestJsonListFormat:
    """load_dreambench_plus must parse JSON list metadata."""

    def test_json_list_one_concept_one_image_two_prompts(self, tmp_path, monkeypatch):
        """JSON list format: 1 concept dir, 1 image, 2 prompts in metadata."""
        # Create concept directory with one image
        concept_dir = tmp_path / "my_concept"
        img_path = concept_dir / "image_000.jpg"
        _make_tiny_image(img_path)

        # Create metadata.json in JSON list format
        metadata = [
            {
                "image": "image_000.jpg",
                "prompts": ["a photo of my_concept", "my_concept on a table"],
            }
        ]
        (concept_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

        # Monkeypatch validation to skip PIL verify
        from fbd_lora.imagen import data as data_module
        monkeypatch.setattr(data_module, "_validate_dreambench_plus", _noop_validate)

        cfg = _make_data_cfg(local_data_root=str(tmp_path))
        concepts = load_dreambench_plus(cfg)

        assert len(concepts) == 1
        c = concepts[0]
        assert c["concept_id"] == "my_concept"
        assert len(c["train_images"]) == 1
        assert len(c["eval_prompts"]) == 2
        assert "a photo of my_concept" in c["eval_prompts"]
        assert "my_concept on a table" in c["eval_prompts"]
        assert c["metadata"]["num_train_images_original"] == 1
        assert c["metadata"]["num_eval_prompts_original"] == 2


# ---------------------------------------------------------------------------
# Test 2: JSON dict format with `items` key
# ---------------------------------------------------------------------------

class TestJsonDictItemsFormat:
    """load_dreambench_plus must parse JSON dict with 'items' key."""

    def test_json_dict_items_format(self, tmp_path, monkeypatch):
        """JSON dict with items key: 1 concept, 1 image, 1 prompt."""
        concept_dir = tmp_path / "cat_concept"
        img_path = concept_dir / "cat.jpg"
        _make_tiny_image(img_path)

        metadata = {
            "items": [
                {
                    "image_path": "cat.jpg",
                    "prompt": "a photo of a cat statue",
                }
            ]
        }
        (concept_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

        from fbd_lora.imagen import data as data_module
        monkeypatch.setattr(data_module, "_validate_dreambench_plus", _noop_validate)

        cfg = _make_data_cfg(local_data_root=str(tmp_path))
        concepts = load_dreambench_plus(cfg)

        assert len(concepts) == 1
        c = concepts[0]
        assert c["concept_id"] == "cat_concept"
        assert len(c["train_images"]) >= 1
        assert len(c["eval_prompts"]) >= 1
        assert "a photo of a cat statue" in c["eval_prompts"]


# ---------------------------------------------------------------------------
# Test 3: CSV format
# ---------------------------------------------------------------------------

class TestCsvFormat:
    """load_dreambench_plus must parse CSV metadata."""

    def test_csv_format(self, tmp_path, monkeypatch):
        """CSV format: columns 'image' and 'prompt'."""
        concept_dir = tmp_path / "teapot"
        img_path = concept_dir / "teapot_001.jpg"
        _make_tiny_image(img_path)

        # Write CSV metadata
        csv_path = concept_dir / "metadata.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "prompt"])
            writer.writeheader()
            writer.writerow({"image": "teapot_001.jpg", "prompt": "a red teapot"})

        from fbd_lora.imagen import data as data_module
        monkeypatch.setattr(data_module, "_validate_dreambench_plus", _noop_validate)

        cfg = _make_data_cfg(local_data_root=str(tmp_path))
        concepts = load_dreambench_plus(cfg)

        assert len(concepts) == 1
        c = concepts[0]
        assert c["concept_id"] == "teapot"
        assert len(c["train_images"]) == 1
        assert "a red teapot" in c["eval_prompts"]


# ---------------------------------------------------------------------------
# Test 4: No prompt file + allow_sanity_prompt_fallback=False → error
# ---------------------------------------------------------------------------

class TestNoPromptFallbackDisabled:
    """When no prompt metadata exists and fallback is disabled, raise RuntimeError."""

    def test_no_prompt_raises_error(self, tmp_path, monkeypatch):
        """No metadata file + allow_sanity_prompt_fallback=false → RuntimeError."""
        concept_dir = tmp_path / "mystery_concept"
        img_path = concept_dir / "image_000.jpg"
        _make_tiny_image(img_path)
        # No metadata file — no prompts

        from fbd_lora.imagen import data as data_module
        monkeypatch.setattr(data_module, "_validate_dreambench_plus", _noop_validate)

        cfg = _make_data_cfg(
            local_data_root=str(tmp_path),
            allow_sanity_prompt_fallback=False,
        )
        with pytest.raises(RuntimeError, match="allow_sanity_prompt_fallback"):
            load_dreambench_plus(cfg)


# ---------------------------------------------------------------------------
# Test 5: No prompt file + allow_sanity_prompt_fallback=True → loads with flag
# ---------------------------------------------------------------------------

class TestNoPromptFallbackEnabled:
    """When no prompt metadata and fallback enabled, load with fallback prompts."""

    def test_fallback_prompt_used(self, tmp_path, monkeypatch):
        """No metadata file + allow_sanity_prompt_fallback=true → loads with fallback."""
        concept_dir = tmp_path / "mystery_subject"
        img_path = concept_dir / "image_000.jpg"
        _make_tiny_image(img_path)
        # No metadata file

        from fbd_lora.imagen import data as data_module
        monkeypatch.setattr(data_module, "_validate_dreambench_plus", _noop_validate)

        cfg = _make_data_cfg(
            local_data_root=str(tmp_path),
            allow_sanity_prompt_fallback=True,
        )
        concepts = load_dreambench_plus(cfg)

        assert len(concepts) == 1
        c = concepts[0]
        assert c["concept_id"] == "mystery_subject"
        assert len(c["eval_prompts"]) >= 1
        assert c["metadata"]["used_sanity_prompt_fallback"] is True
        # Fallback prompt should contain the concept name
        assert "mystery" in c["eval_prompts"][0] or "photo" in c["eval_prompts"][0]


# ---------------------------------------------------------------------------
# Test: resolve_dreambench_plus_root
# ---------------------------------------------------------------------------

class TestResolveRoot:
    """resolve_dreambench_plus_root expands env vars and ~ correctly."""

    def test_none_when_empty(self):
        cfg = {"local_data_root": ""}
        assert resolve_dreambench_plus_root(cfg) is None

    def test_none_when_not_set(self):
        cfg = {}
        assert resolve_dreambench_plus_root(cfg) is None

    def test_expands_tilde(self, tmp_path):
        cfg = {"local_data_root": "~/some/path"}
        result = resolve_dreambench_plus_root(cfg)
        assert result is not None
        assert not result.startswith("~")

    def test_expands_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MY_TEST_VAR", str(tmp_path))
        cfg = {"local_data_root": "${MY_TEST_VAR}"}
        result = resolve_dreambench_plus_root(cfg)
        assert result == str(tmp_path)

    def test_returns_none_for_unresolved_var(self):
        # If var is not set, os.path.expandvars leaves it as-is
        # We can't guarantee the env is clean, so just check it returns a string or None
        import os
        var_name = "FBD_UNSET_VAR_XYZ_123"
        if var_name in os.environ:
            return  # skip if somehow set
        cfg = {"local_data_root": f"${{{var_name}}}"}
        result = resolve_dreambench_plus_root(cfg)
        # With expandvars, unset vars remain as-is; our function should return None
        assert result is None


# ---------------------------------------------------------------------------
# Test: max_concepts limit
# ---------------------------------------------------------------------------

class TestMaxConcepts:
    """max_concepts limits the number of concepts loaded."""

    def test_max_concepts_one(self, tmp_path, monkeypatch):
        """With max_concepts=1, only 1 concept is returned."""
        for name, prompt in [("dog", "a dog"), ("cat", "a cat"), ("bird", "a bird")]:
            concept_dir = tmp_path / name
            img_path = concept_dir / "image.jpg"
            _make_tiny_image(img_path)
            meta = [{"image": "image.jpg", "prompts": [prompt]}]
            (concept_dir / "metadata.json").write_text(json.dumps(meta))

        from fbd_lora.imagen import data as data_module
        monkeypatch.setattr(data_module, "_validate_dreambench_plus", _noop_validate)

        cfg = _make_data_cfg(local_data_root=str(tmp_path), max_concepts=1)
        concepts = load_dreambench_plus(cfg)
        assert len(concepts) == 1


# ---------------------------------------------------------------------------
# Test: max_eval_prompts_per_concept
# ---------------------------------------------------------------------------

class TestMaxEvalPrompts:
    """max_eval_prompts_per_concept limits the number of eval prompts."""

    def test_max_prompts_two(self, tmp_path, monkeypatch):
        """With max_eval_prompts_per_concept=2, only 2 prompts are returned."""
        concept_dir = tmp_path / "multi_prompt"
        img_path = concept_dir / "image.jpg"
        _make_tiny_image(img_path)
        prompts = [f"prompt {i}" for i in range(10)]
        meta = [{"image": "image.jpg", "prompts": prompts}]
        (concept_dir / "metadata.json").write_text(json.dumps(meta))

        from fbd_lora.imagen import data as data_module
        monkeypatch.setattr(data_module, "_validate_dreambench_plus", _noop_validate)

        cfg = _make_data_cfg(
            local_data_root=str(tmp_path),
            max_eval_prompts_per_concept=2,
        )
        concepts = load_dreambench_plus(cfg)
        assert len(concepts) == 1
        assert len(concepts[0]["eval_prompts"]) == 2


# ---------------------------------------------------------------------------
# Test: txt prompt file fallback
# ---------------------------------------------------------------------------

class TestTxtPromptFallback:
    """eval_prompts.txt in concept dir is read as prompt fallback."""

    def test_txt_prompt_file(self, tmp_path, monkeypatch):
        """eval_prompts.txt file is used when no metadata JSON/CSV exists."""
        concept_dir = tmp_path / "backpack"
        img_path = concept_dir / "image.jpg"
        _make_tiny_image(img_path)
        (concept_dir / "eval_prompts.txt").write_text(
            "a blue backpack\na backpack on a table\n", encoding="utf-8"
        )

        from fbd_lora.imagen import data as data_module
        monkeypatch.setattr(data_module, "_validate_dreambench_plus", _noop_validate)

        cfg = _make_data_cfg(local_data_root=str(tmp_path))
        concepts = load_dreambench_plus(cfg)

        assert len(concepts) == 1
        c = concepts[0]
        assert "a blue backpack" in c["eval_prompts"]
        assert "a backpack on a table" in c["eval_prompts"]
        assert c["metadata"]["used_sanity_prompt_fallback"] is False
