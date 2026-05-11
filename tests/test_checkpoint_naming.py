"""Test run naming collision resistance and sanitization."""

import pytest
from fbd_lora.naming import make_run_name, sanitize, make_target_modules_token


class TestCheckpointNaming:
    """Run name must be collision-resistant and sanitized."""

    def test_name_length_within_limit(self):
        """Run name must not exceed 180 characters."""
        name = make_run_name(
            seed=42, modality="nlg", task="metamath",
            backbone="mistralai/Mistral-7B-v0.1",
            adapter="fbd", rank=16, alpha=16,
            batch_size=4, grad_accum=8, lr=2e-4,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            routing="pullback_metric_mixed", lambda_route=0.25,
        )
        assert len(name) <= 180, f"Name too long: {len(name)} chars: {name}"

    def test_name_contains_required_fields(self):
        """Name must contain seed, rank, routing info."""
        name = make_run_name(
            seed=42, modality="nlg", task="metamath",
            backbone="mistralai/Mistral-7B-v0.1",
            adapter="fbd", rank=16, alpha=16,
            batch_size=4, grad_accum=8, lr=2e-4,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            routing="pullback_metric_mixed", lambda_route=0.25,
        )
        assert "seed42" in name
        assert "r16" in name
        assert "nlg" in name
        assert "metamath" in name

    def test_different_seeds_produce_different_hashes(self):
        """Different seeds should produce different run hashes."""
        common = dict(
            modality="nlg", task="metamath",
            backbone="mistralai/Mistral-7B-v0.1",
            adapter="fbd", rank=16, alpha=16,
            batch_size=4, grad_accum=8, lr=2e-4,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            routing="pullback_metric_mixed", lambda_route=0.25,
        )
        name1 = make_run_name(seed=42, **common)
        name2 = make_run_name(seed=43, **common)
        # Hash portion should differ (last 8 chars before end)
        assert name1 != name2

    def test_sanitize_removes_special_chars(self):
        """Sanitize must remove/replace unsafe characters."""
        raw = "mistralai/Mistral-7B-v0.1 (test):foo"
        s = sanitize(raw)
        assert "/" not in s
        assert " " not in s
        assert ":" not in s
        assert "(" not in s
        assert ")" not in s

    def test_sanitize_lowercase(self):
        """Sanitize must lowercase everything."""
        assert sanitize("AbCdEf") == "abcdef"

    def test_target_modules_token(self):
        """Target modules token should produce known compact forms."""
        tok = make_target_modules_token(["q_proj", "k_proj", "v_proj", "o_proj"])
        assert tok == "qkvo"

        tok2 = make_target_modules_token(["q_proj", "v_proj"])
        assert tok2 == "qv"

        tok3 = make_target_modules_token([])
        assert tok3 == "none"

    def test_collision_resistance_via_hash(self):
        """Two runs with same params but different backbone should have different hashes."""
        common = dict(
            seed=42, modality="nlg", task="metamath",
            adapter="fbd", rank=16, alpha=16,
            batch_size=4, grad_accum=8, lr=2e-4,
            target_modules=["q_proj", "v_proj"],
            routing="pullback_metric_mixed", lambda_route=0.25,
        )
        n1 = make_run_name(backbone="mistralai/Mistral-7B-v0.1", **common)
        n2 = make_run_name(backbone="meta-llama/Llama-2-7b-hf", **common)
        # Extract the hash (last component)
        h1 = n1.split("_")[-1]
        h2 = n2.split("_")[-1]
        assert h1 != h2, "Different backbone should give different hash"

    def test_lambda_in_name(self):
        """Lambda value must appear in name."""
        name = make_run_name(
            seed=42, modality="nlg", task="test",
            backbone="gpt2", adapter="fbd",
            rank=4, alpha=4, batch_size=2, grad_accum=1, lr=1e-4,
            target_modules=[], routing="pullback_metric", lambda_route=0.5,
        )
        assert "lam" in name
