"""Test that FBD-LoRA forward pass is identical to standard PEFT LoRA.

Spec Section 3.3: For every FBD-LoRA layer, the forward output must match
standard LoRA with the same A, B, scaling, dropout, and target module.

KEY: Both models are built with the same seed so all weights are identical.
FBD hooks affect only the backward pass, not the forward.
"""

import torch
import torch.nn as nn
import pytest
from peft import LoraConfig, get_peft_model


class _TinyModel(nn.Module):
    def __init__(self, in_f: int = 16, out_f: int = 32):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f, bias=False)

    def forward(self, x):
        return self.linear(x)


def _make_identical_pair(in_f: int, out_f: int, rank: int, seed: int = 0):
    """Create two PEFT LoRA models with identical weights.

    Both are built from the same seed, guaranteeing identical:
    - base layer weights
    - lora_A (random init)
    - lora_B (zero init by default)
    """
    torch.manual_seed(seed)
    m1 = _TinyModel(in_f, out_f)
    cfg = LoraConfig(r=rank, lora_alpha=rank, target_modules=["linear"], bias="none", lora_dropout=0.0)
    lora_model = get_peft_model(m1, cfg)

    # Reset seed to same value so m2 gets identical weights
    torch.manual_seed(seed)
    m2 = _TinyModel(in_f, out_f)
    fbd_model = get_peft_model(m2, cfg)

    # Verify they match before adding FBD hooks
    lp = dict(lora_model.named_parameters())
    fp = dict(fbd_model.named_parameters())
    for k in lp:
        if not torch.allclose(lp[k], fp[k], atol=1e-9):
            # Fall back to explicit copy if seeding isn't deterministic
            fp[k].data.copy_(lp[k].data)

    return lora_model, fbd_model


class TestForwardEquivalence:
    """Forward pass must be identical for FBD and standard LoRA."""

    def test_forward_identical_fp32(self):
        """FBD forward == LoRA forward with same weights (fp32)."""
        in_f, out_f, rank = 16, 32, 4
        lora_model, fbd_model = _make_identical_pair(in_f, out_f, rank, seed=0)

        # Apply FBD hooks to fbd_model (hooks only affect backward, not forward)
        from fbd_lora.config import FBDConfig
        from fbd_lora.fbd.hooks import apply_fbd_to_peft_model
        fbd_cfg = FBDConfig(enabled=True, lambda_route=0.25, metric_mode="diag")
        fbd_state = apply_fbd_to_peft_model(fbd_model, fbd_cfg)

        # Verify same weights before forward
        lora_params = dict(lora_model.named_parameters())
        fbd_params = dict(fbd_model.named_parameters())
        for key in lora_params:
            assert torch.allclose(lora_params[key], fbd_params[key], atol=1e-9), \
                f"Weight mismatch before forward: {key}"

        # Forward pass (no gradient, eval mode)
        lora_model.eval()
        fbd_model.eval()
        torch.manual_seed(42)
        x = torch.randn(4, in_f)

        with torch.no_grad():
            y_lora = lora_model(x)
            y_fbd = fbd_model(x)

        assert y_lora.shape == y_fbd.shape, f"Shape mismatch: {y_lora.shape} vs {y_fbd.shape}"
        assert torch.allclose(y_lora, y_fbd, atol=1e-6), \
            f"Forward outputs differ! Max diff: {(y_lora - y_fbd).abs().max().item()}"

        fbd_state.remove_all()

    def test_forward_identical_zero_lambda(self):
        """With lambda=0, FBD == LoRA forward exactly."""
        in_f, out_f, rank = 32, 64, 8
        lora_model, fbd_model = _make_identical_pair(in_f, out_f, rank, seed=42)

        from fbd_lora.config import FBDConfig
        from fbd_lora.fbd.hooks import apply_fbd_to_peft_model
        fbd_cfg = FBDConfig(enabled=True, lambda_route=0.0)
        fbd_state = apply_fbd_to_peft_model(fbd_model, fbd_cfg)

        lora_model.eval()
        fbd_model.eval()
        torch.manual_seed(1)
        x = torch.randn(8, in_f)
        with torch.no_grad():
            y_lora = lora_model(x)
            y_fbd = fbd_model(x)

        assert torch.allclose(y_lora, y_fbd, atol=1e-7), \
            f"Forward outputs differ with lambda=0! Max diff: {(y_lora - y_fbd).abs().max().item()}"

        fbd_state.remove_all()

    def test_forward_identical_full_metric(self):
        """FBD with full metric also has identical forward (hooks don't touch forward)."""
        in_f, out_f, rank = 16, 32, 4
        lora_model, fbd_model = _make_identical_pair(in_f, out_f, rank, seed=7)

        from fbd_lora.config import FBDConfig
        from fbd_lora.fbd.hooks import apply_fbd_to_peft_model
        fbd_cfg = FBDConfig(enabled=True, lambda_route=0.5, metric_mode="full")
        fbd_state = apply_fbd_to_peft_model(fbd_model, fbd_cfg)

        lora_model.eval()
        fbd_model.eval()
        torch.manual_seed(2)
        x = torch.randn(4, in_f)
        with torch.no_grad():
            y_lora = lora_model(x)
            y_fbd = fbd_model(x)

        assert torch.allclose(y_lora, y_fbd, atol=1e-6), \
            f"Full metric: forward mismatch! Max diff: {(y_lora - y_fbd).abs().max().item()}"
        fbd_state.remove_all()
