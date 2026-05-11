"""Test that FBD adapter can be merged like standard LoRA.

Spec Section 3.5: After training, the FBD adapter must be deployable as standard LoRA.
W_deploy = W0 + scaling * B @ A
"""

import torch
import torch.nn as nn
import pytest
from peft import LoraConfig, get_peft_model


class _TinyModel(nn.Module):
    def __init__(self, in_f=16, out_f=32):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f, bias=False)

    def forward(self, x):
        return self.linear(x)


class TestMergeEquivalence:
    """FBD-LoRA adapter must be mergeable without changing inference output."""

    def test_merge_equivalence_after_training_steps(self):
        """Train FBD for a few steps, then merge. Merged output should match unmerged."""
        torch.manual_seed(99)
        in_f, out_f, rank = 16, 32, 4

        base = _TinyModel(in_f, out_f)
        cfg = LoraConfig(r=rank, lora_alpha=rank, target_modules=["linear"], bias="none", lora_dropout=0.0)
        fbd_model = get_peft_model(base, cfg)

        from fbd_lora.config import FBDConfig
        from fbd_lora.fbd.hooks import apply_fbd_to_peft_model, remove_fbd_hooks

        fbd_cfg = FBDConfig(enabled=True, lambda_route=0.25, metric_mode="diag")
        fbd_state = apply_fbd_to_peft_model(fbd_model, fbd_cfg)

        # Simulate a few training steps
        optimizer = torch.optim.SGD([p for p in fbd_model.parameters() if p.requires_grad], lr=1e-3)
        fbd_model.train()
        for _ in range(3):
            x = torch.randn(4, in_f)
            loss = fbd_model(x).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Get unmerged output
        fbd_model.eval()
        x_test = torch.randn(4, in_f)
        with torch.no_grad():
            y_unmerged = fbd_model(x_test)

        # Remove hooks before merge to avoid interference
        remove_fbd_hooks(fbd_state)

        # Merge adapter
        merged_model = fbd_model.merge_and_unload()
        merged_model.eval()

        with torch.no_grad():
            y_merged = merged_model(x_test)

        assert y_unmerged.shape == y_merged.shape
        max_diff = (y_unmerged - y_merged).abs().max().item()
        assert max_diff < 1e-4, \
            f"Merge equivalence failed! Max diff: {max_diff} (expected < 1e-4)"

    def test_merge_zero_init(self):
        """At initialization (B=0), merged model output must equal base model output."""
        torch.manual_seed(100)
        in_f, out_f, rank = 16, 32, 4

        base = _TinyModel(in_f, out_f)
        base_weight = base.linear.weight.data.clone()

        cfg = LoraConfig(r=rank, lora_alpha=rank, target_modules=["linear"], bias="none", lora_dropout=0.0)
        fbd_model = get_peft_model(base, cfg)

        # At init, lora_B = 0, so LoRA contribution = 0
        from fbd_lora.fbd.hooks import apply_fbd_to_peft_model, remove_fbd_hooks
        from fbd_lora.config import FBDConfig
        fbd_state = apply_fbd_to_peft_model(fbd_model, FBDConfig(enabled=True, lambda_route=0.25))

        x_test = torch.randn(4, in_f)
        fbd_model.eval()
        with torch.no_grad():
            y_fbd = fbd_model(x_test)

        remove_fbd_hooks(fbd_state)
        merged = fbd_model.merge_and_unload()
        merged.eval()
        with torch.no_grad():
            y_merged = merged(x_test)

        # Both should give same result (B=0 at init)
        assert torch.allclose(y_fbd, y_merged, atol=1e-5), \
            f"Zero-init merge failed. Max diff: {(y_fbd - y_merged).abs().max()}"
