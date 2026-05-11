"""Test that FBD-LoRA backward routing modifies lora_A gradient.

Spec Section 3.4:
- B.grad equals standard LoRA when route_b=False
- A.grad differs from LoRA when lambda > 0
- A.grad equals LoRA when lambda = 0
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


def _build_lora_model(in_f, out_f, rank, seed=0, init_b_nonzero=True):
    torch.manual_seed(seed)
    base = _TinyModel(in_f, out_f)
    cfg = LoraConfig(r=rank, lora_alpha=rank, target_modules=["linear"], bias="none", lora_dropout=0.0)
    model = get_peft_model(base, cfg)
    if init_b_nonzero:
        # Override default zero init of lora_B to get nonzero gradients for lora_A
        for name, module in model.named_modules():
            if hasattr(module, "lora_B"):
                for k, sub in module.lora_B.items():
                    torch.manual_seed(seed + 200)
                    nn.init.normal_(sub.weight, std=0.02)
    return model


def _get_lora_a_grad(model):
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "base_layer"):
            for k, sub in module.lora_A.items():
                if sub.weight.grad is not None:
                    return sub.weight.grad.clone()
    return None


def _get_lora_b_grad(model):
    for name, module in model.named_modules():
        if hasattr(module, "lora_B") and hasattr(module, "base_layer"):
            for k, sub in module.lora_B.items():
                if sub.weight.grad is not None:
                    return sub.weight.grad.clone()
    return None


def _copy_all_params(src, dst):
    """Copy ALL parameters from src to dst for fair comparison."""
    src_dict = dict(src.named_parameters())
    dst_dict = dict(dst.named_parameters())
    for k in src_dict:
        if k in dst_dict:
            dst_dict[k].data.copy_(src_dict[k].data)
        else:
            pass  # skip missing (shouldn't happen for same architecture)


class TestBackwardRouting:
    """Test gradient modification behavior."""

    def test_a_grad_differs_when_lambda_positive(self):
        """lora_A grad must differ between LoRA and FBD when lambda > 0."""
        torch.manual_seed(1)
        in_f, out_f, rank = 16, 32, 4

        lora_model = _build_lora_model(in_f, out_f, rank, seed=1)
        fbd_model = _build_lora_model(in_f, out_f, rank, seed=1)
        _copy_all_params(lora_model, fbd_model)

        from fbd_lora.config import FBDConfig
        from fbd_lora.fbd.hooks import apply_fbd_to_peft_model
        fbd_cfg = FBDConfig(enabled=True, lambda_route=0.25, route_a=True, route_b=False, metric_mode="diag")
        fbd_state = apply_fbd_to_peft_model(fbd_model, fbd_cfg)

        x = torch.randn(4, in_f, requires_grad=False)

        # Forward + backward LoRA
        lora_model.train()
        out_lora = lora_model(x)
        loss_lora = out_lora.sum()
        loss_lora.backward()
        grad_a_lora = _get_lora_a_grad(lora_model)
        assert grad_a_lora is not None, "LoRA A grad is None"

        # Forward + backward FBD
        fbd_model.train()
        out_fbd = fbd_model(x)
        loss_fbd = out_fbd.sum()
        loss_fbd.backward()
        grad_a_fbd = _get_lora_a_grad(fbd_model)
        assert grad_a_fbd is not None, "FBD A grad is None"

        # A grad must DIFFER when lambda > 0 and base weight is non-trivial
        assert not torch.allclose(grad_a_lora, grad_a_fbd, atol=1e-7), \
            "FBD A grad should differ from LoRA A grad when lambda=0.25"

        fbd_state.remove_all()

    def test_b_grad_unchanged_when_route_b_false(self):
        """lora_B grad must be identical to standard LoRA when route_b=False."""
        torch.manual_seed(2)
        in_f, out_f, rank = 16, 32, 4

        lora_model = _build_lora_model(in_f, out_f, rank, seed=2)
        fbd_model = _build_lora_model(in_f, out_f, rank, seed=2)
        _copy_all_params(lora_model, fbd_model)

        from fbd_lora.config import FBDConfig
        from fbd_lora.fbd.hooks import apply_fbd_to_peft_model
        fbd_cfg = FBDConfig(enabled=True, lambda_route=0.5, route_a=True, route_b=False, metric_mode="diag")
        fbd_state = apply_fbd_to_peft_model(fbd_model, fbd_cfg)

        x = torch.randn(4, in_f)

        lora_model.train()
        lora_model(x).sum().backward()
        grad_b_lora = _get_lora_b_grad(lora_model)

        fbd_model.train()
        fbd_model(x).sum().backward()
        grad_b_fbd = _get_lora_b_grad(fbd_model)

        assert grad_b_lora is not None and grad_b_fbd is not None
        assert torch.allclose(grad_b_lora, grad_b_fbd, atol=1e-6), \
            f"B grad should be identical when route_b=False. Max diff: {(grad_b_lora - grad_b_fbd).abs().max()}"

        fbd_state.remove_all()

    def test_a_grad_equals_lora_when_lambda_zero(self):
        """With lambda=0, A gradient must be identical to standard LoRA."""
        torch.manual_seed(3)
        in_f, out_f, rank = 16, 32, 4

        lora_model = _build_lora_model(in_f, out_f, rank, seed=3)
        fbd_model = _build_lora_model(in_f, out_f, rank, seed=3)
        _copy_all_params(lora_model, fbd_model)

        from fbd_lora.config import FBDConfig
        from fbd_lora.fbd.hooks import apply_fbd_to_peft_model
        # lambda=0 means no routing -> grad should be unchanged
        fbd_cfg = FBDConfig(enabled=True, lambda_route=0.0, route_a=True)
        fbd_state = apply_fbd_to_peft_model(fbd_model, fbd_cfg)

        x = torch.randn(4, in_f)

        lora_model.train()
        lora_model(x).sum().backward()
        grad_a_lora = _get_lora_a_grad(lora_model)

        fbd_model.train()
        fbd_model(x).sum().backward()
        grad_a_fbd = _get_lora_a_grad(fbd_model)

        assert torch.allclose(grad_a_lora, grad_a_fbd, atol=1e-7), \
            f"With lambda=0, A grad should be identical. Max diff: {(grad_a_lora - grad_a_fbd).abs().max()}"
        fbd_state.remove_all()
