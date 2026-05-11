"""Custom autograd fallback for FBD-LoRA (not used by default).

The primary implementation uses gradient hooks (hooks.py).
This module provides a custom autograd.Function fallback
if hooks prove insufficient for specific PEFT versions.
"""

import torch
import torch.nn.functional as F
from torch.autograd import Function


class FBDLoRAFunction(Function):
    """Custom autograd function implementing FBD-LoRA forward/backward.

    Forward: h = W0 x + B A x   (identical to standard LoRA)
    Backward: modifies gradient of A using pullback metric.

    This is the fallback implementation. Prefer hook-based approach
    in hooks.py for compatibility with PEFT save/load.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        W0: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        scaling: float,
        metric: torch.Tensor,
        lambda_route: float,
        norm_match: bool,
    ) -> torch.Tensor:
        """Standard LoRA forward: h = W0 x + scaling * B @ A @ x."""
        ctx.save_for_backward(x, W0, A, B, metric)
        ctx.scaling = scaling
        ctx.lambda_route = lambda_route
        ctx.norm_match = norm_match

        # h = W0 x + scaling * B A x
        h = F.linear(x, W0) + scaling * F.linear(F.linear(x, A), B)
        return h

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute gradients; route grad_A via pullback metric."""
        x, W0, A, B, metric = ctx.saved_tensors
        scaling = ctx.scaling
        lambda_route = ctx.lambda_route
        norm_match = ctx.norm_match

        # Standard gradients
        # grad_x = grad_output @ (W0 + scaling * B @ A)
        grad_x = grad_output @ W0 + scaling * (grad_output @ B) @ A

        # grad_B = scaling * grad_output^T @ (A @ x^T)^T
        # Standard: grad_B[out, rank] = scaling * delta^T @ (A x^T)^T
        Ax = F.linear(x, A)  # [batch, rank]
        grad_B = scaling * grad_output.t() @ Ax  # [out, rank]

        # Standard grad_A [rank, in]
        # delta = grad_output @ B.T  [batch, rank]
        delta = grad_output @ B  # [batch, rank]
        # grad_A = delta^T @ x    [rank, in]
        grad_A_true = delta.t() @ x  # [rank, in]

        # Pullback routing on grad_A
        from fbd_lora.fbd.routing import route_gradient_pullback
        grad_A_routed, _ = route_gradient_pullback(
            grad_a=grad_A_true,
            metric=metric,
            lambda_route=lambda_route,
            norm_match=norm_match,
            alignment_gate=True,
            gate_type="hard",
            gate_temperature=10.0,
            eps=1e-8,
        )

        return grad_x, None, grad_A_routed, grad_B, None, None, None, None
