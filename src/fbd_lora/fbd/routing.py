"""Pullback metric computation and gradient routing for FBD-LoRA.

Mathematical specification:
    Forward: h = W0 x + B A x  (standard LoRA, unchanged)
    Backward A: g_A = B^T delta x^T  (standard LoRA gradient)
    Routed:  g_A^R = g_A @ P0  where P0 = normalize(W0^T W0) + eps I
    Mixed:   tilde_g_A = (1-lambda) g_A + lambda g_A^R

Modes:
    "diag":    P0 = diag(W0^T W0), elementwise multiply  -> O(d_in)
    "full":    P0 = W0^T W0 + eps I                      -> O(d_in^2)
    "lowrank": P0 approx V_k Sigma_k^2 V_k^T             -> O(k d_in)
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def _normalize_matrix(M: torch.Tensor, mode: str) -> torch.Tensor:
    """Normalize matrix M by given mode.

    Args:
        M: Square matrix [d, d] or vector [d].
        mode: "none", "fro", "spectral", "rms".

    Returns:
        Normalized M (same shape).
    """
    if mode == "none":
        return M
    elif mode == "fro":
        norm = M.norm(p="fro") if M.dim() == 2 else M.norm()
        return M / (norm + 1e-12)
    elif mode == "spectral":
        # Approximate: divide by largest singular value (expensive for large M)
        if M.dim() == 2:
            s = torch.linalg.svdvals(M)[0]
            return M / (s + 1e-12)
        else:
            return M / (M.abs().max() + 1e-12)
    elif mode == "rms":
        rms = M.pow(2).mean().sqrt()
        return M / (rms + 1e-12)
    else:
        raise ValueError(f"Unknown normalize mode: {mode}")


def compute_pullback_metric(
    weight: torch.Tensor,
    epsilon: float = 1e-4,
    normalize: str = "rms",
    mode: str = "diag",
    rank_approx: Optional[int] = None,
) -> torch.Tensor:
    """Compute the pretrained-weight-induced pullback metric P0.

    Args:
        weight: Frozen pretrained weight W0, shape [out_features, in_features].
        epsilon: Regularization for PSD guarantee (added after normalization).
        normalize: Normalization mode ("none", "fro", "spectral", "rms").
        mode: Metric approximation mode.
            "diag"    -> returns vector [in_features]
            "full"    -> returns matrix [in_features, in_features]
            "lowrank" -> returns matrix [in_features, in_features]
        rank_approx: Number of SVD components for lowrank mode.

    Returns:
        Metric tensor:
            mode="diag"    -> shape [in_features]
            mode="full"    -> shape [in_features, in_features]
            mode="lowrank" -> shape [in_features, in_features]

    Mathematical guarantee:
        For "full" and "lowrank" modes with epsilon > 0, the returned
        matrix is strictly positive definite, ensuring descent alignment:
            <g_A, g_A @ P0> = tr(g_A^T g_A P0) >= 0
    """
    # weight: [out, in] = [d_out, d_in]
    W = weight.detach().float()  # compute in fp32 for numerical stability

    if mode == "diag":
        # diag(W^T W) = sum of squares along out_features dim
        # shape: [in_features]
        p_diag = (W ** 2).sum(dim=0)          # [d_in]
        p_diag = _normalize_matrix(p_diag, normalize)
        p_diag = p_diag + epsilon
        return p_diag

    elif mode == "full":
        # P0 = W^T W / normalize + eps I
        # [in, in] = [d_in, d_in]
        P = W.t() @ W                          # [d_in, d_in]
        P = _normalize_matrix(P, normalize)
        d_in = P.shape[0]
        P = P + epsilon * torch.eye(d_in, dtype=P.dtype, device=P.device)
        return P

    elif mode == "lowrank":
        # Approximate W^T W via truncated SVD: V_k Sigma_k^2 V_k^T
        k = rank_approx if rank_approx is not None else min(64, W.shape[0], W.shape[1])
        k = min(k, min(W.shape) - 1)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        # W = U S Vh  =>  W^T W = Vh^T S^2 Vh  (approx with top-k)
        Vk = Vh[:k].t()    # [d_in, k]
        Sk2 = S[:k] ** 2   # [k]
        # P0 = Vk diag(Sk2) Vk^T
        P = Vk * Sk2.unsqueeze(0)   # [d_in, k]
        P = P @ Vk.t()              # [d_in, d_in]
        P = _normalize_matrix(P, normalize)
        d_in = P.shape[0]
        P = P + epsilon * torch.eye(d_in, dtype=P.dtype, device=P.device)
        return P

    else:
        raise ValueError(f"Unknown metric mode: {mode}. Choose 'diag', 'full', or 'lowrank'.")


# ---------------------------------------------------------------------------
# Gradient routing
# ---------------------------------------------------------------------------


def route_gradient_pullback(
    grad_a: torch.Tensor,
    metric: torch.Tensor,
    lambda_route: float,
    norm_match: bool,
    alignment_gate: bool,
    gate_type: str = "hard",
    gate_temperature: float = 10.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, dict]:
    """Apply pullback metric routing to the lora_A gradient.

    Implements Sections 2.3-2.6 of the FBD-LoRA spec.

    Args:
        grad_a: True LoRA gradient for A, shape [rank, in_features].
        metric: Pullback metric P0.
            If 1D [in_features]: diagonal mode, elementwise multiply.
            If 2D [in_features, in_features]: full mode, matrix multiply.
        lambda_route: Routing strength in [0, 1].
            lambda=0 -> unchanged, lambda=1 -> fully routed.
        norm_match: If True, rescale routed grad to match true grad Frobenius norm.
        alignment_gate: If True, apply descent-direction gate on lambda.
        gate_type: "hard" (binary gate) or "sigmoid" (soft sigmoid gate).
        gate_temperature: Temperature for sigmoid gate (higher = sharper).
        eps: Small value for numerical stability.

    Returns:
        Tuple of (routed_grad, stats_dict) where:
            routed_grad: Modified gradient, same shape as grad_a [rank, in_features].
            stats_dict: Dict with keys:
                cos_true_routed: cosine similarity between g_A and tilde_g_A.
                norm_ratio: ||tilde_g_A|| / ||g_A||.
                alignment_gate_active: bool, whether gate passed.
                lambda_eff: effective lambda used.
    """
    if lambda_route == 0.0:
        stats = {
            "cos_true_routed": 1.0,
            "norm_ratio": 1.0,
            "alignment_gate_active": True,
            "lambda_eff": 0.0,
        }
        return grad_a, stats

    # Compute routed gradient: g_A^R = g_A @ P0
    # grad_a: [rank, d_in]
    # metric: [d_in] (diag) or [d_in, d_in] (full)
    grad_float = grad_a.float()   # compute in fp32 for stability

    if metric.dim() == 1:
        # Diagonal mode: elementwise multiply along d_in
        # g_A^R[i, j] = g_A[i, j] * p0[j]
        metric_float = metric.float()
        grad_routed = grad_float * metric_float.unsqueeze(0)   # [rank, d_in]
    elif metric.dim() == 2:
        # Full mode: matrix multiply
        # g_A^R = g_A @ P0   [rank, d_in] @ [d_in, d_in] = [rank, d_in]
        metric_float = metric.float()
        grad_routed = grad_float @ metric_float   # [rank, d_in]
    else:
        raise ValueError(f"Unexpected metric shape: {metric.shape}")

    # Norm matching (Section 2.6)
    if norm_match:
        true_norm = grad_float.norm(p="fro")
        routed_norm = grad_routed.norm(p="fro")
        if routed_norm > eps:
            grad_routed = grad_routed * (true_norm / (routed_norm + eps))

    # Alignment gating (Section 2.5)
    # Compute inner product <g_A, g_A^R>
    inner = (grad_float * grad_routed).sum()
    cos_val = inner / (grad_float.norm() * grad_routed.norm() + eps)

    alignment_gate_active = True
    lambda_eff = lambda_route

    if alignment_gate:
        if gate_type == "hard":
            # Binary gate: lambda_eff = lambda if <g, g^R> > 0 else 0
            if inner.item() <= 0:
                lambda_eff = 0.0
                alignment_gate_active = False
        elif gate_type == "sigmoid":
            # Soft gate: lambda_eff = lambda * sigmoid(tau * cos(g, g^R))
            gate = torch.sigmoid(torch.tensor(gate_temperature * cos_val.item()))
            lambda_eff = lambda_route * gate.item()
            alignment_gate_active = cos_val.item() > 0
        else:
            raise ValueError(f"Unknown gate_type: {gate_type}")

    # Mixed update: tilde_g_A = (1 - lambda_eff) g_A + lambda_eff g_A^R
    # Section 2.5: descent-safe mixture
    grad_mixed = (1.0 - lambda_eff) * grad_float + lambda_eff * grad_routed

    # Cast back to original dtype
    grad_mixed = grad_mixed.to(grad_a.dtype)

    # Compute stats for logging
    mixed_norm = grad_mixed.float().norm(p="fro")
    true_norm_final = grad_float.norm(p="fro")
    cos_mixed = (grad_float * grad_mixed.float()).sum() / (true_norm_final * mixed_norm + eps)

    stats = {
        "cos_true_routed": cos_mixed.item(),
        "norm_ratio": (mixed_norm / (true_norm_final + eps)).item(),
        "alignment_gate_active": alignment_gate_active,
        "lambda_eff": lambda_eff,
    }

    return grad_mixed, stats
