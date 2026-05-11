"""Test PSD descent alignment property of pullback routing.

Spec Section 2.4: PSD pullback routing is descent-aligned.
    <g_A, g_A^R> = tr(g_A^T g_A P0) >= 0
"""

import torch
import pytest
from fbd_lora.fbd.routing import compute_pullback_metric, route_gradient_pullback


class TestPSDRouting:
    """Pullback metric routing must preserve descent alignment."""

    def test_descent_alignment_full_metric(self):
        """Full PSD metric routing must have positive inner product."""
        torch.manual_seed(10)
        rank, d_in, d_out = 4, 16, 32

        W = torch.randn(d_out, d_in)
        P = compute_pullback_metric(W, epsilon=1e-4, normalize="rms", mode="full")
        assert P.shape == (d_in, d_in), f"Metric shape: {P.shape}"

        grad = torch.randn(rank, d_in)
        # g_A^R = g_A @ P0
        grad_routed = grad @ P

        inner = (grad * grad_routed).sum()
        assert inner.item() > 0, f"Inner product should be positive for PSD metric, got {inner.item()}"

    def test_descent_alignment_diag_metric(self):
        """Diagonal metric routing (elementwise) must have non-negative inner product."""
        torch.manual_seed(11)
        rank, d_in, d_out = 4, 16, 32

        W = torch.randn(d_out, d_in)
        p_diag = compute_pullback_metric(W, epsilon=1e-4, normalize="rms", mode="diag")
        assert p_diag.shape == (d_in,), f"Diag metric shape: {p_diag.shape}"
        assert (p_diag > 0).all(), "Diag metric entries must be positive (epsilon > 0)"

        grad = torch.randn(rank, d_in)
        # Elementwise: g_A^R[i,j] = g_A[i,j] * p0[j]
        grad_routed = grad * p_diag.unsqueeze(0)

        inner = (grad * grad_routed).sum()
        assert inner.item() >= 0, f"Inner product should be >= 0 for diag metric, got {inner.item()}"

    def test_mixed_routing_descent_aligned(self):
        """Mixed routing tilde_g = (1-lambda) g + lambda g^R must be descent-aligned."""
        torch.manual_seed(12)
        rank, d_in, d_out = 4, 16, 32

        W = torch.randn(d_out, d_in)
        metric = compute_pullback_metric(W, epsilon=1e-4, normalize="rms", mode="diag")
        grad = torch.randn(rank, d_in)

        for lambda_r in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
            routed, stats = route_gradient_pullback(
                grad_a=grad,
                metric=metric,
                lambda_route=lambda_r,
                norm_match=False,
                alignment_gate=False,  # no gating to test pure math
                gate_type="hard",
                gate_temperature=10.0,
                eps=1e-8,
            )
            inner = (grad * routed).sum()
            assert inner.item() >= -1e-5, \
                f"lambda={lambda_r}: mixed gradient not descent-aligned, inner={inner.item()}"

    def test_norm_matching_preserves_scale(self):
        """With norm_match=True, routed grad should have ~same norm as true grad."""
        torch.manual_seed(13)
        rank, d_in, d_out = 8, 32, 64

        W = torch.randn(d_out, d_in) * 10  # large weights to create scale difference
        metric = compute_pullback_metric(W, epsilon=1e-4, normalize="rms", mode="diag")
        grad = torch.randn(rank, d_in)

        # Without norm matching, norm could change significantly
        routed_no_match, _ = route_gradient_pullback(
            grad, metric, lambda_route=1.0, norm_match=False,
            alignment_gate=False, gate_type="hard", gate_temperature=10.0, eps=1e-8,
        )

        # With norm matching, norm should be preserved (approximately)
        routed_match, _ = route_gradient_pullback(
            grad, metric, lambda_route=1.0, norm_match=True,
            alignment_gate=False, gate_type="hard", gate_temperature=10.0, eps=1e-8,
        )

        true_norm = grad.norm(p="fro")
        match_norm = routed_match.norm(p="fro")

        # Norm should be close to original (within 2x after mixing with original weight)
        # lambda=1.0 fully routed with norm matching -> exactly true_norm
        assert abs(match_norm.item() - true_norm.item()) / (true_norm.item() + 1e-8) < 0.02, \
            f"Norm mismatch after norm_match: got {match_norm.item():.4f}, expected {true_norm.item():.4f}"

    def test_psd_guarantee_epsilon(self):
        """Metric with epsilon > 0 must have all positive eigenvalues (full mode)."""
        torch.manual_seed(14)
        d_in, d_out = 8, 4  # more out than in -> W^T W may be rank-deficient without eps

        W = torch.randn(d_out, d_in) * 0.01  # small weights
        P = compute_pullback_metric(W, epsilon=1e-4, normalize="rms", mode="full")

        eigvals = torch.linalg.eigvalsh(P)
        assert (eigvals > 0).all(), f"All eigenvalues must be positive. Min: {eigvals.min().item()}"
