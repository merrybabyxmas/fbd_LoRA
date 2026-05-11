"""Gradient statistics aggregation for FBD-LoRA logging."""

import logging
from typing import Dict, List

import torch

logger = logging.getLogger(__name__)


def aggregate_gradient_stats(per_layer_stats: List[dict]) -> dict:
    """Aggregate per-layer gradient stats across all FBD hooks.

    Args:
        per_layer_stats: List of stat dicts from route_gradient_pullback.
            Each dict has: cos_true_routed, norm_ratio, alignment_gate_active, lambda_eff.

    Returns:
        Aggregated dict with mean values across layers:
            train/routed_true_grad_cosine: mean cosine between true and routed grad.
            train/descent_alignment_ratio: fraction of layers with cosine > 0.
            train/routed_grad_norm_ratio: mean norm(routed) / norm(true).
    """
    if not per_layer_stats:
        return {}

    cosines = [s["cos_true_routed"] for s in per_layer_stats if "cos_true_routed" in s]
    norm_ratios = [s["norm_ratio"] for s in per_layer_stats if "norm_ratio" in s]
    gate_actives = [s["alignment_gate_active"] for s in per_layer_stats if "alignment_gate_active" in s]
    lambdas = [s["lambda_eff"] for s in per_layer_stats if "lambda_eff" in s]

    result = {}
    if cosines:
        result["train/routed_true_grad_cosine"] = sum(cosines) / len(cosines)
        result["train/descent_alignment_ratio"] = sum(c > 0 for c in cosines) / len(cosines)
    if norm_ratios:
        result["train/routed_grad_norm_ratio"] = sum(norm_ratios) / len(norm_ratios)
    if gate_actives:
        result["train/gate_active_ratio"] = sum(gate_actives) / len(gate_actives)
    if lambdas:
        result["train/lambda_eff_mean"] = sum(lambdas) / len(lambdas)

    return result
