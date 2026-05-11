"""Gradient hook registration for FBD-LoRA.

Registers backward hooks on PEFT LoRA lora_A parameters.
The hook intercepts the gradient of lora_A and replaces it with the
pullback-metric-routed surrogate gradient.

The forward pass is UNCHANGED - hooks affect only gradient computation.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from fbd_lora.config import FBDConfig
from fbd_lora.fbd.metrics import aggregate_gradient_stats
from fbd_lora.fbd.peft_patch import get_lora_b_param, iter_fbd_targets
from fbd_lora.fbd.routing import compute_pullback_metric, route_gradient_pullback

logger = logging.getLogger(__name__)


class FBDHookState:
    """Container for all FBD hook handles and accumulated stats.

    Attributes:
        handles: List of torch hook handles (for cleanup).
        step: Current training step counter.
        per_layer_stats: Last logged stats per layer.
        all_stats_buffer: Buffer accumulating stats between log intervals.
    """

    def __init__(self) -> None:
        self.handles: List = []
        self.step: int = 0
        self.per_layer_stats: List[dict] = []
        self.all_stats_buffer: List[List[dict]] = []
        self._layer_names: List[str] = []

    def add_handle(self, handle: torch.utils.hooks.RemovableHandle, layer_name: str) -> None:
        self.handles.append(handle)
        self._layer_names.append(layer_name)

    def remove_all(self) -> None:
        """Remove all registered hooks."""
        for h in self.handles:
            h.remove()
        self.handles.clear()
        logger.info("Removed %d FBD gradient hooks.", len(self._layer_names))

    def get_and_clear_stats(self) -> dict:
        """Return aggregated gradient stats and reset buffer."""
        if not self.all_stats_buffer:
            return {}
        # Flatten all collected stats across steps
        all_stats = [s for step_stats in self.all_stats_buffer for s in step_stats]
        result = aggregate_gradient_stats(all_stats)
        self.all_stats_buffer.clear()
        return result


def _make_fbd_hook(
    metric: torch.Tensor,
    fbd_config: FBDConfig,
    layer_name: str,
    state: FBDHookState,
) -> callable:
    """Create a gradient hook closure for a single lora_A parameter.

    The hook is called with the gradient of lora_A after each backward pass.
    It replaces the gradient with the pullback-metric-routed version.

    Args:
        metric: Precomputed pullback metric P0 (cached on CPU or device).
        fbd_config: FBD configuration.
        layer_name: Human-readable name for logging.
        state: Shared FBDHookState for stats accumulation.

    Returns:
        Hook function suitable for parameter.register_hook().
    """
    # Cache metric on the same device class - will be moved in hook
    _metric = metric

    def hook(grad: torch.Tensor) -> torch.Tensor:
        """Intercept and route lora_A gradient. grad: [rank, in_features]."""
        if grad is None:
            return grad

        # Move metric to same device/dtype context as gradient
        metric_device = _metric.to(grad.device)

        routed_grad, stats = route_gradient_pullback(
            grad_a=grad,
            metric=metric_device,
            lambda_route=fbd_config.lambda_route,
            norm_match=fbd_config.norm_match,
            alignment_gate=fbd_config.alignment_gate,
            gate_type=fbd_config.gate_type,
            gate_temperature=fbd_config.gate_temperature,
            eps=1e-8,
        )

        stats["layer"] = layer_name
        # Accumulate stats (buffer indexed by step group)
        if not state.all_stats_buffer or len(state.all_stats_buffer[-1]) >= 1000:
            state.all_stats_buffer.append([])
        if state.all_stats_buffer:
            state.all_stats_buffer[-1].append(stats)

        return routed_grad

    return hook


def apply_fbd_to_peft_model(
    model: nn.Module,
    fbd_config: FBDConfig,
    adapter_name: str = "default",
) -> FBDHookState:
    """Register FBD gradient hooks on all eligible PEFT LoRA layers.

    This function:
    1. Identifies PEFT LoRA layers in the model.
    2. Precomputes the pullback metric from the frozen base weight.
    3. Registers a gradient hook on lora_A.weight that replaces the gradient
       with the descent-aligned pullback-metric-routed surrogate.
    4. Returns an FBDHookState for monitoring and cleanup.

    IMPORTANT: The forward pass is completely unchanged. FBD operates
    only on the backward gradient of lora_A.

    Args:
        model: Model with PEFT LoRA applied (output of get_peft_model).
        fbd_config: FBD configuration dataclass.
        adapter_name: LoRA adapter name (default: 'default').

    Returns:
        FBDHookState containing hook handles and accumulated stats.
    """
    if not fbd_config.enabled or fbd_config.lambda_route == 0.0:
        logger.info("FBD disabled (enabled=%s, lambda=%.3f). No hooks registered.",
                    fbd_config.enabled, fbd_config.lambda_route)
        return FBDHookState()

    state = FBDHookState()
    n_registered = 0

    for name, module, lora_a_param, base_weight in iter_fbd_targets(
        model,
        target_modules=list(fbd_config.target_modules) if fbd_config.target_modules else None,
        adapter_name=adapter_name,
    ):
        # lora_a_param: [rank, in_features]
        in_features = lora_a_param.shape[1]

        # base_weight shape depends on module type:
        #   nn.Linear: [out_features, in_features] -> use as-is
        #   Conv1D (GPT-2 style, fan_in_fan_out=True): [in_features, out_features] -> transpose
        # We normalize to [out_features, in_features] before computing the metric.
        bw = base_weight
        if bw.shape[1] == in_features:
            # Already [out, in] layout -- standard nn.Linear
            weight_for_metric = bw
        elif bw.shape[0] == in_features:
            # [in, out] layout -- Conv1D with fan_in_fan_out=True
            weight_for_metric = bw.t()
        else:
            # Fallback: use whatever shape and let compute_pullback_metric handle it
            weight_for_metric = bw
            logger.warning(
                "Unexpected base weight shape %s for lora_A in_features=%d at %s",
                tuple(bw.shape), in_features, name,
            )

        # Precompute metric from frozen base weight
        metric = compute_pullback_metric(
            weight=weight_for_metric,
            epsilon=fbd_config.epsilon,
            normalize=fbd_config.normalize_metric,
            mode=fbd_config.metric_mode,
            rank_approx=fbd_config.metric_rank_approx,
        )

        # Sanity check: metric must be compatible with lora_A
        if fbd_config.metric_mode == "diag":
            assert metric.shape == (in_features,), \
                f"Metric shape mismatch: got {metric.shape}, expected ({in_features},) at {name}"
        elif fbd_config.metric_mode in ("full", "lowrank"):
            assert metric.shape == (in_features, in_features), \
                f"Metric shape mismatch: got {metric.shape}, expected ({in_features}, {in_features}) at {name}"

        # Register hook on lora_A gradient
        if fbd_config.route_a:
            hook_fn = _make_fbd_hook(metric, fbd_config, name, state)
            handle = lora_a_param.register_hook(hook_fn)
            state.add_handle(handle, name)
            n_registered += 1
            logger.debug(
                "FBD hook registered: %s | base_weight=%s | metric=%s | rank=%s",
                name, tuple(base_weight.shape), tuple(metric.shape), lora_a_param.shape[0]
            )

        # Optionally route lora_B gradient
        if fbd_config.route_b:
            lora_b_param = get_lora_b_param(module, adapter_name)
            if lora_b_param is not None:
                # For lora_B [out, rank], compute metric from W0 transpose
                # Metric for B: P0_B = W0 @ W0^T  [out, out]
                # Using diagonal for efficiency
                metric_b = compute_pullback_metric(
                    weight=base_weight.t(),   # [in, out] -> treat as [out, in] for diag
                    epsilon=fbd_config.epsilon,
                    normalize=fbd_config.normalize_metric,
                    mode="diag",  # always diag for B to save memory
                    rank_approx=None,
                )
                hook_b = _make_fbd_hook(metric_b, fbd_config, name + ".lora_B", state)
                handle_b = lora_b_param.register_hook(hook_b)
                state.add_handle(handle_b, name + ".lora_B")

    logger.info(
        "FBD-LoRA: registered %d gradient hooks (route_a=%s, route_b=%s, lambda=%.3f, mode=%s)",
        n_registered, fbd_config.route_a, fbd_config.route_b,
        fbd_config.lambda_route, fbd_config.metric_mode,
    )

    assert n_registered > 0, (
        "No FBD hooks registered! Check that PEFT LoRA is applied "
        "and target_modules are correct."
    )

    return state


def remove_fbd_hooks(state: FBDHookState) -> None:
    """Remove all FBD gradient hooks.

    Should be called after training to clean up.

    Args:
        state: FBDHookState returned by apply_fbd_to_peft_model.
    """
    state.remove_all()
