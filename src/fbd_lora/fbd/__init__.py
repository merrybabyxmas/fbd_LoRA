"""FBD-LoRA core: routing, hooks, metrics."""

from fbd_lora.fbd.hooks import apply_fbd_to_peft_model, remove_fbd_hooks
from fbd_lora.fbd.routing import compute_pullback_metric, route_gradient_pullback

__all__ = [
    "apply_fbd_to_peft_model",
    "remove_fbd_hooks",
    "compute_pullback_metric",
    "route_gradient_pullback",
]
