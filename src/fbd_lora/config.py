"""FBD-LoRA configuration dataclass."""

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

RoutingType = Literal[
    "none",
    "direct_activation",
    "pullback_metric",
    "pullback_metric_mixed",
    "pullback_metric_mixed_gated",
]

MetricMode = Literal["full", "diag", "lowrank"]

NormalizeMode = Literal["none", "fro", "spectral", "rms"]


@dataclass
class FBDConfig:
    """Configuration for Forward-Backward Decoupled LoRA.

    The forward pass is identical to standard PEFT LoRA.
    Only the gradient of lora_A is modified during backward.
    """

    enabled: bool = True
    routing_type: RoutingType = "pullback_metric_mixed"
    metric_mode: MetricMode = "diag"
    route_a: bool = True
    route_b: bool = False
    lambda_route: float = 0.25
    epsilon: float = 1e-4
    normalize_metric: NormalizeMode = "rms"
    norm_match: bool = True
    alignment_gate: bool = True
    gate_type: str = "hard"            # "hard" or "sigmoid"
    gate_temperature: float = 10.0
    metric_cache: bool = True
    metric_rank_approx: Optional[int] = None  # for lowrank mode
    log_gradient_stats: bool = True
    gradient_stats_interval: int = 10
    target_modules: Sequence[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "FBDConfig":
        """Create FBDConfig from a plain dict (e.g., loaded from YAML)."""
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)
