"""FBD-LoRA configuration dataclass and CLI validator.

CLI usage:
    python -m fbd_lora.config --validate --config <path> --mode train
    python -m fbd_lora.config --validate --config <path> --mode eval
    python -m fbd_lora.config --print-run-name --config <path>
"""

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


# ---------------------------------------------------------------------------
# Config validation helpers
# ---------------------------------------------------------------------------

_COMMON_REQUIRED = [
    "modality",
    "task",
]

_TRAIN_REQUIRED_PATHS = [
    ("model", "backbone"),
    ("adapter", "name"),
    ("adapter", "rank"),
    ("adapter", "alpha"),
    ("adapter", "dropout"),
    ("training",),  # block must exist
]

_EVAL_REQUIRED_PATHS = [
    ("evaluation",),
]


def _get_nested(cfg: dict, *keys: str):
    """Traverse nested dict; return value or None if missing."""
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def validate_config(cfg: dict, mode: str = "train") -> list:
    """Validate config dict against required fields.

    Returns a list of error strings (empty = valid).
    """
    errors = []

    # Skip placeholder configs — they self-report as unimplemented
    if _get_nested(cfg, "project", "implemented") is False:
        errors.append("Config is marked as placeholder (project.implemented: false).")
        return errors

    # Common fields
    for key in _COMMON_REQUIRED:
        if key not in cfg or cfg[key] in (None, ""):
            errors.append(f"Missing required top-level field: '{key}'")

    if mode == "train":
        # model
        backbone = _get_nested(cfg, "model", "backbone")
        if backbone is None or str(backbone).startswith("PLACEHOLDER"):
            errors.append("Missing or placeholder: model.backbone")

        # adapter
        adapter_type = _get_nested(cfg, "adapter", "type") or _get_nested(cfg, "adapter", "name") or ""
        modality = cfg.get("modality", "nlg")
        # Full fine-tuning imagen methods (dreambooth, custom_diffusion) don't use LoRA rank/alpha/dropout
        _no_rank_types = frozenset({"dreambooth", "custom_diffusion"})
        for key in ("name", "rank", "alpha", "dropout"):
            val = _get_nested(cfg, "adapter", key)
            if val is None:
                if key == "rank" and str(adapter_type).lower() == "adalora":
                    # AdaLoRA uses init_r/target_r instead of rank
                    if _get_nested(cfg, "adapter", "init_r") is None:
                        errors.append("Missing required field: adapter.rank or adapter.init_r (for adalora)")
                elif key in ("rank", "alpha", "dropout") and str(adapter_type).lower() in _no_rank_types:
                    # Full fine-tuning methods don't have LoRA rank/alpha/dropout
                    pass
                elif key == "name":
                    # Also accept adapter.type as an alias for adapter.name
                    if _get_nested(cfg, "adapter", "type") is None:
                        errors.append("Missing required field: adapter.name or adapter.type")
                else:
                    errors.append(f"Missing required field: adapter.{key}")

        # training block must exist
        if "training" not in cfg or not isinstance(cfg.get("training"), dict):
            errors.append("Missing required block: 'training'")
        else:
            train = cfg["training"]
            for key in ("learning_rate", "per_device_train_batch_size", "gradient_accumulation_steps"):
                if key not in train and "train_batch_size" not in train:
                    # imagen uses train_batch_size instead of per_device_train_batch_size
                    if key == "per_device_train_batch_size":
                        continue
                    errors.append(f"Missing required field: training.{key}")

        # Check num_train_epochs or max_steps or max_train_steps
        train = cfg.get("training", {})
        has_epoch = "num_train_epochs" in train
        has_steps = "max_steps" in train or "max_train_steps" in train
        if not has_epoch and not has_steps:
            errors.append("Missing required field: training.num_train_epochs or training.max_steps")

    if mode == "eval":
        if "evaluation" not in cfg:
            errors.append("Missing required block: 'evaluation'")

    return errors


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    import argparse
    import sys
    try:
        from omegaconf import OmegaConf
    except ImportError:
        print("[ERROR] omegaconf is required for config validation.")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="FBD-LoRA config validator and run name printer.",
        prog="python -m fbd_lora.config",
    )
    parser.add_argument("--validate", action="store_true", help="Validate a config file.")
    parser.add_argument("--print-run-name", action="store_true",
                        help="Print the run name for a config and exit.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "eval"], help="Validation mode.")
    args = parser.parse_args()

    if args.config is None:
        parser.print_help()
        sys.exit(1)

    try:
        cfg_obj = OmegaConf.load(args.config)
        cfg = OmegaConf.to_container(cfg_obj, resolve=False)
    except Exception as e:
        print(f"[ERROR] Could not load config '{args.config}': {e}")
        sys.exit(1)

    if args.validate:
        errors = validate_config(cfg, mode=args.mode)
        if errors:
            print(f"[ERROR] Config validation FAILED for: {args.config}")
            for err in errors:
                print(f"  - {err}")
            sys.exit(1)
        else:
            print(f"[OK] Config validation passed: {args.config} (mode={args.mode})")
            sys.exit(0)

    if args.print_run_name:
        from fbd_lora.naming import make_run_name
        seed = int(_get_nested(cfg, "run", "seed") or 42)
        modality = cfg.get("modality", "nlg")
        task = cfg.get("task", "unknown")
        backbone = _get_nested(cfg, "model", "backbone") or "unknown"
        adapter = _get_nested(cfg, "adapter", "name") or "lora"
        rank = int(_get_nested(cfg, "adapter", "rank") or 16)
        alpha = int(_get_nested(cfg, "adapter", "alpha") or 16)
        bs = int(_get_nested(cfg, "training", "per_device_train_batch_size")
                 or _get_nested(cfg, "training", "train_batch_size") or 4)
        ga = int(_get_nested(cfg, "training", "gradient_accumulation_steps") or 1)
        lr = float(_get_nested(cfg, "training", "learning_rate") or 2e-4)
        target_mods = list(_get_nested(cfg, "adapter", "target_modules") or [])
        routing = _get_nested(cfg, "fbd", "routing_type") or "none"
        lambda_r = float(_get_nested(cfg, "fbd", "lambda_route") or 0.0)

        run_name = make_run_name(
            seed=seed, modality=modality, task=task, backbone=backbone,
            adapter=adapter, rank=rank, alpha=alpha,
            batch_size=bs, grad_accum=ga, lr=lr,
            target_modules=target_mods, routing=routing, lambda_route=lambda_r,
            full_config=cfg,
        )
        print(run_name)
        sys.exit(0)

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    _cli_main()
