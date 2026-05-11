"""Collision-resistant run naming convention for FBD-LoRA experiments."""

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def sanitize(s: str) -> str:
    """Sanitize a string for use in filesystem paths.

    - Lowercase
    - Replace /:.whitespace with -
    - Remove unsafe shell characters
    - Collapse consecutive dashes
    """
    s = str(s).lower()
    s = re.sub(r"[/:.\s]+", "-", s)
    s = re.sub(r"[^a-z0-9\-_]", "", s)
    s = re.sub(r"-+", "-", s)
    s = s.strip("-")
    return s


def make_target_modules_token(target_modules: list) -> str:
    """Create compact token for target modules list.

    Examples:
        [q_proj, k_proj, v_proj, o_proj] -> qkvo
        [q_proj, v_proj] -> qv
        [to_q, to_k, to_v, to_out.0] -> unet_crossattn
    """
    if not target_modules:
        return "none"
    joined = "".join(sorted(target_modules))
    # Predefined compact tokens
    if set(target_modules) == {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return "qkvo"
    if set(target_modules) == {"q_proj", "v_proj"}:
        return "qv"
    if set(target_modules) >= {"to_q", "to_k", "to_v"}:
        return "unet_crossattn"
    # Generic: first letters
    letters = "".join(m[0] for m in sorted(target_modules))[:8]
    return letters or "custom"


def make_run_name(
    seed: int,
    modality: str,
    task: str,
    backbone: str,
    adapter: str,
    rank: int,
    alpha: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    target_modules: list,
    routing: str,
    lambda_route: float,
    full_config: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None,
) -> str:
    """Generate a collision-resistant run name.

    Format:
        {timestamp}_{seed}_{modality}_{task}_{backbone}_{adapter}_{rank}_{alpha}
        _{bs}_{ga}_{lr}_{target_modules}_{routing}_{lambda}_{runhash}

    Args:
        seed: Random seed.
        modality: 'nlg' or 'imagen'.
        task: Task name (metamath, dreambench, etc.).
        backbone: Model name.
        adapter: Adapter type ('lora', 'fbd', etc.).
        rank: LoRA rank.
        alpha: LoRA alpha.
        batch_size: Per-device batch size.
        grad_accum: Gradient accumulation steps.
        lr: Learning rate.
        target_modules: List of target module names.
        routing: Routing type string.
        lambda_route: Routing strength.
        full_config: Full config dict for hashing. If None, uses above args.
        timestamp: UTC timestamp string. Auto-generated if None.

    Returns:
        Collision-resistant run name string (max 180 chars).
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    # Format lr compactly
    lr_str = f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")

    # Format lambda
    lam_str = f"lam{lambda_route:.2f}".replace(".", "p")

    # Sanitize components
    backbone_tok = sanitize(backbone.split("/")[-1])[:20]
    task_tok = sanitize(task)
    modality_tok = sanitize(modality)
    adapter_tok = sanitize(adapter)
    routing_tok = sanitize(routing)[:20]
    target_tok = make_target_modules_token(target_modules)

    # Hash of full config for collision resistance
    if full_config is not None:
        hash_src = json.dumps(full_config, sort_keys=True, default=str)
    else:
        hash_src = json.dumps({
            "seed": seed, "modality": modality, "task": task, "backbone": backbone,
            "adapter": adapter, "rank": rank, "alpha": alpha, "bs": batch_size,
            "ga": grad_accum, "lr": lr, "target_modules": target_modules,
            "routing": routing, "lambda": lambda_route,
        }, sort_keys=True)
    runhash = hashlib.sha256(hash_src.encode()).hexdigest()[:8]

    parts = [
        timestamp,
        f"seed{seed}",
        modality_tok,
        task_tok,
        backbone_tok,
        adapter_tok,
        f"r{rank}",
        f"a{alpha}",
        f"bs{batch_size}",
        f"ga{grad_accum}",
        f"lr{lr_str}",
        target_tok,
        routing_tok,
        lam_str,
        runhash,
    ]

    name = "_".join(parts)

    # Enforce max length
    if len(name) > 180:
        # Truncate backbone token more aggressively
        name = name[:180]

    return name


# ---------------------------------------------------------------------------
# CLI entry point: python -m fbd_lora.naming --config <path> --print-run-name
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Print the run name for a given config.",
        prog="python -m fbd_lora.naming",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--print-run-name", action="store_true", help="Print run name and exit.")
    args = parser.parse_args()

    if not args.print_run_name:
        parser.print_help()
        sys.exit(1)

    try:
        from omegaconf import OmegaConf
    except ImportError:
        print("[ERROR] omegaconf is required.")
        sys.exit(1)

    try:
        cfg_obj = OmegaConf.load(args.config)
        cfg = OmegaConf.to_container(cfg_obj, resolve=False)
    except Exception as e:
        print(f"[ERROR] Could not load config '{args.config}': {e}")
        sys.exit(1)

    def _get(d, *keys):
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    seed = int(_get(cfg, "run", "seed") or 42)
    modality = cfg.get("modality", "nlg")
    task = cfg.get("task", "unknown")
    backbone = _get(cfg, "model", "backbone") or "unknown"
    adapter = _get(cfg, "adapter", "name") or "lora"
    rank = int(_get(cfg, "adapter", "rank") or 16)
    alpha = int(_get(cfg, "adapter", "alpha") or 16)
    bs = int(_get(cfg, "training", "per_device_train_batch_size")
             or _get(cfg, "training", "train_batch_size") or 4)
    ga = int(_get(cfg, "training", "gradient_accumulation_steps") or 1)
    lr = float(_get(cfg, "training", "learning_rate") or 2e-4)
    target_mods = list(_get(cfg, "adapter", "target_modules") or [])
    routing = _get(cfg, "fbd", "routing_type") or "none"
    lambda_r = float(_get(cfg, "fbd", "lambda_route") or 0.0)

    run_name = make_run_name(
        seed=seed, modality=modality, task=task, backbone=backbone,
        adapter=adapter, rank=rank, alpha=alpha,
        batch_size=bs, grad_accum=ga, lr=lr,
        target_modules=target_mods, routing=routing, lambda_route=lambda_r,
        full_config=cfg,
    )
    print(run_name)
