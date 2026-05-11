"""Utilities for identifying and extracting LoRA layers in diffusers UNet.

Provides helper functions to:
- List all PEFT LoRA-patched modules in a diffusers UNet.
- Extract base (frozen) weights for metric computation.
- Verify LoRA target module naming conventions for diffusers 0.37.x + peft 0.17.x.

These functions reuse the logic in src/fbd_lora/fbd/peft_patch.py but add
diffusers-specific naming utilities.
"""

import logging
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

from fbd_lora.fbd.peft_patch import (
    get_base_weight,
    get_lora_a_param,
    is_peft_lora_layer,
    iter_fbd_targets,
)

logger = logging.getLogger(__name__)


# Standard cross-attention target module suffixes for SD v1.5 UNet
# Module path pattern: down_blocks.X.attentions.Y.transformer_blocks.Z.attnW.<suffix>
SD15_ATTENTION_SUFFIXES = ("to_q", "to_k", "to_v", "to_out.0")


def list_lora_modules(
    unet: nn.Module,
    adapter_name: str = "default",
) -> List[Tuple[str, nn.Module]]:
    """Return all (name, module) pairs that are PEFT LoRA layers in the UNet.

    Args:
        unet: Diffusers UNet model with PEFT LoRA applied.
        adapter_name: LoRA adapter name.

    Returns:
        List of (module_name, peft_layer_module) tuples.
    """
    result = []
    for name, module in unet.named_modules():
        if is_peft_lora_layer(module):
            lora_a = get_lora_a_param(module, adapter_name)
            if lora_a is not None:
                result.append((name, module))
    logger.info("Found %d PEFT LoRA modules in UNet.", len(result))
    return result


def get_unet_lora_target_names(
    unet: nn.Module,
    target_suffixes: Tuple[str, ...] = SD15_ATTENTION_SUFFIXES,
) -> List[str]:
    """Return all UNet module names whose suffix matches target_suffixes.

    Used to verify which modules will be targeted by LoRA before training.

    Args:
        unet: Diffusers UNet (before or after PEFT patching).
        target_suffixes: Tuple of module name suffixes to match.

    Returns:
        Sorted list of matching module names.
    """
    result = []
    for name, module in unet.named_modules():
        if any(name.endswith(s) for s in target_suffixes):
            if isinstance(module, nn.Linear):
                result.append(name)
    return sorted(result)


def extract_base_weight_map(
    unet: nn.Module,
    target_modules: Optional[List[str]] = None,
    adapter_name: str = "default",
) -> Dict[str, torch.Tensor]:
    """Extract {module_name: base_weight} mapping for FBD metric pre-computation.

    Only includes modules that have both lora_A and a base weight.

    Args:
        unet: UNet with PEFT LoRA applied.
        target_modules: Optional filter list (module name substrings).
        adapter_name: LoRA adapter name.

    Returns:
        Dict mapping module_name -> base_weight (frozen, on CPU).
    """
    weight_map: Dict[str, torch.Tensor] = {}
    for name, module, lora_a, base_w in iter_fbd_targets(
        unet,
        target_modules=target_modules,
        adapter_name=adapter_name,
    ):
        # Store on CPU to avoid holding extra GPU memory
        weight_map[name] = base_w.detach().cpu()
    logger.info("Extracted %d base weights from UNet.", len(weight_map))
    return weight_map


def verify_lora_applied(
    unet: nn.Module,
    expected_target_modules: List[str],
    adapter_name: str = "default",
) -> bool:
    """Verify that LoRA was applied to the expected target modules.

    Logs warnings for any missing modules.

    Args:
        unet: UNet with PEFT LoRA applied.
        expected_target_modules: List of module name suffixes expected to have LoRA.
        adapter_name: LoRA adapter name.

    Returns:
        True if all expected modules have LoRA applied, False otherwise.
    """
    lora_names = {name for name, _ in list_lora_modules(unet, adapter_name)}
    missing = []
    for suffix in expected_target_modules:
        if not any(name.endswith(suffix) for name in lora_names):
            missing.append(suffix)
            logger.warning("Expected LoRA target '%s' not found in UNet!", suffix)

    if missing:
        logger.error("Missing LoRA modules: %s", missing)
        return False

    logger.info(
        "LoRA verification OK: all %d expected target suffixes found in %d LoRA modules.",
        len(expected_target_modules), len(lora_names)
    )
    return True


def count_lora_parameters(unet: nn.Module, adapter_name: str = "default") -> int:
    """Count total trainable LoRA parameters in the UNet.

    Args:
        unet: UNet with PEFT LoRA applied.
        adapter_name: LoRA adapter name.

    Returns:
        Total number of LoRA trainable parameters.
    """
    total = 0
    for name, module in unet.named_modules():
        if not is_peft_lora_layer(module):
            continue
        lora_a = get_lora_a_param(module, adapter_name)
        if lora_a is not None:
            total += lora_a.numel()
        # Count lora_B too
        if hasattr(module, "lora_B") and adapter_name in module.lora_B:
            lora_b_layer = module.lora_B[adapter_name]
            if hasattr(lora_b_layer, "weight"):
                total += lora_b_layer.weight.numel()
    return total
