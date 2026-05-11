"""Utilities to identify PEFT LoRA layers and extract base weights.

Compatible with peft >= 0.14 where LoRA layers use:
    module.lora_A: ModuleDict  (keys: adapter names)
    module.base_layer: the original nn.Module being wrapped
"""

import logging
from typing import Iterator, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def is_peft_lora_layer(module: nn.Module) -> bool:
    """Return True if module is a PEFT LoRA linear layer.

    Checks for the presence of lora_A (ModuleDict) and base_layer.
    """
    return (
        hasattr(module, "lora_A")
        and hasattr(module, "base_layer")
        and isinstance(module.lora_A, nn.ModuleDict)
    )


def get_base_weight(module: nn.Module) -> Optional[torch.Tensor]:
    """Extract the frozen base weight from a PEFT LoRA layer.

    Args:
        module: A PEFT LoRA layer (has base_layer attribute).

    Returns:
        The base weight tensor, or None if not found.
    """
    if not hasattr(module, "base_layer"):
        return None
    base = module.base_layer
    if hasattr(base, "weight"):
        return base.weight
    return None


def get_lora_a_param(
    module: nn.Module,
    adapter_name: str = "default",
) -> Optional[nn.Parameter]:
    """Return the lora_A weight parameter for a given adapter.

    Args:
        module: PEFT LoRA layer.
        adapter_name: Name of the adapter (default: 'default').

    Returns:
        nn.Parameter of shape [rank, in_features], or None.
    """
    if not hasattr(module, "lora_A"):
        return None
    if adapter_name not in module.lora_A:
        return None
    lora_a_layer = module.lora_A[adapter_name]
    if hasattr(lora_a_layer, "weight"):
        return lora_a_layer.weight
    return None


def get_lora_b_param(
    module: nn.Module,
    adapter_name: str = "default",
) -> Optional[nn.Parameter]:
    """Return the lora_B weight parameter for a given adapter.

    Args:
        module: PEFT LoRA layer.
        adapter_name: Name of the adapter (default: 'default').

    Returns:
        nn.Parameter of shape [out_features, rank], or None.
    """
    if not hasattr(module, "lora_B"):
        return None
    if adapter_name not in module.lora_B:
        return None
    lora_b_layer = module.lora_B[adapter_name]
    if hasattr(lora_b_layer, "weight"):
        return lora_b_layer.weight
    return None


def iter_fbd_targets(
    model: nn.Module,
    target_modules: Optional[list] = None,
    adapter_name: str = "default",
) -> Iterator[Tuple[str, nn.Module, nn.Parameter, torch.Tensor]]:
    """Iterate over PEFT LoRA layers eligible for FBD routing.

    For each eligible layer, yields:
        (module_name, peft_layer, lora_a_param, base_weight)

    Args:
        model: Model with PEFT LoRA applied.
        target_modules: If provided, only include modules whose name
            contains one of these strings. If None/empty, include all LoRA layers.
        adapter_name: LoRA adapter name.

    Yields:
        Tuples of (name, module, lora_a_weight_param, base_weight_tensor).
    """
    for name, module in model.named_modules():
        if not is_peft_lora_layer(module):
            continue

        # Filter by target_modules if specified
        if target_modules:
            if not any(t in name for t in target_modules):
                continue

        lora_a = get_lora_a_param(module, adapter_name)
        base_w = get_base_weight(module)

        if lora_a is None or base_w is None:
            logger.debug("Skipping %s: lora_a=%s, base_w=%s", name, lora_a, base_w)
            continue

        yield name, module, lora_a, base_w
