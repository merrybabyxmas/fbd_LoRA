"""Reproducible seed control for all RNG sources."""

import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducibility.

    Covers: Python random, NumPy, PyTorch CPU, PyTorch CUDA,
    and optionally Transformers / HuggingFace.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Transformers seed (optional import)
    try:
        import transformers
        transformers.set_seed(seed)
    except ImportError:
        pass
