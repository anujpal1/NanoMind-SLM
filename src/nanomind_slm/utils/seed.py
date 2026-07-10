"""Utilities for making experiments more reproducible."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds used by Python, NumPy and PyTorch.

    Args:
        seed: Integer between 0 and 4,294,967,295.
        deterministic: Request deterministic PyTorch operations when possible.

    Raises:
        ValueError: If the seed is outside the supported range.
    """
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be between 0 and 4,294,967,295")

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True