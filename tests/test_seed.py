"""Tests for the reproducibility seed utility."""

import random

import numpy as np
import torch

from nanomind_slm.utils import set_seed


def test_same_seed_produces_same_values() -> None:
    """The same seed should reproduce the same random values."""
    set_seed(123)

    first_python_value = random.random()
    first_numpy_value = np.random.random()
    first_torch_values = torch.rand(3)

    set_seed(123)

    second_python_value = random.random()
    second_numpy_value = np.random.random()
    second_torch_values = torch.rand(3)

    assert first_python_value == second_python_value
    assert first_numpy_value == second_numpy_value
    assert torch.equal(first_torch_values, second_torch_values)


def test_invalid_seed_raises_error() -> None:
    """A negative seed should be rejected."""
    try:
        set_seed(-1)
    except ValueError:
        return

    raise AssertionError("set_seed(-1) should raise ValueError")