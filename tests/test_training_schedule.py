"""Tests for the learning-rate schedule."""

import pytest

from nanomind_slm.training.schedule import cosine_learning_rate


def learning_rate(step: int) -> float:
    return cosine_learning_rate(
        step,
        maximum_learning_rate=3e-4,
        minimum_learning_rate=3e-5,
        warmup_steps=10,
        maximum_steps=100,
    )


def test_learning_rate_warms_up() -> None:
    assert learning_rate(0) == pytest.approx(3e-5)
    assert learning_rate(9) == pytest.approx(3e-4)


def test_learning_rate_decays() -> None:
    assert learning_rate(50) < learning_rate(10)


def test_learning_rate_reaches_minimum() -> None:
    assert learning_rate(100) == pytest.approx(3e-5)


def test_invalid_schedule_is_rejected() -> None:
    with pytest.raises(ValueError):
        cosine_learning_rate(
            0,
            maximum_learning_rate=3e-4,
            minimum_learning_rate=3e-5,
            warmup_steps=100,
            maximum_steps=100,
        )