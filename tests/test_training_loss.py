"""Tests for causal language-model loss."""

import pytest
import torch

from nanomind_slm.training.loss import (
    causal_language_model_loss,
)


def test_causal_loss_returns_scalar() -> None:
    logits = torch.randn(
        2,
        5,
        16,
        requires_grad=True,
    )
    token_ids = torch.randint(0, 16, (2, 5))

    loss = causal_language_model_loss(logits, token_ids)

    assert loss.ndim == 0


def test_causal_loss_supports_backpropagation() -> None:
    logits = torch.randn(
        2,
        5,
        16,
        requires_grad=True,
    )
    token_ids = torch.randint(0, 16, (2, 5))

    loss = causal_language_model_loss(logits, token_ids)
    loss.backward()

    assert logits.grad is not None


def test_causal_loss_is_small_for_correct_predictions() -> None:
    logits = torch.zeros(1, 3, 3)
    logits[0, 0, 1] = 10.0
    logits[0, 1, 2] = 10.0
    token_ids = torch.tensor([[0, 1, 2]])

    loss = causal_language_model_loss(logits, token_ids)

    assert loss.item() < 0.001


def test_causal_loss_rejects_wrong_shape() -> None:
    logits = torch.randn(2, 5, 16)
    token_ids = torch.randint(0, 16, (2, 4))

    with pytest.raises(ValueError):
        causal_language_model_loss(logits, token_ids)