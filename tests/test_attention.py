"""Tests for grouped-query causal attention."""

import pytest
import torch

from nanomind_slm.model.attention import GroupedQueryAttention
from nanomind_slm.model.config import NanoMindConfig


def small_config() -> NanoMindConfig:
    return NanoMindConfig(
        vocabulary_size=32,
        context_length=8,
        hidden_size=16,
        number_of_layers=2,
        number_of_heads=4,
        number_of_kv_heads=2,
        intermediate_size=32,
        dropout=0.0,
    )


def test_attention_preserves_shape() -> None:
    attention = GroupedQueryAttention(small_config())
    inputs = torch.randn(2, 6, 16)

    outputs = attention(inputs)

    assert outputs.shape == inputs.shape


def test_attention_is_causal() -> None:
    attention = GroupedQueryAttention(small_config())
    attention.eval()

    original = torch.randn(1, 6, 16)
    changed = original.clone()
    changed[:, 3:] += 100.0

    original_outputs = attention(original)
    changed_outputs = attention(changed)

    assert torch.allclose(
        original_outputs[:, :3],
        changed_outputs[:, :3],
        atol=1e-5,
    )


def test_attention_supports_gradients() -> None:
    attention = GroupedQueryAttention(small_config())
    inputs = torch.randn(
        2,
        6,
        16,
        requires_grad=True,
    )

    attention(inputs).sum().backward()

    assert inputs.grad is not None
    assert attention.query_projection.weight.grad is not None


def test_attention_rejects_long_sequence() -> None:
    attention = GroupedQueryAttention(small_config())
    inputs = torch.randn(1, 9, 16)

    with pytest.raises(ValueError):
        attention(inputs)