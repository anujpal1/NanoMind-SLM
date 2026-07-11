"""Tests for NanoMind model layers."""

import torch

from nanomind_slm.model.layers import (
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    apply_rotary_position_embedding,
)


def test_rms_norm_preserves_shape() -> None:
    layer = RMSNorm(hidden_size=16)
    inputs = torch.randn(2, 4, 16)

    outputs = layer(inputs)

    assert outputs.shape == inputs.shape


def test_rms_norm_produces_unit_rms() -> None:
    layer = RMSNorm(hidden_size=16)
    inputs = torch.randn(2, 4, 16)

    outputs = layer(inputs)
    root_mean_square = outputs.pow(2).mean(dim=-1).sqrt()

    assert torch.allclose(
        root_mean_square,
        torch.ones_like(root_mean_square),
        atol=1e-4,
    )


def test_rotary_embedding_shapes() -> None:
    rotary = RotaryEmbedding(head_dimension=8)

    cosine, sine = rotary(
        6,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert cosine.shape == (1, 1, 6, 8)
    assert sine.shape == (1, 1, 6, 8)


def test_rotary_embedding_preserves_norm() -> None:
    rotary = RotaryEmbedding(head_dimension=8)
    inputs = torch.randn(2, 3, 6, 8)

    cosine, sine = rotary(
        6,
        device=inputs.device,
        dtype=inputs.dtype,
    )
    outputs = apply_rotary_position_embedding(
        inputs,
        cosine,
        sine,
    )

    assert outputs.shape == inputs.shape
    assert torch.allclose(
        outputs.norm(dim=-1),
        inputs.norm(dim=-1),
        atol=1e-5,
    )


def test_swiglu_preserves_hidden_size() -> None:
    layer = SwiGLU(
        hidden_size=16,
        intermediate_size=32,
    )
    inputs = torch.randn(2, 4, 16)

    outputs = layer(inputs)

    assert outputs.shape == inputs.shape