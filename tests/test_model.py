"""Tests for the complete NanoMind model."""

import torch

from nanomind_slm.model.config import NanoMindConfig
from nanomind_slm.model.model import NanoMindModel


def small_config() -> NanoMindConfig:
    return NanoMindConfig(
        vocabulary_size=32,
        context_length=8,
        hidden_size=16,
        number_of_layers=2,
        number_of_heads=4,
        number_of_kv_heads=2,
        intermediate_size=32,
    )


def test_model_produces_vocabulary_logits() -> None:
    model = NanoMindModel(small_config())
    input_ids = torch.randint(0, 32, (2, 6))

    logits = model(input_ids)

    assert logits.shape == (2, 6, 32)


def test_model_supports_backpropagation() -> None:
    model = NanoMindModel(small_config())
    input_ids = torch.randint(0, 32, (2, 6))

    model(input_ids).mean().backward()

    assert model.token_embedding.weight.grad is not None


def test_embedding_weights_are_tied() -> None:
    model = NanoMindModel(small_config())

    assert (
        model.token_embedding.weight.data_ptr()
        == model.output_projection.weight.data_ptr()
    )


def test_default_model_matches_parameter_target() -> None:
    model = NanoMindModel(NanoMindConfig())

    assert 15_000_000 <= model.parameter_count() <= 25_000_000