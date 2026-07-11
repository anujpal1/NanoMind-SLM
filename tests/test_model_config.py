"""Tests for the NanoMind model configuration."""

import pytest

from nanomind_slm.model.config import NanoMindConfig


def test_default_configuration_matches_project() -> None:
    config = NanoMindConfig()

    assert config.vocabulary_size == 8000
    assert config.context_length == 256
    assert config.hidden_size == 384
    assert config.number_of_layers == 10
    assert config.head_dimension == 64


def test_hidden_size_must_match_attention_heads() -> None:
    with pytest.raises(ValueError):
        NanoMindConfig(
            hidden_size=385,
            number_of_heads=6,
        )


def test_query_heads_must_match_kv_heads() -> None:
    with pytest.raises(ValueError):
        NanoMindConfig(
            number_of_heads=6,
            number_of_kv_heads=4,
        )


def test_dropout_must_be_valid() -> None:
    with pytest.raises(ValueError):
        NanoMindConfig(dropout=1.0)
        