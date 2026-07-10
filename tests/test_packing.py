"""Tests for fixed-length token packing."""

import pytest

from nanomind_slm.data.packing import pack_token_sequences


def test_packer_creates_fixed_length_blocks() -> None:
    blocks = list(
        pack_token_sequences(
            [[1, 2], [3, 4, 5]],
            sequence_length=4,
            eos_token_id=9,
        )
    )

    assert blocks == [[1, 2, 9, 3]]
    assert all(len(block) == 4 for block in blocks)


def test_packer_adds_eos_token() -> None:
    blocks = list(
        pack_token_sequences(
            [[1, 2, 3]],
            sequence_length=4,
            eos_token_id=9,
        )
    )

    assert blocks == [[1, 2, 3, 9]]


def test_packer_is_deterministic() -> None:
    sequences = [[1, 2], [3, 4]]

    first = list(
        pack_token_sequences(
            sequences,
            sequence_length=3,
            eos_token_id=9,
        )
    )
    second = list(
        pack_token_sequences(
            sequences,
            sequence_length=3,
            eos_token_id=9,
        )
    )

    assert first == second


def test_packer_rejects_invalid_sequence_length() -> None:
    with pytest.raises(ValueError):
        list(
            pack_token_sequences(
                [[1, 2]],
                sequence_length=1,
                eos_token_id=9,
            )
        )


def test_packer_rejects_invalid_token_id() -> None:
    with pytest.raises(ValueError):
        list(
            pack_token_sequences(
                [[1, -1]],
                sequence_length=4,
                eos_token_id=9,
            )
        )