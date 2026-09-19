"""Tests for deterministic dataset sampling utilities."""

import pytest

from nanomind_slm.data.sampling import (
    assign_split,
)

SEED = 42
VALIDATION_FRACTION = 0.10


def make_record(
    *,
    repo_name: str = "example/project",
    content_hash: str = "content-hash-1",
) -> dict[str, object]:
    """Create a small artificial metadata record."""
    return {
        "repo_name": repo_name,
        "hash": content_hash,
    }


def test_same_repository_always_receives_same_split() -> None:
    first = make_record(
        repo_name="owner/project",
        content_hash="first-file",
    )
    second = make_record(
        repo_name="owner/project",
        content_hash="second-file",
    )

    first_split = assign_split(
        first,
        seed=SEED,
        validation_fraction=VALIDATION_FRACTION,
    )
    second_split = assign_split(
        second,
        seed=SEED,
        validation_fraction=VALIDATION_FRACTION,
    )

    assert first_split == second_split


def test_split_assignment_is_repeatable() -> None:
    record = make_record()

    first = assign_split(
        record,
        seed=SEED,
        validation_fraction=VALIDATION_FRACTION,
    )
    second = assign_split(
        record,
        seed=SEED,
        validation_fraction=VALIDATION_FRACTION,
    )

    assert first == second


def test_invalid_validation_fractions_are_rejected() -> None:
    record = make_record()

    for invalid_fraction in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            assign_split(
                record,
                seed=SEED,
                validation_fraction=invalid_fraction,
            )
