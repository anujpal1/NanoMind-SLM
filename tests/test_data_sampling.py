"""Tests for deterministic dataset sampling utilities."""

import pytest

from nanomind_slm.data.sampling import (
    assign_split,
    find_cross_split_hashes,
    record_fingerprint,
    sampling_priority,
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


def test_record_fingerprint_uses_dataset_hash() -> None:
    record = make_record(content_hash="abc123")

    assert record_fingerprint(record) == "abc123"


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


def test_sampling_priority_is_repeatable() -> None:
    record = make_record()

    first = sampling_priority(record, seed=SEED)
    second = sampling_priority(record, seed=SEED)

    assert first == second


def test_seed_changes_sampling_priority() -> None:
    record = make_record()

    assert sampling_priority(record, seed=42) != sampling_priority(
        record,
        seed=43,
    )


def test_invalid_validation_fractions_are_rejected() -> None:
    record = make_record()

    for invalid_fraction in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            assign_split(
                record,
                seed=SEED,
                validation_fraction=invalid_fraction,
            )


def test_cross_split_duplicate_is_detected() -> None:
    repositories: dict[str, str] = {}

    for index in range(1000):
        record = make_record(
            repo_name=f"owner/project-{index}",
            content_hash=f"unique-{index}",
        )
        split = assign_split(
            record,
            seed=SEED,
            validation_fraction=VALIDATION_FRACTION,
        )
        repositories.setdefault(split, f"owner/project-{index}")

        if len(repositories) == 2:
            break

    assert set(repositories) == {"train", "validation"}

    duplicate_records = [
        make_record(
            repo_name=repositories["train"],
            content_hash="duplicated-content",
        ),
        make_record(
            repo_name=repositories["validation"],
            content_hash="duplicated-content",
        ),
    ]

    leaking_hashes = find_cross_split_hashes(
        duplicate_records,
        seed=SEED,
        validation_fraction=VALIDATION_FRACTION,
    )

    assert leaking_hashes == {"duplicated-content"}