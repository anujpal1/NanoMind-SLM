"""Pure utilities for deterministic, leakage-aware dataset sampling."""

from collections.abc import Iterable, Mapping
from hashlib import sha256
from typing import Any, Literal

DatasetSplit = Literal["train", "validation"]


def _required_string(record: Mapping[str, Any], field: str) -> str:
    """Return a required non-empty string field."""
    value = record.get(field)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Record field {field!r} must be a non-empty string")

    return value.strip()


def stable_digest(
    value: str,
    *,
    seed: int,
    namespace: str,
) -> bytes:
    """Create a repeatable SHA-256 digest for one value."""
    payload = f"{seed}\0{namespace}\0{value}".encode()
    return sha256(payload).digest()


def record_fingerprint(record: Mapping[str, Any]) -> str:
    """Return the dataset-provided content identity."""
    return _required_string(record, "hash")


def repository_key(record: Mapping[str, Any]) -> str:
    """Return the repository used to keep related files together."""
    return _required_string(record, "repo_name").lower()


def assign_split(
    record: Mapping[str, Any],
    *,
    seed: int,
    validation_fraction: float,
) -> DatasetSplit:
    """Assign an entire repository to train or validation."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    digest = stable_digest(
        repository_key(record),
        seed=seed,
        namespace="repository_split",
    )
    fraction = int.from_bytes(digest[:8], "big") / 2**64

    if fraction < validation_fraction:
        return "validation"

    return "train"


def sampling_priority(
    record: Mapping[str, Any],
    *,
    seed: int,
) -> str:
    """Return a stable priority shared by both experiments."""
    digest = stable_digest(
        record_fingerprint(record),
        seed=seed,
        namespace="sampling_priority",
    )
    return digest.hex()


def find_cross_split_hashes(
    records: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    validation_fraction: float,
) -> set[str]:
    """Find exact content hashes appearing in both dataset splits."""
    observed_splits: dict[str, DatasetSplit] = {}
    leaking_hashes: set[str] = set()

    for record in records:
        fingerprint = record_fingerprint(record)
        current_split = assign_split(
            record,
            seed=seed,
            validation_fraction=validation_fraction,
        )
        previous_split = observed_splits.get(fingerprint)

        if previous_split is not None and previous_split != current_split:
            leaking_hashes.add(fingerprint)
        else:
            observed_splits[fingerprint] = current_split

    return leaking_hashes