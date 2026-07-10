"""Tests for the memory-mapped token dataset."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nanomind_slm.data.token_dataset import TokenShardDataset


def make_dataset(directory: Path) -> TokenShardDataset:
    first = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ],
        dtype=np.uint16,
    )
    second = np.array(
        [[9, 10, 11, 12]],
        dtype=np.uint16,
    )

    np.save(directory / "shard_00000.npy", first)
    np.save(directory / "shard_00001.npy", second)

    manifest = {
        "context_length": 4,
        "total_blocks": 3,
        "shards": [
            "shard_00000.npy",
            "shard_00001.npy",
        ],
    }

    with (directory / "manifest.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(manifest, file)

    return TokenShardDataset(directory)


def test_dataset_reports_total_blocks(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)

    assert len(dataset) == 3


def test_dataset_returns_long_tensor(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)

    block = dataset[0]

    assert block.dtype == torch.long
    assert block.tolist() == [1, 2, 3, 4]


def test_dataset_reads_across_shards(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)

    assert dataset[2].tolist() == [9, 10, 11, 12]


def test_dataset_supports_negative_index(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)

    assert dataset[-1].tolist() == [9, 10, 11, 12]


def test_dataset_rejects_invalid_index(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)

    with pytest.raises(IndexError):
        _ = dataset[3]