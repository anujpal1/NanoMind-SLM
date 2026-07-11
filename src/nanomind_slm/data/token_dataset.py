"""Memory-mapped PyTorch dataset for token shards."""

import json
from bisect import bisect_right
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class TokenShardDataset(Dataset[Tensor]):
    """Read token blocks without loading complete shards into RAM."""

    def __init__(self, split_directory: str | Path) -> None:
        self.split_directory = Path(split_directory)
        manifest_path = self.split_directory / "manifest.json"

        with manifest_path.open(encoding="utf-8") as file:
            manifest: dict[str, Any] = json.load(file)

        self.context_length = int(manifest["context_length"])
        self.shards: list[np.ndarray] = []
        self.cumulative_lengths: list[int] = []

        total_blocks = 0

        for shard_name in manifest["shards"]:
            shard = np.load(
                self.split_directory / shard_name,
                mmap_mode="r",
            )

            if shard.ndim != 2:
                raise ValueError("Token shard must have two dimensions")

            if shard.shape[1] != self.context_length:
                raise ValueError("Token shard context length is incorrect")

            self.shards.append(shard)
            total_blocks += len(shard)
            self.cumulative_lengths.append(total_blocks)

        if total_blocks != manifest["total_blocks"]:
            raise ValueError("Manifest block count does not match shards")

        self.total_blocks = total_blocks

    def __len__(self) -> int:
        return self.total_blocks

    def __getitem__(self, index: int) -> Tensor:
        if index < 0:
            index += len(self)

        if index < 0 or index >= len(self):
            raise IndexError("Token block index is out of range")

        shard_index = bisect_right(self.cumulative_lengths, index)
        previous_length = (
            self.cumulative_lengths[shard_index - 1]
            if shard_index > 0
            else 0
        )
        local_index = index - previous_length
        block = self.shards[shard_index][local_index]

        return torch.tensor(block, dtype=torch.long)