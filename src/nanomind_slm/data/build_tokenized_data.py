"""Build bounded fixed-length token shards without loading full corpora into RAM."""

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tokenizers import Tokenizer

from nanomind_slm.data.packing import pack_token_sequences

CORPUS_FORMAT = "legacy-blank-line-delimited-sections-v1"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a local input file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def iter_documents(path: Path) -> Iterator[str]:
    """Read one bounded code section at a time."""
    lines: list[str] = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                lines.append(line)
            elif lines:
                yield "".join(lines)
                lines.clear()

    if lines:
        yield "".join(lines)


def iter_token_ids(
    path: Path,
    tokenizer: Tokenizer,
) -> Iterator[list[int]]:
    """Tokenize documents one at a time."""
    for document in iter_documents(path):
        yield tokenizer.encode(
            document,
            add_special_tokens=False,
        ).ids


def build_split(
    *,
    experiment: str,
    split: str,
    corpus_path: Path,
    tokenizer: Tokenizer,
    output_root: Path,
    context_length: int,
    shard_token_limit: int,
    dtype: str,
    eos_token: str,
    tokenizer_path: Path,
    tokenizer_sha256: str,
    corpus_sha256: str,
) -> None:
    """Write fixed-length NumPy shards for one experiment and split."""
    eos_token_id = tokenizer.token_to_id(eos_token)

    if eos_token_id is None:
        raise RuntimeError(f"Tokenizer does not contain {eos_token}")

    output_dir = output_root / experiment / split
    output_dir.mkdir(parents=True, exist_ok=True)

    for old_shard in output_dir.glob("shard_*.npy"):
        old_shard.unlink()

    blocks_per_shard = max(1, shard_token_limit // context_length)
    shard_buffer = np.empty(
        (blocks_per_shard, context_length),
        dtype=dtype,
    )

    shard_index = 0
    buffer_index = 0
    total_blocks = 0
    shard_names: list[str] = []

    sequences = iter_token_ids(corpus_path, tokenizer)
    blocks = pack_token_sequences(
        sequences,
        sequence_length=context_length,
        eos_token_id=eos_token_id,
    )

    for block in blocks:
        shard_buffer[buffer_index] = block
        buffer_index += 1
        total_blocks += 1

        if buffer_index == blocks_per_shard:
            shard_name = f"shard_{shard_index:05d}.npy"
            np.save(output_dir / shard_name, shard_buffer)
            shard_names.append(shard_name)

            print(f"Saved {experiment}/{split}/{shard_name}")

            shard_index += 1
            buffer_index = 0

    if buffer_index:
        shard_name = f"shard_{shard_index:05d}.npy"
        np.save(output_dir / shard_name, shard_buffer[:buffer_index])
        shard_names.append(shard_name)

        print(f"Saved {experiment}/{split}/{shard_name}")

    manifest = {
        "experiment": experiment,
        "split": split,
        "context_length": context_length,
        "dtype": dtype,
        "total_blocks": total_blocks,
        "total_tokens": total_blocks * context_length,
        "tokenizer_file": str(tokenizer_path),
        "tokenizer_sha256": tokenizer_sha256,
        "corpus_file": str(corpus_path),
        "corpus_sha256": corpus_sha256,
        "corpus_format": CORPUS_FORMAT,
        "shards": shard_names,
        "shard_sha256": {
            shard_name: sha256_file(output_dir / shard_name)
            for shard_name in shard_names
        },
    }

    with (output_dir / "manifest.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(manifest, file, indent=2)

    print(
        f"{experiment:8} {split:10} "
        f"blocks={total_blocks:6} "
        f"tokens={total_blocks * context_length:9}"
    )


def main() -> None:
    with Path("configs/tokenization.yaml").open(encoding="utf-8") as file:
        config: dict[str, Any] = yaml.safe_load(file)

    tokenization = config["tokenization"]
    output_root = Path(config["output"]["directory"])

    for experiment, paths in config["experiments"].items():
        tokenizer_path = Path(paths["tokenizer"])
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer_sha256 = sha256_file(tokenizer_path)
        vocabulary_size = int(tokenization["vocabulary_size"])

        if tokenizer.get_vocab_size() != vocabulary_size:
            raise RuntimeError(
                f"{experiment} tokenizer has {tokenizer.get_vocab_size()} tokens; "
                f"expected {vocabulary_size}"
            )

        for split in ("train", "validation"):
            corpus_path = Path(paths[f"{split}_corpus"])
            build_split(
                experiment=experiment,
                split=split,
                corpus_path=corpus_path,
                tokenizer=tokenizer,
                output_root=output_root,
                context_length=tokenization["context_length"],
                shard_token_limit=tokenization["shard_token_limit"],
                dtype=tokenization["output_dtype"],
                eos_token=tokenization["eos_token"],
                tokenizer_path=tokenizer_path,
                tokenizer_sha256=tokenizer_sha256,
                corpus_sha256=sha256_file(corpus_path),
            )


if __name__ == "__main__":
    main()
