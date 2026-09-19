"""Tests for token-shard manifest provenance."""

import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from nanomind_slm.data.build_tokenized_data import (
    CORPUS_FORMAT,
    build_split,
    sha256_file,
)


def test_build_split_records_exact_input_hashes(tmp_path: Path) -> None:
    tokenizer = Tokenizer(
        WordLevel(
            vocab={"<unk>": 0, "<eos>": 1, "alpha": 2, "beta": 3},
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()

    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    corpus_path = tmp_path / "validation.txt"
    corpus_path.write_text("alpha\n\nbeta\n", encoding="utf-8")

    tokenizer_sha256 = sha256_file(tokenizer_path)
    corpus_sha256 = sha256_file(corpus_path)
    output_root = tmp_path / "tokenized"

    build_split(
        experiment="quality",
        split="validation",
        corpus_path=corpus_path,
        tokenizer=tokenizer,
        output_root=output_root,
        context_length=4,
        shard_token_limit=8,
        dtype="uint16",
        eos_token="<eos>",
        tokenizer_path=tokenizer_path,
        tokenizer_sha256=tokenizer_sha256,
        corpus_sha256=corpus_sha256,
    )

    manifest_path = output_root / "quality" / "validation" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["tokenizer_sha256"] == hashlib.sha256(
        tokenizer_path.read_bytes()
    ).hexdigest()
    assert manifest["corpus_sha256"] == hashlib.sha256(
        corpus_path.read_bytes()
    ).hexdigest()
    assert manifest["tokenizer_file"] == str(tokenizer_path)
    assert manifest["corpus_file"] == str(corpus_path)
    assert manifest["corpus_format"] == CORPUS_FORMAT
