"""Optional checks against a downloaded published release."""

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml
from tokenizers import Tokenizer

from nanomind_slm.model import NanoMindConfig, NanoMindModel

PROJECT_ROOT = Path(__file__).parents[1]
RELEASE_DIRECTORY = PROJECT_ROOT / "release" / "NanoMind-SLM-60M"
CHECKPOINT_PATH = RELEASE_DIRECTORY / "model.pt"
EXPECTED_CHECKPOINT_SHA256 = (
    "154bf1b3847bbca7ba735f4152468e3a3e2f1ba9c45adb236767b0d1f6cc5734"
)

requires_release = pytest.mark.skipif(
    not CHECKPOINT_PATH.exists(),
    reason="published checkpoint is not downloaded",
)


@requires_release
def test_published_checkpoint_loads_with_expected_layout() -> None:
    with CHECKPOINT_PATH.open("rb") as checkpoint_file:
        digest = hashlib.file_digest(
            checkpoint_file,
            "sha256",
        ).hexdigest()
    config_data = yaml.safe_load(
        (RELEASE_DIRECTORY / "model.yaml").read_text(encoding="utf-8")
    )
    config = NanoMindConfig(**config_data["model"])
    model = NanoMindModel(config)
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model"])
    tokenizer = Tokenizer.from_file(
        str(RELEASE_DIRECTORY / "tokenizer" / "tokenizer.json")
    )

    assert digest == EXPECTED_CHECKPOINT_SHA256
    assert model.parameter_count() == 18_808_704
    assert config.vocabulary_size == 8_000
    assert tokenizer.get_vocab_size() == 8_000
    assert [
        tokenizer.token_to_id(token)
        for token in ("<pad>", "<unk>", "<bos>", "<eos>")
    ] == [0, 1, 2, 3]


@requires_release
def test_published_checkpoint_runs_through_real_cli() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate.py",
            "--model-dir",
            str(RELEASE_DIRECTORY),
            "--prompt",
            "def add(a, b):",
            "--max-new-tokens",
            "1",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )

    assert "Generated output" in result.stdout
    assert "def add(a, b):" in result.stdout
