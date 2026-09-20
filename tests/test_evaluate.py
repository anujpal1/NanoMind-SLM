"""Regression tests for the reproducible evaluation command."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch import Tensor, nn

from nanomind_slm.model import NanoMindConfig, NanoMindModel

EVALUATE_SPEC = importlib.util.spec_from_file_location(
    "nanomind_evaluate",
    Path(__file__).parents[1] / "scripts" / "evaluate.py",
)
assert EVALUATE_SPEC is not None and EVALUATE_SPEC.loader is not None
evaluate = importlib.util.module_from_spec(EVALUATE_SPEC)
sys.modules[EVALUATE_SPEC.name] = evaluate
EVALUATE_SPEC.loader.exec_module(evaluate)

GENERATE_SPEC = importlib.util.spec_from_file_location(
    "nanomind_generate",
    Path(__file__).parents[1] / "scripts" / "generate.py",
)
assert GENERATE_SPEC is not None and GENERATE_SPEC.loader is not None
generate = importlib.util.module_from_spec(GENERATE_SPEC)
sys.modules[GENERATE_SPEC.name] = generate
GENERATE_SPEC.loader.exec_module(generate)


class UniformModel(nn.Module):
    """Return a uniform four-token distribution for exact metric checks."""

    def forward(self, token_ids: Tensor) -> Tensor:
        return torch.zeros(
            (*token_ids.shape, 4),
            dtype=torch.float32,
            device=token_ids.device,
        )


class FakeEncoding:
    def __init__(self, token_ids: list[int]) -> None:
        self.ids = token_ids


class FakeTokenizer:
    def encode(self, prompt: str, *, add_special_tokens: bool) -> FakeEncoding:
        assert prompt == "pass"
        assert add_special_tokens is False
        return FakeEncoding([4])

    def token_to_id(self, token: str) -> int | None:
        return {"<bos>": 2, "<eos>": 3}.get(token)

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        assert token_ids == [4, 5]
        return "pass"


class ScriptedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[list[int]] = []

    def forward(self, token_ids: Tensor) -> Tensor:
        self.inputs.append(token_ids[0].tolist())
        next_token = 5 if len(self.inputs) == 1 else 3
        logits = torch.zeros((*token_ids.shape, 6), device=token_ids.device)
        logits[:, -1, next_token] = 1.0
        return logits


def write_manifest(directory: Path, blocks: np.ndarray) -> Path:
    np.save(directory / "shard_00000.npy", blocks)
    manifest = {
        "experiment": "fixture",
        "split": "validation",
        "context_length": int(blocks.shape[1]),
        "dtype": str(blocks.dtype),
        "total_blocks": int(blocks.shape[0]),
        "total_tokens": int(blocks.size),
        "shards": ["shard_00000.npy"],
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def tiny_config() -> NanoMindConfig:
    return NanoMindConfig(
        vocabulary_size=5,
        context_length=4,
        hidden_size=4,
        number_of_layers=1,
        number_of_heads=1,
        number_of_kv_heads=1,
        intermediate_size=8,
    )


def write_tiny_artifacts(directory: Path) -> dict[str, Path]:
    config = tiny_config()
    config_path = directory / "model.yaml"
    config_path.write_text(
        yaml.safe_dump({"model": config.__dict__}, sort_keys=False),
        encoding="utf-8",
    )

    model = NanoMindModel(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.token_embedding.weight[4, 0] = 1.0
        model.token_embedding.weight[3, 0] = 2.0

    checkpoint_path = directory / "model.pt"
    torch.save({"model": model.state_dict(), "step": 1}, checkpoint_path)

    tokenizer = Tokenizer(
        WordLevel(
            vocab={
                "<pad>": 0,
                "<unk>": 1,
                "<bos>": 2,
                "<eos>": 3,
                "pass": 4,
            },
            unk_token="<unk>",
        )
    )
    tokenizer.add_special_tokens(["<pad>", "<unk>", "<bos>", "<eos>"])
    tokenizer_path = directory / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    manifest_path = write_manifest(
        directory,
        np.array([[2, 4, 3, 0], [2, 4, 3, 0]], dtype=np.uint16),
    )
    prompts_path = directory / "prompts.json"
    prompts_path.write_text('["pass"]', encoding="utf-8")
    return {
        "checkpoint": checkpoint_path,
        "config": config_path,
        "manifest": manifest_path,
        "tokenizer": tokenizer_path,
        "prompts": prompts_path,
    }


def test_checkpoint_loader_requests_safe_weights_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tiny_config()
    config_path = tmp_path / "model.yaml"
    config_path.write_text(
        yaml.safe_dump({"model": config.__dict__}),
        encoding="utf-8",
    )
    expected_state = NanoMindModel(config).state_dict()
    calls: list[dict[str, Any]] = []

    def fake_torch_load(path: Path, **kwargs: Any) -> dict[str, Any]:
        calls.append({"path": path, **kwargs})
        return {"model": expected_state, "step": 7}

    monkeypatch.setattr(evaluate.torch, "load", fake_torch_load)
    checkpoint_path = tmp_path / "model.pt"
    model, loaded_config = evaluate.load_model(
        checkpoint_path=checkpoint_path,
        model_config_path=config_path,
        device=torch.device("cpu"),
    )

    assert loaded_config == config
    assert model.parameter_count() == NanoMindModel(config).parameter_count()
    assert calls == [
        {
            "path": checkpoint_path,
            "map_location": "cpu",
            "weights_only": True,
        }
    ]


def test_validation_metrics_are_weighted_by_predicted_tokens(tmp_path: Path) -> None:
    manifest_path = write_manifest(
        tmp_path,
        np.array(
            [
                [0, 1, 2, 3],
                [3, 2, 1, 0],
                [1, 2, 3, 0],
            ],
            dtype=np.uint16,
        ),
    )

    metrics = evaluate.evaluate_validation(
        model=UniformModel(),  # type: ignore[arg-type]
        manifest_path=manifest_path,
        device=torch.device("cpu"),
        batch_size=2,
        maximum_batches=None,
    )

    assert metrics["loss"] == pytest.approx(np.log(4.0))
    assert metrics["perplexity"] == pytest.approx(4.0)
    assert metrics["evaluated_batches"] == 2
    assert metrics["evaluated_sequences"] == 3
    assert metrics["input_tokens"] == 12
    assert metrics["predicted_tokens"] == 9


@pytest.mark.parametrize(
    ("use_bos", "expected_inputs"),
    [
        (True, [[2, 4], [2, 4, 5]]),
        (False, [[4], [4, 5]]),
    ],
)
def test_greedy_generation_records_bos_policy_and_stops_before_eos(
    use_bos: bool,
    expected_inputs: list[list[int]],
) -> None:
    model = ScriptedModel()

    generated = evaluate.generate_greedy(
        prompt="pass",
        model=model,  # type: ignore[arg-type]
        config=SimpleNamespace(context_length=4),  # type: ignore[arg-type]
        tokenizer=FakeTokenizer(),  # type: ignore[arg-type]
        device=torch.device("cpu"),
        maximum_new_tokens=3,
        use_bos=use_bos,
    )

    assert generated == "pass"
    assert model.inputs == expected_inputs


def test_release_demo_and_evaluator_generation_stay_in_parity() -> None:
    demo_model = ScriptedModel()
    evaluator_model = ScriptedModel()
    config = SimpleNamespace(context_length=4)
    tokenizer = FakeTokenizer()

    demo_output = generate.generate(
        prompt="pass",
        model=demo_model,  # type: ignore[arg-type]
        config=config,  # type: ignore[arg-type]
        tokenizer=tokenizer,  # type: ignore[arg-type]
        device=torch.device("cpu"),
        max_new_tokens=3,
    )
    evaluator_output = evaluate.generate_greedy(
        prompt="pass",
        model=evaluator_model,  # type: ignore[arg-type]
        config=config,  # type: ignore[arg-type]
        tokenizer=tokenizer,  # type: ignore[arg-type]
        device=torch.device("cpu"),
        maximum_new_tokens=3,
        use_bos=True,
    )

    assert demo_output == evaluator_output == "pass"
    assert demo_model.inputs == evaluator_model.inputs


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("def valid():\n    pass\n", True),
        ("def invalid(:\n    pass\n", False),
    ],
)
def test_ast_parseability_uses_python_parser(source: str, expected: bool) -> None:
    assert evaluate.is_ast_parseable(source) is expected


def test_compile_rejects_duplicate_arguments_accepted_by_ast_parser() -> None:
    source = "def duplicate(value, value):\n    return value\n"

    assert evaluate.is_ast_parseable(source) is True
    assert evaluate.is_compile_valid(source) is False


def test_manifest_rejects_recorded_tokenizer_mismatch(tmp_path: Path) -> None:
    manifest_path = write_manifest(
        tmp_path,
        np.array([[0, 1, 2, 3]], dtype=np.uint16),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tokenizer_sha256"] = "a" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        evaluate.inspect_validation_provenance(
            manifest_path=manifest_path,
            tokenizer_sha256="b" * 64,
            allow_unverified=True,
        )


def test_manifest_requires_opt_in_when_tokenizer_hash_is_missing(tmp_path: Path) -> None:
    manifest_path = write_manifest(
        tmp_path,
        np.array([[0, 1, 2, 3]], dtype=np.uint16),
    )

    with pytest.raises(ValueError, match="allow-unverified-validation-data"):
        evaluate.inspect_validation_provenance(
            manifest_path=manifest_path,
            tokenizer_sha256="b" * 64,
        )

    provenance = evaluate.inspect_validation_provenance(
        manifest_path=manifest_path,
        tokenizer_sha256="b" * 64,
        allow_unverified=True,
    )
    assert provenance["tokenizer_compatibility"]["status"] == "unverified"


def test_manifest_provenance_includes_recorded_input_hashes(tmp_path: Path) -> None:
    manifest_path = write_manifest(
        tmp_path,
        np.array([[0, 1, 2, 3]], dtype=np.uint16),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "tokenizer_sha256": "a" * 64,
            "tokenizer_file": "tokenizer.json",
            "corpus_file": "validation.txt",
            "corpus_sha256": "b" * 64,
            "corpus_format": "test-format-v1",
            "shard_sha256": {"shard_00000.npy": "c" * 64},
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    provenance = evaluate.inspect_validation_provenance(
        manifest_path=manifest_path,
        tokenizer_sha256="a" * 64,
    )

    assert provenance["tokenizer_file"] == "tokenizer.json"
    assert provenance["corpus_file"] == "validation.txt"
    assert provenance["corpus_sha256"] == "b" * 64
    assert provenance["corpus_format"] == "test-format-v1"
    assert provenance["shard_sha256"] == {"shard_00000.npy": "c" * 64}


def test_cli_runs_both_evaluations_on_tiny_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = write_tiny_artifacts(tmp_path)

    evaluate.main(
        [
            "--checkpoint",
            str(paths["checkpoint"]),
            "--model-config",
            str(paths["config"]),
            "--validation-manifest",
            str(paths["manifest"]),
            "--allow-unverified-validation-data",
            "--tokenizer",
            str(paths["tokenizer"]),
            "--prompts-json",
            str(paths["prompts"]),
            "--validation-batch-size",
            "2",
            "--max-new-tokens",
            "2",
            "--device",
            "cpu",
        ]
    )

    result = json.loads(capsys.readouterr().out)
    assert result["validation_data"]["tokenizer_compatibility"] == {
        "status": "unverified",
        "reason": "manifest does not record tokenizer_sha256",
    }
    assert result["validation"]["evaluated_sequences"] == 2
    assert result["validation"]["status"] == "completed"
    assert result["validation"]["input_tokens"] == 8
    assert result["syntax"]["ast_parseable_count"] == 1
    assert result["syntax"]["compile_valid_count"] == 1
    assert result["syntax"]["total"] == 1
    assert result["syntax"]["ast_parseable_rate"] == 1.0
    assert result["syntax"]["compile_valid_rate"] == 1.0
    assert result["syntax"]["results"] == [
        {
            "prompt": "pass",
            "generated_code": "pass",
            "ast_parseable": True,
            "compile_valid": True,
        }
    ]
    assert result["syntax"]["decoding"] == {
        "strategy": "greedy_argmax",
        "add_special_tokens": False,
        "bos_prefix": True,
        "stop_at_eos": True,
        "maximum_new_tokens": 2,
        "context_length": 4,
        "context_truncation": "left",
        "source_scope": "prompt_and_generated_text",
    }


def test_cli_can_run_syntax_only_without_a_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = write_tiny_artifacts(tmp_path)

    evaluate.main(
        [
            "--checkpoint",
            str(paths["checkpoint"]),
            "--model-config",
            str(paths["config"]),
            "--tokenizer",
            str(paths["tokenizer"]),
            "--prompts-json",
            str(paths["prompts"]),
            "--max-new-tokens",
            "2",
            "--no-bos",
            "--device",
            "cpu",
        ]
    )

    result = json.loads(capsys.readouterr().out)
    assert result["validation_data"] is None
    assert result["validation"] == {
        "status": "skipped",
        "reason": "no validation manifest supplied",
    }
    assert result["syntax"]["decoding"]["bos_prefix"] is False


def test_cli_refuses_to_overwrite_results(tmp_path: Path) -> None:
    output_path = tmp_path / "existing.json"
    output_path.write_text("historical", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        evaluate.main(
            [
                "--checkpoint",
                "unused.pt",
                "--model-config",
                "unused.yaml",
                "--validation-manifest",
                "manifest.json",
                "--tokenizer",
                "tokenizer.json",
                "--output",
                str(output_path),
            ]
        )

    assert output_path.read_text(encoding="utf-8") == "historical"


RELEASE_DIRECTORY = Path("release/NanoMind-SLM-60M")


@pytest.mark.skipif(
    not (RELEASE_DIRECTORY / "model.pt").is_file(),
    reason="ignored release checkpoint is not available locally",
)
def test_local_release_checkpoint_loads_with_repository_model() -> None:
    model, config = evaluate.load_model(
        checkpoint_path=RELEASE_DIRECTORY / "model.pt",
        model_config_path=RELEASE_DIRECTORY / "model.yaml",
        device=torch.device("cpu"),
    )

    assert config.vocabulary_size == 8000
    assert model.parameter_count() == 18_808_704
