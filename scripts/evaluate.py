"""Reproduce NanoMind validation and Python source-code measurements.

This evaluator is deliberately finite and read-only: it walks the prepared
validation manifest once (or stops at ``--validation-batches``), uses greedy
decoding for a fixed prompt set, and never updates the historical reports.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import platform
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
import yaml
from tokenizers import Tokenizer
from torch import Tensor
from torch.utils.data import DataLoader

from nanomind_slm.data.token_dataset import TokenShardDataset
from nanomind_slm.model import NanoMindConfig, NanoMindModel

DEFAULT_SYNTAX_PROMPTS = (
    "def add(a, b):",
    "def is_even(n):",
    "def factorial(n):",
    "def fibonacci(n):",
    "def is_prime(n):",
    "def reverse_string(text):",
    "def is_palindrome(text):",
    "def find_maximum(values):",
    "class Stack:",
    "class Counter:",
)


def sha256_file(path: Path) -> str:
    """Return a stable content hash for one evaluation input."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_validation_provenance(
    *,
    manifest_path: Path,
    tokenizer_sha256: str,
    allow_unverified: bool = False,
) -> dict[str, Any]:
    """Read validation provenance and check the recorded tokenizer hash."""
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)

    recorded_hash = manifest.get("tokenizer_sha256")
    if recorded_hash is None:
        if not allow_unverified:
            raise ValueError(
                "Validation manifest does not record tokenizer_sha256; rebuild it with "
                "tokenizer provenance or pass --allow-unverified-validation-data"
            )
        compatibility = {
            "status": "unverified",
            "reason": "manifest does not record tokenizer_sha256",
        }
    elif not isinstance(recorded_hash, str):
        raise ValueError("Validation manifest tokenizer_sha256 must be a string")
    elif recorded_hash.lower() != tokenizer_sha256.lower():
        raise ValueError(
            "Validation manifest tokenizer_sha256 does not match the supplied tokenizer"
        )
    else:
        compatibility = {
            "status": "verified",
            "tokenizer_sha256": recorded_hash.lower(),
        }

    return {
        "experiment": manifest.get("experiment"),
        "split": manifest.get("split"),
        "context_length": manifest.get("context_length"),
        "declared_total_blocks": manifest.get("total_blocks"),
        "declared_total_tokens": manifest.get("total_tokens"),
        "tokenizer_file": manifest.get("tokenizer_file"),
        "corpus_file": manifest.get("corpus_file"),
        "corpus_sha256": manifest.get("corpus_sha256"),
        "corpus_format": manifest.get("corpus_format"),
        "shard_sha256": manifest.get("shard_sha256"),
        "tokenizer_compatibility": compatibility,
    }


def load_model(
    *,
    checkpoint_path: Path,
    model_config_path: Path,
    device: torch.device,
) -> tuple[NanoMindModel, NanoMindConfig]:
    """Load a state-dictionary checkpoint without enabling pickle execution."""
    with model_config_path.open(encoding="utf-8") as file:
        config_document = yaml.safe_load(file)

    if not isinstance(config_document, Mapping) or not isinstance(
        config_document.get("model"), Mapping
    ):
        raise ValueError("Model configuration must contain a 'model' mapping")

    config = NanoMindConfig(**dict(config_document["model"]))

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if isinstance(checkpoint, Mapping) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("Checkpoint must be a state dictionary or contain 'model'")
    if not all(
        isinstance(name, str) and isinstance(value, Tensor) for name, value in state_dict.items()
    ):
        raise ValueError("Checkpoint model state must contain only named tensors")

    model = NanoMindModel(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config


@torch.inference_mode()
def evaluate_validation(
    *,
    model: NanoMindModel,
    manifest_path: Path,
    device: torch.device,
    batch_size: int,
    maximum_batches: int | None,
) -> dict[str, int | float]:
    """Compute token-weighted loss and perplexity over a finite manifest."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if maximum_batches is not None and maximum_batches <= 0:
        raise ValueError("maximum_batches must be positive when provided")
    if manifest_path.name != "manifest.json":
        raise ValueError("validation manifest must be named manifest.json")

    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("split", "validation") != "validation":
        raise ValueError("Manifest is not a validation split")

    model.eval()
    dataset = TokenShardDataset(manifest_path.parent)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    total_loss = 0.0
    evaluated_batches = 0
    evaluated_sequences = 0
    input_tokens = 0
    predicted_tokens = 0

    for batch_index, token_ids in enumerate(loader):
        if maximum_batches is not None and batch_index >= maximum_batches:
            break

        token_ids = token_ids.to(device)
        logits = model(token_ids)
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_targets = token_ids[:, 1:].contiguous()
        batch_predicted_tokens = shifted_targets.numel()

        loss_sum = functional.cross_entropy(
            shifted_logits.view(-1, shifted_logits.shape[-1]),
            shifted_targets.view(-1),
            reduction="sum",
        )

        total_loss += float(loss_sum.item())
        evaluated_batches += 1
        evaluated_sequences += token_ids.shape[0]
        input_tokens += token_ids.numel()
        predicted_tokens += batch_predicted_tokens

    if predicted_tokens == 0:
        raise ValueError("Validation manifest contains no predictable tokens")

    mean_loss = total_loss / predicted_tokens
    return {
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "evaluated_batches": evaluated_batches,
        "evaluated_sequences": evaluated_sequences,
        "input_tokens": input_tokens,
        "predicted_tokens": predicted_tokens,
    }


@torch.inference_mode()
def generate_greedy(
    *,
    prompt: str,
    model: NanoMindModel,
    config: NanoMindConfig,
    tokenizer: Tokenizer,
    device: torch.device,
    maximum_new_tokens: int,
    use_bos: bool,
) -> str:
    """Generate with the release demo's BOS-prefixed greedy policy."""
    if maximum_new_tokens <= 0:
        raise ValueError("maximum_new_tokens must be positive")

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None or (use_bos and bos_id is None):
        raise ValueError("Tokenizer must define the requested BOS/EOS tokens")

    generated = [*prompt_ids]
    if use_bos:
        generated.insert(0, bos_id)
    if not generated:
        raise ValueError("Prompt must encode to at least one token when BOS is disabled")

    for _ in range(maximum_new_tokens):
        input_ids = torch.tensor(
            [generated[-config.context_length :]],
            dtype=torch.long,
            device=device,
        )
        logits = model(input_ids)
        next_token = int(torch.argmax(logits[0, -1]).item())
        if next_token == eos_id:
            break
        generated.append(next_token)

    decoded_ids = generated[1:] if use_bos else generated
    return tokenizer.decode(decoded_ids, skip_special_tokens=True)


def is_ast_parseable(source: str) -> bool:
    """Return whether ``ast.parse`` accepts the complete generated text."""
    try:
        ast.parse(source)
    except SyntaxError:
        return False
    return True


def is_compile_valid(source: str) -> bool:
    """Return whether Python can compile the text, without executing it."""
    try:
        compile(source, "<generated>", "exec")
    except SyntaxError:
        return False
    return True


def evaluate_syntax(
    *,
    model: NanoMindModel,
    config: NanoMindConfig,
    tokenizer: Tokenizer,
    prompts: Sequence[str],
    device: torch.device,
    maximum_new_tokens: int,
    use_bos: bool,
) -> dict[str, Any]:
    """Measure AST parseability and compile validity for generated Python."""
    if not prompts:
        raise ValueError("At least one syntax prompt is required")
    if not all(isinstance(prompt, str) and prompt for prompt in prompts):
        raise ValueError("Syntax prompts must be non-empty strings")

    model.eval()
    results = []
    ast_parseable_count = 0
    compile_valid_count = 0
    for prompt in prompts:
        generated_code = generate_greedy(
            prompt=prompt,
            model=model,
            config=config,
            tokenizer=tokenizer,
            device=device,
            maximum_new_tokens=maximum_new_tokens,
            use_bos=use_bos,
        )
        ast_parseable = is_ast_parseable(generated_code)
        compile_valid = is_compile_valid(generated_code)
        ast_parseable_count += int(ast_parseable)
        compile_valid_count += int(compile_valid)
        results.append(
            {
                "prompt": prompt,
                "generated_code": generated_code,
                "ast_parseable": ast_parseable,
                "compile_valid": compile_valid,
            }
        )

    return {
        "ast_parseable_count": ast_parseable_count,
        "compile_valid_count": compile_valid_count,
        "total": len(prompts),
        "ast_parseable_rate": ast_parseable_count / len(prompts),
        "compile_valid_rate": compile_valid_count / len(prompts),
        "decoding": {
            "strategy": "greedy_argmax",
            "add_special_tokens": False,
            "bos_prefix": use_bos,
            "stop_at_eos": True,
            "maximum_new_tokens": maximum_new_tokens,
            "context_length": config.context_length,
            "context_truncation": "left",
            "source_scope": "prompt_and_generated_text",
        },
        "results": results,
    }


def load_prompts(path: Path | None) -> tuple[list[str], str]:
    """Load a JSON prompt list, or return the versioned built-in prompt set."""
    if path is None:
        return list(DEFAULT_SYNTAX_PROMPTS), "built_in_v1"

    with path.open(encoding="utf-8") as file:
        prompts = json.load(file)
    if not isinstance(prompts, list) or not all(isinstance(item, str) for item in prompts):
        raise ValueError("Prompt JSON must be an array of strings")
    return prompts, str(path)


def resolve_device(requested: str) -> torch.device:
    """Resolve the CLI device choice and fail clearly for unavailable CUDA."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a NanoMind checkpoint on finite validation shards and "
            "fixed Python source-code prompts."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        help="Prepared validation manifest; omit for source-check-only evaluation.",
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--prompts-json",
        type=Path,
        help="JSON string array; defaults to the ten historical prompts.",
    )
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument(
        "--validation-batches",
        type=int,
        help="Maximum finite batches; omit to evaluate the manifest once.",
    )
    parser.add_argument(
        "--allow-unverified-validation-data",
        action="store_true",
        help=(
            "Evaluate a legacy manifest with no tokenizer_sha256. Known hash "
            "mismatches are always rejected."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--no-bos",
        action="store_true",
        help="Run the source-check A/B policy without the default <bos> prefix.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write a new JSON file. Existing files are never overwritten.",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(arguments)
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {args.output}")

    tokenizer_sha256 = sha256_file(args.tokenizer)
    validation_data = None
    validation_result: dict[str, Any] = {
        "status": "skipped",
        "reason": "no validation manifest supplied",
    }
    if args.validation_manifest is not None:
        validation_data = inspect_validation_provenance(
            manifest_path=args.validation_manifest,
            tokenizer_sha256=tokenizer_sha256,
            allow_unverified=args.allow_unverified_validation_data,
        )

    device = resolve_device(args.device)
    model, config = load_model(
        checkpoint_path=args.checkpoint,
        model_config_path=args.model_config,
        device=device,
    )
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    prompts, prompt_source = load_prompts(args.prompts_json)

    artifacts = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "model_config": str(args.model_config),
        "model_config_sha256": sha256_file(args.model_config),
        "tokenizer": str(args.tokenizer),
        "tokenizer_sha256": tokenizer_sha256,
        "prompt_source": prompt_source,
    }
    if args.validation_manifest is not None:
        artifacts["validation_manifest"] = str(args.validation_manifest)
        artifacts["validation_manifest_sha256"] = sha256_file(args.validation_manifest)
        validation_result = {
            "status": "completed",
            **evaluate_validation(
                model=model,
                manifest_path=args.validation_manifest,
                device=device,
                batch_size=args.validation_batch_size,
                maximum_batches=args.validation_batches,
            ),
        }

    result = {
        "artifacts": artifacts,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
        },
        "validation_data": validation_data,
        "validation": validation_result,
        "syntax": evaluate_syntax(
            model=model,
            config=config,
            tokenizer=tokenizer,
            prompts=prompts,
            device=device,
            maximum_new_tokens=args.max_new_tokens,
            use_bos=not args.no_bos,
        ),
    }

    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(serialized, end="")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
