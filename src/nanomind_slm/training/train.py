"""Resumable NanoMind training loop for local or cloud GPUs."""

import argparse
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader

from nanomind_slm.data.token_dataset import TokenShardDataset
from nanomind_slm.model import NanoMindConfig, NanoMindModel
from nanomind_slm.training.loss import causal_language_model_loss
from nanomind_slm.training.schedule import cosine_learning_rate


def infinite_batches(loader: DataLoader[Tensor]) -> Iterator[Tensor]:
    """Repeat the DataLoader for as many steps as required."""
    while True:
        yield from loader


def automatic_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_latest_checkpoint(
    *,
    model: NanoMindModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    directory: Path,
    device: torch.device,
) -> int:
    checkpoints = sorted(directory.glob("step_*.pt"))

    if not checkpoints:
        return 0

    checkpoint_path = checkpoints[-1]
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])

    completed_step = int(checkpoint["step"])
    print(f"Resumed from {checkpoint_path}")

    return completed_step


def save_checkpoint(
    *,
    model: NanoMindModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    directory: Path,
    step: int,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / f"step_{step:08d}.pt"

    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        },
        checkpoint_path,
    )
    print(f"Saved {checkpoint_path}")


@torch.no_grad()
def validate(
    *,
    model: NanoMindModel,
    batches: Iterator[Tensor],
    device: torch.device,
    sequence_length: int,
    number_of_batches: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> float:
    model.eval()
    total_loss = 0.0

    for _ in range(number_of_batches):
        token_ids = next(batches)[:, :sequence_length].to(device)

        context = (
            torch.autocast("cuda", dtype=amp_dtype)
            if use_amp
            else nullcontext()
        )

        with context:
            logits = model(token_ids)
            loss = causal_language_model_loss(logits, token_ids)

        total_loss += loss.item()

    model.train()
    return total_loss / number_of_batches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one small CPU/GPU training step",
    )
    arguments = parser.parse_args()

    with Path("configs/model.yaml").open(encoding="utf-8") as file:
        model_data: dict[str, Any] = yaml.safe_load(file)

    with Path("configs/training.yaml").open(encoding="utf-8") as file:
        training_data: dict[str, Any] = yaml.safe_load(file)

    model_config = NanoMindConfig(**model_data["model"])
    config = training_data["training"]

    torch.manual_seed(config["seed"])
    device = automatic_device()

    experiment = config["experiment"]
    data_root = Path("data/tokenized") / experiment

    batch_size = 1 if arguments.smoke else config["batch_size"]
    sequence_length = (
        64 if arguments.smoke else model_config.context_length
    )
    maximum_steps = 1 if arguments.smoke else config["maximum_steps"]
    accumulation_steps = (
        1
        if arguments.smoke
        else config["gradient_accumulation_steps"]
    )

    generator = torch.Generator().manual_seed(config["seed"])

    train_loader = DataLoader(
        TokenShardDataset(data_root / "train"),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(
        TokenShardDataset(data_root / "validation"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    train_batches = infinite_batches(train_loader)
    validation_batches = infinite_batches(validation_loader)

    model = NanoMindModel(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(config["beta1"], config["beta2"]),
        weight_decay=config["weight_decay"],
    )

    use_amp = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp and amp_dtype == torch.float16,
    )

    checkpoint_directory = Path(config["checkpoint_directory"])
    start_step = 0

    if config["resume_from_latest"] and not arguments.smoke:
        start_step = load_latest_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            directory=checkpoint_directory,
            device=device,
        )

    model.train()

    for step in range(start_step, maximum_steps):
        learning_rate = cosine_learning_rate(
            step,
            maximum_learning_rate=config["learning_rate"],
            minimum_learning_rate=config["minimum_learning_rate"],
            warmup_steps=config["warmup_steps"],
            maximum_steps=config["maximum_steps"],
        )

        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        for _ in range(accumulation_steps):
            token_ids = next(train_batches)
            token_ids = token_ids[:, :sequence_length].to(device)

            context = (
                torch.autocast("cuda", dtype=amp_dtype)
                if use_amp
                else nullcontext()
            )

            with context:
                logits = model(token_ids)
                loss = causal_language_model_loss(
                    logits,
                    token_ids,
                )
                scaled_loss = loss / accumulation_steps

            scaler.scale(scaled_loss).backward()
            accumulated_loss += loss.item()

        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config["gradient_clip_norm"],
        )
        scaler.step(optimizer)
        scaler.update()

        completed_step = step + 1
        average_loss = accumulated_loss / accumulation_steps

        print(
            f"step={completed_step} "
            f"loss={average_loss:.4f} "
            f"lr={learning_rate:.8f} "
            f"grad_norm={gradient_norm.item():.4f}"
        )

        should_validate = (
            arguments.smoke
            or completed_step % config["validation_interval"] == 0
        )

        if should_validate:
            validation_loss = validate(
                model=model,
                batches=validation_batches,
                device=device,
                sequence_length=sequence_length,
                number_of_batches=(
                    1
                    if arguments.smoke
                    else config["validation_batches"]
                ),
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
            print(f"validation_loss={validation_loss:.4f}")

        if (
            not arguments.smoke
            and completed_step % config["checkpoint_interval"] == 0
        ):
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                directory=checkpoint_directory,
                step=completed_step,
            )

    print(f"Training run completed on {device}")


if __name__ == "__main__":
    main()