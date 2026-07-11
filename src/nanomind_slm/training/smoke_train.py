"""Run one CPU training step to verify the complete pipeline."""

from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from nanomind_slm.data.token_dataset import TokenShardDataset
from nanomind_slm.model import NanoMindConfig, NanoMindModel
from nanomind_slm.training.loss import (
    causal_language_model_loss,
)


def main() -> None:
    with Path("configs/model.yaml").open(encoding="utf-8") as file:
        model_data: dict[str, Any] = yaml.safe_load(file)

    with Path("configs/training.yaml").open(encoding="utf-8") as file:
        training_data: dict[str, Any] = yaml.safe_load(file)

    config = NanoMindConfig(**model_data["model"])
    training = training_data["training"]
    smoke = training_data["smoke_test"]

    torch.manual_seed(training["seed"])

    experiment = training["experiment"]
    dataset = TokenShardDataset(
        Path("data/tokenized") / experiment / "train"
    )

    generator = torch.Generator()
    generator.manual_seed(training["seed"])

    loader = DataLoader(
        dataset,
        batch_size=smoke["batch_size"],
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    device = torch.device(smoke["device"])
    model = NanoMindModel(config).to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        betas=(
            training["beta1"],
            training["beta2"],
        ),
        weight_decay=training["weight_decay"],
    )

    token_ids = next(iter(loader))
    token_ids = token_ids[
        :,
        : smoke["sequence_length"],
    ].to(device)

    optimizer.zero_grad(set_to_none=True)

    logits = model(token_ids)
    loss = causal_language_model_loss(logits, token_ids)

    if not torch.isfinite(loss):
        raise RuntimeError("Smoke-test loss is not finite")

    loss.backward()

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        training["gradient_clip_norm"],
    )
    optimizer.step()

    print(f"Device: {device}")
    print(f"Parameters: {model.parameter_count():,}")
    print(f"Batch shape: {tuple(token_ids.shape)}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Gradient norm: {gradient_norm.item():.4f}")
    print("Smoke training passed")


if __name__ == "__main__":
    main()