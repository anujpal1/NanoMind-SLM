"""Small integration checks for the training path."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from nanomind_slm.model import NanoMindConfig, NanoMindModel
from nanomind_slm.training.loss import causal_language_model_loss
from nanomind_slm.training.train import (
    load_latest_checkpoint,
    save_checkpoint,
    validate,
)


def small_config() -> NanoMindConfig:
    return NanoMindConfig(
        vocabulary_size=32,
        context_length=8,
        hidden_size=16,
        number_of_layers=2,
        number_of_heads=4,
        number_of_kv_heads=2,
        intermediate_size=32,
    )


def test_tiny_training_step_updates_parameters() -> None:
    torch.manual_seed(42)
    model = NanoMindModel(small_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    token_ids = torch.randint(0, 32, (2, 8))
    original_weights = model.token_embedding.weight.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    loss = causal_language_model_loss(model(token_ids), token_ids)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert not torch.equal(model.token_embedding.weight, original_weights)


def test_validation_restarts_from_same_held_out_prefix() -> None:
    torch.manual_seed(42)
    model = NanoMindModel(small_config())
    loader = DataLoader(
        torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [8, 7, 6, 5, 4, 3, 2, 1],
            ]
        ),
        batch_size=1,
        shuffle=False,
    )

    first = validate(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
        sequence_length=8,
        number_of_batches=1,
        use_amp=False,
        amp_dtype=torch.float32,
    )
    second = validate(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
        sequence_length=8,
        number_of_batches=1,
        use_amp=False,
        amp_dtype=torch.float32,
    )

    assert first == second


def test_training_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(42)
    model = NanoMindModel(small_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    original_weights = model.token_embedding.weight.detach().clone()

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        directory=tmp_path,
        step=3,
    )
    with torch.no_grad():
        model.token_embedding.weight.add_(1.0)

    completed_step = load_latest_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        directory=tmp_path,
        device=torch.device("cpu"),
    )

    assert completed_step == 3
    assert torch.equal(model.token_embedding.weight, original_weights)
