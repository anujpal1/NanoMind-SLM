"""Causal language-model training loss."""

import torch.nn.functional as functional
from torch import Tensor


def causal_language_model_loss(
    logits: Tensor,
    token_ids: Tensor,
) -> Tensor:
    """Predict each token from all previous tokens."""
    if logits.ndim != 3:
        raise ValueError(
            "logits must have shape [batch, sequence, vocabulary]"
        )

    if token_ids.ndim != 2:
        raise ValueError(
            "token_ids must have shape [batch, sequence]"
        )

    if logits.shape[:2] != token_ids.shape:
        raise ValueError(
            "logits and token_ids must share batch and sequence sizes"
        )

    if token_ids.shape[1] < 2:
        raise ValueError("sequence length must be at least 2")

    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_targets = token_ids[:, 1:].contiguous()

    return functional.cross_entropy(
        shifted_logits.view(
            -1,
            shifted_logits.shape[-1],
        ),
        shifted_targets.view(-1),
    )