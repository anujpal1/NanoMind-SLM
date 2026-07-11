"""Warmup and cosine-decay learning-rate schedule."""

import math


def cosine_learning_rate(
    step: int,
    *,
    maximum_learning_rate: float,
    minimum_learning_rate: float,
    warmup_steps: int,
    maximum_steps: int,
) -> float:
    """Return the learning rate for one optimizer step."""
    if step < 0:
        raise ValueError("step must be non-negative")

    if maximum_steps <= warmup_steps:
        raise ValueError(
            "maximum_steps must be greater than warmup_steps"
        )

    if not 0.0 <= minimum_learning_rate <= maximum_learning_rate:
        raise ValueError("learning-rate limits are invalid")

    if step < warmup_steps:
        return maximum_learning_rate * (
            (step + 1) / warmup_steps
        )

    if step >= maximum_steps:
        return minimum_learning_rate

    progress = (
        (step - warmup_steps)
        / (maximum_steps - warmup_steps)
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

    return minimum_learning_rate + (
        maximum_learning_rate - minimum_learning_rate
    ) * cosine