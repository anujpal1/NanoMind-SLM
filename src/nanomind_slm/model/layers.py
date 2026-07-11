"""Core layers used by the NanoMind Llama-style model."""

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root mean square normalization."""

    def __init__(
        self,
        hidden_size: int,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()

        variance = hidden_states.pow(2).mean(
            dim=-1,
            keepdim=True,
        )
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.epsilon
        )

        return (hidden_states * self.weight.float()).to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Generate rotary position cosine and sine values."""

    def __init__(
        self,
        head_dimension: int,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()

        if head_dimension % 2 != 0:
            raise ValueError("head_dimension must be even")

        inverse_frequency = 1.0 / (
            theta
            ** (
                torch.arange(0, head_dimension, 2).float()
                / head_dimension
            )
        )
        self.register_buffer(
            "inverse_frequency",
            inverse_frequency,
            persistent=False,
        )

    def forward(
        self,
        sequence_length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        positions = torch.arange(
            sequence_length,
            device=device,
            dtype=torch.float32,
        )
        frequencies = torch.outer(
            positions,
            self.inverse_frequency.float(),
        )
        embeddings = torch.cat(
            (frequencies, frequencies),
            dim=-1,
        )

        cosine = embeddings.cos().to(dtype)[None, None, :, :]
        sine = embeddings.sin().to(dtype)[None, None, :, :]

        return cosine, sine


def rotate_half(hidden_states: Tensor) -> Tensor:
    """Rotate the final tensor dimension by half."""
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary_position_embedding(
    hidden_states: Tensor,
    cosine: Tensor,
    sine: Tensor,
) -> Tensor:
    """Apply rotary position information."""
    return (hidden_states * cosine) + (
        rotate_half(hidden_states) * sine
    )


class SwiGLU(nn.Module):
    """Llama-style gated feed-forward network."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()
        self.gate_projection = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )
        self.up_projection = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )
        self.down_projection = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        gated = functional.silu(
            self.gate_projection(hidden_states)
        )
        values = self.up_projection(hidden_states)
        return self.down_projection(gated * values)