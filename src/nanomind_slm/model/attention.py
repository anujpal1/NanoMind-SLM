"""Grouped-query causal self-attention."""

import torch.nn.functional as functional
from torch import Tensor, nn

from nanomind_slm.model.config import NanoMindConfig
from nanomind_slm.model.layers import (
    RotaryEmbedding,
    apply_rotary_position_embedding,
)


class GroupedQueryAttention(nn.Module):
    """Llama-style grouped-query self-attention."""

    def __init__(self, config: NanoMindConfig) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.number_of_heads = config.number_of_heads
        self.number_of_kv_heads = config.number_of_kv_heads
        self.head_dimension = config.head_dimension
        self.context_length = config.context_length
        self.dropout = config.dropout

        kv_size = self.number_of_kv_heads * self.head_dimension

        self.query_projection = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
        )
        self.key_projection = nn.Linear(
            self.hidden_size,
            kv_size,
            bias=False,
        )
        self.value_projection = nn.Linear(
            self.hidden_size,
            kv_size,
            bias=False,
        )
        self.output_projection = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
        )

        self.rotary = RotaryEmbedding(
            self.head_dimension,
            theta=config.rope_theta,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch_size, sequence_length, _ = hidden_states.shape

        if sequence_length > self.context_length:
            raise ValueError("Sequence exceeds configured context length")

        queries = self.query_projection(hidden_states)
        keys = self.key_projection(hidden_states)
        values = self.value_projection(hidden_states)

        queries = queries.reshape(
            batch_size,
            sequence_length,
            self.number_of_heads,
            self.head_dimension,
        ).transpose(1, 2)

        keys = keys.reshape(
            batch_size,
            sequence_length,
            self.number_of_kv_heads,
            self.head_dimension,
        ).transpose(1, 2)

        values = values.reshape(
            batch_size,
            sequence_length,
            self.number_of_kv_heads,
            self.head_dimension,
        ).transpose(1, 2)

        cosine, sine = self.rotary(
            sequence_length,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        queries = apply_rotary_position_embedding(
            queries,
            cosine,
            sine,
        )
        keys = apply_rotary_position_embedding(
            keys,
            cosine,
            sine,
        )

        repeat_count = (
            self.number_of_heads // self.number_of_kv_heads
        )
        keys = keys.repeat_interleave(repeat_count, dim=1)
        values = values.repeat_interleave(repeat_count, dim=1)

        attention_output = functional.scaled_dot_product_attention(
            queries,
            keys,
            values,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        attention_output = attention_output.transpose(
            1,
            2,
        ).contiguous()
        attention_output = attention_output.reshape(
            batch_size,
            sequence_length,
            self.hidden_size,
        )

        return self.output_projection(attention_output)