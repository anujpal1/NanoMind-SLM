"""Complete NanoMind Llama-style decoder-only Transformer."""

from torch import Tensor, nn

from nanomind_slm.model.attention import GroupedQueryAttention
from nanomind_slm.model.config import NanoMindConfig
from nanomind_slm.model.layers import RMSNorm, SwiGLU


class TransformerBlock(nn.Module):
    """Pre-normalized Llama-style Transformer block."""

    def __init__(self, config: NanoMindConfig) -> None:
        super().__init__()

        self.attention_norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_epsilon,
        )
        self.attention = GroupedQueryAttention(config)

        self.feed_forward_norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_epsilon,
        )
        self.feed_forward = SwiGLU(
            config.hidden_size,
            config.intermediate_size,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = hidden_states + self.attention(
            self.attention_norm(hidden_states)
        )
        hidden_states = hidden_states + self.feed_forward(
            self.feed_forward_norm(hidden_states)
        )
        return hidden_states


class NanoMindModel(nn.Module):
    """NanoMind decoder-only code language model."""

    def __init__(self, config: NanoMindConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocabulary_size,
            config.hidden_size,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.number_of_layers)
            ]
        )
        self.final_norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_epsilon,
        )
        self.output_projection = nn.Linear(
            config.hidden_size,
            config.vocabulary_size,
            bias=False,
        )

        self.apply(self._initialize_weights)

        if config.tie_word_embeddings:
            self.output_projection.weight = (
                self.token_embedding.weight
            )

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape [batch, sequence]"
            )

        if input_ids.shape[1] > self.config.context_length:
            raise ValueError(
                "Input exceeds configured context length"
            )

        hidden_states = self.token_embedding(input_ids)

        for block in self.blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.final_norm(hidden_states)
        return self.output_projection(hidden_states)

    def parameter_count(self) -> int:
        """Return the number of trainable model parameters."""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )