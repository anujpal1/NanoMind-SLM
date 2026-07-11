"""Configuration for the NanoMind Llama-style model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NanoMindConfig:
    """Validated model architecture configuration."""

    vocabulary_size: int = 8000
    context_length: int = 256
    hidden_size: int = 384
    number_of_layers: int = 10
    number_of_heads: int = 6
    number_of_kv_heads: int = 2
    intermediate_size: int = 1024
    rope_theta: float = 10000.0
    rms_norm_epsilon: float = 1e-5
    dropout: float = 0.0
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.vocabulary_size <= 0:
            raise ValueError("vocabulary_size must be positive")

        if self.context_length <= 0:
            raise ValueError("context_length must be positive")

        if self.hidden_size % self.number_of_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by number_of_heads"
            )

        if self.number_of_heads % self.number_of_kv_heads != 0:
            raise ValueError(
                "number_of_heads must be divisible by number_of_kv_heads"
            )

        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be between 0 and 1")

    @property
    def head_dimension(self) -> int:
        return self.hidden_size // self.number_of_heads