"""Pack token sequences into fixed-length language-model blocks."""

from collections.abc import Iterable, Iterator, Sequence


def pack_token_sequences(
    sequences: Iterable[Sequence[int]],
    *,
    sequence_length: int,
    eos_token_id: int,
) -> Iterator[list[int]]:
    """Yield fixed-length blocks while keeping memory bounded."""
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")

    if eos_token_id < 0:
        raise ValueError("eos_token_id must be non-negative")

    buffer: list[int] = []

    for sequence in sequences:
        for token_id in sequence:
            if not isinstance(token_id, int) or token_id < 0:
                raise ValueError("token IDs must be non-negative integers")

        buffer.extend(sequence)
        buffer.append(eos_token_id)

        while len(buffer) >= sequence_length:
            yield buffer[:sequence_length]
            del buffer[:sequence_length]