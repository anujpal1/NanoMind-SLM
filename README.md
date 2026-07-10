# NanoMind-SLM

NanoMind-SLM is a two-week machine-learning research project focused on
building a compact Llama-style decoder-only Transformer for Python code.

The main model will contain approximately 20 million parameters and will be
trained from random initialization using a custom byte-level BPE tokenizer.

## Research questions

1. How much Python syntax, structure and code-generation ability can a compact
   Llama-style model learn with limited data and free-tier GPU compute?
2. Does AST-based filtering and deduplication improve a small code model when
   architecture and training-token budget are controlled?

## Planned experiment

Two models will use the same architecture and training-token budget:

- **Raw model:** trained on minimally filtered Python code.
- **Filtered model:** trained on AST-validated and deduplicated Python code.

Both will be evaluated using loss, syntax validity, runtime behaviour,
selected MBPP problems and an original private benchmark.

Qwen2.5-Coder-0.5B Base will provide a contextual pretrained baseline.
NanoMind-SLM is not expected to outperform it because the baseline is much
larger and has seen substantially more data and compute.

## Current status

Day 1: repository and development-environment setup.