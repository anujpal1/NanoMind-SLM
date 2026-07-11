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


## Final Results

| Metric | Result |
|---|---:|
| Architecture | Llama-style decoder-only Transformer |
| Parameters | 18,808,704 |
| Vocabulary size | 8,000 |
| Context length | 256 |
| Training tokens | 62,531,072 |
| Validation tokens | 6,277,632 |
| Validation loss | 2.1563 |
| Perplexity | 8.64 |
| Syntax validity | 5/10 (50%) |
| Model weights | 71.8 MB |

### Research Comparison

| Model | Training tokens | Syntax validity |
|---|---:|---:|
| Baseline | Approximately 31M | 1/10 |
| Earlier quality model | Approximately 31M | 0/10 |
| Final quality model | 62.5M | **5/10** |

The final model learned recognizable Python structure, but its logical
correctness remains limited. This is expected for an 18.8M-parameter
model trained entirely from random initialization.