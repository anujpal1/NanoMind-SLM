# NanoMind-SLM Research Log

This log records research decisions, hypotheses, configurations, errors,
changes and measured results throughout the project.

## Project start

- Date: 10 July 2026
- Planned duration: 14 days
- Primary platform: Lightning AI
- Backup platforms: Kaggle and Google Colab
- Local hardware: Intel i5-12500H, 15.70 GiB RAM detected; 3.51 GiB available during the Day 1 check, no dedicated GPU
- Local Python version: Python 3.14
- Package manager: uv
- Training approach: random initialization
- Initial programming language: Python

## Primary hypothesis

After training from random initialization on 30–75 million Python-code
tokens, a Llama-style model with approximately 20 million parameters will
learn measurable Python syntax and code structure.

Evidence for learning will include:

- lower validation loss than the initial untrained model;
- improved Python syntax-validity rate;
- more complete functions;
- less meaningless repetition;
- some successful solutions on selected beginner-level programming tasks.

The model is not expected to outperform Qwen2.5-Coder-0.5B because Qwen
has substantially more parameters, training data and training compute.

## Data-quality hypothesis

When architecture, initialization, optimizer, sequence length and training-token
budget are held constant, training on AST-validated and deduplicated Python code
is expected to produce:

- higher syntax-validity rates;
- fewer repeated or malformed completions;
- fewer runtime errors;
- better pass@1 on selected programming tasks;

compared with training on minimally filtered raw Python code.

Validation loss may or may not be lower, so it will be measured rather than assumed.

## Day 1 decisions

1. Use Python 3.14 initially and fall back only if a confirmed dependency issue occurs.
2. Use uv for Python, environment and dependency management.
3. Use a `src`-based Python package layout.
4. Keep large datasets and checkpoints outside Git.
5. Use streaming or chunked data processing to protect local memory.


## Day 1 issues and resolutions

### NumPy compatibility conflict

- Initial dependency: NumPy 2.5.1
- Problem: NumPy 2.5.1 requires Python 3.12 or newer, but the project supports Python 3.11–3.14.
- Resolution: Changed the requirement to `numpy>=2.4,<2.5`.
- Installed version after resolution: NumPy 2.4.6
- Validation: Dependency locking, seed tests and Ruff checks passed.