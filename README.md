# NanoMind-SLM

NanoMind-SLM is an educational, decoder-only Transformer trained from random
initialization on Python source code. The project covers the complete path from
a revision-pinned streaming dataset through filtering, tokenizer training,
fixed-length shards, model training, checkpointing, greedy inference, and
evaluation.

The released model is deliberately small and limited. It is useful for learning
how a code language model is built; it is not a production coding assistant.

## Released model

| Property | Value |
|---|---:|
| Parameters | 18,808,704 |
| Vocabulary | 8,000-token byte-level BPE |
| Context length | 256 tokens |
| Transformer width | 384 |
| Layers | 10 |
| Attention heads | 6 query / 2 key-value |
| Feed-forward width | 1,024 |
| Components | GQA, RoPE, RMSNorm, SwiGLU, tied embeddings |
| Training source | `codeparrot/codeparrot-clean` |
| Checkpoint size | 75,265,675 bytes (75.3 MB / 71.8 MiB) |

The checkpoint, tokenizer, release configuration, and saved reports are hosted
in [NanoMind-SLM-60M on Hugging Face](https://huggingface.co/Anujpal01/NanoMind-SLM-60M).
“60M” describes the roughly 62.5-million-token training corpus, not the model's
parameter count.

### Reported release results

| Measurement | Reported value | What it means |
|---|---:|---|
| Packed training corpus | 62,531,072 tokens | Tokens stored in complete 256-token blocks |
| Nominal full-batch schedule | 7,634 optimizer steps | 62,537,728 positions at batch 4 × accumulation 8 × context 256 |
| Packed validation corpus | 6,277,632 tokens | The release-time prepared validation data |
| Validation loss | 2.156258 | Saved historical result over 500 batches |
| Perplexity | 8.638752 | `exp(validation loss)` |
| Evaluated validation tokens | 1,024,000 | Saved historical result |
| AST-parseable saved outputs | 5/10 | `ast.parse` accepts five saved generations |
| Compile-valid saved outputs | 5/10 | `compile(..., "exec")` accepts the same five saved generations |

The historical values are preserved unchanged in `reports/final_metrics.json`
and `reports/syntax_results.json`. The older syntax report uses the ambiguous
field name `syntax_valid`; it actually records `ast.parse` acceptance. A new
post-hoc compile check of those preserved outputs also accepts 5/10. Compilation
uses `compile(source, "<generated>", "exec")` and never executes generated code.
Neither AST parsing nor compilation measures functional correctness, pass@1, or
coding success.

The exact release-time validation corpus, shard hashes, and evaluation command
were not retained, so the historical loss and perplexity cannot be claimed as
freshly reproduced. A 2026-09-20 rebuild with the current pinned pipeline and
released tokenizer produced 6,289,664 packed validation tokens, rather than the
historical 6,277,632. A new finite 500-batch CPU evaluation on that rebuilt input
measured loss **2.159504** and perplexity **8.666837** over 4,000 sequences and
1,020,000 predicted tokens. This is a separate measurement, not a reproduction
or replacement of the historical 2.156258 loss. The
[pinned-rebuild report](reports/pinned_rebuild_evaluation_bos_2026-09-20.json)
records the configuration and input hashes.

For example, this saved completion parses but is not a correct implementation:

```python
def is_even(n):
    return n.is_even(n)
```

Another saved completion starts an unterminated docstring and does not parse.
The negative examples remain in the repository because they are important
evidence of the model's limitations.

### Fresh checkpoint verification

On 2026-09-20, the published checkpoint was loaded on CPU with Python 3.14.0
and PyTorch 2.13.0, then run on the same ten prompts with greedy decoding and a
96-token limit:

| Inference policy | AST-parseable | Compile-valid |
|---|---:|---:|
| Current default, with `<bos>` | 8/10 | 7/10 |
| A/B variant, without `<bos>` | 6/10 | 4/10 |

`ast.parse` builds an abstract syntax tree, while `compile(..., "exec")` applies
additional compiler checks such as rejecting duplicate function arguments.
Neither check executes the generated code. The corrected
[BOS report](reports/code_validity_verification_bos_2026-09-20.json) and
[no-BOS report](reports/code_validity_verification_no_bos_2026-09-20.json)
record both measurements, artifact hashes, runtime, decoding policy, and full
outputs. The original `syntax_verification_*.json` files remain unchanged as
historical AST-only reports.

## Run the released checkpoint

The project supports Python 3.11–3.14 and uses
[uv](https://docs.astral.sh/uv/). From a fresh clone:

```console
uv sync --locked
uv run --with huggingface-hub hf download Anujpal01/NanoMind-SLM-60M --revision e748230aaef5e660221758c2c04330bd17be32a2 --local-dir release/NanoMind-SLM-60M
uv run python scripts/generate.py --model-dir release/NanoMind-SLM-60M --prompt "def add(a, b):" --max-new-tokens 32
```

Inference is greedy and deterministic for a fixed software/hardware environment.
It prepends `<bos>`, stops at `<eos>`, and loads the state-dictionary checkpoint
with PyTorch's restricted `weights_only=True` loader.

## Quick verification

The smoke command uses deterministic synthetic tokens; it does not require a
dataset download or prebuilt shards:

```console
uv run python -m nanomind_slm.training.train --smoke
```

Run the fixed ten-prompt source-validity evaluation against the downloaded
release:

```console
uv run python scripts/evaluate.py --checkpoint release/NanoMind-SLM-60M/model.pt --model-config release/NanoMind-SLM-60M/model.yaml --tokenizer release/NanoMind-SLM-60M/tokenizer/tokenizer.json --device cpu
```

Validation loss additionally requires `--validation-manifest` pointing to
prepared shards produced by the same tokenizer. New manifests record tokenizer,
corpus, and shard SHA-256 values, and evaluation rejects a recorded tokenizer
mismatch. The small ignored shards in the original development folder use an
earlier tokenizer and are not valid release-checkpoint evaluation data.

## Data-to-demo flow

```text
revision-pinned CodeParrot stream
  → bounded metadata/license filtering and repository-level split
  → separate baseline and quality byte-level BPE tokenizers
  → packed 256-token NumPy shards
  → Llama-style causal-language-model training
  → state-dictionary checkpoint
  → greedy generation and finite evaluation
```

The active commands are:

```console
uv run python scripts/audit_dataset.py
uv run python -m nanomind_slm.data.build_tokenizer_corpora
uv run python -m nanomind_slm.data.train_tokenizers
uv run python -m nanomind_slm.data.build_tokenized_data
uv run python -m nanomind_slm.training.train
```

`configs/training.yaml` now matches the published final-run schedule. Earlier
10,000-step exploratory settings remain visible in Git history rather than being
presented as the release reproduction default.

## Important limitations and provenance

- The implemented “quality” filter checks license, content length,
  autogenerated metadata, and longest-line length. It does **not** call
  `ast.parse`; the released corpus must not be described as AST-validated.
- The published corpus writer separated files with blank lines, while the
  tokenizer-shard builder treated every blank line as a document boundary.
  Internal blank lines could therefore add `<eos>` inside a source file. This
  historical behavior is retained for checkpoint provenance. Correcting it
  requires a versioned preprocessing pipeline, retraining, and reevaluation.
- Baseline and quality experiments used separately trained tokenizers and did
  not have a demonstrated equal training budget. Their saved 1/10, 0/10, and
  5/10 syntax observations are historical notes, not a controlled proof that
  filtering caused the difference.
- `<bos>` was defined by tokenizer training but was not inserted in the packed
  training sequences; the released inference demo prepends it. The recorded
  96-token CPU A/B produced 8/10 AST-parseable and 7/10 compile-valid outputs
  with `<bos>`, versus 6/10 and 4/10 without it. This does not establish semantic
  or functional correctness.
- Checkpoints restore the model, optimizer, scaler, and completed step, but not
  the exact shuffled-sampler/RNG position. Resume is practical, not bit-exact.
- Current periodic validation restarts from the same deterministic validation
  prefix. The released run's older loop advanced through successive validation
  batches, so its intermediate validation points should not be compared as if
  they used an identical subset.
- No Qwen comparison, MBPP evaluation, functional-correctness benchmark, or
  controlled filtering-only ablation was completed.
- Neither this repository nor the published model currently declares a license.
  Do not describe either artifact as licensed for open-source reuse until an
  appropriate license is deliberately selected and added.

See `docs/dataset_plan.md` for detailed data provenance and the boundary-format
decision. A publish-ready corrected Hugging Face description is in
`docs/huggingface_model_card.md`; the downloaded release directory is a snapshot,
not a writable model-card checkout. The Git history and `v1.0.0` tag preserve
the real development milestones rather than replacing them with a cleaned-up
origin story.
