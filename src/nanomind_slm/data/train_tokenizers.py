"""Train comparable 8K Byte-Level BPE tokenizers."""

import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
VOCAB_SIZE = 8000


def train_tokenizer(experiment: str) -> None:
    corpus_dir = Path("data/tokenizer")
    train_file = corpus_dir / f"{experiment}_train.txt"
    validation_file = corpus_dir / f"{experiment}_validation.txt"

    output_dir = Path("artifacts/tokenizers") / experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=False,
        use_regex=True,
    )
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )

    tokenizer.train(
        files=[str(train_file)],
        trainer=trainer,
    )

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    if bos_id is None or eos_id is None:
        raise RuntimeError("BOS or EOS token is missing")

    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[
            ("<bos>", bos_id),
            ("<eos>", eos_id),
        ],
    )

    tokenizer.save(str(output_dir / "tokenizer.json"))
    tokenizer.model.save(str(output_dir))

    unknown_id = tokenizer.token_to_id("<unk>")
    validation_tokens = 0
    unknown_tokens = 0

    with validation_file.open(encoding="utf-8") as file:
        for line in file:
            encoding = tokenizer.encode(
                line,
                add_special_tokens=False,
            )
            validation_tokens += len(encoding.ids)
            unknown_tokens += encoding.ids.count(unknown_id)

    sample = "def add(a, b):\n    return a + b\n"
    sample_encoding = tokenizer.encode(
        sample,
        add_special_tokens=False,
    )
    decoded_sample = tokenizer.decode(sample_encoding.ids)

    report = {
        "experiment": experiment,
        "algorithm": "byte-level-bpe",
        "vocabulary_size": tokenizer.get_vocab_size(),
        "special_tokens": SPECIAL_TOKENS,
        "validation_tokens": validation_tokens,
        "unknown_tokens": unknown_tokens,
        "sample_round_trip": decoded_sample == sample,
    }

    with (output_dir / "report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(
        f"{experiment}: "
        f"vocab={report['vocabulary_size']} "
        f"validation_tokens={validation_tokens} "
        f"unknown_tokens={unknown_tokens} "
        f"round_trip={report['sample_round_trip']}"
    )


def main() -> None:
    train_tokenizer("baseline")
    train_tokenizer("quality")


if __name__ == "__main__":
    main()