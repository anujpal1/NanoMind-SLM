"""Generate Python code using the trained NanoMind-SLM model."""

import argparse
from pathlib import Path

import torch
import yaml
from tokenizers import Tokenizer

from nanomind_slm.model import NanoMindConfig, NanoMindModel


def load_model(model_directory: Path, device: torch.device):
    """Load the model, configuration, and tokenizer."""
    config_data = yaml.safe_load(
        (model_directory / "model.yaml").read_text()
    )
    config = NanoMindConfig(**config_data["model"])

    model = NanoMindModel(config).to(device)

    checkpoint = torch.load(
        model_directory / "model.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tokenizer = Tokenizer.from_file(
        str(model_directory / "tokenizer" / "tokenizer.json")
    )

    return model, config, tokenizer


@torch.inference_mode()
def generate(
    prompt: str,
    model: NanoMindModel,
    config: NanoMindConfig,
    tokenizer: Tokenizer,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    """Generate text using deterministic greedy decoding."""
    prompt_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
    ).ids

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    generated = [bos_id, *prompt_ids]

    for _ in range(max_new_tokens):
        input_ids = torch.tensor(
            [generated[-config.context_length :]],
            dtype=torch.long,
            device=device,
        )

        logits = model(input_ids)
        next_token = int(torch.argmax(logits[0, -1]).item())

        if next_token == eos_id:
            break

        generated.append(next_token)

    return tokenizer.decode(
        generated[1:],
        skip_special_tokens=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate code with NanoMind-SLM."
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Starting text for generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=96,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("release/NanoMind-SLM-60M"),
        help="Directory containing the released model.",
    )
    args = parser.parse_args()

    if not args.model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {args.model_dir}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Loading model on {device}...")

    model, config, tokenizer = load_model(
        args.model_dir,
        device,
    )

    generated_code = generate(
        prompt=args.prompt,
        model=model,
        config=config,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )

    print("\nGenerated output")
    print("=" * 70)
    print(generated_code)


if __name__ == "__main__":
    main()