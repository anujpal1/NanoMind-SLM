"""Display hardware and PyTorch runtime information."""

from __future__ import annotations

import platform
import sys

import psutil
import torch


def bytes_to_gib(number_of_bytes: int) -> float:
    """Convert a number of bytes into gibibytes."""
    return number_of_bytes / (1024**3)


def main() -> None:
    """Print information about the current computer and ML runtime."""
    memory = psutil.virtual_memory()

    print("=== System information ===")
    print(f"Operating system: {platform.system()} {platform.release()}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Processor: {platform.processor()}")
    print(f"Logical CPU cores: {psutil.cpu_count(logical=True)}")
    print(f"Physical CPU cores: {psutil.cpu_count(logical=False)}")
    print(f"Total RAM: {bytes_to_gib(memory.total):.2f} GiB")
    print(f"Available RAM: {bytes_to_gib(memory.available):.2f} GiB")

    print("\n=== PyTorch information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Current device: CPU")
        print("Main-model training will use a cloud GPU.")


if __name__ == "__main__":
    main()