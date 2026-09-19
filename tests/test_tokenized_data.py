"""Tests for the published corpus-to-token boundary behavior."""

from pathlib import Path

from nanomind_slm.data.build_tokenized_data import iter_documents


def test_published_pipeline_uses_blank_lines_as_boundaries(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "def example():\n"
        "    value = 1\n"
        "\n"
        "    return value\n"
        "\n"
        "class Next:\n"
        "    pass\n",
        encoding="utf-8",
    )

    assert list(iter_documents(corpus)) == [
        "def example():\n    value = 1\n",
        "    return value\n",
        "class Next:\n    pass\n",
    ]
