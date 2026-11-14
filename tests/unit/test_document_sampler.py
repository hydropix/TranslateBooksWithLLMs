"""
Unit tests for src.utils.document_sampler.

Covers the distributed-sampling contract (clamping, non-overlap, idempotency)
and the upload extraction entry point (unsupported extensions, EPUB fixture).
"""
import sys
from pathlib import Path


# Make the project importable regardless of where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.document_sampler import (
    extract_samples_from_upload,
    take_distributed_samples,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def test_short_text_returned_untouched():
    text = "a short text"
    joined, count = take_distributed_samples(text, total_budget=1000, num_samples=5)
    assert joined == text
    assert count == 1


def test_many_samples_stay_within_budget_plus_separators():
    text = "x" * 100_000
    joined, count = take_distributed_samples(
        text, total_budget=6000, num_samples=10, min_sample_size=500
    )
    assert count == 10
    separator = "\n\n[…]\n\n"
    max_len = 6000 + separator.__len__() * (count - 1)
    assert len(joined) <= max_len


def test_num_samples_clamped_to_12():
    text = "y" * 100_000
    joined, count = take_distributed_samples(
        text, total_budget=6000, num_samples=50, min_sample_size=500
    )
    assert count == 12


def test_min_sample_size_clamps_to_8():
    text = "z" * 100_000
    joined, count = take_distributed_samples(
        text, total_budget=10_000, num_samples=20, min_sample_size=1200
    )
    assert count == 8


def test_pieces_do_not_overlap():
    # Marker-seeded text: each 10-char block encodes its own start offset,
    # so we can verify no two returned pieces share source characters.
    text = "".join(f"{i:010d}" for i in range(10_000))
    separator = "\n\n[…]\n\n"
    joined, count = take_distributed_samples(
        text, total_budget=6000, num_samples=10, min_sample_size=500, separator=separator
    )
    assert count > 1
    pieces = joined.split(separator)
    assert len(pieces) == count

    # Every piece must be a contiguous substring of the source text, and
    # successive pieces' locations in the source must not overlap.
    search_from = 0
    for piece in pieces:
        idx = text.find(piece, search_from)
        assert idx != -1, "piece not found in source text after previous piece"
        search_from = idx + len(piece)


def test_idempotent_no_randomness():
    text = "abcdefgh" * 20_000
    result_a = take_distributed_samples(text, total_budget=6000, num_samples=10)
    result_b = take_distributed_samples(text, total_budget=6000, num_samples=10)
    assert result_a == result_b


def test_unsupported_extension_returns_none_zero_zero():
    joined, count, full_len = extract_samples_from_upload(
        b"whatever bytes", "notes.odt", max_chars=6000, num_samples=1
    )
    assert (joined, count, full_len) == (None, 0, 0)


def test_epub_fixture_yields_non_empty_text():
    epub_path = FIXTURES_DIR / "translation_sampler.epub"
    data = epub_path.read_bytes()
    joined, count, full_len = extract_samples_from_upload(
        data, "translation_sampler.epub", max_chars=6000, num_samples=3
    )
    assert joined
    assert count >= 1
    assert full_len > 0


def test_txt_fixture_yields_non_empty_text():
    txt_path = FIXTURES_DIR / "sample.txt"
    data = txt_path.read_bytes()
    joined, count, full_len = extract_samples_from_upload(
        data, "sample.txt", max_chars=6000, num_samples=1
    )
    assert joined
    assert count == 1
    assert full_len > 0
