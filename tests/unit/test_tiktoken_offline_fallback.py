"""Token counting must keep working when the tiktoken encoding cannot load (#339).

tiktoken downloads cl100k_base on first use. Offline, behind a firewall, or in
a build without the bundled file, that download fails; chunking used to crash
with it ("Failed to prepare samples: ... cl100k_base.tiktoken").
"""
import pytest

from src.core.chunking import token_chunker
from src.core.chunking.token_chunker import (
    TokenChunker,
    approximate_token_count,
    count_tokens,
)


@pytest.fixture
def offline_tiktoken(monkeypatch):
    """Simulate a tiktoken that cannot fetch its encoding file."""
    import tiktoken

    calls = []

    def failing_get_encoding(name):
        calls.append(name)
        raise ConnectionError("WinError 10013: socket access forbidden")

    monkeypatch.setattr(tiktoken, "get_encoding", failing_get_encoding)
    monkeypatch.setattr(token_chunker, "_encoder", None)
    monkeypatch.setattr(token_chunker, "_encoder_failed", False)
    return calls


def test_count_tokens_falls_back_when_encoding_unavailable(offline_tiktoken):
    assert count_tokens("Hello world, this is a test.") > 0
    assert count_tokens("") == 0


def test_failed_load_is_attempted_only_once(offline_tiktoken):
    count_tokens("one")
    count_tokens("two")
    TokenChunker(max_tokens=100)
    assert offline_tiktoken == ["cl100k_base"]


def test_chunker_still_splits_text_offline(offline_tiktoken):
    paragraphs = ["Paragraph number %d with a few words in it." % i for i in range(60)]
    chunks = TokenChunker(max_tokens=60).chunk_text("\n\n".join(paragraphs))
    assert len(chunks) > 1
    joined = " ".join(c["main_content"] for c in chunks)
    for p in paragraphs:
        assert p in joined


def test_context_estimation_falls_back_offline(offline_tiktoken):
    from src.core.context_optimizer import estimate_tokens_with_margin

    result = estimate_tokens_with_margin("Some text to estimate.", language="English")
    assert result.estimation_method == "character_based"
    assert result.estimated_tokens > 0


def test_approximation_counts_dense_scripts_per_character():
    assert approximate_token_count("翻訳する本") == 5
    assert approximate_token_count("번역할 책") == 4 + 1  # 4 Hangul + 1 space


def test_approximation_does_not_undercount_latin_text():
    """The estimate should not be far below the real count, or chunks overflow."""
    text = (
        "The quick brown fox jumps over the lazy dog. It was a bright cold day "
        "in April, and the clocks were striking thirteen. "
    ) * 20
    real = count_tokens(text)  # real encoder in the dev environment
    assert approximate_token_count(text) >= real * 0.9
