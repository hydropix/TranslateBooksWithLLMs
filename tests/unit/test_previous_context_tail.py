"""One previous-context policy, bounded for unspaced scripts too.

Three policies coexisted (last 25 words in two places of translator.py and in
plain_text_pipeline, last 200 characters in generic_translator). For CJK
output the word-based one passed the entire previous translation into every
prompt, since Chinese and Japanese have no spaces.
"""

import pytest

import src.core.common.plain_text_pipeline as plain_pipeline
from src.core.translator import (
    PREVIOUS_CONTEXT_MAX_WEIGHTED_CHARS,
    PREVIOUS_CONTEXT_MAX_WORDS,
    _weighted_length,
    previous_context_tail,
)


def test_cjk_tail_is_bounded():
    chinese = "他走进了房间，看见桌子上放着一封信。" * 40
    tail = previous_context_tail(chinese)
    assert chinese.endswith(tail)
    assert 0 < _weighted_length(tail) <= PREVIOUS_CONTEXT_MAX_WEIGHTED_CHARS
    assert len(tail) < len(chinese) // 5


def test_latin_tail_keeps_the_last_words():
    text = " ".join(f"word{k}" for k in range(100))
    tail = previous_context_tail(text)
    assert tail.split() == text.split()[-PREVIOUS_CONTEXT_MAX_WORDS:]


def test_short_text_is_unchanged():
    assert previous_context_tail("Bonjour le monde.") == "Bonjour le monde."
    assert previous_context_tail(None) == ""


def test_long_words_do_not_start_mid_word():
    text = " ".join(["x" * 40] * 20)
    tail = previous_context_tail(text)
    assert all(word == "x" * 40 for word in tail.split())


@pytest.mark.asyncio
async def test_plain_text_pipeline_uses_the_bounded_tail(monkeypatch):
    chinese = "他走进了房间，看见桌子上放着一封信。" * 40
    seen = []

    async def fake_request(*, main_content, previous_translation_context="", **kwargs):
        seen.append(previous_translation_context)
        return chinese

    monkeypatch.setattr(plain_pipeline, "generate_translation_request", fake_request)
    # Two segments: a tiny budget puts each paragraph in its own segment.
    await plain_pipeline.translate_paragraphs_plain(
        ["First paragraph here.", "Second paragraph here."],
        "English", "Chinese", "m", None, 8, prompt_options={},
    )
    assert seen[0] == ""
    assert 0 < _weighted_length(seen[1]) <= PREVIOUS_CONTEXT_MAX_WEIGHTED_CHARS
