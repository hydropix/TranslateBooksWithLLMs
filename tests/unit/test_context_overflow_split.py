"""Context-overflow splitting must keep the text's layout and never lose a tail.

On a ContextOverflowError the translator splits the chunk and translates the
pieces separately. The pieces used to be stripped and rejoined with "\\n",
which turned a sentence split inside a paragraph into a line break and a
paragraph break into a single newline; and a later split whose first part came
back empty ended the loop, dropping the content still queued.
"""

from unittest.mock import Mock

import pytest

from src.core.llm import ContextOverflowError
from src.core.llm.base import LLMResponse
from src.core.translator import (
    _make_llm_request_with_adaptive_context,
    split_chunk_preserving_separator,
)


def _client(max_len):
    """Translates by upper-casing; overflows above max_len source characters."""
    client = Mock()

    async def generate(prompt, system_prompt=None):
        text = prompt.split("<<<")[1].split(">>>")[0]
        if len(text) > max_len:
            raise ContextOverflowError("too long")
        return LLMResponse(content=f"<T>{text.upper()}</T>", context_used=1, context_limit=4096)

    client.generate = generate
    client.extract_translation = lambda raw: raw[3:-4]
    return client


@pytest.fixture(autouse=True)
def _bare_prompt(monkeypatch):
    import src.core.translator as translator

    def fake_prompt(content, *args, **kwargs):
        return Mock(system="", user=f"<<<{content}>>>")

    monkeypatch.setattr(translator, "generate_translation_prompt", fake_prompt)


async def _translate(content, max_len):
    translated, _, _ = await _make_llm_request_with_adaptive_context(
        main_content=content,
        previous_translation_context="", source_language="English",
        target_language="French", model="m", llm_client=_client(max_len),
        log_callback=None, has_placeholders=False,
    )
    return translated


@pytest.mark.asyncio
async def test_paragraph_break_is_preserved():
    p1 = "First paragraph. " * 15
    p2 = "Second paragraph. " * 15
    content = f"{p1.strip()}\n\n{p2.strip()}"
    assert await _translate(content, max_len=300) == content.upper()


@pytest.mark.asyncio
async def test_sentence_split_inside_a_paragraph_keeps_the_space():
    content = ("A sentence of a single long paragraph. " * 12).strip()
    assert await _translate(content, max_len=300) == content.upper()


@pytest.mark.asyncio
async def test_no_tail_is_lost_on_repeated_splits():
    # Lines without sentence terminators: the second reduction (36 %) of a
    # two-line piece used to put the cut before the first line.
    lines = [chr(ord("a") + k) * 250 for k in range(5)]
    content = "\n".join(lines)
    assert await _translate(content, max_len=300) == content.upper()


@pytest.mark.parametrize("content,ratio", [
    ("aaa\nbbb", 0.1),
    ("aaa\nbbb", 0.99),
    ("\n\naaa.\n\nbbb\n", 0.36),
    ("abcdef", 0.0),
])
def test_split_never_yields_an_empty_part(content, ratio):
    first, sep, second = split_chunk_preserving_separator(content, ratio)
    assert first and second
    assert first + sep + second == content.strip()
