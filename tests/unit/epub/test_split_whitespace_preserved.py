"""Regression tests: splitting an oversized paragraph must not eat spaces.

TextSplitter split on "(?<=[.!?,;:])\\s+" and rejoined the parts with "", and
LLM replies are stripped, so a paragraph cut into several chunks came back as
"on the bank,and of having nothing to do:once or twice...".
"""
import pytest
from lxml import etree

import src.core.epub.xhtml_translator as xt
from src.core.epub.html_chunker import HtmlChunker
from src.core.epub.tag_preservation import TagPreserver
from src.core.epub.xhtml_translator import PlaceholderManager

PARAGRAPH = (
    "Alice was beginning to get very tired of sitting by her sister on the bank, "
    "and of having nothing to do: once or twice she had peeped into the book her "
    "sister was reading, but it had no pictures or conversations in it. And what "
    "is the use of a book, thought Alice, without pictures or conversation? From "
    "Aesop, after V. S. Vernon Jones;  two spaces here.\nA new line follows."
)
XHTML_NS = "http://www.w3.org/1999/xhtml"


@pytest.mark.parametrize("max_tokens", [8, 15, 25, 60])
def test_chunks_join_back_to_the_source(max_tokens):
    text, tag_map = TagPreserver().preserve_tags(f"<p>{PARAGRAPH}</p>")
    chunks = HtmlChunker(max_tokens=max_tokens).chunk_html_with_placeholders(text, tag_map)
    assert len(chunks) > 1
    joined = "".join(
        PlaceholderManager.restore_to_global(c["text"], c["global_indices"]) for c in chunks
    )
    assert joined == text


@pytest.mark.asyncio
async def test_stripped_llm_replies_keep_the_space_between_chunks(monkeypatch):
    async def stripping_echo(main_content, *args, **kwargs):
        return main_content.strip()

    monkeypatch.setattr(xt, "generate_translation_request", stripping_echo)
    doc = etree.fromstring(
        f'<html xmlns="{XHTML_NS}"><head><title>t</title></head>'
        f"<body><p>{PARAGRAPH}</p></body></html>".encode("utf-8")
    )
    success, stats = await xt.translate_xhtml_simplified(
        doc, "English", "French", "fake", llm_client=object(), max_tokens_per_chunk=15,
    )
    assert success is True
    assert stats.total_chunks > 1
    body_text = "".join(doc.find(f".//{{{XHTML_NS}}}p").itertext())
    # extract_body_html normalizes whitespace runs on purpose; nothing else
    # may change.
    assert body_text == " ".join(PARAGRAPH.split())
