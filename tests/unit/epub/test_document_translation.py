"""File-level entry points: one extraction per file, prepared chunks reused.

``translate_document`` extracts a body once (or not at all when the book
layer hands over the chunks it prepared for the precount), and the output
must not depend on which of the two paths produced the chunks.
"""
import pytest
from lxml import etree

from src.core.epub import chunk_fallback, document_translation
from src.core.epub.document_translation import prepare_document
from src.core.epub.exceptions import BodyExtractionError
from src.core.epub.placeholder_renumberer import (
    globalize_chunks,
    globalize_placeholders,
    localize_placeholders,
)
from tests.epub_engine_helpers import translate_xhtml

XHTML_NS = "http://www.w3.org/1999/xhtml"
BODY = (
    '<div class="share-buttons"><a href="#">Share</a></div>'
    '<h1 class="title">Chapter <em>one</em></h1>'
    '<p>It was a <b>bright</b> cold day in April.</p>'
    '<p>The <ruby>clocks<rt>c</rt></ruby> were striking thirteen.</p>'
)


def _doc(body=BODY):
    return etree.fromstring(
        f'<html xmlns="{XHTML_NS}"><head><title>t</title></head>'
        f'<body class="chapter">{body}</body></html>'.encode()
    )


def _echo(monkeypatch):
    calls = []

    async def echo(main_content, *args, **kwargs):
        calls.append(main_content)
        return main_content.upper()

    monkeypatch.setattr(chunk_fallback, "generate_translation_request", echo)
    return calls


@pytest.mark.asyncio
async def test_prepared_chunks_give_the_same_output(monkeypatch):
    _echo(monkeypatch)
    extracted = _doc()
    ok_a, _ = await translate_xhtml(extracted, max_tokens_per_chunk=60)

    prepared_doc = _doc()
    prepared = prepare_document(_doc(), 60)  # the precount, on its own tree
    ok_b, _ = await translate_xhtml(prepared_doc, max_tokens_per_chunk=60, prepared=prepared)

    assert ok_a is ok_b is True
    assert etree.tostring(extracted) == etree.tostring(prepared_doc)
    assert b"BRIGHT" in etree.tostring(prepared_doc)


@pytest.mark.asyncio
async def test_prepared_chunks_skip_the_extraction(monkeypatch):
    _echo(monkeypatch)
    calls = []
    real_extract = document_translation.extract_body_html

    def counting_extract(*args, **kwargs):
        calls.append(1)
        return real_extract(*args, **kwargs)

    monkeypatch.setattr(document_translation, "extract_body_html", counting_extract)
    prepared = prepare_document(_doc(), 60)
    calls.clear()
    await translate_xhtml(_doc(), max_tokens_per_chunk=60, prepared=prepared)
    assert calls == []
    await translate_xhtml(_doc(), max_tokens_per_chunk=60)
    assert calls == [1]


@pytest.mark.asyncio
async def test_empty_body_is_finalized_without_llm(monkeypatch):
    calls = _echo(monkeypatch)
    for prepared in (None, prepare_document(_doc("   "), 60)):
        success, stats = await translate_xhtml(_doc("   "), prepared=prepared)
        assert success is True
        assert stats.total_chunks == 0
    assert calls == []


@pytest.mark.asyncio
async def test_missing_body_raises(monkeypatch):
    _echo(monkeypatch)
    doc = etree.fromstring(f'<html xmlns="{XHTML_NS}"><head/></html>'.encode())
    with pytest.raises(BodyExtractionError):
        await translate_xhtml(doc)


def test_placeholder_renumbering_round_trip():
    fmt = ("[id", "]")
    local = "[id0]Hello [id1]world[id2] [id1]"
    indices = [7, 1, 12]
    global_text = globalize_placeholders(local, indices, fmt)
    assert global_text == "[id7]Hello [id1]world[id12] [id1]"
    assert globalize_placeholders(local, indices) == global_text  # format detected
    assert localize_placeholders(global_text, indices, fmt) == local
    chunks = [{"text": local, "global_indices": indices}, {"text": "x", "global_indices": []}]
    assert globalize_chunks(chunks, fmt) == [global_text, "x"]
