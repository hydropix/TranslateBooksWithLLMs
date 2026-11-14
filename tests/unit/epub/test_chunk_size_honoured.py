"""Regression tests: the caller's chunk size must win on the placeholder path.

The EPUB and DOCX adapters built a default
TranslationContainer whose chunker used MAX_TOKENS_PER_CHUNK captured at
import time, and chunking preferred that chunker over the max_tokens argument,
so the user's chunk size was ignored for EPUB and DOCX.
"""
import pytest

from src import config
from src.core.docx.docx_translation_adapter import DocxTranslationAdapter
from lxml import etree

from src.core.epub import document_chunking
from src.core.epub.container import TranslationConfig, TranslationContainer
from src.core.epub.document_translation import prepare_document
from src.core.epub.tag_preservation import TagPreserver


def _placeholder_text(paragraphs: int = 40):
    html = "".join(
        f"<p>Paragraph {i}: the quick brown fox jumps over the lazy dog, "
        f"again and again, until the evening falls on the quiet valley.</p>"
        for i in range(paragraphs)
    )
    return TagPreserver().preserve_tags(html)


def test_prepare_document_uses_the_requested_size():
    paragraphs = "".join(
        f"<p>Paragraph {i}: the quick brown fox jumps over the lazy dog, "
        f"again and again, until the evening falls on the quiet valley.</p>"
        for i in range(40)
    )
    xhtml = f'<html xmlns="http://www.w3.org/1999/xhtml"><body>{paragraphs}</body></html>'
    small = prepare_document(etree.fromstring(xhtml), 60)
    large = prepare_document(etree.fromstring(xhtml), 4000)
    assert len(small.chunks) > len(large.chunks)
    assert len(large.chunks) == 1


@pytest.mark.parametrize("adapter_cls", [DocxTranslationAdapter])
def test_adapter_create_chunks_uses_the_requested_size(adapter_cls):
    text, tag_map = _placeholder_text()
    adapter = adapter_cls()
    small = adapter.create_chunks(text, tag_map, max_tokens=60, log_callback=None)
    large = adapter.create_chunks(text, tag_map, max_tokens=4000, log_callback=None)
    assert len(small) > len(large)
    assert len(large) == 1


def test_create_chunks_with_container_uses_the_requested_size():
    text, tag_map = _placeholder_text()
    container = TranslationContainer(TranslationConfig(max_tokens_per_chunk=4000))
    chunks = document_chunking.create_chunks(text, tag_map, 60, container=container)
    assert len(chunks) > 1
    same = document_chunking.create_chunks(text, tag_map, 4000, container=container)
    assert len(same) == 1


def test_container_reuses_its_chunker_for_the_configured_size():
    container = TranslationContainer(TranslationConfig(max_tokens_per_chunk=300))
    assert container.chunker_for(300) is container.chunker
    assert container.chunker_for(None) is container.chunker
    assert container.chunker_for(120).max_tokens == 120


def test_default_config_reads_the_setting_at_creation(monkeypatch):
    monkeypatch.setattr(config, "MAX_TOKENS_PER_CHUNK", 777)
    assert TranslationConfig().max_tokens_per_chunk == 777
