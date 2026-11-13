"""Regression tests: a lossy body re-parse must fail loudly, never erase a chapter.

replace_body_content parses the restored body with recover=True. A mis-nested
restoration such as "</p>Bonjour<p>" recovers into an EMPTY body, and the
chapter was erased while every chunk status still said "translated".
"""
import pytest
from lxml import etree

import src.core.epub.xhtml_translator as xt
from src.core.epub.body_serializer import replace_body_content
from src.core.epub.exceptions import XmlParsingError
from src.core.epub.xhtml_translation_state import (
    CHUNK_UNTRANSLATED,
    unfinished_chunk_indices,
)
from src.persistence.checkpoint_manager import CheckpointManager

XHTML_NS = "http://www.w3.org/1999/xhtml"


def _doc(body_inner: str) -> etree._Element:
    return etree.fromstring(
        f'<html xmlns="{XHTML_NS}"><head><title>t</title></head>'
        f'<body>{body_inner}</body></html>'.encode("utf-8")
    )


def _body(doc):
    return doc.find(f".//{{{XHTML_NS}}}body")


@pytest.mark.parametrize("broken", [
    "</p>Bonjour tout le monde<p>",
    "<p>a</p></div><p>Lots of text here that is long</p><p>more</p>",
])
def test_lossy_parse_raises_and_leaves_body_untouched(broken):
    doc = _doc("<p>Original chapter text</p>")
    body = _body(doc)
    before = etree.tostring(body)
    with pytest.raises(XmlParsingError):
        replace_body_content(body, broken)
    assert etree.tostring(body) == before


def test_well_formed_body_is_replaced():
    doc = _doc("<p>Original</p>")
    body = _body(doc)
    replace_body_content(body, "<p>Traduit</p><p>Encore &amp; encore</p>")
    assert "".join(body.itertext()) == "TraduitEncore & encore"


def test_lossless_recovery_is_still_accepted():
    # A stray closing inline tag is dropped by recovery without losing text.
    doc = _doc("<p>Original</p>")
    body = _body(doc)
    replace_body_content(body, "<p>Hello</span> world and a long sentence</p>")
    assert "Hello" in "".join(body.itertext())
    assert "long sentence" in "".join(body.itertext())


@pytest.fixture
def checkpoint_manager(tmp_path):
    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    manager.uploads_dir = tmp_path / "uploads"
    manager.uploads_dir.mkdir(parents=True, exist_ok=True)
    return manager


@pytest.mark.asyncio
async def test_failed_body_replacement_marks_file_unfinished(monkeypatch, checkpoint_manager):
    async def fake_request(main_content, *args, **kwargs):
        return main_content.replace("Hello", "Bonjour")

    monkeypatch.setattr(xt, "generate_translation_request", fake_request)
    # Simulate a reconstruction whose re-parse would erase the chapter.
    monkeypatch.setattr(xt, "_reconstruct_html",
                        lambda *a, **k: "</p>Bonjour tout le monde<p>")

    doc = _doc("<p>Hello world, this is a chapter.</p>")
    translation_id, href = "job_a3", "OEBPS/ch1.xhtml"
    success, stats = await xt.translate_xhtml_simplified(
        doc, "English", "French", "fake", llm_client=object(),
        checkpoint_manager=checkpoint_manager, translation_id=translation_id,
        file_href=href,
    )

    assert success is False
    # The source chapter is still there.
    assert "Hello world" in "".join(_body(doc).itertext())
    assert set(stats.chunk_outcomes.values()) == {CHUNK_UNTRANSLATED}
    state = checkpoint_manager.load_xhtml_partial_state(translation_id, href)
    assert state is not None and state.validate()
    assert unfinished_chunk_indices(state.chunk_statuses) == list(range(len(state.chunks)))
