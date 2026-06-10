"""
Regression tests for issue #203: Plain Text Mode paragraph realignment.

The reassembly step must keep a 1:1 mapping between source and translated
paragraph indices, even when:
- (a) the source contains empty paragraphs (image-only blocks create "" slots);
- (b) a single paragraph exceeds max_tokens and is split into sentence chunks.

Before the fix, both cases shifted every subsequent paragraph by one or more
slots (headings got body text, images anchored after the wrong paragraph,
bilingual mode paired source i with translation i+1).
"""
import re

import pytest

import src.core.common.plain_text_pipeline as plain_pipeline


def _fake_perfect_llm(prefix="T::"):
    """A fake LLM that translates each paragraph of the blob 1:1."""
    async def fake_request(*, main_content, **kwargs):
        paragraphs = re.split(r"\n{2,}", main_content)
        return "\n\n".join(
            (prefix + p) if p.strip() else p for p in paragraphs
        )
    return fake_request


async def _run(paragraphs, max_tokens=1000, workers=1):
    out, stats, interrupted = await plain_pipeline.translate_paragraphs_plain(
        paragraphs=paragraphs,
        source_language="English",
        target_language="French",
        model_name="m",
        llm_client=object(),
        max_tokens_per_chunk=max_tokens,
        parallel_workers=workers,
    )
    assert not interrupted
    return out


@pytest.mark.asyncio
async def test_empty_source_paragraph_keeps_alignment(monkeypatch):
    """An empty slot (image-only block) must not shift later paragraphs."""
    monkeypatch.setattr(plain_pipeline, "generate_translation_request", _fake_perfect_llm())
    monkeypatch.setattr(plain_pipeline, "clean_translated_text", lambda s: s)

    source = ["Chapter One", "", "First body paragraph.", "Second body paragraph."]
    out = await _run(source)

    assert len(out) == len(source)
    assert out[0] == "T::Chapter One"
    assert out[1].strip() == ""  # the image-only slot stays empty
    assert out[2] == "T::First body paragraph."
    assert out[3] == "T::Second body paragraph."


@pytest.mark.asyncio
async def test_multiple_empty_slots_keep_alignment(monkeypatch):
    """Several empty slots (e.g. leading synthetic anchor) must all be preserved."""
    monkeypatch.setattr(plain_pipeline, "generate_translation_request", _fake_perfect_llm())
    monkeypatch.setattr(plain_pipeline, "clean_translated_text", lambda s: s)

    source = ["", "Heading", "", "Body text here.", "Tail paragraph."]
    out = await _run(source)

    assert len(out) == len(source)
    assert out[0].strip() == ""
    assert out[1] == "T::Heading"
    assert out[2].strip() == ""
    assert out[3] == "T::Body text here."
    assert out[4] == "T::Tail paragraph."


@pytest.mark.asyncio
async def test_oversized_paragraph_does_not_shift_tail(monkeypatch):
    """A paragraph split into sentence chunks must collapse back into ONE slot."""
    monkeypatch.setattr(plain_pipeline, "generate_translation_request", _fake_perfect_llm())
    monkeypatch.setattr(plain_pipeline, "clean_translated_text", lambda s: s)

    big = " ".join(
        f"This is sentence number {i} of an extremely long paragraph that "
        f"keeps going on and on with plenty of words inside it."
        for i in range(12)
    )
    source = [
        "Introduction paragraph with enough words to stand alone as a chunk.",
        big,
        "Tail paragraph A after the big one.",
        "Tail paragraph B closing the document.",
    ]
    # Small budget so `big` exceeds max_tokens and gets sentence-split.
    out = await _run(source, max_tokens=60)

    assert len(out) == len(source)
    assert out[0].startswith("T::Introduction paragraph")
    # All pieces of the big paragraph land in slot 1, nothing leaks into slot 2+.
    assert "sentence number" in out[1]
    assert "Tail paragraph" not in out[1]
    assert out[2] == "T::Tail paragraph A after the big one."
    assert out[3] == "T::Tail paragraph B closing the document."


@pytest.mark.asyncio
async def test_empty_and_oversized_combined(monkeypatch):
    """Both failure modes at once: empty slot + oversized paragraph."""
    monkeypatch.setattr(plain_pipeline, "generate_translation_request", _fake_perfect_llm())
    monkeypatch.setattr(plain_pipeline, "clean_translated_text", lambda s: s)

    big = " ".join(
        f"Sentence {i} keeps flowing with many additional descriptive words in it."
        for i in range(12)
    )
    source = ["Title", "", big, "Closing words of the chapter."]
    out = await _run(source, max_tokens=60)

    assert len(out) == len(source)
    assert out[0] == "T::Title"
    assert out[1].strip() == ""
    assert "Sentence" in out[2]
    assert "Closing words" not in out[2]
    assert out[3] == "T::Closing words of the chapter."


@pytest.mark.asyncio
async def test_epub_image_anchor_and_heading_stay_aligned(monkeypatch):
    """End-to-end EPUB plain mode: a leading image creates a synthetic ""
    slot; the heading and body must not shift into the wrong tags."""
    from lxml import etree

    from src.core.epub.plain_extractor import (
        extract_plain_paragraphs,
        replace_body_with_paragraphs,
    )

    monkeypatch.setattr(plain_pipeline, "generate_translation_request", _fake_perfect_llm())
    monkeypatch.setattr(plain_pipeline, "clean_translated_text", lambda s: s)

    body = etree.fromstring(
        "<body>"
        '<img src="cover.png"/>'
        "<h1>Chapter One</h1>"
        "<p>First body paragraph.</p>"
        "<p>Second body paragraph.</p>"
        "</body>"
    )
    paragraphs, tags, images = extract_plain_paragraphs(body)
    assert paragraphs[0] == ""  # synthetic anchor for the leading image
    assert 0 in images

    translated = await _run(paragraphs)
    replace_body_with_paragraphs(body, translated, tags, images)

    children = [(c.tag, (c.text or "").strip()) for c in body]
    # Image wrapper first (its anchor slot is empty), then the heading with
    # heading text, then the two body paragraphs.
    assert children[0][0] == "p" and body[0].get("class") == "plain-text-images"
    assert children[1] == ("h1", "T::Chapter One")
    assert children[2] == ("p", "T::First body paragraph.")
    assert children[3] == ("p", "T::Second body paragraph.")


@pytest.mark.asyncio
async def test_alignment_holds_in_parallel_mode(monkeypatch):
    """Same invariants with parallel workers (out-of-order completion)."""
    monkeypatch.setattr(plain_pipeline, "generate_translation_request", _fake_perfect_llm())
    monkeypatch.setattr(plain_pipeline, "clean_translated_text", lambda s: s)

    source = []
    for i in range(8):
        source.append(f"Paragraph number {i} with enough words to be its own chunk.")
        if i % 3 == 0:
            source.append("")
    out = await _run(source, max_tokens=20, workers=4)

    assert len(out) == len(source)
    for i, src in enumerate(source):
        if not src.strip():
            assert out[i].strip() == ""
        else:
            assert out[i] == f"T::{src}"
