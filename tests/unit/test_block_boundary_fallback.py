"""Lost placeholders never move text across blocks, in every format.

When a model drops the placeholders of a chunk spanning several blocks
(paragraphs, headings, list items, table cells...), the fallback ladder
translates each block on its own instead of letting token alignment
reinsert the block tags proportionally over the whole chunk, which bled
text from one block into its neighbours. FB2 has its own end-to-end case in
``test_fb2_adapter.py``; EPUB and DOCX are covered here.
"""
import re

import docx
import pytest
from lxml import etree

from src.core.epub import chunk_fallback
from src.core.epub.tag_classifier import is_block_boundary_markup
from tests.characterization import fake_llm
from tests.characterization.recorder import record_translation
from tests.epub_engine_helpers import translate_xhtml

XHTML_NS = "http://www.w3.org/1999/xhtml"
PLACEHOLDER = re.compile(r"\[id\d+\]")
JUNK = "NOTE DU TRADUCTEUR, TRÈS LONGUE: "


def _upper_outside_placeholders(text: str) -> str:
    return "".join(part if PLACEHOLDER.fullmatch(part) else part.upper()
                   for part in re.split(r"(\[id\d+\])", text))


def careless_llm(block_texts, calls):
    """A fake request that drops every placeholder of any multi-block chunk.

    Its multi-block reply is also padded, so proportional reinsertion lands
    in the wrong place. A single-block request is translated faithfully
    (upper-cased, placeholders kept).
    """
    async def request(text, *args, **kwargs):
        clean = PLACEHOLDER.sub("", text)
        if sum(block in clean for block in block_texts) < 2:
            return _upper_outside_placeholders(text)
        calls["multi_block"] += 1
        return JUNK + clean.upper()
    return request


class TestIsBlockBoundaryMarkup:
    @pytest.mark.parametrize("markup", [
        "<p>", "</p>", '<p class="x">', "</em></p><p>", "<h2>", "</h3>",
        "<li>", "</td><td>", "<th>", "<blockquote>", "<dt>", "</dd>",
        "<pre>", "<figcaption>", "</tr>", "<html:p>", '<p data-b="3">',
        "</div>\n<div>",
    ])
    def test_block_markup(self, markup):
        assert is_block_boundary_markup(markup)

    @pytest.mark.parametrize("markup", [
        "<em>", "</strong>", '<a href="#n1">', "<span>", "<br/>", "<sup>1</sup>",
        "<emphasis>", "</strikethrough>", "<fb2-keep/>", "<!-- p -->", "",
    ])
    def test_inline_markup(self, markup):
        assert not is_block_boundary_markup(markup)


EPUB_BLOCKS = [
    ("h2", "The Old Harbour"),
    ("p", "The boats came back at dawn, heavy with fish and salt."),
    ("p", "Nobody on the quay said a word about the missing sailor."),
    ("li", "Three nets mended before noon"),
    ("td", "Harbour master"),
    ("td", "Paid in silver"),
    ("blockquote", "The sea keeps what it takes."),
]


def _epub_doc():
    body = (
        "<h2>The Old Harbour</h2>"
        "<p>The boats came back at dawn, heavy with <em>fish</em> and salt.</p>"
        "<p>Nobody on the quay said a word about the <b>missing</b> sailor.</p>"
        "<ul><li>Three nets mended before noon</li></ul>"
        "<table><tr><td>Harbour master</td><td>Paid in silver</td></tr></table>"
        "<blockquote>The sea keeps what it takes.</blockquote>"
    )
    return etree.fromstring(
        f'<html xmlns="{XHTML_NS}"><head><title>t</title></head>'
        f"<body>{body}</body></html>".encode("utf-8"))


@pytest.mark.asyncio
async def test_epub_lost_placeholders_keep_each_block_text(monkeypatch):
    calls = {"multi_block": 0}
    monkeypatch.setattr("src.config.EPUB_TOKEN_ALIGNMENT_ENABLED", True)
    monkeypatch.setattr(chunk_fallback, "generate_translation_request",
                        careless_llm([text for _, text in EPUB_BLOCKS], calls))
    doc = _epub_doc()
    success, stats = await translate_xhtml(doc, max_retries=1, max_tokens_per_chunk=400)

    assert success is True
    assert calls["multi_block"] > 0, "the document must yield a multi-block chunk"
    for tag, text in EPUB_BLOCKS:
        found = ["".join(el.itertext()) for el in doc.iter(f"{{{XHTML_NS}}}{tag}")]
        assert text.upper() in found, (tag, found)
    assert JUNK not in etree.tostring(doc, encoding="unicode")
    # Inline markup still sits inside its own block.
    em = next(doc.iter(f"{{{XHTML_NS}}}em"))
    assert em.text == "FISH"
    assert stats.token_alignment_used == 0


@pytest.mark.asyncio
async def test_single_block_chunk_still_uses_token_alignment(monkeypatch):
    """A chunk holding one block keeps the whole-chunk ladder (Phase 2)."""
    async def drops_placeholders(text, *args, **kwargs):
        return PLACEHOLDER.sub("", text).upper()

    monkeypatch.setattr("src.config.EPUB_TOKEN_ALIGNMENT_ENABLED", True)
    monkeypatch.setattr(chunk_fallback, "generate_translation_request", drops_placeholders)
    doc = etree.fromstring(
        f'<html xmlns="{XHTML_NS}"><head><title>t</title></head>'
        "<body><p>The boats came back with <em>fish</em> and salt.</p></body></html>"
        .encode("utf-8"))
    success, stats = await translate_xhtml(doc, max_retries=1)

    assert success is True
    assert stats.token_alignment_used == 1
    p = next(doc.iter(f"{{{XHTML_NS}}}p"))
    assert "".join(p.itertext()) == "THE BOATS CAME BACK WITH FISH AND SALT."


DOCX_PARAGRAPHS = [
    "The Old Harbour",
    "The boats came back at dawn, heavy with fish and salt.",
    "Nobody on the quay said a word about the missing sailor.",
    "Three nets mended before noon",
]
DOCX_CELLS = ["Harbour master", "Paid in silver"]


def _build_docx(path):
    document = docx.Document()
    document.add_heading(DOCX_PARAGRAPHS[0], level=2)
    p = document.add_paragraph("The boats came back at dawn, heavy with ")
    p.add_run("fish").bold = True
    p.add_run(" and salt.")
    document.add_paragraph(DOCX_PARAGRAPHS[2])
    document.add_paragraph(DOCX_PARAGRAPHS[3], style="List Bullet")
    table = document.add_table(rows=1, cols=2)
    for cell, text in zip(table.rows[0].cells, DOCX_CELLS):
        cell.text = text
    document.save(str(path))
    return path


def test_docx_lost_placeholders_keep_each_block_text(tmp_path, monkeypatch):
    calls = {"multi_block": 0}
    fake_llm.install(monkeypatch)
    monkeypatch.setattr("src.config.EPUB_TOKEN_ALIGNMENT_ENABLED", True)
    monkeypatch.setattr(chunk_fallback, "generate_translation_request",
                        careless_llm(DOCX_PARAGRAPHS + DOCX_CELLS, calls))
    src = _build_docx(tmp_path / "in.docx")
    out = tmp_path / "out.docx"
    record_translation(tmp_path, src, out, max_tokens_per_chunk=400)

    assert calls["multi_block"] > 0, "the document must yield a multi-block chunk"
    result = docx.Document(str(out))
    paragraphs = [p.text for p in result.paragraphs if p.text.strip()]
    assert paragraphs == [text.upper() for text in DOCX_PARAGRAPHS]
    cells = [cell.text for cell in result.tables[0].rows[0].cells]
    assert cells == [text.upper() for text in DOCX_CELLS]
    bold = [run.text for run in result.paragraphs[1].runs if run.bold]
    assert bold == ["FISH"]
