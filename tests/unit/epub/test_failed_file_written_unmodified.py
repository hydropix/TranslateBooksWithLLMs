"""A file whose translation fails is written back exactly as it was read.

The translation pipeline mutates the parsed tree before any chunk is sent
(boilerplate removal, ruby folding). Writing that tree back for a failed file
silently dropped the share bars and ruby readings the source still had.
"""
import zipfile
from pathlib import Path

import pytest

import src.core.epub.book_loop as book_loop
import src.core.epub.translator as translator_module
from src.core.epub.body_serializer import extract_body_html
from src.core.epub.translator import translate_epub_file
from src.core.job import JobStatus

from tests.job_helpers import make_ctx, make_job
from tests.unit.epub.conftest import _disable_attribution, _echo_llm_client

CONTAINER = """<?xml version="1.0" encoding="utf-8"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
  <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>
"""

OPF = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" xmlns:dc="http://purl.org/dc/elements/1.1/" version="3.0" unique-identifier="id">
  <metadata><dc:title>Book</dc:title><dc:language>ja</dc:language><dc:identifier id="id">x</dc:identifier></metadata>
  <manifest><item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/></manifest>
  <spine><itemref idref="ch1"/></spine>
</package>
"""

CHAPTER = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Ch</title></head>
<body>
<p>彼は<ruby>宇宙<rt>そら</rt></ruby>を見上げた。</p>
<div class="sharedaddy"><a href="https://example.invalid/share">Share on X</a></div>
</body></html>
"""


def _book(tmp_path: Path) -> Path:
    dest = tmp_path / "in.epub"
    with zipfile.ZipFile(dest, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        archive.writestr("META-INF/container.xml", CONTAINER)
        archive.writestr("OEBPS/content.opf", OPF)
        archive.writestr("OEBPS/ch1.xhtml", CHAPTER)
    return dest


@pytest.mark.asyncio
async def test_failed_file_keeps_its_boilerplate_and_ruby(tmp_path, monkeypatch):
    async def failing_translation(doc_root, run, task):
        # What the real pipeline does to the tree before it fails.
        extract_body_html(doc_root)
        return False, None

    monkeypatch.setattr(translator_module, "create_llm_client",
                        lambda *args, **kwargs: _echo_llm_client())
    monkeypatch.setattr(book_loop, "translate_document", failing_translation)
    _disable_attribution(monkeypatch)

    out = tmp_path / "out.epub"
    result = await translate_epub_file(
        make_job(str(_book(tmp_path)), str(out), file_type="epub",
                 source_language="Japanese", target_language="English"),
        make_ctx(),
    )

    assert result.status is JobStatus.PARTIAL
    with zipfile.ZipFile(out) as archive:
        written = archive.read("OEBPS/ch1.xhtml").decode("utf-8")
    assert "sharedaddy" in written
    assert "Share on X" in written
    assert "<ruby>" in written and "<rt>そら</rt>" in written
