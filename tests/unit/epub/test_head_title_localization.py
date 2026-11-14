"""Regression tests: a translated document's <head><title> follows its body.

Only the body is translated, so every <title> used to stay in the source
language. It is now set, without an LLM call, to the translated first heading
when the source title equals the source first heading, else to the translated
TOC label when it equals the source TOC label of that file; any other title is
left unchanged.
"""
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from lxml import etree

import src.core.epub.translator as translator_module
from src.config import INPUT_TAG_IN, INPUT_TAG_OUT
from src.core.epub.archive import parse_xhtml
from src.core.epub.head_titles import capture_source_titles, localize_head_titles
from src.core.epub.translator import translate_epub_file
from src.core.job import JobStatus
from src.core.llm.base import LLMResponse

from tests.job_helpers import make_ctx, make_job
from tests.unit.epub.conftest import _disable_attribution, _read, _write, _zip_dir_as_epub




OPF = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" xmlns:dc="http://purl.org/dc/elements/1.1/" version="2.0" unique-identifier="id">
  <metadata><dc:title>Book</dc:title><dc:language>en</dc:language></metadata>
  <manifest>
    <item id="ch01" href="Text/ch01.xhtml" media-type="application/xhtml+xml"/>
    <item id="ch02" href="Text/ch02.xhtml" media-type="application/xhtml+xml"/>
    <item id="ch03" href="Text/ch03.xhtml" media-type="application/xhtml+xml"/>
    <item id="ch04" href="Text/ch04.xhtml" media-type="application/xhtml+xml"/>
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
  </manifest>
  <spine toc="ncx">
    <itemref idref="ch01"/><itemref idref="ch02"/><itemref idref="ch03"/><itemref idref="ch04"/>
  </spine>
</package>
"""

NCX = """<?xml version="1.0" encoding="utf-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <docTitle><text>Book</text></docTitle>
  <navMap>
    <navPoint id="n1" playOrder="1"><navLabel><text>The rain</text></navLabel><content src="Text/ch01.xhtml"/></navPoint>
    <navPoint id="n2" playOrder="2"><navLabel><text>Chapter two</text></navLabel><content src="Text/ch02.xhtml#start"/></navPoint>
    <navPoint id="n3" playOrder="3"><navLabel><text>The night</text></navLabel><content src="Text/ch03.xhtml"/></navPoint>
    <navPoint id="n4" playOrder="4"><navLabel><text>The end</text></navLabel><content src="Text/ch04.xhtml"/></navPoint>
  </navMap>
</ncx>
"""


def _xhtml(title: str, body: str) -> str:
    return ('<?xml version="1.0" encoding="utf-8"?>\n'
            '<html xmlns="http://www.w3.org/1999/xhtml">\n'
            f'<head><title>{title}</title></head>\n'
            f'<body>{body}</body>\n</html>\n')


DOCS = {
    # Title equals the first heading.
    "ch01.xhtml": _xhtml("The rain", "<h1>The rain</h1><p>It rained all day long.</p>"),
    # Title equals the TOC label, not the heading; the TOC entry points at
    # the heading through a fragment.
    "ch02.xhtml": _xhtml("Chapter two",
                         '<h2 id="start">Two: the wind</h2><p>The wind blew hard.</p>'),
    # Title equals neither.
    "ch03.xhtml": _xhtml("Book &amp; Co", "<h1>The night</h1><p>Dark and cold.</p>"),
    # Title equals the heading up to whitespace.
    "ch04.xhtml": _xhtml("  The\n   end ", "<h1>The end</h1><p>All is over.</p>"),
}


def _build_book(root: Path) -> Path:
    _write(root / "META-INF" / "container.xml",
           '<?xml version="1.0" encoding="utf-8"?>\n'
           '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">\n'
           '  <rootfiles><rootfile full-path="OEBPS/content.opf" '
           'media-type="application/oebps-package+xml"/></rootfiles>\n'
           '</container>\n')
    _write(root / "mimetype", "application/epub+zip")
    _write(root / "OEBPS" / "content.opf", OPF)
    _write(root / "OEBPS" / "toc.ncx", NCX)
    for name, text in DOCS.items():
        _write(root / "OEBPS" / "Text" / name, text)
    return root


def _upper_llm_client() -> MagicMock:
    """Fake translator: upper-cases the source text, so translation is visible."""
    client = MagicMock()
    client.context_window = 2048

    async def generate(user_prompt, system_prompt=None, **kwargs):
        start = user_prompt.find(INPUT_TAG_IN)
        end = user_prompt.find(INPUT_TAG_OUT)
        content = user_prompt[start + len(INPUT_TAG_IN):end].strip("\n")
        # Placeholders ([id0], [[0]], ...) must survive byte-identical.
        content = re.sub(r"(\[[^\]]*\])|([^\[]+)",
                         lambda m: m.group(1) or m.group(2).upper(), content)
        return LLMResponse(content=content, prompt_tokens=10, completion_tokens=10,
                           context_used=20, context_limit=4096, was_truncated=False)

    client.generate = generate
    client.extract_translation = lambda response: response
    return client


def _title(xhtml: str) -> str:
    root = parse_xhtml(xhtml)
    return root.find(".//{http://www.w3.org/1999/xhtml}title").text


@pytest.fixture
def translated_epub(tmp_path, monkeypatch) -> Path:
    input_epub = _zip_dir_as_epub(_build_book(tmp_path / "src"), tmp_path / "in.epub")
    monkeypatch.setattr(translator_module, "create_llm_client",
                        lambda *args, **kwargs: _upper_llm_client())
    _disable_attribution(monkeypatch)
    output = tmp_path / "out.epub"

    import asyncio
    result = asyncio.run(translate_epub_file(
        make_job(str(input_epub), str(output), file_type="epub",
                 source_language="English", target_language="French"),
        make_ctx(log_callback=lambda *a, **k: None),
    ))
    assert result.status is JobStatus.COMPLETED
    return output


def test_title_equal_to_first_heading_is_translated(translated_epub):
    assert _title(_read(translated_epub, "OEBPS/Text/ch01.xhtml")) == "THE RAIN"


def test_title_equal_to_toc_label_takes_translated_toc_label(translated_epub):
    title = _title(_read(translated_epub, "OEBPS/Text/ch02.xhtml"))
    assert title == "TWO: THE WIND"
    # It is the very label the NCX pass wrote for that entry.
    assert "<text>TWO: THE WIND</text>" in _read(translated_epub, "OEBPS/toc.ncx")


def test_title_matching_nothing_is_unchanged(translated_epub):
    xhtml = _read(translated_epub, "OEBPS/Text/ch03.xhtml")
    assert _title(xhtml) == "Book & Co"
    assert "<title>Book &amp; Co</title>" in xhtml


def test_whitespace_only_difference_still_matches(translated_epub):
    assert _title(_read(translated_epub, "OEBPS/Text/ch04.xhtml")) == "THE END"


def test_case_difference_does_not_match(tmp_path):
    root = _build_book(tmp_path / "book")
    path = root / "OEBPS" / "Text" / "ch03.xhtml"
    path.write_text(_xhtml("the night", "<h1>The night</h1>"), encoding="utf-8")
    opf_dir = str(root / "OEBPS")
    source = capture_source_titles([str(path)], opf_dir, etree.parse(str(root / "OEBPS" / "content.opf")),
                                    book_root=str(root))

    doc = parse_xhtml(_xhtml("the night", "<h1>LA NUIT</h1>"))
    assert localize_head_titles({str(path): doc}, source) == 0
    assert doc.find(".//{http://www.w3.org/1999/xhtml}title").text == "the night"


def test_restored_document_uses_source_captured_before_restore(tmp_path):
    """Resume: the restore overwrites the extracted file with its translation;
    the source side was captured before, so the title is still localized."""
    root = _build_book(tmp_path / "book")
    path = root / "OEBPS" / "Text" / "ch01.xhtml"
    opf_dir = str(root / "OEBPS")
    source = capture_source_titles([str(path)], opf_dir, etree.parse(str(root / "OEBPS" / "content.opf")),
                                    book_root=str(root))

    # The checkpointed translation keeps the source <title> (the checkpoint is
    # written before titles are localized).
    restored = _xhtml("The rain", "<h1>LA PLUIE</h1><p>IL A PLU.</p>")
    path.write_text(restored, encoding="utf-8")
    doc = parse_xhtml(restored)
    assert localize_head_titles({str(path): doc}, source) == 1
    assert doc.find(".//{http://www.w3.org/1999/xhtml}title").text == "LA PLUIE"


def test_epub3_nav_label_is_used(tmp_path):
    root = tmp_path / "book" / "OEBPS"
    _write(root / "content.opf", """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <manifest>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="c" href="Text/c.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
</package>""")
    _write(root / "nav.xhtml", """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head><title>Nav</title></head><body>
<nav epub:type="toc"><ol><li><a href="Text/c.xhtml#s"><span>Part</span> one</a></li></ol></nav>
</body></html>""")
    path = root / "Text" / "c.xhtml"
    _write(path, _xhtml("Part one", '<h1 id="s">1. Beginnings</h1>'))
    source = capture_source_titles([str(path)], str(root), etree.parse(str(root / "content.opf")),
                                    book_root=str(root.parent))

    doc = parse_xhtml(_xhtml("Part one", '<h1 id="s">1. DÉBUTS</h1>'))
    assert localize_head_titles({str(path): doc}, source) == 1
    assert doc.find(".//{http://www.w3.org/1999/xhtml}title").text == "1. DÉBUTS"
