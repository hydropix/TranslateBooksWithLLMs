"""
Tests for the term_fix_file / scan_file entry points: TXT, SRT, EPUB, DOCX,
literal and smart (fake LLM) modes.
"""
import re
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.adapters import scan_file, term_fix_file
from src.core.adapters.exceptions import UnsupportedFormatError
from src.core.term_fix import STATUS_FIXED, parse_rules_text


class _Response:
    def __init__(self, content):
        self.content = content
        self.was_truncated = False
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.context_used = 0
        self.context_limit = 0


class FakeLLM:
    """Answers term-fix prompts by swapping each marked span for `answer(span)`."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    async def make_request(self, prompt, model=None, system_prompt=None):
        self.calls += 1
        items = re.findall(r'<s id="(\d+)">(.*?)</s>', prompt, re.DOTALL)
        out = []
        for ident, text in items:
            fixed = re.sub(r"⟦(.*?)⟧", lambda m: self.answer(m.group(1)), text)
            out.append(f'<s id="{ident}">{fixed}</s>')
        return _Response("\n".join(out))

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_txt_bytes_outside_matches_are_preserved(tmp_path):
    src = tmp_path / "book.txt"
    raw = "﻿Chapter 1\r\nAshe came.\r\n\r\nNo match here — ünïcode.\r\n"
    src.write_bytes(raw.encode("utf-8"))
    out = tmp_path / "book (term-fix).txt"
    logs = []

    result = await term_fix_file(
        str(src), str(out), parse_rules_text("Ashe -> Askan"),
        log_callback=lambda key, msg: logs.append(key),
    )

    assert out.read_bytes() == raw.replace("Ashe", "Askan").encode("utf-8")
    assert len(result.occurrences) == 1
    report = (tmp_path / "book (term-fix).term-fix.md").read_text(encoding="utf-8")
    assert "| line 2 | Ashe | Ashe came. | Askan came. | literal |" in report
    assert logs == ["term_fix_start", "term_fix_saved", "term_fix_report"]


@pytest.mark.asyncio
async def test_srt_keeps_timings(tmp_path):
    src = tmp_path / "film.srt"
    raw = "1\n00:00:01,000 --> 00:00:02,000\nAshe runs.\n\n2\n00:00:03,000 --> 00:00:04,000\nHi\n"
    src.write_text(raw, encoding="utf-8")
    out = tmp_path / "out.srt"
    await term_fix_file(str(src), str(out), parse_rules_text("Ashe -> Askan"))
    assert out.read_text(encoding="utf-8") == raw.replace("Ashe", "Askan")


def _make_epub(path: Path, chapters: dict) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(zipfile.ZipInfo("mimetype"), "application/epub+zip",
                    compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", "<container/>")
        for name, body in chapters.items():
            zf.writestr(name, body, compress_type=zipfile.ZIP_DEFLATED)


@pytest.mark.asyncio
async def test_epub_edits_text_only_and_copies_other_entries(tmp_path):
    chapter = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Ashe</title>'
        '<style>.Ashe{}</style></head>\n'
        '<body><p class="Ashe">Ashe &amp; friends met <i>Ashe</i>.</p>'
        '<img alt="Ashe" src="a.png"/></body></html>'
    )
    untouched = '<html><body><p>Nothing&nbsp;here.</p></body></html>'
    src, out = tmp_path / "book.epub", tmp_path / "out.epub"
    _make_epub(src, {"OEBPS/ch1.xhtml": chapter, "OEBPS/ch2.xhtml": untouched})

    result = await term_fix_file(str(src), str(out), parse_rules_text("Ashe -> Askan"))

    assert [o.location for o in result.occurrences] == ["OEBPS/ch1.xhtml"] * 2
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(out) as zout:
        assert zout.namelist() == zin.namelist()
        assert zout.infolist()[0].compress_type == zipfile.ZIP_STORED
        assert zout.read("OEBPS/ch2.xhtml") == zin.read("OEBPS/ch2.xhtml")
        fixed = zout.read("OEBPS/ch1.xhtml").decode("utf-8")
    # Head, style, attributes untouched; both text occurrences fixed; the
    # entity of an edited segment stays valid XML.
    assert fixed == chapter.replace(
        "Ashe &amp; friends met <i>Ashe</i>", "Askan &amp; friends met <i>Askan</i>")


def _make_docx(path: Path) -> None:
    from docx import Document
    doc = Document()
    p = doc.add_paragraph()
    p.add_run("Lord ")
    bold = p.add_run("As")
    bold.bold = True
    p.add_run("he spoke. ")
    p.add_run("Untouched.").italic = True
    doc.add_paragraph("Ashe again.")
    table = doc.add_table(rows=1, cols=1)
    table.cell(0, 0).text = "Ashe in a table."
    doc.save(str(path))


@pytest.mark.asyncio
async def test_docx_edits_runs_in_place(tmp_path):
    from docx import Document
    src, out = tmp_path / "book.docx", tmp_path / "out.docx"
    _make_docx(src)

    result = await term_fix_file(str(src), str(out), parse_rules_text("Ashe -> Askan"))

    assert len(result.occurrences) == 3
    doc = Document(str(out))
    runs = doc.paragraphs[0].runs
    # The match spanned a bold run: the replacement lands in the first run it
    # touched, the other runs keep their own text and formatting.
    assert [r.text for r in runs] == ["Lord ", "Askan", " spoke. ", "Untouched."]
    assert runs[1].bold and runs[3].italic
    assert doc.paragraphs[1].text == "Askan again."
    assert doc.tables[0].cell(0, 0).text == "Askan in a table."


@pytest.mark.asyncio
async def test_docx_without_match_is_copied_verbatim(tmp_path):
    src, out = tmp_path / "book.docx", tmp_path / "out.docx"
    _make_docx(src)
    await term_fix_file(str(src), str(out), parse_rules_text("Zorro -> Zed"))
    assert out.read_bytes() == src.read_bytes()


@pytest.mark.asyncio
async def test_smart_mode_inflects_through_llm(tmp_path, monkeypatch):
    fake = FakeLLM(lambda found: "Eldsdrakens" if found.endswith("s") else "Eldsdraken")
    import src.core.refine.client_setup as client_setup
    monkeypatch.setattr(client_setup, "build_refine_client", lambda **kw: (fake, None))

    src, out = tmp_path / "bok.txt", tmp_path / "out.txt"
    src.write_text("Drakbranden vaknade. Drakbrandens eld brann.\n", encoding="utf-8")
    result = await term_fix_file(
        str(src), str(out), parse_rules_text("Drakbrand* -> Eldsdraken"),
        mode="smart", target_language="Swedish",
    )

    assert out.read_text(encoding="utf-8") == "Eldsdraken vaknade. Eldsdrakens eld brann.\n"
    assert {o.status for o in result.occurrences} == {STATUS_FIXED}
    assert fake.calls == 1  # both snippets in one batch


@pytest.mark.asyncio
async def test_literal_mode_refuses_wildcards(tmp_path):
    src = tmp_path / "bok.txt"
    src.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="wildcard"):
        await term_fix_file(str(src), str(tmp_path / "o.txt"),
                            parse_rules_text("Drakbrand* -> Eldsdraken"))


@pytest.mark.asyncio
async def test_scan_file_writes_nothing(tmp_path):
    src = tmp_path / "book.txt"
    src.write_text("Ashe. Ashe.", encoding="utf-8")
    [res] = await scan_file(str(src), parse_rules_text("Ashe -> Askan"))
    assert res.total == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["book.txt"]


@pytest.mark.asyncio
async def test_scan_file_epub(tmp_path):
    src = tmp_path / "book.epub"
    _make_epub(src, {"a.xhtml": "<html><head><title>Ashe</title></head><body>Ashe</body></html>"})
    [res] = await scan_file(str(src), parse_rules_text("Ashe -> Askan"))
    assert res.total == 1  # the <head> title is not book prose


@pytest.mark.asyncio
async def test_unsupported_format(tmp_path):
    src = tmp_path / "book.pdf"
    src.write_bytes(b"%PDF-1.4 Ashe")
    with pytest.raises(UnsupportedFormatError):
        await term_fix_file(str(src), str(tmp_path / "o.pdf"), parse_rules_text("Ashe -> Askan"))


@pytest.mark.asyncio
async def test_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        await term_fix_file(str(tmp_path / "nope.txt"), str(tmp_path / "o.txt"),
                            parse_rules_text("Ashe -> Askan"))
