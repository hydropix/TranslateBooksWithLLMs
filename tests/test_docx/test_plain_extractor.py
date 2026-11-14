"""Regression tests for the DOCX Plain Text Mode extractor and rebuilder."""
import io
import struct
import zlib

from docx import Document
from docx.shared import Emu
from docx.oxml.ns import qn

from src.core.docx.plain_extractor import build_minimal_docx, extract_plain_paragraphs


def _png_bytes(width: int = 2, height: int = 2) -> bytes:
    """Return a tiny valid RGB PNG built in memory."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _build_doc_with_table(path):
    doc = Document()
    doc.add_paragraph("Before the table.")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Cell A1"
    table.cell(0, 1).text = "Cell B1"
    table.cell(1, 0).text = "Cell A2"
    # (1, 1) intentionally left empty
    doc.add_paragraph("After the table.")
    doc.save(path)


def test_table_cells_are_extracted_in_document_order(tmp_path):
    source = tmp_path / "table.docx"
    _build_doc_with_table(source)

    content = extract_plain_paragraphs(str(source))

    assert content.paragraphs_text == [
        "Before the table.",
        "Cell A1",
        "Cell B1",
        "Cell A2",
        "After the table.",
    ]


def test_table_cells_are_rebuilt_inside_a_table(tmp_path):
    source = tmp_path / "table.docx"
    _build_doc_with_table(source)
    content = extract_plain_paragraphs(str(source))

    translated = [text.upper() for text in content.paragraphs_text]
    output = tmp_path / "out.docx"
    build_minimal_docx(translated, content, str(output))

    rebuilt = Document(str(output))
    body_paragraphs = [p.text for p in rebuilt.paragraphs if p.text]
    assert body_paragraphs == ["BEFORE THE TABLE.", "AFTER THE TABLE."]

    assert len(rebuilt.tables) == 1
    table = rebuilt.tables[0]
    assert len(table.rows) == 2
    assert len(table.columns) == 2
    assert table.cell(0, 0).text == "CELL A1"
    assert table.cell(0, 1).text == "CELL B1"
    assert table.cell(1, 0).text == "CELL A2"
    assert table.cell(1, 1).text == ""

    # The table sits between the two body paragraphs.
    body_tags = [child.tag for child in rebuilt.element.body]
    tbl_pos = body_tags.index(qn("w:tbl"))
    texts_before = [
        "".join(t.text or "" for t in child.iter(qn("w:t")))
        for child in list(rebuilt.element.body)[:tbl_pos]
    ]
    assert "BEFORE THE TABLE." in texts_before


def test_bilingual_table_cells_keep_source_and_translation(tmp_path):
    source = tmp_path / "table.docx"
    _build_doc_with_table(source)
    content = extract_plain_paragraphs(str(source))

    translated = [f"T:{text}" for text in content.paragraphs_text]
    output = tmp_path / "out.docx"
    build_minimal_docx(translated, content, str(output), bilingual=True)

    cell = Document(str(output)).tables[0].cell(0, 0)
    assert [p.text for p in cell.paragraphs] == ["Cell A1", "T:Cell A1"]


def test_each_image_keeps_its_own_size(tmp_path):
    source = tmp_path / "images.docx"
    doc = Document()
    paragraph = doc.add_paragraph("Two pictures:")
    run = paragraph.add_run()
    run.add_picture(io.BytesIO(_png_bytes()), width=Emu(914400), height=Emu(914400))
    run = paragraph.add_run()
    run.add_picture(io.BytesIO(_png_bytes()), width=Emu(1828800), height=Emu(457200))
    doc.save(source)

    content = extract_plain_paragraphs(str(source))

    images = content.images_by_paragraph[0]
    assert [(i.width_emu, i.height_emu) for i in images] == [
        (914400, 914400),
        (1828800, 457200),
    ]


def test_image_in_table_cell_stays_in_the_cell(tmp_path):
    source = tmp_path / "cell_image.docx"
    doc = Document()
    doc.add_paragraph("Intro.")
    table = doc.add_table(rows=1, cols=1)
    cell_paragraph = table.cell(0, 0).paragraphs[0]
    cell_paragraph.add_run().add_picture(
        io.BytesIO(_png_bytes()), width=Emu(914400), height=Emu(914400)
    )
    doc.save(source)

    content = extract_plain_paragraphs(str(source))
    output = tmp_path / "out.docx"
    build_minimal_docx(list(content.paragraphs_text), content, str(output))

    rebuilt = Document(str(output))
    cell_element = rebuilt.tables[0].cell(0, 0)._element
    assert cell_element.findall(".//" + qn("a:blip"))
    body_level_blips = [
        p for p in rebuilt.paragraphs if p._element.findall(".//" + qn("a:blip"))
    ]
    assert body_level_blips == []


def test_merged_cells_and_nested_tables_survive_round_trip(tmp_path):
    source = tmp_path / "merged.docx"
    doc = Document()
    table = doc.add_table(rows=3, cols=2)
    table.cell(0, 0).merge(table.cell(0, 1)).text = "Wide header"
    table.cell(1, 0).merge(table.cell(2, 0)).text = "Tall cell"
    table.cell(1, 1).text = "Right 1"
    nested = table.cell(2, 1).add_table(rows=1, cols=1)
    nested.cell(0, 0).text = "Nested text"
    doc.save(source)

    content = extract_plain_paragraphs(str(source))
    assert content.paragraphs_text == ["Wide header", "Tall cell", "Right 1", "Nested text"]

    output = tmp_path / "out.docx"
    build_minimal_docx(list(content.paragraphs_text), content, str(output))
    rebuilt = Document(str(output)).tables[0]

    assert rebuilt.cell(0, 0)._tc is rebuilt.cell(0, 1)._tc
    assert rebuilt.cell(1, 0)._tc is rebuilt.cell(2, 0)._tc
    assert rebuilt.cell(0, 0).text == "Wide header"
    assert rebuilt.cell(1, 0).text == "Tall cell"
    assert rebuilt.cell(1, 1).text == "Right 1"
    assert "Nested text" in rebuilt.cell(2, 1).text


def test_every_table_cell_keeps_a_single_paragraph(tmp_path):
    """Each rebuilt cell must reuse its initial empty paragraph.

    Used cells were once tracked by ``id()`` of transient lxml proxies; a
    recycled id made a fresh cell look already used, so it got a stray empty
    leading paragraph. A large grid makes id reuse near-certain.
    """
    rows, cols = 40, 5
    source = tmp_path / "grid.docx"
    doc = Document()
    table = doc.add_table(rows=rows, cols=cols)
    for r in range(rows):
        for c in range(cols):
            table.cell(r, c).text = f"R{r}C{c}"
    doc.save(source)

    content = extract_plain_paragraphs(str(source))
    output = tmp_path / "out.docx"
    build_minimal_docx(list(content.paragraphs_text), content, str(output))
    rebuilt = Document(str(output)).tables[0]

    wrong = [
        (r, c, rebuilt.cell(r, c).text)
        for r in range(rows)
        for c in range(cols)
        if rebuilt.cell(r, c).text != f"R{r}C{c}"
    ]
    assert wrong == []
