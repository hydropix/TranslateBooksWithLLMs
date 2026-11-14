"""
Tests for the PDF extractor (PDF -> paragraphs).

Fixtures are generated with PyMuPDF in tests/test_pdf/conftest.py.
"""

import importlib
import io
import sys
from pathlib import Path

import pytest

pymupdf = pytest.importorskip("pymupdf")

from src.core.pdf import (  # noqa: E402
    PARAGRAPH_STYLES,
    PdfEncryptedError,
    PdfNoTextLayerError,
    PdfOpenError,
)
from src.core.pdf import extractor  # noqa: E402
from src.core.pdf.extractor import (  # noqa: E402
    extract_pdf_paragraphs,
    extract_pdf_text,
    join_lines,
)

ALL_TEXT_FIXTURES = ["simple_pdf_path", "multipage_pdf_path", "list_pdf_path", "image_pdf_path"]


def _assert_contract(content):
    """The PdfPlainContent contract shared by every successful extraction."""
    assert len(content.paragraphs_text) == len(content.paragraphs_style)
    assert all(style in PARAGRAPH_STYLES for style in content.paragraphs_style)
    assert all(text and text == text.strip() for text in content.paragraphs_text)
    assert all("\n" not in text for text in content.paragraphs_text)
    for key, images in content.images_by_paragraph.items():
        assert -1 <= key <= len(content.paragraphs_text) - 1
        assert all(image.ext in {"png", "jpeg"} for image in images)


# --- 1. Simple document: heading + two body paragraphs -----------------------


def test_simple_pdf_paragraphs_and_styles(simple_pdf_path):
    content = extract_pdf_paragraphs(simple_pdf_path)

    assert content.paragraphs_style == ["heading1", "normal", "normal"]
    assert content.paragraphs_text == [
        "A Simple Title",
        "This is the first body paragraph of the document. "
        "It spans two lines of ordinary running text.",
        "Here is the second body paragraph, a separate block.",
    ]
    assert content.page_count == 1
    assert content.page_size == (595, 842)
    _assert_contract(content)


# --- 2. Header/footer removal and page-boundary reassembly -------------------


def test_multipage_drops_headers_and_page_numbers(multipage_pdf_path):
    content = extract_pdf_paragraphs(multipage_pdf_path)

    assert "My Book" not in content.paragraphs_text
    assert not any(text in {"1", "2", "3", "4"} for text in content.paragraphs_text)
    assert not any("My Book" in text for text in content.paragraphs_text)
    assert content.page_count == 4
    _assert_contract(content)


def test_multipage_rejoins_hyphenated_page_break(multipage_pdf_path):
    content = extract_pdf_paragraphs(multipage_pdf_path)

    # The page break is mended (one paragraph, no stray space) and the
    # line-end hyphen is dropped: the document never spells "conti-nuation"
    # with a hyphen elsewhere (see test_extractor_hyphenation).
    merged = [text for text in content.paragraphs_text if "the continuation of the sentence." in text]
    assert len(merged) == 1
    # Pages 1 and 2 form one paragraph; pages 3 and 4 start with a capital
    assert len(content.paragraphs_text) == 3


def test_repeated_header_kept_below_min_pages(tmp_path):
    """The repetition rule needs REPEAT_MIN_PAGES pages; a lone header line survives."""
    doc = pymupdf.open()
    for _ in range(2):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 30), "My Book", fontsize=9, fontname="helv")
        page.insert_text((72, 120), "Some body text that is long enough to count.", fontsize=11, fontname="helv")
    path = tmp_path / "two_pages.pdf"
    doc.save(str(path))
    doc.close()

    content = extract_pdf_paragraphs(str(path))
    assert content.paragraphs_text.count("My Book") == 2


# --- 3. List splitting -------------------------------------------------------


def test_list_block_split_into_list_items(list_pdf_path):
    content = extract_pdf_paragraphs(list_pdf_path)

    assert content.paragraphs_style == ["list", "list", "list"]
    assert content.paragraphs_text[0].startswith("\u2022 first item")
    assert content.paragraphs_text[1].startswith("\u2022 second item")
    assert content.paragraphs_text[2].startswith("1) third item")
    _assert_contract(content)


# --- 4. Images ---------------------------------------------------------------


def test_image_kept_and_anchored(image_pdf_path):
    content = extract_pdf_paragraphs(image_pdf_path)

    assert list(content.images_by_paragraph) == [0]
    images = content.images_by_paragraph[0]
    assert len(images) == 1
    image = images[0]
    assert image.ext == "png"
    assert image.width_pt == pytest.approx(200, abs=1)
    assert image.height_pt == pytest.approx(100, abs=1)
    assert image.data.startswith(b"\x89PNG")
    assert len(content.paragraphs_text) == 2
    _assert_contract(content)


def test_images_skipped_when_not_requested(image_pdf_path):
    content = extract_pdf_paragraphs(image_pdf_path, include_images=False)
    assert content.images_by_paragraph == {}
    assert len(content.paragraphs_text) == 2


def test_unsupported_image_format_converted_to_png():
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), (10, 120, 30)).save(buffer, format="BMP")
    block = {"bbox": (0, 0, 40, 30), "image": buffer.getvalue(), "ext": "bmp"}

    image = extractor._to_pdf_image(block)
    assert image is not None
    assert image.ext == "png"
    assert image.data.startswith(b"\x89PNG")


def test_unconvertible_image_dropped():
    block = {"bbox": (0, 0, 40, 30), "image": b"garbage", "ext": "jbig2"}
    assert extractor._to_pdf_image(block) is None


# --- 5. Scanned document -----------------------------------------------------


def test_scanned_pdf_raises_no_text_layer(scanned_pdf_path):
    with pytest.raises(PdfNoTextLayerError, match="no extractable text layer"):
        extract_pdf_paragraphs(scanned_pdf_path)
    assert extract_pdf_text(scanned_pdf_path) == ""


# --- 6. Encrypted document ---------------------------------------------------


def test_encrypted_pdf_raises(encrypted_pdf_path):
    with pytest.raises(PdfEncryptedError, match="password-protected"):
        extract_pdf_paragraphs(encrypted_pdf_path)
    assert extract_pdf_text(encrypted_pdf_path) == ""


# --- 7. Not a PDF ------------------------------------------------------------


def test_garbage_bytes_raise_open_error():
    with pytest.raises(PdfOpenError):
        extract_pdf_paragraphs(b"not a pdf")
    assert extract_pdf_text(b"not a pdf") == ""


def test_missing_file_raises_open_error(tmp_path):
    with pytest.raises(PdfOpenError):
        extract_pdf_paragraphs(str(tmp_path / "missing.pdf"))


def test_non_pdf_document_path_raises_open_error(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("A plain text file that PyMuPDF could open as a document.\n", encoding="utf-8")
    with pytest.raises(PdfOpenError):
        extract_pdf_paragraphs(str(path))


# --- 8. join_lines rules -----------------------------------------------------


@pytest.mark.parametrize(
    "prev, nxt, expected",
    [
        # (a) soft hyphen
        ("hyphen\u00ad", "  ation", "hyphenation"),
        ("Soft\u00ad", "Capital", "SoftCapital"),
        # (b) letter + hyphen, next line starts with a letter: the hyphen is
        # kept and no space is added (dropping it needs whole-document evidence)
        ("the conti-", "nuation", "the conti-nuation"),
        ("the conti-", "  nuation", "the conti-nuation"),
        ("well-", "Known", "well-Known"),
        # (b) does not apply: hyphen not after a letter
        ("pages 10-", "twelve", "pages 10- twelve"),
        # (c) CJK joined without a space
        ("中文", "段落", "中文段落"),
        ("日本語の", "文章", "日本語の文章"),
        ("中文 ", " 段落", "中文段落"),
        # (d) Hangul and everything else joined with one space
        ("한국어", "문장", "한국어 문장"),
        ("first line ", "  second line", "first line second line"),
    ],
)
def test_join_lines(prev, nxt, expected):
    assert join_lines(prev, nxt) == expected


# --- 9. extract_pdf_text cap and input types ---------------------------------


def test_extract_pdf_text_hard_cap(simple_pdf_path):
    text = extract_pdf_text(simple_pdf_path, hard_cap=10)
    assert len(text) == 10
    assert text == extract_pdf_text(simple_pdf_path)[:10]


def test_extract_pdf_text_joins_paragraphs(simple_pdf_path):
    content = extract_pdf_paragraphs(simple_pdf_path)
    assert extract_pdf_text(simple_pdf_path) == "\n\n".join(content.paragraphs_text)


@pytest.mark.parametrize("fixture_name", ALL_TEXT_FIXTURES)
def test_bytes_and_path_inputs_identical(fixture_name, request):
    path = request.getfixturevalue(fixture_name)
    data = Path(path).read_bytes()

    from_path = extract_pdf_paragraphs(path)
    from_bytes = extract_pdf_paragraphs(data)
    assert from_path == from_bytes
    assert extract_pdf_text(path) == extract_pdf_text(data)


@pytest.mark.parametrize("fixture_name", ALL_TEXT_FIXTURES)
def test_extraction_is_deterministic(fixture_name, request):
    path = request.getfixturevalue(fixture_name)
    assert extract_pdf_paragraphs(path) == extract_pdf_paragraphs(path)


def test_package_extract_pdf_text_without_pymupdf(monkeypatch, simple_pdf_path):
    """The package-level wrapper never raises, even when PyMuPDF is missing."""
    monkeypatch.setitem(sys.modules, "pymupdf", None)
    monkeypatch.setitem(sys.modules, "fitz", None)
    for name in [m for m in sys.modules if m == "src.core.pdf" or m.startswith("src.core.pdf.")]:
        monkeypatch.delitem(sys.modules, name)

    package = importlib.import_module("src.core.pdf")
    assert package.extract_pdf_text(simple_pdf_path) == ""
    with pytest.raises(ImportError):
        package.extract_pdf_paragraphs(simple_pdf_path)


# --- 10. No newline in any paragraph, on every fixture -----------------------


@pytest.mark.parametrize("fixture_name", ALL_TEXT_FIXTURES)
def test_no_newline_in_any_paragraph(fixture_name, request):
    content = extract_pdf_paragraphs(request.getfixturevalue(fixture_name))
    assert content.paragraphs_text
    _assert_contract(content)


def test_no_newline_with_ligatures_soft_hyphens_and_mixed_lists(tmp_path):
    """Normalisation runs on list-split and merged paragraphs too."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_font(fontname="F0", fontbuffer=pymupdf.Font("helv").buffer)
    page.insert_text(
        (72, 120),
        "Intro text with an \ufb01ne ligature and a soft\u00ad\nhyphen, then\n"
        "\u2022 an item that wraps\nonto a second line\n- dash item",
        fontsize=11, fontname="F0",
    )
    page.insert_text((72, 260), "continued in a new block.", fontsize=11, fontname="F0")
    path = tmp_path / "mixed.pdf"
    doc.save(str(path))
    doc.close()

    content = extract_pdf_paragraphs(str(path))
    _assert_contract(content)
    assert content.paragraphs_text == [
        "Intro text with an fine ligature and a softhyphen, then",
        "\u2022 an item that wraps onto a second line",
        "- dash item",
        # A list item never absorbs the next block, even a lowercase one
        "continued in a new block.",
    ]
    assert content.paragraphs_style == ["normal", "list", "list", "normal"]


def test_heading_levels_from_size_and_bold(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 80), "Second Level", fontsize=15, fontname="helv")
    page.insert_text((72, 130), "Third Level", fontsize=13, fontname="helv")
    page.insert_text((72, 180), "Bold Body Heading", fontsize=11, fontname="hebo")
    page.insert_text(
        (72, 230),
        "Body text written at the regular size, long enough to dominate the\n"
        "size statistics of this small page so that eleven points is the body.",
        fontsize=11, fontname="helv",
    )
    page.insert_text((72, 300), "A bold sentence that ends with a period.", fontsize=11, fontname="hebo")
    path = tmp_path / "headings.pdf"
    doc.save(str(path))
    doc.close()

    content = extract_pdf_paragraphs(str(path))
    assert content.paragraphs_style == ["heading2", "heading3", "heading3", "normal", "normal"]


# --- Internal rules ----------------------------------------------------------


def test_merge_rule():
    assert extractor._should_merge("a sentence without end", "normal", "continues here")
    assert extractor._should_merge("hyphen-", "normal", "ated")
    assert not extractor._should_merge("A full sentence.", "normal", "lowercase start")
    assert not extractor._should_merge("no end", "normal", "Uppercase start")
    assert not extractor._should_merge("A heading", "heading2", "lowercase start")
    assert not extractor._should_merge("\u2022 a list item", "list", "lowercase start")


def test_page_number_regex():
    for text in ["1", "42", "iv", "Page 3", "page 3 of 10", "3 / 10"]:
        assert extractor.PAGE_NUMBER_RE.fullmatch(text), text
    for text in ["My Book", "Chapter 1", "1984 was a year"]:
        assert not extractor.PAGE_NUMBER_RE.fullmatch(text), text


def test_list_marker_regex_excludes_dialogue_dashes():
    assert extractor.LIST_MARKER_RE.match("\u2022 item")
    assert extractor.LIST_MARKER_RE.match("- item")
    assert extractor.LIST_MARKER_RE.match("(a) item")
    assert not extractor.LIST_MARKER_RE.match("— Bonjour, dit-il.")
    assert not extractor.LIST_MARKER_RE.match("– Hola.")
