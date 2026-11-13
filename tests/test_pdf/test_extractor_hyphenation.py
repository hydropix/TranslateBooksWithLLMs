"""
Regression tests for line-end hyphen repair.

Fixtures are generated with PyMuPDF at test time (no binary fixtures).
"""

import pytest

pymupdf = pytest.importorskip("pymupdf")

from src.core.pdf import extractor  # noqa: E402
from src.core.pdf.extractor import extract_pdf_paragraphs, join_lines  # noqa: E402


def _single_page_pdf(tmp_path, body: str, fontname: str = "helv") -> str:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 120), body, fontsize=11, fontname=fontname)
    path = tmp_path / "doc.pdf"
    doc.save(str(path))
    doc.close()
    return str(path)


# --- join_lines without document evidence -------------------------------------


@pytest.mark.parametrize(
    "prev, nxt, expected",
    [
        ("a well-", "known fact", "a well-known fact"),
        ("peut-", "être demain", "peut-être demain"),
        ("Jean-", "Paul", "Jean-Paul"),
        ("hyphen­", "ation", "hyphenation"),
    ],
)
def test_join_lines_keeps_real_compound_hyphens(prev, nxt, expected):
    assert join_lines(prev, nxt) == expected


# --- Whole-document evidence ----------------------------------------------------


def test_compound_hyphen_kept_without_evidence(tmp_path):
    path = _single_page_pdf(
        tmp_path,
        "It is a well-\nknown fact that the river runs north.\n"
        "Il viendra peut-\nêtre demain matin avec ses amis.",
    )

    text = " ".join(extract_pdf_paragraphs(path).paragraphs_text)

    assert "well-known" in text
    assert "wellknown" not in text
    assert "peut-être" in text
    assert "peutêtre" not in text


def test_word_split_at_line_end_rejoined_when_document_confirms_it(tmp_path):
    path = _single_page_pdf(
        tmp_path,
        "The continuation of the story was long awaited by all.\n"
        "Everyone hoped the conti-\nnuation would arrive soon enough.",
    )

    text = " ".join(extract_pdf_paragraphs(path).paragraphs_text)

    assert "hoped the continuation would" in text
    assert "conti-nuation" not in text


def test_hyphenated_spelling_elsewhere_wins_over_joined_spelling(tmp_path):
    path = _single_page_pdf(
        tmp_path,
        "A well-known saying and a wellknown typo appear here once.\n"
        "Then it is a well-\nknown saying again in this line.",
    )

    text = " ".join(extract_pdf_paragraphs(path).paragraphs_text)

    assert text.count("well-known saying") == 2


def test_no_break_marker_leaks_into_output(tmp_path):
    path = _single_page_pdf(tmp_path, "Some text ending in a dangling well-\n42 numbers follow.")

    for text in extract_pdf_paragraphs(path).paragraphs_text:
        assert extractor._BREAK_HYPHEN not in text
