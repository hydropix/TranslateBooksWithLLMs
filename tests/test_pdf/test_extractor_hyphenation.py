"""
Regression tests for line-end hyphen repair and caseless cross-block merges.

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


def _two_page_pdf(tmp_path, first: str, second: str, fontname: str) -> str:
    doc = pymupdf.open()
    for body in (first, second):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 120), body, fontsize=11, fontname=fontname)
    path = tmp_path / "pages.pdf"
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


# --- Caseless scripts across blocks/pages ------------------------------------------


@pytest.mark.parametrize(
    "prev_text, text",
    [
        ("这是一个没有结束的句子", "继续写在下一页上。"),
        ("هذه جملة لم تنته", "بعد وتستمر هنا."),
        ("זה משפט שלא נגמר", "וממשיך כאן."),
    ],
)
def test_caseless_continuation_merges(prev_text, text):
    assert extractor._should_merge(prev_text, "normal", text)


@pytest.mark.parametrize(
    "prev_text, text",
    [
        ("这是一个完整的句子。", "新的段落开始了。"),
        ("这是一个问题吗？", "新的段落开始了。"),
        ("هل هذا سؤال؟", "فقرة جديدة."),
        ("Une phrase sans fin", "Majuscule ensuite"),
    ],
)
def test_terminated_or_capitalised_paragraphs_do_not_merge(prev_text, text):
    assert not extractor._should_merge(prev_text, "normal", text)


def test_cjk_paragraph_split_by_page_break_is_reassembled(tmp_path):
    path = _two_page_pdf(
        tmp_path,
        "这是第一页上的一个很长的句子，它在页面的末尾还没有结束，而且",
        "在第二页上继续写下去，直到这里才终于写完了这个句子。",
        fontname="china-s",
    )

    content = extract_pdf_paragraphs(path)

    assert content.paragraphs_text == [
        "这是第一页上的一个很长的句子，它在页面的末尾还没有结束，而且"
        "在第二页上继续写下去，直到这里才终于写完了这个句子。"
    ]
