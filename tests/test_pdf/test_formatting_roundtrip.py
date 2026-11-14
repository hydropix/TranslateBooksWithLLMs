"""Round trip: extract a formatted PDF, rebuild it, and check the formatting survived."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

pymupdf = pytest.importorskip("pymupdf")

from src.core.epub.translation_metrics import TranslationMetrics
from src.core.pdf.builder import build_pdf
from src.core.pdf.extractor import extract_pdf_paragraphs
from src.core.pdf.translator import translate_pdf_file
from tests.job_helpers import make_ctx, make_job

PIPELINE = "src.core.pdf.translator.translate_paragraphs_plain"


def _channels(value):
    """0xRRGGBB int -> (r, g, b) on 0..255."""
    return (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF


def _close_rgb(actual, expected, tolerance=1):
    return all(abs(a - e) <= tolerance for a, e in zip(actual, expected))


def _fill_rgb(drawing):
    fill = drawing.get("fill")
    if fill is None:
        return None
    return tuple(round(c * 255) for c in fill)


def _all_spans(doc):
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    yield span


def _baseline_y(doc, needle):
    """Baseline y of the first span containing `needle` (first match across pages)."""
    for span in _all_spans(doc):
        if needle in span["text"]:
            return span["origin"][1]
    raise AssertionError(f"{needle!r} not found in the rebuilt PDF")


@pytest.fixture
def rebuilt(formatted_pdf_path, tmp_path):
    content = extract_pdf_paragraphs(formatted_pdf_path)
    output = str(tmp_path / "rebuilt.pdf")
    build_pdf([f"T:{p}" for p in content.paragraphs_text], content, output, "French")
    with pymupdf.open(output) as doc:
        yield content, doc


def test_page_count(rebuilt):
    _, doc = rebuilt
    assert doc.page_count >= 1


def test_eyebrow_keeps_colour_and_monospace_font(rebuilt):
    _, doc = rebuilt
    eyebrows = [
        s for s in _all_spans(doc)
        if _close_rgb(_channels(s["color"]), _channels(0x00897C)) and "Mono" in s["font"]
    ]
    assert eyebrows, "no teal monospace span in the rebuilt PDF"


def test_heading_size_is_kept(rebuilt):
    _, doc = rebuilt
    assert any(abs(s["size"] - 20) <= 1.5 for s in _all_spans(doc) if s["text"].strip())


def test_box_and_bar_fills_are_drawn(rebuilt):
    _, doc = rebuilt
    fills = [_fill_rgb(d) for page in doc for d in page.get_drawings()]
    fills = [f for f in fills if f is not None]
    assert any(_close_rgb(f, _channels(0xF3F6F5)) for f in fills), "no #f3f6f5 box fill"
    assert any(_close_rgb(f, _channels(0x9A5B2D)) for f in fills), "no #9a5b2d bar fill"


def test_table_cells_share_their_row(rebuilt):
    content, doc = rebuilt
    rows = content.tables[0].rows
    texts = content.paragraphs_text
    for needle in ("L1222-3", "ARTICLE"):
        index = texts.index(needle)
        row = next(r for r in rows if index in r)
        mates = [i for i in row if i is not None and i != index]
        assert mates, f"{needle} has no row-mates"
        y = _baseline_y(doc, f"T:{needle}")
        for mate in mates:
            mate_y = _baseline_y(doc, f"T:{texts[mate]}"[:12])
            assert abs(mate_y - y) <= 1, f"{needle} and {texts[mate]!r} are on different rows"


@pytest.mark.asyncio
async def test_translate_pdf_file_logs_table_count(formatted_pdf_path, tmp_path):
    async def _run(*args, paragraphs, **kwargs):
        return [f"T:{p}" for p in paragraphs], TranslationMetrics(), False

    log = Mock()
    output = str(tmp_path / "out.pdf")
    with patch(PIPELINE, AsyncMock(side_effect=_run)):
        result = await translate_pdf_file(
            make_job(
                input_path=formatted_pdf_path,
                output_path=output,
                source_language="English",
                target_language="French",
                model="test-model",
            ),
            make_ctx(log_callback=log),
            llm_client=Mock(),
        )

    assert result["success"] is True
    messages = {c.args[0]: c.args[1] for c in log.call_args_list}
    assert "2 tables" in messages["pdf_extracted"]
