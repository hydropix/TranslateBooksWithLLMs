"""Tests for translate_pdf_file (extractor + plain pipeline + builder)."""

import os
import unicodedata
from unittest.mock import AsyncMock, Mock, patch

import pytest

pymupdf = pytest.importorskip("pymupdf")

from src.core.epub.translation_metrics import TranslationMetrics
from src.core.pdf.content import PdfNoTextLayerError
from src.core.pdf.extractor import extract_pdf_paragraphs
from src.core.pdf.translator import translate_pdf_file
from tests.job_helpers import make_ctx, make_job

PIPELINE = "src.core.pdf.translator.translate_paragraphs_plain"


def _fake_pipeline(was_interrupted=False):
    """AsyncMock standing in for translate_paragraphs_plain."""
    async def _run(*args, paragraphs, **kwargs):
        return [f"T:{p}" for p in paragraphs], TranslationMetrics(), was_interrupted

    return AsyncMock(side_effect=_run)


def _normalized_text(path):
    """Full text with whitespace collapsed and ligatures (e.g. U+FB01) expanded."""
    with pymupdf.open(path) as doc:
        text = "".join(page.get_text() for page in doc)
    return " ".join(unicodedata.normalize("NFKC", text).split())


_CTX_FIELDS = ("log_callback", "checkpoint_manager", "translation_id", "stats_callback")


async def _translate(input_path, output_path, **kwargs):
    ctx_fields = {k: kwargs.pop(k) for k in _CTX_FIELDS if k in kwargs}
    return await translate_pdf_file(
        make_job(input_path, output_path, model="test-model",
                 source_language="English", target_language="French", **kwargs),
        make_ctx(**ctx_fields),
        llm_client=Mock(),
    )


@pytest.mark.asyncio
async def test_success_writes_translated_pdf(simple_pdf_path, tmp_path):
    output = str(tmp_path / "out.pdf")
    first = extract_pdf_paragraphs(simple_pdf_path).paragraphs_text[0]
    log = Mock()

    with patch(PIPELINE, _fake_pipeline()) as pipeline:
        result = await _translate(simple_pdf_path, output, log_callback=log)

    pipeline.assert_awaited_once()
    assert result["success"] is True
    assert result["output_path"] == output
    assert isinstance(result["stats"], dict)
    assert os.path.exists(output)
    assert f"T:{first}" in _normalized_text(output)
    keys = [c.args[0] for c in log.call_args_list]
    assert "pdf_extracted" in keys
    assert "pdf_rebuilt" in keys


@pytest.mark.asyncio
async def test_interrupted_writes_no_output(simple_pdf_path, tmp_path):
    output = str(tmp_path / "out.pdf")
    log = Mock()

    with patch(PIPELINE, _fake_pipeline(was_interrupted=True)):
        result = await _translate(simple_pdf_path, output, log_callback=log)

    assert result["success"] is False
    assert result["interrupted"] is True
    assert result["output_path"] is None
    assert not os.path.exists(output)
    assert "pdf_interrupted" in [c.args[0] for c in log.call_args_list]


@pytest.mark.asyncio
async def test_scanned_pdf_raises_and_logs(scanned_pdf_path, tmp_path):
    output = str(tmp_path / "out.pdf")
    log = Mock()

    with patch(PIPELINE, _fake_pipeline()) as pipeline:
        with pytest.raises(PdfNoTextLayerError):
            await _translate(scanned_pdf_path, output, log_callback=log)

    pipeline.assert_not_awaited()
    assert "pdf_extraction_failed" in [c.args[0] for c in log.call_args_list]
    assert not os.path.exists(output)


@pytest.mark.asyncio
async def test_bilingual_keeps_source_and_translation(simple_pdf_path, tmp_path):
    output = str(tmp_path / "out.pdf")
    paragraphs = extract_pdf_paragraphs(simple_pdf_path).paragraphs_text

    with patch(PIPELINE, _fake_pipeline()):
        result = await _translate(
            simple_pdf_path, output, prompt_options={"bilingual": True}
        )

    assert result["success"] is True
    text = _normalized_text(output)
    for paragraph in paragraphs:
        assert f"T:{paragraph}" in text
        # The source paragraph appears on its own, not only inside "T:..."
        assert text.count(paragraph) >= 2


@pytest.mark.asyncio
async def test_checkpoint_deleted_only_after_success(simple_pdf_path, tmp_path):
    expected_href = os.path.basename(simple_pdf_path)

    manager = Mock()
    manager.load_xhtml_partial_state.return_value = None
    with patch(PIPELINE, _fake_pipeline()):
        result = await _translate(
            simple_pdf_path, str(tmp_path / "ok.pdf"),
            checkpoint_manager=manager, translation_id="job1",
        )
    assert result["success"] is True
    manager.load_xhtml_partial_state.assert_called_once_with("job1", expected_href)
    manager.delete_xhtml_partial_state.assert_called_once_with("job1", expected_href)

    manager = Mock()
    manager.load_xhtml_partial_state.return_value = None
    with patch(PIPELINE, _fake_pipeline(was_interrupted=True)):
        result = await _translate(
            simple_pdf_path, str(tmp_path / "interrupted.pdf"),
            checkpoint_manager=manager, translation_id="job1",
        )
    assert result["interrupted"] is True
    manager.delete_xhtml_partial_state.assert_not_called()
