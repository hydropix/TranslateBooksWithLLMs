"""Tests for the PDF branch of the unified translate_file() router."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.core.adapters.translate_file import get_file_type_from_path, translate_file
from src.core.job import JobStatus
from tests.job_helpers import make_ctx, make_job


def _checkpoint_manager_without_job():
    manager = Mock()
    manager.db.get_job.return_value = None
    return manager


async def _route(input_path, output_path, checkpoint_manager):
    return await translate_file(
        make_job(input_path, output_path, provider="ollama", model="test-model",
                 endpoint="http://localhost:11434/api/generate"),
        make_ctx(checkpoint_manager=checkpoint_manager, translation_id="job1"),
    )


@pytest.mark.asyncio
async def test_pdf_routes_to_translate_pdf_file(simple_pdf_path, tmp_path):
    output = str(tmp_path / "out.pdf")
    manager = _checkpoint_manager_without_job()

    async def fake_pdf(job, ctx, **kwargs):
        with open(job.output_path, "wb") as f:
            f.write(b"%PDF")
        return {'success': True, 'stats': {}, 'output_path': job.output_path}

    pdf_mock = AsyncMock(side_effect=fake_pdf)

    with patch("src.core.pdf.translator.translate_pdf_file", pdf_mock):
        result = await _route(simple_pdf_path, output, manager)

    assert result.status is JobStatus.COMPLETED
    assert result.output_path == output
    pdf_mock.assert_awaited_once()
    job, ctx = pdf_mock.await_args.args
    assert job.input_path == simple_pdf_path
    assert job.output_path == output
    assert ctx.checkpoint_manager is manager
    assert ctx.translation_id == "job1"
    # The PDF engine builds its own client from the job.
    assert "llm_client" not in pdf_mock.await_args.kwargs
    # The parent checkpoint job is created for PDF, like EPUB/DOCX
    manager.start_job.assert_called_once()
    assert manager.start_job.call_args.kwargs["file_type"] == "pdf"


@pytest.mark.asyncio
async def test_pdf_interrupted_returns_false(simple_pdf_path, tmp_path):
    output = str(tmp_path / "out.pdf")
    pdf_mock = AsyncMock(return_value={
        'success': False, 'stats': {}, 'output_path': None, 'interrupted': True,
    })

    with patch("src.core.pdf.translator.translate_pdf_file", pdf_mock):
        result = await _route(simple_pdf_path, output, _checkpoint_manager_without_job())

    assert not result
    pdf_mock.assert_awaited_once()


def test_get_file_type_from_path_pdf():
    assert get_file_type_from_path("a.PDF") == "pdf"
    assert get_file_type_from_path("dir/paper.pdf") == "pdf"
    # Existing mappings are unchanged
    assert get_file_type_from_path("a.docx") == "docx"
    assert get_file_type_from_path("a.epub") == "epub"
