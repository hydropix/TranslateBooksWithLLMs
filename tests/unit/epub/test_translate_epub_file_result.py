"""Regression tests: translate_epub_file reports whether the book was translated.

It used to return None in every case (client-creation failure, swallowed
exception, partial result), and the caller reported EPUB jobs as successful
unconditionally. The contract is now
`translate_epub_file(job, ctx) -> JobResult`.
"""
from pathlib import Path

import pytest

import src.core.epub.book_loop as book_loop
import src.core.epub.translator as translator_module
from src.core.epub.book_loop import FileResult
from src.core.epub.llm_setup import create_llm_client
from src.core.epub.translator import translate_epub_file
from src.core.job import JobStatus
from src.core.llm.exceptions import LLMClientConfigError, RateLimitError

from tests.job_helpers import make_ctx, make_job
from tests.unit.epub.conftest import _disable_attribution, _echo_llm_client


async def _run(input_epub: Path, tmp_path: Path, monkeypatch, client_factory=None):
    monkeypatch.setattr(
        translator_module, "create_llm_client",
        client_factory or (lambda *args, **kwargs: _echo_llm_client()),
    )
    _disable_attribution(monkeypatch)
    events = []
    result = await translate_epub_file(
        make_job(str(input_epub), str(tmp_path / "out.epub"), file_type="epub",
                 source_language="Chinese", target_language="French"),
        make_ctx(log_callback=lambda event, message, **_k: events.append((event, message))),
    )
    return result, events


@pytest.mark.asyncio
async def test_complete_book_returns_completed(tmp_path, input_epub, monkeypatch):
    result, _ = await _run(input_epub, tmp_path, monkeypatch)
    assert result.status is JobStatus.COMPLETED
    assert result.output_path == str(tmp_path / "out.epub")
    assert (tmp_path / "out.epub").exists()


@pytest.mark.asyncio
async def test_missing_input_fails(tmp_path, monkeypatch):
    result, _ = await _run(tmp_path / "missing.epub", tmp_path, monkeypatch)
    assert result.status is JobStatus.FAILED
    assert "not found" in result.error


@pytest.mark.asyncio
async def test_client_creation_failure_fails(tmp_path, input_epub, monkeypatch):
    def failing_factory(*args, **kwargs):
        raise LLMClientConfigError("Gemini provider ('gemini') requires an API key.")

    result, _ = await _run(input_epub, tmp_path, monkeypatch, client_factory=failing_factory)
    assert result.status is JobStatus.FAILED
    assert not (tmp_path / "out.epub").exists()


@pytest.mark.asyncio
async def test_unexpected_error_fails(tmp_path, input_epub, monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(translator_module, "process_content_files", boom)
    result, events = await _run(input_epub, tmp_path, monkeypatch)
    assert result.status is JobStatus.FAILED
    assert "disk on fire" in result.error
    assert any(event == "epub_major_error" for event, _ in events)


@pytest.mark.asyncio
async def test_failed_file_is_partial(tmp_path, input_epub, monkeypatch):
    async def failing_file(*args, **kwargs):
        return FileResult(None, False, None, None)

    monkeypatch.setattr(book_loop, "translate_xhtml_file", failing_file)
    result, _ = await _run(input_epub, tmp_path, monkeypatch)
    assert not result.ok
    assert result.status is JobStatus.PARTIAL


@pytest.mark.asyncio
async def test_rate_limit_still_propagates(tmp_path, input_epub, monkeypatch):
    async def limited(*args, **kwargs):
        raise RateLimitError("429")

    monkeypatch.setattr(translator_module, "process_content_files", limited)
    with pytest.raises(RateLimitError):
        await _run(input_epub, tmp_path, monkeypatch)


def test_create_llm_client_reports_unknown_provider():
    events = []
    with pytest.raises(LLMClientConfigError):
        create_llm_client(
            make_job(provider="no-such-provider", model="m", endpoint="http://localhost:1"),
            2048,
            lambda event, message: events.append((event, message)),
        )
    assert events and events[0][0] == "llm_client_error"
    assert "no-such-provider" in events[0][1]
