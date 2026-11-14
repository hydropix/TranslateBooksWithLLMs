"""Regression tests: EPUB chunk errors must reach the job's own log callback.

The error reporter used to send every error to the global logger singleton
whenever a callback was given, so with concurrent jobs the errors reached the
wrong job or none at all. Without a callback it now uses the standard logging
module, never the process-wide unified logger.
"""
import logging

import pytest

from src.core.epub import chunk_fallback
from src.core.epub.engine_log import report_error
from src.core.epub.translation_metrics import TranslationMetrics
from tests.epub_engine_helpers import translate_chunk


def _fail_on_unified_logger(monkeypatch):
    class _Logger:
        def error(self, message):
            raise AssertionError(f"unified logger used: {message}")

    monkeypatch.setattr("src.utils.unified_logger.get_logger", lambda *a, **k: _Logger())


def test_log_error_uses_the_job_callback(monkeypatch):
    _fail_on_unified_logger(monkeypatch)
    received = []
    report_error(lambda event, message: received.append((event, message)),
                 "phase2_error", "boom")
    assert received == [("phase2_error", "boom")]


def test_log_error_without_callback_uses_the_module_logger(monkeypatch, caplog):
    _fail_on_unified_logger(monkeypatch)
    with caplog.at_level(logging.ERROR, logger="src.core.epub.engine_log"):
        report_error(None, "phase2_error", "boom")
    assert [record.getMessage() for record in caplog.records] == ["boom"]


@pytest.mark.asyncio
async def test_concurrent_jobs_keep_their_own_errors(monkeypatch):
    async def failing_request(*args, **kwargs):
        return None

    monkeypatch.setattr(chunk_fallback, "generate_translation_request", failing_request)
    monkeypatch.setattr("src.config.EPUB_TOKEN_ALIGNMENT_ENABLED", False)

    logs = {"a": [], "b": []}

    async def run(job):
        await translate_chunk(
            chunk_text=f"Text of job {job}",
            local_tag_map={},
            global_indices=[],
            source_language="English",
            target_language="French",
            model_name="fake",
            llm_client=object(),
            stats=TranslationMetrics(),
            log_callback=lambda event, message: logs[job].append(event),
            max_retries=1,
        )

    await run("a")
    await run("b")
    for job in ("a", "b"):
        assert "chunk_translation_failed" in logs[job]
        assert "fallback_untranslated" in logs[job]
