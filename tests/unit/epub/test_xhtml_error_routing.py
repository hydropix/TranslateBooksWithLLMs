"""Regression tests: EPUB chunk errors must reach the job's own log callback.

_log_error used to send every error to the global logger singleton whenever a
callback was given, so with concurrent jobs the errors reached the wrong job
or none at all.
"""
import pytest

import src.core.epub.xhtml_translator as xt
from src.core.epub.translation_metrics import TranslationMetrics


def test_log_error_uses_the_job_callback(monkeypatch):
    global_errors = []

    class _Logger:
        def error(self, message):
            global_errors.append(message)

    monkeypatch.setattr("src.utils.unified_logger.get_logger", lambda: _Logger())
    received = []
    xt._log_error(lambda event, message: received.append((event, message)),
                  "phase2_error", "boom")
    assert received == [("phase2_error", "boom")]
    assert global_errors == []


def test_log_error_without_callback_falls_back_to_global_logger(monkeypatch):
    global_errors = []

    class _Logger:
        def error(self, message):
            global_errors.append(message)

    monkeypatch.setattr("src.utils.unified_logger.get_logger", lambda: _Logger())
    xt._log_error(None, "phase2_error", "boom")
    assert global_errors == ["boom"]


@pytest.mark.asyncio
async def test_concurrent_jobs_keep_their_own_errors(monkeypatch):
    async def failing_request(*args, **kwargs):
        return None

    monkeypatch.setattr(xt, "generate_translation_request", failing_request)
    monkeypatch.setattr("src.config.EPUB_TOKEN_ALIGNMENT_ENABLED", False)

    logs = {"a": [], "b": []}

    async def run(job):
        await xt.translate_chunk_with_fallback(
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
