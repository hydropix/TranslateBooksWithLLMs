"""Regression tests: translate_epub_file reports whether the book was translated.

It used to return None in every case (client-creation failure, swallowed
exception, partial result), and the caller reported EPUB jobs as successful
unconditionally. The contract is now `translate_epub_file(...) -> bool`.
"""
from pathlib import Path

import pytest

import src.core.epub.translator as translator_module
from src.core.epub.translator import _create_llm_client, translate_epub_file
from src.core.llm.exceptions import LLMClientConfigError, RateLimitError

from tests.unit.epub.conftest import _disable_attribution, _echo_llm_client


async def _run(input_epub: Path, tmp_path: Path, monkeypatch, client_factory=None):
    monkeypatch.setattr(
        translator_module, "_create_llm_client",
        client_factory or (lambda **kwargs: _echo_llm_client()),
    )
    _disable_attribution(monkeypatch)
    events = []
    result = await translate_epub_file(
        input_filepath=str(input_epub),
        output_filepath=str(tmp_path / "out.epub"),
        source_language="Chinese",
        target_language="French",
        log_callback=lambda event, message, **_k: events.append((event, message)),
    )
    return result, events


@pytest.mark.asyncio
async def test_complete_book_returns_true(tmp_path, input_epub, monkeypatch):
    result, _ = await _run(input_epub, tmp_path, monkeypatch)
    assert result is True
    assert (tmp_path / "out.epub").exists()


@pytest.mark.asyncio
async def test_missing_input_returns_false(tmp_path, monkeypatch):
    result, _ = await _run(tmp_path / "missing.epub", tmp_path, monkeypatch)
    assert result is False


@pytest.mark.asyncio
async def test_client_creation_failure_returns_false(tmp_path, input_epub, monkeypatch):
    def failing_factory(**kwargs):
        raise LLMClientConfigError("Gemini provider ('gemini') requires an API key.")

    result, _ = await _run(input_epub, tmp_path, monkeypatch, client_factory=failing_factory)
    assert result is False
    assert not (tmp_path / "out.epub").exists()


@pytest.mark.asyncio
async def test_unexpected_error_returns_false(tmp_path, input_epub, monkeypatch):
    async def boom(**kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(translator_module, "_process_all_content_files", boom)
    result, events = await _run(input_epub, tmp_path, monkeypatch)
    assert result is False
    assert any(event == "epub_major_error" for event, _ in events)


@pytest.mark.asyncio
async def test_failed_file_returns_false(tmp_path, input_epub, monkeypatch):
    async def failing_file(**kwargs):
        return None, False, None

    monkeypatch.setattr(translator_module, "_translate_single_xhtml_file", failing_file)
    result, _ = await _run(input_epub, tmp_path, monkeypatch)
    assert result is False


@pytest.mark.asyncio
async def test_rate_limit_still_propagates(tmp_path, input_epub, monkeypatch):
    async def limited(**kwargs):
        raise RateLimitError("429")

    monkeypatch.setattr(translator_module, "_process_all_content_files", limited)
    with pytest.raises(RateLimitError):
        await _run(input_epub, tmp_path, monkeypatch)


def test_create_llm_client_reports_unknown_provider():
    events = []
    with pytest.raises(LLMClientConfigError):
        _create_llm_client(
            llm_provider="no-such-provider", model_name="m",
            gemini_api_key=None, openai_api_key=None, openrouter_api_key=None,
            mistral_api_key=None, deepseek_api_key=None, poe_api_key=None,
            nim_api_key=None, cli_api_endpoint="http://localhost:1",
            initial_context=2048,
        log_callback=lambda event, message: events.append((event, message)),
        )
    assert events and events[0][0] == "llm_client_error"
    assert "no-such-provider" in events[0][1]
