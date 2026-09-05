"""
Unit tests for HTTP error handling in OllamaProvider.generate().

Regression tests for item 2 of issue #231: raise_for_status() runs inside the
client.stream(...) block, before the body has been fetched, so reading the
error payload from the exception handler raised httpx.ResponseNotRead. The
bare except swallowed it, the "context"/"length" keyword check never matched,
and Ollama's num_ctx-overflow 400 was retried as a generic HTTP error instead
of being raised as ContextOverflowError.

https://github.com/hydropix/TranslateBooksWithLLMs/issues/231
"""
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.llm.exceptions import ContextOverflowError
from src.core.llm.providers.ollama import OllamaProvider
from src.core.llm.thinking.behavior import ThinkingBehavior


def _streaming_response(status_code, body, content_type="application/json"):
    """Build a response whose body is only available after an explicit read.

    httpx.Response(content=b"...") pre-populates _content, which would hide the
    bug under test. An async iterator reproduces a real streamed response:
    .json() raises ResponseNotRead until aread() is awaited.
    """
    async def stream():
        yield body

    return httpx.Response(
        status_code,
        headers={"Content-Type": content_type},
        content=stream(),
    )


def _provider_with_transport(handler):
    provider = OllamaProvider(
        api_endpoint="http://localhost:11434/api/chat",
        model="test-model",
        context_window=2048,
    )
    # Skip the thinking-behavior probe: it would issue its own request.
    provider._thinking_behavior = ThinkingBehavior.STANDARD
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return provider


@pytest.mark.asyncio
async def test_context_overflow_400_raises_context_overflow_error():
    """Ollama's num_ctx overflow must surface as ContextOverflowError."""
    def handler(request):
        return _streaming_response(
            400,
            b'{"error": "input length exceeds context length"}',
        )

    provider = _provider_with_transport(handler)
    try:
        with pytest.raises(ContextOverflowError) as excinfo:
            await provider.generate("prompt")
        assert "context length" in str(excinfo.value)
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_context_overflow_detected_from_plain_text_body():
    """A non-JSON error body is still inspected for the overflow keywords."""
    def handler(request):
        return _streaming_response(
            400,
            b"prompt is too long for this model",
            content_type="text/plain",
        )

    provider = _provider_with_transport(handler)
    try:
        with pytest.raises(ContextOverflowError):
            await provider.generate("prompt")
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_unrelated_http_error_is_retried_then_gives_up(monkeypatch):
    """A 500 with no overflow keyword stays a plain retryable HTTP error."""
    monkeypatch.setattr("src.core.llm.providers.ollama.MAX_TRANSLATION_ATTEMPTS", 2)
    monkeypatch.setattr("asyncio.sleep", _no_sleep)

    calls = []

    def handler(request):
        calls.append(request)
        return _streaming_response(500, b'{"error": "internal server error"}')

    provider = _provider_with_transport(handler)
    try:
        assert await provider.generate("prompt") is None
        assert len(calls) == 2
    finally:
        await provider.close()


async def _no_sleep(_seconds):
    return None
