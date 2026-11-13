"""
Unit tests for the Ollama provider's HTTP 429 handling.

Covers (issue #279 - an EPUB finished with 271 untranslated chunks because a
rate-limited Ollama Cloud daemon was retried twice and then degraded to None,
which the pipeline silently interpreted as "keep the source text"):
    - 429 with a Retry-After hint -> RateLimitError carrying retry_after
    - 429 without any rate-limit header -> RateLimitError with retry_after=None
    - 429 body text (quota message) survives into the raised exception
    - 429 fails fast: exactly one request, no retry, no sleep
    - 500 keeps the historical retry-then-None behaviour (regression guard)
    - 200 streaming happy path still returns an LLMResponse
"""
import asyncio
import json

import httpx
import pytest

from src.core.llm.exceptions import RateLimitError
from src.core.llm.providers import ollama as ollama_mod
from src.core.llm.thinking.behavior import ThinkingBehavior

TEST_ENDPOINT = "http://localhost:11434/api/chat"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeStreamResponse:
    """Minimal stand-in for a streamed 200 response from /api/chat.

    Ollama answers with NDJSON: one JSON object per line, the last one carrying
    done=true plus the token counters.
    """

    def __init__(self, lines, status_code=200, headers=None):
        self._lines = list(lines)
        self.status_code = status_code
        self.headers = httpx.Headers(headers or {})
        self.closed = False

    @property
    def is_error(self):
        return self.status_code >= 400

    @property
    def text(self):
        return "\n".join(self._lines)

    def json(self):
        return json.loads(self._lines[-1]) if self._lines else {}

    async def aread(self):
        return self.text.encode("utf-8")

    def raise_for_status(self):
        return None

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aclose(self):
        self.closed = True


class _FakeStreamContext:
    """Async context manager returned by the fake client's stream()."""

    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeClient:
    """Records every stream() call and hands back a caller-supplied response."""

    def __init__(self, responder):
        self._responder = responder
        self.stream_calls = []

    def stream(self, method, url, json=None, timeout=None):
        self.stream_calls.append((method, url))
        return _FakeStreamContext(self._responder())


class _AsyncioShim:
    """Proxies the real asyncio module but records sleeps instead of waiting.

    Patched over the provider module's `asyncio` name so the retry sleeps become
    instant and observable, without touching asyncio globally (pytest-asyncio
    itself relies on the real one).
    """

    def __init__(self, sleeps):
        self.sleeps = sleeps

    def __getattr__(self, name):
        return getattr(asyncio, name)

    async def sleep(self, delay, *args, **kwargs):
        self.sleeps.append(delay)


def _make_error_response(status_code, headers=None, body=None):
    """Build a real httpx.Response so raise_for_status() raises a real
    HTTPStatusError and .headers behaves like real (case-insensitive) headers."""
    request = httpx.Request("POST", TEST_ENDPOINT)
    return httpx.Response(
        status_code,
        headers=headers or {},
        json=body if body is not None else {"error": "rate limited"},
        request=request,
    )


def _make_provider(monkeypatch, responder, max_attempts=2):
    """Build an OllamaProvider wired to a fake client, with thinking detection
    short-circuited and the retry sleeps neutralised."""
    from src.core.llm import base as base_mod
    from src.core.llm.policy import RetryPolicy

    sleeps = []
    # The retry loop (and its sleeps) lives in the provider base.
    monkeypatch.setattr(base_mod, "asyncio", _AsyncioShim(sleeps))

    provider = ollama_mod.OllamaProvider(
        api_endpoint=TEST_ENDPOINT, model="test-model",
        retry_policy=RetryPolicy(max_attempts=max_attempts, jitter=0),
    )
    # Skip _detect_thinking_behavior(), which would issue its own HTTP request.
    provider._thinking_behavior = ThinkingBehavior.STANDARD

    client = _FakeClient(responder)

    async def fake_get_client():
        return client

    monkeypatch.setattr(provider, "_get_client", fake_get_client)
    return provider, client, sleeps


# ---------------------------------------------------------------------------
# HTTP 429 -> RateLimitError
# ---------------------------------------------------------------------------

class TestOllamaRateLimit:
    """A 429 must surface as RateLimitError immediately, so the pipeline can
    pause and checkpoint instead of writing the source text to the output."""

    @pytest.mark.asyncio
    async def test_retry_after_header_is_propagated(self, monkeypatch):
        """T1: Retry-After: 30 -> retry_after == 30, provider == 'ollama'."""
        def responder():
            return _make_error_response(429, headers={"Retry-After": "30"})

        provider, _client, _sleeps = _make_provider(monkeypatch, responder)

        with pytest.raises(RateLimitError) as exc_info:
            await provider.generate("hello")

        assert exc_info.value.retry_after == 30
        assert exc_info.value.provider == "ollama"

    @pytest.mark.asyncio
    async def test_without_headers_retry_after_is_none(self, monkeypatch):
        """T2: no rate-limit hint -> retry_after stays None.

        The header-less backoff fallback would hammer an exhausted monthly
        quota; None lets the caller apply its own auto-resume delay.
        """
        def responder():
            return _make_error_response(429)

        provider, _client, _sleeps = _make_provider(monkeypatch, responder)

        with pytest.raises(RateLimitError) as exc_info:
            await provider.generate("hello")

        assert exc_info.value.retry_after is None
        assert exc_info.value.provider == "ollama"

    @pytest.mark.asyncio
    async def test_quota_message_survives_to_the_user(self, monkeypatch):
        """T3: the actionable body text must reach the raised exception."""
        quota_body = {
            "error": (
                "you (x) have reached your monthly usage limit, "
                "upgrade for higher limits: https://ollama.com/upgrade"
            )
        }

        def responder():
            return _make_error_response(429, body=quota_body)

        provider, _client, _sleeps = _make_provider(monkeypatch, responder)

        with pytest.raises(RateLimitError) as exc_info:
            await provider.generate("hello")

        assert "monthly usage limit" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fails_fast_without_retry_or_sleep(self, monkeypatch):
        """T4: exactly one request is issued, and no retry delay is awaited."""
        def responder():
            return _make_error_response(429, headers={"Retry-After": "30"})

        provider, client, sleeps = _make_provider(monkeypatch, responder)

        with pytest.raises(RateLimitError):
            await provider.generate("hello")

        assert len(client.stream_calls) == 1, (
            f"429 must not burn a retry attempt, got {client.stream_calls}"
        )
        assert sleeps == [], f"429 must not sleep before raising, got {sleeps}"


# ---------------------------------------------------------------------------
# Untouched behaviour (regression guards)
# ---------------------------------------------------------------------------

class TestOllamaNonRateLimitPaths:
    """The 429 branch must not change how other statuses behave."""

    @pytest.mark.asyncio
    async def test_server_error_still_retries_then_returns_none(self, monkeypatch):
        """T5: a persistent 500 exhausts the attempts and returns None."""
        def responder():
            return _make_error_response(500, body={"error": "internal failure"})

        provider, client, _sleeps = _make_provider(
            monkeypatch, responder, max_attempts=2
        )

        result = await provider.generate("hello")

        assert result is None
        assert len(client.stream_calls) == 2, (
            f"expected MAX_TRANSLATION_ATTEMPTS requests, got {client.stream_calls}"
        )

    @pytest.mark.asyncio
    async def test_streaming_success_still_returns_a_response(self, monkeypatch):
        """T6: the happy path is untouched by the new 429 branch."""
        lines = [
            json.dumps({"message": {"role": "assistant", "content": "Bonjour"},
                        "done": False}),
            json.dumps({"message": {"role": "assistant", "content": " le monde"},
                        "done": False}),
            json.dumps({"message": {"role": "assistant", "content": ""},
                        "done": True,
                        "prompt_eval_count": 12,
                        "eval_count": 7}),
        ]

        def responder():
            return _FakeStreamResponse(lines)

        provider, client, _sleeps = _make_provider(monkeypatch, responder)

        result = await provider.generate("hello")

        assert result is not None
        assert result.content == "Bonjour le monde"
        assert result.prompt_tokens == 12
        assert len(client.stream_calls) == 1
