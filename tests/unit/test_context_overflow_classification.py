"""
Context-overflow classification must not catch auth, credit or rate-limit
errors: a false positive makes the pipeline split the chunk again and again.
"""

import asyncio

import httpx
import pytest

from src.core.llm.exceptions import ContextOverflowError
from src.core.llm.utils.error_classification import is_context_overflow_error

FAKE_KEY = "YOUR_API_KEY_HERE"


@pytest.mark.parametrize("message", [
    "This model's maximum context length is 4096 tokens.",
    "input length exceeds context length",
    "the request exceeds the available context size, try increasing it",
    "prompt is too long: 210000 tokens > 200000 maximum",
    "The input token count (1200000) exceeds the maximum number of tokens allowed (1048576).",
    "context_length_exceeded",
])
def test_real_overflow_messages_are_detected(message):
    assert is_context_overflow_error(message, 400)


@pytest.mark.parametrize("status", [401, 402, 403, 429])
def test_auth_credit_and_rate_limit_statuses_are_never_overflow(status):
    assert not is_context_overflow_error("maximum context length exceeded", status)


@pytest.mark.parametrize("message", [
    "Invalid authentication: the key length is wrong",
    "max_tokens must be less than or equal to 8192",
    "Your credit balance exceeds the allowed limit",
    "Missing context parameter",
    "Response was truncated by the proxy",
])
def test_broad_words_alone_are_not_overflow(message):
    assert not is_context_overflow_error(message, 400)


def test_extra_keywords_are_honored():
    assert is_context_overflow_error("RESOURCE_EXHAUSTED", 400, extra_keywords=("resource_exhausted",))


async def _no_sleep(*_args, **_kwargs):
    return None


def _provider(name):
    from src.core.llm.providers.ollama import OllamaProvider
    from src.core.llm.providers.openai import OpenAICompatibleProvider
    if name == "openai":
        return OpenAICompatibleProvider(
            api_endpoint="https://api.example.test/v1/chat/completions",
            model="gpt-4o", api_key=FAKE_KEY)
    return OllamaProvider(api_endpoint="http://localhost:11434/api/generate",
                          model="llama3", context_window=2048)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["openai", "ollama"])
@pytest.mark.parametrize("status,error", [
    (401, "Incorrect API key provided: the context of this request is unauthorized"),
    (402, "Insufficient credits for the requested max_tokens length"),
    (403, "Access to this model is forbidden in this context"),
])
async def test_auth_and_credit_errors_do_not_raise_overflow(monkeypatch, name, status, error):
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    provider = _provider(name)
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(status, json={"error": {"message": error}})))
    try:
        assert await provider.generate("Translate this") is None
    except ContextOverflowError:
        pytest.fail(f"HTTP {status} was classified as a context overflow")
    finally:
        await provider.close()
