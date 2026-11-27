"""
Wire-format and error-handling pins for every HTTP provider (no network).

Complements tests/unit/test_provider_conformance.py: that file checks the
shared behaviour with a generic error body; this one pins the exact request
each provider sends (URL, auth header form, payload keys) and the handling of
each provider's documented error shapes, so an accidental wire change shows
up as a test failure.

It also pins three fixes over the pre-refactor code: a keyless 429 waits for
its Retry-After hint (capped), an OpenRouter 200 carrying only an error object
is classified by the status it names and retried when transient, and a partial
answer ending with finish_reason "error" is never accepted as a translation.
"""

import json

import httpx
import pytest

from src.core.llm import base as base_mod
from src.core.llm.exceptions import ContextOverflowError, RateLimitError
from src.core.llm.factory import create_llm_provider
from src.core.llm.policy import RetryPolicy
from src.core.llm.thinking.behavior import ThinkingBehavior

FAKE_KEY = "sk-xxxxxxxx"
POLICY = RetryPolicy(max_attempts=2, base_delay=2.0, jitter=0)


def _ok(text="<TRANSLATION>ok</TRANSLATION>", finish="stop"):
    return httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2}})


class Wire:
    """Mock transport recording every request; answers from a queue."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def handler(self, request):
        body = json.loads(request.content) if request.content else None
        self.requests.append({"method": request.method, "url": str(request.url),
                              "headers": dict(request.headers), "json": body})
        if request.method == "GET":  # model catalogs (OpenRouter, Poe)
            return httpx.Response(200, json={"data": []})
        return self.responses.pop(0)

    @property
    def posts(self):
        return [r for r in self.requests if r["method"] == "POST"]


@pytest.fixture
def sleeps(monkeypatch):
    recorded = []

    async def _instant(delay):
        recorded.append(delay)
    monkeypatch.setattr(base_mod.asyncio, "sleep", _instant)
    import src.core.llm.rate_limit_handler as rl
    monkeypatch.setattr(rl.asyncio, "sleep", _instant)
    return recorded


def _attach(provider, wire):
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(wire.handler))
    if hasattr(provider, "_thinking_behavior"):
        provider._thinking_behavior = ThinkingBehavior.STANDARD
    return provider


def _make(provider_type, **kwargs):
    kwargs.setdefault("retry_policy", POLICY)
    kwargs.setdefault("temperature", 0.3)
    return create_llm_provider(provider_type, **kwargs)


# --------------------------------------------------------------- wire format
CASES = {
    # name: (factory args, expected URL, auth header (or None), extra payload keys)
    "openai": (dict(provider_type="openai", model="gpt-4o", api_key=FAKE_KEY,
                    api_endpoint="https://api.openai.com/v1"),
               "https://api.openai.com/v1/chat/completions", f"Bearer {FAKE_KEY}",
               {"temperature": 0.3}),
    "openai_local": (dict(provider_type="openai", model="qwen3", api_key="",
                          api_endpoint="http://localhost:8080/v1/"),
                     "http://localhost:8080/v1/chat/completions", None,
                     {"temperature": 0.3, "enable_thinking": False,
                      "chat_template_kwargs": {"enable_thinking": False}}),
    "nim": (dict(provider_type="nim", model="meta/llama-3.1-70b-instruct", api_key=FAKE_KEY),
            "https://integrate.api.nvidia.com/v1/chat/completions", f"Bearer {FAKE_KEY}",
            {"temperature": 0.3}),
    "llmtr": (dict(provider_type="llmtr", model="qwen/qwen3.8-27b-free", api_key=FAKE_KEY),
              "https://llmtr.com/v1/chat/completions", f"Bearer {FAKE_KEY}",
              {"temperature": 0.3}),
    "mistral": (dict(provider_type="mistral", model="mistral-large-latest", api_key=FAKE_KEY),
                "https://api.mistral.ai/v1/chat/completions", f"Bearer {FAKE_KEY}",
                {"temperature": 0.3}),
    "deepseek_v4": (dict(provider_type="deepseek", model="deepseek-v4-flash", api_key=FAKE_KEY),
                    "https://api.deepseek.com/chat/completions", f"Bearer {FAKE_KEY}",
                    {"temperature": 0.3, "thinking": {"type": "disabled"}}),
    "poe": (dict(provider_type="poe", model="Claude-Sonnet-4", api_key=FAKE_KEY),
            "https://api.poe.com/v1/chat/completions", f"Bearer {FAKE_KEY}",
            {"temperature": 0.3}),
    "openrouter_preset": (dict(provider_type="openrouter", model="@preset/x", api_key=FAKE_KEY),
                          "https://openrouter.ai/api/v1/chat/completions", f"Bearer {FAKE_KEY}",
                          {}),
}


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.asyncio
async def test_chat_completions_request_shape(name, sleeps):
    args, url, auth, extra = CASES[name]
    args = dict(args)
    provider = _make(args.pop("provider_type"), **args)
    wire = Wire([_ok()])
    _attach(provider, wire)
    try:
        response = await provider.generate("USER", system_prompt="SYS")
    finally:
        await provider.close()
    assert response.content == "<TRANSLATION>ok</TRANSLATION>"
    (post,) = wire.posts
    assert post["url"] == url
    assert post["headers"].get("authorization") == auth
    assert post["headers"]["content-type"] == "application/json"
    expected = {"model": args["model"], "stream": False,
                "messages": [{"role": "system", "content": "SYS"},
                             {"role": "user", "content": "USER"}], **extra}
    assert post["json"] == expected
    # No output cap is sent by any chat provider (unchanged from main).
    assert "max_tokens" not in post["json"]


@pytest.mark.asyncio
async def test_openrouter_sends_attribution_headers(sleeps):
    wire = Wire([_ok()])
    provider = _attach(_make("openrouter", model="@preset/x", api_key=FAKE_KEY), wire)
    try:
        await provider.generate("USER")
    finally:
        await provider.close()
    headers = wire.posts[0]["headers"]
    assert headers["http-referer"] == "https://github.com/hydropix/TranslateBookWithLLM"
    assert headers["x-title"] == "TranslateBookWithLLM"


@pytest.mark.asyncio
async def test_gemini_request_shape(sleeps):
    wire = Wire([httpx.Response(200, json={"candidates": [
        {"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}]})])
    provider = _attach(_make("gemini", model="gemini-2.5-flash", api_key=FAKE_KEY), wire)
    try:
        await provider.generate("USER", system_prompt="SYS")
    finally:
        await provider.close()
    (post,) = wire.posts
    assert post["url"] == ("https://generativelanguage.googleapis.com/v1beta/models/"
                           "gemini-2.5-flash:generateContent")
    assert post["headers"]["x-goog-api-key"] == FAKE_KEY
    assert "key=" not in post["url"] and "authorization" not in post["headers"]
    body = post["json"]
    assert body["generationConfig"] == {"temperature": 0.3,
                                        "thinkingConfig": {"thinkingBudget": 0}}
    assert body["systemInstruction"] == {"parts": [{"text": "SYS"}]}
    assert body["contents"] == [{"role": "user", "parts": [{"text": "USER"}]}]
    assert len(body["safetySettings"]) == 4


@pytest.mark.asyncio
async def test_ollama_request_shape(sleeps):
    stream = "\n".join(json.dumps(x) for x in [
        {"message": {"content": "ok"}, "done": False},
        {"message": {"content": ""}, "done": True, "done_reason": "stop",
         "prompt_eval_count": 3, "eval_count": 1}])
    wire = Wire([httpx.Response(200, text=stream)])
    provider = _attach(_make("ollama", model="llama3.1:8b", context_window=4096,
                             api_endpoint="http://ollama.test:11434/api/generate"), wire)
    try:
        response = await provider.generate("USER", system_prompt="SYS", context_window=8192)
    finally:
        await provider.close()
    (post,) = wire.posts
    assert post["url"] == "http://ollama.test:11434/api/chat"
    assert "authorization" not in post["headers"]
    assert post["json"] == {
        "model": "llama3.1:8b", "stream": True, "think": False,
        "messages": [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}],
        "options": {"num_ctx": 8192, "truncate": False},  # no temperature: Modelfile sampling
    }
    assert response.context_limit == 8192


@pytest.mark.asyncio
async def test_explicit_empty_key_is_not_refilled_from_env(monkeypatch, sleeps):
    """MAIN sent OPENAI_API_KEY to any OpenAI-compatible host when the caller passed ''."""
    monkeypatch.setenv("OPENAI_API_KEY", "<REDACTED>-env-key-must-not-leak")
    wire = Wire([_ok()])
    provider = _attach(_make("openai", model="m", api_key="",
                             api_endpoint="http://192.168.1.50:8000/v1"), wire)
    try:
        await provider.generate("USER")
    finally:
        await provider.close()
    assert "authorization" not in wire.posts[0]["headers"]


# ---------------------------------------------------- documented error shapes
OVERFLOW_SHAPES = {
    "openai": ("openai", dict(api_endpoint="https://api.openai.com/v1"), {"error": {
        "message": "This model's maximum context length is 128000 tokens. However, your "
                   "messages resulted in 130000 tokens.", "type": "invalid_request_error",
        "param": "messages", "code": "context_length_exceeded"}}),
    "vllm_or_nim": ("nim", {}, {"object": "error", "message": "This model's maximum context "
                                "length is 4096 tokens. However, you requested 5000 tokens.",
                                "type": "BadRequestError", "param": None, "code": 400}),
    "llamacpp": ("openai", dict(api_endpoint="http://localhost:8080/v1"), {"error": {
        "code": 400, "message": "the request exceeds the available context size, try "
                                "increasing it", "type": "exceed_context_size_error"}}),
    "mistral": ("mistral", {}, {"object": "error", "message": "Prompt contains 40000 tokens and "
                                "0 draft tokens, too large for model with 32768 maximum context "
                                "length", "type": "invalid_request_message_error", "code": "3051"}),
    "deepseek": ("deepseek", {}, {"error": {"message": "This model's maximum context length is "
                                            "65536 tokens. However, you requested 70000 tokens.",
                                            "type": "invalid_request_error"}}),
    "openrouter": ("openrouter", dict(model="@preset/x"), {"error": {
        "code": 400, "message": "This endpoint's maximum context length is 163840 tokens. "
                                "However, you requested about 170000 tokens."}}),
    "gemini": ("gemini", {}, {"error": {"code": 400, "message": "The input token count (1200000) "
                                        "exceeds the maximum number of tokens allowed (1048576).",
                                        "status": "INVALID_ARGUMENT"}}),
}


@pytest.mark.parametrize("name", list(OVERFLOW_SHAPES))
@pytest.mark.asyncio
async def test_documented_overflow_shapes_raise_context_overflow(name, sleeps):
    ptype, extra, body = OVERFLOW_SHAPES[name]
    extra = dict(extra)
    extra.setdefault("model", "m")
    wire = Wire([httpx.Response(400, json=body)])
    provider = _attach(_make(ptype, api_key=FAKE_KEY, **extra), wire)
    try:
        with pytest.raises(ContextOverflowError):
            await provider.generate("USER")
    finally:
        await provider.close()
    assert len(wire.posts) == 1


@pytest.mark.asyncio
async def test_openrouter_402_naming_max_tokens_is_not_an_overflow(sleeps):
    """MAIN read this 402 as an overflow (keyword 'max_tokens') and split the chunk."""
    wire = Wire([httpx.Response(402, json={"error": {"code": 402, "message":
        "This request requires more credits, or fewer max_tokens. You requested up to 4096 "
        "tokens, but can only afford 1000."}})])
    provider = _attach(_make("openrouter", model="@preset/x", api_key=FAKE_KEY), wire)
    try:
        assert await provider.generate("USER") is None
    finally:
        await provider.close()
    assert len(wire.posts) == 1


@pytest.mark.parametrize("ptype", ["mistral", "deepseek", "poe"])
@pytest.mark.asyncio
async def test_persistent_429_pauses_with_retry_after(ptype, sleeps):
    """One shared loop: MAIN's Mistral/DeepSeek/Poe returned None (chunk failed) instead."""
    wire = Wire([httpx.Response(429, headers={"Retry-After": "7"},
                                json={"message": "Requests rate limit exceeded"})] * 4)
    provider = _attach(_make(ptype, model="Claude-Sonnet-4" if ptype == "poe" else "m",
                             api_key=FAKE_KEY), wire)
    try:
        with pytest.raises(RateLimitError) as info:
            await provider.generate("USER")
    finally:
        await provider.close()
    assert info.value.retry_after == 7
    assert sleeps == [7]


# ------------------------------------------------------------------- DEFECTS
@pytest.mark.asyncio
async def test_rejected_temperature_is_remembered_across_requests(sleeps):
    """DEFECT: a model that rejects 'temperature' (OpenAI gpt-5 / o-series) costs a
    failed 400 on EVERY chunk, doubling the request count, because the drop is
    only kept for one generate() call. MAIN sent no temperature to OpenAI."""
    rejection = httpx.Response(400, json={"error": {
        "message": "Unsupported value: 'temperature' does not support 0.3 with this model. "
                   "Only the default (1) value is supported.",
        "type": "invalid_request_error", "param": "temperature", "code": "unsupported_value"}})
    wire = Wire([rejection, _ok(), _ok(), _ok()])
    provider = _attach(_make("openai", model="gpt-5", api_key=FAKE_KEY,
                             api_endpoint="https://api.openai.com/v1"), wire)
    try:
        for _ in range(3):
            assert (await provider.generate("USER")).content
    finally:
        await provider.close()
    assert len(wire.posts) == 4
    assert all("temperature" not in p["json"] for p in wire.posts[1:])


@pytest.mark.asyncio
async def test_keyless_429_waits_for_retry_after(sleeps):
    """With no key pool (keyless OpenAI-compatible gateway), the wait before the
    retry honours Retry-After (7 s) instead of the backoff policy alone (2 s)."""
    wire = Wire([httpx.Response(429, headers={"Retry-After": "7"},
                                json={"error": {"message": "slow down"}}), _ok()])
    provider = _attach(_make("openai", model="m", api_key="",
                             api_endpoint="http://localhost:8080/v1"), wire)
    try:
        assert (await provider.generate("USER")).content
    finally:
        await provider.close()
    assert sleeps and sleeps[0] >= 7


@pytest.mark.asyncio
async def test_openrouter_200_error_body_is_retried(sleeps):
    """OpenRouter documents a 200 whose body holds only an 'error' object (upstream
    failure before any output). Code 502 is transient: it is retried (MAIN failed
    it without retry as 'unexpected response format')."""
    wire = Wire([httpx.Response(200, json={"id": "gen-1", "error": {
        "code": 502, "message": "Upstream provider error"}}), _ok()])
    provider = _attach(_make("openrouter", model="@preset/x", api_key=FAKE_KEY), wire)
    try:
        response = await provider.generate("USER")
    finally:
        await provider.close()
    assert response is not None and response.content == "<TRANSLATION>ok</TRANSLATION>"
    assert len(wire.posts) == 2


@pytest.mark.asyncio
async def test_finish_reason_error_partial_output_is_not_accepted(sleeps):
    """OpenRouter returns 200 with partial content and finish_reason 'error' when the
    upstream dies mid-answer. The partial text must not be returned as a
    translation (MAIN did); the request is retried instead."""
    partial = httpx.Response(200, json={"id": "gen-2", "choices": [
        {"message": {"role": "assistant", "content": "<TRANSLATION>Bonj"},
         "finish_reason": "error"}], "error": {"code": 502, "message": "disconnected"}})
    wire = Wire([partial, _ok()])
    provider = _attach(_make("openrouter", model="@preset/x", api_key=FAKE_KEY), wire)
    try:
        response = await provider.generate("USER")
    finally:
        await provider.close()
    assert response is not None and response.content == "<TRANSLATION>ok</TRANSLATION>"


@pytest.mark.asyncio
async def test_keyless_429_huge_retry_after_is_capped(sleeps):
    """A huge Retry-After (RFC 6585's own example is 3600 s) must not stall a worker:
    the retry wait is capped, and the final RateLimitError keeps the full hint so
    the pipeline can pause."""
    wire = Wire([httpx.Response(429, headers={"Retry-After": "3600"},
                                json={"error": {"message": "slow down"}})] * 2)
    provider = _attach(_make("openai", model="m", api_key="",
                             api_endpoint="http://localhost:8080/v1"), wire)
    try:
        with pytest.raises(RateLimitError) as info:
            await provider.generate("USER")
    finally:
        await provider.close()
    assert sleeps == [base_mod.MAX_HINTED_RETRY_WAIT]
    assert info.value.retry_after == 3600


@pytest.mark.asyncio
async def test_openrouter_200_error_with_402_is_not_retried(sleeps):
    """A 200 whose error object names 402 (out of credits) is classified like an
    HTTP 402: not retryable, and not read as a context overflow."""
    wire = Wire([httpx.Response(200, json={"id": "gen-3", "error": {
        "code": 402, "message": "Insufficient credits"}}), _ok()])
    provider = _attach(_make("openrouter", model="@preset/x", api_key=FAKE_KEY), wire)
    try:
        assert await provider.generate("USER") is None
    finally:
        await provider.close()
    assert len(wire.posts) == 1


@pytest.mark.asyncio
async def test_openrouter_200_error_with_429_is_a_rate_limit(sleeps):
    """A 200 whose error object names 429 goes through the rate-limit path: the
    persistent case raises RateLimitError (pipeline pause), never returns text."""
    wire = Wire([httpx.Response(200, json={"id": "gen-4", "error": {
        "code": 429, "message": "Rate limit exceeded"}})] * 4)
    provider = _attach(_make("openrouter", model="@preset/x", api_key=FAKE_KEY), wire)
    try:
        with pytest.raises(RateLimitError):
            await provider.generate("USER")
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_finish_reason_error_partial_output_is_never_returned(sleeps):
    """When every attempt ends mid-answer with finish_reason 'error', the request
    fails (None) rather than returning the partial text."""
    partial = httpx.Response(200, json={"id": "gen-5", "choices": [
        {"message": {"role": "assistant", "content": "<TRANSLATION>Bonj"},
         "finish_reason": "error"}], "error": {"code": 502, "message": "disconnected"}})
    wire = Wire([partial] * 2)
    provider = _attach(_make("openrouter", model="@preset/x", api_key=FAKE_KEY), wire)
    try:
        assert await provider.generate("USER") is None
    finally:
        await provider.close()
    assert len(wire.posts) == 2
