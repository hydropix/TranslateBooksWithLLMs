"""
Conformance test run against every HTTP provider class.

Each provider keeps its own wire format, but all of them must behave the
same through the shared request loop: parse the answer and its usage,
report usage to the job tracker, retry a 5xx, fail fast on 401, raise
ContextOverflowError on an overflow answer, and send (or deliberately not
send) the sampling temperature as documented in src/config.py.
"""

import json

import httpx
import pytest

from src.core.llm import base as base_mod
from src.core.llm.cost_tracker import begin_job_cost_tracking, end_job_cost_tracking
from src.core.llm.exceptions import ContextOverflowError
from src.core.llm.policy import RetryPolicy
from src.core.llm.thinking.behavior import ThinkingBehavior

FAKE_KEY = "YOUR_API_KEY_HERE"
POLICY = RetryPolicy(max_attempts=2, jitter=0)


def _openai_body(text):
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7}}


def _gemini_body(text):
    return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 7}}


def _ollama_body(text):
    lines = [
        {"message": {"content": text}, "prompt_eval_count": 11},
        {"message": {"content": ""}, "done": True, "done_reason": "stop",
         "prompt_eval_count": 11, "eval_count": 7},
    ]
    return "\n".join(json.dumps(line) for line in lines)


def _build(name):
    from src.core.llm.providers.deepseek import DeepSeekProvider
    from src.core.llm.providers.gemini import GeminiProvider
    from src.core.llm.providers.mistral import MistralProvider
    from src.core.llm.providers.ollama import OllamaProvider
    from src.core.llm.providers.openai import OpenAICompatibleProvider
    from src.core.llm.providers.openrouter import OpenRouterProvider
    from src.core.llm.providers.poe import PoeProvider

    common = dict(retry_policy=POLICY, temperature=0.42)
    if name == "openai":
        return OpenAICompatibleProvider("https://api.example.test/v1", "gpt-x",
                                        api_key=FAKE_KEY, **common)
    if name == "nim":
        return OpenAICompatibleProvider("https://integrate.api.example.test/v1", "nim-x",
                                        api_key=FAKE_KEY, provider_name="nim", **common)
    if name == "mistral":
        return MistralProvider(api_key=FAKE_KEY, model="mistral-small", **common)
    if name == "deepseek":
        return DeepSeekProvider(api_key=FAKE_KEY, model="deepseek-chat", **common)
    if name == "poe":
        return PoeProvider(api_key=FAKE_KEY, model="GPT-4o", disable_thinking=False,
                           disable_web_search=False, **common)
    if name == "openrouter":
        return OpenRouterProvider(api_key=FAKE_KEY, model="openai/gpt-4o",
                                  disable_thinking=False, **common)
    if name == "gemini":
        return GeminiProvider(api_key=FAKE_KEY, model="gemini-2.0-flash", **common)
    provider = OllamaProvider(api_endpoint="http://localhost:11434/api/chat", model="llama3",
                              context_window=4096, **common)
    provider._thinking_behavior = ThinkingBehavior.STANDARD  # skip detection requests
    return provider


PROVIDERS = ["openai", "nim", "mistral", "deepseek", "poe", "openrouter", "gemini", "ollama"]
# Ollama keeps the Modelfile sampling (documented in src/config.py).
SENDS_TEMPERATURE = {name: name != "ollama" for name in PROVIDERS}


def _success(name, text="translated"):
    if name == "ollama":
        return httpx.Response(200, text=_ollama_body(text))
    if name == "gemini":
        return httpx.Response(200, json=_gemini_body(text))
    return httpx.Response(200, json=_openai_body(text))


def _attach(provider, responses):
    sent = []

    def handler(request):
        sent.append(json.loads(request.content or b"{}"))
        return responses.pop(0)

    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return sent


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def _instant(_delay):
        return None
    monkeypatch.setattr(base_mod.asyncio, "sleep", _instant)


def _temperature_of(name, payload):
    if name == "gemini":
        return payload.get("generationConfig", {}).get("temperature")
    if name == "ollama":
        return payload.get("options", {}).get("temperature")
    return payload.get("temperature")


@pytest.mark.parametrize("name", PROVIDERS)
@pytest.mark.asyncio
async def test_success_parses_text_and_reports_usage(name):
    provider = _build(name)
    sent = _attach(provider, [_success(name)])
    tracker, token = begin_job_cost_tracking()
    try:
        response = await provider.generate("Translate this", system_prompt="You translate.")
    finally:
        end_job_cost_tracking(token)
        await provider.close()
    assert response.content == "translated"
    assert (response.prompt_tokens, response.completion_tokens) == (11, 7)
    assert tracker.snapshot()[1] == {"prompt": 11, "completion": 7}
    temperature = _temperature_of(name, sent[0])
    assert (temperature == 0.42) if SENDS_TEMPERATURE[name] else (temperature is None)


@pytest.mark.parametrize("name", PROVIDERS)
@pytest.mark.asyncio
async def test_server_error_is_retried(name):
    provider = _build(name)
    sent = _attach(provider, [httpx.Response(503, json={"error": "busy"}), _success(name)])
    try:
        response = await provider.generate("Translate this")
    finally:
        await provider.close()
    assert response.content == "translated"
    assert len(sent) == 2


@pytest.mark.parametrize("name", PROVIDERS)
@pytest.mark.asyncio
async def test_unauthorized_fails_fast(name):
    provider = _build(name)
    sent = _attach(provider, [httpx.Response(401, json={"error": {"message": "bad key"}})])
    try:
        assert await provider.generate("Translate this") is None
    finally:
        await provider.close()
    assert len(sent) == 1


@pytest.mark.parametrize("name", PROVIDERS)
@pytest.mark.asyncio
async def test_overflow_answer_raises_context_overflow(name):
    provider = _build(name)
    _attach(provider, [httpx.Response(400, json={"error": {
        "message": "This model's maximum context length is 4096 tokens",
        "code": "context_length_exceeded"}})])
    try:
        with pytest.raises(ContextOverflowError):
            await provider.generate("Translate this")
    finally:
        await provider.close()


@pytest.mark.parametrize("name", ["openai", "mistral", "deepseek", "openrouter", "poe"])
@pytest.mark.asyncio
async def test_rejected_temperature_is_dropped_once(name):
    """A model that rejects the sampling parameter gets one retry without it."""
    provider = _build(name)
    sent = _attach(provider, [
        httpx.Response(400, json={"error": {
            "message": "Unsupported value: 'temperature' does not support 0.42 with this model."}}),
        _success(name),
    ])
    try:
        response = await provider.generate("Translate this")
    finally:
        await provider.close()
    assert response.content == "translated"
    assert "temperature" in sent[0] and "temperature" not in sent[1]
