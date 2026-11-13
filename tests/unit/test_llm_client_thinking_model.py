"""
Unit tests for LLMClient's thinking-model API (issue #220).

After the Ollama refactor, OllamaProvider exposes ``_detect_thinking_behavior()``
and ``_thinking_behavior`` instead of ``_detect_thinking_model()`` and
``_is_thinking_model``, so ``LLMClient.detect_thinking_model()`` silently did
nothing. ``get_is_thinking_model()`` also returned the bound
``_is_thinking_model`` method of the Gemini and DeepSeek providers, which is
truthy for every model.

https://github.com/hydropix/TranslateBooksWithLLMs/issues/220
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.llm.providers.deepseek import DeepSeekProvider
from src.core.llm.providers.gemini import GeminiProvider
from src.core.llm.providers.ollama import OllamaProvider
from src.core.llm.thinking.behavior import ThinkingBehavior
from src.core.llm_client import LLMClient


def _client_with(provider) -> LLMClient:
    client = LLMClient(provider_type="ollama", model=provider.model)
    client._provider = provider
    return client


def _ollama_with_detection(behavior: ThinkingBehavior):
    provider = OllamaProvider(model="some-model:7b")
    calls = []

    async def fake_detect():
        calls.append(1)
        return behavior

    provider._detect_thinking_behavior = fake_detect
    return provider, calls


class TestOllamaDetection:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "behavior, expected",
        [
            (ThinkingBehavior.STANDARD, False),
            (ThinkingBehavior.CONTROLLABLE, True),
            (ThinkingBehavior.UNCONTROLLABLE, True),
        ],
    )
    async def test_detect_runs_and_caches_behavior(self, behavior, expected):
        provider, calls = _ollama_with_detection(behavior)
        client = _client_with(provider)

        assert await client.detect_thinking_model() is expected
        assert provider._thinking_behavior is behavior
        assert client.get_is_thinking_model() is expected

        # Detection runs once; later calls reuse the stored behavior.
        assert await client.detect_thinking_model() is expected
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_detect_reuses_behavior_already_known(self):
        provider, calls = _ollama_with_detection(ThinkingBehavior.STANDARD)
        provider._thinking_behavior = ThinkingBehavior.UNCONTROLLABLE
        client = _client_with(provider)

        assert await client.detect_thinking_model() is True
        assert calls == []

    def test_status_is_unknown_before_detection(self):
        client = _client_with(OllamaProvider(model="some-model:7b"))
        assert client.get_is_thinking_model() is None


class TestMethodBasedProviders:
    @pytest.mark.parametrize(
        "provider, expected",
        [
            (GeminiProvider(api_key="YOUR_API_KEY_HERE", model="gemini-2.0-flash"), False),
            (GeminiProvider(api_key="YOUR_API_KEY_HERE", model="gemini-2.5-flash"), True),
            (DeepSeekProvider(api_key="YOUR_API_KEY_HERE", model="deepseek-chat"), False),
            (DeepSeekProvider(api_key="YOUR_API_KEY_HERE", model="deepseek-reasoner"), True),
        ],
        ids=["gemini-2.0", "gemini-2.5", "deepseek-chat", "deepseek-reasoner"],
    )
    def test_get_is_thinking_model_returns_a_bool(self, provider, expected):
        assert _client_with(provider).get_is_thinking_model() is expected

    @pytest.mark.asyncio
    async def test_detect_reports_status_without_dynamic_detection(self):
        provider = GeminiProvider(api_key="YOUR_API_KEY_HERE", model="gemini-2.0-flash")
        assert await _client_with(provider).detect_thinking_model() is False
