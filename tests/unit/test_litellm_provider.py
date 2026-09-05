"""
Unit tests for the LiteLLM provider.

LiteLLM is stubbed via sys.modules so these tests never touch the network and
do not require the optional `litellm` package to be installed. The real
end-to-end behaviour is covered by tests/standalone/manual_litellm_smoke.py.
"""

import sys
import types
from unittest import mock

import pytest


def _install_litellm_stub():
    fake = types.ModuleType("litellm")
    fake.acompletion = mock.AsyncMock(name="litellm.acompletion")
    sys.modules["litellm"] = fake
    return fake


@pytest.fixture(autouse=True)
def litellm_stub():
    fake = _install_litellm_stub()
    yield fake
    sys.modules.pop("litellm", None)


def _mock_response(content="Hello!", prompt_tokens=10, completion_tokens=5):
    from types import SimpleNamespace

    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        ),
    )


@pytest.mark.asyncio
async def test_generate_calls_acompletion(litellm_stub):
    litellm_stub.acompletion.return_value = _mock_response("translated text")

    from src.core.llm.providers.litellm import LiteLLMProvider

    provider = LiteLLMProvider(model="anthropic/claude-haiku-4-5", api_key="sk-test")
    result = await provider.generate("Translate this")

    litellm_stub.acompletion.assert_called_once()
    kwargs = litellm_stub.acompletion.call_args.kwargs
    assert kwargs["model"] == "anthropic/claude-haiku-4-5"
    assert kwargs["api_key"] == "sk-test"
    assert kwargs["drop_params"] is True
    assert result.content == "translated text"
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5


@pytest.mark.asyncio
async def test_generate_forwards_temperature(litellm_stub):
    litellm_stub.acompletion.return_value = _mock_response()

    from src.config import TEMPERATURE
    from src.core.llm.providers.litellm import LiteLLMProvider

    provider = LiteLLMProvider(model="openai/gpt-4o", api_key="k")
    await provider.generate("Hello")

    kwargs = litellm_stub.acompletion.call_args.kwargs
    assert kwargs["temperature"] == TEMPERATURE


@pytest.mark.asyncio
async def test_generate_omits_blank_credentials(litellm_stub):
    litellm_stub.acompletion.return_value = _mock_response()

    from src.core.llm.providers.litellm import LiteLLMProvider

    # No key and no api_base: LiteLLM should fall back to native env vars.
    provider = LiteLLMProvider(model="openai/gpt-4o")
    await provider.generate("Hello")

    kwargs = litellm_stub.acompletion.call_args.kwargs
    assert "api_key" not in kwargs
    assert "api_base" not in kwargs


@pytest.mark.asyncio
async def test_generate_forwards_system_prompt(litellm_stub):
    litellm_stub.acompletion.return_value = _mock_response()

    from src.core.llm.providers.litellm import LiteLLMProvider

    provider = LiteLLMProvider(model="openai/gpt-4o", api_key="k")
    await provider.generate("Translate this", system_prompt="You are a translator")

    kwargs = litellm_stub.acompletion.call_args.kwargs
    messages = kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a translator"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Translate this"


@pytest.mark.asyncio
async def test_generate_forwards_api_base(litellm_stub):
    litellm_stub.acompletion.return_value = _mock_response()

    from src.core.llm.providers.litellm import LiteLLMProvider

    provider = LiteLLMProvider(
        model="openai/gpt-4o", api_key="k", api_base="https://proxy.example/v1"
    )
    await provider.generate("Hello")

    kwargs = litellm_stub.acompletion.call_args.kwargs
    assert kwargs["api_base"] == "https://proxy.example/v1"


@pytest.mark.asyncio
async def test_context_overflow_is_raised(litellm_stub):
    litellm_stub.acompletion.side_effect = RuntimeError(
        "This model's maximum context length is 8192 tokens"
    )

    from src.core.llm.providers.litellm import LiteLLMProvider
    from src.core.llm.exceptions import ContextOverflowError

    provider = LiteLLMProvider(model="openai/gpt-4o", api_key="k")
    with pytest.raises(ContextOverflowError):
        await provider.generate("way too long")


def test_init_does_not_shadow_base_api_key_property():
    """Regression: the base class exposes `api_key` as a read-only property.

    A previous draft assigned `self.api_key = ...` in __init__, which raised
    AttributeError on instantiation against the current base class. Construction
    must succeed and the property must stay readable.
    """
    from src.core.llm.providers.litellm import LiteLLMProvider

    # No explicit key -> no KeyPool -> property returns None, never raises.
    provider = LiteLLMProvider(model="openai/gpt-4o")
    assert provider.api_key is None

    # Explicit key -> readable through the inherited KeyPool-backed property.
    provider_with_key = LiteLLMProvider(model="openai/gpt-4o", api_key="sk-test")
    assert provider_with_key.api_key == "sk-test"


# ---------------------------------------------------------------------------
# Key rotation on 429 (item 6 of issue #231)
#
# The provider used to peek() a single key once, before the retry loop, and
# never call acquire()/mark_throttled(), so a multi-key LiteLLM setup never
# rotated on RateLimitError.
# ---------------------------------------------------------------------------

class FakeLiteLLMRateLimitError(Exception):
    """Stand-in for litellm.exceptions.RateLimitError.

    The provider classifies exceptions by qualname, so the module and class
    names have to match the real ones. The message must stay free of the
    context-overflow keywords or it would be re-raised as ContextOverflowError.
    """
    __module__ = "litellm.exceptions"


FakeLiteLLMRateLimitError.__name__ = "RateLimitError"
FakeLiteLLMRateLimitError.__qualname__ = "RateLimitError"


@pytest.fixture
def no_sleep(monkeypatch):
    """Neutralise every backoff so the retry loops run instantly."""
    async def _instant(_seconds):
        return None

    monkeypatch.setattr("asyncio.sleep", _instant)


@pytest.mark.asyncio
async def test_rate_limit_rotates_to_the_next_key(litellm_stub, no_sleep):
    """A 429 on the first key must retry on the second one, not give up."""
    from src.core.llm.providers.litellm import LiteLLMProvider

    litellm_stub.acompletion.side_effect = [
        FakeLiteLLMRateLimitError("429 too many requests"),
        _mock_response("translated"),
    ]

    provider = LiteLLMProvider(model="openai/gpt-4o", api_key="key-a,key-b")
    result = await provider.generate("Hello")

    assert result.content == "translated"
    used_keys = [c.kwargs["api_key"] for c in litellm_stub.acompletion.call_args_list]
    assert used_keys == ["key-a", "key-b"]


@pytest.mark.asyncio
async def test_rate_limit_does_not_consume_a_retry_attempt(litellm_stub, no_sleep):
    """Rotations have their own budget: a 3-key pool gets more than
    MAX_TRANSLATION_ATTEMPTS calls before the transient retries even start."""
    from src.config import MAX_TRANSLATION_ATTEMPTS
    from src.core.llm.providers.litellm import LiteLLMProvider

    litellm_stub.acompletion.side_effect = (
        [FakeLiteLLMRateLimitError("429 too many requests")] * MAX_TRANSLATION_ATTEMPTS
        + [_mock_response("translated")]
    )

    provider = LiteLLMProvider(model="openai/gpt-4o", api_key="k1,k2,k3")
    result = await provider.generate("Hello")

    assert result.content == "translated"
    assert litellm_stub.acompletion.call_count == MAX_TRANSLATION_ATTEMPTS + 1


@pytest.mark.asyncio
async def test_persistent_rate_limit_raises_for_upstream_pause(litellm_stub, no_sleep):
    """Once the rotation budget is spent, RateLimitError propagates so the
    job can be auto-paused instead of silently returning None."""
    from src.core.llm.exceptions import RateLimitError
    from src.core.llm.providers.litellm import LiteLLMProvider

    litellm_stub.acompletion.side_effect = FakeLiteLLMRateLimitError(
        "429 too many requests"
    )

    provider = LiteLLMProvider(model="openai/gpt-4o", api_key="k1,k2")
    with pytest.raises(RateLimitError):
        await provider.generate("Hello")


@pytest.mark.asyncio
async def test_rate_limit_without_pool_keeps_plain_backoff(litellm_stub, no_sleep):
    """With credentials in env vars there is no pool and nothing to rotate:
    the 429 stays an ordinary transient retry that ends in None."""
    from src.config import MAX_TRANSLATION_ATTEMPTS
    from src.core.llm.providers.litellm import LiteLLMProvider

    litellm_stub.acompletion.side_effect = FakeLiteLLMRateLimitError(
        "429 too many requests"
    )

    provider = LiteLLMProvider(model="openai/gpt-4o")
    assert await provider.generate("Hello") is None
    assert litellm_stub.acompletion.call_count == MAX_TRANSLATION_ATTEMPTS


@pytest.mark.asyncio
async def test_non_rate_limit_errors_still_retry_on_the_same_key(litellm_stub, no_sleep):
    """A transient connection error is not a rate limit: no key is throttled."""
    from src.core.llm.providers.litellm import LiteLLMProvider

    litellm_stub.acompletion.side_effect = [
        RuntimeError("connection reset"),
        _mock_response("translated"),
    ]

    provider = LiteLLMProvider(model="openai/gpt-4o", api_key="only-key")
    result = await provider.generate("Hello")

    assert result.content == "translated"
    assert litellm_stub.acompletion.call_count == 2


def test_factory_creates_litellm_provider():
    from src.core.llm.factory import create_llm_provider
    from src.core.llm.providers.litellm import LiteLLMProvider

    provider = create_llm_provider(
        "litellm", model="anthropic/claude-haiku-4-5", api_key="k"
    )
    assert isinstance(provider, LiteLLMProvider)
    assert provider.model == "anthropic/claude-haiku-4-5"


def test_factory_ignores_generic_endpoint_for_api_base():
    """The txt/srt pipeline passes the Ollama endpoint as `endpoint`; LiteLLM
    must not adopt it as api_base or native routing would break."""
    from src.core.llm.factory import create_llm_provider

    provider = create_llm_provider(
        "litellm",
        model="gemini/gemini-2.5-flash",
        endpoint="http://localhost:11434/api/generate",
    )
    assert provider.api_base is None
