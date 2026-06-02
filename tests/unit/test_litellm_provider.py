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
