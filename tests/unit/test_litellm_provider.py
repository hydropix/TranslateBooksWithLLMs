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
async def test_generate_omits_blank_credentials(litellm_stub):
    litellm_stub.acompletion.return_value = _mock_response()

    from src.core.llm.providers.litellm import LiteLLMProvider

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


def test_factory_creates_litellm_provider():
    from src.core.llm.factory import create_llm_provider
    from src.core.llm.providers.litellm import LiteLLMProvider

    provider = create_llm_provider("litellm", model="anthropic/claude-haiku-4-5", api_key="k")
    assert isinstance(provider, LiteLLMProvider)
    assert provider.model == "anthropic/claude-haiku-4-5"
