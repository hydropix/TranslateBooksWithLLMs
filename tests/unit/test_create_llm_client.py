"""
create_llm_client() must build a real client for every provider the factory
supports, and fail loudly instead of returning None: callers used to fall back
to the module-level Ollama client, so a LiteLLM refine or a keyless Gemini job
silently talked to a local Ollama server.
"""

import pytest

from src.core.llm import LLMClientConfigError
from src.core.llm.providers.deepseek import DeepSeekProvider
from src.core.llm.providers.gemini import GeminiProvider
from src.core.llm.providers.litellm import LiteLLMProvider
from src.core.llm.providers.mistral import MistralProvider
from src.core.llm.providers.ollama import OllamaProvider
from src.core.llm.providers.openai import OpenAICompatibleProvider
from src.core.llm.providers.openrouter import OpenRouterProvider
from src.core.llm.providers.poe import PoeProvider
from src.core.llm_client import (
    SUPPORTED_CLIENT_PROVIDERS, LLMClient, create_llm_client, default_client,
)

FAKE_KEY = "YOUR_API_KEY_HERE"
ENDPOINT = "http://localhost:11434/api/generate"

KEY_ENV_VARS = ("GEMINI_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "MISTRAL_API_KEY",
                "DEEPSEEK_API_KEY", "POE_API_KEY", "NIM_API_KEY")


@pytest.fixture
def no_env_keys(monkeypatch):
    """No key from the environment or from src.config defaults."""
    import src.core.llm.factory as factory
    for var in KEY_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
        if hasattr(factory, var):
            monkeypatch.setattr(factory, var, "")


def _build(provider, **overrides):
    kwargs = dict(
        llm_provider=provider, gemini_api_key=FAKE_KEY, api_endpoint=ENDPOINT,
        model_name="some-model", openai_api_key=FAKE_KEY, openrouter_api_key=FAKE_KEY,
        mistral_api_key=FAKE_KEY, deepseek_api_key=FAKE_KEY, poe_api_key=FAKE_KEY,
        nim_api_key=FAKE_KEY, context_window=4096,
    )
    kwargs.update(overrides)
    return create_llm_client(**kwargs)


EXPECTED_PROVIDER_CLASS = {
    "ollama": OllamaProvider,
    "openai": OpenAICompatibleProvider,
    "gemini": GeminiProvider,
    "openrouter": OpenRouterProvider,
    "mistral": MistralProvider,
    "deepseek": DeepSeekProvider,
    "poe": PoeProvider,
    "nim": OpenAICompatibleProvider,
    "litellm": LiteLLMProvider,
}


def test_every_factory_provider_is_supported():
    assert set(SUPPORTED_CLIENT_PROVIDERS) == set(EXPECTED_PROVIDER_CLASS)


@pytest.mark.parametrize("provider", sorted(EXPECTED_PROVIDER_CLASS))
def test_builds_the_matching_provider(provider):
    client = _build(provider)
    assert isinstance(client, LLMClient)
    assert client is not default_client
    assert isinstance(client._get_provider(), EXPECTED_PROVIDER_CLASS[provider])
    assert client._get_provider().model == "some-model"


def test_litellm_is_not_routed_to_ollama():
    client = _build("litellm", model_name="anthropic/claude-haiku-4-5")
    provider = client._get_provider()
    assert isinstance(provider, LiteLLMProvider)
    assert provider.model == "anthropic/claude-haiku-4-5"


def test_ollama_and_openai_keep_endpoint_and_context_window():
    ollama = _build("ollama")._get_provider()
    assert ollama.context_window == 4096
    openai = _build("openai", api_endpoint="http://localhost:8080/v1/chat/completions")
    assert openai._get_provider().api_endpoint.startswith("http://localhost:8080")
    assert openai._get_provider().context_window == 4096


def test_provider_name_is_case_insensitive():
    assert isinstance(_build("Gemini")._get_provider(), GeminiProvider)


def test_gemini_without_explicit_key_uses_the_environment(monkeypatch, no_env_keys):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    client = _build("gemini", gemini_api_key="")
    assert isinstance(client._get_provider(), GeminiProvider)


@pytest.mark.parametrize("provider", ["gemini", "openrouter", "mistral", "deepseek", "poe", "nim"])
def test_missing_key_raises_instead_of_returning_none(no_env_keys, provider):
    with pytest.raises(LLMClientConfigError, match=provider):
        _build(provider, **{f"{provider}_api_key": ""})


def test_unknown_provider_raises():
    with pytest.raises(LLMClientConfigError, match="Unknown LLM provider 'nope'"):
        _build("nope")


def test_config_error_is_a_value_error():
    assert issubclass(LLMClientConfigError, ValueError)
