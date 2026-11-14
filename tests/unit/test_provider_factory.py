"""
create_llm_provider() is the only construction path for LLM providers.

It must build the matching provider for every supported type, fail loudly on
misconfiguration (callers used to fall back to a module-level Ollama client,
so a LiteLLM refine or a keyless Gemini job silently talked to a local Ollama
server), and never re-fill a key the caller deliberately left empty: the web
layer drops a stored key when a request picks its own endpoint.
"""

import pytest

from src.core.llm import LLMClientConfigError
from src.core.llm.credentials import resolve_provider_key
from src.core.llm.factory import create_llm_provider
from src.core.llm.policy import RetryPolicy
from src.core.llm.provider_config import (
    SUPPORTED_PROVIDERS, ProviderConfig,
)
from src.core.llm.providers.deepseek import DeepSeekProvider
from src.core.llm.providers.gemini import GeminiProvider
from src.core.llm.providers.litellm import LiteLLMProvider
from src.core.llm.providers.mistral import MistralProvider
from src.core.llm.providers.ollama import OllamaProvider
from src.core.llm.providers.openai import OpenAICompatibleProvider
from src.core.llm.providers.openrouter import OpenRouterProvider
from src.core.llm.providers.poe import PoeProvider

FAKE_KEY = "YOUR_API_KEY_HERE"
ENDPOINT = "http://localhost:11434/api/generate"

KEY_ENV_VARS = ("GEMINI_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "MISTRAL_API_KEY",
                "DEEPSEEK_API_KEY", "POE_API_KEY", "NIM_API_KEY")


@pytest.fixture
def no_env_keys(monkeypatch):
    """No key from the environment or from src.config defaults."""
    import src.config as config
    for var in KEY_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(config, var, "", raising=False)


def _build(provider, api_key=FAKE_KEY, model_name="some-model", api_endpoint=ENDPOINT,
           context_window=4096):
    return create_llm_provider(ProviderConfig(
        provider=provider, model=model_name, endpoint=api_endpoint, api_key=api_key,
        context_window=context_window,
    ))


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
    assert set(SUPPORTED_PROVIDERS) == set(EXPECTED_PROVIDER_CLASS)


@pytest.mark.parametrize("provider", sorted(EXPECTED_PROVIDER_CLASS))
def test_builds_the_matching_provider(provider):
    built = _build(provider)
    assert isinstance(built, EXPECTED_PROVIDER_CLASS[provider])
    assert built.model == "some-model"


def test_each_call_builds_a_new_provider():
    assert _build("ollama") is not _build("ollama")


def test_litellm_is_not_routed_to_ollama():
    built = _build("litellm", model_name="anthropic/claude-haiku-4-5")
    assert isinstance(built, LiteLLMProvider)
    assert built.model == "anthropic/claude-haiku-4-5"


def test_ollama_and_openai_keep_endpoint_and_context_window():
    assert _build("ollama").context_window == 4096
    openai = _build("openai", api_endpoint="http://localhost:8080/v1/chat/completions")
    assert openai.api_endpoint.startswith("http://localhost:8080")
    assert openai.context_window == 4096


def test_provider_name_is_case_insensitive():
    assert isinstance(_build("Gemini"), GeminiProvider)


def test_key_left_out_falls_back_to_the_environment(monkeypatch, no_env_keys):
    """The .env fallback lives in the credential resolver the job builders use."""
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    key = resolve_provider_key("gemini", None)
    assert key == FAKE_KEY
    assert isinstance(_build("gemini", api_key=key), GeminiProvider)
    assert resolve_provider_key("gemini", None, allow_env_fallback=False) == ""


def test_key_left_empty_on_purpose_is_not_refilled(monkeypatch, no_env_keys):
    """An explicit empty key stays empty even when .env holds one."""
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    with pytest.raises(LLMClientConfigError, match="gemini"):
        _build("gemini", api_key="")


def test_dropped_openai_key_never_follows_an_overridden_endpoint(monkeypatch, no_env_keys):
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    built = _build("openai", api_key="",
                   api_endpoint="http://192.168.1.20:8080/v1/chat/completions")
    assert built.api_key is None


def test_legacy_kwargs_keep_empty_keys_empty(monkeypatch, no_env_keys):
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    built = create_llm_provider("openai", api_endpoint="http://localhost:8080/v1",
                                model="m", api_key="")
    assert built.api_key is None
    built = create_llm_provider("openai", api_endpoint="http://localhost:8080/v1", model="m")
    assert built.api_key == FAKE_KEY


@pytest.mark.parametrize("provider", ["gemini", "openrouter", "mistral", "deepseek", "poe", "nim"])
def test_missing_key_raises(no_env_keys, provider):
    with pytest.raises(LLMClientConfigError, match=provider):
        _build(provider, api_key="")


def test_unknown_provider_raises():
    with pytest.raises(LLMClientConfigError, match="Unknown LLM provider 'nope'"):
        _build("nope")


def test_config_error_is_a_value_error():
    assert issubclass(LLMClientConfigError, ValueError)


def test_provider_config_never_prints_its_key():
    cfg = ProviderConfig(provider="mistral", model="m", api_key="k-m")
    assert "k-m" not in repr(cfg)


def test_provider_config_carries_retry_policy_and_sampling():
    policy = RetryPolicy(max_attempts=5, timeout=42)
    built = create_llm_provider(ProviderConfig(
        provider="deepseek", model="deepseek-chat", api_key=FAKE_KEY,
        retry=policy, temperature=0.7,
    ))
    assert built.effective_retry_policy() is policy
    assert built.effective_temperature() == 0.7


def test_config_and_kwargs_are_exclusive():
    with pytest.raises(TypeError):
        create_llm_provider(ProviderConfig(provider="ollama"), model="x")
