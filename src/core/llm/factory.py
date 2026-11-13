"""
The one construction path for LLM providers.

``create_llm_provider`` takes a ``ProviderConfig`` (or, for existing callers,
a provider type plus keyword arguments that are normalised into one) and
returns a ready provider, raising ``LLMClientConfigError`` (a ``ValueError``)
on misconfiguration: unknown provider, or a provider that needs a key and got
none.

Credentials: a key passed explicitly as an empty string stays empty; the
``.env`` key is used only when the caller passed no key at all (``None`` /
absent). The web layer relies on this to keep a stored key from following a
request-chosen endpoint (``src/api/services/credential_guard.py``).
"""

from typing import Any, Union

from src import config as _config

from .base import LLMProvider, normalize_api_keys
from .exceptions import LLMClientConfigError
from .provider_config import (
    KEY_REQUIRED_PROVIDERS,
    SUPPORTED_PROVIDERS,
    ProviderConfig,
    _env_key,
)
from .providers.deepseek import DeepSeekProvider
from .providers.gemini import GeminiProvider
from .providers.litellm import LiteLLMProvider
from .providers.mistral import MistralProvider
from .providers.ollama import OllamaProvider
from .providers.openai import OpenAICompatibleProvider
from .providers.openrouter import OpenRouterProvider
from .providers.poe import PoeProvider

_KEY_HINTS = {
    "gemini": "Set GEMINI_API_KEY in .env or pass an API key.",
    "openrouter": "Set OPENROUTER_API_KEY in .env or pass an API key.",
    "mistral": "Set MISTRAL_API_KEY in .env or pass an API key.",
    "deepseek": "Set DEEPSEEK_API_KEY in .env or pass an API key.",
    "poe": "Get your key at https://poe.com/api_key",
    "nim": "Get your key at https://build.nvidia.com/",
}

_PROVIDER_LABELS = {
    "gemini": "Gemini", "openrouter": "OpenRouter", "mistral": "Mistral",
    "deepseek": "DeepSeek", "poe": "Poe", "nim": "NVIDIA NIM",
}


def _config_from_kwargs(provider_type: str, kwargs: dict) -> ProviderConfig:
    """Normalise the legacy keyword arguments into a ``ProviderConfig``."""
    provider = (provider_type or "ollama").strip().lower()
    model = kwargs.get("model") or ""
    # Auto-detect: an "ollama" job asking for a Gemini model goes to Gemini.
    if provider == "ollama" and model.startswith("gemini"):
        provider = "gemini"

    api_key = kwargs.get("api_key")
    if api_key is None:
        api_key = _env_key(provider) or None

    options = {}
    for option in ("disable_thinking", "disable_web_search"):
        value = kwargs.get(f"{provider}_{option}", kwargs.get(option))
        if value is not None:
            options[option] = value
    if kwargs.get("litellm_api_base"):
        options["api_base"] = kwargs["litellm_api_base"]

    return ProviderConfig(
        provider=provider,
        model=model,
        endpoint=kwargs.get("api_endpoint") or kwargs.get("endpoint"),
        api_key=api_key,
        context_window=kwargs.get("context_window"),
        retry=kwargs.get("retry_policy") or kwargs.get("retry"),
        temperature=kwargs.get("temperature"),
        options=options,
        log_callback=kwargs.get("log_callback"),
    )


def _require_key(cfg: ProviderConfig) -> str:
    if not normalize_api_keys(cfg.api_key):
        label = _PROVIDER_LABELS.get(cfg.provider, cfg.provider)
        raise LLMClientConfigError(
            f"{label} provider ('{cfg.provider}') requires an API key. "
            f"{_KEY_HINTS.get(cfg.provider, '')}".strip()
        )
    return cfg.api_key


def create_llm_provider(
    provider: Union[ProviderConfig, str, None] = "ollama", **kwargs: Any
) -> LLMProvider:
    """
    Create an LLM provider instance.

    Args:
        provider: A ``ProviderConfig`` (preferred), or a provider type
            ("ollama", "openai", "gemini", "openrouter", "mistral", "deepseek",
            "poe", "nim", "litellm") followed by keyword arguments: ``model``,
            ``api_endpoint``/``endpoint``, ``api_key``,
            ``context_window``, ``log_callback``, ``retry_policy``,
            ``temperature``, ``<provider>_disable_thinking``,
            ``poe_disable_web_search``, ``litellm_api_base``.

    Returns:
        A provider instance. Each call builds a new one; providers are never
        shared between jobs.

    Raises:
        LLMClientConfigError: Unknown provider, or no usable key for a provider
            that requires one.
    """
    if isinstance(provider, ProviderConfig):
        cfg = provider
        if kwargs:
            raise TypeError("create_llm_provider: pass either a ProviderConfig or keyword "
                            "arguments, not both")
        if cfg.provider == "ollama" and (cfg.model or "").startswith("gemini"):
            cfg = cfg.with_changes(provider="gemini")
    else:
        if provider is None:
            provider = kwargs.pop("provider_type", "ollama")
        cfg = _config_from_kwargs(provider, kwargs)

    if cfg.provider not in SUPPORTED_PROVIDERS:
        raise LLMClientConfigError(
            f"Unknown LLM provider '{cfg.provider}'. "
            f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}."
        )
    if cfg.provider in KEY_REQUIRED_PROVIDERS:
        _require_key(cfg)

    common = dict(retry_policy=cfg.retry, temperature=cfg.temperature)
    opts = cfg.options or {}

    if cfg.provider == "ollama":
        return OllamaProvider(
            api_endpoint=cfg.endpoint or _config.API_ENDPOINT,
            model=cfg.model or _config.DEFAULT_MODEL,
            context_window=cfg.context_window or _config.OLLAMA_NUM_CTX,
            log_callback=cfg.log_callback,
            **common,
        )
    if cfg.provider == "openai":
        endpoint = cfg.endpoint or ""
        # Distinguish official OpenAI from local llama.cpp/vLLM/LM Studio in logs.
        # The key stays optional: local OpenAI-compatible servers need none.
        return OpenAICompatibleProvider(
            api_endpoint=endpoint,
            model=cfg.model or _config.DEFAULT_MODEL,
            api_key=cfg.api_key or None,
            context_window=cfg.context_window or _config.OLLAMA_NUM_CTX,
            log_callback=cfg.log_callback,
            provider_name="openai" if "api.openai.com" in endpoint else "openai-compatible",
            **common,
        )
    if cfg.provider == "nim":
        return OpenAICompatibleProvider(
            api_endpoint=cfg.endpoint or _config.NIM_API_ENDPOINT,
            model=cfg.model or _config.NIM_MODEL,
            api_key=cfg.api_key,
            context_window=cfg.context_window or _config.OLLAMA_NUM_CTX,
            log_callback=cfg.log_callback,
            provider_name="nim",
            **common,
        )
    if cfg.provider == "gemini":
        return GeminiProvider(
            api_key=cfg.api_key,
            model=cfg.model or "gemini-3.6-flash",
            log_callback=cfg.log_callback,
            **common,
        )
    if cfg.provider == "openrouter":
        return OpenRouterProvider(
            api_key=cfg.api_key,
            model=cfg.model or _config.OPENROUTER_MODEL,
            disable_thinking=opts.get("disable_thinking", _config.OPENROUTER_DISABLE_THINKING),
            log_callback=cfg.log_callback,
            **common,
        )
    if cfg.provider == "mistral":
        return MistralProvider(
            api_key=cfg.api_key,
            model=cfg.model or _config.MISTRAL_MODEL,
            api_endpoint=_config.MISTRAL_API_ENDPOINT,
            log_callback=cfg.log_callback,
            **common,
        )
    if cfg.provider == "deepseek":
        return DeepSeekProvider(
            api_key=cfg.api_key,
            model=cfg.model or _config.DEEPSEEK_MODEL,
            api_endpoint=_config.DEEPSEEK_API_ENDPOINT,
            disable_thinking=opts.get("disable_thinking", _config.DEEPSEEK_DISABLE_THINKING),
            log_callback=cfg.log_callback,
            **common,
        )
    if cfg.provider == "poe":
        return PoeProvider(
            api_key=cfg.api_key,
            model=cfg.model or _config.POE_MODEL,
            api_endpoint=_config.POE_API_ENDPOINT,
            disable_thinking=opts.get("disable_thinking", _config.POE_DISABLE_THINKING),
            disable_web_search=opts.get("disable_web_search", _config.POE_DISABLE_WEB_SEARCH),
            log_callback=cfg.log_callback,
            **common,
        )
    # litellm: credentials come from each upstream provider's own env var
    # unless a key is given. api_base only from its dedicated option, never
    # from the generic endpoint (which defaults to the Ollama URL).
    return LiteLLMProvider(
        model=cfg.model or _config.LITELLM_MODEL or _config.DEFAULT_MODEL,
        api_key=cfg.api_key or None,
        api_base=opts.get("api_base"),
        log_callback=cfg.log_callback,
        **common,
    )
