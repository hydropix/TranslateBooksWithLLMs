"""
Factory for creating LLM provider instances.

This module provides the create_llm_provider() function which instantiates
the appropriate provider based on the provider_type parameter.
"""

import os

from src.config import (
    API_ENDPOINT, DEFAULT_MODEL, OLLAMA_NUM_CTX,
    OPENROUTER_API_KEY, OPENROUTER_MODEL,
    MISTRAL_API_KEY, MISTRAL_MODEL, MISTRAL_API_ENDPOINT,
    DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_API_ENDPOINT,
    DEEPSEEK_DISABLE_THINKING,
    POE_API_KEY, POE_MODEL, POE_API_ENDPOINT,
    NIM_API_KEY, NIM_MODEL, NIM_API_ENDPOINT,
    LITELLM_MODEL
)
from .base import LLMProvider, normalize_api_keys
from .providers.ollama import OllamaProvider
from .providers.openai import OpenAICompatibleProvider
from .providers.gemini import GeminiProvider
from .providers.openrouter import OpenRouterProvider
from .providers.mistral import MistralProvider
from .providers.deepseek import DeepSeekProvider
from .providers.poe import PoeProvider
from .providers.litellm import LiteLLMProvider


def _require_key(raw, error_message: str):
    """Validate that at least one usable key is present in `raw`.

    Used by cloud providers in the factory to surface a clear error before
    instantiation. Returns the original `raw` value so providers receive the
    multi-key string unchanged (the base class handles splitting).
    """
    if not normalize_api_keys(raw):
        raise ValueError(error_message)
    return raw


def create_llm_provider(provider_type: str = "ollama", **kwargs) -> LLMProvider:
    """
    Create an LLM provider instance.

    Auto-detection: If provider_type is "ollama" and model name starts with "gemini",
    automatically switches to Gemini provider.

    Args:
        provider_type: Type of provider ("ollama", "openai", "gemini", "openrouter", "mistral", "deepseek", "poe", "nim", "litellm")
        **kwargs: Provider-specific parameters:
            - api_endpoint: API endpoint URL (Ollama, OpenAI)
            - model: Model name/identifier
            - api_key: API key (Gemini, OpenAI, OpenRouter)
            - context_window: Context window size (Ollama, OpenAI)
            - log_callback: Logging callback function (Ollama, OpenAI)

    Returns:
        Instantiated LLMProvider subclass

    Raises:
        ValueError: If provider_type is unknown or required parameters are missing

    Examples:
        >>> # Ollama provider
        >>> provider = create_llm_provider("ollama", model="llama3")

        >>> # OpenAI-compatible provider
        >>> provider = create_llm_provider("openai", api_key="sk-...", model="gpt-4")

        >>> # Gemini provider (auto-detected from model name)
        >>> provider = create_llm_provider("ollama", model="gemini-2.0-flash")

        >>> # OpenRouter provider
        >>> provider = create_llm_provider("openrouter", api_key="sk-or-...", model="anthropic/claude-sonnet-4")
    """
    # Auto-detect provider from model name if not explicitly set
    model = kwargs.get("model", DEFAULT_MODEL)
    if provider_type == "ollama" and model and model.startswith("gemini"):
        # Auto-switch to Gemini provider when Gemini model is detected
        provider_type = "gemini"

    if provider_type.lower() == "ollama":
        return OllamaProvider(
            api_endpoint=kwargs.get("api_endpoint") or kwargs.get("endpoint") or API_ENDPOINT,
            model=kwargs.get("model", DEFAULT_MODEL),
            context_window=kwargs.get("context_window") or OLLAMA_NUM_CTX,
            log_callback=kwargs.get("log_callback")
        )
    elif provider_type.lower() == "openai":
        api_endpoint = kwargs.get("api_endpoint") or kwargs.get("endpoint") or ""
        # Distinguish official OpenAI from local llama.cpp/vLLM/LM Studio for log clarity
        pname = "openai" if "api.openai.com" in api_endpoint else "openai-compatible"
        return OpenAICompatibleProvider(
            api_endpoint=api_endpoint,
            model=kwargs.get("model", DEFAULT_MODEL),
            api_key=kwargs.get("api_key") or kwargs.get("openai_api_key"),
            context_window=kwargs.get("context_window") or OLLAMA_NUM_CTX,
            log_callback=kwargs.get("log_callback"),
            provider_name=pname,
        )
    elif provider_type.lower() == "gemini":
        api_key = _require_key(
            kwargs.get("api_key") or kwargs.get("gemini_api_key") or os.getenv("GEMINI_API_KEY"),
            "Gemini provider requires an API key. Set GEMINI_API_KEY environment variable or pass api_key parameter."
        )
        return GeminiProvider(
            api_key=api_key,
            model=kwargs.get("model", "gemini-2.0-flash")
        )
    elif provider_type.lower() == "openrouter":
        api_key = _require_key(
            kwargs.get("api_key") or kwargs.get("openrouter_api_key")
            or os.getenv("OPENROUTER_API_KEY", OPENROUTER_API_KEY),
            "OpenRouter provider requires an API key. Set OPENROUTER_API_KEY environment variable or pass api_key parameter."
        )
        return OpenRouterProvider(
            api_key=api_key,
            model=kwargs.get("model", OPENROUTER_MODEL)
        )
    elif provider_type.lower() == "mistral":
        api_key = _require_key(
            kwargs.get("api_key") or kwargs.get("mistral_api_key")
            or os.getenv("MISTRAL_API_KEY", MISTRAL_API_KEY),
            "Mistral provider requires an API key. Set MISTRAL_API_KEY environment variable or pass api_key parameter."
        )
        return MistralProvider(
            api_key=api_key,
            model=kwargs.get("model", MISTRAL_MODEL),
            api_endpoint=MISTRAL_API_ENDPOINT
        )
    elif provider_type.lower() == "deepseek":
        api_key = _require_key(
            kwargs.get("api_key") or kwargs.get("deepseek_api_key")
            or os.getenv("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY),
            "DeepSeek provider requires an API key. Set DEEPSEEK_API_KEY environment variable or pass api_key parameter."
        )
        return DeepSeekProvider(
            api_key=api_key,
            model=kwargs.get("model", DEEPSEEK_MODEL),
            api_endpoint=DEEPSEEK_API_ENDPOINT,
            disable_thinking=kwargs.get("deepseek_disable_thinking", DEEPSEEK_DISABLE_THINKING)
        )
    elif provider_type.lower() == "poe":
        api_key = _require_key(
            kwargs.get("api_key") or kwargs.get("poe_api_key")
            or os.getenv("POE_API_KEY", POE_API_KEY),
            "Poe provider requires an API key. Get your key at https://poe.com/api_key"
        )
        return PoeProvider(
            api_key=api_key,
            model=kwargs.get("model", POE_MODEL),
            api_endpoint=POE_API_ENDPOINT
        )
    elif provider_type.lower() == "nim":
        api_key = _require_key(
            kwargs.get("api_key") or kwargs.get("nim_api_key")
            or os.getenv("NIM_API_KEY", NIM_API_KEY),
            "NVIDIA NIM provider requires an API key. Get your key at https://build.nvidia.com/"
        )
        return OpenAICompatibleProvider(
            api_key=api_key,
            model=kwargs.get("model", NIM_MODEL),
            api_endpoint=kwargs.get("api_endpoint", NIM_API_ENDPOINT),
            provider_name="nim",
        )

    elif provider_type.lower() == "litellm":
        # LiteLLM reads credentials from each provider's native env var, so no
        # key is required here. api_base is taken only from a dedicated kwarg,
        # never from the generic `endpoint` (which defaults to the Ollama URL).
        return LiteLLMProvider(
            model=kwargs.get("model") or LITELLM_MODEL or DEFAULT_MODEL,
            api_key=kwargs.get("api_key") or kwargs.get("litellm_api_key"),
            api_base=kwargs.get("litellm_api_base"),
        )

    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
