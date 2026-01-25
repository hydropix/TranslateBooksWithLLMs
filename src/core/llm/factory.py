"""
Factory for creating LLM provider instances.

This module provides the create_llm_provider() function which instantiates
the appropriate provider based on the provider_type parameter.
"""

import os
from typing import Optional

from src.config import (
    API_ENDPOINT, DEFAULT_MODEL, OLLAMA_NUM_CTX,
    OPENROUTER_API_KEY, OPENROUTER_MODEL,
    MISTRAL_API_KEY, MISTRAL_MODEL,
    DEEPSEEK_API_KEY, DEEPSEEK_MODEL
)
from .base import LLMProvider
from .providers.ollama import OllamaProvider
from .providers.openai import OpenAICompatibleProvider
from .providers.gemini import GeminiProvider
from .providers.openrouter import OpenRouterProvider
from .providers.mistral import MistralProvider
from .providers.deepseek import DeepSeekProvider


def create_llm_provider(provider_type: str = "ollama", **kwargs) -> LLMProvider:
    """
    Create an LLM provider instance.

    Auto-detection: If provider_type is "ollama" and model name starts with "gemini",
    automatically switches to Gemini provider.

    Args:
        provider_type: Type of provider ("ollama", "openai", "gemini", "openrouter", "mistral", "deepseek")
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
            api_endpoint=kwargs.get("api_endpoint", API_ENDPOINT),
            model=kwargs.get("model", DEFAULT_MODEL),
            context_window=kwargs.get("context_window") or OLLAMA_NUM_CTX,
            log_callback=kwargs.get("log_callback")
        )
    elif provider_type.lower() == "openai":
        return OpenAICompatibleProvider(
            api_endpoint=kwargs.get("api_endpoint"),
            model=kwargs.get("model", DEFAULT_MODEL),
            api_key=kwargs.get("api_key"),
            context_window=kwargs.get("context_window") or OLLAMA_NUM_CTX,
            log_callback=kwargs.get("log_callback")
        )
    elif provider_type.lower() == "gemini":
        api_key = kwargs.get("api_key")
        if not api_key:
            # Try to get from environment
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Gemini provider requires an API key. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
        return GeminiProvider(
            api_key=api_key,
            model=kwargs.get("model", "gemini-2.0-flash")
        )
    elif provider_type.lower() == "openrouter":
        api_key = kwargs.get("api_key")
        if not api_key:
            # Try to get from environment
            api_key = os.getenv("OPENROUTER_API_KEY", OPENROUTER_API_KEY)
            if not api_key:
                raise ValueError("OpenRouter provider requires an API key. Set OPENROUTER_API_KEY environment variable or pass api_key parameter.")
        return OpenRouterProvider(
            api_key=api_key,
            model=kwargs.get("model", OPENROUTER_MODEL)
        )
    elif provider_type.lower() == "mistral":
        api_key = kwargs.get("api_key")
        if not api_key:
            # Try to get from environment
            api_key = os.getenv("MISTRAL_API_KEY", MISTRAL_API_KEY)
            if not api_key:
                raise ValueError("Mistral provider requires an API key. Set MISTRAL_API_KEY environment variable or pass api_key parameter.")
        return MistralProvider(
            api_key=api_key,
            model=kwargs.get("model", MISTRAL_MODEL),
            api_endpoint=kwargs.get("api_endpoint")
        )
    elif provider_type.lower() == "deepseek":
        api_key = kwargs.get("api_key")
        if not api_key:
            # Try to get from environment
            api_key = os.getenv("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY)
            if not api_key:
                raise ValueError("DeepSeek provider requires an API key. Set DEEPSEEK_API_KEY environment variable or pass api_key parameter.")
        return DeepSeekProvider(
            api_key=api_key,
            model=kwargs.get("model", DEEPSEEK_MODEL),
            api_endpoint=kwargs.get("api_endpoint")
        )
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
