"""
Factory for creating LLM provider instances.

This module provides the create_llm_provider() function which instantiates
the appropriate provider based on the provider_type parameter.
"""

import os
from typing import Optional

from src.config import (
    API_ENDPOINT, DEFAULT_MODEL, OLLAMA_NUM_CTX,
    OPENAI_API_KEY, OPENAI_API_ENDPOINT,
    OPENROUTER_API_KEY, OPENROUTER_MODEL,
    MISTRAL_API_KEY, MISTRAL_MODEL, MISTRAL_API_ENDPOINT,
    DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_API_ENDPOINT,
    POE_API_KEY, POE_MODEL, POE_API_ENDPOINT,
    NIM_API_KEY, NIM_MODEL, NIM_API_ENDPOINT,
    FIREWORKS_API_KEY, FIREWORKS_MODEL, FIREWORKS_API_ENDPOINT
)
from .base import LLMProvider
from .providers.ollama import OllamaProvider
from .providers.openai import OpenAICompatibleProvider
from .providers.gemini import GeminiProvider
from .providers.openrouter import OpenRouterProvider
from .providers.mistral import MistralProvider
from .providers.deepseek import DeepSeekProvider
from .providers.poe import PoeProvider


def create_llm_provider(provider_type: str = "ollama", **kwargs) -> LLMProvider:
    """
    Create and return an LLM provider instance for the requested provider type.
    
    Auto-detects Gemini when `provider_type` is "ollama" and the `model` name starts with "gemini".
    
    Parameters:
        provider_type (str): Provider identifier (e.g., "ollama", "openai", "gemini", "openrouter", "mistral", "deepseek", "poe", "nim", "fireworks").
        **kwargs: Provider-specific overrides:
            - api_endpoint (str): API endpoint URL (used by Ollama, OpenAI-compatible providers).
            - model (str): Model name or identifier.
            - api_key (str): API key for providers that require authentication.
            - context_window (int): Context window size (Ollama, OpenAI-compatible).
            - log_callback (callable): Optional logging callback (Ollama, OpenAI-compatible).
    
    Returns:
        LLMProvider: An instantiated provider subclass configured according to `provider_type` and provided overrides.
    
    Raises:
        ValueError: If `provider_type` is unknown or a required API key is missing for providers that mandate one.
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
        api_key = kwargs.get("api_key") or kwargs.get("openai_api_key") or os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
        return OpenAICompatibleProvider(
            api_endpoint=kwargs.get("api_endpoint") or kwargs.get("endpoint") or OPENAI_API_ENDPOINT,
            model=kwargs.get("model", DEFAULT_MODEL),
            api_key=api_key,
            context_window=kwargs.get("context_window") or OLLAMA_NUM_CTX,
            log_callback=kwargs.get("log_callback")
        )
    elif provider_type.lower() == "gemini":
        api_key = kwargs.get("api_key") or kwargs.get("gemini_api_key")
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
        api_key = kwargs.get("api_key") or kwargs.get("openrouter_api_key")
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
        api_key = kwargs.get("api_key") or kwargs.get("mistral_api_key")
        if not api_key:
            # Try to get from environment
            api_key = os.getenv("MISTRAL_API_KEY", MISTRAL_API_KEY)
            if not api_key:
                raise ValueError("Mistral provider requires an API key. Set MISTRAL_API_KEY environment variable or pass api_key parameter.")
        return MistralProvider(
            api_key=api_key,
            model=kwargs.get("model", MISTRAL_MODEL),
            api_endpoint=MISTRAL_API_ENDPOINT
        )
    elif provider_type.lower() == "deepseek":
        api_key = kwargs.get("api_key") or kwargs.get("deepseek_api_key")
        if not api_key:
            # Try to get from environment
            api_key = os.getenv("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY)
            if not api_key:
                raise ValueError("DeepSeek provider requires an API key. Set DEEPSEEK_API_KEY environment variable or pass api_key parameter.")
        return DeepSeekProvider(
            api_key=api_key,
            model=kwargs.get("model", DEEPSEEK_MODEL),
            api_endpoint=DEEPSEEK_API_ENDPOINT
        )
    elif provider_type.lower() == "poe":
        api_key = kwargs.get("api_key") or kwargs.get("poe_api_key")
        if not api_key:
            # Try to get from environment
            api_key = os.getenv("POE_API_KEY", POE_API_KEY)
            if not api_key:
                raise ValueError("Poe provider requires an API key. Get your key at https://poe.com/api_key")
        return PoeProvider(
            api_key=api_key,
            model=kwargs.get("model", POE_MODEL),
            api_endpoint=POE_API_ENDPOINT
        )
    elif provider_type.lower() == "nim":
        api_key = kwargs.get("api_key") or kwargs.get("nim_api_key")
        if not api_key:
            api_key = os.getenv("NIM_API_KEY", NIM_API_KEY)
            if not api_key:
                raise ValueError("NVIDIA NIM provider requires an API key. Get your key at https://build.nvidia.com/")
        return OpenAICompatibleProvider(
            api_key=api_key,
            model=kwargs.get("model", NIM_MODEL),
            api_endpoint=kwargs.get("api_endpoint", NIM_API_ENDPOINT)
        )
    elif provider_type.lower() == "fireworks":
        api_key = kwargs.get("api_key") or kwargs.get("fireworks_api_key")
        if not api_key:
            api_key = os.getenv("FIREWORKS_API_KEY", FIREWORKS_API_KEY)
            if not api_key:
                raise ValueError("Fireworks provider requires an API key. Get your key at https://fireworks.ai/")
        return OpenAICompatibleProvider(
            api_key=api_key,
            model=kwargs.get("model", FIREWORKS_MODEL),
            api_endpoint=kwargs.get("api_endpoint", FIREWORKS_API_ENDPOINT)
        )

    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
