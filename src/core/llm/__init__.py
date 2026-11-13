"""
LLM Provider System

This package provides a modular system for interacting with various LLM providers.

Public API:
    - Exceptions: ContextOverflowError, RepetitionLoopError, RateLimitError, LLMClientConfigError
    - Base classes: LLMProvider (owns the shared request loop), LLMRequest, LLMResponse
    - Configuration: ProviderConfig, provider_config_from_args, RetryPolicy
    - Thinking system: ThinkingBehavior, get_thinking_behavior_sync, get_model_warning_message, detect_repetition_loop
    - Utilities: ContextDetector, TranslationExtractor
    - Providers: OllamaProvider, OpenAICompatibleProvider, OpenRouterProvider, GeminiProvider
    - Factory: create_llm_provider (the only construction path), create_provider_from_args

Example usage:
    >>> from src.core.llm import create_llm_provider, OllamaProvider
    >>> provider = create_llm_provider("ollama", model="llama3")
    >>> response = await provider.generate("Translate: Hello")
"""

# Exceptions
from .exceptions import (
    ContextOverflowError, RepetitionLoopError, RateLimitError, LLMClientConfigError,
)

# Base classes, configuration and policy
from .base import LLMProvider, LLMRequest, LLMResponse
from .policy import RetryPolicy, begin_job_retry_policy, current_retry_policy, end_job_retry_policy
from .provider_config import ProviderConfig, provider_config_from_args

# Thinking system
from .thinking.behavior import ThinkingBehavior, get_thinking_behavior_sync, get_model_warning_message
from .thinking.cache import ThinkingCache, get_thinking_cache
from .thinking.detection import detect_repetition_loop

# Utilities
from .utils.context_detection import ContextDetector
from .utils.extraction import TranslationExtractor

# Providers
from .providers.ollama import OllamaProvider
from .providers.openai import OpenAICompatibleProvider
from .providers.openai_chat import OpenAIChatProvider
from .providers.openrouter import OpenRouterProvider
from .providers.gemini import GeminiProvider
from .providers.mistral import MistralProvider
from .providers.deepseek import DeepSeekProvider
from .providers.poe import PoeProvider
from .providers.litellm import LiteLLMProvider

# Factory
from .factory import create_llm_provider, create_provider_from_args

__all__ = [
    # Exceptions
    'ContextOverflowError',
    'RepetitionLoopError',
    'RateLimitError',
    'LLMClientConfigError',

    # Base
    'LLMProvider',
    'LLMRequest',
    'LLMResponse',
    'RetryPolicy',
    'begin_job_retry_policy',
    'current_retry_policy',
    'end_job_retry_policy',
    'ProviderConfig',
    'provider_config_from_args',

    # Thinking
    'ThinkingBehavior',
    'ThinkingCache',
    'get_thinking_cache',
    'get_thinking_behavior_sync',
    'get_model_warning_message',
    'detect_repetition_loop',

    # Utilities
    'ContextDetector',
    'TranslationExtractor',

    # Providers
    'OllamaProvider',
    'OpenAICompatibleProvider',
    'OpenAIChatProvider',
    'LiteLLMProvider',
    'OpenRouterProvider',
    'GeminiProvider',
    'MistralProvider',
    'DeepSeekProvider',
    'PoeProvider',

    # Factory
    'create_llm_provider',
    'create_provider_from_args',
]
