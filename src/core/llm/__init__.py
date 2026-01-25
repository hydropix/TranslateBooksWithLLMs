"""
LLM Provider System

This package provides a modular system for interacting with various LLM providers.

Public API:
    - Exceptions: ContextOverflowError, RepetitionLoopError
    - Base classes: LLMProvider, LLMResponse
    - Thinking system: ThinkingBehavior, get_thinking_behavior_sync, get_model_warning_message, detect_repetition_loop
    - Utilities: ContextDetector, TranslationExtractor
    - Providers: OllamaProvider, OpenAICompatibleProvider, OpenRouterProvider, GeminiProvider
    - Factory: create_llm_provider

Example usage:
    >>> from src.core.llm import create_llm_provider, OllamaProvider
    >>> provider = create_llm_provider("ollama", model="llama3")
    >>> response = await provider.generate("Translate: Hello")
"""

# Exceptions
from .exceptions import ContextOverflowError, RepetitionLoopError

# Base classes
from .base import LLMProvider, LLMResponse

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
from .providers.openrouter import OpenRouterProvider
from .providers.gemini import GeminiProvider
from .providers.mistral import MistralProvider
from .providers.deepseek import DeepSeekProvider

# Factory
from .factory import create_llm_provider

__all__ = [
    # Exceptions
    'ContextOverflowError',
    'RepetitionLoopError',

    # Base
    'LLMProvider',
    'LLMResponse',

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
    'OpenRouterProvider',
    'GeminiProvider',
    'MistralProvider',
    'DeepSeekProvider',

    # Factory
    'create_llm_provider',
]
