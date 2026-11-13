"""
Centralized LLM client for all API communication
"""
from typing import Optional

from src.config import API_ENDPOINT, DEFAULT_MODEL
from src.core.llm import (
    create_llm_provider, LLMProvider, ContextOverflowError, RepetitionLoopError, LLMResponse,
    LLMClientConfigError,
)
from src.core.llm.thinking.behavior import ThinkingBehavior

# Re-export for convenience
__all__ = ['LLMClient', 'default_client', 'create_llm_client', 'LLMClientConfigError',
           'SUPPORTED_CLIENT_PROVIDERS', 'ContextOverflowError', 'RepetitionLoopError', 'LLMResponse']

# Providers create_llm_client() can build (all those create_llm_provider() supports).
SUPPORTED_CLIENT_PROVIDERS = (
    'ollama', 'openai', 'gemini', 'openrouter', 'mistral', 'deepseek', 'poe', 'nim', 'litellm',
)


class LLMClient:
    """Centralized client for LLM API communication"""
    
    def __init__(self, provider_type: str = "ollama", **kwargs):
        self.provider_type = provider_type
        self.provider_kwargs = kwargs
        self._provider: Optional[LLMProvider] = None
        
        # For backward compatibility
        if "api_endpoint" in kwargs and "model" in kwargs:
            self.api_endpoint = kwargs["api_endpoint"]
            self.model = kwargs["model"]
        else:
            self.api_endpoint = API_ENDPOINT
            self.model = DEFAULT_MODEL
    
    def _get_provider(self) -> LLMProvider:
        """Get or create the LLM provider"""
        if not self._provider:
            self._provider = create_llm_provider(self.provider_type, **self.provider_kwargs)
        return self._provider
    
    @property
    def context_window(self) -> int:
        """Get the current context window size from the provider"""
        if self._provider and hasattr(self._provider, 'context_window'):
            return self._provider.context_window
        return self.provider_kwargs.get('context_window', 2048)

    @context_window.setter
    def context_window(self, value: int):
        """Set the context window size on the provider"""
        if self._provider and hasattr(self._provider, 'context_window'):
            self._provider.context_window = value
        self.provider_kwargs['context_window'] = value

    async def generate(self, prompt: str, system_prompt: Optional[str] = None,
                      timeout: int = None) -> Optional[LLMResponse]:
        """
        Generate a response from the LLM (alias for make_request for backward compatibility)

        Args:
            prompt: The user prompt to send
            system_prompt: Optional system prompt (role/instructions)
            timeout: Request timeout in seconds

        Returns:
            LLMResponse with content and token usage info, or None if failed
        """
        provider = self._get_provider()

        if timeout:
            return await provider.generate(prompt, timeout, system_prompt=system_prompt)
        else:
            return await provider.generate(prompt, system_prompt=system_prompt)

    async def make_request(self, prompt: str, model: Optional[str] = None,
                    timeout: int = None, system_prompt: Optional[str] = None) -> Optional[LLMResponse]:
        """
        Make a request to the LLM API with error handling and retries

        Args:
            prompt: The user prompt to send (content to process)
            model: Model to use (defaults to instance model)
            timeout: Request timeout in seconds
            system_prompt: Optional system prompt (role/instructions)

        Returns:
            LLMResponse with content and token usage info, or None if failed
        """
        provider = self._get_provider()

        # Update model if specified
        if model:
            provider.model = model

        if timeout:
            return await provider.generate(prompt, timeout, system_prompt=system_prompt)
        else:
            return await provider.generate(prompt, system_prompt=system_prompt)
    
    def extract_translation(self, response: str) -> Optional[str]:
        """
        Extract translation from response using configured tags
        
        Args:
            response: Raw LLM response
            
        Returns:
            Extracted translation or None if not found
        """
        provider = self._get_provider()
        return provider.extract_translation(response)
    
    async def translate_text(self, prompt: str, model: Optional[str] = None) -> Optional[str]:
        """
        Complete translation workflow: request + extraction
        
        Args:
            prompt: Translation prompt
            model: Model to use
            
        Returns:
            Extracted translation or None if failed
        """
        provider = self._get_provider()
        
        # Update model if specified
        if model:
            provider.model = model
            
        return await provider.translate_text(prompt)
    
    async def close(self):
        """Close the HTTP client and clean up resources"""
        if self._provider:
            await self._provider.close()
            self._provider = None

    def get_is_thinking_model(self) -> Optional[bool]:
        """
        Get the thinking model status from the provider (if available).

        Returns:
            True if model produces thinking output, False if not, None if unknown/not detected yet
        """
        if not self._provider:
            return None
        provider = self._provider
        if hasattr(provider, '_thinking_behavior'):
            # Ollama: behavior enum, None until detected
            behavior = provider._thinking_behavior
            if behavior is None:
                return None
            return behavior != ThinkingBehavior.STANDARD
        status = getattr(provider, '_is_thinking_model', None)
        if callable(status):
            # Gemini, DeepSeek: name-based check exposed as a method
            status = status()
        return None if status is None else bool(status)

    async def detect_thinking_model(self) -> Optional[bool]:
        """
        Trigger thinking model detection (for Ollama provider).

        Ollama classifies the model's thinking behavior (from its cache, the
        known model lists, or test prompts) and keeps the result for later
        requests. Other providers report their name-based status.

        Returns:
            True if model produces thinking output, False if not, None if unknown
        """
        provider = self._get_provider()
        if hasattr(provider, '_detect_thinking_behavior') and provider._thinking_behavior is None:
            provider._thinking_behavior = await provider._detect_thinking_behavior()
        return self.get_is_thinking_model()


# Global instance for backward compatibility
default_client = LLMClient(provider_type="ollama", api_endpoint=API_ENDPOINT, model=DEFAULT_MODEL)


def create_llm_client(llm_provider: str, gemini_api_key: Optional[str],
                      api_endpoint: str, model_name: str,
                      openai_api_key: Optional[str] = None,
                      openrouter_api_key: Optional[str] = None,
                      mistral_api_key: Optional[str] = None,
                      deepseek_api_key: Optional[str] = None,
                      poe_api_key: Optional[str] = None,
                      nim_api_key: Optional[str] = None,
                      context_window: Optional[int] = None,
                      log_callback: Optional[callable] = None) -> LLMClient:
    """
    Build an LLMClient for a provider, delegating to create_llm_provider().

    The provider is instantiated eagerly so a bad configuration fails here,
    with a clear message, instead of at the first request (or, as before,
    by silently falling back to the module-level Ollama client).

    Args:
        llm_provider: One of SUPPORTED_CLIENT_PROVIDERS ('ollama', 'openai',
            'gemini', 'openrouter', 'mistral', 'deepseek', 'poe', 'nim',
            'litellm'). Case-insensitive.
        gemini_api_key: API key for Gemini provider
        api_endpoint: Endpoint for Ollama or an OpenAI-compatible API
            (ignored by the other providers)
        model_name: Model name to use
        openai_api_key: API key for OpenAI provider
        openrouter_api_key: API key for OpenRouter provider
        mistral_api_key: API key for Mistral provider
        deepseek_api_key: API key for DeepSeek provider
        poe_api_key: API key for Poe provider
        nim_api_key: API key for NVIDIA NIM provider
        context_window: Context window size (Ollama and OpenAI-compatible)
        log_callback: Logging callback (Ollama and OpenAI-compatible)

    Empty keys fall back to the provider's environment variable, as in
    create_llm_provider(). LiteLLM reads credentials from each upstream
    provider's own environment variable.

    Returns:
        A ready LLMClient (never None).

    Raises:
        LLMClientConfigError: Unknown provider, or no usable API key for a
            provider that requires one.
    """
    provider = (llm_provider or "").strip().lower()
    per_provider_kwargs = {
        "ollama": dict(api_endpoint=api_endpoint, model=model_name,
                       context_window=context_window, log_callback=log_callback),
        "openai": dict(api_endpoint=api_endpoint, model=model_name, api_key=openai_api_key,
                       context_window=context_window, log_callback=log_callback),
        "gemini": dict(api_key=gemini_api_key, model=model_name),
        "openrouter": dict(api_key=openrouter_api_key, model=model_name),
        "mistral": dict(api_key=mistral_api_key, model=model_name),
        "deepseek": dict(api_key=deepseek_api_key, model=model_name),
        "poe": dict(api_key=poe_api_key, model=model_name),
        "nim": dict(api_key=nim_api_key, model=model_name),
        "litellm": dict(model=model_name),
    }
    if provider not in per_provider_kwargs:
        raise LLMClientConfigError(
            f"Unknown LLM provider '{llm_provider}'. "
            f"Supported providers: {', '.join(SUPPORTED_CLIENT_PROVIDERS)}."
        )

    kwargs = per_provider_kwargs[provider]
    client = LLMClient(provider_type=provider, **kwargs)
    try:
        client._provider = create_llm_provider(provider, **kwargs)
    except ValueError as e:
        raise LLMClientConfigError(f"Cannot create the '{provider}' LLM client: {e}") from e
    return client
