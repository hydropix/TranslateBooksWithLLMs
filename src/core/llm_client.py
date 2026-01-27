"""
Centralized LLM client for all API communication
"""
from typing import Optional, Dict, Any

from src.config import API_ENDPOINT, DEFAULT_MODEL
from src.core.llm import create_llm_provider, LLMProvider, ContextOverflowError, RepetitionLoopError, LLMResponse

# Re-export for convenience
__all__ = ['LLMClient', 'default_client', 'create_llm_client', 'ContextOverflowError', 'RepetitionLoopError', 'LLMResponse']


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
        if self._provider and hasattr(self._provider, '_is_thinking_model'):
            return self._provider._is_thinking_model
        return None

    async def detect_thinking_model(self) -> Optional[bool]:
        """
        Trigger thinking model detection (for Ollama provider).

        This sends a simple test prompt to detect if the model produces
        thinking output, and caches the result for future use.

        Returns:
            True if model produces thinking output, False if not, None if detection not supported
        """
        provider = self._get_provider()
        if hasattr(provider, '_detect_thinking_model'):
            # Trigger detection if not already done
            if provider._is_thinking_model is None:
                provider._is_thinking_model = await provider._detect_thinking_model()
            return provider._is_thinking_model
        return None


# Global instance for backward compatibility
default_client = LLMClient(provider_type="ollama", api_endpoint=API_ENDPOINT, model=DEFAULT_MODEL)


def create_llm_client(llm_provider: str, gemini_api_key: Optional[str],
                      api_endpoint: str, model_name: str,
                      openai_api_key: Optional[str] = None,
                      openrouter_api_key: Optional[str] = None,
                      mistral_api_key: Optional[str] = None,
                      deepseek_api_key: Optional[str] = None,
                      poe_api_key: Optional[str] = None,
                      context_window: Optional[int] = None,
                      log_callback: Optional[callable] = None) -> Optional[LLMClient]:
    """
    Factory function to create LLM client based on provider or custom endpoint

    Args:
        llm_provider: Provider type ('ollama', 'gemini', 'openai', 'openrouter', 'mistral', 'deepseek', or 'poe')
        gemini_api_key: API key for Gemini provider
        api_endpoint: API endpoint for custom Ollama instance or OpenAI-compatible API
        model_name: Model name to use
        openai_api_key: API key for OpenAI provider
        openrouter_api_key: API key for OpenRouter provider
        mistral_api_key: API key for Mistral provider
        deepseek_api_key: API key for DeepSeek provider
        poe_api_key: API key for Poe provider
        context_window: Context window size for the model
        log_callback: Callback function for logging

    Returns:
        LLMClient instance or None if using default client
    """
    if llm_provider == "gemini" and gemini_api_key:
        return LLMClient(provider_type="gemini", api_key=gemini_api_key, model=model_name)
    if llm_provider == "openai":
        return LLMClient(provider_type="openai", api_endpoint=api_endpoint, model=model_name,
                         api_key=openai_api_key, context_window=context_window, log_callback=log_callback)
    if llm_provider == "openrouter":
        return LLMClient(provider_type="openrouter", model=model_name, api_key=openrouter_api_key)
    if llm_provider == "mistral":
        return LLMClient(provider_type="mistral", model=model_name, api_key=mistral_api_key)
    if llm_provider == "deepseek":
        return LLMClient(provider_type="deepseek", model=model_name, api_key=deepseek_api_key)
    if llm_provider == "poe":
        return LLMClient(provider_type="poe", model=model_name, api_key=poe_api_key)
    if llm_provider == "ollama":
        # Always create a new client for Ollama to ensure proper configuration
        return LLMClient(provider_type="ollama", api_endpoint=api_endpoint, model=model_name,
                         context_window=context_window, log_callback=log_callback)
    return None