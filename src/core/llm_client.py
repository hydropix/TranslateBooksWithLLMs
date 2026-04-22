"""
Centralized LLM client for all API communication
"""
from typing import Optional, Dict, Any
import asyncio
import time

from src.config import API_ENDPOINT, DEFAULT_MODEL
from src.core.llm import create_llm_provider, LLMProvider, ContextOverflowError, RepetitionLoopError, LLMResponse

# Re-export for convenience
__all__ = ['LLMClient', 'default_client', 'create_llm_client', 'ContextOverflowError', 'RepetitionLoopError', 'LLMResponse']


class LLMClient:
    """Centralized client for LLM API communication"""
    
    def __init__(self, provider_type: str = "ollama", **kwargs):
        """
        Initialize an LLMClient and configure request pacing, provider selection, and provider-specific kwargs.
        
        Parameters:
        	provider_type (str): The provider identifier (e.g., "ollama", "openai") used to create the underlying provider.
        	**kwargs: Additional configuration for the client and provider. Recognized keys:
        		- min_request_interval (float): Minimum interval, in seconds, between requests (default 0).
        		- max_request_interval (float): Maximum interval, in seconds, for backoff (default 5.0).
        		- adaptive_request_backoff (bool): Enable adaptive backoff on failures (default True).
        		- api_endpoint (str): Optional provider API endpoint; if present alongside `model`, used for backward-compatible attributes.
        		- model (str): Optional model name; if present alongside `api_endpoint`, used for backward-compatible attributes.
        		Any other keys are retained in `self.provider_kwargs` for provider construction.
        
        Notes:
        	Sets up internal pacing state (`_dynamic_request_interval`, `_next_request_at`) and an asyncio lock (`_request_gate_lock`) to serialize pacing waits. The actual provider is created lazily and stored in `self._provider`.
        """
        self.min_request_interval = float(kwargs.pop("min_request_interval", 0) or 0)
        self.max_request_interval = float(kwargs.pop("max_request_interval", 5.0) or 5.0)
        self.adaptive_request_backoff = bool(kwargs.pop("adaptive_request_backoff", True))
        self._dynamic_request_interval = self.min_request_interval
        self._next_request_at = 0.0
        self._request_gate_lock = asyncio.Lock()

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
        """
        Return the cached LLM provider, creating and caching it if it does not already exist.
        
        Returns:
            LLMProvider: The provider instance associated with this client.
        """
        if not self._provider:
            self._provider = create_llm_provider(self.provider_type, **self.provider_kwargs)
        return self._provider

    async def _wait_for_request_slot(self):
        """
        Waits until the next permitted request time according to the client's dynamic pacing interval.
        
        If the computed interval is greater than zero, acquires the internal request gate lock, sleeps until the next scheduled request time if needed, and advances the internal next-request timestamp to enforce the interval.
        """
        interval = max(0.0, self._dynamic_request_interval)
        if interval <= 0:
            return

        async with self._request_gate_lock:
            now = time.monotonic()
            if self._next_request_at > now:
                await asyncio.sleep(self._next_request_at - now)
            self._next_request_at = time.monotonic() + interval

    def _update_request_pacing(self, success: bool):
        """
        Adjust the client's dynamic request interval based on the outcome of the most recent request.
        
        If adaptive backoff is disabled, this is a no-op. On success, reduce the dynamic interval by 10%, not going below `min_request_interval`. On failure or empty response, increase the interval (doubling a non-zero current interval or using `max(min_request_interval, 0.2)` as a base) and clamp the result between `min_request_interval` and `max_request_interval`.
        
        Parameters:
            success (bool): `True` if the previous request produced a successful/non-empty response; `False` for failure or empty response.
        """
        if not self.adaptive_request_backoff:
            return

        if success:
            self._dynamic_request_interval = max(
                self.min_request_interval,
                self._dynamic_request_interval * 0.9
            )
            return

        # Failure/empty response: back off aggressively to avoid hammering APIs
        base = self._dynamic_request_interval if self._dynamic_request_interval > 0 else max(self.min_request_interval, 0.2)
        self._dynamic_request_interval = min(self.max_request_interval, max(self.min_request_interval, base * 2))
    
    @property
    def context_window(self) -> int:
        """
        Get the current context window size used by the client.
        
        Returns:
            int: The provider's context window size in tokens, or the configured fallback from `provider_kwargs`. If no value is set, returns 2048.
        """
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
                      Generate a response from the configured LLM for the given prompt.
                      
                      Parameters:
                          prompt (str): The user-visible prompt to send to the model.
                          system_prompt (Optional[str]): Optional system-level instructions or role context to include.
                          timeout (int): Optional request timeout in seconds.
                      
                      Returns:
                          LLMResponse: The model response including content and token usage information, or `None` if the request failed.
                      """
        provider = self._get_provider()
        await self._wait_for_request_slot()

        if timeout:
            response = await provider.generate(prompt, timeout, system_prompt=system_prompt)
        else:
            response = await provider.generate(prompt, system_prompt=system_prompt)

        self._update_request_pacing(bool(response))
        return response

    async def make_request(self, prompt: str, model: Optional[str] = None,
                    timeout: int = None, system_prompt: Optional[str] = None) -> Optional[LLMResponse]:
        """
                    Send a prompt to the configured LLM provider and return its response.
                    
                    Parameters:
                    	prompt (str): The user prompt to send to the model.
                    	model (Optional[str]): Optional model name to override the client's default.
                    	timeout (int): Optional request timeout in seconds.
                    	system_prompt (Optional[str]): Optional system-level instructions to include with the prompt.
                    
                    Returns:
                    	LLMResponse | None: The provider's response containing generated content and token-usage info, or `None` if no response was produced.
                    """
        provider = self._get_provider()

        # Update model if specified
        if model:
            provider.model = model

        await self._wait_for_request_slot()
        if timeout:
            response = await provider.generate(prompt, timeout, system_prompt=system_prompt)
        else:
            response = await provider.generate(prompt, system_prompt=system_prompt)

        self._update_request_pacing(bool(response))
        return response
    
    def extract_translation(self, response: str) -> Optional[str]:
        """
        Extracts a translated string from an LLM response using the client's provider-configured tags.
        
        Parameters:
            response (str): Raw LLM response text to parse for a translation.
        
        Returns:
            str | None: The extracted translation if present, otherwise `None`.
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
                      fireworks_api_key: Optional[str] = None,
                      nim_api_key: Optional[str] = None,
                      min_request_interval: float = 0.0,
                      adaptive_request_backoff: bool = True,
                      max_request_interval: float = 5.0,
                      context_window: Optional[int] = None,
                      log_callback: Optional[callable] = None) -> Optional[LLMClient]:
    """
                      Create and configure an LLMClient for the specified provider.
                      
                      Parameters:
                          llm_provider (str): Provider identifier ('ollama', 'gemini', 'openai', 'openrouter',
                              'mistral', 'deepseek', 'poe', 'nim', or 'fireworks').
                          gemini_api_key (Optional[str]): API key required for the 'gemini' provider.
                          api_endpoint (str): API endpoint or host for providers that use a custom endpoint (e.g., Ollama, OpenAI-compatible).
                          model_name (str): Model name to configure on the client.
                          openai_api_key (Optional[str]): API key for OpenAI.
                          openrouter_api_key (Optional[str]): API key for OpenRouter.
                          mistral_api_key (Optional[str]): API key for Mistral.
                          deepseek_api_key (Optional[str]): API key for DeepSeek.
                          poe_api_key (Optional[str]): API key for Poe.
                          fireworks_api_key (Optional[str]): API key for Fireworks.
                          nim_api_key (Optional[str]): API key for NVIDIA NIM.
                          min_request_interval (float): Minimum delay between requests in seconds.
                          adaptive_request_backoff (bool): If true, increase the inter-request delay after failures.
                          max_request_interval (float): Maximum adaptive delay in seconds.
                          context_window (Optional[int]): Optional context window size to pass to providers that support it.
                          log_callback (Optional[callable]): Optional logging callback to attach to the client.
                      
                      Returns:
                          Optional[LLMClient]: A configured LLMClient for the requested provider, or `None` if the provider is unsupported
                          or required credentials (e.g., Gemini API key for 'gemini') are missing.
                      """
    if llm_provider == "gemini" and gemini_api_key:
        return LLMClient(provider_type="gemini", api_key=gemini_api_key, model=model_name,
                         min_request_interval=min_request_interval,
                         adaptive_request_backoff=adaptive_request_backoff,
                         max_request_interval=max_request_interval)
    if llm_provider == "openai":
        return LLMClient(provider_type="openai", api_endpoint=api_endpoint, model=model_name,
                         api_key=openai_api_key, context_window=context_window, log_callback=log_callback,
                         min_request_interval=min_request_interval,
                         adaptive_request_backoff=adaptive_request_backoff,
                         max_request_interval=max_request_interval)
    if llm_provider == "openrouter":
        return LLMClient(provider_type="openrouter", model=model_name, api_key=openrouter_api_key,
                         min_request_interval=min_request_interval,
                         adaptive_request_backoff=adaptive_request_backoff,
                         max_request_interval=max_request_interval)
    if llm_provider == "mistral":
        return LLMClient(provider_type="mistral", model=model_name, api_key=mistral_api_key,
                         min_request_interval=min_request_interval,
                         adaptive_request_backoff=adaptive_request_backoff,
                         max_request_interval=max_request_interval)
    if llm_provider == "deepseek":
        return LLMClient(provider_type="deepseek", model=model_name, api_key=deepseek_api_key,
                         min_request_interval=min_request_interval,
                         adaptive_request_backoff=adaptive_request_backoff,
                         max_request_interval=max_request_interval)
    if llm_provider == "poe":
        return LLMClient(provider_type="poe", model=model_name, api_key=poe_api_key,
                         min_request_interval=min_request_interval,
                         adaptive_request_backoff=adaptive_request_backoff,
                         max_request_interval=max_request_interval)
    if llm_provider == "nim":
        return LLMClient(provider_type="nim", model=model_name, api_key=nim_api_key,
                         min_request_interval=min_request_interval,
                         adaptive_request_backoff=adaptive_request_backoff,
                         max_request_interval=max_request_interval)
    if llm_provider == "fireworks":
        return LLMClient(provider_type="fireworks", model=model_name,
                         api_endpoint=api_endpoint, api_key=fireworks_api_key,
                         min_request_interval=min_request_interval,
                         adaptive_request_backoff=adaptive_request_backoff,
                         max_request_interval=max_request_interval)
    if llm_provider == "ollama":
        # Always create a new client for Ollama to ensure proper configuration
        return LLMClient(provider_type="ollama", api_endpoint=api_endpoint, model=model_name,
                         context_window=context_window, log_callback=log_callback,
                         min_request_interval=min_request_interval,
                         adaptive_request_backoff=adaptive_request_backoff,
                         max_request_interval=max_request_interval)
    return None
