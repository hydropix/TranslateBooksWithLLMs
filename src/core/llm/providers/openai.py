"""
OpenAI-compatible provider.

Works with the official OpenAI API and every OpenAI-compatible endpoint
(llama.cpp, LM Studio, vLLM, NVIDIA NIM, ...). The wire format and request
loop are shared (see ``openai_chat.OpenAIChatProvider``); this module only
adds endpoint normalisation, the local-server thinking switches and context
size detection.
"""

from typing import Any, Callable, Dict, List, Optional, Union

from src.config import OLLAMA_NUM_CTX

from ..base import LLMRequest
from ..policy import RetryPolicy
from ..utils.context_detection import ContextDetector
from .openai_chat import OpenAIChatProvider


class OpenAICompatibleProvider(OpenAIChatProvider):
    """OpenAI-compatible API provider (llama.cpp, LM Studio, vLLM, OpenAI, NIM, ...)."""

    DISPLAY_NAME = ""  # provider_name is used ("openai", "openai-compatible", "nim")
    STATUS_HINTS = {
        404: "unknown model or wrong endpoint path (expected .../v1/chat/completions)",
    }

    def __init__(self, api_endpoint: str, model: str,
                 api_key: Optional[Union[str, List[str]]] = None,
                 context_window: int = OLLAMA_NUM_CTX, log_callback: Optional[Callable] = None,
                 provider_name: str = "openai-compatible",
                 *,
                 retry_policy: Optional[RetryPolicy] = None,
                 temperature: Optional[float] = None):
        # No key pool for keyless local servers (llama.cpp, LM Studio, vLLM).
        super().__init__(
            api_key=api_key, model=model, api_endpoint=self._normalize_endpoint(api_endpoint),
            provider_name=provider_name, context_window=context_window,
            retry_policy=retry_policy, temperature=temperature, log_callback=log_callback,
        )
        self._detected_context_size: Optional[int] = None
        self._context_detector = ContextDetector()

    def _is_official_openai_endpoint(self) -> bool:
        """Check if the endpoint is the official OpenAI API."""
        return "api.openai.com" in (self.api_endpoint or "")

    def _is_local_endpoint(self) -> bool:
        """Check if the endpoint is a local server (llama.cpp, vLLM, LM Studio, etc.)."""
        endpoint = self.api_endpoint or ""
        return "localhost" in endpoint or "127.0.0.1" in endpoint

    @staticmethod
    def _normalize_endpoint(endpoint: str) -> str:
        """
        Normalize API endpoint URL for OpenAI-compatible APIs.

        Adds '/chat/completions' when the URL ends with '/v1' or '/v1/':
        - http://localhost:11434/v1 -> http://localhost:11434/v1/chat/completions
        - https://api.example.com/v1/ -> https://api.example.com/v1/chat/completions
        Any other path is kept as the user wrote it.
        """
        if not endpoint:
            return endpoint
        endpoint = endpoint.rstrip('/')
        if endpoint.endswith('/v1/chat/completions'):
            return endpoint
        if endpoint.endswith('/v1'):
            return endpoint + '/chat/completions'
        return endpoint

    def _response_context_limit(self, request: LLMRequest) -> int:
        return request.context_window or self.context_window or 0

    async def _fixed_params(self, request: LLMRequest) -> Dict[str, Any]:
        # Thinking-disable switches only for local servers (llama.cpp, vLLM,
        # LM Studio); the official API and cloud gateways (NIM) reject them.
        # No top-level "thinking" key: strict servers such as Unsloth Studio
        # validate it as an object and reject a boolean with HTTP 400 (#339).
        if not self._is_official_openai_endpoint() and self._is_local_endpoint():
            return {
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        return {}

    async def get_model_context_size(self) -> int:
        """Query server to get model's context size using ContextDetector."""
        if self._detected_context_size:
            return self._detected_context_size

        client = await self._get_client()
        ctx = await self._context_detector.detect(
            client=client,
            model=self.model,
            endpoint=self.api_endpoint,
            api_key=self.api_key,
            log_callback=self.log_callback
        )

        self._detected_context_size = ctx
        return ctx
