"""
DeepSeek LLM Provider.

This module provides the DeepSeekProvider class for interacting with
DeepSeek's API, which offers cost-effective models with strong capabilities.

Features:
    - DeepSeek V3 (chat) and V4 (flash/pro) models
    - OpenAI-compatible API format
    - Cost-effective pricing (~5-10x cheaper than OpenAI)
    - 64K context window
    - Auto-disables V4 reasoning by default (translation-friendly)
"""

import logging
from typing import Any, Dict, List, Optional, Union

from ..base import LLMRequest
from ..policy import RetryPolicy
from .openai_chat import OpenAIChatProvider

logger = logging.getLogger("llm.deepseek")


class DeepSeekProvider(OpenAIChatProvider):
    """
    Provider for DeepSeek API.

    DeepSeek provides powerful language models with excellent price/performance:
        - deepseek-v4-pro: Recommended high-quality model for translation
        - deepseek-v4-flash: Faster economical model

    Configuration:
        endpoint: https://api.deepseek.com/chat/completions
        model: Model identifier (e.g., "deepseek-v4-pro")
        api_key: DeepSeek API key

    Example:
        >>> provider = DeepSeekProvider(
        ...     api_key="your-api-key",
        ...     model="deepseek-v4-pro"
        ... )
        >>> response = await provider.generate("Translate: Hello")
    """

    API_URL = "https://api.deepseek.com/chat/completions"
    MODELS_URL = "https://api.deepseek.com/models"

    MODEL_CONTEXT_SIZES = {
        "deepseek-v4-pro": 64000,
        "deepseek-v4-flash": 64000,
        "deepseek-chat": 64000,
        "deepseek-reasoner": 64000,
        "deepseek-coder": 16000,
        "deepseek-v4": 64000,
    }

    FALLBACK_MODELS = [
        "deepseek-v4-pro",
        "deepseek-v4-flash",
    ]

    DEFAULT_CONTEXT_LIMIT = 64000
    DISPLAY_NAME = "DeepSeek"
    STATUS_HINTS = {404: "model not found, check https://platform.deepseek.com/"}

    THINKING_MODELS = ["deepseek-reasoner", "deepseek-r1"]
    THINKING_BY_DEFAULT_MODELS = ["deepseek-v4"]

    def __init__(
        self,
        api_key: Union[str, List[str]],
        model: str = "deepseek-v4-pro",
        api_endpoint: Optional[str] = None,
        disable_thinking: bool = True,
        *,
        retry_policy: Optional[RetryPolicy] = None,
        temperature: Optional[float] = None,
        log_callback=None,
    ):
        """
        Initialize the DeepSeek provider.

        Args:
            api_key: DeepSeek API key
            model: Model identifier (default: deepseek-v4-pro)
            api_endpoint: Optional custom API endpoint
            disable_thinking: For models that think by default (V4 family),
                inject ``thinking={"type":"disabled"}`` to skip reasoning tokens.
        """
        super().__init__(
            api_key=api_key, model=model, api_endpoint=api_endpoint, provider_name="deepseek",
            retry_policy=retry_policy, temperature=temperature, log_callback=log_callback,
        )
        self.disable_thinking = disable_thinking

    def _is_thinking_model(self) -> bool:
        """Check if the current model uses thinking mode."""
        return any(tm in self.model.lower() for tm in self.THINKING_MODELS)

    def _thinking_enabled_by_default(self) -> bool:
        """True for models (V4 family) that think unless `thinking.type=disabled` is sent."""
        model_lower = self.model.lower()
        return any(tm in model_lower for tm in self.THINKING_BY_DEFAULT_MODELS)

    async def get_available_models(self) -> list:
        """
        Fetch available DeepSeek models from API.

        Returns:
            List of model dicts with id, name, and context_length
        """
        if not self.api_key:
            return self._get_fallback_models()

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json"
            }
            client = await self._get_client()

            response = await client.get(
                self.MODELS_URL,
                headers=headers,
                timeout=15
            )
            response.raise_for_status()

            models_data = response.json().get("data", [])
            filtered_models = []

            for model in models_data:
                model_id = model.get("id", "")
                # Skip deepseek-reasoner: always thinks, no toggle. V4 models
                # are kept (they think by default but we override that).
                if "reasoner" in model_id.lower():
                    continue
                if "deepseek" in model_id.lower():
                    context_length = model.get("max_context_length")
                    if not context_length:
                        for prefix, size in self.MODEL_CONTEXT_SIZES.items():
                            if prefix in model_id.lower():
                                context_length = size
                                break
                        if not context_length:
                            context_length = 64000

                    filtered_models.append({
                        "id": model_id,
                        "name": model_id,
                        "context_length": context_length
                    })

            filtered_models.sort(key=lambda x: x["name"])

            if len(filtered_models) < 1:
                return self._get_fallback_models()

            return filtered_models

        except Exception as e:
            logger.warning(f"Failed to fetch DeepSeek models: {e}")
            return self._get_fallback_models()

    def _get_fallback_models(self) -> list:
        """Return fallback models list when API fetch fails."""
        return [
            {
                "id": m,
                "name": m,
                "context_length": self._get_context_limit_for_model(m)
            }
            for m in self.FALLBACK_MODELS
        ]

    def _get_context_limit_for_model(self, model_name: str) -> int:
        """Get context limit for a specific model name."""
        return self._context_limit_for(model_name)

    async def _fixed_params(self, request: LLMRequest) -> Dict[str, Any]:
        # V4 models think unless told not to; reasoning tokens are billed.
        model_lower = request.model.lower()
        if self.disable_thinking and any(tm in model_lower for tm in self.THINKING_BY_DEFAULT_MODELS):
            return {"thinking": {"type": "disabled"}}
        return {}
