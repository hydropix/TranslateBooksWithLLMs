"""
LLMTR provider.

LLMTR is a prepaid OpenAI-compatible gateway. Chat Completions and the
public model catalog both live under https://llmtr.com/v1. Model IDs use
the vendor/model form, for example qwen/qwen3.8-27b-free.
"""

import logging
from typing import List, Optional, Union

from ..policy import RetryPolicy
from .openai_chat import OpenAIChatProvider

logger = logging.getLogger("llm.llmtr")


class LlmtrProvider(OpenAIChatProvider):
    """Provider for the LLMTR OpenAI-compatible API."""

    API_URL = "https://llmtr.com/v1/chat/completions"
    MODELS_URL = "https://llmtr.com/v1/models"

    DEFAULT_CONTEXT_LIMIT = 0
    DISPLAY_NAME = "LLMTR"
    STATUS_HINTS = {404: "model not found, check https://llmtr.com"}

    FALLBACK_MODELS = [
        "qwen/qwen3.8-27b-free",
    ]

    def __init__(
        self,
        api_key: Union[str, List[str]],
        model: str = "qwen/qwen3.8-27b-free",
        api_endpoint: Optional[str] = None,
        *,
        retry_policy: Optional[RetryPolicy] = None,
        temperature: Optional[float] = None,
        log_callback=None,
    ):
        """
        Initialize the LLMTR provider.

        Args:
            api_key: LLMTR API key
            model: Model identifier (vendor/model form)
            api_endpoint: Optional custom API endpoint
        """
        super().__init__(
            api_key=api_key, model=model, api_endpoint=api_endpoint, provider_name="llmtr",
            retry_policy=retry_policy, temperature=temperature, log_callback=log_callback,
        )

    async def get_available_models(self) -> list:
        """Fetch available LLMTR models from the public catalog."""
        if not self.api_key:
            return self._get_fallback_models()

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            }
            client = await self._get_client()
            response = await client.get(self.MODELS_URL, headers=headers, timeout=15)
            response.raise_for_status()

            models_data = response.json().get("data", [])
            filtered_models = []
            for model in models_data:
                model_id = model.get("id", "")
                if not model_id:
                    continue
                context_length = (
                    model.get("context_length")
                    or model.get("max_context_length")
                    or model.get("context_window")
                    or self.DEFAULT_CONTEXT_LIMIT
                )
                filtered_models.append({
                    "id": model_id,
                    "name": model.get("name") or model_id,
                    "context_length": context_length,
                })

            filtered_models.sort(key=lambda x: x["name"].lower())
            if not filtered_models:
                return self._get_fallback_models()
            return filtered_models

        except Exception as e:
            logger.warning(f"Failed to fetch LLMTR models: {e}")
            return self._get_fallback_models()

    def _get_fallback_models(self) -> list:
        """Return fallback models list when API fetch fails."""
        return [
            {
                "id": m,
                "name": m,
                "context_length": self._get_context_limit_for_model(m),
            }
            for m in self.FALLBACK_MODELS
        ]

    def _get_context_limit_for_model(self, model_name: str) -> int:
        """Get context limit for a specific model name."""
        return self._context_limit_for(model_name)
