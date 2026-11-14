"""
Mistral AI LLM Provider.

This module provides the MistralProvider class for interacting with
Mistral's API, which offers various models with excellent multilingual capabilities.

Features:
    - Access to Mistral models (large, medium, small, codestral)
    - OpenAI-compatible API format
    - Automatic context size detection
    - Rate limiting with exponential backoff
"""

import logging
from typing import List, Optional, Union

from ..policy import RetryPolicy
from .openai_chat import OpenAIChatProvider

logger = logging.getLogger("llm.mistral")


class MistralProvider(OpenAIChatProvider):
    """
    Provider for Mistral AI API.

    Mistral AI provides powerful language models with strong multilingual support:
        - mistral-large-latest: Most capable, 128K context
        - mistral-medium-latest: Balanced performance, 128K context
        - mistral-small-latest: Fast and economical, 32K context
        - codestral-latest: Optimized for code, 32K context
        - ministral-8b-latest: Small efficient model, 128K context

    Configuration:
        endpoint: https://api.mistral.ai/v1/chat/completions
        model: Model identifier (e.g., "mistral-large-latest")
        api_key: Mistral API key

    Example:
        >>> provider = MistralProvider(
        ...     api_key="your-api-key",
        ...     model="mistral-large-latest"
        ... )
        >>> response = await provider.generate("Translate: Hello")
    """

    # Mistral API endpoints
    API_URL = "https://api.mistral.ai/v1/chat/completions"
    MODELS_URL = "https://api.mistral.ai/v1/models"

    # Context sizes by model family
    MODEL_CONTEXT_SIZES = {
        "mistral-large": 128000,
        "mistral-medium": 128000,
        "mistral-small": 32000,
        "codestral": 32000,
        "ministral": 128000,
        "pixtral": 128000,
        "open-mistral": 32000,
        "open-mixtral": 32000,
    }

    DEFAULT_CONTEXT_LIMIT = 32000
    DISPLAY_NAME = "Mistral"
    STATUS_HINTS = {404: "model not found, check https://docs.mistral.ai/"}

    # Fallback models (sorted by capability)
    FALLBACK_MODELS = [
        "mistral-large-latest",
        "mistral-medium-latest",
        "mistral-small-latest",
        "codestral-latest",
        "ministral-8b-latest",
        "ministral-3b-latest",
    ]

    def __init__(
        self,
        api_key: Union[str, List[str]],
        model: str = "mistral-large-latest",
        api_endpoint: Optional[str] = None,
        *,
        retry_policy: Optional[RetryPolicy] = None,
        temperature: Optional[float] = None,
        log_callback=None,
    ):
        """
        Initialize the Mistral provider.

        Args:
            api_key: Mistral API key
            model: Model identifier (default: mistral-large-latest)
            api_endpoint: Optional custom API endpoint
        """
        super().__init__(
            api_key=api_key, model=model, api_endpoint=api_endpoint, provider_name="mistral",
            retry_policy=retry_policy, temperature=temperature, log_callback=log_callback,
        )

    async def get_available_models(self) -> list:
        """
        Fetch available Mistral models from API.

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
                # Include chat/completion models
                if any(x in model_id.lower() for x in ['mistral', 'codestral', 'ministral', 'pixtral']):
                    # Get context length from model info or estimate from ID
                    context_length = model.get("max_context_length")
                    if not context_length:
                        # Estimate based on model family
                        for prefix, size in self.MODEL_CONTEXT_SIZES.items():
                            if prefix in model_id.lower():
                                context_length = size
                                break
                        if not context_length:
                            context_length = 32000

                    filtered_models.append({
                        "id": model_id,
                        "name": model_id,
                        "context_length": context_length
                    })

            # Sort by name
            filtered_models.sort(key=lambda x: x["name"])

            if len(filtered_models) < 1:
                return self._get_fallback_models()

            return filtered_models

        except Exception as e:
            logger.warning(f"Failed to fetch Mistral models: {e}")
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
