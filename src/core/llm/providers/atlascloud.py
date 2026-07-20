"""
Atlas Cloud LLM provider.

Atlas Cloud exposes an OpenAI-compatible LLM API while serving a curated
catalog of third-party text models behind one endpoint.
"""

from typing import List, Optional, Union

from .openai import OpenAICompatibleProvider


class AtlasCloudProvider(OpenAICompatibleProvider):
    """OpenAI-compatible provider for Atlas Cloud."""

    API_URL = "https://api.atlascloud.ai/v1/chat/completions"
    MODELS_URL = "https://api.atlascloud.ai/v1/models"

    FALLBACK_MODELS = [
        "qwen/qwen3.5-flash",
        "deepseek-ai/deepseek-v4-pro",
    ]

    def __init__(
        self,
        api_key: Union[str, List[str]],
        model: str = "qwen/qwen3.5-flash",
        api_endpoint: Optional[str] = None,
    ):
        super().__init__(
            api_endpoint=api_endpoint or self.API_URL,
            model=model,
            api_key=api_key,
            context_window=64000,
            provider_name="atlascloud",
        )

    async def get_available_models(self) -> list:
        """Fetch available Atlas Cloud text models from the OpenAI-compatible catalog."""
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

            models = []
            for model in response.json().get("data", []):
                model_id = model.get("id", "")
                if not model_id:
                    continue
                model_lower = model_id.lower()
                if "embedding" in model_lower or "whisper" in model_lower:
                    continue
                models.append({
                    "id": model_id,
                    "name": model_id,
                    "owned_by": model.get("owned_by", "atlascloud"),
                    "context_length": model.get("context_length") or model.get("max_context_length"),
                })

            models.sort(key=lambda item: item["name"].lower())
            return models or self._get_fallback_models()

        except Exception as exc:
            print(f"Warning: Failed to fetch Atlas Cloud models: {exc}")
            return self._get_fallback_models()

    def _get_fallback_models(self) -> list:
        return [
            {"id": model, "name": model, "owned_by": "atlascloud", "context_length": 64000}
            for model in self.FALLBACK_MODELS
        ]
