"""xAI Grok provider using the supported OpenAI-compatible chat contract."""

from typing import Callable, List, Optional, Union
from src.config import OLLAMA_NUM_CTX
from .openai import OpenAICompatibleProvider


class XAIProvider(OpenAICompatibleProvider):
    DEFAULT_API_ENDPOINT = "https://api.x.ai/v1"
    FALLBACK_MODELS = ["grok-4.5", "grok-4", "grok-3-mini"]

    def __init__(
        self,
        api_key: Union[str, List[str]],
        model: str = FALLBACK_MODELS[0],
        api_endpoint: Optional[str] = None,
        context_window: Optional[int] = None,
        log_callback: Optional[Callable] = None,
    ):
        super().__init__(
            api_endpoint or self.DEFAULT_API_ENDPOINT,
            model,
            api_key=api_key,
            context_window=context_window or OLLAMA_NUM_CTX,
            log_callback=log_callback,
            provider_name="xai",
        )

    async def get_available_models(self) -> list:
        return await self._get_compatible_models(self.api_endpoint.rsplit("/chat/completions", 1)[0] + "/models")

    async def _get_compatible_models(self, url: str) -> list:
        try:
            response = await (await self._get_client()).get(url, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=15)
            response.raise_for_status()
            return [{"id": m.get("id", ""), "name": m.get("id", ""), "context_length": m.get("context_length")} for m in response.json().get("data", []) if m.get("id")]
        except Exception:
            return [{"id": m, "name": m} for m in self.FALLBACK_MODELS]
