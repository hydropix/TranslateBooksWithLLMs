"""Ollama Cloud provider — hosted models at ollama.com, OpenAI-compatible."""

from typing import Callable, List, Optional, Union

from src.config import OLLAMA_NUM_CTX
from .openai import OpenAICompatibleProvider


class OllamaCloudProvider(OpenAICompatibleProvider):
    """Direct Ollama Cloud client (no local Ollama daemon).

    Auth is a Bearer API key from ollama.com/settings/keys. Chat uses
    ``https://ollama.com/v1/chat/completions``; models are listed from
    ``GET /v1/models``.
    """

    DEFAULT_API_ENDPOINT = "https://ollama.com/v1"
    FALLBACK_MODELS = [
        "gpt-oss:120b",
        "kimi-k2.6",
        "glm-5.1",
        "deepseek-v4-flash",
        "minimax-m2.7",
    ]

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
            provider_name="ollamacloud",
        )

    async def get_available_models(self) -> list:
        try:
            base = self.api_endpoint.rsplit("/chat/completions", 1)[0]
            response = await (await self._get_client()).get(
                f"{base}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=15,
            )
            response.raise_for_status()
            models = response.json().get("data", [])
            if models:
                return [
                    {
                        "id": m.get("id", ""),
                        "name": m.get("id", ""),
                        "context_length": m.get("context_length"),
                    }
                    for m in models
                    if m.get("id")
                ]
        except Exception:
            pass
        return [{"id": m, "name": m} for m in self.FALLBACK_MODELS]
