"""OpenCode Zen and OpenCode Go OpenAI-compatible providers."""

from typing import Callable, List, Optional, Union
from src.config import OLLAMA_NUM_CTX, REQUEST_TIMEOUT
from .openai import OpenAICompatibleProvider


class _OpenCodeCompatibleProvider(OpenAICompatibleProvider):
    """Shared Chat Completions client for OpenCode Zen and Go."""

    DEFAULT_API_ENDPOINT = ""
    FALLBACK_MODELS: List[str] = []
    PROVIDER_NAME = "opencode"

    # Reasoning models (deepseek-v4-*) routinely spend the completion budget
    # on chain-of-thought and return empty ``content``. Keep a large budget
    # and ask the gateway to disable thinking.
    DEFAULT_MAX_OUTPUT_TOKENS = 32768

    def __init__(
        self,
        api_key: Union[str, List[str]],
        model: str = "",
        api_endpoint: Optional[str] = None,
        context_window: Optional[int] = None,
        log_callback: Optional[Callable] = None,
    ):
        super().__init__(
            api_endpoint or self.DEFAULT_API_ENDPOINT,
            model or self.FALLBACK_MODELS[0],
            api_key=api_key,
            context_window=context_window or OLLAMA_NUM_CTX,
            log_callback=log_callback,
            provider_name=self.PROVIDER_NAME,
        )

    async def generate(self, prompt: str, timeout: int = REQUEST_TIMEOUT,
                      system_prompt: Optional[str] = None,
                      **generation_options):
        generation_options.setdefault("max_tokens", self.DEFAULT_MAX_OUTPUT_TOKENS)
        # DeepSeek V4 (default on Zen/Go) rejects a boolean thinking flag
        # with HTTP 400 "expected struct ThinkingOptions".
        if "deepseek" in (self.model or "").lower():
            generation_options.setdefault("thinking", {"type": "disabled"})
        return await super().generate(
            prompt, timeout=timeout, system_prompt=system_prompt, **generation_options
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
                return [{"id": m.get("id", ""), "name": m.get("id", "")} for m in models if m.get("id")]
        except Exception:
            pass
        return [{"id": m, "name": m} for m in self.FALLBACK_MODELS]


class OpenCodeProvider(_OpenCodeCompatibleProvider):
    """OpenCode Zen pay-as-you-go gateway."""

    DEFAULT_API_ENDPOINT = "https://opencode.ai/zen/v1"
    FALLBACK_MODELS = ["deepseek-v4-flash", "kimi-k3", "glm-5.2", "minimax-m2.7"]
    PROVIDER_NAME = "opencode"


class OpenCodeGoProvider(_OpenCodeCompatibleProvider):
    """OpenCode Go subscription gateway."""

    DEFAULT_API_ENDPOINT = "https://opencode.ai/zen/go/v1"
    FALLBACK_MODELS = ["deepseek-v4-pro", "deepseek-v4-flash", "kimi-k3", "glm-5.2"]
    PROVIDER_NAME = "opencodego"
