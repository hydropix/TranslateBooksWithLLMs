"""
Opencode LLM Provider.

This module provides the OpencodeProvider class for the Opencode API gateway.
It is a drop-in OpenAI-compatible provider (chat/completions + models listing),
with a single additional requirement: every request must carry an
``x-opencode-session`` header holding a stable identifier for the conversation.

The identifier is generated once when the provider is constructed and reused on
every subsequent call. Because one provider instance is created per translation
job (see GenericTranslator.translate -> LLMClient), the id is stable for the
whole conversation and changes from one job to the next. It is intentionally
NOT derived from the machine, the install, or any user data.
"""

from typing import Callable, List, Optional, Union
import uuid

from src.config import OLLAMA_NUM_CTX, OPENCODE_API_ENDPOINT
from .openai import OpenAICompatibleProvider


class OpencodeProvider(OpenAICompatibleProvider):
    """
    OpenAI-compatible provider that adds the mandatory Opencode session header.

    Example:
        >>> provider = OpencodeProvider(
        ...     api_endpoint="https://opencode.ai/zen/go/v1/chat/completions",
        ...     model="opencode-go/deepseek-v4.1-flash",
        ...     api_key="YOUR_API_KEY_HERE",
        ... )
        >>> response = await provider.generate("Translate: Hello")
    """

    SESSION_HEADER = "x-opencode-session"

    def __init__(
        self,
        model: str = "",
        api_endpoint: Optional[str] = None,
        api_key: Optional[Union[str, List[str]]] = None,
        context_window: int = OLLAMA_NUM_CTX,
        log_callback: Optional[Callable] = None,
        session_id: Optional[str] = None,
        extra_headers: Optional[dict] = None,
    ):
        """
        Initialize the Opencode provider.

        Args:
            model: Model identifier (e.g. "opencode-go/deepseek-v4.1-flash").
            api_endpoint: Optional custom endpoint; defaults to
                OPENCODE_API_ENDPOINT from src.config.
            api_key: Opencode API key (a single key or an iterable for rotation).
            context_window: Context window size in tokens.
            log_callback: Optional logging callback.
            session_id: Optional explicit conversation id. When omitted, a
                random id is generated once and reused for this instance.
            extra_headers: Optional additional headers merged into every call.
        """
        # A "conversation" is the lifetime of this provider instance (one
        # translation job). Reuse a caller-supplied id when one is threaded
        # through; otherwise mint a fresh random one. Never machine-derived.
        self.session_id = session_id or f"opencode-session-{uuid.uuid4()}"
        headers = dict(extra_headers or {})
        headers[self.SESSION_HEADER] = self.session_id

        super().__init__(
            api_endpoint=api_endpoint or OPENCODE_API_ENDPOINT,
            model=model,
            api_key=api_key,
            context_window=context_window,
            log_callback=log_callback,
            provider_name="opencode",
            extra_headers=headers,
        )

    async def get_available_models(self) -> list:
        """
        Fetch available models from the Opencode API.

        Returns:
            List of model dicts with id, name, and (when the API exposes it)
            context_length. Empty list when no key is configured or the API
            call fails.
        """
        if not self.api_key:
            return []

        models_url = self.api_endpoint.replace("/chat/completions", "").rstrip("/") + "/models"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            self.SESSION_HEADER: self.session_id,
        }

        try:
            client = await self._get_client()
            response = await client.get(models_url, headers=headers, timeout=15)
            response.raise_for_status()

            models = []
            for model in response.json().get("data", []):
                model_id = model.get("id", "")
                if not model_id:
                    continue
                entry = {"id": model_id, "name": model.get("name") or model_id}
                context_length = model.get("context_length") or model.get("max_context_length")
                if context_length:
                    entry["context_length"] = context_length
                models.append(entry)

            models.sort(key=lambda m: m["name"].lower())
            return models

        except Exception as e:
            print(f"⚠️ Failed to fetch Opencode models: {e}")
            return []
