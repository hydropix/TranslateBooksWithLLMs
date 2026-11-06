"""
Opencode LLM Provider.

This module provides the OpencodeProvider class for the OpenCode gateway
(https://opencode.ai). One OpenCode API key reaches two catalogs:

- OpenCode Go (``https://opencode.ai/zen/go/v1``): the subscription plan,
  billed against the workspace's Go usage limits.
- OpenCode Zen (``https://opencode.ai/zen/v1``): pay-as-you-go, billed against
  the workspace balance.

Model ids follow opencode's own ``<catalog>/<model>`` convention, as printed
by ``opencode models``: ``opencode-go/deepseek-v4.1-flash`` routes to Go and
``opencode/deepseek-v4.1-flash`` routes to Zen. A bare id routes to Go, so a
typo never spends balance. The API itself only ever receives the bare id.

Every request carries an ``x-opencode-session`` header holding a stable
identifier for the conversation; Go rejects requests without it
(``MissingSessionID``). The identifier is generated once when the provider is
constructed and reused on every subsequent call. Because one provider instance
is created per translation job (see GenericTranslator.translate -> LLMClient),
every chunk of a job shares it, which keeps the gateway's prompt cache warm.
It is intentionally NOT derived from the machine, the install, or any user
data.

The gateway serves each model through one API family: chat/completions,
OpenAI Responses, Anthropic Messages, or Google generateContent. Requests are
routed by model-id prefix, following opencode's own client and hermes-agent;
any other model (including models that rotate in and out of the catalogs)
goes to chat/completions. Only the System One API (``jev-`` on Zen) is not
implemented: those models are rejected up front and hidden from the model
list. Gateway refusals are turned into actionable errors.
"""

from typing import Callable, List, Optional, Tuple, Union
import asyncio
import uuid

import httpx

from src.config import OLLAMA_NUM_CTX, OPENCODE_API_BASE
from .openai import OpenAICompatibleProvider


class OpencodeProvider(OpenAICompatibleProvider):
    """
    OpenAI-compatible provider for the OpenCode Go and Zen catalogs.

    Example:
        >>> provider = OpencodeProvider(
        ...     model="opencode-go/deepseek-v4.1-flash",
        ...     api_key="YOUR_API_KEY_HERE",
        ... )
        >>> response = await provider.generate("Translate: Hello")
    """

    SESSION_HEADER = "x-opencode-session"

    GO = "opencode-go"
    ZEN = "opencode"
    DEFAULT_CATALOG = GO

    # Catalog -> (path under OPENCODE_API_BASE, label shown in the model list).
    # Go is listed first: it is the subscription, Zen spends balance.
    CATALOGS = {
        GO: ("/go/v1", "OpenCode Go"),
        ZEN: ("/v1", "OpenCode Zen"),
    }

    CHAT = "chat"
    RESPONSES = "responses"
    MESSAGES = "messages"
    GOOGLE = "google"
    UNSUPPORTED = "unsupported"

    # Model-id prefix -> API family, per catalog; first match wins and anything
    # else is chat/completions. From the endpoint tables at
    # https://opencode.ai/docs/go/ and the Zen docs, matching the routing in
    # opencode and hermes-agent. The gateway answers ModelProtocolUnsupported
    # when a model is sent to the wrong family. "jev-" is the System One API,
    # which is not implemented.
    API_FAMILY_PREFIXES = {
        GO: (
            (("gpt-", "grok-", "muse-spark"), RESPONSES),
            (("claude-", "minimax-", "qwen", "union-alpha"), MESSAGES),
        ),
        ZEN: (
            (("gpt-", "grok-", "muse-spark"), RESPONSES),
            (("claude-", "qwen", "union-alpha"), MESSAGES),
            (("gemini-",), GOOGLE),
            (("jev-",), UNSUPPORTED),
        ),
    }

    # Anthropic Messages requires max_tokens; generous enough for any chunk.
    MESSAGES_MAX_TOKENS = 16384
    ANTHROPIC_VERSION = "2023-06-01"

    @classmethod
    def split_model_id(cls, model: Optional[str]) -> Tuple[str, str]:
        """Return (catalog, bare model id); a bare id belongs to Go."""
        model = (model or "").strip()
        # Check "opencode-go/" before "opencode/": neither is a prefix of the
        # other, but keep the longer one first should that ever change.
        for catalog in sorted(cls.CATALOGS, key=len, reverse=True):
            prefix = f"{catalog}/"
            if model.lower().startswith(prefix):
                return catalog, model[len(prefix):]
        return cls.DEFAULT_CATALOG, model

    @classmethod
    def canonical_model_id(cls, model: Optional[str]) -> str:
        """The "<catalog>/<model>" form used in the model list ("" stays "")."""
        catalog, bare = cls.split_model_id(model)
        return f"{catalog}/{bare}" if bare else ""

    @classmethod
    def api_family(cls, model: Optional[str]) -> str:
        """The API family the gateway serves this model through."""
        catalog, bare = cls.split_model_id(model)
        bare = bare.lower()
        for prefixes, family in cls.API_FAMILY_PREFIXES[catalog]:
            if bare.startswith(prefixes):
                return family
        return cls.CHAT

    @classmethod
    def is_supported(cls, model: Optional[str]) -> bool:
        """False for models served only through an API this provider lacks."""
        return cls.api_family(model) != cls.UNSUPPORTED

    def __init__(
        self,
        model: str = "",
        api_base: Optional[str] = None,
        api_key: Optional[Union[str, List[str]]] = None,
        context_window: int = OLLAMA_NUM_CTX,
        log_callback: Optional[Callable] = None,
        session_id: Optional[str] = None,
        extra_headers: Optional[dict] = None,
    ):
        """
        Initialize the Opencode provider.

        Args:
            model: Model identifier, "opencode-go/<id>" (Go) or "opencode/<id>"
                (Zen). A bare "<id>" is routed to Go.
            api_base: Optional gateway root; defaults to OPENCODE_API_BASE from
                src.config. Catalog paths are appended to it.
            api_key: Opencode API key (a single key or an iterable for rotation).
            context_window: Context window size in tokens.
            log_callback: Optional logging callback.
            session_id: Optional explicit conversation id. When omitted, a
                random id is generated once and reused for this instance.
            extra_headers: Optional additional headers merged into every call.
        """
        self.api_base = (api_base or OPENCODE_API_BASE).rstrip("/")
        # Validates the model and picks its catalog (see the model setter).
        self.model = model

        # A "conversation" is the lifetime of this provider instance (one
        # translation job). Reuse a caller-supplied id when one is threaded
        # through; otherwise mint a fresh random one. Never machine-derived.
        self.session_id = session_id or f"opencode-session-{uuid.uuid4()}"
        headers = dict(extra_headers or {})
        headers[self.SESSION_HEADER] = self.session_id

        super().__init__(
            api_endpoint=self._chat_url(),
            model=model,
            api_key=api_key,
            context_window=context_window,
            log_callback=log_callback,
            provider_name="opencode",
            extra_headers=headers,
        )

    @property
    def model(self) -> str:
        """The bare model id sent to the API."""
        return self._model

    @model.setter
    def model(self, model: Optional[str]) -> None:
        # Callers (e.g. LLMClient.make_request during refinement) assign the
        # "<catalog>/<model>" id from settings; strip it here so the API only
        # ever receives the bare id and the request goes to the right catalog.
        catalog, bare_model = self.split_model_id(model)
        if bare_model and not self.is_supported(model):
            raise ValueError(
                f"Opencode model '{self.canonical_model_id(model)}' is only served "
                "through the System One API, which this provider does not "
                "support yet. Choose another model."
            )
        self.catalog = catalog
        self.api_family_name = self.api_family(model)
        self._model = bare_model
        # chat/completions URL: used for logs and context detection; the
        # request itself goes to _request_url() for the model's family.
        self.api_endpoint = self._chat_url()

    def _chat_url(self) -> str:
        return self._catalog_url(self.catalog) + "/chat/completions"

    def _request_url(self) -> str:
        base = self._catalog_url(self.catalog)
        if self.api_family_name == self.RESPONSES:
            return base + "/responses"
        if self.api_family_name == self.MESSAGES:
            return base + "/messages"
        if self.api_family_name == self.GOOGLE:
            return f"{base}/models/{self.model}:generateContent"
        return base + "/chat/completions"

    def _auth_headers(self, api_key: str) -> dict:
        # The gateway reads the key from each family's native header.
        if self.api_family_name == self.MESSAGES:
            return {"x-api-key": api_key, "anthropic-version": self.ANTHROPIC_VERSION}
        if self.api_family_name == self.GOOGLE:
            return {"x-goog-api-key": api_key}
        return super()._auth_headers(api_key)

    def _build_payload(self, prompt: str, system_prompt: Optional[str]) -> dict:
        if self.api_family_name == self.RESPONSES:
            payload = {"model": self.model, "input": prompt, "store": False}
            if system_prompt:
                payload["instructions"] = system_prompt
            return payload
        if self.api_family_name == self.MESSAGES:
            payload = {
                "model": self.model,
                "max_tokens": self.MESSAGES_MAX_TOKENS,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system_prompt:
                payload["system"] = system_prompt
            return payload
        if self.api_family_name == self.GOOGLE:
            payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
            if system_prompt:
                payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
            return payload
        return super()._build_payload(prompt, system_prompt)

    def _parse_response(self, response_json: dict) -> Tuple[str, Optional[str], int, int]:
        if self.api_family_name == self.RESPONSES:
            text = "".join(
                part.get("text", "")
                for item in response_json.get("output", [])
                if item.get("type") == "message"
                for part in item.get("content", [])
                if part.get("type") == "output_text"
            )
            finish_reason = (response_json.get("incomplete_details") or {}).get("reason")
            if finish_reason is None and response_json.get("status") == "completed":
                finish_reason = "stop"
            usage = response_json.get("usage") or {}
            return text, finish_reason, usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        if self.api_family_name == self.MESSAGES:
            text = "".join(
                block.get("text", "")
                for block in response_json.get("content", [])
                if block.get("type") == "text"
            )
            usage = response_json.get("usage") or {}
            return (text, response_json.get("stop_reason"),
                    usage.get("input_tokens", 0), usage.get("output_tokens", 0))
        if self.api_family_name == self.GOOGLE:
            candidate = (response_json.get("candidates") or [{}])[0]
            text = "".join(
                part.get("text", "")
                for part in (candidate.get("content") or {}).get("parts", [])
                if not part.get("thought")
            )
            usage = response_json.get("usageMetadata") or {}
            return (text, candidate.get("finishReason"),
                    usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0))
        return super()._parse_response(response_json)

    def _catalog_url(self, catalog: str) -> str:
        return self.api_base + self.CATALOGS[catalog][0]

    async def get_available_models(self) -> list:
        """
        Fetch the chat-capable models of both catalogs.

        Returns:
            List of model dicts with a "<catalog>/<model>" id, name, group
            (catalog label) and, when the API exposes it, context_length. Go
            models come first. A catalog that fails to load is skipped; the
            list is empty when no key is configured.

        Raises:
            RuntimeError: Both catalogs failed; the message names each cause.
        """
        if not self.api_key:
            return []

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            self.SESSION_HEADER: self.session_id,
        }
        client = await self._get_client()

        async def fetch(catalog: str) -> list:
            response = await client.get(
                self._catalog_url(catalog) + "/models", headers=headers, timeout=15
            )
            response.raise_for_status()
            return response.json().get("data", [])

        results = await asyncio.gather(
            *(fetch(catalog) for catalog in self.CATALOGS), return_exceptions=True
        )

        models = []
        errors = []
        for (catalog, (_, label)), data in zip(self.CATALOGS.items(), results):
            if isinstance(data, Exception):
                print(f"⚠️ Failed to fetch {label} models: {data}")
                errors.append(f"{label}: {data}")
                continue

            entries = []
            for model in data:
                bare_id = model.get("id", "")
                model_id = f"{catalog}/{bare_id}"
                if not bare_id or not self.is_supported(model_id):
                    continue
                entry = {"id": model_id, "name": model.get("name") or model_id, "group": label}
                context_length = model.get("context_length") or model.get("max_context_length")
                if context_length:
                    entry["context_length"] = context_length
                entries.append(entry)
            entries.sort(key=lambda m: m["id"].lower())
            models.extend(entries)

        # Both catalogs failed: surface the cause (e.g. 401 invalid key)
        # instead of an empty list the caller can only report generically.
        if not models and len(errors) == len(self.CATALOGS):
            raise RuntimeError("; ".join(errors))

        return models

    # Gateway error types (error.type in the JSON body) -> what to tell the user.
    _ERROR_HINTS = {
        "ModelProtocolUnsupported": (
            "OpenCode serves '{model}' through a different API than the one "
            "this provider picked for it. Choose another model, or report the "
            "model id so its routing can be added."
        ),
        "FreeTierError": (
            "OpenCode currently restricts its free Zen models to its own app. "
            "Choose an opencode-go/ model or a paid opencode/ model."
        ),
        "CreditsError": (
            "opencode/ (Zen) models are paid from your OpenCode workspace "
            "balance. Add balance in the OpenCode console or choose an "
            "opencode-go/ model."
        ),
    }

    def _describe_http_error(self, response: Optional[httpx.Response], error_message: str) -> str:
        """Explain gateway refusals that would otherwise read as opaque 4xx errors."""
        try:
            error_type = response.json().get("error", {}).get("type")
        except Exception:
            # Plain-text/HTML bodies still get the substring check below.
            error_type = None
        hint = self._ERROR_HINTS.get(error_type)
        lowered = error_message.lower()
        if hint is None and "model access is disabled" in lowered:
            hint = (
                "'{model}' is not enabled for your OpenCode workspace. Check the "
                "model and billing settings in the OpenCode console."
            )
        # Messages and Google bodies carry no CreditsError type, only this text.
        if hint is None and "insufficient account funds" in lowered:
            hint = self._ERROR_HINTS["CreditsError"]
        if hint is None:
            return error_message
        model = f"{self.catalog}/{self.model}"
        return f"{error_message} {hint.format(model=model)}"
