"""
OpenRouter provider implementation.

OpenRouter gives unified access to 200+ models through the OpenAI
chat-completions protocol (shared implementation in ``openai_chat``). This
module only configures it:

    - endpoints, attribution headers and the model / preset listings;
    - billed cost read from ``usage.cost`` (flat-rate estimate when absent);
    - reasoning disabled by default (translation-friendly), per the model
      metadata of the catalog; mandatory-reasoning models get their lowest
      effort instead;
    - account presets (@preset/<slug>) listed with the models. Presets hold
      settings TBL has no options for (provider routing, model fallbacks),
      edited on openrouter.ai, so no sampling or reasoning override is sent
      for them.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from src.config import OPENROUTER_DISABLE_THINKING

from ..base import LLMRequest
from ..catalog import CatalogCache
from ..policy import RetryPolicy
from .openai_chat import OpenAIChatProvider

logger = logging.getLogger("llm.openrouter")


class OpenRouterProvider(OpenAIChatProvider):
    """
    Provider for OpenRouter API.

    Configuration:
        endpoint: https://openrouter.ai/api/v1/chat/completions
        model: Model identifier (e.g., "anthropic/claude-3-opus") or "@preset/<slug>"
        api_key: OpenRouter API key (one key or a comma-separated pool)

    Example:
        >>> provider = OpenRouterProvider(
        ...     api_key="sk-or-...",
        ...     model="anthropic/claude-3-opus"
        ... )
        >>> response = await provider.generate("Translate: Hello")
        >>> print(f"Cost: ${provider.get_session_cost()}")
    """

    # OpenRouter API endpoints
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    MODELS_URL = "https://openrouter.ai/api/v1/models"
    PRESETS_URL = "https://openrouter.ai/api/v1/presets"

    DISPLAY_NAME = "OpenRouter"
    STATUS_HINTS = {
        404: "model not found, check https://openrouter.ai/models",
        402: "insufficient OpenRouter credits",
    }

    # Presets are referenced as a model string: "@preset/<slug>"
    PRESET_PREFIX = "@preset/"
    PRESETS_PAGE_SIZE = 100  # API maximum
    PRESETS_MAX_PAGES = 10

    # Rates (USD per 1M tokens) used ONLY when a response carries no usage.cost.
    FALLBACK_PROMPT_RATE_PER_M = 0.50
    FALLBACK_COMPLETION_RATE_PER_M = 1.50

    # Fallback text-only models (sorted by cost, cheapest first)
    FALLBACK_MODELS = [
        # === CHEAP MODELS ===
        "google/gemini-2.0-flash-001",
        "meta-llama/llama-3.3-70b-instruct",
        "qwen/qwen-2.5-72b-instruct",
        "mistralai/mistral-small-24b-instruct-2501",
        # === MID-TIER MODELS ===
        "anthropic/claude-3-5-haiku-20241022",
        "openai/gpt-4o-mini",
        "google/gemini-1.5-pro",
        "deepseek/deepseek-chat",
        # === PREMIUM MODELS ===
        "anthropic/claude-sonnet-4",
        "openai/gpt-4o",
        "anthropic/claude-3-5-sonnet-20241022",
    ]

    # --- Reasoning override ------------------------------------------------
    # OpenRouter exposes one unified `reasoning` request parameter, and
    # /api/v1/models describes each model's reasoning support:
    #   "reasoning": {"mandatory": bool, "default_enabled": bool,
    #                 "supported_efforts": [...], "default_effort": "..."}
    # plus "reasoning" in `supported_parameters`. Many models reason by default
    # (DeepSeek V4.x at effort "high", Qwen 3.x), which multiplies billed output
    # tokens for no gain on translation. Mandatory-reasoning models reject a
    # disable request, so they get their lowest supported effort instead.

    # Effort levels ordered least-thinking first.
    REASONING_EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

    # Model catalog entries, fetched once per process: {model_id: model_dict}.
    # A failed fetch is retried after CATALOG_RETRY_SECONDS. The cache is safe
    # to share across the per-job event loops (see CatalogCache).
    CATALOG_RETRY_SECONDS = 300
    _catalog = CatalogCache("OpenRouter", retry_seconds=CATALOG_RETRY_SECONDS)
    # Resolved override per model: {model_id: reasoning dict, or {} for none}
    _reasoning_overrides: Dict[str, Dict[str, Any]] = {}

    def __init__(
        self,
        api_key: Union[str, List[str]],
        model: str = "anthropic/claude-sonnet-4",
        disable_thinking: bool = OPENROUTER_DISABLE_THINKING,
        *,
        retry_policy: Optional[RetryPolicy] = None,
        temperature: Optional[float] = None,
        log_callback=None,
    ):
        """
        Initialize the OpenRouter provider.

        Args:
            api_key: OpenRouter API key
            model: Model identifier (default: anthropic/claude-sonnet-4)
            disable_thinking: Turn reasoning off (or down to the lowest effort
                on models where it is mandatory). Reasoning tokens are billed
                as output: DeepSeek V4.1 Flash was seen producing 4.7k-31k
                output tokens for ~3k-token translation chunks with it on.
        """
        super().__init__(
            api_key=api_key, model=model, provider_name="openrouter",
            retry_policy=retry_policy, temperature=temperature, log_callback=log_callback,
        )
        self.disable_thinking = disable_thinking
        self._warned_reasoning_tokens = False

    # ------------------------------------------------------- reasoning
    async def _load_model_catalog(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        OpenRouter's model catalog (reasoning metadata included).

        Fetched once per process. Returns None when the catalog is
        unreachable, which means "unknown" and not "no models"; the fetch is
        retried once CATALOG_RETRY_SECONDS have passed.
        """
        async def fetch():
            client = await self._get_client()
            response = await client.get(
                self.MODELS_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30
            )
            response.raise_for_status()
            return {m["id"]: m for m in response.json().get("data", []) if m.get("id")}

        return await type(self)._catalog.get(fetch)

    def _pick_reasoning_override(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Map a model's catalog entry to the `reasoning` value translation wants."""
        if "reasoning" not in (model_info.get("supported_parameters") or []):
            return {}

        reasoning = model_info.get("reasoning") or {}
        if not reasoning.get("mandatory"):
            return {"enabled": False}

        # Mandatory reasoning: disabling is rejected, so ask for the lowest
        # effort the model advertises. Without an effort list, leave its default.
        supported = reasoning.get("supported_efforts") or []
        for effort in self.REASONING_EFFORT_ORDER[1:]:
            if effort in supported:
                return {"effort": effort}
        return {}

    async def _get_reasoning_override(self, model: Optional[str] = None) -> Dict[str, Any]:
        """Return the `reasoning` request value for the model ({} = send none)."""
        model = model or self.model
        if not self.disable_thinking:
            return {}

        # A preset carries its own model and reasoning config; don't override it.
        if model.startswith(self.PRESET_PREFIX):
            return {}

        cached = OpenRouterProvider._reasoning_overrides.get(model)
        if cached is not None:
            return dict(cached)

        catalog = await self._load_model_catalog()
        if catalog is None:
            return {}

        model_info = catalog.get(model)
        if model_info is None:
            logger.warning(f"[OpenRouter] '{model}' is not in the model catalog; "
                           f"reasoning stays at the model default")
            override = {}
        else:
            override = self._pick_reasoning_override(model_info)
            if (model_info.get("reasoning") or {}).get("mandatory"):
                logger.warning(f"[OpenRouter] '{model}' always reasons and cannot "
                               f"turn it off; reasoning tokens are billed as output "
                               f"(requested: {override or 'model default'})")
            elif override:
                logger.info(f"[OpenRouter] {model}: reasoning disabled")

        OpenRouterProvider._reasoning_overrides[model] = override
        return dict(override)

    # ------------------------------------------------- chat-completions hooks
    def _headers(self, request: LLMRequest) -> Dict[str, str]:
        headers = super()._headers(request)
        headers["HTTP-Referer"] = "https://github.com/hydropix/TranslateBookWithLLM"
        headers["X-Title"] = "TranslateBookWithLLM"
        return headers

    def _response_context_limit(self, request: LLMRequest) -> int:
        return 0  # OpenRouter manages context internally

    async def _optional_params(self, request: LLMRequest) -> Dict[str, Any]:
        # A preset carries its own sampling and reasoning settings.
        if request.model.startswith(self.PRESET_PREFIX):
            return {}
        params = await super()._optional_params(request)
        reasoning = await self._get_reasoning_override(request.model)
        if reasoning:
            params["reasoning"] = reasoning
        return params

    @staticmethod
    def _extract_cost(usage: Dict[str, Any]) -> Optional[float]:
        """Return the exact billed cost (USD) from ``usage.cost``, or None if absent."""
        raw = usage.get("cost") if isinstance(usage, dict) else None
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def _parse_cost(self, usage: Dict[str, Any], prompt_tokens: int,
                    completion_tokens: int) -> Tuple[Optional[float], bool]:
        cost = self._extract_cost(usage)
        if cost is not None:
            return cost, False
        # usage.cost is returned automatically (the old usage.include request
        # flag is deprecated); fall back to flat rates if it is absent.
        return (
            prompt_tokens * self.FALLBACK_PROMPT_RATE_PER_M / 1_000_000
            + completion_tokens * self.FALLBACK_COMPLETION_RATE_PER_M / 1_000_000
        ), True

    def _inspect_usage(self, usage: Dict[str, Any], request: LLMRequest) -> None:
        details = usage.get("completion_tokens_details") or {}
        reasoning_tokens = details.get("reasoning_tokens") or 0
        sent = (request.state.get("optional") or {}).get("reasoning") or {}
        if reasoning_tokens and sent.get("enabled") is False and not self._warned_reasoning_tokens:
            self._warned_reasoning_tokens = True
            logger.warning(f"[OpenRouter] '{request.model}' still produced {reasoning_tokens} "
                           f"reasoning tokens with reasoning disabled; the upstream provider "
                           f"may ignore the setting")

    # ---------------------------------------------------------- listings
    async def get_available_models(self, text_only: bool = True) -> list:
        """
        Fetch available OpenRouter models from API.

        Args:
            text_only: If True, filter out vision/multimodal models (default: True)

        Returns:
            List of model dicts with id, name, pricing info, sorted by price
        """
        if not self.api_key:
            return self._get_fallback_models()

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
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
                architecture = model.get("architecture", {})
                modality = architecture.get("modality", "")

                if text_only:
                    if modality == "multimodal":
                        continue
                    model_id_lower = model_id.lower()
                    vision_keywords = ["vision", "vl", "-v-", "image"]
                    if any(kw in model_id_lower for kw in vision_keywords):
                        continue

                pricing = model.get("pricing", {})
                prompt_price = float(pricing.get("prompt", "0") or "0")
                completion_price = float(pricing.get("completion", "0") or "0")

                prompt_per_million = prompt_price * 1_000_000
                completion_per_million = completion_price * 1_000_000

                is_free = ":free" in model_id
                display_name = model.get("name", model_id)
                if is_free and "(free" not in display_name.lower():
                    display_name = f"{display_name} (free, 20 req/min)"

                filtered_models.append({
                    "id": model_id,
                    "name": display_name,
                    "context_length": model.get("context_length", 0),
                    "pricing": {
                        "prompt": prompt_price,
                        "completion": completion_price,
                        "prompt_per_million": prompt_per_million,
                        "completion_per_million": completion_per_million,
                    },
                    "total_price": prompt_price + completion_price,
                    "is_free": is_free,
                })

            # Sort: paid models by price ascending, free models last (shared 20 req/min limit)
            filtered_models.sort(key=lambda x: (x["is_free"], x["total_price"]))

            if len(filtered_models) < 5:
                return self._get_fallback_models()

            # The key's presets go first: they are the user's own configurations
            return await self.get_presets() + filtered_models

        except Exception as e:
            logger.warning(f"[OpenRouter] Failed to fetch models: {e}")
            return self._get_fallback_models()

    async def get_presets(self) -> list:
        """
        Fetch the active presets visible to this API key.

        Each preset is returned in the same shape as a model entry, with
        id "@preset/<slug>", so it can be selected and sent as the model.
        Failures are non-fatal: the model list is still usable without presets.

        Returns:
            List of preset dicts sorted by id (empty on error or when none)
        """
        if not self.api_key:
            return []

        presets = []
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            client = await self._get_client()

            for page in range(self.PRESETS_MAX_PAGES):
                response = await client.get(
                    self.PRESETS_URL,
                    headers=headers,
                    params={"limit": self.PRESETS_PAGE_SIZE,
                            "offset": page * self.PRESETS_PAGE_SIZE},
                    timeout=15
                )
                response.raise_for_status()
                body = response.json()
                page_data = body.get("data", [])

                for preset in page_data:
                    slug = preset.get("slug")
                    if not slug or preset.get("status", "active") != "active":
                        continue
                    preset_id = f"{self.PRESET_PREFIX}{slug}"
                    presets.append({
                        "id": preset_id,
                        "name": preset_id,
                        "description": preset.get("description") or "",
                        "is_preset": True,
                    })

                total = body.get("total_count", 0)
                if len(page_data) < self.PRESETS_PAGE_SIZE or \
                        (page + 1) * self.PRESETS_PAGE_SIZE >= total:
                    break

        except Exception as e:
            logger.warning(f"[OpenRouter] Failed to fetch presets: {e}")

        presets.sort(key=lambda p: p["id"])
        return presets

    def _get_fallback_models(self) -> list:
        """Return fallback models list when API fetch fails."""
        return [{"id": m, "name": m, "pricing": {"prompt": 0, "completion": 0}}
                for m in self.FALLBACK_MODELS]
