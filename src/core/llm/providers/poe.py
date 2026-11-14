"""
POE (Poe.com) LLM Provider.

This module provides the PoeProvider class for interacting with
Poe's OpenAI-compatible API, which provides access to hundreds of models.

Features:
    - Access to Claude, GPT, Gemini, Llama, Grok, and community bots
    - OpenAI-compatible API format
    - 500 requests/minute rate limit
    - Subscription-based pricing (uses Poe points)

API Documentation: https://creator.poe.com/docs/external-applications/openai-compatible-api
"""

import logging
from typing import Any, Dict, Iterable, List, Optional, Union

from src.config import POE_DISABLE_THINKING, POE_DISABLE_WEB_SEARCH

from ..base import LLMRequest
from ..catalog import CatalogCache
from ..policy import RetryPolicy
from .openai_chat import OpenAIChatProvider

logger = logging.getLogger("llm.poe")


class PoeProvider(OpenAIChatProvider):
    """
    Provider for Poe API (OpenAI-compatible).

    Poe provides unified access to multiple LLM providers including:
        - Anthropic (Claude Opus, Sonnet)
        - OpenAI (GPT-4)
        - Google (Gemini)
        - Meta (Llama)
        - xAI (Grok)
        - Millions of community-created bots

    Configuration:
        endpoint: https://api.poe.com/v1/chat/completions
        model: Bot name on Poe (e.g., "Claude-Sonnet-4", "GPT-4o")
        api_key: Poe API key (get from https://poe.com/api_key)

    Example:
        >>> provider = PoeProvider(
        ...     api_key="your-poe-api-key",
        ...     model="Claude-Sonnet-4"
        ... )
        >>> response = await provider.generate("Translate: Hello")
    """

    # Poe API endpoints
    API_URL = "https://api.poe.com/v1/chat/completions"
    MODELS_URL = "https://api.poe.com/v1/models"

    # Model context sizes (approximate, based on underlying models)
    MODEL_CONTEXT_SIZES = {
        "claude": 100000,
        "gpt-4": 128000,
        "gpt-4o": 128000,
        "gemini": 128000,
        "llama": 128000,
        "grok": 131072,
    }

    # Fallback models (popular Poe bots for translation)
    FALLBACK_MODELS = [
        "Claude-Sonnet-4",
        "Claude-Opus-4.1",
        "GPT-4o",
        "Gemini-2.5-Pro",
        "Llama-3.1-405B",
        "Grok-4",
    ]

    # --- Bot default overrides ---------------------------------------------
    # Poe has no universal switch for reasoning or web search: each bot
    # advertises its own knobs in the `parameters` array of /v1/models, and the
    # API REJECTS any knob a bot does not advertise ("API Bots do not support
    # extra_body fields: ...", or "Unrecognized request argument supplied: ...",
    # both HTTP 400). So the overrides we send are derived from the bot's own
    # schema instead of a hardcoded model list.

    # Reasoning enum knobs, ordered least-thinking first: first match wins.
    REASONING_ENUM_PREFERENCES = {
        "thinking_level": ("minimal", "low"),
        "reasoning_effort": ("none", "minimal", "low"),
    }
    # Reasoning numeric knobs: send the schema minimum (0 = no thinking budget).
    REASONING_BUDGET_PARAMS = ("thinking_budget",)
    # Reasoning boolean knobs: off.
    REASONING_TOGGLE_PARAMS = ("enable_thinking",)
    # `output_effort` (Claude 4.6/4.8) is deliberately left alone: it caps the
    # whole answer, not just hidden reasoning, so forcing it down would degrade
    # the translation itself.

    # Retrieval knobs, off: a book is self-contained, so search results are
    # paid prompt tokens at best and context pollution at worst. Default is on
    # for gemini-3.6-flash and grok-4.5, where it inflates every single chunk.
    WEB_SEARCH_PARAMS = ("web_search", "enable_web_search")

    # Advertised parameter schemas, fetched once per process: {model_id: [param]}.
    # A failed fetch is retried after CATALOG_RETRY_SECONDS. The cache is safe
    # to share across the per-job event loops (see CatalogCache).
    CATALOG_RETRY_SECONDS = 300
    _catalog = CatalogCache("Poe", retry_seconds=CATALOG_RETRY_SECONDS)
    # Resolved overrides, keyed by model *and* flags since they decide the
    # result: {(model_id, disable_thinking, disable_web_search): {param: value}}
    _bot_overrides: Dict[tuple, Dict[str, Any]] = {}

    DEFAULT_CONTEXT_LIMIT = 32000  # Conservative default for unknown models
    DISPLAY_NAME = "Poe"
    # Poe's 400s for an unsupported knob do not always name it: when our own
    # overrides are in the request, retry once without them to rule them out.
    DROP_ALL_OPTIONAL_ON_UNNAMED_400 = True
    STATUS_HINTS = {
        401: "invalid API key, get one at https://poe.com/api_key",
        402: "insufficient points/credits, see https://poe.com/subscribe",
        404: "bot not found (only public bots are reachable through the API)",
    }

    def __init__(
        self,
        api_key: Union[str, List[str]],
        model: str = "Claude-Sonnet-4",
        api_endpoint: Optional[str] = None,
        disable_thinking: bool = POE_DISABLE_THINKING,
        disable_web_search: bool = POE_DISABLE_WEB_SEARCH,
        *,
        retry_policy: Optional[RetryPolicy] = None,
        temperature: Optional[float] = None,
        log_callback=None,
    ):
        """
        Initialize the Poe provider.

        Args:
            api_key: Poe API key (get from https://poe.com/api_key)
            model: Bot name on Poe (default: Claude-Sonnet-4)
            api_endpoint: Optional custom API endpoint (default: https://api.poe.com/v1)
            disable_thinking: Ask the bot for its lowest reasoning setting.
                Translation gains nothing from reasoning tokens but pays for
                them: gemini-3.6-flash spends ~1170 reasoning tokens on a
                102-token prompt at its default thinking_level.
            disable_web_search: Turn off retrieval on bots that default to it.
                grok-4.5 sends 1141 prompt tokens for an 84-token translation
                prompt with search on, and every chunk pays that.
        """
        super().__init__(
            api_key=api_key, model=model, api_endpoint=api_endpoint, provider_name="poe",
            retry_policy=retry_policy, temperature=temperature, log_callback=log_callback,
        )
        self.disable_thinking = disable_thinking
        self.disable_web_search = disable_web_search

    async def _load_model_parameters(self) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """
        The parameter schemas Poe advertises for every bot.

        Fetched once per process. Returns None when the catalog is
        unreachable, which means "unknown" and not "no parameters"; the fetch
        is retried once CATALOG_RETRY_SECONDS have passed.
        """
        async def fetch():
            client = await self._get_client()
            response = await client.get(
                self.MODELS_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json"
                },
                timeout=30
            )
            response.raise_for_status()
            return {
                m["id"]: (m.get("parameters") or [])
                for m in response.json().get("data", [])
                if m.get("id")
            }

        return await type(self)._catalog.get(fetch)

    def _pick_bot_overrides(self, advertised: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Map a bot's advertised parameters to the settings translation wants."""
        overrides: Dict[str, Any] = {}
        for param in advertised:
            name = param.get("name")
            schema = param.get("schema") or {}

            if self.disable_thinking and name in self.REASONING_ENUM_PREFERENCES:
                allowed = schema.get("enum") or []
                for candidate in self.REASONING_ENUM_PREFERENCES[name]:
                    if candidate in allowed:
                        overrides[name] = candidate
                        break
            elif self.disable_thinking and name in self.REASONING_BUDGET_PARAMS:
                minimum = schema.get("minimum", 0)
                overrides[name] = minimum if isinstance(minimum, (int, float)) else 0
            elif self.disable_thinking and name in self.REASONING_TOGGLE_PARAMS:
                overrides[name] = False
            elif self.disable_web_search and name in self.WEB_SEARCH_PARAMS:
                overrides[name] = False

        return overrides

    def _overrides_key(self, model: Optional[str] = None) -> tuple:
        return (model or self.model, self.disable_thinking, self.disable_web_search)

    @property
    def _overrides_cache_key(self) -> tuple:
        return self._overrides_key()

    async def _get_bot_overrides(self, model: Optional[str] = None) -> Dict[str, Any]:
        """Return the request params that override this bot's costly defaults."""
        model = model or self.model
        if not (self.disable_thinking or self.disable_web_search):
            return {}

        cached = PoeProvider._bot_overrides.get(self._overrides_key(model))
        if cached is not None:
            return dict(cached)

        catalog = await self._load_model_parameters()
        if catalog is None:
            return {}

        overrides = self._pick_bot_overrides(catalog.get(model, []))
        PoeProvider._bot_overrides[self._overrides_key(model)] = overrides
        if overrides:
            logger.info(f"Poe ({model}): bot defaults overridden: {overrides}")
        return dict(overrides)

    # Known official providers to filter models (excludes community bots)
    # Note: Poe sometimes hosts models via intermediaries like "Novita AI", "Together AI", etc.
    OFFICIAL_PROVIDERS = {
        "openai", "anthropic", "google", "meta", "mistral", "xai",
        "deepseek", "qwen", "alibaba", "amazon", "cohere", "minimax",
        "zhipu", "moonshot", "baidu",
        # Intermediary providers that host official models
        "novita ai", "together ai", "cerebrasai", "empirio labs ai",
        "poe", "poe tools"
    }

    # Model name patterns to exclude (image/video/audio generators, coding-only, etc.)
    EXCLUDED_PATTERNS = [
        "image", "img", "dalle", "flux", "midjourney", "stable-diffusion",
        "video", "veo", "sora", "runway", "kling", "pika",
        "audio", "tts", "whisper", "music", "lyria", "elevenlabs",
        "vision-only", "canvas", "edit", "inpaint",
        "embedding", "embed",
        "banana", "reel"  # image generation bots
    ]

    # Model name patterns to prioritize (good for translation)
    PREFERRED_PATTERNS = [
        "claude", "gpt", "gemini", "llama", "mistral", "grok",
        "deepseek", "qwen", "glm", "command", "nova"
    ]

    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Fetch available models from Poe API.

        Poe provides a /v1/models endpoint that returns all available models
        with pricing information. We filter aggressively to only keep
        text-generation models suitable for translation.

        Returns:
            List of model dicts with id, name, context_length, and pricing info
        """
        try:
            client = await self._get_client()
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json"
            }

            response = await client.get(
                "https://api.poe.com/v1/models",
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                models_data = data.get("data", [])

                # Filter to text-capable models from official providers only
                models = []
                for m in models_data:
                    model_id = m.get("id", "")
                    if not model_id:
                        continue

                    model_id_lower = model_id.lower()

                    # Skip models matching excluded patterns
                    if any(pattern in model_id_lower for pattern in self.EXCLUDED_PATTERNS):
                        continue

                    # Get provider - skip community bots (no owned_by or unknown provider)
                    owned_by = (m.get("owned_by") or "").lower()
                    if not owned_by or owned_by not in self.OFFICIAL_PROVIDERS:
                        continue

                    # Get modalities - filter out image/video only models
                    output_modalities = m.get("output_modalities", ["text"])

                    # Skip if no text output capability or if primary output is not text
                    if "text" not in output_modalities:
                        continue

                    # Skip if output is primarily image/video (even if text is listed)
                    if "image" in output_modalities or "video" in output_modalities:
                        # Only keep if it's a known text model
                        if not any(pattern in model_id_lower for pattern in self.PREFERRED_PATTERNS):
                            continue

                    # Get pricing info - NOTE: Poe returns prices as STRINGS
                    pricing = m.get("pricing") or {}
                    prompt_price_raw = pricing.get("prompt")
                    completion_price_raw = pricing.get("completion")
                    request_price_raw = pricing.get("request")

                    # Convert string prices to float
                    try:
                        prompt_price = float(prompt_price_raw) if prompt_price_raw else 0
                    except (ValueError, TypeError):
                        prompt_price = 0
                    try:
                        completion_price = float(completion_price_raw) if completion_price_raw else 0
                    except (ValueError, TypeError):
                        completion_price = 0
                    try:
                        request_price = float(request_price_raw) if request_price_raw else 0
                    except (ValueError, TypeError):
                        request_price = 0

                    # Convert to per-million tokens for consistency with OpenRouter
                    prompt_per_million = prompt_price * 1_000_000 if prompt_price else 0
                    completion_per_million = completion_price * 1_000_000 if completion_price else 0

                    # Skip models that cost more than $100 per 1M output tokens (very expensive)
                    # This keeps most useful models while filtering extreme outliers
                    if completion_per_million > 100.0:
                        continue

                    # Skip models with very high request-based pricing (> $0.10 per request)
                    if request_price > 0.10:
                        continue

                    # Skip free models (no pricing = free tier, often lower quality/rate limited)
                    if prompt_price == 0 and completion_price == 0 and request_price == 0:
                        continue

                    # Build display name with context info
                    description = m.get("description", "")
                    # Use display name from API if available, otherwise use model_id
                    display_name = m.get("name") or model_id

                    model_info = {
                        "id": model_id,
                        "name": display_name,
                        "description": description[:100] if description else "",
                        "owned_by": m.get("owned_by", ""),
                        "context_length": self._get_context_limit_for_model(model_id),
                        "pricing": {
                            "prompt": prompt_price,
                            "completion": completion_price,
                            "request": request_price,
                            "prompt_per_million": prompt_per_million,
                            "completion_per_million": completion_per_million
                        }
                    }
                    models.append(model_info)

                # Sort by owned_by (provider) then by name
                models.sort(key=lambda x: (x.get("owned_by", "zzz").lower(), x.get("id", "").lower()))

                if models:
                    logger.info(f"Poe: loaded {len(models)} text models from API (filtered from {len(models_data)} total)")
                    return models

            # API error - fall back to static list
            logger.warning(f"Poe API returned status {response.status_code}, using fallback models")

        except Exception as e:
            logger.warning(f"Poe: error fetching models: {e}, using fallback list")

        # Fallback to static list
        return self._get_fallback_models()

    def _get_fallback_models(self) -> List[Dict[str, Any]]:
        """Return fallback models list."""
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

    # ------------------------------------------------- chat-completions hooks
    async def _optional_params(self, request: LLMRequest) -> Dict[str, Any]:
        params = await super()._optional_params(request)
        params.update(await self._get_bot_overrides(request.model))
        return params

    def _on_optional_params_rejected(self, rejected: Iterable[str], request: LLMRequest) -> None:
        # The bot genuinely rejects one of our overrides: stop sending them
        # for the rest of the run.
        key = self._overrides_key(request.model)
        if any(name in (PoeProvider._bot_overrides.get(key) or {}) for name in rejected):
            PoeProvider._bot_overrides[key] = {}
