"""
OpenRouter provider implementation.

This module provides the OpenRouterProvider class for interacting with
OpenRouter's API, which provides access to 200+ models.

Features:
    - Access to 200+ models (Claude, GPT-4, Llama, Mistral, etc.)
    - Built-in cost tracking
    - Model validation
    - Automatic context size detection
    - Reasoning disabled by default (translation-friendly), per model metadata
    - Account presets (@preset/<slug>) listed with the models. Presets hold
      settings TBL has no options for (provider routing, model fallbacks),
      edited on openrouter.ai
"""

from typing import List, Optional, Dict, Any, Callable, Union
import httpx
import asyncio
import json

from src.config import (
    REQUEST_TIMEOUT, MAX_TRANSLATION_ATTEMPTS, OPENROUTER_DISABLE_THINKING
)
from ..base import LLMProvider, LLMResponse, is_output_limit_finish
from ..exceptions import ContextOverflowError
from ..rate_limit_handler import handle_rate_limit, is_retryable_http_status


class OpenRouterProvider(LLMProvider):
    """
    Provider for OpenRouter API.

    OpenRouter provides unified access to multiple LLM providers including:
        - Anthropic (Claude)
        - OpenAI (GPT-4, GPT-3.5)
        - Meta (Llama)
        - Google (Gemini, PaLM)
        - Mistral AI
        - And 200+ more models

    Features:
        - Automatic model validation
        - Per-request cost tracking
        - Session cost accumulation
        - Context size detection

    Configuration:
        endpoint: https://openrouter.ai/api/v1/chat/completions
        model: Model identifier (e.g., "anthropic/claude-3-opus")
        api_key: OpenRouter API key

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

    # Presets are referenced as a model string: "@preset/<slug>"
    PRESET_PREFIX = "@preset/"
    PRESETS_PAGE_SIZE = 100  # API maximum
    PRESETS_MAX_PAGES = 10

    # Session cost tracking (class-level)
    _session_cost = 0.0
    _session_tokens = {"prompt": 0, "completion": 0}
    _cost_callback: Optional[Callable[[Dict[str, Any]], None]] = None

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

    # Model catalog entries, fetched once per process: {model_id: model_dict}
    _model_catalog: Optional[Dict[str, Dict[str, Any]]] = None
    _model_catalog_failed = False
    _model_catalog_lock: Optional[asyncio.Lock] = None
    # Resolved override per model: {model_id: reasoning dict, or {} for none}
    _reasoning_overrides: Dict[str, Dict[str, Any]] = {}

    def __init__(
        self,
        api_key: Union[str, List[str]],
        model: str = "anthropic/claude-sonnet-4",
        disable_thinking: bool = OPENROUTER_DISABLE_THINKING
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
        super().__init__(model, api_keys=api_key, provider_name="openrouter")
        self.disable_thinking = disable_thinking
        self._warned_reasoning_tokens = False

    async def _load_model_catalog(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Fetch OpenRouter's model catalog (reasoning metadata included).

        Fetched once per process and cached class-side. Returns None when the
        catalog is unreachable, which means "unknown" and not "no models".
        """
        if OpenRouterProvider._model_catalog is not None:
            return OpenRouterProvider._model_catalog
        if OpenRouterProvider._model_catalog_failed:
            return None

        # Safe to create here without a double-check race: no await between the
        # test and the assignment, so the event loop cannot interleave.
        if OpenRouterProvider._model_catalog_lock is None:
            OpenRouterProvider._model_catalog_lock = asyncio.Lock()

        async with OpenRouterProvider._model_catalog_lock:
            if OpenRouterProvider._model_catalog is not None:
                return OpenRouterProvider._model_catalog
            if OpenRouterProvider._model_catalog_failed:
                return None
            try:
                client = await self._get_client()
                response = await client.get(
                    self.MODELS_URL,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=30
                )
                response.raise_for_status()
                OpenRouterProvider._model_catalog = {
                    m["id"]: m for m in response.json().get("data", []) if m.get("id")
                }
                return OpenRouterProvider._model_catalog
            except Exception as e:
                print(f"[OpenRouter] WARN: could not read the model catalog ({e}); "
                      f"reasoning stays at the model default")
                OpenRouterProvider._model_catalog_failed = True
                return None

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

    async def _get_reasoning_override(self) -> Dict[str, Any]:
        """Return the `reasoning` request value for this model ({} = send none)."""
        if not self.disable_thinking:
            return {}

        # A preset carries its own model and reasoning config; don't override it.
        if self.model.startswith(self.PRESET_PREFIX):
            return {}

        cached = OpenRouterProvider._reasoning_overrides.get(self.model)
        if cached is not None:
            return dict(cached)

        catalog = await self._load_model_catalog()
        if catalog is None:
            return {}

        model_info = catalog.get(self.model)
        if model_info is None:
            print(f"[OpenRouter] WARN: '{self.model}' is not in the model catalog; "
                  f"reasoning stays at the model default")
            override = {}
        else:
            override = self._pick_reasoning_override(model_info)
            if (model_info.get("reasoning") or {}).get("mandatory"):
                print(f"[OpenRouter] WARN: '{self.model}' always reasons and cannot "
                      f"turn it off; reasoning tokens are billed as output "
                      f"(requested: {override or 'model default'})")
            elif override:
                print(f"[OpenRouter] {self.model}: reasoning disabled")

        OpenRouterProvider._reasoning_overrides[self.model] = override
        return dict(override)

    @classmethod
    def get_session_cost(cls) -> tuple:
        """
        Get the current session cost and token usage.

        Returns:
            Tuple of (total_cost_usd, token_counts_dict)
        """
        return cls._session_cost, cls._session_tokens.copy()

    @classmethod
    def reset_session_cost(cls) -> None:
        """Reset the session cost tracking."""
        cls._session_cost = 0.0
        cls._session_tokens = {"prompt": 0, "completion": 0}

    @classmethod
    def set_cost_callback(cls, callback: Optional[Callable[[Dict[str, Any]], None]]) -> None:
        """
        Set a callback to receive cost updates after each API call.

        Args:
            callback: Function that receives a dict with:
                - request_cost: Cost of this specific request (USD)
                - session_cost: Cumulative session cost (USD)
                - prompt_tokens: Tokens used for this request's prompt
                - completion_tokens: Tokens generated in this request
                - total_prompt_tokens: Cumulative prompt tokens
                - total_completion_tokens: Cumulative completion tokens
        """
        cls._cost_callback = callback

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
            print(f"[OpenRouter] WARN: Failed to fetch models: {e}")
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
            print(f"[OpenRouter] WARN: Failed to fetch presets: {e}")

        presets.sort(key=lambda p: p["id"])
        return presets

    def _get_fallback_models(self) -> list:
        """Return fallback models list when API fetch fails."""
        return [{"id": m, "name": m, "pricing": {"prompt": 0, "completion": 0}}
                for m in self.FALLBACK_MODELS]

    async def generate(self, prompt: str, timeout: int = REQUEST_TIMEOUT,
                      system_prompt: Optional[str] = None) -> Optional[LLMResponse]:
        """
        Generate text using OpenRouter API with cost tracking.

        Args:
            prompt: The user prompt (content to translate)
            timeout: Request timeout in seconds
            system_prompt: Optional system prompt (role/instructions)

        Returns:
            LLMResponse with content and token usage info, or None if failed

        Raises:
            ContextOverflowError: If input exceeds model's context window
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        reasoning_override = await self._get_reasoning_override()
        if reasoning_override:
            payload["reasoning"] = reasoning_override

        client = await self._get_client()
        # 429s have their own budget (rate_limit_events): rotating to a spare
        # key must not consume a transient-retry attempt (issue #217).
        attempt = 0
        rate_limit_events = 0
        while attempt < MAX_TRANSLATION_ATTEMPTS:
            current_key = await self._key_pool.acquire()
            headers = {
                "Authorization": f"Bearer {current_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/hydropix/TranslateBookWithLLM",
                "X-Title": "TranslateBookWithLLM",
            }
            try:
                response = await client.post(
                    self.API_URL,
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )
                response.raise_for_status()

                result = response.json()

                if "choices" not in result or len(result["choices"]) == 0:
                    print(f"[OpenRouter] WARN: Unexpected response format: {result}")
                    return None

                # NOTE: a present-but-null "content" makes .get(..., "") return None,
                # so coalesce with `or ""` to guarantee a string downstream.
                response_text = result["choices"][0].get("message", {}).get("content") or ""
                finish_reason = result["choices"][0].get("finish_reason")

                usage = result.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0

                if (reasoning_tokens and reasoning_override.get("enabled") is False
                        and not self._warned_reasoning_tokens):
                    self._warned_reasoning_tokens = True
                    print(f"[OpenRouter] WARN: '{self.model}' still produced "
                          f"{reasoning_tokens} reasoning tokens with reasoning disabled; "
                          f"the upstream provider may ignore the setting")

                if not response_text.strip():
                    print(f"[OpenRouter] WARN: Empty response from model '{self.model}' "
                          f"({prompt_tokens}+{completion_tokens} tokens). The model likely "
                          f"refused or filtered this chunk (sensitive/policy-flagged content "
                          f"or provider-side moderation). Try a different model.")

                if "cost" in result:
                    cost = float(result.get("cost", 0))
                else:
                    # Fallback estimate when OpenRouter omits cost (typical rates in USD)
                    cost = (prompt_tokens * 0.50 / 1_000_000) + (completion_tokens * 1.50 / 1_000_000)

                OpenRouterProvider._session_cost += cost
                OpenRouterProvider._session_tokens["prompt"] += prompt_tokens
                OpenRouterProvider._session_tokens["completion"] += completion_tokens

                reasoning_note = f" ({reasoning_tokens} reasoning)" if reasoning_tokens else ""
                print(f"[OpenRouter] {prompt_tokens}+{completion_tokens} tokens{reasoning_note} | "
                      f"Cost: ${cost:.6f} (session: ${OpenRouterProvider._session_cost:.4f})")

                if OpenRouterProvider._cost_callback:
                    try:
                        OpenRouterProvider._cost_callback({
                            "request_cost": cost,
                            "session_cost": OpenRouterProvider._session_cost,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_prompt_tokens": OpenRouterProvider._session_tokens["prompt"],
                            "total_completion_tokens": OpenRouterProvider._session_tokens["completion"],
                        })
                    except Exception as cb_err:
                        print(f"[OpenRouter] WARN: Cost callback error: {cb_err}")

                return LLMResponse(
                    content=response_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    context_used=prompt_tokens + completion_tokens,
                    context_limit=0,  # OpenRouter manages context internally
                    was_truncated=False,
                    finish_reason=finish_reason,
                    output_truncated=is_output_limit_finish(finish_reason)
                )

            except httpx.TimeoutException as e:
                print(f"OpenRouter API Timeout (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                attempt += 1
                if attempt < MAX_TRANSLATION_ATTEMPTS:
                    await asyncio.sleep(2)
                    continue
                return None
            except httpx.HTTPStatusError as e:
                error_body = ""
                error_message = str(e)
                if hasattr(e, 'response') and hasattr(e.response, 'text'):
                    error_body = e.response.text[:500]
                    error_message = f"{e} - {error_body}"

                if e.response.status_code == 429:
                    rate_limit_events += 1
                    await handle_rate_limit(
                        self._key_pool, current_key, e.response.headers,
                        rate_limit_events, MAX_TRANSLATION_ATTEMPTS,
                    )
                    continue

                if e.response.status_code == 404:
                    print(f"[OpenRouter] ERROR: Model '{self.model}' not found!")
                    print(f"   Check available models at https://openrouter.ai/models")
                    print(f"   Response: {error_body}")
                elif e.response.status_code == 401:
                    print(f"[OpenRouter] ERROR: Invalid API key!")
                elif e.response.status_code == 402:
                    print(f"[OpenRouter] ERROR: Insufficient credits!")
                else:
                    print(f"OpenRouter API HTTP Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                    print(f"Response details: Status {e.response.status_code}, Body: {error_body}...")

                context_overflow_keywords = ["context_length", "maximum context", "token limit",
                                              "too many tokens", "reduce the length", "max_tokens",
                                              "context window", "exceeds"]
                if any(keyword in error_message.lower() for keyword in context_overflow_keywords):
                    raise ContextOverflowError(f"OpenRouter context overflow: {error_message}")

                # Client errors (404 model, 401 key, 402 credits, 400) won't
                # recover on retry — fail fast instead of retrying 3x.
                if not is_retryable_http_status(e.response.status_code):
                    return None

                attempt += 1
                if attempt < MAX_TRANSLATION_ATTEMPTS:
                    await asyncio.sleep(2)
                    continue
                return None
            except json.JSONDecodeError as e:
                print(f"OpenRouter API JSON Decode Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                attempt += 1
                if attempt < MAX_TRANSLATION_ATTEMPTS:
                    await asyncio.sleep(2)
                    continue
                return None
            except Exception as e:
                print(f"OpenRouter API Unknown Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                attempt += 1
                if attempt < MAX_TRANSLATION_ATTEMPTS:
                    await asyncio.sleep(2)
                    continue
                return None

        return None
