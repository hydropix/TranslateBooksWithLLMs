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

from typing import Optional, Dict, Any, Callable, List, Union
import httpx
import asyncio
import json

from src.config import (
    REQUEST_TIMEOUT, MAX_TRANSLATION_ATTEMPTS, TEMPERATURE,
    POE_DISABLE_THINKING, POE_DISABLE_WEB_SEARCH
)
from ..base import LLMProvider, LLMResponse, is_output_limit_finish
from ..exceptions import ContextOverflowError
from ..cost_tracker import CostTracker, get_current_cost_tracker
from ..rate_limit_handler import handle_rate_limit, is_retryable_http_status


class PoeProvider(LLMProvider):
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

    # Advertised parameter schemas, fetched once per process: {model_id: [param]}
    _model_parameters: Optional[Dict[str, List[Dict[str, Any]]]] = None
    _model_parameters_failed = False
    _model_parameters_lock: Optional[asyncio.Lock] = None
    # Resolved overrides, keyed by model *and* flags since they decide the
    # result: {(model_id, disable_thinking, disable_web_search): {param: value}}
    _bot_overrides: Dict[tuple, Dict[str, Any]] = {}

    def __init__(
        self,
        api_key: Union[str, List[str]],
        model: str = "Claude-Sonnet-4",
        api_endpoint: Optional[str] = None,
        disable_thinking: bool = POE_DISABLE_THINKING,
        disable_web_search: bool = POE_DISABLE_WEB_SEARCH
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
        super().__init__(model, api_keys=api_key, provider_name="poe")
        self.api_endpoint = api_endpoint or self.API_URL
        self.disable_thinking = disable_thinking
        self.disable_web_search = disable_web_search
        # Used when no per-job tracker is active (direct use outside a job).
        self._cost_tracker = CostTracker()

    def _get_context_limit(self) -> int:
        """
        Determine context limit based on model name.

        Returns:
            Context limit in tokens
        """
        model_lower = self.model.lower()
        for prefix, limit in self.MODEL_CONTEXT_SIZES.items():
            if prefix in model_lower:
                return limit
        return 32000  # Conservative default for unknown models

    async def _load_model_parameters(self) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """
        Fetch the parameter schemas Poe advertises for every bot.

        Fetched once per process and cached class-side. Returns None when the
        catalog is unreachable, which means "unknown" and not "no parameters".
        """
        if PoeProvider._model_parameters is not None:
            return PoeProvider._model_parameters
        if PoeProvider._model_parameters_failed:
            return None

        # Safe to create here without a double-check race: no await between the
        # test and the assignment, so the event loop cannot interleave.
        if PoeProvider._model_parameters_lock is None:
            PoeProvider._model_parameters_lock = asyncio.Lock()

        async with PoeProvider._model_parameters_lock:
            if PoeProvider._model_parameters is not None:
                return PoeProvider._model_parameters
            if PoeProvider._model_parameters_failed:
                return None
            try:
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
                PoeProvider._model_parameters = {
                    m["id"]: (m.get("parameters") or [])
                    for m in response.json().get("data", [])
                    if m.get("id")
                }
                return PoeProvider._model_parameters
            except Exception as e:
                print(f"⚠️ Poe: could not read model parameters ({e});"
                      f" reasoning stays at the bot's default")
                PoeProvider._model_parameters_failed = True
                return None

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

    @property
    def _overrides_cache_key(self) -> tuple:
        return (self.model, self.disable_thinking, self.disable_web_search)

    async def _get_bot_overrides(self) -> Dict[str, Any]:
        """Return the request params that override this bot's costly defaults."""
        if not (self.disable_thinking or self.disable_web_search):
            return {}

        cached = PoeProvider._bot_overrides.get(self._overrides_cache_key)
        if cached is not None:
            return dict(cached)

        catalog = await self._load_model_parameters()
        if catalog is None:
            return {}

        overrides = self._pick_bot_overrides(catalog.get(self.model, []))
        PoeProvider._bot_overrides[self._overrides_cache_key] = overrides
        if overrides:
            print(f"⚙️ Poe ({self.model}): bot defaults overridden: {overrides}")
        return dict(overrides)

    def get_session_cost(self) -> tuple:
        """
        Get this provider instance's token usage.

        Poe bills in compute points, not USD, so the cost is always 0.0.

        Returns:
            Tuple of (total_cost_usd, token_counts_dict)
        """
        return self._cost_tracker.snapshot()

    def reset_session_cost(self) -> None:
        """Reset this provider instance's usage tracking."""
        self._cost_tracker.reset()

    def set_cost_callback(self, callback: Optional[Callable[[Dict[str, Any]], None]]) -> None:
        """Set a callback receiving token usage after each call on this instance."""
        self._cost_tracker.callback = callback

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
                    print(f"✅ Poe: Loaded {len(models)} text models from API (filtered from {len(models_data)} total)")
                    return models

            # API error - fall back to static list
            print(f"⚠️ Poe API returned status {response.status_code}, using fallback models")

        except Exception as e:
            print(f"⚠️ Poe: Error fetching models: {e}, using fallback list")

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
        model_lower = model_name.lower()
        for prefix, limit in self.MODEL_CONTEXT_SIZES.items():
            if prefix in model_lower:
                return limit
        return 32000

    async def generate(
        self,
        prompt: str,
        timeout: int = REQUEST_TIMEOUT,
        system_prompt: Optional[str] = None
    ) -> Optional[LLMResponse]:
        """
        Generate text using Poe API.

        Args:
            prompt: The user prompt (content to translate)
            timeout: Request timeout in seconds
            system_prompt: Optional system prompt (role/instructions)

        Returns:
            LLMResponse with content and token usage info, or None if failed

        Raises:
            ContextOverflowError: If input exceeds model's context window
        """
        # Build messages array
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        bot_overrides = await self._get_bot_overrides()
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": TEMPERATURE,
            **bot_overrides,
        }

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
                "Accept": "application/json"
            }
            try:
                response = await client.post(
                    self.api_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )

                # Handle specific error codes
                if response.status_code == 401:
                    print(f"❌ Poe: Invalid API key!")
                    print(f"   Get your API key at: https://poe.com/api_key")
                    return None

                if response.status_code == 402:
                    print(f"❌ Poe: Insufficient points/credits!")
                    print(f"   Check your subscription at: https://poe.com/subscribe")
                    return None

                if response.status_code == 429:
                    rate_limit_events += 1
                    await handle_rate_limit(
                        self._key_pool, current_key, response.headers,
                        rate_limit_events, MAX_TRANSLATION_ATTEMPTS,
                    )
                    continue

                # Never let our own overrides be the reason a chunk fails: on
                # any 400 we retry once without them. Consumes no attempt, and
                # the branch is one-shot since the overrides are emptied. Poe
                # names the offending argument in the body ("Unrecognized
                # request argument supplied: X", or "do not support extra_body
                # fields: X"); when it names one of ours, the bot genuinely
                # rejects it, so stop sending it this run.
                if response.status_code == 400 and bot_overrides:
                    rejected = [n for n in bot_overrides if n in response.text]
                    if rejected:
                        print(f"⚠️ Poe ({self.model}): params {sorted(rejected)} "
                              f"rejected by the bot, dropping them for this run")
                        PoeProvider._bot_overrides[self._overrides_cache_key] = {}
                    else:
                        print(f"⚠️ Poe ({self.model}): HTTP 400, retrying once without "
                              f"our param overrides to rule them out")
                    for name in bot_overrides:
                        payload.pop(name, None)
                    bot_overrides = {}
                    continue

                response.raise_for_status()
                result = response.json()

                if "choices" not in result or len(result["choices"]) == 0:
                    print(f"⚠️ Poe: Unexpected response format: {result}")
                    return None

                response_text = result["choices"][0].get("message", {}).get("content", "")
                finish_reason = result["choices"][0].get("finish_reason")

                # Track token usage
                usage = result.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                print(f"💬 Poe ({self.model}): {prompt_tokens}+{completion_tokens} tokens")

                # Poe reports no USD cost: record tokens only (cost 0.0).
                self._cost_tracker.record(0.0, prompt_tokens, completion_tokens)
                job_tracker = get_current_cost_tracker()
                if job_tracker is not None:
                    job_tracker.record(0.0, prompt_tokens, completion_tokens)

                return LLMResponse(
                    content=response_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    context_used=prompt_tokens + completion_tokens,
                    context_limit=self._get_context_limit(),
                    was_truncated=False,
                    finish_reason=finish_reason,
                    output_truncated=is_output_limit_finish(finish_reason)
                )

            except httpx.TimeoutException as e:
                print(f"Poe API Timeout (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
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

                if e.response.status_code == 404:
                    print(f"❌ Poe: Bot '{self.model}' not found!")
                    print(f"   Note: Only public bots are accessible via API")
                    print(f"   Find bots at: https://poe.com/explore")
                elif e.response.status_code not in [401, 402, 429]:  # Already handled above
                    print(f"Poe API HTTP Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                    print(f"Response details: Status {e.response.status_code}, Body: {error_body}...")

                # Detect context overflow errors
                context_overflow_keywords = [
                    "context_length", "maximum context", "token limit",
                    "too many tokens", "reduce the length", "max_tokens",
                    "context window", "exceeds"
                ]
                if any(keyword in error_message.lower() for keyword in context_overflow_keywords):
                    raise ContextOverflowError(f"Poe context overflow: {error_message}")

                # Client errors (404 bot, 401 key, 402 credits, 400) won't
                # recover on retry — fail fast.
                if not is_retryable_http_status(e.response.status_code):
                    return None

                attempt += 1
                if attempt < MAX_TRANSLATION_ATTEMPTS:
                    await asyncio.sleep(2)
                    continue
                return None

            except json.JSONDecodeError as e:
                print(f"Poe API JSON Decode Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                attempt += 1
                if attempt < MAX_TRANSLATION_ATTEMPTS:
                    await asyncio.sleep(2)
                    continue
                return None

            except Exception as e:
                print(f"Poe API Unknown Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                attempt += 1
                if attempt < MAX_TRANSLATION_ATTEMPTS:
                    await asyncio.sleep(2)
                    continue
                return None

        return None
