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

from typing import Optional, Dict, Any, Callable, List
import httpx
import asyncio
import json

from src.config import REQUEST_TIMEOUT, MAX_TRANSLATION_ATTEMPTS
from ..base import LLMProvider, LLMResponse
from ..exceptions import ContextOverflowError


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

    # Session cost tracking (class-level)
    _session_cost = 0.0
    _session_tokens = {"prompt": 0, "completion": 0}
    _cost_callback: Optional[Callable[[Dict[str, Any]], None]] = None

    def __init__(
        self,
        api_key: str,
        model: str = "Claude-Sonnet-4",
        api_endpoint: Optional[str] = None
    ):
        """
        Initialize the Poe provider.

        Args:
            api_key: Poe API key (get from https://poe.com/api_key)
            model: Bot name on Poe (default: Claude-Sonnet-4)
            api_endpoint: Optional custom API endpoint (default: https://api.poe.com/v1)
        """
        super().__init__(model)
        self.api_key = api_key
        self.api_endpoint = api_endpoint or self.API_URL

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
            callback: Function that receives a dict with token usage info
        """
        cls._cost_callback = callback

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
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        # Build messages array
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": 0.3,  # Lower temperature for translation consistency
        }

        client = await self._get_client()
        for attempt in range(MAX_TRANSLATION_ATTEMPTS):
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
                    # Rate limited - use exponential backoff with jitter
                    retry_after = response.headers.get("Retry-After")
                    wait_time = int(retry_after) if retry_after else (2 ** attempt) + (asyncio.get_event_loop().time() % 1)
                    print(f"⚠️ Poe rate limited, waiting {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                    continue

                response.raise_for_status()
                result = response.json()

                if "choices" not in result or len(result["choices"]) == 0:
                    print(f"⚠️ Poe: Unexpected response format: {result}")
                    return None

                response_text = result["choices"][0].get("message", {}).get("content", "")

                # Track token usage
                usage = result.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                # Update session tracking
                PoeProvider._session_tokens["prompt"] += prompt_tokens
                PoeProvider._session_tokens["completion"] += completion_tokens

                print(f"💬 Poe ({self.model}): {prompt_tokens}+{completion_tokens} tokens")

                # Call cost callback if set
                if PoeProvider._cost_callback:
                    try:
                        PoeProvider._cost_callback({
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_prompt_tokens": PoeProvider._session_tokens["prompt"],
                            "total_completion_tokens": PoeProvider._session_tokens["completion"],
                        })
                    except Exception as cb_err:
                        print(f"⚠️ Cost callback error: {cb_err}")

                return LLMResponse(
                    content=response_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    context_used=prompt_tokens + completion_tokens,
                    context_limit=self._get_context_limit(),
                    was_truncated=False
                )

            except httpx.TimeoutException as e:
                print(f"Poe API Timeout (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
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

                if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                    await asyncio.sleep(2)
                    continue
                return None

            except json.JSONDecodeError as e:
                print(f"Poe API JSON Decode Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                    await asyncio.sleep(2)
                    continue
                return None

            except Exception as e:
                print(f"Poe API Unknown Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                    await asyncio.sleep(2)
                    continue
                return None

        return None
