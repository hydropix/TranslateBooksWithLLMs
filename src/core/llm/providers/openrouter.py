"""
OpenRouter provider implementation.

This module provides the OpenRouterProvider class for interacting with
OpenRouter's API, which provides access to 200+ models.

Features:
    - Access to 200+ models (Claude, GPT-4, Llama, Mistral, etc.)
    - Built-in cost tracking
    - Model validation
    - Automatic context size detection
"""

from typing import List, Optional, Dict, Any, Callable, Union
import httpx
import asyncio
import json

from src.config import REQUEST_TIMEOUT, MAX_TRANSLATION_ATTEMPTS
from ..base import LLMProvider, LLMResponse
from ..exceptions import ContextOverflowError
from ..rate_limit_handler import handle_rate_limit


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

    def __init__(self, api_key: Union[str, List[str]], model: str = "anthropic/claude-sonnet-4"):
        """
        Initialize the OpenRouter provider.

        Args:
            api_key: OpenRouter API key
            model: Model identifier (default: anthropic/claude-sonnet-4)
        """
        super().__init__(model, api_keys=api_key, provider_name="openrouter")

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

            return filtered_models

        except Exception as e:
            print(f"[OpenRouter] WARN: Failed to fetch models: {e}")
            return self._get_fallback_models()

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

        # thinking/enable_thinking forwarded to underlying models (DeepSeek, Qwen, etc.)
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "thinking": False,
            "enable_thinking": False,
        }

        client = await self._get_client()
        for attempt in range(MAX_TRANSLATION_ATTEMPTS):
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

                usage = result.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

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

                print(f"[OpenRouter] {prompt_tokens}+{completion_tokens} tokens | "
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
                    was_truncated=False
                )

            except httpx.TimeoutException as e:
                print(f"OpenRouter API Timeout (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
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

                if e.response.status_code == 429:
                    await handle_rate_limit(
                        self._key_pool, current_key, e.response.headers,
                        attempt, MAX_TRANSLATION_ATTEMPTS,
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

                if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                    await asyncio.sleep(2)
                    continue
                return None
            except json.JSONDecodeError as e:
                print(f"OpenRouter API JSON Decode Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                    await asyncio.sleep(2)
                    continue
                return None
            except Exception as e:
                print(f"OpenRouter API Unknown Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                    await asyncio.sleep(2)
                    continue
                return None

        return None
