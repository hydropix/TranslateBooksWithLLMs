"""
Google Gemini provider implementation.

This module provides the GeminiProvider class for interacting with
Google's Gemini API.

Features:
    - Gemini 2.0 Flash and other models
    - Large context windows
    - Efficient batch processing
"""

from typing import List, Optional, Union
import httpx
import asyncio

from src.config import (
    REQUEST_TIMEOUT,
    MAX_TRANSLATION_ATTEMPTS,
    TEMPERATURE,
    GEMINI_SAFETY_THRESHOLD,
    GEMINI_THINKING_LEVEL,
    GEMINI_THINKING_BUDGET,
)
from ..base import LLMProvider, LLMResponse
from ..exceptions import ContextOverflowError
from ..rate_limit_handler import handle_rate_limit, is_retryable_http_status

# Four harm categories returned by the Gemini API. By default Gemini blocks
# anything at MEDIUM and above on all four, which strips a significant portion
# of CN/KR webnovel content (romance, violence, supernatural themes). We send
# explicit safetySettings on every request so the configured threshold (see
# GEMINI_SAFETY_THRESHOLD in config.py) always wins over the API default.
_GEMINI_HARM_CATEGORIES = (
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
)


class GeminiProvider(LLMProvider):
    """
    Provider for Google Gemini API.

    Supports Gemini models including:
        - gemini-2.0-flash-exp
        - gemini-1.5-pro
        - gemini-1.5-flash

    Features:
        - Large context windows (up to 2M tokens for some models)
        - Fast response times
        - Good for batch translation

    Configuration:
        api_key: Google AI API key (required)
        model: Gemini model name

    Example:
        >>> provider = GeminiProvider(
        ...     api_key="AI...",
        ...     model="gemini-2.0-flash-exp"
        ... )
        >>> response = await provider.generate("Translate: Hello")
    """

    def __init__(self, api_key: Union[str, List[str]], model: str = "gemini-2.0-flash"):
        """
        Initialize the Gemini provider.

        Args:
            api_key: Google AI API key. Accepts a single key string OR a list
                of keys for automatic failover on HTTP 429 (the base class wraps
                them in a KeyPool that rotates on rate-limit).
            model: Gemini model name (default: gemini-2.0-flash)
        """
        super().__init__(model, api_keys=api_key, provider_name="gemini")
        self.api_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def _is_thinking_model(self) -> bool:
        """Check if the current model supports thinking mode (Gemini 2.5+ and 3.x)."""
        return "2.5" in self.model or "gemini-3" in self.model

    def _get_thinking_config(self) -> dict:
        """
        Build the thinkingConfig block for this request, based on the model
        generation and the GEMINI_THINKING_LEVEL / GEMINI_THINKING_BUDGET env
        settings.

        Gemini 3.x models use the newer `thinkingLevel` field (minimal/low/
        medium/high). Gemini 2.5.x models use the older `thinkingBudget`
        field (a token count; 0 disables thinking, -1 is dynamic).

        Thinking is billed as output tokens, so for mechanical tasks like
        translation, turning it down (or off, where the model allows it)
        can meaningfully cut cost with little to no quality impact.
        """
        if "gemini-3" in self.model:
            # default levels: medium for Flash, minimal for Flash-Lite, high for Pro
            thinking_level = GEMINI_THINKING_LEVEL if GEMINI_THINKING_LEVEL is not None else 'minimal'
            return {"thinkingConfig": {"thinkingLevel": thinking_level}}

        if "2.5" in self.model:
            # Preserve the previous hardcoded behavior (thinking off) when
            # the user hasn't set an explicit budget, for backward compat.
            budget = GEMINI_THINKING_BUDGET if GEMINI_THINKING_BUDGET is not None else 0
            return {"thinkingConfig": {"thinkingBudget": budget}}

        return {}

    async def get_available_models(self) -> list[dict]:
        """
        Fetch available Gemini models from API, excluding experimental/vision models.

        Returns:
            List of model dictionaries with name, displayName, description, and token limits
        """
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }

        models_endpoint = "https://generativelanguage.googleapis.com/v1beta/models"

        client = await self._get_client()
        try:
            response = await client.get(
                models_endpoint,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            models = []

            for model in data.get("models", []):
                model_name = model.get("name", "").replace("models/", "")

                # Skip experimental, latest, and vision models
                model_name_lower = model_name.lower()
                skip_keywords = ["experimental", "latest", "vision", "-exp-"]
                if any(keyword in model_name_lower for keyword in skip_keywords):
                    continue

                # Only include models that support generateContent
                supported_methods = model.get("supportedGenerationMethods", [])
                if "generateContent" in supported_methods:
                    models.append({
                        "name": model_name,
                        "displayName": model.get("displayName", model_name),
                        "description": model.get("description", ""),
                        "inputTokenLimit": model.get("inputTokenLimit", 0),
                        "outputTokenLimit": model.get("outputTokenLimit", 0)
                    })

            return models

        except httpx.HTTPStatusError as e:
            # Include status and response body so the caller (and the user
            # through the API route) can see why the listing failed. Common
            # cases: 400 (API key not enabled for the project), 403 (key
            # restricted or revoked), 429 (quota exhausted).
            body = ""
            if hasattr(e, "response") and hasattr(e.response, "text"):
                body = e.response.text[:500]
            raise RuntimeError(
                f"Gemini /v1beta/models returned HTTP {e.response.status_code}: {body}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Error fetching Gemini models: {e}") from e

    async def generate(self, prompt: str, timeout: int = REQUEST_TIMEOUT,
                      system_prompt: Optional[str] = None) -> Optional[LLMResponse]:
        """
        Generate text using Gemini API.

        Args:
            prompt: The user prompt (content to translate)
            timeout: Request timeout in seconds
            system_prompt: Optional system prompt (role/instructions)

        Returns:
            LLMResponse with content and token usage info, or None if failed

        Raises:
            ContextOverflowError: If input exceeds Gemini's context window
        """
        payload = {
            "contents": [{
                "role": "user",
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": TEMPERATURE,
                **self._get_thinking_config()
            },
            "safetySettings": [
                {"category": category, "threshold": GEMINI_SAFETY_THRESHOLD}
                for category in _GEMINI_HARM_CATEGORIES
            ],
        }

        # Add system instruction if provided (Gemini API supports systemInstruction field)
        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [{
                    "text": system_prompt
                }]
            }

        client = await self._get_client()
        # 429s have their own budget (rate_limit_events): rotating to a spare
        # key must not consume a transient-retry attempt (issue #217).
        attempt = 0
        rate_limit_events = 0
        while attempt < MAX_TRANSLATION_ATTEMPTS:
            current_key = await self._key_pool.acquire()
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": current_key,
            }
            try:
                response = await client.post(
                    self.api_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )
                response.raise_for_status()

                response_json = response.json()
                # Extract text from Gemini response structure
                response_text = ""
                was_truncated = False
                # When the input prompt itself is blocked, Gemini returns 200
                # with no candidates and promptFeedback.blockReason populated.
                # Surface this so the empty response isn't silently treated as
                # a translation failure.
                prompt_feedback = response_json.get("promptFeedback", {})
                if prompt_feedback.get("blockReason") and not response_json.get("candidates"):
                    print(
                        f"⚠️ Gemini blocked the input prompt "
                        f"(blockReason: {prompt_feedback['blockReason']}). "
                        f"Consider lowering GEMINI_SAFETY_THRESHOLD "
                        f"(current: {GEMINI_SAFETY_THRESHOLD})."
                    )
                if "candidates" in response_json and response_json["candidates"]:
                    candidate = response_json["candidates"][0]
                    content = candidate.get("content", {})
                    parts = content.get("parts", [])
                    if parts:
                        response_text = parts[0].get("text", "")
                    # Detect finishReason: MAX_TOKENS = truncation; SAFETY/RECITATION
                    # = the model produced nothing because the response was blocked
                    # post-generation. We log SAFETY explicitly because a silent
                    # empty content here is the main cause of mixed-language
                    # output in EPUB translations.
                    finish_reason = candidate.get("finishReason", "")
                    if finish_reason == "MAX_TOKENS":
                        was_truncated = True
                        print(f"⚠️ Gemini response was truncated (finishReason: MAX_TOKENS)")
                    elif finish_reason in ("SAFETY", "RECITATION") and not response_text:
                        print(
                            f"⚠️ Gemini returned empty content (finishReason: {finish_reason}). "
                            f"This chunk was blocked post-generation; consider lowering "
                            f"GEMINI_SAFETY_THRESHOLD (current: {GEMINI_SAFETY_THRESHOLD})."
                        )

                # Extract token usage if available
                usage_metadata = response_json.get("usageMetadata", {})
                prompt_tokens = usage_metadata.get("promptTokenCount", 0)
                completion_tokens = usage_metadata.get("candidatesTokenCount", 0)

                return LLMResponse(
                    content=response_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    context_used=prompt_tokens + completion_tokens,
                    context_limit=0,  # Gemini manages context internally
                    was_truncated=was_truncated
                )

            except httpx.TimeoutException as e:
                    print(f"Gemini API Timeout (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                    attempt += 1
                    if attempt < MAX_TRANSLATION_ATTEMPTS:
                        await asyncio.sleep(2)
                        continue
                    return None
            except httpx.HTTPStatusError as e:
                    error_message = str(e)
                    error_body = ""
                    if hasattr(e, 'response') and hasattr(e.response, 'text'):
                        error_body = e.response.text[:500]
                        error_message = f"{e} - {error_body}"

                    # Handle rate limiting (429) — rotate key or sleep, raise if exhausted
                    if e.response.status_code == 429:
                        rate_limit_events += 1
                        await handle_rate_limit(
                            self._key_pool, current_key, e.response.headers,
                            rate_limit_events, MAX_TRANSLATION_ATTEMPTS,
                        )
                        continue

                    print(f"Gemini API HTTP Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                    if error_body:
                        print(f"Response details: Status {e.response.status_code}, Body: {error_body[:200]}...")

                    # Detect context overflow errors (Gemini uses "RESOURCE_EXHAUSTED" or token limits)
                    context_overflow_keywords = ["resource_exhausted", "token limit", "input too long",
                                                  "maximum input", "context length", "too many tokens"]
                    if any(keyword in error_message.lower() for keyword in context_overflow_keywords):
                        raise ContextOverflowError(f"Gemini context overflow: {error_message}")

                    # Client errors (404 model retired, 400/401/403) won't recover
                    # on retry — fail fast instead of hammering the API 3x.
                    if not is_retryable_http_status(e.response.status_code):
                        return None

                    attempt += 1
                    if attempt < MAX_TRANSLATION_ATTEMPTS:
                        await asyncio.sleep(2)
                        continue
                    return None
            except Exception as e:
                    print(f"Gemini API Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                    attempt += 1
                    if attempt < MAX_TRANSLATION_ATTEMPTS:
                        await asyncio.sleep(2)
                        continue
                    return None

        return None
