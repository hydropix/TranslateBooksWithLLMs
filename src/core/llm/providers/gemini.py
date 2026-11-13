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

from src.config import GEMINI_SAFETY_THRESHOLD
from ..base import (
    LLMProvider, LLMRequest, LLMResponse, is_output_limit_finish, provider_error_from_response,
)
from ..policy import RetryPolicy

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
        - gemini-3.6-flash
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

    API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
    DISPLAY_NAME = "Gemini"
    # Gemini may also report an oversized prompt as RESOURCE_EXHAUSTED (a 429
    # with that status is a quota error and is never read as an overflow).
    OVERFLOW_KEYWORDS = ("resource_exhausted",)

    def __init__(self, api_key: Union[str, List[str]], model: str = "gemini-3.6-flash",
                 *, retry_policy: Optional[RetryPolicy] = None,
                 temperature: Optional[float] = None, log_callback=None):
        """
        Initialize the Gemini provider.

        Args:
            api_key: Google AI API key. Accepts a single key string OR a list
                of keys for automatic failover on HTTP 429 (the base class wraps
                them in a KeyPool that rotates on rate-limit).
            model: Gemini model name (default: gemini-3.6-flash)
        """
        super().__init__(model, api_keys=api_key, provider_name="gemini",
                         retry_policy=retry_policy, temperature=temperature,
                         log_callback=log_callback)
        self.api_endpoint = self._endpoint_for(model)

    def _endpoint_for(self, model: str) -> str:
        return f"{self.API_BASE}/{model}:generateContent"

    def _is_thinking_model(self, model: Optional[str] = None) -> bool:
        """Check if the model supports thinking mode (Gemini 2.5+)."""
        return "2.5" in (model or self.model)

    def _get_thinking_config(self, model: Optional[str] = None) -> dict:
        """Return thinkingConfig to disable thinking for supported models."""
        if self._is_thinking_model(model):
            return {"thinkingConfig": {"thinkingBudget": 0}}
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

        models_endpoint = self.API_BASE

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

    async def _send(self, request: LLMRequest) -> LLMResponse:
        """One generateContent call (request loop in ``LLMProvider``)."""
        payload = {
            "contents": [{
                "role": "user",
                "parts": [{
                    "text": request.prompt
                }]
            }],
            "generationConfig": {
                "temperature": self.effective_temperature(),
                **self._get_thinking_config(request.model)
            },
            "safetySettings": [
                {"category": category, "threshold": GEMINI_SAFETY_THRESHOLD}
                for category in _GEMINI_HARM_CATEGORIES
            ],
        }

        # Gemini takes the system prompt as a dedicated systemInstruction field.
        if request.system_prompt:
            payload["systemInstruction"] = {
                "parts": [{
                    "text": request.system_prompt
                }]
            }

        headers = {"Content-Type": "application/json"}
        if request.api_key:
            headers["x-goog-api-key"] = request.api_key

        client = await self._get_client()
        response = await client.post(
            self._endpoint_for(request.model),
            headers=headers,
            json=payload,
            timeout=request.timeout
        )
        if response.status_code >= 400:
            raise provider_error_from_response(response)
        return self._parse_response(response.json())

    def _parse_response(self, response_json: dict) -> LLMResponse:
        response_text = ""
        was_truncated = False
        finish_reason = None
        # When the input prompt itself is blocked, Gemini returns 200 with no
        # candidates and promptFeedback.blockReason populated. Surface this so
        # the empty response isn't silently treated as a translation failure.
        prompt_feedback = response_json.get("promptFeedback", {})
        if prompt_feedback.get("blockReason") and not response_json.get("candidates"):
            self._log("gemini_prompt_blocked",
                      f"⚠️ Gemini blocked the input prompt "
                      f"(blockReason: {prompt_feedback['blockReason']}). "
                      f"Consider lowering GEMINI_SAFETY_THRESHOLD "
                      f"(current: {GEMINI_SAFETY_THRESHOLD}).")
        if "candidates" in response_json and response_json["candidates"]:
            candidate = response_json["candidates"][0]
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            # Gemini can split one answer across several parts, and adds a part
            # flagged "thought" when thinking is enabled. Keep every text part;
            # taking parts[0] truncated long responses.
            response_text = "".join(
                part.get("text", "")
                for part in parts
                if not part.get("thought")
            )
            # finishReason MAX_TOKENS = truncation; SAFETY/RECITATION = the
            # response was blocked post-generation. SAFETY is logged explicitly
            # because a silent empty content here is the main cause of
            # mixed-language output in EPUB translations.
            finish_reason = candidate.get("finishReason", "")
            if finish_reason == "MAX_TOKENS":
                was_truncated = True
                self._log("gemini_max_tokens",
                          "⚠️ Gemini response was truncated (finishReason: MAX_TOKENS)")
            elif finish_reason in ("SAFETY", "RECITATION") and not response_text:
                self._log("gemini_blocked_output",
                          f"⚠️ Gemini returned empty content (finishReason: {finish_reason}). "
                          f"This chunk was blocked post-generation; consider lowering "
                          f"GEMINI_SAFETY_THRESHOLD (current: {GEMINI_SAFETY_THRESHOLD}).")

        usage_metadata = response_json.get("usageMetadata", {})
        prompt_tokens = usage_metadata.get("promptTokenCount", 0)
        completion_tokens = usage_metadata.get("candidatesTokenCount", 0)

        return LLMResponse(
            content=response_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            context_used=prompt_tokens + completion_tokens,
            context_limit=0,  # Gemini manages context internally
            was_truncated=was_truncated,
            finish_reason=finish_reason,
            output_truncated=is_output_limit_finish(finish_reason)
        )
