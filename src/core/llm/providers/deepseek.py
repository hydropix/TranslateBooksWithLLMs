"""
DeepSeek LLM Provider.

This module provides the DeepSeekProvider class for interacting with
DeepSeek's API, which offers cost-effective models with strong capabilities.

Features:
    - DeepSeek V3 models (chat and reasoner)
    - OpenAI-compatible API format
    - Thinking mode support (deepseek-reasoner)
    - Cost-effective pricing (~5-10x cheaper than OpenAI)
    - 64K context window
"""

from typing import Optional
import httpx
import asyncio
import json

from src.config import REQUEST_TIMEOUT, MAX_TRANSLATION_ATTEMPTS
from ..base import LLMProvider, LLMResponse
from ..exceptions import ContextOverflowError


class DeepSeekProvider(LLMProvider):
    """
    Provider for DeepSeek API.

    DeepSeek provides powerful language models with excellent price/performance:
        - deepseek-chat: Non-thinking mode (fast, economical)
        - deepseek-reasoner: Thinking mode with <think> tags

    Configuration:
        endpoint: https://api.deepseek.com/chat/completions
        model: Model identifier (e.g., "deepseek-chat")
        api_key: DeepSeek API key

    Example:
        >>> provider = DeepSeekProvider(
        ...     api_key="your-api-key",
        ...     model="deepseek-chat"
        ... )
        >>> response = await provider.generate("Translate: Hello")
    """

    API_URL = "https://api.deepseek.com/chat/completions"
    MODELS_URL = "https://api.deepseek.com/models"

    MODEL_CONTEXT_SIZES = {
        "deepseek-chat": 64000,
        "deepseek-reasoner": 64000,
        "deepseek-coder": 16000,
    }

    FALLBACK_MODELS = [
        "deepseek-chat",
        "deepseek-reasoner",
    ]

    THINKING_MODELS = ["deepseek-reasoner", "deepseek-r1"]

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        api_endpoint: Optional[str] = None
    ):
        """
        Initialize the DeepSeek provider.

        Args:
            api_key: DeepSeek API key
            model: Model identifier (default: deepseek-chat)
            api_endpoint: Optional custom API endpoint
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
        return 64000  # Default for DeepSeek

    def _is_thinking_model(self) -> bool:
        """Check if the current model uses thinking mode."""
        return any(tm in self.model.lower() for tm in self.THINKING_MODELS)

    async def get_available_models(self) -> list:
        """
        Fetch available DeepSeek models from API.

        Returns:
            List of model dicts with id, name, and context_length
        """
        if not self.api_key:
            return self._get_fallback_models()

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json"
            }
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
                if "deepseek" in model_id.lower():
                    context_length = model.get("max_context_length")
                    if not context_length:
                        for prefix, size in self.MODEL_CONTEXT_SIZES.items():
                            if prefix in model_id.lower():
                                context_length = size
                                break
                        if not context_length:
                            context_length = 64000

                    filtered_models.append({
                        "id": model_id,
                        "name": model_id,
                        "context_length": context_length
                    })

            filtered_models.sort(key=lambda x: x["name"])

            if len(filtered_models) < 1:
                return self._get_fallback_models()

            return filtered_models

        except Exception as e:
            print(f"⚠️ Failed to fetch DeepSeek models: {e}")
            return self._get_fallback_models()

    def _get_fallback_models(self) -> list:
        """Return fallback models list when API fetch fails."""
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
        return 64000

    async def generate(
        self,
        prompt: str,
        timeout: int = REQUEST_TIMEOUT,
        system_prompt: Optional[str] = None
    ) -> Optional[LLMResponse]:
        """
        Generate text using DeepSeek API.

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

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "stream": False
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

                if response.status_code == 401:
                    raise ValueError("Invalid DeepSeek API key")

                if response.status_code == 429:
                    wait_time = 2 ** attempt
                    print(f"⚠️ DeepSeek rate limited, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue

                response.raise_for_status()
                result = response.json()

                if "choices" not in result or len(result["choices"]) == 0:
                    print(f"⚠️ DeepSeek: Unexpected response format: {result}")
                    return None

                response_text = result["choices"][0].get("message", {}).get("content", "")

                usage = result.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                print(f"💬 DeepSeek: {prompt_tokens}+{completion_tokens} tokens")

                return LLMResponse(
                    content=response_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    context_used=prompt_tokens + completion_tokens,
                    context_limit=self._get_context_limit(),
                    was_truncated=False
                )

            except httpx.TimeoutException as e:
                print(f"DeepSeek API Timeout (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
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
                    print(f"❌ DeepSeek: Model '{self.model}' not found!")
                    print(f"   Check available models at https://platform.deepseek.com/")
                elif e.response.status_code == 401:
                    print(f"❌ DeepSeek: Invalid API key!")
                elif e.response.status_code == 402:
                    print(f"❌ DeepSeek: Insufficient credits!")
                else:
                    print(f"DeepSeek API HTTP Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                    print(f"Response details: Status {e.response.status_code}, Body: {error_body}...")

                context_overflow_keywords = [
                    "context_length", "maximum context", "token limit",
                    "too many tokens", "reduce the length", "max_tokens",
                    "context window", "exceeds"
                ]
                if any(keyword in error_message.lower() for keyword in context_overflow_keywords):
                    raise ContextOverflowError(f"DeepSeek context overflow: {error_message}")

                if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                    await asyncio.sleep(2)
                    continue
                return None

            except json.JSONDecodeError as e:
                print(f"DeepSeek API JSON Decode Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                    await asyncio.sleep(2)
                    continue
                return None

            except Exception as e:
                print(f"DeepSeek API Unknown Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                    await asyncio.sleep(2)
                    continue
                return None

        return None
