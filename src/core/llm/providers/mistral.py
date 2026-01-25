"""
Mistral AI LLM Provider.

This module provides the MistralProvider class for interacting with
Mistral's API, which offers various models with excellent multilingual capabilities.

Features:
    - Access to Mistral models (large, medium, small, codestral)
    - OpenAI-compatible API format
    - Automatic context size detection
    - Rate limiting with exponential backoff
"""

from typing import Optional
import httpx
import asyncio
import json

from src.config import REQUEST_TIMEOUT, MAX_TRANSLATION_ATTEMPTS
from ..base import LLMProvider, LLMResponse
from ..exceptions import ContextOverflowError


class MistralProvider(LLMProvider):
    """
    Provider for Mistral AI API.

    Mistral AI provides powerful language models with strong multilingual support:
        - mistral-large-latest: Most capable, 128K context
        - mistral-medium-latest: Balanced performance, 128K context
        - mistral-small-latest: Fast and economical, 32K context
        - codestral-latest: Optimized for code, 32K context
        - ministral-8b-latest: Small efficient model, 128K context

    Configuration:
        endpoint: https://api.mistral.ai/v1/chat/completions
        model: Model identifier (e.g., "mistral-large-latest")
        api_key: Mistral API key

    Example:
        >>> provider = MistralProvider(
        ...     api_key="your-api-key",
        ...     model="mistral-large-latest"
        ... )
        >>> response = await provider.generate("Translate: Hello")
    """

    # Mistral API endpoints
    API_URL = "https://api.mistral.ai/v1/chat/completions"
    MODELS_URL = "https://api.mistral.ai/v1/models"

    # Context sizes by model family
    MODEL_CONTEXT_SIZES = {
        "mistral-large": 128000,
        "mistral-medium": 128000,
        "mistral-small": 32000,
        "codestral": 32000,
        "ministral": 128000,
        "pixtral": 128000,
        "open-mistral": 32000,
        "open-mixtral": 32000,
    }

    # Fallback models (sorted by capability)
    FALLBACK_MODELS = [
        "mistral-large-latest",
        "mistral-medium-latest",
        "mistral-small-latest",
        "codestral-latest",
        "ministral-8b-latest",
        "ministral-3b-latest",
    ]

    def __init__(
        self,
        api_key: str,
        model: str = "mistral-large-latest",
        api_endpoint: Optional[str] = None
    ):
        """
        Initialize the Mistral provider.

        Args:
            api_key: Mistral API key
            model: Model identifier (default: mistral-large-latest)
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
        return 32000  # Default for unknown models

    async def get_available_models(self) -> list:
        """
        Fetch available Mistral models from API.

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
                # Include chat/completion models
                if any(x in model_id.lower() for x in ['mistral', 'codestral', 'ministral', 'pixtral']):
                    # Get context length from model info or estimate from ID
                    context_length = model.get("max_context_length")
                    if not context_length:
                        # Estimate based on model family
                        for prefix, size in self.MODEL_CONTEXT_SIZES.items():
                            if prefix in model_id.lower():
                                context_length = size
                                break
                        if not context_length:
                            context_length = 32000

                    filtered_models.append({
                        "id": model_id,
                        "name": model_id,
                        "context_length": context_length
                    })

            # Sort by name
            filtered_models.sort(key=lambda x: x["name"])

            if len(filtered_models) < 1:
                return self._get_fallback_models()

            return filtered_models

        except Exception as e:
            print(f"⚠️ Failed to fetch Mistral models: {e}")
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
        return 32000

    async def generate(
        self,
        prompt: str,
        timeout: int = REQUEST_TIMEOUT,
        system_prompt: Optional[str] = None
    ) -> Optional[LLMResponse]:
        """
        Generate text using Mistral API.

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

                # Handle specific error codes
                if response.status_code == 401:
                    raise ValueError("Invalid Mistral API key")

                if response.status_code == 429:
                    # Rate limited - exponential backoff
                    wait_time = 2 ** attempt
                    print(f"⚠️ Mistral rate limited, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue

                response.raise_for_status()
                result = response.json()

                if "choices" not in result or len(result["choices"]) == 0:
                    print(f"⚠️ Mistral: Unexpected response format: {result}")
                    return None

                response_text = result["choices"][0].get("message", {}).get("content", "")

                # Extract token usage
                usage = result.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                print(f"💬 Mistral: {prompt_tokens}+{completion_tokens} tokens")

                return LLMResponse(
                    content=response_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    context_used=prompt_tokens + completion_tokens,
                    context_limit=self._get_context_limit(),
                    was_truncated=False
                )

            except httpx.TimeoutException as e:
                print(f"Mistral API Timeout (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
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

                # Handle specific error codes
                if e.response.status_code == 404:
                    print(f"❌ Mistral: Model '{self.model}' not found!")
                    print(f"   Check available models at https://docs.mistral.ai/")
                elif e.response.status_code == 401:
                    print(f"❌ Mistral: Invalid API key!")
                elif e.response.status_code == 402:
                    print(f"❌ Mistral: Insufficient credits!")
                else:
                    print(f"Mistral API HTTP Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                    print(f"Response details: Status {e.response.status_code}, Body: {error_body}...")

                # Detect context overflow errors
                context_overflow_keywords = [
                    "context_length", "maximum context", "token limit",
                    "too many tokens", "reduce the length", "max_tokens",
                    "context window", "exceeds"
                ]
                if any(keyword in error_message.lower() for keyword in context_overflow_keywords):
                    raise ContextOverflowError(f"Mistral context overflow: {error_message}")

                if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                    await asyncio.sleep(2)
                    continue
                return None

            except json.JSONDecodeError as e:
                print(f"Mistral API JSON Decode Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                    await asyncio.sleep(2)
                    continue
                return None

            except Exception as e:
                print(f"Mistral API Unknown Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                    await asyncio.sleep(2)
                    continue
                return None

        return None
