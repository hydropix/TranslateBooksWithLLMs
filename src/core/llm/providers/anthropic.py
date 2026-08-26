"""Anthropic Messages API provider."""

from typing import Callable, List, Optional, Union
import asyncio
import httpx

from src.config import REQUEST_TIMEOUT, MAX_TRANSLATION_ATTEMPTS
from ..base import LLMProvider, LLMResponse
from ..exceptions import ContextOverflowError
from ..rate_limit_handler import handle_rate_limit, is_retryable_http_status


class AnthropicProvider(LLMProvider):
    API_URL = "https://api.anthropic.com/v1/messages"
    MODELS_URL = "https://api.anthropic.com/v1/models"
    API_VERSION = "2023-06-01"
    FALLBACK_MODELS = ["claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5-20251001"]

    def __init__(
        self,
        api_key: Union[str, List[str]],
        model: str = FALLBACK_MODELS[0],
        api_endpoint: Optional[str] = None,
        context_window: Optional[int] = None,
        log_callback: Optional[Callable] = None,
        max_output_tokens: int = 16384,
    ):
        super().__init__(model, api_keys=api_key, provider_name="anthropic")
        self.context_window = context_window or 200000
        self.log_callback = log_callback
        self.max_output_tokens = max(1, int(max_output_tokens))
        endpoint = (api_endpoint or self.API_URL).rstrip("/")
        self.api_endpoint = endpoint if endpoint.endswith("/messages") else endpoint + "/messages"
        base = self.api_endpoint[:-len("/messages")] if self.api_endpoint.endswith("/messages") else self.api_endpoint
        self.models_url = base + "/models"

    async def get_available_models(self) -> list:
        if not self.api_key:
            return [{"id": m, "name": m} for m in self.FALLBACK_MODELS]
        try:
            response = await (await self._get_client()).get(
                self.models_url,
                headers={"x-api-key": self.api_key, "anthropic-version": self.API_VERSION},
                timeout=15,
            )
            response.raise_for_status()
            models = response.json().get("data", [])
            return [{"id": m.get("id", ""), "name": m.get("display_name") or m.get("id", "")} for m in models if m.get("id")]
        except Exception:
            return [{"id": m, "name": m} for m in self.FALLBACK_MODELS]

    async def generate(self, prompt: str, timeout: int = REQUEST_TIMEOUT,
                       system_prompt: Optional[str] = None,
                       **generation_options) -> Optional[LLMResponse]:
        payload = {
            "model": self.model,
            "max_tokens": generation_options.get("max_tokens", self.max_output_tokens),
            "messages": [{"role": "user", "content": prompt}],
        }
        for option in ("temperature", "top_p"):
            value = generation_options.get(option)
            if value is not None:
                payload[option] = value
        if system_prompt:
            payload["system"] = system_prompt
        client = await self._get_client()
        attempt = 0
        rate_limit_events = 0
        while attempt < MAX_TRANSLATION_ATTEMPTS:
            key = await self._key_pool.acquire()
            try:
                response = await client.post(
                    self.api_endpoint,
                    headers={"x-api-key": key, "anthropic-version": self.API_VERSION, "content-type": "application/json"},
                    json=payload,
                    timeout=timeout,
                )
                if response.status_code == 429:
                    rate_limit_events += 1
                    await handle_rate_limit(self._key_pool, key, response.headers, rate_limit_events, MAX_TRANSLATION_ATTEMPTS)
                    continue
                response.raise_for_status()
                body = response.json()
                content = "".join(block.get("text", "") for block in body.get("content", []) if block.get("type") == "text")
                usage = body.get("usage", {})
                prompt_tokens = usage.get("input_tokens", 0)
                completion_tokens = usage.get("output_tokens", 0)
                was_truncated = body.get("stop_reason") == "max_tokens"
                if self.log_callback and (prompt_tokens or completion_tokens):
                    self.log_callback(
                        "token_usage",
                        f"Tokens: prompt={prompt_tokens}, response={completion_tokens}, "
                        f"total={prompt_tokens + completion_tokens}",
                    )
                return LLMResponse(
                    content=content,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    context_used=prompt_tokens + completion_tokens,
                    context_limit=self.context_window,
                    was_truncated=was_truncated,
                )
            except httpx.TimeoutException:
                attempt += 1
            except httpx.HTTPStatusError as exc:
                message = str(exc)
                if any(word in message.lower() for word in ("context", "token limit", "too long")):
                    raise ContextOverflowError(message) from exc
                if exc.response is not None and not is_retryable_http_status(exc.response.status_code):
                    return None
                attempt += 1
            except (ValueError, httpx.DecodingError):
                attempt += 1
            if attempt < MAX_TRANSLATION_ATTEMPTS:
                await asyncio.sleep(2)
        return None
