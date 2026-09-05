"""
LiteLLM provider implementation.

Routes to 100+ LLM providers (OpenAI, Anthropic, Gemini, Bedrock, Vertex AI,
Groq, etc.) through a single unified interface using provider-prefixed model
names. LiteLLM is a client-side library, not a hosted gateway: it runs in this
process and dispatches each call to the underlying provider's native API.

API keys are read from each provider's native environment variable
(OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, ...) unless an explicit
key is passed. This is why the factory does not plumb a single LiteLLM key.

Optional dependency: pip install "litellm>=1.65,<1.85"

Example:
    >>> provider = LiteLLMProvider(model="anthropic/claude-sonnet-4-6")
    >>> response = await provider.generate("Translate: Hello")
"""

import asyncio
from collections.abc import Mapping
from typing import Optional, Union, List

from src.config import REQUEST_TIMEOUT, MAX_TRANSLATION_ATTEMPTS, TEMPERATURE
from ..base import LLMProvider, LLMResponse
from ..exceptions import ContextOverflowError
from ..rate_limit_handler import handle_rate_limit


# Substrings that mark a provider-side context/length overflow, surfaced as
# ContextOverflowError so the chunking layer can react (shrink and retry).
_CONTEXT_OVERFLOW_KEYWORDS = (
    "context_length", "maximum context", "token limit",
    "too many tokens", "reduce the length", "max_tokens",
    "context window", "exceeds", "context window exceeded",
)

# Raised by LiteLLM on a provider 429; routed to the key pool when there is one.
_RATE_LIMIT_EXCEPTION = "litellm.exceptions.RateLimitError"

# LiteLLM exception qualnames worth retrying with backoff (transient).
_TRANSIENT_EXCEPTIONS = (
    _RATE_LIMIT_EXCEPTION,
    "litellm.exceptions.APIConnectionError",
    "litellm.exceptions.Timeout",
    "litellm.exceptions.InternalServerError",
    "litellm.exceptions.ServiceUnavailableError",
)


class LiteLLMProvider(LLMProvider):
    """
    Provider that uses LiteLLM to reach 100+ LLM providers.

    Model names carry a provider prefix that selects the route AND the
    expected credential env var:
        - "openai/gpt-4o"            -> OPENAI_API_KEY
        - "anthropic/claude-sonnet-4-6" -> ANTHROPIC_API_KEY
        - "gemini/gemini-2.5-flash"  -> GEMINI_API_KEY
        - "bedrock/anthropic.claude-v2" -> AWS_* env vars

    LiteLLM handles authentication and parameter translation itself, so this
    wrapper stays thin. A KeyPool is used only when explicit keys are passed:
    each attempt acquires a key from it and a provider 429 rotates to the next
    one instead of consuming a transient-retry attempt.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[Union[str, List[str]]] = None,
        api_base: Optional[str] = None,
    ):
        """
        Initialize the LiteLLM provider.

        Args:
            model: Provider-prefixed model name (e.g. "anthropic/claude-sonnet-4-6").
            api_key: Optional explicit key. When omitted, LiteLLM reads the
                provider's native env var. A KeyPool is created only when a key
                is given so the inherited `api_key` property stays consistent.
            api_base: Optional custom endpoint (self-hosted gateways, proxies).
        """
        super().__init__(model, api_keys=api_key, provider_name="litellm")
        self.api_base = api_base

    def _build_kwargs(self, api_key: Optional[str] = None) -> dict:
        """Assemble the keyword arguments forwarded to litellm.acompletion.

        Args:
            api_key: The key acquired from the pool for this attempt. None when
                no explicit key was given, in which case LiteLLM falls back to
                the provider's native env var.
        """
        kwargs: dict = {"drop_params": True, "temperature": TEMPERATURE}
        if api_key:
            kwargs["api_key"] = api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        return kwargs

    async def generate(
        self,
        prompt: str,
        timeout: int = REQUEST_TIMEOUT,
        system_prompt: Optional[str] = None,
    ) -> Optional[LLMResponse]:
        """
        Generate text via LiteLLM.

        Args:
            prompt: The user prompt (content to translate).
            timeout: Request timeout in seconds.
            system_prompt: Optional system prompt (role/instructions).

        Returns:
            LLMResponse with content and token usage, or None on failure.

        Raises:
            ImportError: If the optional `litellm` package is not installed.
            ContextOverflowError: If the input exceeds the model's context window.
        """
        try:
            import litellm
        except ImportError as e:
            raise ImportError(
                "LiteLLM provider requires the 'litellm' package. "
                'Install it with: pip install "litellm>=1.65,<1.85"'
            ) from e

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # 429s have their own budget: rotating to a spare key must never consume
        # a transient-retry attempt (same contract as the other cloud providers).
        attempt = 0
        rate_limit_events = 0
        while attempt < MAX_TRANSLATION_ATTEMPTS:
            current_key = await self._key_pool.acquire() if self._key_pool else None
            try:
                response = await litellm.acompletion(
                    model=self.model,
                    messages=messages,
                    timeout=timeout,
                    **self._build_kwargs(current_key),
                )

                choice = response.choices[0]
                content = getattr(choice.message, "content", "") or ""

                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

                # Plain ASCII: an emoji here crashes on Windows cp1252 consoles,
                # and a failing print would be caught below as a request error.
                print(f"[LiteLLM] ({self.model}): {prompt_tokens}+{completion_tokens} tokens")

                return LLMResponse(
                    content=content,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    context_used=prompt_tokens + completion_tokens,
                    context_limit=0,  # Unknown across providers; not enforced here.
                    was_truncated=False,
                )

            except Exception as e:
                error_str = str(e).lower()
                if any(kw in error_str for kw in _CONTEXT_OVERFLOW_KEYWORDS):
                    raise ContextOverflowError(f"LiteLLM context overflow: {e}") from e

                qualname = f"{type(e).__module__}.{type(e).__name__}"

                # With a multi-key pool, a 429 rotates to the next key instead of
                # burning an attempt. handle_rate_limit() sleeps when every key is
                # throttled and raises RateLimitError once the budget is spent,
                # which pauses the job upstream.
                if qualname == _RATE_LIMIT_EXCEPTION and current_key:
                    rate_limit_events += 1
                    await handle_rate_limit(
                        self._key_pool,
                        current_key,
                        _response_headers(e),
                        rate_limit_events,
                        MAX_TRANSLATION_ATTEMPTS,
                    )
                    continue

                attempt += 1
                print(
                    f"[LiteLLM] Error (attempt {attempt}/"
                    f"{MAX_TRANSLATION_ATTEMPTS}): {e}"
                )
                if attempt >= MAX_TRANSLATION_ATTEMPTS:
                    return None

                if qualname in _TRANSIENT_EXCEPTIONS:
                    await asyncio.sleep(min(2 ** attempt, 10))
                else:
                    await asyncio.sleep(2)

        return None


def _response_headers(exc: Exception) -> Mapping:
    """Best-effort extraction of the response headers from a LiteLLM error.

    LiteLLM wraps each provider's native error, so the underlying response is
    exposed as `.response.headers` on some routes, as `.headers` on others, and
    not at all on the rest. An empty mapping makes compute_wait_time() fall back
    to exponential backoff.
    """
    for holder in (getattr(exc, "response", None), exc):
        headers = getattr(holder, "headers", None)
        if isinstance(headers, Mapping):
            return headers
    return {}
