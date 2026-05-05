"""
LiteLLM provider implementation.

Routes to 100+ LLM providers (OpenAI, Anthropic, Gemini, Bedrock, Vertex AI,
Groq, etc.) via a unified interface using provider-prefixed model names.

Install: pip install litellm

Example:
    >>> provider = LiteLLMProvider(model="anthropic/claude-sonnet-4-6")
    >>> response = await provider.generate("Translate: Hello")
"""

from typing import Optional

from src.config import REQUEST_TIMEOUT, MAX_TRANSLATION_ATTEMPTS
from ..base import LLMProvider, LLMResponse
from ..exceptions import ContextOverflowError


class LiteLLMProvider(LLMProvider):
    """
    Provider that uses LiteLLM to access 100+ LLM providers.

    Uses provider-prefixed model names for routing:
        - "openai/gpt-4o"
        - "anthropic/claude-sonnet-4-6"
        - "gemini/gemini-2.5-flash"
        - "bedrock/anthropic.claude-v2"

    API keys are read from provider-specific env vars (OPENAI_API_KEY,
    ANTHROPIC_API_KEY, etc.) or passed explicitly.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        super().__init__(model)
        self.api_key = api_key
        self.api_base = api_base

    def _build_kwargs(self) -> dict:
        kwargs: dict = {"drop_params": True}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        return kwargs

    async def generate(
        self,
        prompt: str,
        timeout: int = REQUEST_TIMEOUT,
        system_prompt: Optional[str] = None,
    ) -> Optional[LLMResponse]:
        import litellm

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = self._build_kwargs()

        for attempt in range(MAX_TRANSLATION_ATTEMPTS):
            try:
                response = await litellm.acompletion(
                    model=self.model,
                    messages=messages,
                    timeout=timeout,
                    **kwargs,
                )

                choice = response.choices[0]
                content = getattr(choice.message, "content", "") or ""

                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

                return LLMResponse(
                    content=content,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    context_used=prompt_tokens + completion_tokens,
                    context_limit=0,
                    was_truncated=False,
                )

            except Exception as e:
                error_str = str(e).lower()
                context_keywords = [
                    "context_length", "maximum context", "token limit",
                    "too many tokens", "reduce the length", "max_tokens",
                    "context window", "exceeds",
                ]
                if any(kw in error_str for kw in context_keywords):
                    raise ContextOverflowError(
                        f"LiteLLM context overflow: {e}"
                    ) from e

                qualname = f"{type(e).__module__}.{type(e).__name__}"
                transient = {
                    "litellm.exceptions.RateLimitError",
                    "litellm.exceptions.APIConnectionError",
                    "litellm.exceptions.Timeout",
                    "litellm.exceptions.InternalServerError",
                    "litellm.exceptions.ServiceUnavailableError",
                }
                if qualname in transient and attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                    import asyncio
                    await asyncio.sleep(min(2 ** (attempt + 1), 10))
                    continue

                print(f"[LiteLLM] Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                    import asyncio
                    await asyncio.sleep(2)
                    continue
                return None

        return None
