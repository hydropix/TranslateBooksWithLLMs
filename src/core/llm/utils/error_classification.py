"""
Classification of provider error messages.

Providers report a prompt that does not fit the model's context window as a
plain HTTP error, so the only signal is the error text. Raising
ContextOverflowError makes the pipeline split the chunk and retry, which is
wasted (and billed) work when the error is really about credentials, credits
or rate limits. This module holds the one keyword list every provider uses.
"""

from typing import Iterable, Optional

# Phrases that, in practice, only appear in context-overflow errors
# (OpenAI, vLLM, llama.cpp, LM Studio, Ollama, Mistral, DeepSeek, Gemini,
# OpenRouter, Anthropic-compatible gateways). Deliberately no bare
# "context", "length", "max_tokens", "exceeds" or "truncate": those also
# appear in auth, quota and parameter-validation errors.
CONTEXT_OVERFLOW_KEYWORDS = (
    "context_length",
    "context length",
    "maximum context",
    "context window",
    "context size",
    "exceed_context",
    "token limit",
    "too many tokens",
    "reduce the length",
    "input too long",
    "input is too long",
    "prompt too long",
    "prompt is too long",
    "maximum input",
    "input token count",
    "too large for model",
)

# Statuses that are never a context overflow, whatever the body says:
# auth (401/403), billing (402) and rate limiting (429).
NEVER_OVERFLOW_STATUSES = frozenset({401, 402, 403, 429})


def is_context_overflow_error(
    message: Optional[str],
    status_code: Optional[int] = None,
    extra_keywords: Iterable[str] = (),
) -> bool:
    """Whether an error means the prompt exceeded the model's context window.

    Args:
        message: Error text (exception string and/or response body).
        status_code: HTTP status when known. 401/402/403/429 always return
            False, so an auth, credit or rate-limit error is never mistaken
            for an overflow (which would trigger repeated chunk splitting).
        extra_keywords: Provider-specific phrases, checked like the shared ones.
    """
    if status_code in NEVER_OVERFLOW_STATUSES:
        return False
    text = str(message or "").lower()
    if not text:
        return False
    return any(keyword in text for keyword in CONTEXT_OVERFLOW_KEYWORDS) or any(
        keyword.lower() in text for keyword in extra_keywords
    )
