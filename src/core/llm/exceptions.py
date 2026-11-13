"""
LLM-specific exceptions.

This module defines all custom exceptions used in the LLM provider system.
"""

from typing import Any, Mapping, Optional


class ContextOverflowError(Exception):
    """
    Raised when the input text exceeds the model's context window.

    This typically occurs when a chunk is too large for the model to process
    in a single request.
    """
    pass


class RepetitionLoopError(Exception):
    """
    Raised when the model enters a repetition loop.

    This can occur with "thinking" models that get stuck repeating the same
    phrase or pattern, indicating the model has likely exceeded its effective
    context window or encountered an issue.
    """
    pass


class RateLimitError(Exception):
    """
    Raised when the API returns HTTP 429 (Too Many Requests) and all retry
    attempts with backoff have been exhausted.

    This signals the translation pipeline to auto-pause and save a checkpoint
    so the user can resume later.

    Attributes:
        retry_after: Suggested wait time in seconds (from Retry-After header),
                     or None if not provided by the API.
        provider: Name of the LLM provider that was rate-limited.
        partial_result: Best-effort, already-reassembled payload carrying the
                        work completed before the rate limit hit, so the caller
                        can persist it instead of discarding it. Its type is
                        defined by whoever raises (or re-raises) the error;
                        None when the raiser has nothing to hand over.
    """

    def __init__(self, message: str, retry_after: int = None, provider: str = None,
                 partial_result: Any = None):
        super().__init__(message)
        self.retry_after = retry_after
        self.provider = provider
        self.partial_result = partial_result


class LLMClientConfigError(ValueError):
    """
    Raised when an LLM client cannot be built from the given configuration:
    unknown provider name, or a cloud provider with no usable API key
    (neither passed in nor found in the environment).

    Subclasses ValueError, which the provider factory already raises for the
    same conditions, so existing `except ValueError` handlers keep working.
    """
    pass


class ProviderError(Exception):
    """One failed provider attempt, classified for the shared request loop.

    Providers raise it from their transport/parse hooks; ``LLMProvider``
    decides what to do with it (retry, rotate key, raise a pipeline error or
    give up). Never escapes ``LLMProvider.generate``.

    Attributes:
        status_code: HTTP status when the failure came from an HTTP answer.
        error_code: Provider error code/type when the body carries one
            (e.g. OpenAI ``context_length_exceeded``).
        headers: Response headers (rate-limit hints), empty when none.
        retryable: Whether the same request may succeed on another attempt.
            ``None`` lets the loop decide from ``status_code``.
        context_overflow: ``True`` when the provider identified the error as
            a context-window overflow by status or error code.
        rate_limited: ``True`` for a rate-limit answer that carries no HTTP
            429 status (SDK-level errors).
        body: Short excerpt of the response body, for logs.
        transport: ``True`` when no API answer was read (timeout, connection
            error, malformed body): never classified as a context overflow.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
        headers: Optional[Mapping[str, str]] = None,
        retryable: Optional[bool] = None,
        context_overflow: bool = False,
        rate_limited: bool = False,
        body: str = "",
        transport: bool = False,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.headers = headers or {}
        self.retryable = retryable
        self.context_overflow = context_overflow
        self.rate_limited = rate_limited or status_code == 429
        self.body = body
        self.transport = transport
