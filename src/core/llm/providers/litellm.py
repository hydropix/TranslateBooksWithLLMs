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

from collections.abc import Mapping
from typing import List, Optional, Union

from ..base import LLMProvider, LLMRequest, LLMResponse, is_output_limit_finish
from ..exceptions import ProviderError
from ..policy import RetryPolicy
from ..rate_limit_handler import is_retryable_http_status


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

# Raised by LiteLLM when the input does not fit the model; always surfaced as
# ContextOverflowError, even when its message misses every keyword above.
_CONTEXT_WINDOW_EXCEPTION = "litellm.exceptions.ContextWindowExceededError"

# LiteLLM exception qualnames caused by the request or the credentials: a retry
# sends the same thing and fails the same way, so give up on the first one.
# Matched against the whole MRO, so subclasses (ContentPolicyViolationError,
# UnsupportedParamsError, ... all derive from BadRequestError) are covered.
_NON_TRANSIENT_EXCEPTIONS = (
    "litellm.exceptions.AuthenticationError",
    "litellm.exceptions.PermissionDeniedError",
    "litellm.exceptions.NotFoundError",
    "litellm.exceptions.BadRequestError",
    "litellm.exceptions.UnprocessableEntityError",
    "litellm.exceptions.BudgetExceededError",
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
        *,
        retry_policy: Optional[RetryPolicy] = None,
        temperature: Optional[float] = None,
        log_callback=None,
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
        super().__init__(model, api_keys=api_key, provider_name="litellm",
                         retry_policy=retry_policy, temperature=temperature,
                         log_callback=log_callback)
        self.api_base = api_base
        self._litellm = None

    def _build_kwargs(self, api_key: Optional[str] = None) -> dict:
        """Assemble the keyword arguments forwarded to litellm.acompletion.

        Args:
            api_key: The key acquired from the pool for this attempt. None when
                no explicit key was given, in which case LiteLLM falls back to
                the provider's native env var.
        """
        kwargs: dict = {"drop_params": True, "temperature": self.effective_temperature()}
        if api_key:
            kwargs["api_key"] = api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        return kwargs

    async def _before_request(self, model: str) -> None:
        # Import here so a missing optional dependency fails the call loudly
        # instead of being retried as a transient error.
        self._litellm = _import_litellm()

    async def _send(self, request: LLMRequest) -> LLMResponse:
        """One ``litellm.acompletion`` call (request loop in ``LLMProvider``)."""
        litellm = self._litellm or _import_litellm()
        try:
            response = await litellm.acompletion(
                model=request.model,
                messages=request.messages(),
                timeout=request.timeout,
                **self._build_kwargs(request.api_key),
            )
        except Exception as e:
            raise _classify(e) from e

        choice = response.choices[0]
        content = getattr(choice.message, "content", "") or ""
        finish_reason = getattr(choice, "finish_reason", None)

        usage = getattr(response, "usage", None)
        prompt_tokens = (getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        completion_tokens = (getattr(usage, "completion_tokens", 0) or 0) if usage else 0

        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            context_used=prompt_tokens + completion_tokens,
            context_limit=0,  # Unknown across providers; not enforced here.
            was_truncated=False,
            finish_reason=finish_reason,
            output_truncated=is_output_limit_finish(finish_reason),
        )


def _import_litellm():
    try:
        import litellm
    except ImportError as e:
        raise ImportError(
            "LiteLLM provider requires the 'litellm' package. "
            'Install it with: pip install "litellm>=1.65,<1.85"'
        ) from e
    return litellm


def _classify(exc: Exception) -> ProviderError:
    """Map a LiteLLM exception onto the shared error taxonomy."""
    qualnames = _qualnames(exc)
    status = getattr(exc, "status_code", None)
    status = status if isinstance(status, int) else None
    if _CONTEXT_WINDOW_EXCEPTION in qualnames:
        return ProviderError(str(exc), status_code=status, context_overflow=True)
    if qualnames[0] == _RATE_LIMIT_EXCEPTION:
        # A 429 rotates keys when there is a pool; without one it is retried
        # with backoff and pauses the job once the attempts are spent.
        return ProviderError(str(exc), status_code=429, rate_limited=True,
                             headers=_response_headers(exc), retryable=True)
    # Auth, bad request, unknown model, ... won't recover on retry: fail
    # fast, same as the HTTP providers do for non-retryable 4xx.
    return ProviderError(str(exc), status_code=status,
                         retryable=not _is_non_transient(exc, qualnames))


def _qualnames(exc: Exception) -> List[str]:
    """Qualified names of the exception's class and its bases, most specific first.

    Exceptions are classified by name rather than isinstance() because litellm
    is an optional lazy import: its classes cannot be referenced at module load.
    """
    return [f"{cls.__module__}.{cls.__name__}" for cls in type(exc).__mro__]


def _is_non_transient(exc: Exception, qualnames: List[str]) -> bool:
    """Whether retrying the same request is pointless.

    Known transient classes win over everything else. Otherwise a known
    non-transient class, or a non-retryable HTTP status on an exception that
    carries one (LiteLLM sets `status_code` on its errors, including the
    generic APIError), means the request itself is at fault. Anything
    unclassified stays retryable, as before.
    """
    if any(name in _TRANSIENT_EXCEPTIONS for name in qualnames):
        return False
    if any(name in _NON_TRANSIENT_EXCEPTIONS for name in qualnames):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and not is_retryable_http_status(status)


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
