"""
Base classes and data structures for LLM providers.

``LLMProvider`` owns the whole request loop shared by every provider:

    * per-request timeout and transient-attempt counting (``RetryPolicy``),
      with exponential backoff and jitter between attempts;
    * HTTP 429 handling through the ``KeyPool`` / ``handle_rate_limit`` pair
      (separate budget, key rotation without consuming attempts);
    * error classification: HTTP status and provider error code first, the
      shared overflow keyword list only as a last resort and never for
      401/402/403/429; fail fast on non-retryable 4xx;
    * propagation of the pipeline errors (``RateLimitError``,
      ``ContextOverflowError``, ``RepetitionLoopError``);
    * token-usage reporting to the per-job ``CostTracker``;
    * logging (the provider's ``log_callback`` when it has one, the
      ``llm.<provider>`` logger otherwise).

A concrete provider only implements ``_send(request)``: build the request,
perform it (streaming if it wants), parse the answer into an ``LLMResponse``,
and raise ``ProviderError`` (or let an ``httpx`` exception escape) on failure.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import httpx

from src import __version__
from src.config import TRANSLATE_TAG_IN, TRANSLATE_TAG_OUT
from src.core.llm.cost_tracker import CostTracker, get_current_cost_tracker
from src.core.llm.exceptions import (
    ContextOverflowError,
    ProviderError,
    RateLimitError,
    RepetitionLoopError,
)
from src.core.llm.key_pool import KeyPool
from src.core.llm.policy import RetryPolicy, current_retry_policy
from src.core.llm.rate_limit_handler import (
    compute_wait_time,
    handle_rate_limit,
    is_retryable_http_status,
)
from src.core.llm.utils.error_classification import (
    NEVER_OVERFLOW_STATUSES,
    is_context_overflow_error,
)
from src.core.llm.utils.extraction import TranslationExtractor


def normalize_api_keys(raw: Optional[Union[str, Iterable[str]]]) -> List[str]:
    """Split comma/newline-separated key strings into a clean list.

    Accepts a single key, a "k1,k2,k3" string (the documented multi-key
    format used by .env, the Web UI input, and the CLI), or an iterable of
    keys. Whitespace and empty fragments are trimmed; order is preserved
    for round-robin rotation.

    Returns an empty list when no usable key is provided.
    """
    if raw is None:
        return []
    if not isinstance(raw, str):
        return [k for k in raw if k]
    if "," not in raw and "\n" not in raw:
        return [raw] if raw else []
    parts = [p.strip() for p in raw.replace("\n", ",").split(",")]
    return [p for p in parts if p]


@dataclass
class LLMResponse:
    """Response from LLM with token usage information"""
    content: str
    prompt_tokens: int = 0  # Number of tokens in the prompt
    completion_tokens: int = 0  # Number of tokens in the response
    context_used: int = 0  # Total context used (prompt + completion)
    context_limit: int = 0  # Context limit that was set for this request
    was_truncated: bool = False  # True if response was truncated due to context limit
    was_fallback: bool = False  # True if raw response was used because tag extraction failed
    finish_reason: Optional[str] = None  # Stop reason reported by the provider, when it gives one
    output_truncated: bool = False  # True if the provider says generation hit the output token limit
    cost: Optional[float] = None  # Billed USD cost, when the provider reports one
    cost_estimated: bool = False  # True when `cost` comes from a fallback rate


def is_output_limit_finish(finish_reason: Optional[str]) -> bool:
    """True when a provider stop reason means "ran out of output tokens".

    Covers the OpenAI-compatible "length", Gemini's "MAX_TOKENS" and the
    "max_tokens" / "max_output_tokens" spellings used by other gateways.
    """
    return str(finish_reason or "").strip().lower() in {
        "length", "max_tokens", "max_output_tokens",
    }


@dataclass
class LLMRequest:
    """Everything a provider needs to perform one attempt."""
    prompt: str
    system_prompt: Optional[str]
    model: str
    api_key: Optional[str]
    timeout: float
    context_window: Optional[int] = None
    attempt: int = 0
    #: Scratch space shared by every attempt of one ``generate`` call.
    state: Dict[str, Any] = field(default_factory=dict)

    def messages(self) -> List[Dict[str, str]]:
        """Chat-style messages: optional system prompt, then the user prompt."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.prompt})
        return messages


def provider_error_from_response(response: Any, message: Optional[str] = None) -> ProviderError:
    """Classify an HTTP error answer as a ``ProviderError``.

    Reads the error message and code from the usual JSON shapes
    (``{"error": {"message", "code", "type"}}``, ``{"error": "..."}``,
    ``{"message": ...}``) and falls back to the raw body.
    """
    status = getattr(response, "status_code", None)
    headers: Mapping[str, str] = getattr(response, "headers", None) or {}
    message = message or f"HTTP {status}"
    body = ""
    code = None
    try:
        body = (response.text or "")[:500]
    except Exception:
        body = ""
    try:
        data = response.json()
    except Exception:
        data = None
    if isinstance(data, dict):
        err = data.get("error", data)
        if isinstance(err, dict):
            message = str(err.get("message") or err.get("error") or message)
            raw_code = err.get("code") or err.get("type") or err.get("status")
            code = str(raw_code) if raw_code is not None else None
        elif err:
            message = str(err)
    elif body:
        message = f"{message} - {body}"
    return ProviderError(
        message, status_code=status, error_code=code, headers=headers, body=body,
    )


def http_error_to_provider_error(error: httpx.HTTPStatusError) -> ProviderError:
    """Classify an ``httpx.HTTPStatusError`` (see ``provider_error_from_response``)."""
    if error.response is None:
        return ProviderError(str(error))
    return provider_error_from_response(error.response, str(error))


# Provider error codes that name a context-window overflow outright.
CONTEXT_OVERFLOW_ERROR_CODES = frozenset({
    "context_length_exceeded",
    "string_above_max_length",
    "context_window_exceeded",
})

# Headers through which a rate-limited answer says how long to wait.
RATE_LIMIT_HINT_HEADERS = ("Retry-After", "retry-after", "X-RateLimit-Reset", "x-ratelimit-reset")

# Upper bound, in seconds, for one wait taken from a server hint when there is
# no key pool to rotate. A larger hint (RFC 6585's own example is 3600 s) is
# not slept in full: the request is retried after this cap, and once its
# attempts are spent the RateLimitError carries the full hint so the pipeline
# can pause instead of stalling a worker.
MAX_HINTED_RETRY_WAIT = 60


def _has_rate_limit_hint(headers: Mapping[str, str]) -> bool:
    return any(h in headers for h in RATE_LIMIT_HINT_HEADERS)


class LLMProvider(ABC):
    """Abstract base class for LLM providers (owns the shared request loop)."""

    #: Provider-specific phrases that also mean "prompt too large".
    OVERFLOW_KEYWORDS: Tuple[str, ...] = ()
    #: Human-readable name used in logs.
    DISPLAY_NAME: str = ""

    def __init__(
        self,
        model: str,
        api_keys: Optional[Union[str, Iterable[str]]] = None,
        provider_name: str = "",
        *,
        retry_policy: Optional[RetryPolicy] = None,
        temperature: Optional[float] = None,
        log_callback: Optional[Callable] = None,
    ):
        """
        Initialize the LLM provider.

        Args:
            model: Model name/identifier (default for every request).
            api_keys: A single API key or an iterable of keys. When given,
                wrapped in a KeyPool that supports rotation on HTTP 429.
                Pass None for providers that don't use API keys (e.g. Ollama).
            provider_name: Logical name used for log messages and rate-limit
                error reports. Defaults to the empty string ("unknown" in logs).
            retry_policy: Retry/timeout policy. None resolves the active
                job's policy (or the config defaults) at request time.
            temperature: Sampling temperature. None reads ``TEMPERATURE``
                from the config at request time.
            log_callback: Optional ``(event, message)`` logging callback.
        """
        self.model = model
        self.provider_name = provider_name or "unknown"
        self.retry_policy = retry_policy
        self.temperature = temperature
        self.log_callback = log_callback
        self._extractor = TranslationExtractor(TRANSLATE_TAG_IN, TRANSLATE_TAG_OUT)
        self._client = None
        self._key_pool: Optional[KeyPool] = None
        keys_iter = normalize_api_keys(api_keys)
        if keys_iter:
            self._key_pool = KeyPool(keys_iter, provider_name=self.provider_name)
        # Totals of this instance; the per-job tracker (if any) gets them too.
        self._cost_tracker = CostTracker()
        self._logger = logging.getLogger(f"llm.{self.provider_name}")

    # ------------------------------------------------------------------ keys
    @property
    def api_key(self) -> Optional[str]:
        """The current 'primary' API key (first non-throttled if a pool is used).

        Kept for code that reads the key directly (listing models, context
        detection). Requests always go through the pool so rotation works.
        """
        return self._key_pool.peek() if self._key_pool else None

    # ------------------------------------------------------------ settings
    def effective_retry_policy(self) -> RetryPolicy:
        """This provider's policy, else the active job's, else the config's."""
        return self.retry_policy or current_retry_policy()

    def effective_temperature(self) -> float:
        """The sampling temperature to send (config ``TEMPERATURE`` by default)."""
        if self.temperature is not None:
            return self.temperature
        from src import config
        return config.TEMPERATURE

    @property
    def display_name(self) -> str:
        return self.DISPLAY_NAME or self.provider_name

    # --------------------------------------------------------------- client
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create a persistent HTTP client with connection pooling"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                timeout=httpx.Timeout(self.effective_retry_policy().timeout),
                headers={"User-Agent": f"TranslateBookWithLLM/{__version__}"}
            )
        return self._client

    async def close(self):
        """Close the HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None

    # -------------------------------------------------------------- logging
    def _log(self, event: str, message: str, level: int = logging.WARNING) -> None:
        """Route a provider message to the job log, or to the module logger."""
        if self.log_callback:
            try:
                self.log_callback(event, message)
                return
            except Exception:
                pass
        self._logger.log(level, message)

    # ---------------------------------------------------------- the loop
    async def generate(
        self,
        prompt: str,
        timeout: Optional[float] = None,
        system_prompt: Optional[str] = None,
        *,
        model: Optional[str] = None,
        context_window: Optional[int] = None,
    ) -> Optional[LLMResponse]:
        """
        Send one prompt, retrying transient failures per the retry policy.

        Args:
            prompt: The user prompt (content to process).
            timeout: Per-request timeout in seconds (policy default if None).
            system_prompt: Optional system prompt (role/instructions).
            model: Model for this request only (the provider's own model if
                None). The provider object is never mutated.
            context_window: Context window for this request only, for the
                providers that size it (Ollama, OpenAI-compatible).

        Returns:
            The parsed ``LLMResponse``, or None when every attempt failed or
            the error is not retryable.

        Raises:
            RateLimitError: Rate-limit budget exhausted (pipeline pauses).
            ContextOverflowError: The prompt does not fit the model.
            RepetitionLoopError: The model looped (streaming providers).
        """
        policy = self.effective_retry_policy()
        timeout = timeout or policy.timeout
        model = model or self.model
        await self._before_request(model)

        attempt = 0
        rate_limit_events = 0
        state: Dict[str, Any] = {}
        while attempt < policy.max_attempts:
            current_key = await self._key_pool.acquire() if self._key_pool else None
            request = LLMRequest(
                prompt=prompt, system_prompt=system_prompt, model=model,
                api_key=current_key, timeout=timeout,
                context_window=context_window, attempt=attempt, state=state,
            )
            try:
                response = await self._send(request)
            except (RateLimitError, ContextOverflowError, RepetitionLoopError):
                raise
            except ProviderError as e:
                error = e
            except httpx.HTTPStatusError as e:
                error = http_error_to_provider_error(e)
            except httpx.TimeoutException as e:
                error = ProviderError(
                    f"request timed out after {timeout:.0f}s ({type(e).__name__})",
                    retryable=True, transport=True,
                )
            except httpx.TransportError as e:
                error = ProviderError(f"connection error: {type(e).__name__}: {e}", retryable=True,
                                      transport=True)
            except json.JSONDecodeError as e:
                error = ProviderError(f"invalid JSON in the response: {e}", retryable=True,
                                      transport=True)
            except Exception as e:
                error = ProviderError(f"unexpected error: {type(e).__name__}: {e}", retryable=True,
                                      transport=True)
            else:
                if response is None:
                    error = ProviderError("the provider returned no response", retryable=True,
                                      transport=True)
                else:
                    self._after_response(response, request)
                    return response

            # --- classify the failure -------------------------------------
            if error.rate_limited:
                if self._key_pool is not None and current_key is not None:
                    rate_limit_events += 1
                    await handle_rate_limit(
                        self._key_pool, current_key, error.headers,
                        rate_limit_events, policy.max_attempts, self.log_callback,
                    )
                    continue
                self._on_keyless_rate_limit(error, request)

            if self._is_context_overflow(error):
                self._log("llm_context_overflow",
                          f"❌ [{self.display_name}] prompt too large for the model's context "
                          f"window ({request.context_window or getattr(self, 'context_window', '?')} "
                          f"tokens): {error.message}. Lower the chunk size or use a model "
                          f"with a larger context.")
                raise ContextOverflowError(f"{self.display_name} context overflow: {error.message}")

            if await self._retry_without_optional_params(error, request):
                # One-shot fallback owned by the provider: consumes no attempt.
                continue

            retryable = error.retryable
            if retryable is None:
                retryable = error.status_code is None or is_retryable_http_status(error.status_code)

            self._log_failure(error, request, policy)
            if not retryable:
                return None

            attempt += 1
            if attempt >= policy.max_attempts:
                if error.rate_limited:
                    hinted = _has_rate_limit_hint(error.headers)
                    raise RateLimitError(
                        f"{self.provider_name} rate limit persisted after "
                        f"{attempt} attempt(s): {error.message}",
                        retry_after=compute_wait_time(error.headers, attempt) if hinted else None,
                        provider=self.provider_name,
                    )
                self._log("llm_retries_exhausted",
                          f"❌ [{self.display_name}] all {policy.max_attempts} attempt(s) failed "
                          f"for model '{request.model}'; giving up on this request.")
                return None

            delay = policy.delay_before(attempt)
            if error.rate_limited and _has_rate_limit_hint(error.headers):
                # No key pool to rotate: honour the server's wait hint
                # (Retry-After, RFC 6585 section 4), within a cap.
                hint = compute_wait_time(error.headers, attempt)
                delay = max(delay, min(hint, MAX_HINTED_RETRY_WAIT))
            self._log("llm_retry", f"   Retrying in {delay:.1f}s "
                                   f"(attempt {attempt + 1}/{policy.max_attempts})...",
                      level=logging.INFO)
            await asyncio.sleep(delay)

        return None

    # ----------------------------------------------------- provider hooks
    @abstractmethod
    async def _send(self, request: LLMRequest) -> LLMResponse:
        """Perform one attempt and return the parsed response.

        Raise ``ProviderError`` (or let ``httpx`` errors escape) on failure;
        raise ``ContextOverflowError`` / ``RepetitionLoopError`` /
        ``RateLimitError`` directly when the provider knows better.
        """

    async def _before_request(self, model: str) -> None:
        """Hook run once per ``generate`` call, before the first attempt."""

    def _on_keyless_rate_limit(self, error: ProviderError, request: LLMRequest) -> None:
        """Hook for a 429 with no key pool to rotate. Default: retry with backoff."""

    async def _retry_without_optional_params(
        self, error: ProviderError, request: LLMRequest
    ) -> bool:
        """Hook: return True to re-send without optional request parameters.

        Used by providers that add parameters some models reject (sampling,
        reasoning overrides). Must be one-shot per request.
        """
        return False

    # --------------------------------------------------- classification
    def _is_context_overflow(self, error: ProviderError) -> bool:
        """HTTP status and error code first; keywords only as a last resort."""
        if error.rate_limited or error.status_code in NEVER_OVERFLOW_STATUSES:
            return False
        if error.context_overflow:
            return True
        if (error.error_code or "").lower() in CONTEXT_OVERFLOW_ERROR_CODES:
            return True
        if error.transport:
            return False
        return is_context_overflow_error(
            f"{error.message} {error.body}", error.status_code, self.OVERFLOW_KEYWORDS,
        )

    def _log_failure(self, error: ProviderError, request: LLMRequest, policy: RetryPolicy) -> None:
        status = f"HTTP {error.status_code}" if error.status_code else "error"
        hint = self._status_hint(error.status_code)
        self._log("llm_request_failed",
                  f"⚠️ [{self.display_name}] {status} for model '{request.model}' "
                  f"(attempt {request.attempt + 1}/{policy.max_attempts}): {error.message}"
                  + (f" — {hint}" if hint else ""))

    def _status_hint(self, status_code: Optional[int]) -> str:
        """Short advice for common statuses; providers may refine it."""
        return {
            401: "invalid API key",
            402: "insufficient credits",
            403: "access denied for this key or model",
            404: "unknown model or endpoint",
        }.get(status_code or 0, "")

    # ------------------------------------------------------------- usage
    def _after_response(self, response: LLMResponse, request: LLMRequest) -> None:
        """Report usage to the cost trackers and log a one-line summary."""
        if not (response.content or "").strip():
            self._logger.warning(
                      f"⚠️ [{self.display_name}] empty response from '{request.model}' "
                      f"({response.prompt_tokens}+{response.completion_tokens} tokens): the "
                      f"model likely refused or filtered this chunk.")
        if response.prompt_tokens or response.completion_tokens or response.cost is not None:
            cost = response.cost if response.cost is not None else 0.0
            self._cost_tracker.record(
                cost, response.prompt_tokens, response.completion_tokens, response.cost_estimated)
            job_tracker = get_current_cost_tracker()
            if job_tracker is not None:
                job_tracker.record(
                    cost, response.prompt_tokens, response.completion_tokens,
                    response.cost_estimated)
            summary = (f"[{self.display_name}] {request.model}: "
                       f"{response.prompt_tokens}+{response.completion_tokens} tokens")
            if response.cost is not None:
                session = (job_tracker or self._cost_tracker).snapshot()[0]
                label = "estimated cost" if response.cost_estimated else "cost"
                summary += f" | {label} ${response.cost:.6f} (session ${session:.4f})"
            self._logger.info(summary)

    def get_session_cost(self) -> Tuple[float, Dict[str, int]]:
        """This instance's (cost_usd, {"prompt", "completion"}) totals."""
        return self._cost_tracker.snapshot()

    def reset_session_cost(self) -> None:
        """Reset this instance's usage totals."""
        self._cost_tracker.reset()

    def set_cost_callback(self, callback: Optional[Callable[[Dict[str, Any]], None]]) -> None:
        """Callback receiving this instance's usage after each request.

        Jobs should use ``begin_job_cost_tracking`` instead, which covers
        every provider instance created during the job.
        """
        self._cost_tracker.callback = callback

    # ------------------------------------------------------ thinking status
    def get_is_thinking_model(self) -> Optional[bool]:
        """Whether the model produces thinking output (None when unknown)."""
        behavior = getattr(self, "_thinking_behavior", None)
        if behavior is not None:
            from src.core.llm.thinking.behavior import ThinkingBehavior
            return behavior != ThinkingBehavior.STANDARD
        status = getattr(self, "_is_thinking_model", None)
        if callable(status):
            status = status()
        return None if status is None else bool(status)

    async def detect_thinking_model(self) -> Optional[bool]:
        """Run the provider's thinking detection (Ollama) and report the status."""
        return self.get_is_thinking_model()

    # ------------------------------------------------------- extraction
    def extract_translation(self, response: str) -> Optional[str]:
        """
        Extract translation from response using configured tags with strict validation.

        Returns the content between TRANSLATE_TAG_IN and TRANSLATE_TAG_OUT,
        ignoring <think>/<thinking>/<reasoning> blocks.
        """
        return self._extractor.extract(response)

    async def translate_text(self, prompt: str) -> Optional[str]:
        """Complete translation workflow: request + extraction"""
        response = await self.generate(prompt)
        if response:
            return self.extract_translation(response.content)
        return None
