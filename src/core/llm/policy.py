"""
Retry / timeout policy for LLM requests.

One small value object owns every knob of the provider request loop: how many
transient attempts a request gets, how long to wait between them, and the
per-request timeout. It is resolved per job, never bound at import:

    * defaults come from ``src.config`` (read at call time, so a settings
      reload is honoured by the next job);
    * the web UI's per-job ``timeout`` / ``max_attempts`` / ``retry_delay``
      override them through ``RetryPolicy.from_job_config``;
    * the active job installs its policy in a ``ContextVar`` (same pattern as
      the per-job ``CostTracker``), so providers built anywhere inside that
      job pick it up without threading a parameter through every layer.

HTTP 429 handling keeps its own budget (see ``rate_limit_handler``): rotating
to a spare API key never consumes one of these attempts.
"""

from __future__ import annotations

import random
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class RetryPolicy:
    """How a provider retries one request.

    Attributes:
        max_attempts: Transient attempts per request (>= 1). 429 rotations
            and rate-limit waits do not count against it.
        base_delay: Seconds to wait before the second attempt.
        backoff_factor: Multiplier applied per further attempt
            (1.0 = fixed delay).
        max_delay: Upper bound for one wait, before jitter.
        jitter: Fraction of the delay added at random (0.25 = up to +25%),
            so parallel workers do not retry in lockstep.
        timeout: Per-request timeout in seconds.
    """

    max_attempts: int = 2
    base_delay: float = 2.0
    backoff_factor: float = 2.0
    max_delay: float = 30.0
    jitter: float = 0.25
    timeout: float = 300.0

    def __post_init__(self):
        object.__setattr__(self, "max_attempts", max(1, int(self.max_attempts)))
        object.__setattr__(self, "base_delay", max(0.0, float(self.base_delay)))
        object.__setattr__(self, "backoff_factor", max(1.0, float(self.backoff_factor)))
        object.__setattr__(self, "max_delay", max(0.0, float(self.max_delay)))
        object.__setattr__(self, "jitter", max(0.0, float(self.jitter)))
        object.__setattr__(self, "timeout", max(1.0, float(self.timeout)))

    def delay_before(self, attempt: int, rng: Optional[random.Random] = None) -> float:
        """Seconds to wait before 0-based ``attempt`` (``attempt >= 1``).

        ``base_delay * backoff_factor ** (attempt - 1)``, capped at
        ``max_delay``, plus up to ``jitter`` of itself at random.
        """
        if attempt <= 0 or self.base_delay <= 0:
            return 0.0
        delay = min(self.base_delay * self.backoff_factor ** (attempt - 1), self.max_delay)
        if self.jitter:
            delay += delay * self.jitter * (rng or random).random()
        return delay

    def with_overrides(self, **changes: Any) -> "RetryPolicy":
        """Copy with the given fields replaced; ``None`` values are ignored."""
        return replace(self, **{k: v for k, v in changes.items() if v is not None})

    @classmethod
    def from_config(cls) -> "RetryPolicy":
        """Defaults from ``src.config``, read now (not at import)."""
        from src import config

        return cls(
            max_attempts=getattr(config, "MAX_TRANSLATION_ATTEMPTS", 2),
            base_delay=getattr(config, "RETRY_BASE_DELAY", 2.0),
            backoff_factor=getattr(config, "RETRY_BACKOFF_FACTOR", 2.0),
            max_delay=getattr(config, "RETRY_MAX_DELAY", 30.0),
            jitter=getattr(config, "RETRY_JITTER", 0.25),
            timeout=getattr(config, "REQUEST_TIMEOUT", 300),
        )

    @classmethod
    def from_job_config(cls, job_config: Optional[Mapping[str, Any]]) -> "RetryPolicy":
        """Config defaults overridden by a web job's stored settings.

        Reads ``request_timeout`` (or ``timeout``), ``max_attempts`` and
        ``retry_delay``. Missing or malformed values keep the default.
        """
        policy = cls.from_config()
        if not job_config:
            return policy
        return policy.with_overrides(
            timeout=_positive_number(
                job_config.get("request_timeout", job_config.get("timeout"))),
            max_attempts=_positive_number(job_config.get("max_attempts"), integer=True),
            base_delay=_non_negative_number(job_config.get("retry_delay")),
        )


def _positive_number(value: Any, integer: bool = False):
    try:
        number = int(value) if integer else float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _non_negative_number(value: Any):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


_current_policy: ContextVar[Optional[RetryPolicy]] = ContextVar(
    "llm_retry_policy", default=None
)


def current_retry_policy() -> RetryPolicy:
    """The active job's policy, or the config defaults outside a job."""
    return _current_policy.get() or RetryPolicy.from_config()


def begin_job_retry_policy(policy: Optional[RetryPolicy]) -> Token:
    """Install ``policy`` for the current job. Pass the token to ``end_job_retry_policy``."""
    return _current_policy.set(policy)


def end_job_retry_policy(token: Token) -> None:
    """Restore the policy that was active before ``begin_job_retry_policy``."""
    try:
        _current_policy.reset(token)
    except ValueError:
        _current_policy.set(None)
