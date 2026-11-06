"""
Per-job LLM cost tracking.

A job runs in its own thread with its own event loop, so the active tracker is
carried in a ``ContextVar``: every provider call made from that job (translate,
refine, term-fix, ...) reports to the same tracker, and concurrent jobs never
see each other's totals or callbacks.
"""
import threading
from contextvars import ContextVar, Token
from typing import Any, Callable, Dict, Optional, Tuple

CostCallback = Callable[[Dict[str, Any]], None]


class CostTracker:
    """Thread-safe accumulator for the cost and token usage of one job."""

    def __init__(self, callback: Optional[CostCallback] = None):
        self._lock = threading.Lock()
        self.callback = callback
        self.total_cost = 0.0
        self.tokens = {"prompt": 0, "completion": 0}
        # Requests whose cost came from a fallback rate instead of the API.
        self.estimated_requests = 0

    def reset(self) -> None:
        with self._lock:
            self.total_cost = 0.0
            self.tokens = {"prompt": 0, "completion": 0}
            self.estimated_requests = 0

    @property
    def is_estimated(self) -> bool:
        """True when at least one request's cost is an estimate."""
        return self.estimated_requests > 0

    def snapshot(self) -> Tuple[float, Dict[str, int]]:
        with self._lock:
            return self.total_cost, dict(self.tokens)

    def record(self, cost: float, prompt_tokens: int, completion_tokens: int,
               estimated: bool = False) -> Dict[str, Any]:
        """Add one request and notify the callback. Returns the cost payload."""
        with self._lock:
            self.total_cost += cost
            self.tokens["prompt"] += prompt_tokens
            self.tokens["completion"] += completion_tokens
            if estimated:
                self.estimated_requests += 1
            data = {
                "request_cost": cost,
                "request_cost_estimated": estimated,
                "session_cost": self.total_cost,
                "cost_is_estimated": self.estimated_requests > 0,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_prompt_tokens": self.tokens["prompt"],
                "total_completion_tokens": self.tokens["completion"],
            }
            callback = self.callback
        if callback:
            try:
                callback(data)
            except Exception as cb_err:
                print(f"[CostTracker] WARN: Cost callback error: {cb_err}")
        return data


_current_tracker: ContextVar[Optional[CostTracker]] = ContextVar(
    "llm_cost_tracker", default=None
)


def get_current_cost_tracker() -> Optional[CostTracker]:
    """Return the tracker of the job running in this context, if any."""
    return _current_tracker.get()


def begin_job_cost_tracking(callback: Optional[CostCallback] = None) -> Tuple[CostTracker, Token]:
    """Install a fresh tracker for the current job. Pass the token to ``end_job_cost_tracking``."""
    tracker = CostTracker(callback)
    return tracker, _current_tracker.set(tracker)


def end_job_cost_tracking(token: Token) -> None:
    """Remove the job's tracker, restoring the previous one."""
    try:
        _current_tracker.reset(token)
    except ValueError:
        # Token created in another context; just clear.
        _current_tracker.set(None)
