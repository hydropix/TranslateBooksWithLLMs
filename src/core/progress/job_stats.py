"""Canonical progress of a job, derived from its merged stats.

The engines emit :class:`~src.core.progress.reporter.ProgressReporter`
payloads (chunk counts of the current phase plus quality fields); the job
runner tags every emit with the workflow phase (``enable_refinement``,
``current_phase``, ``refine_only``); the web sink (``src/api/handlers.py``)
merges them into the job's stats. :func:`snapshot_from_job_stats` turns that
merged dict into a :class:`ProgressSnapshot`, whose ``percent`` / ``phase``
the frontend displays verbatim. This is the one place the global percent is
derived; it shares :func:`global_percent` with :class:`ProgressTracker`, so
the two never diverge.
"""

from __future__ import annotations

from typing import Any, Dict

from .snapshot import Phase, ProgressSnapshot, global_percent

_REFINE_SPLIT = 50.0


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def snapshot_from_job_stats(stats: Dict[str, Any]) -> ProgressSnapshot:
    """Derive a :class:`ProgressSnapshot` from a job's merged stats dict."""
    total = _as_int(stats.get("total_chunks"))
    done = _as_int(stats.get("completed_chunks"))
    failed = _as_int(stats.get("failed_chunks"))
    refinement_enabled = bool(stats.get("enable_refinement"))

    if refinement_enabled:
        # Two-phase refine-after: translation in [0, 50], refinement in [50, 100].
        phase = Phase.REFINING if stats.get("current_phase") == 2 else Phase.TRANSLATING
    else:
        phase = Phase.REFINING if stats.get("refine_only") else Phase.TRANSLATING

    return ProgressSnapshot(
        phase=phase,
        units_done=done,
        units_total=total,
        units_succeeded=max(0, done - failed),
        units_failed=failed,
        percent=global_percent(phase, done, total, refinement_enabled, _REFINE_SPLIT),
        refinement_enabled=refinement_enabled,
    )
