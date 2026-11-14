"""Unified progress domain model for the translation/refine pipeline.

* :class:`Phase` — the coarse phase of work (translating vs refining).
* :class:`ProgressSnapshot` — an immutable, serializable view of progress at
  a point in time. ``percent`` is the *global* 0–100 value the UI displays;
  no consumer recomputes it.
* :class:`ProgressTracker` — owns the totals, the phase and the percent math
  (including the two-phase segment mapping).
* :class:`ProgressReporter` — what an engine owns for one run: it wraps a
  tracker and builds every stats emit through :func:`progress_payload` (the
  chunk counts from the tracker, the engine's quality fields appended).
* :func:`snapshot_from_job_stats` — the web sink's derivation of the
  canonical ``percent`` / ``phase`` from the job's merged stats (engine counts
  plus the runner's phase tags).
"""

from .job_stats import snapshot_from_job_stats
from .reporter import COUNT_FIELDS, ProgressReporter, progress_payload
from .snapshot import Phase, ProgressSnapshot, global_percent
from .tracker import ProgressTracker

__all__ = [
    "COUNT_FIELDS",
    "Phase",
    "ProgressReporter",
    "ProgressSnapshot",
    "ProgressTracker",
    "global_percent",
    "progress_payload",
    "snapshot_from_job_stats",
]
