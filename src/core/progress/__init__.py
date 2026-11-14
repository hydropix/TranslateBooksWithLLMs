"""Unified progress domain model for the translation/refine pipeline.

This package defines the single source of truth for progress reporting:

* :class:`Phase` — the coarse phase of work (translating vs refining).
* :class:`ProgressSnapshot` — an immutable, serializable view of progress at
  a point in time. ``percent`` is the *global* 0–100 value the UI displays;
  no consumer recomputes it.
* :class:`ProgressTracker` — the single authority that owns the totals, the
  phase, and the percent math (including the two-phase segment mapping that
  replaces the ad-hoc "double the total when refining" trick scattered across
  the legacy code).

In production the percent authority is reached through the
:func:`snapshot_from_legacy_stats` bridge: the format engines still emit their
historical stats dicts, and the bridge maps them onto a single canonical
``percent`` / ``phase`` at the handler seam (``src/api/handlers.py``). The
:class:`ProgressTracker` driver — formats calling ``record_unit()`` directly,
with the bridge deleted — is the intended end state but is **not** wired into
the pipeline; today it backs the unit tests and shares its segment math
(:func:`global_percent`) with the bridge so both agree by construction.
"""

from .legacy import snapshot_from_legacy_stats
from .reporter import COUNT_FIELDS, ProgressReporter, progress_payload
from .snapshot import Phase, ProgressSnapshot, global_percent
from .tracker import ProgressTracker

__all__ = [
    "COUNT_FIELDS",
    "Phase",
    "ProgressReporter",
    "ProgressSnapshot",
    "ProgressTracker",
    "progress_payload",
    "global_percent",
    "snapshot_from_legacy_stats",
]
