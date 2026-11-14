"""Immutable progress snapshot — the single wire/serialization contract."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class Phase(str, Enum):
    """Coarse phase of a translation job.

    A job is either single-phase (``TRANSLATING`` only) or two-phase
    (``TRANSLATING`` then ``REFINING``). Inheriting from ``str`` makes the
    enum trivially JSON-serializable and comparable to the wire string.
    """

    TRANSLATING = "translating"
    REFINING = "refining"


def global_percent(
    phase: "Phase",
    done: int,
    total: int,
    refinement_enabled: bool,
    split: float = 50.0,
) -> float:
    """Map per-phase progress onto the global ``[0, 100]`` bar.

    Single source of the segment math, used by both :class:`ProgressTracker`
    and :func:`snapshot_from_job_stats`. Single-phase jobs fill the whole bar;
    two-phase jobs put translation in ``[0, split]`` and refinement in
    ``[split, 100]``. Per-phase totals may differ — each phase is scaled into
    its own segment independently.
    """
    ratio = min(done / total, 1.0) if total > 0 else 0.0
    if not refinement_enabled:
        pct = ratio * 100.0
    elif phase is Phase.TRANSLATING:
        pct = ratio * split
    else:  # REFINING
        pct = split + ratio * (100.0 - split)
    return min(pct, 100.0)


@dataclass(frozen=True)
class ProgressSnapshot:
    """A point-in-time view of progress.

    ``percent`` is the **global** progress in ``[0, 100]`` — already mapped
    across phases by the tracker — so the UI displays it verbatim and never
    recomputes it. The per-phase counts (``units_done`` / ``units_total`` …)
    describe only the *current* phase and exist for labels like
    "Refining 3/10".

    ``units_done`` advances on every processed unit regardless of success;
    ``units_succeeded`` and ``units_failed`` split that count for honest
    quality reporting (so "100%" can still surface that N units fell back to
    untranslated output).
    """

    phase: Phase
    units_done: int
    units_total: int
    units_succeeded: int
    units_failed: int
    percent: float
    refinement_enabled: bool
    elapsed_seconds: Optional[float] = None
    eta_seconds: Optional[float] = None

    @property
    def quality_degraded(self) -> bool:
        """True when at least one unit did not translate cleanly."""
        return self.units_failed > 0

    @property
    def is_complete(self) -> bool:
        return self.percent >= 100.0

    def to_dict(self) -> Dict[str, Any]:
        """Canonical serialization. ``percent`` is rounded for a stable wire."""
        data: Dict[str, Any] = {
            "phase": self.phase.value,
            "percent": round(self.percent, 2),
            "units_done": self.units_done,
            "units_total": self.units_total,
            "units_succeeded": self.units_succeeded,
            "units_failed": self.units_failed,
            "refinement_enabled": self.refinement_enabled,
            "quality_degraded": self.quality_degraded,
        }
        if self.elapsed_seconds is not None:
            data["elapsed_seconds"] = round(self.elapsed_seconds, 2)
        if self.eta_seconds is not None:
            data["eta_seconds"] = round(self.eta_seconds, 2)
        return data
