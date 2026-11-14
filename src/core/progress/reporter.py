"""The engine side of progress reporting.

An engine owns one :class:`ProgressReporter` per run. The reporter wraps a
:class:`ProgressTracker` (phase ``TRANSLATING`` for translation engines,
``REFINING`` for refine engines; whether a refine phase follows a translation
is the job runner's business) and turns it into the stats dicts the engine
hands to its ``stats_callback``.

Every payload is built by :func:`progress_payload`: the chunk-count fields
(``total_chunks`` / ``completed_chunks`` / ``failed_chunks``) come from the
tracker snapshot, the engine's own quality fields (retries, fallbacks,
unfinished-chunk maps, ``run_*`` twins, ...) are appended as ``extra``. Engines
never write the three count keys by hand.

``completed_chunks`` counts every processed unit, failed ones included (the
progress bar advances on a failure too); ``failed_chunks`` is the part of it
that failed. Engines that need "still owed" report it as ``unfinished_chunks``
in ``extra``.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Mapping, Optional

from .snapshot import Phase, ProgressSnapshot
from .tracker import ProgressTracker

#: The chunk-count fields every stats payload carries, built from the tracker.
COUNT_FIELDS = ("total_chunks", "completed_chunks", "failed_chunks")

StatsCallback = Callable[[Dict[str, Any]], Any]


def progress_payload(snapshot: ProgressSnapshot,
                     extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The stats dict for ``snapshot`` plus the engine's quality fields.

    The single builder of the chunk-count fields. ``extra`` must not carry
    them: an engine that does is overriding the tracker, which is a bug.
    """
    fields = dict(extra or {})
    clash = [key for key in COUNT_FIELDS if key in fields]
    if clash:
        raise ValueError(f"extra progress fields must not set {', '.join(clash)}")
    return {
        "total_chunks": snapshot.units_total,
        "completed_chunks": snapshot.units_done,
        "failed_chunks": snapshot.units_failed,
        **fields,
    }


class ProgressReporter:
    """Chunk counts of one engine run, and the stats emits built from them.

    Args:
        callback: Receives every emitted payload (usually
            ``ctx.stats_callback``); None makes :meth:`emit` a no-op that
            still returns the payload.
        phase: ``TRANSLATING`` or ``REFINING``.

    An emit whose payload equals the previous one is skipped: it carries no
    information, and a repeated tick at 100% would read as a run that
    finished twice.
    """

    def __init__(self, callback: Optional[StatsCallback] = None, *,
                 phase: Phase = Phase.TRANSLATING) -> None:
        self._callback = callback
        self._phase = phase
        self._tracker = ProgressTracker()
        self._tracker.begin_phase(phase, 0)
        self._last: Optional[Dict[str, Any]] = None

    # -- driving -------------------------------------------------------------

    def begin(self, total: int, done: int = 0, failed: int = 0) -> None:
        """Start (or restart) the run with ``total`` units, ``done`` of them
        already processed (restored from a checkpoint), ``failed`` of those
        failed."""
        self._tracker.begin_phase(self._phase, total)
        self._tracker.set_counts(done=done, failed=failed)

    def record(self, succeeded: bool = True, count: int = 1) -> None:
        """``count`` more units processed."""
        self._tracker.record_unit(succeeded=succeeded, count=count)

    def set_counts(self, done: int, failed: Optional[int] = None,
                   total: Optional[int] = None) -> None:
        """Absolute counts, for engines that compute them themselves."""
        self._tracker.set_counts(done=done, failed=failed, total=total)

    # -- reading -------------------------------------------------------------

    def snapshot(self) -> ProgressSnapshot:
        return self._tracker.snapshot()

    @property
    def total(self) -> int:
        return self.snapshot().units_total

    @property
    def done(self) -> int:
        return self.snapshot().units_done

    @property
    def failed(self) -> int:
        return self.snapshot().units_failed

    @property
    def succeeded(self) -> int:
        return self.snapshot().units_succeeded

    # -- emitting ------------------------------------------------------------

    def payload(self, extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """The stats dict for the current counts, without emitting it."""
        return progress_payload(self.snapshot(), extra)

    def emit(self, extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Build the payload and hand it to the callback (unless unchanged)."""
        payload = self.payload(extra)
        if self._callback is not None and payload != self._last:
            self._last = copy.deepcopy(payload)
            self._callback(payload)
        return payload
