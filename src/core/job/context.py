"""
The runtime half of a job: what an engine reports to and asks of its caller.

``JobConfig`` says *what* to do; ``JobContext`` carries the live hooks the
engine needs while doing it: the log callback, the stats sink, the
interruption check, the checkpoint manager, the job id and the index to
resume from. The runner builds one per phase; tests and direct engine callers
build one by hand (every field is optional).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class JobContext:
    """Live hooks of one engine run.

    Attributes:
        translation_id: Job id (checkpoint key). None disables checkpointing.
        checkpoint_manager: ``CheckpointManager`` or None.
        log_callback: ``(key, message[, data])`` job log.
        stats_callback: Receives every stats dict the engine emits.
        check_interruption: Returns True once the user asked to stop.
        resume_from_index: Unit index to resume from (0 for a fresh run).
        checkpoint_phase: Checkpoint namespace the engine writes its units
            to: ``translate`` or ``refine`` (the refine pass of a job never
            overwrites its translation chunks).
    """

    translation_id: Optional[str] = None
    checkpoint_manager: Any = None
    log_callback: Optional[Callable[..., Any]] = None
    stats_callback: Optional[Callable[[Dict[str, Any]], Any]] = None
    check_interruption: Optional[Callable[[], bool]] = None
    resume_from_index: int = 0
    checkpoint_phase: str = 'translate'

    def with_changes(self, **changes: Any) -> "JobContext":
        return replace(self, **changes)

    def log(self, key: str, message: str = "", data: Optional[dict] = None) -> None:
        if self.log_callback is None:
            return
        if data is None:
            self.log_callback(key, message)
        else:
            self.log_callback(key, message, data)

    def emit(self, stats: Dict[str, Any]) -> None:
        if self.stats_callback is not None:
            self.stats_callback(stats)

    def interrupted(self) -> bool:
        return bool(self.check_interruption and self.check_interruption())

    def tap_stats(self) -> "tuple[JobContext, Dict[str, Any]]":
        """A copy whose stats are also merged into the returned dict.

        Engines emit partial dicts; the merged view is what the verdict is
        computed from once the engine returns.
        """
        seen: Dict[str, Any] = {}
        downstream = self.stats_callback

        def _tap(stats: Dict[str, Any]) -> None:
            if isinstance(stats, dict):
                seen.update(stats)
            if downstream is not None:
                downstream(stats)

        return self.with_changes(stats_callback=_tap), seen
