"""
The job result: what every engine and the runner return.

One typed object replaces the bool / None / dict returns and the Phase 1 side
channels (``raw_fallback_used`` in the stats, ``failed_doc_paths`` inside the
EPUB pipeline). The status of a finished run always comes from
:func:`src.core.job.verdict.classify_completion`, the single verdict shared by
the web handler and the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .verdict import CompletionVerdict, classify_completion


class JobStatus(str, Enum):
    COMPLETED = 'completed'
    PARTIAL = 'partial'
    FAILED = 'failed'
    INTERRUPTED = 'interrupted'
    RATE_LIMITED = 'rate_limited'

    @property
    def web_status(self) -> str:
        """The status string the web state and Socket.IO payloads use."""
        return 'error' if self is JobStatus.FAILED else self.value


def _as_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


@dataclass(frozen=True)
class JobResult:
    """Outcome of one engine run or of a whole job.

    Attributes:
        status: See :class:`JobStatus`.
        output_path: The file written (or the partial file, when marked).
        output_paths: Every file the run produced (output, report, audio).
        units_total / units_translated / units_failed / units_fallback /
            units_untranslated / units_raw_fallback: Unit counts, from the
            last stats the engine emitted.
        failed_units: Indices of units that failed, when the engine knows them.
        failed_documents: EPUB documents whose translation failed.
        error: Human-readable reason when the run failed.
        usage: Token/cost totals when tracked.
        stats: The merged stats dict the counts were read from.
        verdict: The ``classify_completion`` verdict, when one was taken.
        detail: Engine-specific payload (the term-fix result, for example).
    """

    status: JobStatus
    output_path: Optional[str] = None
    output_paths: Tuple[str, ...] = ()
    units_total: int = 0
    units_translated: int = 0
    units_failed: int = 0
    units_fallback: int = 0
    units_untranslated: int = 0
    units_raw_fallback: int = 0
    failed_units: Tuple[int, ...] = ()
    failed_documents: Tuple[str, ...] = ()
    error: Optional[str] = None
    usage: Optional[Mapping[str, Any]] = None
    stats: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)
    verdict: Optional[CompletionVerdict] = field(default=None, compare=False)
    detail: Any = field(default=None, compare=False, repr=False)

    @property
    def ok(self) -> bool:
        return self.status is JobStatus.COMPLETED

    def __bool__(self) -> bool:  # `if await translate_file(...)` reads naturally
        return self.ok

    def with_changes(self, **changes: Any) -> "JobResult":
        return replace(self, **changes)

    @property
    def exit_code(self) -> int:
        """CLI exit code: 0 only for a completed job."""
        return 0 if self.ok else 1

    # -- construction ------------------------------------------------------

    @classmethod
    def failed(cls, error: str, *, output_path: Optional[str] = None,
               stats: Optional[Mapping[str, Any]] = None) -> "JobResult":
        return cls._with_counts(JobStatus.FAILED, stats or {}, output_path=output_path,
                                error=error)

    @classmethod
    def interrupted(cls, *, output_path: Optional[str] = None,
                    stats: Optional[Mapping[str, Any]] = None) -> "JobResult":
        return cls._with_counts(JobStatus.INTERRUPTED, stats or {}, output_path=output_path)

    @classmethod
    def from_run(cls, stats: Mapping[str, Any], output_path: Optional[str], *,
                 engine_ok: Optional[bool] = None, interrupted: bool = False,
                 failed_documents: Iterable[str] = (), error: Optional[str] = None,
                 detail: Any = None) -> "JobResult":
        """Turn a finished engine run into a result.

        ``interrupted`` wins. Otherwise the verdict decides; an engine that
        reported failure (``engine_ok is False``) while the verdict found
        nothing wrong is reported as partial, never as completed: the engine
        knows about a problem the stats do not show.
        """
        stats = dict(stats or {})
        if interrupted:
            return cls._with_counts(JobStatus.INTERRUPTED, stats, output_path=output_path,
                                    failed_documents=failed_documents, detail=detail)
        verdict = classify_completion(stats, output_path)
        if verdict.status == 'error':
            status = JobStatus.FAILED
            error = error or verdict.error
        elif verdict.status == 'partial':
            status = JobStatus.PARTIAL
        elif engine_ok is False or failed_documents:
            status = JobStatus.PARTIAL
            error = error or "The engine reported an incomplete run; see the job log."
        else:
            status = JobStatus.COMPLETED
        return cls._with_counts(status, stats, output_path=output_path, verdict=verdict,
                                failed_documents=failed_documents, error=error,
                                detail=detail)

    @classmethod
    def _with_counts(cls, status: JobStatus, stats: Mapping[str, Any], **kwargs: Any) -> "JobResult":
        total = _as_int(stats.get('total_chunks'))
        completed = _as_int(stats.get('completed_chunks'))
        failed = _as_int(stats.get('failed_chunks'))
        unfinished = (_as_int(stats.get('unfinished_chunks')) if 'unfinished_chunks' in stats
                      else _as_int(stats.get('fallback_used')))
        usage: Optional[Dict[str, Any]] = None
        if any(k in stats for k in ('openrouter_cost', 'openrouter_prompt_tokens')):
            usage = {
                'cost': stats.get('openrouter_cost', 0.0),
                'prompt_tokens': _as_int(stats.get('openrouter_prompt_tokens')),
                'completion_tokens': _as_int(stats.get('openrouter_completion_tokens')),
                'cost_estimated': bool(stats.get('openrouter_cost_estimated', False)),
            }
        failed_documents = tuple(sorted(str(p) for p in (kwargs.pop('failed_documents', ()) or ())))
        output_path = kwargs.get('output_path')
        return cls(
            status=status,
            output_paths=(output_path,) if output_path else (),
            units_total=total,
            units_translated=max(0, completed - failed),
            units_failed=failed,
            units_fallback=_as_int(stats.get('fallback_used')),
            units_untranslated=unfinished,
            units_raw_fallback=_as_int(stats.get('raw_fallback_used')),
            failed_documents=failed_documents,
            usage=usage,
            stats=dict(stats),
            **kwargs,
        )
