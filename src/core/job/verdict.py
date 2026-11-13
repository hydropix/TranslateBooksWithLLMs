"""
Completion classifier for finished translation jobs.

A translation loop that ran to the end is not the same thing as a translation
that succeeded. Chunks can fall back to their source text, and the output file
can be missing or empty even though nothing raised. This module turns the raw
job stats plus the state of the output file into a single verdict, so job
finalization has exactly one place that decides between 'completed', 'partial'
and 'error'.

The function is pure apart from two ``os.path`` lookups: no logging, no state
manager, no socket, no checkpoint manager. It is meant to be callable with a
plain dict and a path string from a unit test.
"""
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class CompletionVerdict:
    """Outcome of classifying a finished translation job.

    Attributes:
        status: Exactly one of 'completed', 'partial' or 'error'.
        failed_chunks: Number of chunks that failed outright (>= 0).
        fallback_chunks: Number of chunks left as source text (>= 0). Reported
            for the message only; see ``_unfinished`` for how it does (and
            does not) affect the verdict.
        unfinished_chunks: Number of chunks the job still owes — pending or
            untranslated — as of the last stats update (>= 0). This is the
            authority for rule 4 (issue #261, D9).
        error: Human-readable reason, non-None if and only if status is 'error'.
    """

    status: str
    failed_chunks: int
    fallback_chunks: int
    unfinished_chunks: int
    error: Optional[str]


def _unfinished(stats: Mapping[str, Any]) -> int:
    """Number of chunks still owed, per design decision D9.

    ``unfinished_chunks`` is the authority once it is present — even when it
    is legitimately ``0`` while ``fallback_used`` is not (every chunk that
    fell back was successfully retried). It is only when the key is entirely
    *absent* — a job that predates this counter, or a path that never learned
    to emit it — that a missing key must not be silently read as "all good":
    fall back to the historical ``fallback_used`` counter instead (the #239
    regression this guards against).
    """
    if 'unfinished_chunks' in stats:
        return int(stats.get('unfinished_chunks') or 0)
    return int(stats.get('fallback_used') or 0)


def classify_completion(
    stats: Mapping[str, Any],
    output_path: Optional[str],
) -> CompletionVerdict:
    """Classify a finished translation job as completed, partial or errored.

    The rules below are exhaustive and evaluated in order:

    1. Read ``failed_chunks``, ``fallback_used`` and ``total_chunks`` from the
       stats, coercing a missing or ``None`` value to 0.
    2. Missing output: ``output_path`` is falsy, or the file does not exist
       → 'error'.
    3. Empty output: the file exists, is 0 bytes, and ``total_chunks`` > 0
       → 'error'. The ``total_chunks`` > 0 condition is deliberate: an empty
       source file legitimately produces an empty output and stays 'completed'.
    4. Unfinished work: any failed chunk, or any chunk still pending/untranslated
       per ``_unfinished(stats)`` → 'partial'.
    5. Otherwise → 'completed'.

    IMPORTANT: ``token_alignment_used`` and ``placeholder_errors`` NEVER
    contribute to the verdict, and must never be added to the rules above.
    Those chunks *are* translated; only their placeholder positions were
    approximated. Counting them would report healthy jobs as degraded.

    ``fallback_used`` no longer drives rule 4 directly (design decision D9):
    a job that retried every fallen-back chunk to a successful outcome must
    be able to reach 'completed'. It stays informational — reported via
    ``fallback_chunks`` for the message — and is used as the verdict's input
    only when the ``unfinished_chunks`` key is absent from ``stats`` (see
    ``_unfinished``). Nothing about ``fallback_used``'s meaning changes
    (design decision D10): it still counts every Phase 3 fallback that
    happened during the run, retried or not.

    ``failed_chunks`` and ``fallback_chunks`` are reported separately and are
    never summed into a single number: "failed outright" and "left as source
    text" are different outcomes for the user.

    Args:
        stats: The job stats payload (any mapping; missing keys are tolerated).
        output_path: Path the job was supposed to write, or None.

    Returns:
        A :class:`CompletionVerdict`.
    """
    failed = int(stats.get('failed_chunks') or 0)
    fallback = int(stats.get('fallback_used') or 0)
    total = int(stats.get('total_chunks') or 0)
    unfinished = _unfinished(stats)

    # Rule 2 — the job claims to be done but there is nothing on disk.
    if not output_path or not os.path.exists(output_path):
        return CompletionVerdict(
            status='error',
            failed_chunks=failed,
            fallback_chunks=fallback,
            unfinished_chunks=unfinished,
            error=f"Output file was not written: {output_path or '(none)'}",
        )

    # Rule 3 — a 0-byte output is only an error when there was work to do.
    if os.path.getsize(output_path) == 0 and total > 0:
        return CompletionVerdict(
            status='error',
            failed_chunks=failed,
            fallback_chunks=fallback,
            unfinished_chunks=unfinished,
            error=f"Output file is empty (0 bytes): {output_path}",
        )

    # Rule 4 — the file is there, but part of the content is not translated.
    if failed > 0 or unfinished > 0:
        return CompletionVerdict(
            status='partial',
            failed_chunks=failed,
            fallback_chunks=fallback,
            unfinished_chunks=unfinished,
            error=None,
        )

    # Rule 5 — nothing left to flag.
    return CompletionVerdict(
        status='completed',
        failed_chunks=failed,
        fallback_chunks=fallback,
        unfinished_chunks=unfinished,
        error=None,
    )
