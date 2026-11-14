"""Mutable state of one pass over a book, and the stats payload it emits.

``BookRunState`` holds what the file loop accumulates: counters, the
unfinished/untranslated/degraded chunk maps, the merged metrics, the per-run
baseline, the monotonic checkpoint pointer and the parsed documents to write
back. Its :class:`ProgressReporter` builds the book-level stats payload,
emitted per chunk and per file: the chunk counts from the book counters,
the quality fields from ``book_progress_fields``.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from lxml import etree

from src.core.progress import ProgressReporter

from .run_context import PreparedDocument
from .translation_metrics import TranslationMetrics


# Counters the UI's rate context (retry %, fallback %, avg placeholder errors)
# divides by or displays. They are the only ones that need a per-run twin: the
# accumulated versions are deliberately rehydrated across resume passes
# (issue #180, the Fallbacks card must not reset to zero), which makes any
# percentage derived from them a cross-pass average nobody asked for.
RUN_RATE_COUNTERS = (
    'processed_chunks',
    'successful_after_retry',
    'token_alignment_used',
    'fallback_used',
    'placeholder_errors',
)


def book_failed_chunks(acc, file_stats=None) -> int:
    """Failed chunks of the book: accumulated plus the current file's."""
    return acc.failed_chunks + (file_stats or {}).get('failed_chunks', 0)


def book_progress_fields(acc, file_stats=None, unfinished_units=None,
                         run_prior_counts=None, untranslated_units=None,
                         degraded_units=None, run_total_chunks=0,
                         run_is_repair=False, skipped_chunks=0):
    """The quality fields of the EPUB book-level stats payload.

    Single source of the cross-file payload shape, shared by the resume-initial
    emit, the per-chunk wrapper, and the post-file emit. The chunk counts are
    not here: ``BookRunState``'s ProgressReporter adds them. The cumulative
    counters come from ``acc`` (a TranslationMetrics) plus, when given, the
    current file's not-yet-merged ``file_stats`` (its progress payload).

    ``unfinished_units`` (issue #261, design decisions D8/D9) is the job-level
    ``{file_href: [chunk_index, ...]}`` map of chunks still owed. It is emitted
    unconditionally as ``unfinished_chunks`` (the flat count) and
    ``unfinished_files`` (the map itself, so the completion card can name the
    files - Phase 5 of the plan reuses this one field, no second source of
    truth). Callers pass a live reference; it is copied here so a later
    in-place mutation of the caller's dict never reaches back into an
    already-emitted payload. When the current file's own outcome has not yet
    been folded into the map (the per-chunk callback fires mid-file, before the
    file returns), the emitted count legitimately lags by one file's worth of
    work; the post-file emit folds it in first, so it is always exact.

    ``untranslated_units`` is the same map narrowed to the chunks that actually
    fell back to their source text (CHUNK_UNTRANSLATED only, never
    CHUNK_PENDING). Its total is emitted as ``untranslated_chunks`` and is what
    the live Fallbacks stat card counts, so a retry that heals a chunk makes the
    card go down. It is deliberately NOT the same number as
    ``unfinished_chunks``: an interrupted job owes every chunk it never reached,
    and none of those is a fallback.

    ``degraded_units`` is the job-level ``{file_href: [chunk_index, ...]}`` map
    of chunks that ARE translated but whose inline tags were only
    approximately repositioned by token alignment. Emitted unconditionally as
    ``degraded_chunks`` (the flat count) and ``degraded_files`` (the map), for
    the same key-by-key merge reason as the keys below. It is what the
    completion card must read instead of ``token_alignment_used``: that counter
    is an accumulated, cross-pass tally of Phase 2 *events* and stays positive
    after those chunks have been repaired, so gating anything on it would
    claim defects on a clean book. It never contributes to the verdict:
    ``unfinished_chunks`` stays free of these chunks (D3/D10).

    ``run_total_chunks`` / ``run_is_repair`` describe the scope of *this pass*
    so the live panel can report a repair pass as its own little job ("1 TOTAL
    / 1 COMPLETED") instead of the book it is patching ("12 TOTAL / 10
    COMPLETED, 83%"). ``run_total_chunks`` is the number of chunks the pass will
    attempt - known before the first one, so the bar has a denominator
    immediately - and ``run_processed_chunks`` below is its completed
    counterpart. ``run_is_repair`` is True only when the work set came from
    retry tickets; it is an explicit flag rather than a shape the frontend has
    to infer from the numbers. Both are live-payload only: nothing here is
    persisted, and ``total_chunks`` / ``completed_chunks`` keep their
    book-level meaning for the checkpoint and the resumable-job card.

    ``run_prior_counts`` is the part of those cumulative counters that belongs
    to *earlier* passes of a resumed job (see ``BookRunState.run_prior_counts``).
    Subtracting it yields the ``run_*`` twins the UI needs to express honest
    per-run percentages. They are emitted unconditionally - equal to the
    accumulated values on a fresh run, where nothing was restored - because
    ``state_manager.update_stats`` merges key by key: a key that appears only
    sometimes would leave a stale value behind.
    """
    fs = file_stats or {}
    uu = unfinished_units or {}
    ut = untranslated_units or {}
    du = degraded_units or {}
    prior = run_prior_counts or {}
    combined = {
        name: getattr(acc, name) + fs.get(name, 0)
        for name in RUN_RATE_COUNTERS
    }
    payload = {
        'token_alignment_used': combined['token_alignment_used'],
        'fallback_used': combined['fallback_used'],
        'placeholder_errors': combined['placeholder_errors'],
        'processed_chunks': combined['processed_chunks'],
        'successful_after_retry': combined['successful_after_retry'],
        'quality_warning_fired': acc.quality_warning_fired or fs.get('quality_warning_fired', False),
        # Plain Text Mode paragraph alignment (issue #253). Emitted for every
        # format: they are 0 on any path that is not Plain Text Mode, and a
        # counter nobody can read is a counter that does not exist.
        'paragraph_count_mismatches': (acc.paragraph_count_mismatches
                                       + fs.get('paragraph_count_mismatches', 0)),
        'paragraph_retry_recovered': (acc.paragraph_retry_recovered
                                      + fs.get('paragraph_retry_recovered', 0)),
        'paragraph_repair_failed': (acc.paragraph_repair_failed
                                    + fs.get('paragraph_repair_failed', 0)),
        'total_tokens': (acc.total_tokens_processed + acc.total_tokens_generated
                         + fs.get('total_tokens_processed', 0) + fs.get('total_tokens_generated', 0)),
        # Chunk-level unfinished work (issue #261, D8/D9). Emitted
        # unconditionally (0 / {} when there is nothing) because
        # state_manager.update_stats merges key by key: once these keys exist
        # they must be refreshed on every payload, or a stale non-zero value
        # would survive to the completion verdict.
        'unfinished_chunks': sum(len(v) for v in uu.values()),
        'unfinished_files': dict(uu),
        # Chunks currently sitting in their source text (Phase 3 fallback), as
        # opposed to chunks merely not reached yet. Emitted unconditionally for
        # the same key-by-key merge reason as above: the live Fallbacks card
        # trusts this key whenever it is present, so it must always be present
        # and always be fresh.
        'untranslated_chunks': sum(len(v) for v in ut.values()),
        # Chunks translated with approximate tag positions, as they stand right
        # now (not the accumulated Phase 2 event count). Emitted
        # unconditionally, same merge reason as above: the completion card
        # trusts these keys whenever they are present, so a repaired book must
        # see them refreshed to 0 / {} rather than keeping a stale value.
        'degraded_chunks': sum(len(v) for v in du.values()),
        'degraded_files': dict(du),
        # Scope of this pass, for the live progress panel (see the docstring).
        # Emitted unconditionally for the same key-by-key merge reason as the
        # keys above: a key that appears only on repair passes would leave a
        # stale True (or a stale denominator) behind on the next pass.
        'run_total_chunks': max(0, int(run_total_chunks or 0)),
        'run_is_repair': bool(run_is_repair),
        # Chunks with nothing to translate (passed through without an LLM
        # call). Static for a book; the job verdict subtracts them before
        # deciding that no unit at all was translated.
        'skipped_chunks': max(0, int(skipped_chunks or 0)),
    }
    # Per-run twins. Clamped at 0: a stale snapshot must never produce a
    # negative counter, whatever the arithmetic says.
    for name in RUN_RATE_COUNTERS:
        payload[f'run_{name}'] = max(0, combined[name] - prior.get(name, 0))
    return payload


@dataclass
class BookRunState:
    """What one pass over a book accumulates, file after file.

    Attributes:
        content_files: Spine hrefs, in reading order.
        opf_dir / temp_dir: Where the extracted book lives.
        total_chunks / chunks_per_file: The chunk precount of the source.
        skipped_chunks: Text-free chunks of the precount (no LLM call).
        prepared: Precount output per file index, handed to the file it was
            computed for and dropped once used.
        parsed_docs: file path -> document to write back.
        accumulated: Cross-file metrics, rehydrated on resume (issue #180).
        run_prior_counts: The part of the cumulative counters produced by
            earlier passes (baseline of the ``run_*`` twins).
        unfinished_units / untranslated_units / degraded_units: Job-level
            ``{href: [chunk_index, ...]}`` maps (issue #261, D8/D10).
        retry_tickets: Files below the resume pointer re-entered this pass.
        completed_chunks: Chunks done in files before the current one.
        run_total_chunks / run_is_repair: Scope of this pass.
        last_completed_file_idx: Monotonic checkpoint pointer.
        progress: Builds (and, once the file loop binds the job's stats
            callback, emits) the book-level payload.
    """

    content_files: List[str]
    opf_dir: str
    temp_dir: str
    total_chunks: int = 0
    skipped_chunks: int = 0
    chunks_per_file: List[int] = field(default_factory=list)
    prepared: Dict[int, PreparedDocument] = field(default_factory=dict)
    parsed_docs: Dict[str, etree._Element] = field(default_factory=dict)
    completed_files: int = 0
    failed_files: int = 0
    # Files kept in their source language because translating them failed.
    failed_doc_paths: Set[str] = field(default_factory=set)
    was_interrupted: bool = False
    accumulated: TranslationMetrics = field(default_factory=TranslationMetrics)
    run_prior_counts: Dict[str, int] = field(default_factory=dict)
    unfinished_units: Dict[str, List[int]] = field(default_factory=dict)
    untranslated_units: Dict[str, List[int]] = field(default_factory=dict)
    degraded_units: Dict[str, List[int]] = field(default_factory=dict)
    retry_tickets: Dict[str, List[int]] = field(default_factory=dict)
    completed_chunks: int = 0
    run_total_chunks: int = 0
    run_is_repair: bool = False
    last_completed_file_idx: int = -1
    progress: ProgressReporter = field(default_factory=ProgressReporter)

    @property
    def total_files(self) -> int:
        return len(self.content_files)

    def chunks_in_file(self, file_idx: int) -> int:
        return self.chunks_per_file[file_idx] if file_idx < len(self.chunks_per_file) else 0

    def take_prepared(self, file_idx: int) -> Optional[PreparedDocument]:
        """The file's precount output, released so it is not held any longer."""
        return self.prepared.pop(file_idx, None)

    def _sync_progress(self, completed_chunks: Optional[int],
                       file_stats: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Set the reporter's counts; return the payload's quality fields."""
        self.progress.set_counts(
            self.completed_chunks if completed_chunks is None else completed_chunks,
            failed=book_failed_chunks(self.accumulated, file_stats),
            total=self.total_chunks)
        return book_progress_fields(
            self.accumulated, file_stats,
            unfinished_units=self.unfinished_units,
            run_prior_counts=self.run_prior_counts,
            untranslated_units=self.untranslated_units,
            degraded_units=self.degraded_units,
            run_total_chunks=self.run_total_chunks,
            run_is_repair=self.run_is_repair,
            skipped_chunks=self.skipped_chunks)

    def payload(self, completed_chunks: Optional[int] = None,
                file_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """The book-level stats payload for the current picture (not emitted).

        ``completed_chunks`` defaults to the chunks of the finished files;
        ``file_stats`` is the current file's progress payload, folded in
        before the file is merged into ``accumulated``.
        """
        return self.progress.payload(self._sync_progress(completed_chunks, file_stats))

    def emit(self, completed_chunks: Optional[int] = None,
             file_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Emit :meth:`payload` through the bound reporter (unless unchanged)."""
        return self.progress.emit(self._sync_progress(completed_chunks, file_stats))

    def record_file_units(self, href: str, unfinished: List[int],
                          untranslated: List[int], degraded: List[int]) -> None:
        """Overwrite (or clear) one file's entry in the three chunk maps."""
        for units, indices in ((self.unfinished_units, unfinished),
                               (self.untranslated_units, untranslated),
                               (self.degraded_units, degraded)):
            if indices:
                units[href] = indices
            else:
                units.pop(href, None)

    @property
    def book_complete(self) -> bool:
        """Every file translated, no chunk owed, not interrupted."""
        return (not self.was_interrupted and not self.failed_files
                and not any(self.unfinished_units))
