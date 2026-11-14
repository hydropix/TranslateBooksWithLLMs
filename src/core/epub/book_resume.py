"""Resume planning of a book pass: what earlier passes left behind.

Rehydrates the cross-file counters, grants the retry tickets (files below the
resume pointer that still owe chunks), and computes the progress arithmetic
of this pass (chunks already done, chunks this pass will attempt, the per-run
baseline). Everything reads the per-file partial states, which are the
payload; the stored maps are only their index.
"""

from typing import Any, Dict, List, Optional, Tuple

from .book_state import RUN_RATE_COUNTERS, BookRunState
from .run_context import EpubRun
from .xhtml_translation_state import (
    token_aligned_chunk_indices,
    unfinished_chunk_indices,
    untranslated_chunk_indices,
)


def load_partial_state(checkpoint_manager, translation_id, content_href: str) -> Any:
    """The file's XHTML partial state, or None (none, unreadable, no checkpoint)."""
    if not (checkpoint_manager and translation_id):
        return None
    try:
        return checkpoint_manager.load_xhtml_partial_state(translation_id, content_href)
    except Exception:
        return None


def chunk_projections(state) -> Tuple[List[int], List[int], List[int]]:
    """(unfinished, untranslated, degraded) chunk indices of a partial state.

    Three projections of one truth, not three sources of truth: the persisted
    `unfinished` set (pending + untranslated, D8), the persisted `degraded` set
    (token-aligned only) and the in-memory `untranslated`-only set all come
    from this single read of `chunk_statuses`, so they cannot drift apart.
    """
    statuses = state.chunk_statuses if state is not None else None
    return (unfinished_chunk_indices(statuses),
            untranslated_chunk_indices(statuses),
            token_aligned_chunk_indices(statuses))


def begin_pass(run: EpubRun, book: BookRunState) -> None:
    """Fill ``book`` with what this pass starts from, then emit it on resume."""
    resume_from_index = run.ctx.resume_from_index
    job_progress = _rehydrate_accumulated_stats(run, book)

    # Everything in the emitted cumulative counters that was NOT produced by
    # this pass. The UI divides by a per-run denominator, so it needs the
    # accumulated values minus this baseline; the accumulated values themselves
    # stay untouched (issue #180).
    #
    # It starts as the cross-file snapshot just restored, and grows by each
    # re-entered file's own restored counters: `document_translation` rebuilds a
    # file's TranslationMetrics from its XHTML partial state, so a re-entered
    # file's `file_stats` replay the chunks it already did in an earlier pass -
    # chunks the snapshot above *also* counts (they were merged into
    # `accumulated` before the pause). Without that second term the
    # per-run numbers would carry every re-entered file's prior work.
    book.run_prior_counts = {name: getattr(book.accumulated, name)
                             for name in RUN_RATE_COUNTERS}

    if resume_from_index > 0:
        _grant_retry_tickets(run, book, job_progress)

    book.completed_chunks = _completed_chunks_before(book, resume_from_index)
    book.run_is_repair = bool(book.retry_tickets)
    book.run_total_chunks = _run_total_chunks(run, book, resume_from_index)

    # The file index written to the checkpoint must never rewind. `chunk_index`
    # is turned back into `resume_from_index = current_chunk_index + 1` by
    # load_checkpoint, so writing the index of a re-entered *early* file (say
    # file 0 of 3, resumed at 3) would drop the pointer to 1 - and the next
    # resume would re-enter files 1 and 2 with no partial state, re-chunk their
    # already-translated bodies and translate them a second time. Tracking the
    # last completed index monotonically, floored at the pointer we resumed
    # from, is what prevents that. DO NOT replace this with `file_idx`.
    book.last_completed_file_idx = resume_from_index - 1

    # Send initial stats if resuming (to update UI immediately). Forward the
    # restored fallback counters so the UI hydrates the Fallbacks card with
    # the work already done before the pause, instead of showing 0.
    if run.ctx.stats_callback and resume_from_index > 0:
        run.ctx.emit(book.payload())


def _rehydrate_accumulated_stats(run: EpubRun, book: BookRunState) -> Dict[str, Any]:
    """Load the job's stored progress and its cross-file counters.

    On resume, rehydrate the cross-file fallback counters from the checkpoint
    so the Fallbacks stat card does not restart at zero (issue #180). Per-file
    metrics are still restored from the partial XHTML state inside
    document_translation. Returns the stored progress dict ({} when none).
    """
    if not (run.can_checkpoint and run.ctx.resume_from_index > 0):
        return {}
    try:
        job = run.checkpoint_manager.get_job(run.translation_id)
    except Exception:
        job = None
    if not job:
        return {}
    job_progress = job.get('progress') or {}
    book.accumulated.restore_accumulated(job_progress.get('epub_accumulated_stats'))
    # Only Plain Text Mode ever counted EPUB `failed_chunks`, and it now
    # records those segments as retryable fallbacks instead. A count
    # restored here was written before that: the segments kept their
    # source text and nothing recorded where they are, so no pass can
    # retry them and the job stays partial. Say so, once, rather than
    # letting a resume look like it silently ignored them (issue #285).
    if book.accumulated.failed_chunks:
        run.log("epub_legacy_failed_chunks",
                f"⚠️ {book.accumulated.failed_chunks} chunk(s) from an earlier "
                f"pass kept their source text but were not recorded for "
                f"retry, so they cannot be retried. Translate the book "
                f"again to fix them.")
    return job_progress


def _stored_units(job_progress: Dict[str, Any], key: str) -> Optional[Dict[str, List[int]]]:
    listed = job_progress.get(key)
    if not isinstance(listed, dict):
        return None
    return {str(href): list(indices or []) for href, indices in listed.items()}


def _grant_retry_tickets(run: EpubRun, book: BookRunState,
                         job_progress: Dict[str, Any]) -> None:
    """Re-entry tickets: files below the resume pointer that still hold
    unfinished chunks (issue #261).

    `unfinished_units` is the job-level index of remaining work
    ({file_href: [chunk_index, ...]}, design decision D8). It starts from what
    the job already recorded, so a file that is not re-entered during this
    pass keeps its ticket; every file this pass processes overwrites (or
    clears) its own entry, and the whole dict is written back - never merged.

    A ticket is only granted when the per-file partial state exists, validates
    and still reports unfinished chunks (D7). The state is the payload, the map
    is only the index. Without a state the file must stay skipped: the copy on
    disk was overwritten with its translated version by restore_epub_files, so
    re-entering would re-chunk an already-translated body and translate it a
    second time.

    The lookup is restricted to the hrefs the map lists, so a clean resume
    touches no extra file on disk. `untranslated_units` is the in-memory
    projection narrowed to the chunks that fell back to their source text (it
    feeds the live Fallbacks card and is deliberately NOT persisted).
    `degraded_units` lists the translated but approximately-tagged chunks;
    persisted next to `unfinished_units` and never merged into it (D3/D10).
    """
    book.unfinished_units = _stored_units(job_progress, 'epub_unfinished_units') or {}
    book.degraded_units = _stored_units(job_progress, 'epub_degraded_units') or {}
    # Both stored indices point at files whose partial state has to be read,
    # so the loop walks their union once. A file listed only as degraded is
    # loaded to refresh the map (and, in repair mode only, to grant it a
    # ticket); in the normal mode it never earns one.
    listed_hrefs = list(book.unfinished_units)
    listed_hrefs += [href for href in book.degraded_units
                     if href not in book.unfinished_units]
    for href in listed_hrefs:
        state = load_partial_state(run.checkpoint_manager, run.translation_id, href)
        indices, ticket_untranslated, ticket_degraded = chunk_projections(state)
        if ticket_untranslated:
            book.untranslated_units[href] = ticket_untranslated
        if state is not None:
            # The state is the payload, the map only the index: refresh the
            # index from what the state actually says, so a file repaired by
            # an earlier pass stops being listed.
            if ticket_degraded:
                book.degraded_units[href] = ticket_degraded
            else:
                book.degraded_units.pop(href, None)
        _grant_ticket(run, book, href, state, indices, ticket_degraded)


def _grant_ticket(run: EpubRun, book: BookRunState, href: str, state,
                  indices: List[int], ticket_degraded: List[int]) -> None:
    ticket = (sorted(set(indices) | set(ticket_degraded))
              if run.retry_token_aligned else list(indices))
    if ticket:
        book.retry_tickets[href] = ticket
        if run.retry_token_aligned and ticket_degraded:
            run.log("epub_retry_token_aligned_requested",
                    f"🔁 Retrying {len(ticket_degraded)} "
                    f"approximately-tagged chunk(s) of {href} on "
                    f"explicit request: {ticket_degraded}")
    elif state is None:
        run.log("epub_retry_state_missing",
                f"WARNING: {href} is listed as unfinished but has no "
                f"usable partial state - skipped (re-entering it "
                f"without one would re-translate a translated file)")


def _completed_chunks_before(book: BookRunState, resume_from_index: int) -> int:
    """Chunks of the files below the resume pointer that are already done.

    A file that is going to be re-entered must NOT be pre-counted here: the
    loop adds its full chunk count again once it has been processed, and
    counting it twice pushes the progress past 100%.
    """
    completed = 0
    for idx in range(resume_from_index):
        if idx < len(book.chunks_per_file) and book.content_files[idx] not in book.retry_tickets:
            completed += book.chunks_per_file[idx]
    return completed


def _run_total_chunks(run: EpubRun, book: BookRunState, resume_from_index: int) -> int:
    """Scope of THIS pass, for the live progress panel.

    A repair pass re-translates a handful of chunks of an otherwise finished
    book. Reporting it against the book ("12 TOTAL / 10 COMPLETED, 83%",
    creeping to 100% in tiny steps, ETA computed from a denominator that has
    nothing to do with the work in flight) describes nothing the user asked
    for, so the panel gets the pass's own numerator/denominator instead.

    The arithmetic, per file index:
      - below the resume pointer: only the chunks its retry ticket lists. The
        ticket IS the file's work set - `_translate_all_chunks_with_checkpoint`
        computes `unfinished_chunk_indices(statuses) | range(start, len)` and
        the second term is already inside the first (invariant D4: every chunk
        at or above `current_chunk_index` is `pending`, hence unfinished). A
        file with no ticket is skipped and contributes nothing.
      - at or above the pointer: its full pre-counted chunk count.
    Indexing by position rather than summing the ticket dict keeps the two
    terms disjoint, exactly like `_completed_chunks_before`.

    The file *at* the pointer is the one a previous pass was interrupted
    inside, so it carries a partial state and will attempt only its remaining
    chunks - fewer than `chunks_per_file` says. It has no ticket (a ticket is
    written by `save_file_checkpoint`, which only runs for files that
    completed), so its state has to be read directly. Only that one index can
    be in this situation: files above it were never touched.

    This matters whenever a book was interrupted *and* an earlier file left a
    fallback behind: the tickets make `run_is_repair` True, so the panel does
    use these numbers, and an over-counted denominator would leave the repair
    bar stalled below 100%.
    """
    total = 0
    for idx, href in enumerate(book.content_files):
        if idx < resume_from_index:
            total += len(book.retry_tickets.get(href) or ())
        elif idx >= len(book.chunks_per_file):
            continue
        elif idx == resume_from_index and resume_from_index > 0:
            total += pending_chunk_count(run.checkpoint_manager, run.translation_id,
                                         href, book.chunks_per_file[idx])
        else:
            total += book.chunks_per_file[idx]
    return total


def pending_chunk_count(checkpoint_manager, translation_id, content_href: str,
                        precounted: int) -> int:
    """How many chunks a file will actually attempt on this pass.

    A file that a previous pass was interrupted inside carries a partial state,
    so it resumes from `current_chunk_index` and attempts only what that state
    still reports as unfinished - fewer than the pre-counted total. Used for the
    repair-progress denominator, where over-counting would leave the bar stalled
    below 100%.

    Falls back to `precounted` whenever there is no usable state, which is the
    correct answer for a file that has never been entered.
    """
    state = load_partial_state(checkpoint_manager, translation_id, content_href)
    if state is None:
        return precounted
    return len(unfinished_chunk_indices(state.chunk_statuses))


def add_file_prior_counts(prior_counts: Dict, checkpoint_manager, translation_id,
                          content_href: str) -> None:
    """Fold one file's already-counted work into the per-run baseline.

    A file re-entered on a resume has its TranslationMetrics rebuilt from its
    XHTML partial state, so the `file_stats` it reports include the chunks it
    translated in an earlier pass. Those same chunks are already inside the
    restored cross-file snapshot, so they appear twice in the cumulative
    counters the payload emits; counting them once more in the baseline is what
    keeps the `run_*` twins equal to this pass's own work.

    Best effort by design: no state (fresh file), an unreadable one, or a state
    the translator ends up ignoring simply means nothing is subtracted, and the
    payload's `max(0, ...)` clamp absorbs the rest.
    """
    state = load_partial_state(checkpoint_manager, translation_id, content_href)
    stats = getattr(state, 'stats', None) if state is not None else None
    if not isinstance(stats, dict):
        return
    for name in RUN_RATE_COUNTERS:
        try:
            prior_counts[name] = prior_counts.get(name, 0) + int(stats.get(name, 0) or 0)
        except (TypeError, ValueError):
            continue
