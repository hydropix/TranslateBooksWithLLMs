"""The checkpointed chunk loop of the placeholder pipeline.

``translate_chunks`` translates the chunks of one document (an EPUB XHTML
file or a DOCX body) with:

- an ordered-concurrent scheduler (up to ``parallel_workers`` requests at
  once, results delivered in index order);
- interruption checks before each chunk;
- periodic checkpoints of the partial state (every few chunks, at the last
  scheduled chunk, and when the pass stops early);
- resume from a partial state, including the retry of the chunks an earlier
  pass left in the source language (issue #261).

The two authoritative views of a pass live in :class:`ChunkSlots`.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from src.core.progress import ProgressReporter

from .chunk_fallback import translate_chunk_with_fallback
from .chunk_settings import ChunkLoopSettings
from .html_utils import is_text_free_chunk
from .placeholder_renumberer import globalize_placeholders
from .translation_metrics import TranslationMetrics
from .xhtml_translation_state import (
    CHUNK_PENDING,
    CHUNK_TOKEN_ALIGNED,
    CHUNK_TRANSLATED,
    CHUNK_UNTRANSLATED,
    token_aligned_chunk_indices,
    unfinished_chunk_indices,
)

CHECKPOINT_FREQUENCY = 5  # Save every 5 chunks


class ChunkSlots:
    """Per-chunk results and statuses of one pass, addressed by chunk index.

    `statuses` decides what gets translated and what gets persisted; the
    length of a growing list does not. A resumed pass gets the statuses from
    the partial state, so a chunk that fell back to its source text is
    re-translated even though it sits *below* start_chunk_index (issue #261).
    Without a statuses list (fresh start, or a caller that knows nothing
    about them) the legacy reading of the pointer applies: everything below it
    is translated, everything above is pending.

    A length mismatch means the incoming list does not describe the chunks (a
    re-chunked file, a truncated payload), and trusting it would retry
    arbitrary indices - so it is discarded rather than repaired.
    """

    def __init__(self, chunk_count: int, start_chunk_index: int = 0,
                 translated_chunks: Optional[List[str]] = None,
                 chunk_statuses: Optional[List[str]] = None):
        self.chunk_count = chunk_count
        if chunk_statuses is not None and len(chunk_statuses) == chunk_count:
            self.statuses = list(chunk_statuses)  # copy: never mutate the caller's list
        else:
            done_prefix = max(0, min(start_chunk_index, chunk_count))
            self.statuses = ([CHUNK_TRANSLATED] * done_prefix
                             + [CHUNK_PENDING] * (chunk_count - done_prefix))

        # Results are written by index, never appended (design decision D6): a
        # retry pass fills a hole in the middle of an otherwise complete list,
        # and appending would shift every chunk after it. `translated_chunks` is
        # an index-aligned prefix, so it pre-fills the slots as-is.
        self.slots: List[Optional[str]] = [None] * chunk_count
        for slot_index, slot_text in enumerate((translated_chunks or [])[:chunk_count]):
            self.slots[slot_index] = slot_text

        # Keep the two views in lockstep: a slot with text is not pending, and
        # a slot without text has nothing but 'pending' to say about itself.
        # Every state this module writes already satisfies both; a hand-edited
        # or truncated payload may not, and the loop must not act on a
        # contradiction (it is what keeps `len(translated_chunks) ==
        # current_chunk_index` true for validate(), invariant D4).
        for slot_index in range(chunk_count):
            if self.slots[slot_index] is None:
                self.statuses[slot_index] = CHUNK_PENDING
            elif self.statuses[slot_index] == CHUNK_PENDING:
                self.statuses[slot_index] = CHUNK_TRANSLATED

    def next_pending_index(self) -> int:
        """First never-attempted chunk, or the chunk count when there is none.

        This is what `current_chunk_index` means in the persisted state. It is
        NOT the number of delivered results any more: a retry pass translates a
        chunk below the pointer without moving it.
        """
        for index, status in enumerate(self.statuses):
            if status == CHUNK_PENDING:
                return index
        return self.chunk_count

    def completed_prefix(self) -> List[str]:
        """The delivered text, up to the first never-attempted chunk.

        Invariant D4 (any status other than 'pending' has text in the slot)
        makes this the same list as `[s for s in slots if s is not None]`.
        Stopping at the first hole is the shape validate() requires:
        len(translated_chunks) == current_chunk_index, with no hole below it.
        """
        prefix: List[str] = []
        for text in self.slots:
            if text is None:
                break
            prefix.append(text)
        return prefix

    def has_first_result(self) -> bool:
        return bool(self.slots) and self.slots[0] is not None

    def work_set(self, start_chunk_index: int, retry_token_aligned: bool) -> List[int]:
        """The chunks this pass schedules, in index order.

        The work set (design decision D5): everything never attempted from the
        resume pointer on, plus every chunk the previous passes left
        unfinished (a Phase 3 fallback below the pointer). One expression
        covers both mid-file interruption resume and post-completion retry.
        CHUNK_TOKEN_ALIGNED is deliberately absent: those chunks are
        translated (D3), retrying them would re-translate healthy content.

        `retry_token_aligned` is the one, explicit exception: the user asked
        from the completion card to retranslate those chunks. It is opt-in per
        pass and never inferred, so the automatic path keeps ignoring them.
        """
        work = (set(unfinished_chunk_indices(self.statuses))
                | set(range(start_chunk_index, self.chunk_count)))
        if retry_token_aligned:
            work |= set(token_aligned_chunk_indices(self.statuses))
        return sorted(work)

    def deliver(self, index: int, text: str, outcome: str) -> bool:
        """Store a delivered result; False when the previous text was kept.

        A repair pass asked to improve a chunk that was already translated
        (only its tag positions were approximate) and the retry came back with
        the source text (Phase 3). Overwriting would make the book WORSE than
        before the user pressed the button - and would move the verdict from
        'completed' to 'partial'. Keep what we had; the accumulated fallback
        counter still records that the attempt happened (D10).
        """
        if (self.statuses[index] == CHUNK_TOKEN_ALIGNED
                and outcome == CHUNK_UNTRANSLATED
                and self.slots[index] is not None):
            return False
        self.slots[index] = text
        self.statuses[index] = outcome
        return True


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + 'Z'


class _ChunkPass:
    """One run of :func:`translate_chunks` over a document's chunks."""

    def __init__(self, chunks: List[Dict], settings: ChunkLoopSettings,
                 placeholder_format: Tuple[str, str], global_tag_map: Optional[Dict[str, str]],
                 slots: ChunkSlots, stats: TranslationMetrics,
                 original_chunks: Optional[List[Dict]], progress: ProgressReporter):
        self.chunks = chunks
        self.settings = settings
        self.placeholder_format = placeholder_format
        self.global_tag_map = global_tag_map
        self.slots = slots
        self.stats = stats
        self.original_chunks = original_chunks
        self.progress = progress

    def save_state(self) -> None:
        """Persist the resume state for this file from the slots and statuses.

        Both persisted progress fields come from the two authoritative views,
        so they can never disagree: `current_chunk_index` is the first
        never-attempted chunk and `translated_chunks` is the text delivered
        below it.
        """
        settings = self.settings
        if not settings.can_checkpoint:
            return
        from .xhtml_translation_state import XHTMLTranslationState

        stats, chunks = self.stats, self.chunks
        global_stats_dict = None
        if settings.global_total_chunks is not None and settings.global_completed_chunks is not None:
            completed = stats.successful_first_try + stats.successful_after_retry
            global_stats_dict = {
                'total_chunks': settings.global_total_chunks,
                'completed_chunks': settings.global_completed_chunks + completed,
                'failed_chunks': stats.failed_chunks,
            }

        state = XHTMLTranslationState(
            file_path=settings.file_path or settings.file_href,
            translation_id=settings.translation_id,
            file_href=settings.file_href,
            source_language=settings.source_language,
            target_language=settings.target_language,
            model_name=settings.model_name,
            max_tokens_per_chunk=max(len(c.get('text', '')) for c in chunks) if chunks else 1000,
            max_retries=settings.max_retries,
            chunks=chunks,
            global_tag_map=self.global_tag_map or {},
            placeholder_format=self.placeholder_format,
            translated_chunks=self.slots.completed_prefix(),
            current_chunk_index=self.slots.next_pending_index(),
            original_body_html="",
            doc_metadata={},
            stats=stats.to_dict(),
            prompt_options=settings.prompt_options,
            bilingual=settings.bilingual,
            original_chunks=self.original_chunks,
            chunk_statuses=self.slots.statuses,
            created_at=_utc_now_iso(),
            updated_at=_utc_now_iso(),
            global_stats=global_stats_dict,
        )
        settings.checkpoint_manager.save_xhtml_partial_state(
            settings.translation_id, settings.file_href, state)

    async def translate_one(self, i: int) -> str:
        chunk = self.chunks[i]

        # A chunk with no translatable character (pure markup, a '==' separator,
        # a bare number, CJK ellipsis) has nothing for the LLM to do. Sending it
        # anyway invites an empty answer, and an empty answer used to wipe the
        # body — the cover page's <svg> image was destroyed this way. Pass it
        # through verbatim instead, restoring global indices exactly like the
        # untranslated fallback does.
        if is_text_free_chunk(chunk['text'], self.placeholder_format):
            self.settings.log("chunk_text_free_skipped",
                f"⏭️ Chunk {i + 1}/{len(self.chunks)} has no translatable text - kept as-is")
            # Same completed-chunk bookkeeping as a first-try success, minus the
            # LLM-call, token and retry counters (no request was made).
            self.stats.successful_first_try += 1
            self.stats.record_processed()
            self.stats.record_chunk_outcome(i, CHUNK_TRANSLATED)
            return globalize_placeholders(chunk['text'], chunk['global_indices'])

        return await translate_chunk_with_fallback(
            chunk['text'],
            chunk['local_tag_map'],
            chunk['global_indices'],
            self.settings,
            self.stats,
            self.placeholder_format,
            chunk_index=i,
        )

    def on_delivered(self, i: int, result: str, last_scheduled: int) -> None:
        """Record one delivered chunk, checkpoint and report progress."""
        settings, stats = self.settings, self.stats
        # A delivered result always reached a terminal outcome, so the default
        # only covers a caller that monkeypatches translate_chunk_with_fallback
        # with a fake recording nothing.
        outcome = stats.chunk_outcomes.get(i, CHUNK_TRANSLATED)
        if not self.slots.deliver(i, result, outcome):
            settings.log("chunk_retry_kept_previous",
                f"↩️ Chunk {i + 1}/{len(self.chunks)} retry fell back to the "
                f"source text - keeping the previous translation")

        # === HEALTH CHECK ===
        # Warn loudly once if the LLM is failing to preserve placeholders at
        # a high rate. Without this, retries pile up silently.
        quality_warning = stats.check_quality_warning()
        if quality_warning:
            settings.log("quality_warning", quality_warning)

        # === PERIODIC CHECKPOINT === (every N chunks and at the last one)
        # `last_scheduled` (pending[-1]) rather than `len(chunks) - 1`: a retry
        # pass whose last scheduled chunk sits in the middle of the file must
        # still persist the statuses it produced, otherwise the file loop reads
        # back a state that still calls the chunk unfinished.
        if (i + 1) % CHECKPOINT_FREQUENCY == 0 or i == last_scheduled:
            self.save_state()
            settings.log("xhtml_checkpoint_saved",
                f"💾 Checkpoint saved: chunk {i + 1}/{len(self.chunks)}")

        # Report progress after completing each chunk
        stats.report_progress(self.progress)

    async def run(self, pending: List[int]) -> int:
        """Translate the `pending` chunks; the number delivered.

        Continuous concurrency with in-order delivery: the delivered slots
        stay a contiguous prefix (resume-safe) while up to `workers` requests
        run at once. aclosing() is required: the rate-limit branch breaks out
        of the loop, and only closing the generator cancels the requests still
        in flight.
        """
        from src.core.common.parallel import aclosing, iter_ordered_concurrent
        from src.core.llm.exceptions import RateLimitError

        # EPUB chunks carry no cross-chunk translation context (the prompt is
        # built with previous_translation_context=""), so they parallelize
        # cleanly. parallel_workers is already resolved by the caller (local
        # providers forced to 1).
        workers = max(1, int(self.settings.parallel_workers))
        rate_limit_error = None
        delivered = 0
        async with aclosing(iter_ordered_concurrent(
            pending, workers, self.translate_one, self.settings.check_interruption
        )) as stream:
            async for i, result in stream:
                if isinstance(result, RateLimitError):
                    # Stop before recording this chunk so resume restarts at it.
                    rate_limit_error = result
                    break
                if isinstance(result, Exception):
                    # Non-rate-limit errors propagate as before (after persisting
                    # the contiguous prefix already delivered).
                    if self.slots.has_first_result():
                        self.save_state()
                    raise result
                self.on_delivered(i, result, pending[-1])
                delivered += 1

        if rate_limit_error is not None:
            # Persist the contiguous prefix delivered before the limit, then
            # propagate to let the caller pause/resume. current_chunk_index
            # still means "first never-attempted chunk", so the resumed pass
            # restarts at the chunk that hit the limit.
            if self.slots.has_first_result():
                self.save_state()
            raise rate_limit_error
        return delivered


async def translate_chunks(
    chunks: List[Dict],
    settings: ChunkLoopSettings,
    placeholder_format: Tuple[str, str],
    *,
    global_tag_map: Optional[Dict[str, str]] = None,
    start_chunk_index: int = 0,
    translated_chunks: Optional[List[str]] = None,
    chunk_statuses: Optional[List[str]] = None,
    stats: Optional[TranslationMetrics] = None,
    original_chunks: Optional[List[Dict]] = None,
) -> Tuple[List[str], TranslationMetrics, bool]:
    """Translate all chunks with checkpoint support.

    Args:
        chunks: Chunk dicts (``text``, ``local_tag_map``, ``global_indices``).
        settings: Languages, client, hooks, checkpoint target (see
            :class:`ChunkLoopSettings`).
        placeholder_format: ``(prefix, suffix)`` of the placeholders.
        global_tag_map: Global tag map (for state serialization).
        start_chunk_index: Index to start/resume from (default: 0).
        translated_chunks: Pre-existing translated chunks (for resume).
        chunk_statuses: Per-chunk statuses restored from the partial state
            (one entry per chunk, values from xhtml_translation_state's
            CHUNK_* constants). This is what makes a retry pass possible:
            chunks marked CHUNK_UNTRANSLATED are re-translated even though they
            sit below start_chunk_index (issue #261). None (or a list whose
            length does not match `chunks`) falls back to the legacy reading
            of start_chunk_index: everything below it is translated, the rest
            is pending.
        stats: Pre-existing stats (for resume).
        original_chunks: Source chunks (bilingual mode, persisted).

    Returns:
        Tuple of (translated_chunks, statistics, was_interrupted)
    """
    if stats is None:
        stats = TranslationMetrics()
        stats.total_chunks = len(chunks)
    slots = ChunkSlots(len(chunks), start_chunk_index, translated_chunks, chunk_statuses)

    # Report initial stats
    progress = ProgressReporter(settings.stats_callback)
    stats.report_progress(progress)

    chunk_pass = _ChunkPass(chunks, settings, placeholder_format, global_tag_map,
                            slots, stats, original_chunks, progress)
    pending = slots.work_set(start_chunk_index, settings.retry_token_aligned)
    delivered = await chunk_pass.run(pending)

    # Interruption: the scheduler stopped launching new chunks. The delivered
    # slots form a contiguous prefix, so the state saved here resumes exactly
    # where this pass stopped.
    next_index = slots.next_pending_index()
    interrupted = bool(next_index < len(chunks)
                       and settings.check_interruption
                       and settings.check_interruption())

    # Persist a pass that stopped before its last scheduled chunk. Two cases
    # reach here:
    #   - a genuine interruption, including one that happened before the first
    #     chunk was delivered (the file still needs a resume point);
    #   - a retry pass interrupted after fixing chunks that all sit below the
    #     resume pointer: nothing looks unfinished (next_index is already
    #     len(chunks)) yet the statuses this pass produced still have to reach
    #     disk, because the EPUB file loop reads them back to decide whether
    #     the file is clean.
    # A pass that delivered everything it scheduled was already persisted by
    # the periodic checkpoint above (its last save fires on pending[-1]), so
    # this never double-writes a completed pass.
    if interrupted or delivered < len(pending):
        chunk_pass.save_state()

    if interrupted:
        settings.log("xhtml_translation_interrupted",
            f"⏸️ Translation interrupted at chunk {next_index}/{len(chunks)}")
    return slots.completed_prefix(), stats, interrupted
