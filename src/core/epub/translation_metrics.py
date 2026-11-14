"""Translation metrics and statistics tracking.

This module contains classes for tracking translation statistics and metrics.
Extracted from html_chunker.py as part of Phase 2 refactoring.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.core.progress import ProgressReporter


# Cross-file counters of an EPUB run that survive a resume, in the order and
# under the keys of the checkpoint's stored `epub_accumulated_stats` dict
# (issue #180). Per-file metrics are rehydrated by document_translation from the
# partial state instead. Changing a key breaks the resume of existing jobs.
ACCUMULATED_COUNTERS = (
    'token_alignment_used',
    'token_alignment_success',
    'fallback_used',
    'failed_chunks',
    'placeholder_errors',
    'processed_chunks',
    'successful_first_try',
    'successful_after_retry',
    'retry_attempts',
    'quality_warning_fired',
    'fallback_warning_fired',
    'paragraph_count_mismatches',
    'paragraph_retry_recovered',
    'paragraph_repair_failed',
    'correction_attempts',
    'correction_success',
    'total_tokens_processed',
    'total_tokens_generated',
    'refinement_chunks_completed',
)
_ACCUMULATED_FLAGS = frozenset({'quality_warning_fired', 'fallback_warning_fired'})

# Quality counters a metrics-driven progress emit carries next to the chunk
# counts (see TranslationMetrics.report_progress). The web UI reads them live
# (quality warning, fallback and paragraph-alignment cards), and the EPUB book
# loop folds the current file's values into the book-level payload.
PROGRESS_FIELDS = (
    'processed_chunks',
    'successful_first_try',
    'successful_after_retry',
    'fallback_used',
    'placeholder_errors',
    'quality_warning_fired',
    'token_alignment_used',
    'paragraph_count_mismatches',
    'paragraph_retry_recovered',
    'paragraph_repair_failed',
    'total_tokens_processed',
    'total_tokens_generated',
)


@dataclass
class TranslationMetrics:
    """Comprehensive translation metrics.

    Tracks counts, timing, token usage, and retry distribution.
    This is an enhanced version of TranslationStats for Phase 3 refactoring.

    Translation flow:
    1. Phase 1: Normal translation (with retry attempts)
    2. Phase 2: Token alignment fallback (translate without placeholders, reinsert proportionally)
    3. Phase 3: Untranslated fallback (if all retries fail, returns original text)

    Refinement support:
    - When refinement is enabled, total_chunks represents the ORIGINAL chunk count
    - enable_refinement flag tracks if this is a two-phase workflow
    - In refinement phase, use refinement_chunks_completed to track progress
    """
    # === Counts ===
    total_chunks: int = 0
    successful_first_try: int = 0
    successful_after_retry: int = 0
    fallback_used: int = 0  # Phase 3: Chunks returned untranslated after all phases failed
    failed_chunks: int = 0
    
    # === Progress tracking ===
    processed_chunks: int = 0  # Chunks fully processed (regardless of success/failure)
    # This is used for progress calculation to avoid fluctuations during retries

    # === Refinement tracking ===
    enable_refinement: bool = False  # If True, this is a two-phase workflow
    refinement_phase: bool = False  # If True, currently in refinement phase
    refinement_chunks_completed: int = 0  # Chunks completed in refinement phase

    # === Retry & Error Tracking ===
    retry_attempts: int = 0  # Total number of retry attempts made
    placeholder_errors: int = 0  # Total placeholder validation errors encountered
    quality_warning_fired: bool = False  # One-shot flag for the high-failure-rate warning
    fallback_warning_fired: bool = False  # Set once the first-fallback recommendation has been emitted
    # Last counts reported by check_fallback_warning, so we re-fire when totals grow
    _last_reported_token_alignment: int = 0
    _last_reported_fallback: int = 0

    # === Plain Text Mode paragraph alignment (issue #253) ===
    paragraph_count_mismatches: int = 0  # Segments the LLM returned with the wrong paragraph count
    paragraph_retry_recovered: int = 0  # ...of which one count-stating retry realigned, for 1 call
    paragraph_repair_failed: int = 0  # Segments still misaligned after the per-paragraph repair

    # === Phase 2: Token Alignment Fallback ===
    token_alignment_used: int = 0  # Phase 2: Token alignment fallback used
    token_alignment_success: int = 0  # Phase 2: Token alignment succeeded

    # === LLM Correction (legacy, kept for compatibility) ===
    correction_attempts: int = 0  # Total LLM correction attempts made
    correction_success: int = 0  # Successful LLM corrections

    # === Timing ===
    total_time_seconds: float = 0.0
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0

    # === Token Usage ===
    total_tokens_processed: int = 0
    total_tokens_generated: int = 0

    # === Retry Distribution ===
    retry_distribution: Dict[int, int] = field(default_factory=dict)
    """Map of retry_count -> number_of_chunks. Example: {0: 85, 1: 10, 2: 5}"""

    # === Chunk Size Stats ===
    min_chunk_size: int = field(default_factory=lambda: float('inf'))
    max_chunk_size: int = 0
    total_chunk_size: int = 0

    # === Per-chunk outcomes (issue #261) ===
    chunk_outcomes: Dict[int, str] = field(default_factory=dict)
    """Map of chunk_index -> chunk status constant from xhtml_translation_state.

    File-local: the indices are positions inside the current XHTML file's chunk
    list. The chunk loop consumes them to build the persisted `chunk_statuses`
    of the file's partial state.

    Deliberately NOT part of to_dict()/from_dict() and NOT merged by merge():
      - to_dict() is the serialized form of the metrics; the authoritative
        statuses live in XHTMLTranslationState.chunk_statuses. Shipping them
        twice would create two truths that can disagree.
      - merge() combines per-file metrics into job-wide totals, where these
        file-local indices would collide across files.

    Concurrency: entries are written from inside translate_chunk_with_fallback
    while several chunks are in flight. That is safe because each write is a
    single-key dict assignment in one asyncio event loop, with no `await`
    between reading and writing. A future *thread*-based executor would break
    this and would need explicit synchronisation.
    """

    def record_success(self, attempt: int, chunk_size: int) -> None:
        """Record successful translation.

        Args:
            attempt: Attempt number (0 = first try)
            chunk_size: Size of chunk in tokens
        """
        # Note: total_chunks is set by the caller before the chunk loop, not incremented here

        if attempt == 0:
            self.successful_first_try += 1
        else:
            self.successful_after_retry += 1

        # Update retry distribution
        self.retry_distribution[attempt] = self.retry_distribution.get(attempt, 0) + 1

        # Update chunk size stats
        self._update_chunk_stats(chunk_size)

    def record_fallback(self, chunk_size: int) -> None:
        """Record fallback usage (untranslated chunk returned).

        Args:
            chunk_size: Size of chunk in tokens
        """
        # Note: total_chunks is set by the caller before the chunk loop, not incremented here
        self.fallback_used += 1
        self._update_chunk_stats(chunk_size)

    def record_failure(self, chunk_size: int) -> None:
        """Record failed translation.

        Args:
            chunk_size: Size of chunk in tokens
        """
        # Note: total_chunks is set by the caller before the chunk loop, not incremented here
        self.failed_chunks += 1
        self._update_chunk_stats(chunk_size)
    
    def record_chunk_outcome(self, chunk_index: Optional[int], status: str) -> None:
        """Record the terminal outcome of one chunk (issue #261).

        Overwrites any previous status for that index: a retry supersedes the
        outcome of the earlier attempt on the same chunk.

        Args:
            chunk_index: Index of the chunk inside the current file's chunk
                list. None means "caller does not track indices" and the call
                is a no-op.
            status: One of the CHUNK_* constants of xhtml_translation_state.
        """
        if chunk_index is None:
            return
        self.chunk_outcomes[chunk_index] = status

    def record_processed(self) -> None:
        """Record that a chunk has been fully processed (success or failure).
        
        This is used for progress tracking to ensure the progress bar only moves forward.
        """
        self.processed_chunks += 1

    def _update_chunk_stats(self, chunk_size: int) -> None:
        """Update chunk size statistics."""
        self.min_chunk_size = min(self.min_chunk_size, chunk_size)
        self.max_chunk_size = max(self.max_chunk_size, chunk_size)
        self.total_chunk_size += chunk_size

    def finalize(self) -> None:
        """Finalize metrics (call when translation completes)."""
        self.end_time = time.time()
        self.total_time_seconds = self.end_time - self.start_time

    @property
    def avg_time_per_chunk(self) -> float:
        """Average time per chunk in seconds."""
        if self.total_chunks == 0:
            return 0.0
        return self.total_time_seconds / self.total_chunks

    @property
    def avg_chunk_size(self) -> float:
        """Average chunk size in tokens."""
        if self.total_chunks == 0:
            return 0.0
        return self.total_chunk_size / self.total_chunks

    @property
    def success_rate(self) -> float:
        """Success rate (excludes fallbacks)."""
        if self.total_chunks == 0:
            return 0.0
        successful = self.successful_first_try + self.successful_after_retry
        return successful / self.total_chunks

    @property
    def first_try_rate(self) -> float:
        """First-try success rate."""
        if self.total_chunks == 0:
            return 0.0
        return self.successful_first_try / self.total_chunks

    # === Quality warning thresholds (aggressive: warn early) ===
    _QUALITY_MIN_PROCESSED = 5
    _QUALITY_RETRY_RATE_THRESHOLD = 0.30
    _QUALITY_FALLBACK_RATE_THRESHOLD = 0.10
    _QUALITY_AVG_ERRORS_THRESHOLD = 1.0

    def check_quality_warning(self) -> Optional[str]:
        """Detect high placeholder-failure rate and return a one-shot warning.

        Returns a formatted warning string the first time the threshold is crossed,
        then None on every subsequent call (the metric is mutated to prevent reuse).
        Returns None until at least _QUALITY_MIN_PROCESSED chunks have been processed
        so early noise on tiny documents does not trigger a false positive.
        """
        if self.quality_warning_fired:
            return None
        if self.processed_chunks < self._QUALITY_MIN_PROCESSED:
            return None

        processed = self.processed_chunks
        not_first_try = self.successful_after_retry + self.token_alignment_used + self.fallback_used
        fallback_count = self.token_alignment_used + self.fallback_used

        retry_rate = not_first_try / processed
        fallback_rate = fallback_count / processed
        avg_errors = self.placeholder_errors / processed

        triggered = (
            retry_rate > self._QUALITY_RETRY_RATE_THRESHOLD
            or fallback_rate > self._QUALITY_FALLBACK_RATE_THRESHOLD
            or avg_errors > self._QUALITY_AVG_ERRORS_THRESHOLD
        )
        if not triggered:
            return None

        self.quality_warning_fired = True

        retry_pct = round(retry_rate * 100)
        fallback_pct = round(fallback_rate * 100)

        return (
            "⚠️ HIGH PLACEHOLDER FAILURE RATE — the LLM is struggling to preserve "
            f"structural tags. After {processed} chunks: {retry_pct}% needed retries, "
            f"{fallback_pct}% fell back to alignment/untranslated mode "
            f"(avg {avg_errors:.1f} placeholder errors per chunk).\n"
            "   This wastes compute and degrades output quality. "
            "Consider stopping and adjusting one of:\n"
            "   • Switch to a more capable LLM (placeholder preservation needs strong "
            "instruction-following)\n"
            "   • Reduce MAX_TOKENS_PER_CHUNK in .env (smaller chunks = fewer "
            "placeholders per call)\n"
            "   • Enable Plain Text Mode (EPUB / DOCX only) — feeds the LLM near-plain "
            "text instead of marked-up content, so translations come out much "
            "cleaner and virtually error-free (trade-off: very limited formatting "
            "in the output)"
        )

    def check_fallback_warning(self) -> Optional[str]:
        """Surface the Recommendations block whenever the fallback counts grow.

        Phase 2 (token alignment) and Phase 3 (untranslated) are quality regressions
        the user almost always wants to know about live — waiting for the final
        summary means a long run can be half-degraded before the user sees it.
        Re-emits whenever either counter increases so the displayed numbers stay
        current; the final summary still prints the cumulative block.
        """
        if self.token_alignment_used == 0 and self.fallback_used == 0:
            return None
        if (self.token_alignment_used == self._last_reported_token_alignment
                and self.fallback_used == self._last_reported_fallback):
            return None

        self._last_reported_token_alignment = self.token_alignment_used
        self._last_reported_fallback = self.fallback_used
        self.fallback_warning_fired = True

        lines = ["=== Recommendations ==="]
        if self.token_alignment_used > 0:
            lines.append(
                f"⚠️ {self.token_alignment_used} chunk(s) used token alignment "
                "fallback (Phase 2)."
            )
            lines.append(
                "   This can cause minor layout imperfections due to proportional "
                "tag repositioning."
            )
        if self.fallback_used > 0:
            lines.append(
                f"⚠️ {self.fallback_used} chunk(s) could not be translated "
                "(Phase 3 fallback)."
            )
            lines.append("   These chunks remain in the source language.")
        lines.extend([
            "",
            "To improve translation quality, consider:",
            "  • Using a more capable LLM model",
            "  • Reducing MAX_TOKENS_PER_CHUNK in .env (e.g., from 400 to 150)",
            "  • Enabling Plain Text Mode (EPUB / DOCX only) — feeds the LLM near-plain "
            "text instead of marked-up content, so translations come out much "
            "cleaner and virtually error-free (trade-off: very limited formatting "
            "in the output)",
        ])
        return "\n".join(lines)

    def progress_counts(self) -> "tuple[int, int, int]":
        """``(done, failed, total)`` chunk counts of this document.

        processed_chunks (not the success count) is the done count, so the
        bar does not fluctuate during retries. A document in its refinement
        phase reports its translation as complete.
        """
        done = self.total_chunks if self.refinement_phase else self.processed_chunks
        return done, self.failed_chunks, self.total_chunks

    def progress_fields(self) -> Dict[str, Any]:
        """The quality counters of a progress emit (see PROGRESS_FIELDS)."""
        return {name: getattr(self, name) for name in PROGRESS_FIELDS}

    def report_progress(self, progress: ProgressReporter) -> Dict[str, Any]:
        """Emit the current picture through ``progress``; returns the payload."""
        done, failed, total = self.progress_counts()
        progress.set_counts(done, failed=failed, total=total)
        return progress.emit(self.progress_fields())

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for serialization.

        This is the persisted form (the per-file partial state stores it and
        ``from_dict`` reads it back, so its keys must stay readable by old
        checkpoints), not the progress payload: progress goes through
        :meth:`report_progress`. ``completed_chunks`` mirrors
        :meth:`progress_counts`.
        """
        effective_total_chunks = self.total_chunks
        if self.refinement_phase:
            # Translation for this file is done; refinement does not advance
            # the main progress denominator.
            effective_completed = self.total_chunks
        else:
            effective_completed = self.processed_chunks

        return {
            "total_chunks": effective_total_chunks,
            "completed_chunks": effective_completed,
            "successful_first_try": self.successful_first_try,
            "successful_after_retry": self.successful_after_retry,
            "fallback_used": self.fallback_used,
            "failed_chunks": self.failed_chunks,
            "retry_attempts": self.retry_attempts,
            "placeholder_errors": self.placeholder_errors,
            "quality_warning_fired": self.quality_warning_fired,
            "fallback_warning_fired": self.fallback_warning_fired,
            "token_alignment_used": self.token_alignment_used,
            "token_alignment_success": self.token_alignment_success,
            "paragraph_count_mismatches": self.paragraph_count_mismatches,
            "paragraph_retry_recovered": self.paragraph_retry_recovered,
            "paragraph_repair_failed": self.paragraph_repair_failed,
            "correction_attempts": self.correction_attempts,
            "correction_success": self.correction_success,
            "total_time_seconds": self.total_time_seconds,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "avg_time_per_chunk": self.avg_time_per_chunk,
            "total_tokens_processed": self.total_tokens_processed,
            "total_tokens_generated": self.total_tokens_generated,
            "total_chunk_size": self.total_chunk_size,
            "avg_chunk_size": self.avg_chunk_size,
            "min_chunk_size": self.min_chunk_size if self.min_chunk_size != float('inf') else 0,
            "max_chunk_size": self.max_chunk_size,
            "success_rate": self.success_rate,
            "first_try_rate": self.first_try_rate,
            "retry_distribution": self.retry_distribution,
            # Add refinement info for debugging
            "enable_refinement": self.enable_refinement,
            "refinement_phase": self.refinement_phase,
            "refinement_chunks_completed": self.refinement_chunks_completed,
            # Progress tracking
            "processed_chunks": self.processed_chunks
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'TranslationMetrics':
        """Create TranslationMetrics instance from dictionary.

        Args:
            data: Dictionary containing metrics data

        Returns:
            TranslationMetrics instance
        """
        metrics = cls()

        # Basic counts
        metrics.total_chunks = data.get("total_chunks", 0)
        metrics.successful_first_try = data.get("successful_first_try", 0)
        metrics.successful_after_retry = data.get("successful_after_retry", 0)
        metrics.fallback_used = data.get("fallback_used", 0)
        metrics.failed_chunks = data.get("failed_chunks", 0)
        # Progress counter. progress_counts() reports done = processed_chunks,
        # so this MUST round-trip or the progress bar restarts near zero on
        # resume (it would then count only the chunks done in the new session,
        # not the restored prefix).
        metrics.processed_chunks = data.get("processed_chunks", 0)
        metrics.refinement_chunks_completed = data.get("refinement_chunks_completed", 0)

        # Retry & error tracking
        metrics.retry_attempts = data.get("retry_attempts", 0)
        metrics.placeholder_errors = data.get("placeholder_errors", 0)
        metrics.quality_warning_fired = data.get("quality_warning_fired", False)
        metrics.fallback_warning_fired = data.get("fallback_warning_fired", False)

        # Phase 2: Token alignment
        metrics.token_alignment_used = data.get("token_alignment_used", 0)
        metrics.token_alignment_success = data.get("token_alignment_success", 0)

        # Plain Text Mode paragraph alignment
        metrics.paragraph_count_mismatches = data.get("paragraph_count_mismatches", 0)
        metrics.paragraph_retry_recovered = data.get("paragraph_retry_recovered", 0)
        metrics.paragraph_repair_failed = data.get("paragraph_repair_failed", 0)

        # LLM correction
        metrics.correction_attempts = data.get("correction_attempts", 0)
        metrics.correction_success = data.get("correction_success", 0)

        # Timing
        metrics.total_time_seconds = data.get("total_time_seconds", 0.0)
        metrics.start_time = data.get("start_time", time.time())
        metrics.end_time = data.get("end_time", 0.0)

        # Token usage
        metrics.total_tokens_processed = data.get("total_tokens_processed", 0)
        metrics.total_tokens_generated = data.get("total_tokens_generated", 0)

        # Chunk size stats
        min_size = data.get("min_chunk_size", 0)
        metrics.min_chunk_size = min_size if min_size > 0 else float('inf')
        metrics.max_chunk_size = data.get("max_chunk_size", 0)
        metrics.total_chunk_size = data.get("total_chunk_size", 0)

        # Retry distribution
        retry_dist = data.get("retry_distribution", {})
        if isinstance(retry_dist, dict):
            # Convert string keys back to int if needed
            metrics.retry_distribution = {int(k): v for k, v in retry_dist.items()}

        return metrics

    def _pct(self, value: int) -> float:
        """Calculate percentage of total chunks."""
        if self.total_chunks == 0:
            return 0.0
        return round(value / self.total_chunks * 100, 1)

    def _pct_of(self, value: int, total: int) -> float:
        """Calculate percentage of a specific total."""
        if total == 0:
            return 0.0
        return round(value / total * 100, 1)

    def log_summary(self, log_callback=None) -> str:
        """Log comprehensive summary.

        Args:
            log_callback: Optional callback for logging

        Returns:
            Summary string
        """
        summary_lines = [
            "=== Translation Summary ===",
            f"Total chunks: {self.total_chunks}",
            f"Success 1st try: {self.successful_first_try} ({self._pct(self.successful_first_try)}%)",
            f"Success after retry: {self.successful_after_retry} ({self._pct(self.successful_after_retry)}%)",
            f"Total retry attempts: {self.retry_attempts}",
        ]

        # Phase 2 stats (token alignment)
        if self.token_alignment_used > 0:
            summary_lines.extend([
                f"Token alignment fallback used: {self.token_alignment_used} ({self._pct(self.token_alignment_used)}%)",
                f"Token alignment success: {self.token_alignment_success}/{self.token_alignment_used} ({self._pct_of(self.token_alignment_success, self.token_alignment_used)}%)",
            ])

        # Phase 3 stats (untranslated fallback)
        if self.fallback_used > 0:
            summary_lines.append(f"Untranslated chunks (Phase 3 fallback): {self.fallback_used} ({self._pct(self.fallback_used)}%)")

        # Placeholder error tracking
        if self.placeholder_errors > 0:
            summary_lines.extend([
                "",
                "=== Placeholder Issues ===",
                f"Placeholder validation errors: {self.placeholder_errors}",
            ])
            if self.correction_attempts > 0:
                summary_lines.append(f"LLM correction attempts: {self.correction_attempts} (success: {self.correction_success})")

        # Timing info (if finalized)
        if self.total_time_seconds > 0:
            summary_lines.extend([
                "",
                "=== Timing ===",
                f"Total time: {self.total_time_seconds:.2f}s",
                f"Avg per chunk: {self.avg_time_per_chunk:.2f}s",
            ])

        # Token usage (if tracked)
        if self.total_tokens_processed > 0 or self.total_tokens_generated > 0:
            summary_lines.extend([
                "",
                "=== Token Usage ===",
                f"Processed: {self.total_tokens_processed:,}",
                f"Generated: {self.total_tokens_generated:,}",
            ])

        # Chunk size stats (if tracked)
        if self.max_chunk_size > 0:
            summary_lines.extend([
                "",
                "=== Chunk Sizes ===",
                f"Min: {self.min_chunk_size if self.min_chunk_size != float('inf') else 0} tokens",
                f"Max: {self.max_chunk_size} tokens",
                f"Avg: {self.avg_chunk_size:.1f} tokens",
            ])

        # Retry distribution (if tracked)
        if self.retry_distribution:
            summary_lines.append("")
            summary_lines.append("=== Retry Distribution ===")
            for attempt, count in sorted(self.retry_distribution.items()):
                percentage = self._pct(count)
                summary_lines.append(f"  {attempt} retries: {count} chunks ({percentage}%)")

        # Recommendations
        if self.token_alignment_used > 0 or self.fallback_used > 0:
            summary_lines.extend([
                "",
                "=== Recommendations ===",
            ])

            if self.token_alignment_used > 0:
                summary_lines.append(
                    f"⚠️ {self.token_alignment_used} chunks used token alignment fallback (Phase 2)."
                )
                summary_lines.append(
                    "   This can cause minor layout imperfections due to proportional tag repositioning."
                )

            if self.fallback_used > 0:
                summary_lines.append(
                    f"⚠️ {self.fallback_used} chunks could not be translated (Phase 3 fallback)."
                )
                summary_lines.append(
                    "   These chunks remain in the source language."
                )

            summary_lines.extend([
                "",
                "To improve translation quality, consider:",
                "  • Using a more capable LLM model",
                "  • Reducing MAX_TOKENS_PER_CHUNK in .env (e.g., from 400 to 150)",
            ])

        summary = "\n".join(summary_lines)

        if log_callback:
            log_callback("translation_stats", summary)

        return summary

    def snapshot_accumulated(self) -> Dict[str, Any]:
        """The cross-file counters that must survive a resume.

        Stored by the EPUB checkpoint as ``epub_accumulated_stats``. Goes
        through dedicated fields rather than ``to_dict`` (a UI payload with
        derived values).
        """
        return {name: getattr(self, name) for name in ACCUMULATED_COUNTERS}

    def restore_accumulated(self, snapshot: Optional[Dict[str, Any]]) -> None:
        """Load counters captured by :meth:`snapshot_accumulated`.

        A missing snapshot (legacy checkpoint) leaves the metrics untouched; a
        missing key resets that counter to 0 (False for the warning flags).
        """
        if not snapshot:
            return
        for name in ACCUMULATED_COUNTERS:
            default = False if name in _ACCUMULATED_FLAGS else 0
            setattr(self, name, snapshot.get(name, default))

    def merge(self, other: 'TranslationMetrics') -> None:
        """Merge statistics from another TranslationMetrics instance.

        Args:
            other: Another TranslationMetrics instance to merge
        """
        self.total_chunks += other.total_chunks
        self.successful_first_try += other.successful_first_try
        self.successful_after_retry += other.successful_after_retry
        self.fallback_used += other.fallback_used
        self.failed_chunks += other.failed_chunks
        self.retry_attempts += other.retry_attempts
        self.placeholder_errors += other.placeholder_errors
        # If either side already warned the user, don't re-warn after merging.
        self.quality_warning_fired = self.quality_warning_fired or other.quality_warning_fired
        self.fallback_warning_fired = self.fallback_warning_fired or other.fallback_warning_fired
        self.token_alignment_used += other.token_alignment_used
        self.token_alignment_success += other.token_alignment_success
        self.paragraph_count_mismatches += other.paragraph_count_mismatches
        self.paragraph_retry_recovered += other.paragraph_retry_recovered
        self.paragraph_repair_failed += other.paragraph_repair_failed
        self.correction_attempts += other.correction_attempts
        self.correction_success += other.correction_success
        self.total_tokens_processed += other.total_tokens_processed
        self.total_tokens_generated += other.total_tokens_generated
        self.total_chunk_size += other.total_chunk_size
        
        # Merge refinement tracking (needed for accurate progress across multiple files)
        self.refinement_chunks_completed += other.refinement_chunks_completed
        
        # Merge progress tracking
        self.processed_chunks += other.processed_chunks

        # Merge min/max
        if other.min_chunk_size != float('inf'):
            self.min_chunk_size = min(self.min_chunk_size, other.min_chunk_size)
        self.max_chunk_size = max(self.max_chunk_size, other.max_chunk_size)

        # Merge retry distribution
        for attempt, count in other.retry_distribution.items():
            self.retry_distribution[attempt] = self.retry_distribution.get(attempt, 0) + count
