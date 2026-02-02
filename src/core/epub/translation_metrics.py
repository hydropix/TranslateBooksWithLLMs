"""Translation metrics and statistics tracking.

This module contains classes for tracking translation statistics and metrics.
Extracted from html_chunker.py as part of Phase 2 refactoring.
"""

import time
from dataclasses import dataclass, field
from typing import Dict


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

    def record_success(self, attempt: int, chunk_size: int) -> None:
        """Record successful translation.

        Args:
            attempt: Attempt number (0 = first try)
            chunk_size: Size of chunk in tokens
        """
        # Note: total_chunks is initialized in _translate_all_chunks, not incremented here

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
        # Note: total_chunks is initialized in _translate_all_chunks, not incremented here
        self.fallback_used += 1
        self._update_chunk_stats(chunk_size)

    def record_failure(self, chunk_size: int) -> None:
        """Record failed translation.

        Args:
            chunk_size: Size of chunk in tokens
        """
        # Note: total_chunks is initialized in _translate_all_chunks, not incremented here
        self.failed_chunks += 1
        self._update_chunk_stats(chunk_size)
    
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

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for serialization.

        For two-phase workflows (translation + refinement):
        - total_chunks is doubled to reflect both phases
        - completed_chunks accounts for both translation and refinement progress
        - Phase 1 (translation): 0-50% of total work (0 to N chunks)
        - Phase 2 (refinement): 50-100% of total work (N to 2N chunks)
        
        Note: We use processed_chunks for translation progress to avoid fluctuations
        during retries. A chunk is only counted when fully processed (success or failure).
        """
        # Calculate total chunks and completed chunks based on refinement status
        if self.enable_refinement:
            # Two-phase workflow: double the total chunks
            effective_total_chunks = self.total_chunks * 2

            if self.refinement_phase:
                # In refinement phase: translation complete (N) + refinement progress
                effective_completed = self.total_chunks + self.refinement_chunks_completed
            else:
                # In translation phase: use processed_chunks to avoid retry fluctuations
                effective_completed = self.processed_chunks
        else:
            # Single-phase workflow: no adjustment needed
            effective_total_chunks = self.total_chunks
            # Use processed_chunks for consistent progress tracking
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
            "token_alignment_used": self.token_alignment_used,
            "token_alignment_success": self.token_alignment_success,
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

        # Retry & error tracking
        metrics.retry_attempts = data.get("retry_attempts", 0)
        metrics.placeholder_errors = data.get("placeholder_errors", 0)

        # Phase 2: Token alignment
        metrics.token_alignment_used = data.get("token_alignment_used", 0)
        metrics.token_alignment_success = data.get("token_alignment_success", 0)

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
        self.token_alignment_used += other.token_alignment_used
        self.token_alignment_success += other.token_alignment_success
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
