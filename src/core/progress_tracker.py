"""
Token-based progress tracking for translation tasks.

Provides accurate progress calculation and time estimation based on actual token counts
rather than chunk counts, since translation time is proportional to token count.
"""

from dataclasses import dataclass
from time import time
from typing import Optional


@dataclass
class ProgressStats:
    """Immutable progress statistics snapshot."""
    total_tokens: int
    completed_tokens: int
    total_chunks: int
    completed_chunks: int
    failed_chunks: int
    progress_percent: float
    estimated_remaining_seconds: float
    elapsed_seconds: float
    avg_tokens_per_chunk: float
    current_token_rate: float  # tokens/second

    def to_dict(self) -> dict:
        """Convert to dictionary for backwards compatibility with existing code."""
        return {
            'total_tokens': self.total_tokens,
            'completed_tokens': self.completed_tokens,
            'total_chunks': self.total_chunks,
            'completed_chunks': self.completed_chunks,
            'failed_chunks': self.failed_chunks,
            'progress_percent': self.progress_percent,
            'estimated_remaining_seconds': self.estimated_remaining_seconds,
            'elapsed_seconds': self.elapsed_seconds,
            'avg_tokens_per_chunk': self.avg_tokens_per_chunk,
            'current_token_rate': self.current_token_rate
        }


class TokenProgressTracker:
    """
    Tracks translation progress based on token counts.

    Uses real-time calibration to provide accurate time estimates:
    - Fixed overhead: ~3s per chunk (prompt processing)
    - Variable time: proportional to token count
    - Auto-calibrates based on actual performance

    Supports two-phase workflows (translation + refinement):
    - Phase 1 (translation): 0-50% if refinement enabled, 0-100% otherwise
    - Phase 2 (refinement): 50-100% when enabled
    """

    FIXED_PROMPT_OVERHEAD = 3.0  # seconds per chunk
    DEFAULT_TOKEN_RATE = 0.01    # seconds per token (initial estimate)
    CALIBRATION_THRESHOLD = 5    # chunks needed before calibration kicks in
    CALIBRATION_SMOOTHING = 0.3  # weight for new data vs historical

    def __init__(self, enable_refinement: bool = False):
        """
        Initialize progress tracker.

        Args:
            enable_refinement: If True, progress is split 50/50 between translation and refinement
        """
        self._total_tokens = 0
        self._completed_tokens = 0
        self._total_chunks = 0
        self._completed_chunks = 0
        self._failed_chunks = 0
        self._chunk_tokens = []  # token count per chunk
        self._chunk_times = []   # actual elapsed time per chunk
        self._start_time: Optional[float] = None
        self._token_rate = self.DEFAULT_TOKEN_RATE
        self._enable_refinement = enable_refinement
        self._current_phase = 1  # 1 = translation, 2 = refinement

    def start(self):
        """Mark the start of translation."""
        self._start_time = time()

    def register_chunk(self, token_count: int):
        """Register a chunk with its token count before translation."""
        self._total_tokens += token_count
        self._total_chunks += 1
        self._chunk_tokens.append(token_count)

    def mark_completed(self, chunk_index: int, elapsed_time: float):
        """Mark a chunk as successfully translated."""
        if chunk_index >= len(self._chunk_tokens):
            raise ValueError(f"Invalid chunk index: {chunk_index}")

        self._completed_chunks += 1
        self._completed_tokens += self._chunk_tokens[chunk_index]
        self._chunk_times.append(elapsed_time)

        # Auto-calibrate after sufficient data
        if len(self._chunk_times) >= self.CALIBRATION_THRESHOLD:
            self._calibrate_token_rate()

    def mark_failed(self, chunk_index: int):
        """Mark a chunk as failed (still counts toward progress)."""
        if chunk_index >= len(self._chunk_tokens):
            raise ValueError(f"Invalid chunk index: {chunk_index}")

        self._failed_chunks += 1
        self._completed_chunks += 1
        self._completed_tokens += self._chunk_tokens[chunk_index]

    def start_refinement_phase(self):
        """Switch to refinement phase (resets counters for phase 2)."""
        self._current_phase = 2
        self._completed_tokens = 0
        self._completed_chunks = 0
        self._failed_chunks = 0
        self._chunk_times = []

    def get_progress_percent(self) -> float:
        """
        Calculate progress as percentage of total work completed.

        For two-phase workflows (enable_refinement=True):
        - Total work = total_tokens * 2 (translation + refinement)
        - Phase 1 (translation): returns 0-50% based on tokens translated
        - Phase 2 (refinement): returns 50-100% based on tokens refined

        For single-phase workflows:
        - Returns 0-100% based on tokens translated
        """
        if self._total_tokens == 0:
            return 0.0

        if not self._enable_refinement:
            # Single-phase: direct calculation
            return (self._completed_tokens / self._total_tokens) * 100

        # Two-phase workflow: total work is double (translate + refine)
        total_work_tokens = self._total_tokens * 2

        if self._current_phase == 1:
            # Translation phase: 0-50%
            # completed_tokens out of total_work_tokens (which is double)
            return (self._completed_tokens / total_work_tokens) * 100
        else:
            # Refinement phase: 50-100%
            # First phase already contributed 50%, now add refinement progress
            phase1_contribution = 50.0
            phase2_progress = (self._completed_tokens / self._total_tokens) * 50.0
            return phase1_contribution + phase2_progress

    def get_estimated_remaining_seconds(self) -> float:
        """
        Estimate remaining time based on token count and real performance.

        For two-phase workflows, accounts for remaining work in both phases.
        """
        if self._completed_chunks == 0:
            # Initial estimate before any real data
            total_work_chunks = self._total_chunks * 2 if self._enable_refinement else self._total_chunks
            total_work_tokens = self._total_tokens * 2 if self._enable_refinement else self._total_tokens
            return (self.FIXED_PROMPT_OVERHEAD * total_work_chunks) + \
                   (total_work_tokens * self._token_rate)

        # Use calibrated rate
        if not self._enable_refinement:
            # Single-phase: simple calculation
            remaining_tokens = self._total_tokens - self._completed_tokens
            remaining_chunks = self._total_chunks - self._completed_chunks
            return (self.FIXED_PROMPT_OVERHEAD * remaining_chunks) + \
                   (remaining_tokens * self._token_rate)

        # Two-phase workflow
        if self._current_phase == 1:
            # Still in translation phase - need to account for:
            # 1. Remaining translation work
            # 2. All refinement work (entire second phase)
            remaining_translation_tokens = self._total_tokens - self._completed_tokens
            remaining_translation_chunks = self._total_chunks - self._completed_chunks

            phase1_remaining = (self.FIXED_PROMPT_OVERHEAD * remaining_translation_chunks) + \
                              (remaining_translation_tokens * self._token_rate)

            # Phase 2 will process all chunks again
            phase2_total = (self.FIXED_PROMPT_OVERHEAD * self._total_chunks) + \
                          (self._total_tokens * self._token_rate)

            return phase1_remaining + phase2_total
        else:
            # In refinement phase - only refinement work remains
            remaining_refinement_tokens = self._total_tokens - self._completed_tokens
            remaining_refinement_chunks = self._total_chunks - self._completed_chunks

            return (self.FIXED_PROMPT_OVERHEAD * remaining_refinement_chunks) + \
                   (remaining_refinement_tokens * self._token_rate)

    def get_stats(self) -> ProgressStats:
        """Get immutable snapshot of current progress statistics."""
        elapsed = time() - self._start_time if self._start_time else 0.0

        return ProgressStats(
            total_tokens=self._total_tokens,
            completed_tokens=self._completed_tokens,
            total_chunks=self._total_chunks,
            completed_chunks=self._completed_chunks,
            failed_chunks=self._failed_chunks,
            progress_percent=self.get_progress_percent(),
            estimated_remaining_seconds=self.get_estimated_remaining_seconds(),
            elapsed_seconds=elapsed,
            avg_tokens_per_chunk=self._total_tokens / self._total_chunks if self._total_chunks > 0 else 0,
            current_token_rate=self._token_rate
        )

    def _calibrate_token_rate(self):
        """Auto-adjust token processing rate based on actual performance."""
        if not self._chunk_times or self._completed_tokens == 0:
            return

        total_time = sum(self._chunk_times)
        # Subtract fixed overhead to isolate variable (token-dependent) time
        variable_time = total_time - (self.FIXED_PROMPT_OVERHEAD * len(self._chunk_times))

        if variable_time <= 0:
            return

        measured_rate = variable_time / self._completed_tokens

        # Smooth adjustment to avoid oscillations
        self._token_rate = (self._token_rate * (1 - self.CALIBRATION_SMOOTHING)) + \
                          (measured_rate * self.CALIBRATION_SMOOTHING)
