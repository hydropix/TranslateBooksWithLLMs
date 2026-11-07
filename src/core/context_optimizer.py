"""
Context Optimization Module

Handles automatic estimation and adjustment of context size for LLM requests.
Now uses an adaptive strategy based on actual token usage instead of pre-estimation.

Strategy:
- Start with a small context (2048 by default)
- After each request, check if the context was near its limit
- If truncated or near limit: increase context and retry
- Track last N successful chunks to potentially reduce context if all fit with smaller size
"""

import re
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field
from collections import deque

from src.config import (
    MAX_TOKENS_PER_CHUNK, THINKING_MODELS,
    ADAPTIVE_CONTEXT_INITIAL, ADAPTIVE_CONTEXT_STEP, ADAPTIVE_CONTEXT_STABILITY_WINDOW
)

# Try to import tiktoken, fallback to character-based estimation
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


@dataclass
class ContextEstimation:
    """Result of context size estimation"""
    estimated_tokens: int
    prompt_length_chars: int
    estimation_method: str  # "tiktoken" or "character_based"
    language: str
    safety_margin_applied: bool


# Language-specific character-to-token ratios
CHAR_TO_TOKEN_RATIOS = {
    "english": 4.0,
    "french": 3.5,
    "spanish": 3.5,
    "german": 3.8,
    "italian": 3.6,
    "portuguese": 3.5,
    "russian": 3.2,
    "chinese": 1.5,
    "japanese": 2.0,
    "korean": 2.5,
    "arabic": 3.0,
}

# Safety margin for estimation (10% buffer)
SAFETY_MARGIN = 1.1

# Standard context sizes (powers of 2) for optimal Ollama performance
STANDARD_CONTEXT_SIZES = [2048, 4096, 8192, 16384, 32768, 65536, 131072]


def round_to_standard_context_size(required: int) -> int:
    """
    Round up to the nearest standard context size (power of 2).
    Ollama performs better with these standard sizes.

    Args:
        required: Minimum required context size

    Returns:
        Nearest standard context size >= required
    """
    for size in STANDARD_CONTEXT_SIZES:
        if size >= required:
            return size
    # If larger than all standard sizes, return as-is
    return required

# Default context size - most translations fit within 2048 tokens
DEFAULT_CONTEXT_SIZE = 2048

# Maximum context size limit (can be adjusted via OLLAMA_NUM_CTX in .env)
# Most modern models support at least 32K, so we use this as a safe upper bound
MAX_CONTEXT_SIZE = 131072


def estimate_tokens_with_margin(
    text: str,
    language: str = "english",
    apply_margin: bool = True
) -> ContextEstimation:
    """
    Estimate number of tokens in text with safety margin.

    Uses tiktoken if available (more accurate), otherwise falls back
    to character-based estimation with language-specific ratios.

    Args:
        text: The text to estimate
        language: Language of the text (affects character ratio)
        apply_margin: Whether to apply 10% safety margin

    Returns:
        ContextEstimation object with estimation details
    """
    prompt_length = len(text)

    # Method 1: tiktoken (preferred, ~90-95% accurate for Ollama models)
    if TIKTOKEN_AVAILABLE:
        try:
            # Use cl100k_base encoding (GPT-4 tokenizer)
            # Note: Not exact for Ollama models, but close enough
            encoder = tiktoken.get_encoding("cl100k_base")
            base_tokens = len(encoder.encode(text))

            estimated_tokens = int(base_tokens * SAFETY_MARGIN) if apply_margin else base_tokens

            return ContextEstimation(
                estimated_tokens=estimated_tokens,
                prompt_length_chars=prompt_length,
                estimation_method="tiktoken",
                language=language,
                safety_margin_applied=apply_margin
            )
        except Exception:
            # Fall through to character-based method
            pass

    # Method 2: Character-based estimation with language factors
    from src.utils.lang_normalize import normalize_lang_key
    lang_lower = normalize_lang_key(language)
    ratio = CHAR_TO_TOKEN_RATIOS.get(lang_lower, 4.0)  # Default to English

    base_tokens = prompt_length / ratio
    estimated_tokens = int(base_tokens * SAFETY_MARGIN) if apply_margin else int(base_tokens)

    return ContextEstimation(
        estimated_tokens=estimated_tokens,
        prompt_length_chars=prompt_length,
        estimation_method="character_based",
        language=language,
        safety_margin_applied=apply_margin
    )


def calculate_optimal_chunk_size(
    max_context_tokens: int,
    base_overhead: int = 2000,
    tokens_per_line: int = 23,
    min_chunk_size: int = 5,
    max_chunk_size: int = 100
) -> int:
    """
    Calculate optimal chunk size given context window constraints.

    Formula:
        Reserve 50% of context for output tokens
        Input budget = (max_context * 0.5) - base_overhead
        chunk_size = input_budget / tokens_per_line

    Args:
        max_context_tokens: Maximum context window size
        base_overhead: Base prompt overhead (instructions, formatting)
        tokens_per_line: Estimated tokens per line of content
        min_chunk_size: Minimum allowed chunk size
        max_chunk_size: Maximum allowed chunk size

    Returns:
        Optimal chunk size (number of lines)
    """
    # Reserve half the context for output
    input_budget = (max_context_tokens * 0.5) - base_overhead

    if input_budget <= 0:
        return min_chunk_size

    optimal_size = int(input_budget / tokens_per_line)

    # Clamp to min/max bounds
    optimal_size = max(min_chunk_size, min(max_chunk_size, optimal_size))

    return optimal_size


def adjust_parameters_for_context(
    estimated_tokens: int,
    current_num_ctx: int,
    current_chunk_size: int,
    model_name: str = "",
    min_chunk_size: int = 5,
    is_thinking_model: Optional[bool] = None
) -> Tuple[int, int, list[str]]:
    """
    Adjust num_ctx and/or chunk_size to fit prompt within context.

    Strategy:
        1. Priority: Increase num_ctx to next standard size (power of 2)
        2. Last resort: Reduce chunk_size if num_ctx would exceed MAX_CONTEXT_SIZE

    Args:
        estimated_tokens: Estimated prompt size in tokens
        current_num_ctx: Current context window setting (base from .env)
        current_chunk_size: Current chunk size (lines)
        model_name: Name of the model (used to detect thinking models if is_thinking_model not provided)
        min_chunk_size: Minimum allowed chunk size
        is_thinking_model: Explicit flag from runtime detection (overrides model_name check)

    Returns:
        Tuple of (adjusted_num_ctx, adjusted_chunk_size, warnings)
    """
    warnings = []
    adjusted_num_ctx = current_num_ctx
    adjusted_chunk_size = current_chunk_size

    # Check if current context is sufficient
    # Response can be up to 2x MAX_TOKENS_PER_CHUNK (for languages less efficient in tokenization)
    # + ~50 tokens for <Translated> tags
    response_buffer = (MAX_TOKENS_PER_CHUNK * 2) + 50

    # Add thinking buffer only for models that actually produce thinking output
    # Use explicit is_thinking_model if provided (from runtime detection), otherwise fall back to model name check
    if is_thinking_model is None:
        is_thinking_model = any(tm in model_name.lower() for tm in THINKING_MODELS)
    if is_thinking_model:
        thinking_buffer = 2000  # Conservative estimate for model's internal reasoning
        response_buffer += thinking_buffer

    required_ctx = estimated_tokens + response_buffer

    if required_ctx <= current_num_ctx:
        # All good, no adjustment needed
        return adjusted_num_ctx, adjusted_chunk_size, warnings

    # Step 1: Try to increase num_ctx to next standard size (power of 2)
    standard_ctx = round_to_standard_context_size(required_ctx)
    if standard_ctx <= MAX_CONTEXT_SIZE:
        adjusted_num_ctx = standard_ctx
        warnings.append(
            f"Automatically increased context window from {current_num_ctx} to {adjusted_num_ctx} tokens "
            f"to accommodate prompt size (~{estimated_tokens} tokens)."
        )
        return adjusted_num_ctx, adjusted_chunk_size, warnings

    # Step 2: Last resort - reduce chunk_size
    adjusted_num_ctx = MAX_CONTEXT_SIZE

    # Calculate what chunk_size would fit
    new_chunk_size = calculate_optimal_chunk_size(
        max_context_tokens=MAX_CONTEXT_SIZE,
        min_chunk_size=min_chunk_size
    )

    if new_chunk_size < current_chunk_size:
        adjusted_chunk_size = new_chunk_size
        warnings.append(
            f"Prompt too large even at maximum context ({MAX_CONTEXT_SIZE} tokens). "
            f"Automatically reduced chunk_size from {current_chunk_size} to {adjusted_chunk_size} lines."
        )
    else:
        warnings.append(
            f"WARNING: Prompt size ({estimated_tokens} tokens) may exceed maximum context. "
            f"Translation may fail or produce incomplete results."
        )

    return adjusted_num_ctx, adjusted_chunk_size, warnings


def validate_configuration(
    chunk_size: int,
    num_ctx: int,
    model_name: str = ""
) -> list[str]:
    """
    Validate translation configuration and return warnings/recommendations.

    Args:
        chunk_size: Configured chunk size
        num_ctx: Configured context window
        model_name: Model name (unused, kept for compatibility)

    Returns:
        List of warning/recommendation messages
    """
    warnings = []

    # Check if chunk_size is reasonable
    if chunk_size < 5:
        warnings.append(
            f"ℹ️  chunk_size ({chunk_size}) is very small. Translation may be slow.\n"
            f"   Consider increasing chunk_size for better performance."
        )

    if chunk_size > 100:
        # Estimate typical prompt size for this configuration
        estimated_prompt = 2000 + (chunk_size * 23) + 200
        min_recommended_ctx = estimated_prompt * 2
        warnings.append(
            f"ℹ️  chunk_size ({chunk_size}) is very large. Ensure num_ctx is sufficient.\n"
            f"   Minimum recommended num_ctx: {min_recommended_ctx} tokens"
        )

    return warnings


# Convenience function for logging
def format_estimation_info(estimation: ContextEstimation) -> str:
    """Format estimation details for logging"""
    margin_text = " (with 10% safety margin)" if estimation.safety_margin_applied else ""
    return (
        f"Estimated {estimation.estimated_tokens} tokens{margin_text} "
        f"using {estimation.estimation_method} method "
        f"({estimation.prompt_length_chars} characters, {estimation.language})"
    )


# =============================================================================
# ADAPTIVE CONTEXT MANAGER
# =============================================================================

# Use configuration values from config.py
CONTEXT_STEP = ADAPTIVE_CONTEXT_STEP
INITIAL_CONTEXT_SIZE = ADAPTIVE_CONTEXT_INITIAL
STABILITY_WINDOW = ADAPTIVE_CONTEXT_STABILITY_WINDOW

# Repetition-loop retries with a larger window allowed per chunk request (Ollama
# also raises a repetition loop for a stream that overflowed the window). Past
# this the loop is the model's, not the window's, and further growth only
# inflates the cost of every failed attempt.
MAX_REPETITION_RETRIES = 3

# Floor for the refinement pass: its prompt carries the draft chunk plus the
# previous and next chunks as context, so it is roughly 3x a translation prompt.
REFINEMENT_MIN_CONTEXT = 4096

# Threshold for considering context as "near limit" (95% usage)
NEAR_LIMIT_THRESHOLD = 0.95


@dataclass
class ChunkTokenUsage:
    """Token usage information for a single chunk"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    context_limit: int

    @property
    def usage_ratio(self) -> float:
        """Ratio of tokens used vs context limit"""
        if self.context_limit == 0:
            return 0.0
        return self.total_tokens / self.context_limit

    @property
    def is_near_limit(self) -> bool:
        """True if usage is close to the context limit"""
        return self.usage_ratio >= NEAR_LIMIT_THRESHOLD


class AdaptiveContextManager:
    """
    Manages context size adaptively based on actual token usage.

    Strategy:
    1. Start at INITIAL_CONTEXT_SIZE (2048)
    2. After each successful request, record token usage
    3. If usage is near limit (>=95%) or truncated: increase by CONTEXT_STEP and retry
    4. Track last STABILITY_WINDOW chunks
    5. If all recent chunks could fit in a smaller context: reduce by CONTEXT_STEP

    This avoids over-allocating context (which wastes VRAM) while ensuring
    translations complete successfully.
    """

    def __init__(self,
                 initial_context: int = INITIAL_CONTEXT_SIZE,
                 context_step: int = CONTEXT_STEP,
                 stability_window: int = STABILITY_WINDOW,
                 max_context: int = MAX_CONTEXT_SIZE,
                 log_callback: Optional[callable] = None):
        """
        Initialize the adaptive context manager.

        Args:
            initial_context: Starting context size (default: 2048)
            context_step: Amount to increase/decrease context (default: 2048)
            stability_window: Number of chunks to track for stability (default: 5)
            max_context: Maximum allowed context size (default: 131072)
            log_callback: Optional callback for logging
        """
        self.current_context = initial_context
        self.min_context = initial_context  # Never reduce below the initial context
        self.context_step = context_step
        self.stability_window = stability_window
        self.max_context = max_context
        self.log_callback = log_callback

        # Track token usage for recent chunks
        self._usage_history: deque = deque(maxlen=stability_window)

        # Track retry attempts for current chunk
        self._retry_count = 0
        self._max_retries = 10  # Safety limit

        # Per-request state (see begin_request / abandon_request). A request
        # that fails after growing the window must not leave the larger window
        # in place for every following chunk.
        self._request_start_context = initial_context
        self._repetition_retries = 0

    def get_context_size(self) -> int:
        """Get the current context size to use for the next request"""
        return self.current_context

    def record_success(self, prompt_tokens: int, completion_tokens: int, context_limit: int) -> None:
        """
        Record a successful translation with its token usage.

        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            context_limit: Context limit that was used
        """
        usage = ChunkTokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            context_limit=context_limit
        )
        self._usage_history.append(usage)
        self._retry_count = 0  # Reset retry count on success

        # Check if we can reduce context
        self._maybe_reduce_context()

    def should_retry_with_larger_context(self, was_truncated: bool, context_used: int) -> bool:
        """
        Determine if we should retry with a larger context.

        Args:
            was_truncated: True if the response was truncated
            context_used: Total tokens used in the request

        Returns:
            True if we should increase context and retry, False otherwise
        """
        # Check if we're at max context or max retries
        if self.current_context >= self.max_context:
            if self.log_callback:
                self.log_callback("context_adaptive",
                    f"⚠️ Already at maximum context ({self.max_context}), cannot increase further")
            return False

        if self._retry_count >= self._max_retries:
            if self.log_callback:
                self.log_callback("context_adaptive",
                    f"⚠️ Maximum retry attempts ({self._max_retries}) reached")
            return False

        # Check if context was near limit or truncated
        usage_ratio = context_used / self.current_context if self.current_context > 0 else 0
        near_limit = usage_ratio >= NEAR_LIMIT_THRESHOLD

        if was_truncated or near_limit:
            return True

        return False

    def begin_request(self) -> None:
        """Start the retry budget of one chunk request.

        Each chunk gets its own growth budget (``_max_retries`` steps, of
        which at most ``MAX_REPETITION_RETRIES`` for repetition loops), and
        the window it started with is remembered so a failed request can
        give back what it grew.
        """
        self._request_start_context = self.current_context
        self._retry_count = 0
        self._repetition_retries = 0

    def abandon_request(self) -> None:
        """Give back the context grown by a request that still failed.

        Growing the window did not help this chunk, so keeping it would only
        make every following request bigger and slower (and, on metered
        hosts, more expensive).
        """
        if self.current_context > self._request_start_context:
            old_context = self.current_context
            self.current_context = self._request_start_context
            if self.log_callback:
                self.log_callback("context_adaptive",
                    f"↩️ Request failed despite a larger context: restoring "
                    f"{old_context} → {self.current_context}")
        self._retry_count = 0
        self._repetition_retries = 0

    def should_retry_repetition_loop(self) -> bool:
        """Whether a repetition loop is worth one more retry with a larger window.

        A loop caused by a window too small for a thinking model goes away
        after one or two bigger windows. A loop that survives that is the
        model itself looping on this chunk, and growing the window further
        only makes each failed attempt more expensive.
        """
        if self._repetition_retries >= MAX_REPETITION_RETRIES:
            if self.log_callback:
                self.log_callback("context_adaptive",
                    f"⚠️ Repetition loop persists after {MAX_REPETITION_RETRIES} larger "
                    f"context(s): not a context size problem, giving up on this chunk")
            return False
        if not self.should_retry_with_larger_context(True, 0):
            return False
        self._repetition_retries += 1
        return True

    def increase_context(self) -> int:
        """
        Increase the context size by one step.

        Returns:
            The new context size
        """
        old_context = self.current_context
        self.current_context = min(self.current_context + self.context_step, self.max_context)
        self._retry_count += 1

        if self.log_callback:
            self.log_callback("context_adaptive",
                f"📈 Increasing context: {old_context} → {self.current_context} "
                f"(retry {self._retry_count}/{self._max_retries})")

        return self.current_context

    def _maybe_reduce_context(self) -> None:
        """
        Check if we can safely reduce the context size.

        We only reduce if:
        1. We have enough history (stability_window chunks)
        2. ALL recent chunks could have fit in a smaller context
        3. Current context is above the minimum (initial context for this session)
        """
        if len(self._usage_history) < self.stability_window:
            return  # Not enough history yet

        if self.current_context <= self.min_context:
            return  # Already at minimum (the initial context for this model type)

        # Calculate the maximum tokens used across all recent chunks
        max_tokens_used = max(usage.total_tokens for usage in self._usage_history)

        # Check if all chunks could fit in a smaller context
        smaller_context = max(self.current_context - self.context_step, self.min_context)

        # Don't reduce if we're already at minimum
        if smaller_context >= self.current_context:
            return

        # We need some headroom (20%) to avoid oscillation
        headroom_threshold = 0.80
        if max_tokens_used <= smaller_context * headroom_threshold:
            old_context = self.current_context
            self.current_context = smaller_context

            if self.log_callback:
                self.log_callback("context_adaptive",
                    f"📉 Reducing context: {old_context} → {self.current_context} "
                    f"(max usage was {max_tokens_used} tokens over last {self.stability_window} chunks)")

            # Clear history to start fresh tracking at new context level
            self._usage_history.clear()

    def reset(self) -> None:
        """Reset the manager to initial state"""
        self.current_context = self.min_context
        self._usage_history.clear()
        self._retry_count = 0
        self._request_start_context = self.min_context
        self._repetition_retries = 0

    def get_stats(self) -> Dict:
        """Get statistics about context usage"""
        if not self._usage_history:
            return {
                "current_context": self.current_context,
                "chunks_tracked": 0,
                "avg_usage": 0,
                "max_usage": 0,
                "min_usage": 0,
            }

        usages = [u.total_tokens for u in self._usage_history]
        return {
            "current_context": self.current_context,
            "chunks_tracked": len(self._usage_history),
            "avg_usage": sum(usages) / len(usages),
            "max_usage": max(usages),
            "min_usage": min(usages),
        }
