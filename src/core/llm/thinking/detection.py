"""
Repetition loop detection for LLM responses.

This module detects when a model has entered a repetition loop, which typically
indicates the model has exceeded its effective context window or encountered an issue.
"""

from typing import Optional
from src.config import (
    REPETITION_MIN_PHRASE_LENGTH,
    REPETITION_MIN_COUNT,
    REPETITION_MIN_COUNT_THINKING
)


def detect_repetition_loop(
    text: str,
    min_phrase_length: Optional[int] = None,
    min_repetitions: Optional[int] = None,
    is_thinking_content: bool = False
) -> bool:
    """
    Detect if text contains a repetition loop pattern.

    This is common with thinking models (Qwen, DeepSeek) when context window is too small -
    they enter loops like "I'm not sure. I'm not sure. I'm not sure..."

    The detection uses different thresholds for:
    - Regular content: stricter detection (fewer repetitions needed)
    - Thinking content: more lenient (thinking models may naturally repeat phrases)

    Args:
        text: Text to analyze
        min_phrase_length: Minimum phrase length to detect (default from config)
        min_repetitions: Minimum number of repetitions to trigger detection (default from config)
        is_thinking_content: If True, uses more lenient thresholds for thinking model output

    Returns:
        True if repetition loop detected, False otherwise
    """
    # Use config defaults if not specified
    if min_phrase_length is None:
        min_phrase_length = REPETITION_MIN_PHRASE_LENGTH
    if min_repetitions is None:
        min_repetitions = REPETITION_MIN_COUNT_THINKING if is_thinking_content else REPETITION_MIN_COUNT

    if not text or len(text) < min_phrase_length * min_repetitions:
        return False

    # Check last portion of text for repetition patterns
    # Use a larger window for better detection
    check_text = text[-3000:] if len(text) > 3000 else text

    # Look for repeated phrases of various lengths
    # Longer phrases are more indicative of pathological loops
    for phrase_len in range(min_phrase_length, min(80, len(check_text) // min_repetitions)):
        # For longer phrases, we need fewer repetitions (they're more indicative of a loop)
        # Short phrases (5-10 chars) need more repetitions to avoid false positives
        # Longest first: a `>= 20` branch placed before `>= 40` would shadow it
        # and the strongest loop signal would never get the lenient threshold.
        adjusted_min_reps = min_repetitions
        if phrase_len >= 40:
            adjusted_min_reps = max(3, min_repetitions - 8)  # Very long phrases are very suspicious
        elif phrase_len >= 20:
            adjusted_min_reps = max(5, min_repetitions - 5)  # Longer phrases need fewer reps

        # Find potential repeating phrases
        for start in range(len(check_text) - phrase_len * adjusted_min_reps):
            phrase = check_text[start:start + phrase_len]

            # Skip if phrase is just whitespace or punctuation
            if not any(c.isalnum() for c in phrase):
                continue

            # Skip very common short phrases that might naturally repeat
            # These are normal in thinking and don't indicate a loop
            if phrase_len <= 10:
                common_phrases = ['the ', 'and ', 'to ', 'of ', 'in ', 'is ', 'it ', 'that ', 'for ']
                if phrase.lower().strip() in common_phrases:
                    continue

            # Count consecutive occurrences
            count = 1
            pos = start + phrase_len
            while pos + phrase_len <= len(check_text):
                if check_text[pos:pos + phrase_len] == phrase:
                    count += 1
                    pos += phrase_len
                else:
                    break

            if count >= adjusted_min_reps:
                return True

    return False
