"""
Constants for EPUB translation processing

This module defines all magic numbers and configuration values used throughout
the EPUB translation pipeline.
"""

# Context window constants
MIN_CONTEXT_LINES = 3
"""Minimum number of lines to include in translation context"""

MIN_CONTEXT_WORDS = 25
"""Minimum number of words to include in translation context"""

MAX_CONTEXT_LINES = 5
"""Maximum number of lines to include in translation context"""

MAX_CONTEXT_BLOCKS = 10
"""Maximum number of translation blocks to keep in context accumulator"""

# Placeholder configuration - re-exported from src.config for backward compatibility
# The canonical definitions are in src/config.py to avoid circular imports
from src.config import (
    PLACEHOLDER_PREFIX,
    PLACEHOLDER_SUFFIX,
    PLACEHOLDER_PATTERN,
    create_placeholder,
    create_example_placeholder,
)
