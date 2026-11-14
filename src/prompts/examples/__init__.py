"""
Multilingual examples for translation prompts.

This package provides technical examples (generated dynamically):
- Placeholder preservation (HTML/XML tags)
- Simple sentences that focus on WHAT to preserve

All examples use the actual constants from src/config.py to ensure consistency.
"""

# Re-export constants
from .constants import (
    TAG0,
    TAG1,
    TAG2,
)

# Re-export example dictionaries
from .placeholder_examples import PLACEHOLDER_EXAMPLES
from .subtitle_examples import SUBTITLE_EXAMPLES
from .output_examples import OUTPUT_FORMAT_EXAMPLES

# Re-export helper functions
from .helpers import (
    get_placeholder_example,
    get_subtitle_example,
    get_subtitle_refinement_example,
    get_output_format_example,
    get_refinement_output_example,
    build_placeholder_section,
    has_example_for_pair,
    ensure_example_ready,
)

__all__ = [
    # Constants
    "TAG0",
    "TAG1",
    "TAG2",
    # Technical example dictionaries (fallback)
    "PLACEHOLDER_EXAMPLES",
    "SUBTITLE_EXAMPLES",
    "OUTPUT_FORMAT_EXAMPLES",
    # Helper functions
    "get_placeholder_example",
    "get_subtitle_example",
    "get_subtitle_refinement_example",
    "get_output_format_example",
    "get_refinement_output_example",
    "build_placeholder_section",
    "has_example_for_pair",
    "ensure_example_ready",
]
