"""
Technical Content Detector for EPUB Translation

This module provides detection of technical content (code, LaTeX formulas, measurements,
technical identifiers, HTML entity blocks) that should be protected from LLM translation
by replacing them with placeholders.

The detector uses regex patterns with priority levels to identify and extract technical
content while avoiding false positives (e.g., distinguishing LaTeX formulas from currency).

Supported patterns:
- Code blocks (triple backticks)
- LaTeX formulas ($$...$$ and $...$)
- Inline code (`...`)
- Technical measurements (10 Mbps, 5V, etc.)
- Technical identifiers (TIA/EIA-485-A, DS1487, etc.)
- HTML entity blocks (&lt;section&gt;..., escaped code examples)
"""

import re
from dataclasses import dataclass
from typing import List, Optional
from enum import IntEnum


class PatternPriority(IntEnum):
    """Priority levels for pattern matching (higher = matched first)."""
    MULTILINE_BLOCK = 10  # Triple backticks, $$...$$
    HTML_ENTITY_BLOCK = 9 # Blocks of HTML entities (escaped code examples)
    INLINE_CODE = 5       # Single backticks, $...$
    MEASUREMENT = 3       # Numbers + units
    IDENTIFIER = 2        # Technical IDs like TIA/EIA-485-A


@dataclass
class TechnicalPattern:
    """Represents a detected technical content pattern."""
    start: int          # Start position in text
    end: int            # End position in text
    content: str        # Matched content
    pattern_name: str   # Name of pattern (e.g., "latex_inline", "code_block")
    priority: int       # Pattern priority (for overlap resolution)


class TechnicalContentDetector:
    """
    Detects technical content in text using regex patterns.

    The detector identifies various types of technical content:
    - Code blocks (triple backticks)
    - LaTeX formulas ($$...$$ and $...$)
    - Technical measurements (10 Mbps, 32 ULs)
    - Technical identifiers (TIA/EIA-485-A, DS1487)
    - HTML entity blocks (&lt;section&gt;..., escaped code examples)

    Patterns are applied with priority levels to handle overlaps correctly.
    """

    def __init__(self):
        """Initialize detector with pre-compiled regex patterns."""
        # Pattern 0: HTML entity blocks (escaped code examples in documentation)
        # Matches continuous blocks with multiple HTML entities like &lt;section&gt;
        # This captures code examples that are shown as escaped HTML
        # Strategy: Match from first entity to the last nearby entity (within reasonable distance)
        # Minimum 3 entities to ensure we're capturing real code examples
        self.html_entity_block_pattern = re.compile(
            r'&(?:lt|gt|amp|quot|apos|#\d+|#x[0-9a-fA-F]+);'  # First entity
            r'(?:(?!&(?:lt|gt|amp|quot|apos|#)).)*?'           # Non-entity chars (minimal)
            r'(?:&(?:lt|gt|amp|quot|apos|#\d+|#x[0-9a-fA-F]+);' # Second+ entity
            r'(?:(?!&(?:lt|gt|amp|quot|apos|#)).)*?){2,}'      # More entities
            r'&(?:lt|gt|amp|quot|apos|#\d+|#x[0-9a-fA-F]+);',  # Final entity
            re.DOTALL
        )

        # Pattern 1: Triple backtick code blocks (with optional language)
        # Matches: ```python\ncode\n```, ```\ncode\n```, or even ```code``` (inline)
        # Made more flexible to handle various newline situations
        self.code_block_pattern = re.compile(
            r'```[\w]*.*?```',
            re.DOTALL
        )

        # Pattern 2: LaTeX display math ($$...$$)
        # Matches: $$formula$$ (must not be empty)
        self.latex_display_pattern = re.compile(
            r'\$\$(.+?)\$\$',
            re.DOTALL
        )

        # Pattern 3: Single backtick inline code
        # Matches: `code`
        self.inline_code_pattern = re.compile(
            r'`[^`]+`'
        )

        # Pattern 4: LaTeX inline math ($...$)
        # Requires validation to avoid currency false positives
        # Matches: $formula$ but NOT across multiple $ signs
        # Use positive lookahead to ensure we don't match $...$...$
        self.latex_inline_pattern = re.compile(
            r'\$([^\$\n]+?)\$(?!\d)'  # Negative lookahead for digit after closing $
        )

        # Pattern 5: Technical measurements
        # Matches: 10 Mbps, 32 ULs, 3.5 MHz, 12V, 5mA, etc.
        self.measurement_pattern = re.compile(
            r'\b(\d+(?:\.\d+)?)\s*(ULs?|Mbps|kbps|Gbps|kHz|MHz|GHz|Hz|V|mV|mA|µA|uA|ms|ns|µs|us|Ω|ohm|W|kW|F|µF|uF|pF|H|mH|µH|uH)\b',
            re.IGNORECASE
        )

        # Pattern 6: Measurement ranges
        # Matches: +12 to -7 V, 0 to 100 mA
        self.measurement_range_pattern = re.compile(
            r'[+-]?\d+(?:\.\d+)?\s+to\s+[+-]?\d+(?:\.\d+)?\s*[A-Z]+\b',
            re.IGNORECASE
        )

        # Pattern 7: Technical identifiers
        # Matches: TIA/EIA-485-A, DS1487, MAX1482, RS-485
        # Improved to handle slashes and dashes better
        self.technical_id_pattern = re.compile(
            r'\b([A-Z]{2,}(?:[/-][A-Z\d]+)*(?:-[A-Z\d]+)?|[A-Z]+\d{3,})\b'
        )

    def _is_latex_formula(self, content: str) -> bool:
        """
        Determine if content within $...$ is a LaTeX formula or currency/variable.

        Heuristics:
        - Contains LaTeX indicators (_, ^, \\, {, }) → LaTeX formula
        - Simple number (5, 10.50) → Currency
        - Single word (price, total) → Variable/currency
        - Default → LaTeX formula

        Args:
            content: String content between $ signs (without the $)

        Returns:
            True if likely LaTeX formula, False if likely currency/variable

        Examples:
            >>> detector._is_latex_formula("V_{cm}")
            True  # Has subscript
            >>> detector._is_latex_formula("5")
            False  # Simple currency
            >>> detector._is_latex_formula("10.50")
            False  # Currency with cents
            >>> detector._is_latex_formula("price")
            False  # Variable name
            >>> detector._is_latex_formula("x^2 + y^2")
            True  # Has superscript
        """
        # Strong LaTeX indicators: subscript, superscript, backslash, braces
        if re.search(r'[_^\\{}]', content):
            return True

        # Simple number with optional cents → currency
        if re.match(r'^\d+(?:\.\d{1,2})?$', content):
            return False

        # Single word (letters only) → variable/currency reference
        if re.match(r'^[a-zA-Z]+$', content):
            return False

        # Multiple math operators or Greek letters → LaTeX
        if re.search(r'[+\-*/=<>].*[+\-*/=<>]', content):
            return True

        if re.search(r'\\(alpha|beta|gamma|delta|epsilon|theta|lambda|mu|sigma|pi|omega)', content):
            return True

        # Default: assume LaTeX (conservative - prefer protection)
        return True

    def _find_pattern_matches(
        self,
        text: str,
        pattern: re.Pattern,
        pattern_name: str,
        priority: int,
        validator: Optional[callable] = None
    ) -> List[TechnicalPattern]:
        """
        Find all matches of a pattern in text.

        Args:
            text: Text to search
            pattern: Compiled regex pattern
            pattern_name: Name identifier for this pattern
            priority: Priority level for overlap resolution
            validator: Optional function to validate match content

        Returns:
            List of TechnicalPattern objects for valid matches
        """
        matches = []
        for match in pattern.finditer(text):
            content = match.group(0)

            # Apply validator if provided
            if validator:
                # For LaTeX, extract content between $ signs
                if pattern_name == "latex_inline":
                    # Use group(1) to get captured group (content between $)
                    inner_content = match.group(1) if match.lastindex else content[1:-1]
                    if not validator(inner_content):
                        continue  # Skip false positive

            matches.append(TechnicalPattern(
                start=match.start(),
                end=match.end(),
                content=content,
                pattern_name=pattern_name,
                priority=priority
            ))

        return matches

    def _resolve_overlaps(self, patterns: List[TechnicalPattern]) -> List[TechnicalPattern]:
        """
        Resolve overlapping patterns by keeping higher priority ones.

        When two patterns overlap, the one with higher priority is kept.
        If priorities are equal, the longer pattern is kept.

        Args:
            patterns: List of detected patterns (may have overlaps)

        Returns:
            List of non-overlapping patterns sorted by position

        Example:
            Input: ```code with $latex$```
            - code_block (priority 10): covers entire range
            - latex_inline (priority 5): covers $latex$ only
            Result: code_block wins (higher priority)
        """
        if not patterns:
            return []

        # Sort by start position, then by priority (descending)
        sorted_patterns = sorted(patterns, key=lambda p: (p.start, -p.priority))

        result = []
        last_end = -1

        for pattern in sorted_patterns:
            # Check if this pattern overlaps with previously accepted pattern
            if pattern.start >= last_end:
                # No overlap, accept this pattern
                result.append(pattern)
                last_end = pattern.end
            else:
                # Overlap detected - check priorities
                last_pattern = result[-1]

                if pattern.priority > last_pattern.priority:
                    # Higher priority - replace last pattern
                    result[-1] = pattern
                    last_end = pattern.end
                elif pattern.priority == last_pattern.priority:
                    # Same priority - keep longer pattern
                    if (pattern.end - pattern.start) > (last_pattern.end - last_pattern.start):
                        result[-1] = pattern
                        last_end = pattern.end
                # else: lower or equal priority with shorter length - skip

        return result

    def find_all_technical_content(self, text: str) -> List[TechnicalPattern]:
        """
        Find all technical content in text with overlap resolution.

        Applies all patterns in priority order and resolves overlaps by keeping
        higher priority matches.

        Priority order (highest to lowest):
        1. Multiline blocks (```...```, $$...$$) - Priority 10
        2. HTML entity blocks (&lt;code&gt;...) - Priority 9
        3. Inline code/formulas (`...`, $...$) - Priority 5
        4. Measurements (10 Mbps, 32 ULs) - Priority 3
        5. Technical IDs (TIA/EIA-485-A) - Priority 2

        Args:
            text: Text to analyze

        Returns:
            List of TechnicalPattern objects sorted by start position

        Example:
            >>> text = "The $V_{cm}$ voltage is 10 Mbps using `MAX1482` chip."
            >>> patterns = detector.find_all_technical_content(text)
            >>> [p.pattern_name for p in patterns]
            ['latex_inline', 'measurement', 'inline_code']
        """
        all_patterns = []

        # Priority 10: Multiline code blocks
        all_patterns.extend(self._find_pattern_matches(
            text,
            self.code_block_pattern,
            "code_block",
            PatternPriority.MULTILINE_BLOCK
        ))

        # Priority 10: LaTeX display math
        all_patterns.extend(self._find_pattern_matches(
            text,
            self.latex_display_pattern,
            "latex_display",
            PatternPriority.MULTILINE_BLOCK
        ))

        # Priority 9: HTML entity blocks (escaped code examples)
        all_patterns.extend(self._find_pattern_matches(
            text,
            self.html_entity_block_pattern,
            "html_entity_block",
            PatternPriority.HTML_ENTITY_BLOCK
        ))

        # Priority 5: Inline code
        all_patterns.extend(self._find_pattern_matches(
            text,
            self.inline_code_pattern,
            "inline_code",
            PatternPriority.INLINE_CODE
        ))

        # Priority 5: LaTeX inline (with validation)
        all_patterns.extend(self._find_pattern_matches(
            text,
            self.latex_inline_pattern,
            "latex_inline",
            PatternPriority.INLINE_CODE,
            validator=self._is_latex_formula
        ))

        # Priority 3: Technical measurements
        all_patterns.extend(self._find_pattern_matches(
            text,
            self.measurement_pattern,
            "measurement",
            PatternPriority.MEASUREMENT
        ))

        # Priority 3: Measurement ranges
        all_patterns.extend(self._find_pattern_matches(
            text,
            self.measurement_range_pattern,
            "measurement_range",
            PatternPriority.MEASUREMENT
        ))

        # Priority 2: Technical identifiers
        all_patterns.extend(self._find_pattern_matches(
            text,
            self.technical_id_pattern,
            "technical_id",
            PatternPriority.IDENTIFIER
        ))

        # Resolve overlaps and return sorted by position
        return self._resolve_overlaps(all_patterns)

    def get_statistics(self, patterns: List[TechnicalPattern]) -> dict:
        """
        Generate statistics about detected patterns.

        Args:
            patterns: List of detected patterns

        Returns:
            Dictionary with pattern counts by type

        Example:
            >>> stats = detector.get_statistics(patterns)
            >>> stats
            {
                'total': 5,
                'code_block': 1,
                'latex_inline': 2,
                'measurement': 1,
                'inline_code': 1
            }
        """
        stats = {'total': len(patterns)}

        for pattern in patterns:
            pattern_type = pattern.pattern_name
            stats[pattern_type] = stats.get(pattern_type, 0) + 1

        return stats
