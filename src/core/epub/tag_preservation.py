"""
Tag preservation system for EPUB translation

This module handles the preservation of HTML/XML tags during translation by
replacing them with simple placeholders that LLMs won't modify.

Key features:
- Groups adjacent tags AND non-translatable content into single placeholders
- Strict validation ensures placeholder integrity post-translation
- Automatic mutation detection and correction
"""
import re
from typing import Dict, List, Tuple

from .initial_letters import close_initial_letters, merge_initial_letters
from .placeholder_validator import PlaceholderValidator
from src.common.placeholder_format import PlaceholderFormat


def is_non_translatable(text: str) -> bool:
    """
    Check if text contains only non-translatable content.

    Non-translatable content includes:
    - Whitespace only (spaces, tabs, newlines)
    - Non-breaking spaces and invisible Unicode characters
    - Simple numbers (digits, with optional dots/dashes for numbering)
    - Roman numerals (I, II, III, IV, etc.)

    Does NOT include:
    - Punctuation alone (could be meaningful)
    - Emojis or symbols
    - Any alphabetic text (except roman numerals)

    Args:
        text: Text to check

    Returns:
        True if text contains only non-translatable content
    """
    if not text:
        return True

    # Strip whitespace
    stripped = text.strip()
    if not stripped:
        return True  # Whitespace only

    # Check if it's just numbers/roman numerals with optional formatting
    # Matches: "1", "1.", "1.2", "42", "III", "IV.", "1-", "1)", "(1)"
    # Also matches invisible Unicode characters
    non_translatable_pattern = r'^[\d\.\-\–\—\)\(\s\u00A0\u2000-\u200F\u2028\u2029IVXLCDM]+$'
    return bool(re.match(non_translatable_pattern, stripped, re.IGNORECASE))

class TagPreserver:
    """
    Preserves HTML/XML tags during translation by replacing them with simple placeholders

    The TagPreserver converts tags like <p><span>text</span></p> into placeholders
    before translation, then restores them afterward. Adjacent tags are grouped
    into single placeholders to reduce token usage and LLM confusion.

    Unified placeholder format: [id0], [id1], [id2], ...

    Example:
        Input:  <p class="body"><span>Hello world</span></p>
        Output: "[id0]Hello world[id1]"
        Tag map: {"[id0]": "<p class=\"body\"><span>", "[id1]": "</span></p>"}
    """

    def __init__(self, protect_technical: bool = False):
        """
        Initialize TagPreserver.

        Args:
            protect_technical: If True, also protect technical content (code, LaTeX,
                             measurements, technical IDs) using the same placeholder system
        """
        self.tag_map: Dict[str, str] = {}
        self.counter: int = 0
        self.placeholder_format: PlaceholderFormat = PlaceholderFormat.from_config()
        self.protect_technical = protect_technical
        self._detector = None  # Lazy-load TechnicalContentDetector when needed

    def preserve_tags(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Replace HTML/XML tags with grouped placeholders.

        Tags and non-translatable content (whitespace, numbers) are merged into
        single placeholders to reduce the number of placeholders the LLM needs
        to preserve.

        Args:
            text: Text containing HTML/XML tags

        Returns:
            Tuple of (processed_text, tag_map)

        Example:
            >>> preserver = TagPreserver()
            >>> text, tag_map = preserver.preserve_tags("<p><span>Hello</span></p>")
            >>> text
            '[id0]Hello[id1]'

        Example with non-translatable content grouping:
            >>> preserver = TagPreserver()
            >>> text, tag_map = preserver.preserve_tags("<p> </p><p>1.</p><p>Hello</p>")
            >>> text
            '[id0]Hello[id1]'
            # [id0] contains "<p> </p><p>1.</p><p>" (empty paragraphs + chapter number grouped)
        """
        # Reset for new text
        self.tag_map = {}
        self.counter = 0

        text = merge_initial_letters(text)

        # Split text into segments: tags vs non-tags
        # This preserves the order and allows us to analyze each segment
        segments = re.split(r'(<[^>]+>)', text)

        # Build output by grouping tags and non-translatable content
        merged_segments = []
        current_group = []

        for segment in segments:
            if not segment:
                continue

            is_tag = segment.startswith('<') and segment.endswith('>')
            is_non_trans = is_non_translatable(segment)

            if is_tag or is_non_trans:
                # Add to current group (tags and non-translatable content)
                current_group.append(segment)
            else:
                # Found translatable text - flush the group as a placeholder
                if current_group:
                    merged_content = ''.join(current_group)
                    placeholder = self.placeholder_format.create(self.counter)
                    self.tag_map[placeholder] = merged_content
                    merged_segments.append(placeholder)
                    self.counter += 1
                    current_group = []
                # Add the translatable text directly
                merged_segments.append(segment)

        # Flush remaining group at the end
        if current_group:
            merged_content = ''.join(current_group)
            placeholder = self.placeholder_format.create(self.counter)
            self.tag_map[placeholder] = merged_content
            merged_segments.append(placeholder)
            self.counter += 1

        processed_text = ''.join(merged_segments)

        return processed_text, self.tag_map.copy()

    def preserve_tags_and_technical_content(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Replace HTML/XML tags AND technical content with grouped placeholders.

        This unified method handles both tag preservation and technical content protection
        in a single pass, using the same placeholder format [idN] for both types.

        Algorithm:
        1. Extract multiline technical blocks (```...```, $$...$$) → temporary markers
        2. Split text on HTML tags
        3. For each non-tag segment, split on inline technical patterns
        4. Group adjacent non-translatable segments (tags, technical, whitespace, numbers)
        5. Restore multiline blocks into final placeholders

        Args:
            text: Text containing HTML/XML tags and potentially technical content

        Returns:
            Tuple of (processed_text, tag_map)

        Example with technical content:
            Input:  "<p>The $V_{cm}$ voltage</p>"
            Output: ("[id0]The [id1] voltage[id2]",
                    {"[id0]": "<p>", "[id1]": "$V_{cm}$", "[id2]": "</p>"})

        Example with code block (atomic):
            Input:  "<p>Example:</p><pre><code>def f():\\n    pass</code></pre><p>Done</p>"
            Output: ("[id0]Example:[id1][id2]Done[id3]",
                    {"[id0]": "<p>", "[id1]": "</p><pre><code>def f():\\n    pass</code></pre>",
                     "[id2]": "<p>", "[id3]": "</p>"})
        """
        # Reset for new text
        self.tag_map = {}
        self.counter = 0

        # If technical protection is disabled, fallback to standard preserve_tags
        if not self.protect_technical:
            return self.preserve_tags(text)

        text = merge_initial_letters(text)

        # Step 1: Extract multiline blocks (code blocks, $$...$$)
        # These are replaced with temporary markers to keep them atomic
        text_with_markers, multiline_block_map = self._extract_multiline_blocks(text)

        # Step 2: Split on HTML tags
        tag_segments = re.split(r'(<[^>]+>)', text_with_markers)

        # Step 3 & 4: For each segment, split on technical patterns and group
        all_segments = []
        for segment in tag_segments:
            if not segment:
                continue

            is_tag = segment.startswith('<') and segment.endswith('>')

            if is_tag:
                # Keep tag as-is
                all_segments.append(segment)
            else:
                # Check if segment contains block markers - if so, split on them specially
                if '__TECH_BLOCK_' in segment:
                    # Split on block markers manually to preserve them
                    parts = re.split(r'(__TECH_BLOCK_\d+__)', segment)
                    all_segments.extend(parts)
                else:
                    # Split on inline technical patterns (code, LaTeX, measurements)
                    technical_split = self._split_on_technical_patterns(segment)
                    all_segments.extend(technical_split)

        # Step 4: Group adjacent non-translatable segments
        # BUT: Technical content should get its own placeholder (not grouped with tags)
        merged_segments = []
        current_group = []

        def flush_group():
            """Helper to flush current group as a placeholder."""
            if current_group:
                merged_content = ''.join(current_group)
                # Restore multiline blocks in the group
                for marker, block_content in multiline_block_map.items():
                    merged_content = merged_content.replace(marker, block_content)

                placeholder = self.placeholder_format.create(self.counter)
                self.tag_map[placeholder] = merged_content
                merged_segments.append(placeholder)
                self.counter += 1
                current_group.clear()

        for segment in all_segments:
            if not segment:
                continue

            is_tag = segment.startswith('<') and segment.endswith('>')
            is_non_trans = is_non_translatable(segment)
            is_tech = self._is_technical_content(segment)
            is_block_marker = segment.startswith('__TECH_BLOCK_')

            # Technical content and block markers get their own placeholders
            if is_tech or is_block_marker:
                # Flush any pending tag group
                flush_group()

                # Create dedicated placeholder for technical content
                merged_content = segment
                for marker, block_content in multiline_block_map.items():
                    merged_content = merged_content.replace(marker, block_content)

                placeholder = self.placeholder_format.create(self.counter)
                self.tag_map[placeholder] = merged_content
                merged_segments.append(placeholder)
                self.counter += 1
            elif is_tag or is_non_trans:
                # Add to current tag group
                current_group.append(segment)
            else:
                # Found translatable text - flush the tag group
                flush_group()
                # Add the translatable text directly
                merged_segments.append(segment)

        # Flush remaining group at the end
        flush_group()

        processed_text = ''.join(merged_segments)

        return processed_text, self.tag_map.copy()

    def restore_tags(self, text: str, tag_map: Dict[str, str]) -> str:
        """
        Restore HTML/XML tags from placeholders

        Args:
            text: Text with placeholders
            tag_map: Dictionary mapping placeholders to original tags

        Returns:
            Text with restored tags

        Example:
            >>> preserver = TagPreserver()
            >>> tag_map = {'[id0]': '<p><span>', '[id1]': '</span></p>'}
            >>> preserver.restore_tags('[id0]Hello[id1]', tag_map)
            '<p><span>Hello</span></p>'
        """
        restored_text = text

        # Sort placeholders by number in reverse order to avoid partial replacements
        # e.g., replace [id10] before [id1]
        placeholders = sorted(
            tag_map.keys(),
            key=lambda p: self.placeholder_format.parse(p) or 0,
            reverse=True
        )

        for placeholder in placeholders:
            if placeholder in restored_text:
                restored_text = restored_text.replace(placeholder, tag_map[placeholder])

        return close_initial_letters(restored_text)

    def validate_placeholders(self, text: str, tag_map: Dict[str, str]) -> Tuple[bool, List[str], List[Tuple[str, str]]]:
        """
        Validate that all expected placeholders are present in the text

        This function checks if the LLM preserved all placeholders during translation.

        Deprecated: Use PlaceholderValidator.validate_basic() instead.

        Args:
            text: Text to validate
            tag_map: Dictionary mapping placeholders to original tags

        Returns:
            Tuple of (is_valid, missing_placeholders, mutated_placeholders)
            - is_valid: True if all placeholders present
            - missing_placeholders: List of missing placeholder strings
            - mutated_placeholders: Empty list (legacy compatibility, no longer used)
        """
        # Use centralized PlaceholderValidator
        is_valid = PlaceholderValidator.validate_basic(text, tag_map)
        missing_placeholders = PlaceholderValidator.get_missing_placeholders(text, tag_map)
        mutated_placeholders = []  # Legacy compatibility

        return is_valid, missing_placeholders, mutated_placeholders

    def validate_placeholders_strict(
        self,
        translated_text: str,
        tag_map: Dict[str, str]
    ) -> Tuple[bool, str]:
        """
        Strict validation of placeholders post-translation.

        Validates:
        1. Correct number of placeholders
        2. Placeholders appear in sequential order (0, 1, 2, ...)

        Deprecated: Use PlaceholderValidator.validate_strict() instead.

        Args:
            translated_text: Text with placeholders after translation
            tag_map: Dictionary mapping placeholders to original tags

        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if validation passes
            - error_message: Description of the validation failure (empty if valid)
        """
        # Use centralized PlaceholderValidator
        return PlaceholderValidator.validate_strict(translated_text, tag_map)

    def fix_mutated_placeholders(self, text: str, mutated_placeholders: List[Tuple[str, str]]) -> str:
        """
        Attempt to fix common placeholder mutations

        Args:
            text: Text with mutated placeholders
            mutated_placeholders: List of (original, mutated) placeholder pairs

        Returns:
            Text with fixed placeholders
        """
        fixed_text = text
        for original, mutated in mutated_placeholders:
            fixed_text = fixed_text.replace(mutated, original)
        return fixed_text

    def _get_detector(self):
        """Lazy-load the TechnicalContentDetector."""
        if self._detector is None:
            from .technical_content_detector import TechnicalContentDetector
            self._detector = TechnicalContentDetector()
        return self._detector

    def _extract_multiline_blocks(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Extract multiline technical blocks (code blocks, LaTeX display) from text.

        These blocks must be kept atomic (never split across chunks), so they're
        extracted first and replaced with temporary markers.

        Args:
            text: Text containing potential multiline blocks

        Returns:
            Tuple of (text_with_markers, block_map)
            - text_with_markers: Text with blocks replaced by __TECH_BLOCK_N__ markers
            - block_map: Dict mapping markers to original block content

        Example:
            Input:  "Text before ```code\\nblock``` text after"
            Output: ("Text before __TECH_BLOCK_0__ text after",
                    {"__TECH_BLOCK_0__": "```code\\nblock```"})
        """
        if not self.protect_technical:
            return text, {}

        detector = self._get_detector()
        patterns = detector.find_all_technical_content(text)

        # Filter for multiline blocks only (priority 10)
        from .technical_content_detector import PatternPriority
        multiline_blocks = [
            p for p in patterns
            if p.priority == PatternPriority.MULTILINE_BLOCK
        ]

        if not multiline_blocks:
            return text, {}

        # Replace blocks from end to start to preserve positions
        block_map = {}
        result_text = text
        for i, block in enumerate(reversed(multiline_blocks)):
            marker = f"__TECH_BLOCK_{i}__"
            block_map[marker] = block.content
            result_text = result_text[:block.start] + marker + result_text[block.end:]

        return result_text, block_map

    def _split_on_technical_patterns(self, text: str) -> List[str]:
        """
        Split text on inline technical patterns (code, LaTeX, measurements).

        This creates segments where technical content is separated from regular text.

        Args:
            text: Text to split (should have multiline blocks already extracted)

        Returns:
            List of text segments, alternating between regular text and technical content

        Example:
            Input:  "The $V_{cm}$ voltage is `MAX1482` chip"
            Output: ["The ", "$V_{cm}$", " voltage is ", "`MAX1482`", " chip"]
        """
        if not self.protect_technical:
            return [text]

        detector = self._get_detector()
        patterns = detector.find_all_technical_content(text)

        # Filter out multiline blocks (already handled)
        from .technical_content_detector import PatternPriority
        inline_patterns = [
            p for p in patterns
            if p.priority < PatternPriority.MULTILINE_BLOCK
        ]

        if not inline_patterns:
            return [text]

        # Build segments
        segments = []
        last_end = 0

        for pattern in inline_patterns:
            # Add text before pattern
            if pattern.start > last_end:
                segments.append(text[last_end:pattern.start])

            # Add pattern itself
            segments.append(pattern.content)
            last_end = pattern.end

        # Add remaining text
        if last_end < len(text):
            segments.append(text[last_end:])

        return segments

    def _is_technical_content(self, segment: str) -> bool:
        """
        Check if a text segment is technical content.

        This is used during grouping to determine if a segment should be merged
        with adjacent tags into a single placeholder.

        Args:
            segment: Text segment to check

        Returns:
            True if segment contains technical content that should be protected

        Example:
            >>> self._is_technical_content("$V_{cm}$")
            True
            >>> self._is_technical_content("Hello world")
            False
        """
        if not self.protect_technical:
            return False

        if not segment or not segment.strip():
            return False

        # Check if entire segment matches a technical pattern
        detector = self._get_detector()
        patterns = detector.find_all_technical_content(segment.strip())

        # If we find patterns covering most/all of the segment, it's technical
        if patterns:
            # Calculate coverage
            total_coverage = sum(p.end - p.start for p in patterns)
            segment_length = len(segment.strip())
            coverage_ratio = total_coverage / segment_length if segment_length > 0 else 0

            # Consider it technical if patterns cover >80% of content
            return coverage_ratio > 0.8

        return False

