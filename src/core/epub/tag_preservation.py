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


# Numbering made only of digits and numbering punctuation, plus whitespace and
# invisible Unicode characters. No letters at all: "1", "1.", "1.2", "1-", "(1)".
_NUMERIC_NUMBERING_RE = re.compile(
    r'^[\d\.\-\u2013\u2014\)\(\s\u00A0\u2000-\u200F\u2028\u2029]+$'
)

# A strict, canonical, UPPERCASE Roman numeral (1..3999), optionally wrapped in
# parentheses or followed by "." / ")". Matching is case-sensitive on purpose:
# lowercase or mixed-case words built from the same letters ("did", "Civil",
# "vivid", "mild", "Mix", "dim", "lid") are real words and must be translated.
_ROMAN_CORE = r'M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})'
_ROMAN_PUNCTUATED_RE = re.compile(r'^(?:\(' + _ROMAN_CORE + r'\)|' + _ROMAN_CORE + r'[\.\)])$')
_ROMAN_BARE_RE = re.compile(r'^' + _ROMAN_CORE + r'$')

# Inline tags do not end a block of running text. Any other tag (p, div, h1-h6,
# li, td, br, section, ...) is treated as a block boundary when deciding whether
# a text segment stands alone in its block.
_INLINE_TAG_NAMES = frozenset({
    'a', 'abbr', 'b', 'bdi', 'bdo', 'big', 'cite', 'code', 'data', 'del',
    'dfn', 'em', 'font', 'i', 'ins', 'kbd', 'label', 'mark', 'q', 'rp', 'rt',
    'ruby', 's', 'samp', 'small', 'span', 'strike', 'strong', 'sub', 'sup',
    'time', 'tt', 'u', 'var', 'wbr',
})
_TAG_NAME_RE = re.compile(r'^<\s*/?\s*([A-Za-z][\w:.-]*)')


def is_non_translatable(text: str, standalone: bool = False) -> bool:
    """
    Check if text contains only non-translatable content.

    Non-translatable content includes:
    - Whitespace only (spaces, tabs, newlines)
    - Non-breaking spaces and invisible Unicode characters
    - Simple numbers (digits, with optional dots/dashes/parentheses for numbering)
    - Strict uppercase canonical Roman numerals, see the rule below

    Roman numeral rule (case-sensitive, canonical form only):
    - "IV.", "XII)", "(III)": always numbering (explicit numbering punctuation).
    - Bare "IV", "XII", "I": numbering only when ``standalone`` is True, i.e.
      the segment is the only text of its block (``<h2>IV</h2>``). Inside running
      prose a bare uppercase token may be a word (the pronoun "I", an all-caps
      "MIX" or "LI"), so it is left for the LLM. Sending a real numeral to the
      LLM is harmless; freezing a real word is not.
    - Lowercase or mixed case ("did", "Civil", "vivid", "Mix") is never numbering.

    Args:
        text: Text to check
        standalone: True when the caller knows the segment is the only text in
            its enclosing block (see ``_standalone_text_flags``).

    Returns:
        True if text contains only non-translatable content
    """
    if not text:
        return True

    stripped = text.strip()
    if not stripped:
        return True  # Whitespace only

    if _NUMERIC_NUMBERING_RE.match(stripped):
        return True

    if _ROMAN_PUNCTUATED_RE.match(stripped):
        return True

    if standalone and _ROMAN_BARE_RE.match(stripped):
        return True

    return False


def _is_block_boundary_tag(tag: str) -> bool:
    """Return True if ``tag`` (a full ``<...>`` string) is not an inline tag."""
    if tag.startswith('<!') or tag.startswith('<?'):
        return True
    match = _TAG_NAME_RE.match(tag)
    if not match:
        return True
    name = match.group(1).lower()
    if ':' in name:
        name = name.split(':', 1)[1]
    return name not in _INLINE_TAG_NAMES


def _standalone_text_flags(segments: List[str]) -> List[bool]:
    """
    For each segment, tell whether it is a text segment that is the only text
    of its block: walking left and right, only tags, empty or whitespace-only
    segments are met before a block-boundary tag (or the start/end of input).
    """
    def is_tag(seg: str) -> bool:
        return seg.startswith('<') and seg.endswith('>')

    def side_is_empty(indices) -> bool:
        for j in indices:
            seg = segments[j]
            if not seg:
                continue
            if is_tag(seg):
                if _is_block_boundary_tag(seg):
                    return True
                continue
            if seg.strip():
                return False
        return True

    flags = []
    for i, seg in enumerate(segments):
        if not seg or is_tag(seg) or not seg.strip():
            flags.append(False)
            continue
        flags.append(
            side_is_empty(range(i - 1, -1, -1))
            and side_is_empty(range(i + 1, len(segments)))
        )
    return flags

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
        standalone_flags = _standalone_text_flags(segments)

        for index, segment in enumerate(segments):
            if not segment:
                continue

            is_tag = segment.startswith('<') and segment.endswith('>')
            is_non_trans = is_non_translatable(segment, standalone_flags[index])

            if is_tag or is_non_trans:
                # Add to current group (tags and non-translatable content)
                current_group.append(segment)
            else:
                # Found translatable text - flush the group as a placeholder
                if current_group:
                    self._add_placeholder(''.join(current_group), merged_segments)
                    current_group = []
                # Add the translatable text directly
                merged_segments.append(segment)

        # Flush remaining group at the end
        if current_group:
            self._add_placeholder(''.join(current_group), merged_segments)

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

        # Steps 2 and 3: split on HTML tags, then on technical patterns
        all_segments = self._split_tags_and_technical(text_with_markers)

        # Step 4: Group adjacent non-translatable segments
        processed_text = self._group_technical_segments(all_segments, multiline_block_map)

        return processed_text, self.tag_map.copy()

    def _add_placeholder(self, content: str, merged_segments: List[str]) -> None:
        """Register `content` under the next placeholder and append that placeholder."""
        placeholder = self.placeholder_format.create(self.counter)
        self.tag_map[placeholder] = content
        merged_segments.append(placeholder)
        self.counter += 1

    @staticmethod
    def _restore_block_markers(content: str, multiline_block_map: Dict[str, str]) -> str:
        """Swap every temporary multiline-block marker back to its block."""
        for marker, block_content in multiline_block_map.items():
            content = content.replace(marker, block_content)
        return content

    def _split_tags_and_technical(self, text_with_markers: str) -> List[str]:
        """Split on HTML tags, then split each non-tag segment on technical content."""
        # Step 2: Split on HTML tags
        tag_segments = re.split(r'(<[^>]+>)', text_with_markers)

        # Step 3: For each segment, split on technical patterns
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
        return all_segments

    def _group_technical_segments(self, all_segments: List[str],
                                  multiline_block_map: Dict[str, str]) -> str:
        """Replace non-translatable runs with placeholders and join the result.

        Tags and non-translatable text are grouped, BUT technical content and
        block markers get their own placeholder (not grouped with tags).
        """
        merged_segments: List[str] = []
        current_group: List[str] = []

        def flush_group():
            """Helper to flush current group as a placeholder."""
            if current_group:
                # Restore multiline blocks in the group
                merged_content = self._restore_block_markers(
                    ''.join(current_group), multiline_block_map)
                self._add_placeholder(merged_content, merged_segments)
                current_group.clear()

        standalone_flags = _standalone_text_flags(all_segments)

        for index, segment in enumerate(all_segments):
            if not segment:
                continue

            is_tag = segment.startswith('<') and segment.endswith('>')
            is_non_trans = is_non_translatable(segment, standalone_flags[index])
            is_tech = self._is_technical_content(segment)
            is_block_marker = segment.startswith('__TECH_BLOCK_')

            # Technical content and block markers get their own placeholders
            if is_tech or is_block_marker:
                # Flush any pending tag group
                flush_group()

                # Create dedicated placeholder for technical content
                merged_content = self._restore_block_markers(segment, multiline_block_map)
                self._add_placeholder(merged_content, merged_segments)
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

        return ''.join(merged_segments)

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

