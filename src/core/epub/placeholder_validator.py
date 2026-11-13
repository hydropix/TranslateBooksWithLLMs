"""
Centralized placeholder validation for EPUB translation.

This module provides unified validation logic for placeholder integrity checks.
"""

from typing import Dict, Tuple, List, Optional
import re


# Tags that never take a closing tag (HTML void elements). Self-closing XML
# tags ("<br/>") are recognised by their trailing slash.
_VOID_TAGS = frozenset({
    'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link',
    'meta', 'param', 'source', 'track', 'wbr',
})
_TAG_RE = re.compile(r'<[^>]+>')
_TAG_NAME_RE = re.compile(r'^<\s*(/?)\s*([A-Za-z][\w:.-]*)')


def nesting_signature(markup: str) -> Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
    """Simulate a tag stack over ``markup`` and describe its net effect.

    A chunk is a slice of a larger document, so it may legitimately close tags
    opened in an earlier chunk or leave tags open for a later one. The
    signature captures exactly that: the closing tags that reach outside the
    chunk (in order) and the tags still open at the end.

    Returns:
        ``(unmatched_closes, still_open)``, or None when the markup is
        mis-nested (a closing tag does not match the innermost open tag).
    """
    stack: List[str] = []
    outer_closes: List[str] = []
    for match in _TAG_RE.finditer(markup):
        tag = match.group(0)
        if tag.startswith('<!') or tag.startswith('<?') or tag.endswith('/>'):
            continue
        name_match = _TAG_NAME_RE.match(tag)
        if not name_match:
            continue
        is_close, name = name_match.group(1) == '/', name_match.group(2).lower()
        if name in _VOID_TAGS:
            continue
        if not is_close:
            stack.append(name)
        elif stack:
            if stack[-1] != name:
                return None
            stack.pop()
        else:
            outer_closes.append(name)
    return tuple(outer_closes), tuple(stack)


class PlaceholderValidator:
    """Validates placeholder integrity in translated text."""

    @staticmethod
    def validate_basic(text: str, expected_tag_map: Dict[str, str]) -> bool:
        """Quick validation: check all placeholders are present.

        Args:
            text: Text containing placeholders
            expected_tag_map: Map of placeholders to tags

        Returns:
            True if all placeholders present, False otherwise
        """
        for placeholder in expected_tag_map:
            if placeholder not in text:
                return False
        return True

    @staticmethod
    def validate_strict(
        text: str,
        expected_tag_map: Dict[str, str]
    ) -> Tuple[bool, str]:
        """Strict validation with detailed error messages.

        Checks:
        1. All expected placeholders are present
        2. No extra placeholders
        3. Every index 0..N-1 present exactly once
        4. Valid format
        5. Restored markup is well nested (see validate_structure)

        Args:
            text: Text containing placeholders
            expected_tag_map: Map of placeholders to tags

        Returns:
            Tuple of (is_valid, error_message)
            error_message is empty string if valid
        """
        # Extract placeholder pattern (e.g., "[id", "]")
        if not expected_tag_map:
            return True, ""

        # Get first placeholder to detect format
        first_ph = next(iter(expected_tag_map.keys()))
        match = re.match(r'^(.*?)(\d+)(.*)$', first_ph)
        if not match:
            return False, f"Invalid placeholder format: {first_ph}"

        prefix, _, suffix = match.groups()

        # 1. Check count
        expected_count = len(expected_tag_map)
        pattern = re.escape(prefix) + r'(\d+)' + re.escape(suffix)
        found_placeholders = re.findall(pattern, text)
        actual_count = len(found_placeholders)

        if actual_count != expected_count:
            return False, (
                f"Placeholder count mismatch: expected {expected_count}, "
                f"found {actual_count}"
            )

        # 2. Check sequential order
        indices = sorted([int(idx) for idx in found_placeholders])
        expected_indices = list(range(expected_count))

        if indices != expected_indices:
            missing = set(expected_indices) - set(indices)
            extra = set(indices) - set(expected_indices)
            return False, (
                f"Non-sequential placeholders. "
                f"Missing: {missing}, Extra: {extra}"
            )

        # 3. Check all expected placeholders present
        for placeholder in expected_tag_map:
            if placeholder not in text:
                return False, f"Missing placeholder: {placeholder}"

        # 4. Check the markup restored from the placeholders is well nested.
        # Reordering inline spans is legitimate (languages reorder words), so
        # the placeholder order itself is not enforced; but the order must not
        # produce mis-nested or unbalanced tags, e.g. "[id1]Bonjour[id0]"
        # for {"[id0]": "<p>", "[id1]": "</p>"}.
        structure_ok, structure_error = PlaceholderValidator.validate_structure(
            text, expected_tag_map
        )
        if not structure_ok:
            return False, structure_error

        return True, ""

    @staticmethod
    def validate_structure(
        text: str,
        expected_tag_map: Dict[str, str]
    ) -> Tuple[bool, str]:
        """Check that restoring placeholders in ``text`` yields well-nested markup.

        The tag sequence obtained by restoring placeholders in the order they
        appear in ``text`` must have the same nesting signature as the original
        sequence (placeholders in index order, which is source order), and must
        not be mis-nested.

        Args:
            text: Translated text containing placeholders
            expected_tag_map: Map of placeholders to the markup they stand for

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not expected_tag_map:
            return True, ""

        first_ph = next(iter(expected_tag_map.keys()))
        match = re.match(r'^(.*?)(\d+)(.*)$', first_ph)
        if not match:
            return False, f"Invalid placeholder format: {first_ph}"
        prefix, _, suffix = match.groups()

        def index_of(placeholder: str) -> int:
            m = re.match(r'^' + re.escape(prefix) + r'(\d+)' + re.escape(suffix) + r'$', placeholder)
            return int(m.group(1)) if m else -1

        original_markup = ''.join(
            expected_tag_map[ph] for ph in sorted(expected_tag_map, key=index_of)
        )
        original_signature = nesting_signature(original_markup)
        if original_signature is None:
            # The source slice itself cannot be judged; do not block on it.
            return True, ""

        pattern = re.escape(prefix) + r'\d+' + re.escape(suffix)
        translated_markup = ''.join(
            expected_tag_map.get(ph, '') for ph in re.findall(pattern, text)
        )
        translated_signature = nesting_signature(translated_markup)
        if translated_signature is None:
            return False, "Placeholder order produces mis-nested tags"
        if translated_signature != original_signature:
            return False, "Placeholder order produces unbalanced tags"
        return True, ""

    @staticmethod
    def get_missing_placeholders(
        text: str,
        expected_tag_map: Dict[str, str]
    ) -> List[str]:
        """Get list of missing placeholders.

        Args:
            text: Text to check
            expected_tag_map: Expected placeholders

        Returns:
            List of missing placeholder IDs
        """
        missing = []
        for placeholder in expected_tag_map:
            if placeholder not in text:
                missing.append(placeholder)
        return missing
