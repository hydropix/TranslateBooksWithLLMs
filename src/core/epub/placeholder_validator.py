"""
Centralized placeholder validation for EPUB translation.

This module provides unified validation logic for placeholder integrity checks.
"""

from typing import Dict, Tuple, List
import re


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
        3. Sequential order (0, 1, 2, ...)
        4. Valid format

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
