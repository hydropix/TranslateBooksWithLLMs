"""
Unit tests for proportional placeholder reinsertion

Note: We extract the function code to avoid circular import issues.
"""


def find_nearest_word_boundary(text: str, pos: int) -> int:
    """
    Find the nearest word boundary to the given position.
    Avoids cutting in the middle of a word.
    """
    if pos <= 0:
        return 0
    if pos >= len(text):
        return len(text)

    # If we're on a space, perfect
    if text[pos] == ' ':
        return pos

    # Find the nearest space (before or after)
    left = pos
    right = pos

    while left > 0 and text[left] != ' ':
        left -= 1

    while right < len(text) and text[right] != ' ':
        right += 1

    # Choose the nearest boundary
    if text[left] == ' ' and (right >= len(text) or text[right] != ' '):
        return left
    elif right < len(text) and text[right] == ' ':
        return right
    elif text[left] == ' ':
        return left

    return pos


def reinsert_placeholders(
    translated_text: str,
    positions: dict,
    placeholder_format: tuple = None
) -> str:
    """
    Reinsert placeholders at proportional positions.

    Args:
        translated_text: The translated text without placeholders
        positions: Dict mapping placeholder index to relative position (0.0-1.0)
        placeholder_format: Optional (prefix, suffix) tuple for placeholder format
                          If None, uses safe format [[N]]

    Returns:
        Text with placeholders reinserted at proportional positions
    """
    if not positions:
        return translated_text

    # Use provided format or default to safe
    if placeholder_format is None:
        prefix, suffix = "[[", "]]"
    else:
        prefix, suffix = placeholder_format

    text_length = len(translated_text)

    # Calculate absolute positions for each placeholder
    insertions = []
    for idx, rel_pos in positions.items():
        abs_pos = int(rel_pos * text_length)
        # Adjust to not cut a word (find nearest word boundary)
        abs_pos = find_nearest_word_boundary(translated_text, abs_pos)
        insertions.append((abs_pos, idx))

    # Sort by position (reverse order to insert without shifting)
    insertions.sort(key=lambda x: x[0], reverse=True)

    result = translated_text
    for abs_pos, idx in insertions:
        placeholder = f"{prefix}{idx}{suffix}"
        result = result[:abs_pos] + placeholder + result[abs_pos:]

    return result


class TestPlaceholderReinsertion:
    """Test proportional reinsertion of placeholders"""

    def test_reinsert_safe_format_default(self):
        """Should use safe format [[N]] by default"""
        translated = "Bonjour le monde"
        positions = {0: 0.0, 1: 0.5, 2: 1.0}

        result = reinsert_placeholders(translated, positions)

        assert "[[0]]" in result
        assert "[[1]]" in result
        assert "[[2]]" in result
        assert result.startswith("[[0]]")
        assert result.endswith("[[2]]")

    def test_reinsert_safe_format_explicit(self):
        """Should use safe format [[N]] when explicitly specified"""
        translated = "Bonjour le monde"
        positions = {0: 0.0, 1: 0.5, 2: 1.0}

        result = reinsert_placeholders(
            translated,
            positions,
            placeholder_format=("[[", "]]")
        )

        assert "[[0]]" in result
        assert "[[1]]" in result
        assert "[[2]]" in result

    def test_reinsert_simple_format(self):
        """Should use simple format [N] when specified"""
        translated = "Bonjour le monde"
        positions = {0: 0.0, 1: 0.5, 2: 1.0}

        result = reinsert_placeholders(
            translated,
            positions,
            placeholder_format=("[", "]")
        )

        assert "[0]" in result
        assert "[1]" in result
        assert "[2]" in result
        # Should NOT contain double brackets
        assert "[[0]]" not in result
        assert "[[1]]" not in result
        assert "[[2]]" not in result

    def test_empty_positions(self):
        """Should return original text when no positions"""
        translated = "Bonjour le monde"
        result = reinsert_placeholders(translated, {})
        assert result == translated

    def test_single_placeholder(self):
        """Should handle single placeholder"""
        translated = "Bonjour"
        positions = {0: 1.0}

        result = reinsert_placeholders(
            translated,
            positions,
            placeholder_format=("[", "]")
        )

        assert result == "Bonjour[0]"

    def test_proportional_positioning(self):
        """Should place placeholders at proportional positions"""
        translated = "0123456789"  # 10 chars
        positions = {
            0: 0.0,   # Position 0
            1: 0.5,   # Position 5
            2: 1.0    # Position 10
        }

        result = reinsert_placeholders(
            translated,
            positions,
            placeholder_format=("[", "]")
        )

        # Check relative positions
        idx0 = result.index("[0]")
        idx1 = result.index("[1]")
        idx2 = result.index("[2]")

        assert idx0 < idx1 < idx2

    def test_preserves_text_content(self):
        """Should preserve all original text"""
        translated = "Bonjour le monde"
        positions = {0: 0.3, 1: 0.7}

        result = reinsert_placeholders(translated, positions)

        # Remove placeholders and check text is preserved
        cleaned = result.replace("[[0]]", "").replace("[[1]]", "")
        assert cleaned == translated

    def test_realistic_translation_scenario(self):
        """Test with realistic translation content"""
        original_positions = {
            5: 0.0,    # Start
            6: 0.15,   # After first word
            7: 0.85,   # Near end
            8: 1.0     # End
        }

        translated_text = "Laura regardait par la fenêtre."

        # Test with simple format
        result_simple = reinsert_placeholders(
            translated_text,
            original_positions,
            placeholder_format=("[", "]")
        )

        assert "[5]" in result_simple
        assert "[6]" in result_simple
        assert "[7]" in result_simple
        assert "[8]" in result_simple

        # Test with safe format
        result_safe = reinsert_placeholders(
            translated_text,
            original_positions,
            placeholder_format=("[[", "]]")
        )

        assert "[[5]]" in result_safe
        assert "[[6]]" in result_safe
        assert "[[7]]" in result_safe
        assert "[[8]]" in result_safe
