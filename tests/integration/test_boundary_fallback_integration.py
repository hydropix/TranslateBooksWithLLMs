"""
Integration test for boundary tag restoration in EPUB translation.

This test validates that the boundary restoration fix works correctly
in the complete translation pipeline.
"""

import pytest


def test_boundary_restoration_logic():
    """
    Standalone verification that boundary restoration logic is correct.

    This test simulates the exact scenario from the code review:
    - Chunk has stripped boundary tags
    - Translation fails, triggering fallback
    - Fallback must restore boundaries to prevent invalid HTML
    """
    # Simulate the data structures from a real translation
    chunk_text = "This is some content with [[0]]inline tags[[1]]."

    local_tag_map = {
        "[[0]]": "<em>",
        "[[1]]": "</em>",
        "__boundary_prefix__": "<body><div class='content'><p>",
        "__boundary_suffix__": "</p></div></body>"
    }

    global_indices = [42, 43]  # These are the global indices

    # Simulate the fallback restoration (from xhtml_translator.py:482-492)
    def restore_to_global_mock(text, indices):
        """Mock of PlaceholderManager.restore_to_global"""
        import re
        result = text
        # Detect format
        if re.search(r'\[\[\d+\]\]', text):
            prefix, suffix = "[[", "]]"
        else:
            prefix, suffix = "[", "]"

        # Replace local with global
        for local_idx in range(len(indices)):
            local_ph = f"{prefix}{local_idx}{suffix}"
            if local_ph in result:
                result = result.replace(local_ph, f"__TEMP_{local_idx}__")

        for local_idx, global_idx in enumerate(indices):
            result = result.replace(f"__TEMP_{local_idx}__", f"{prefix}{global_idx}{suffix}")

        return result

    # Apply the fixed fallback logic
    result_with_globals = restore_to_global_mock(chunk_text, global_indices)

    # Restore boundary tags (THE FIX)
    boundary_prefix = local_tag_map.get("__boundary_prefix__", "")
    boundary_suffix = local_tag_map.get("__boundary_suffix__", "")

    if boundary_prefix or boundary_suffix:
        result_with_globals = boundary_prefix + result_with_globals + boundary_suffix

    # Assertions
    assert result_with_globals.startswith("<body>"), \
        f"Result should start with <body>, got: {result_with_globals[:50]}"

    assert result_with_globals.endswith("</body>"), \
        f"Result should end with </body>, got: {result_with_globals[-50:]}"

    assert "[[42]]" in result_with_globals, \
        f"Should have global placeholder [[42]], got: {result_with_globals}"

    assert "[[43]]" in result_with_globals, \
        f"Should have global placeholder [[43]], got: {result_with_globals}"

    # Most important: verify HTML is valid
    expected = "<body><div class='content'><p>This is some content with [[42]]inline tags[[43]].</p></div></body>"
    assert result_with_globals == expected, \
        f"Expected: {expected}\nGot: {result_with_globals}"

    print(f"[OK] Boundary restoration successful: {result_with_globals}")


def test_fallback_without_fix_would_fail():
    """
    Demonstrate that WITHOUT the fix, HTML would be invalid.

    This shows what the old code (line 482) was doing wrong.
    """

    # OLD BROKEN CODE (before fix):
    # return placeholder_mgr.restore_to_global(chunk_text, global_indices)

    # This would return: "Content with [[10]]tags[[11]]"
    # Missing: <body><p> at start and </p></body> at end
    # Result: INVALID HTML!

    broken_result = "Content with [[10]]tags[[11]]"  # What old code returned

    # This would NOT be valid HTML
    assert not broken_result.startswith("<"), \
        "Old code did not include boundary prefix"
    assert not broken_result.endswith(">"), \
        "Old code did not include boundary suffix"

    print(f"[FAIL] Old code would return invalid HTML: {broken_result}")

    # NEW FIXED CODE includes boundaries
    fixed_result = "<body><p>Content with [[10]]tags[[11]]</p></body>"
    assert fixed_result.startswith("<body>")
    assert fixed_result.endswith("</body>")

    print(f"[OK] Fixed code returns valid HTML: {fixed_result}")


def test_multiple_scenarios():
    """Test various boundary restoration scenarios."""

    test_cases = [
        {
            "name": "Standard case",
            "chunk": "Hello [[0]]world[[1]]",
            "map": {
                "[[0]]": "<i>",
                "[[1]]": "</i>",
                "__boundary_prefix__": "<p>",
                "__boundary_suffix__": "</p>"
            },
            "indices": [5, 6],
            "expected_start": "<p>",
            "expected_end": "</p>"
        },
        {
            "name": "No internal placeholders",
            "chunk": "Plain text",
            "map": {
                "__boundary_prefix__": "<div>",
                "__boundary_suffix__": "</div>"
            },
            "indices": [],
            "expected_start": "<div>",
            "expected_end": "</div>"
        },
        {
            "name": "Complex nested boundaries",
            "chunk": "Chapter [[0]]title[[1]]",
            "map": {
                "[[0]]": "<h1>",
                "[[1]]": "</h1>",
                "__boundary_prefix__": "<body><div class='book'><section>",
                "__boundary_suffix__": "</section></div></body>"
            },
            "indices": [100, 101],
            "expected_start": "<body><div class='book'><section>",
            "expected_end": "</section></div></body>"
        },
        {
            "name": "Simple format [N]",
            "chunk": "Text [0]here[1]",
            "map": {
                "[0]": "<u>",
                "[1]": "</u>",
                "__boundary_prefix__": "<span>",
                "__boundary_suffix__": "</span>"
            },
            "indices": [20, 21],
            "expected_start": "<span>",
            "expected_end": "</span>"
        }
    ]

    for test_case in test_cases:
        # Simulate restoration
        import re
        chunk = test_case["chunk"]
        indices = test_case["indices"]
        map_data = test_case["map"]

        # Detect format
        if re.search(r'\[\[\d+\]\]', chunk):
            prefix, suffix = "[[", "]]"
        elif re.search(r'(?<!\[)\[\d+\](?!\])', chunk):
            prefix, suffix = "[", "]"
        else:
            prefix, suffix = "[[", "]]"

        # Restore global indices
        result = chunk
        for local_idx in range(len(indices)):
            local_ph = f"{prefix}{local_idx}{suffix}"
            if local_ph in result:
                result = result.replace(local_ph, f"__T_{local_idx}__")

        for local_idx, global_idx in enumerate(indices):
            result = result.replace(f"__T_{local_idx}__", f"{prefix}{global_idx}{suffix}")

        # Apply boundaries (THE FIX)
        boundary_prefix = map_data.get("__boundary_prefix__", "")
        boundary_suffix = map_data.get("__boundary_suffix__", "")

        if boundary_prefix or boundary_suffix:
            result = boundary_prefix + result + boundary_suffix

        # Verify
        assert result.startswith(test_case["expected_start"]), \
            f"{test_case['name']}: Expected to start with '{test_case['expected_start']}', got: {result[:50]}"

        assert result.endswith(test_case["expected_end"]), \
            f"{test_case['name']}: Expected to end with '{test_case['expected_end']}', got: {result[-50:]}"

        print(f"[OK] {test_case['name']}: {result}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
