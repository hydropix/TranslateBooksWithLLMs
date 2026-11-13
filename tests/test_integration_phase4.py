"""
Integration tests for Phase 4 validation.

Ensures all components work together correctly:
- TechnicalContentDetector
- TagPreserver with technical protection
- EPUB translation pipeline
- PlaceholderValidator
- Backward compatibility
"""

import pytest
from src.core.epub.technical_content_detector import TechnicalContentDetector
from src.core.epub.tag_preservation import TagPreserver
from src.core.epub.placeholder_validator import PlaceholderValidator


class TestPhase4Integration:
    """End-to-end integration tests for Phase 4."""

    def test_detector_preserver_integration(self):
        """Detector and TagPreserver work together correctly."""
        detector = TechnicalContentDetector()
        preserver = TagPreserver(protect_technical=True)

        # Use detector directly
        text = "The $V_{cm}$ is 10 Mbps using `chip`."
        patterns = detector.find_all_technical_content(text)

        # Should find 3 patterns
        assert len(patterns) >= 3

        # Use preserver (which uses detector internally)
        html = f"<p>{text}</p>"
        result, tag_map = preserver.preserve_tags_and_technical_content(html)

        # Should have protected technical content
        assert "$V_{cm}$" in tag_map.values()
        assert "10 Mbps" in tag_map.values()
        assert "`chip`" in tag_map.values()

    def test_preserver_validator_integration(self):
        """TagPreserver output validates correctly with PlaceholderValidator."""
        preserver = TagPreserver(protect_technical=True)

        html = "<p>The $V_{cm}$ voltage is 10 Mbps.</p>"
        result, tag_map = preserver.preserve_tags_and_technical_content(html)

        # Simulate translation (preserve placeholders)
        translated = result.replace("The", "La").replace("voltage is", "tension est")

        # Validate using static method
        is_valid = PlaceholderValidator.validate_basic(translated, tag_map)
        assert is_valid, "Validation failed"

    def test_preserver_restore_integration(self):
        """TagPreserver preserve and restore work correctly."""
        preserver = TagPreserver(protect_technical=True)

        original = "<p>The voltage $V_{cm}$ is 10 Mbps using `chip`.</p>"

        # Preserve
        preserved, tag_map = preserver.preserve_tags_and_technical_content(original)

        # Simulate translation
        translated = preserved.replace("voltage", "tension").replace("using", "utilisant")

        # Restore
        restored = preserver.restore_tags(translated, tag_map)

        # Technical content should be unchanged
        assert "$V_{cm}$" in restored
        assert "10 Mbps" in restored
        assert "`chip`" in restored

        # Translated text should be present
        assert "tension" in restored
        assert "utilisant" in restored

    def test_backward_compatibility_no_protection(self):
        """System works correctly when protection is disabled."""
        preserver = TagPreserver(protect_technical=False)

        html = "<p>The $V_{cm}$ voltage</p>"
        result, tag_map = preserver.preserve_tags(html)

        # Should only protect tags, not technical content
        assert "$V_{cm}$" in result  # Formula NOT protected
        assert len(tag_map) == 2  # Only <p> and </p>

    def test_backward_compatibility_with_protection(self):
        """Old code using preserve_tags() still works when protect_technical=True."""
        preserver = TagPreserver(protect_technical=True)

        # Even with protect_technical=True, old method works as before
        html = "<p>Simple text</p>"
        result, tag_map = preserver.preserve_tags(html)

        # Should work without technical protection (old API)
        assert result == "[id0]Simple text[id1]"
        assert len(tag_map) == 2

    def test_feature_toggle_behavior(self):
        """Protection only activates when explicitly requested."""
        html = "<p>The $V_{cm}$ voltage</p>"

        # Without protection
        preserver_off = TagPreserver(protect_technical=False)
        result_off, map_off = preserver_off.preserve_tags(html)

        # With protection
        preserver_on = TagPreserver(protect_technical=True)
        result_on, map_on = preserver_on.preserve_tags_and_technical_content(html)

        # Different results
        assert len(map_off) < len(map_on), "Protection should add more placeholders"
        assert "$V_{cm}$" in result_off, "Without protection, formula in result"
        assert "$V_{cm}$" not in result_on or "$V_{cm}$" in map_on.values(), "With protection, formula protected"

    def test_complex_workflow(self):
        """Complex workflow with multiple steps."""
        preserver = TagPreserver(protect_technical=True)

        # Step 1: Complex HTML with mixed content
        html = """<div class="chapter">
            <h1>Chapter 1</h1>
            <p>The formula $E = mc^2$ is famous.</p>
            <p>Speed: 10 Mbps using `MAX1482` chip.</p>
            <pre><code>def calc():
    return 42</code></pre>
        </div>"""

        # Step 2: Preserve
        preserved, tag_map = preserver.preserve_tags_and_technical_content(html)

        # Step 3: Validate we have technical content protected
        technical_values = [v for v in tag_map.values() if any(
            indicator in v for indicator in ['$', '`', 'Mbps', 'def ', 'return']
        )]
        assert len(technical_values) > 0, "Should have protected technical content"

        # Step 4: Simulate translation
        translated = preserved.replace("Chapter", "Chapitre").replace("famous", "célèbre")

        # Step 5: Validate using static method
        is_valid = PlaceholderValidator.validate_basic(translated, tag_map)
        assert is_valid, "Validation failed"

        # Step 6: Restore
        restored = preserver.restore_tags(translated, tag_map)

        # Step 7: Verify
        assert "$E = mc^2$" in restored or "E = mc^2" in restored
        assert "10 Mbps" in restored
        assert "MAX1482" in restored
        assert "def calc():" in restored
        assert "Chapitre" in restored
        assert "célèbre" in restored


class TestRealWorldScenarios:
    """Real-world usage scenarios."""

    def test_math_textbook_workflow(self):
        """Translate math textbook excerpt."""
        preserver = TagPreserver(protect_technical=True)

        textbook = """<p>The Pythagorean theorem states that $a^2 + b^2 = c^2$ for right triangles.</p>
<p>The integral formula:</p>
<p>$$\\int_{0}^{\\infty} e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}$$</p>"""

        # Preserve
        preserved, tag_map = preserver.preserve_tags_and_technical_content(textbook)

        # Check formulas protected
        assert "$a^2 + b^2 = c^2$" in tag_map.values()
        # Display formula should be protected
        display_formulas = [v for v in tag_map.values() if v.startswith("$$")]
        assert len(display_formulas) >= 1

        # Simulate translation
        translated = preserved.replace("theorem states that", "théorème indique que")
        translated = translated.replace("for right triangles", "pour les triangles rectangles")

        # Restore
        restored = preserver.restore_tags(translated, tag_map)

        # Verify
        assert "$a^2 + b^2 = c^2$" in restored
        assert "théorème" in restored

    def test_programming_tutorial_workflow(self):
        """Translate programming tutorial."""
        preserver = TagPreserver(protect_technical=True)

        tutorial = """<p>Use the `@cache` decorator for optimization:</p>
<pre><code>from functools import cache

@cache
def fibonacci(n):
    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)
</code></pre>
<p>This reduces complexity from `O(2^n)` to `O(n)`.</p>"""

        # Preserve
        preserved, tag_map = preserver.preserve_tags_and_technical_content(tutorial)

        # Check code protected
        assert "`@cache`" in tag_map.values()
        assert "`O(2^n)`" in tag_map.values()
        assert "`O(n)`" in tag_map.values()

        # Code block should be protected
        code_blocks = [v for v in tag_map.values() if "fibonacci" in v]
        assert len(code_blocks) >= 1

        # Restore
        restored = preserver.restore_tags(preserved, tag_map)

        # Verify code intact
        assert "@cache" in restored
        assert "fibonacci" in restored
        assert "O(2^n)" in restored

    def test_electronics_datasheet_workflow(self):
        """Translate electronics datasheet."""
        preserver = TagPreserver(protect_technical=True)

        datasheet = """<p>The MAX1482 operates at 10 Mbps with supply voltage +5V ±10%.</p>
<p>Common-mode range: +12 to -7 V per TIA/EIA-485-A standard.</p>
<p>Differential voltage: $V_{OD} = V_A - V_B$</p>"""

        # Preserve
        preserved, tag_map = preserver.preserve_tags_and_technical_content(datasheet)

        # Check technical content protected
        assert "MAX1482" in str(tag_map.values()) or "MAX1482" in preserved
        assert "10 Mbps" in tag_map.values()
        assert "+5V" in str(tag_map.values()) or "5V" in str(tag_map.values())
        assert "TIA/EIA-485-A" in tag_map.values()
        assert "$V_{OD} = V_A - V_B$" in tag_map.values()

        # Restore
        restored = preserver.restore_tags(preserved, tag_map)

        # Verify all technical specs intact
        assert "MAX1482" in restored
        assert "10 Mbps" in restored
        assert "TIA/EIA-485-A" in restored
        assert "$V_{OD}" in restored


class TestErrorRecovery:
    """Test error handling and recovery."""

    def test_malformed_html_recovery(self):
        """System handles malformed HTML gracefully."""
        preserver = TagPreserver(protect_technical=True)

        # Unclosed tag
        html = "<p>Text with $x^2$ formula"

        try:
            result, tag_map = preserver.preserve_tags_and_technical_content(html)
            # Should complete without crash
            assert result is not None
            # Formula should still be protected if possible
            assert "$x^2$" in tag_map.values() or "$x^2$" in result
        except Exception as e:
            pytest.fail(f"Should handle malformed HTML, got: {e}")

    def test_unclosed_technical_content(self):
        """Unclosed technical delimiters handled correctly."""
        preserver = TagPreserver(protect_technical=True)

        # Unclosed LaTeX
        html = "<p>Price is $50 for item</p>"

        result, tag_map = preserver.preserve_tags_and_technical_content(html)

        # Unclosed $ should not be protected (treated as currency)
        # Just verify no crash
        assert result is not None

    def test_empty_content(self):
        """Empty content handled correctly."""
        preserver = TagPreserver(protect_technical=True)

        result, tag_map = preserver.preserve_tags_and_technical_content("")
        assert result == ""
        assert len(tag_map) == 0

    def test_only_whitespace(self):
        """Only whitespace handled correctly."""
        preserver = TagPreserver(protect_technical=True)

        result, tag_map = preserver.preserve_tags_and_technical_content("   \n\t  ")
        # Should handle gracefully
        assert result is not None


class TestPerformanceIntegration:
    """Integration tests for performance requirements."""

    def test_large_document_processing(self):
        """Large document with mixed content processes efficiently."""
        import time

        preserver = TagPreserver(protect_technical=True)

        # Create large document
        paragraphs = []
        for i in range(100):
            paragraphs.append(f"<p>Section {i}: The formula $V_{{{i}}}$ is measured at {i} MHz using `CHIP{i}`.</p>")

        html = "\n".join(paragraphs)

        start = time.perf_counter()
        result, tag_map = preserver.preserve_tags_and_technical_content(html)
        elapsed = time.perf_counter() - start

        # Should complete in reasonable time (< 500ms for 100 paragraphs)
        assert elapsed < 0.5, f"Too slow: {elapsed*1000:.2f}ms"

        # Should have protected content
        assert len(tag_map) > 100  # At least tags + some technical content

    def test_repeated_operations_stable(self):
        """Repeated operations give consistent results."""
        preserver = TagPreserver(protect_technical=True)

        html = "<p>The $V_{cm}$ is 10 Mbps.</p>"

        # Run multiple times
        results = []
        for _ in range(10):
            result, tag_map = preserver.preserve_tags_and_technical_content(html)
            results.append((result, tag_map))

        # All results should be identical
        first_result = results[0]
        for result, tag_map in results[1:]:
            assert result == first_result[0]
            assert tag_map == first_result[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
