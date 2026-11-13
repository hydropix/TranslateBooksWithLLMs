"""
Unit tests for TechnicalContentDetector

Tests cover:
- Individual pattern detection
- LaTeX heuristic (formula vs currency)
- Overlap resolution
- Edge cases and false positives
- Performance benchmarks
"""

import pytest
import time
from src.core.epub.technical_content_detector import (
    TechnicalContentDetector,
    PatternPriority
)


class TestPatternDetection:
    """Tests for individual pattern detection."""

    def setup_method(self):
        """Initialize detector for each test."""
        self.detector = TechnicalContentDetector()

    def test_code_block_simple(self):
        """Test detection of simple code block."""
        text = """Here is code:
```python
def hello():
    return "world"
```
Done."""
        patterns = self.detector.find_all_technical_content(text)

        assert len(patterns) == 1
        assert patterns[0].pattern_name == "code_block"
        assert "def hello():" in patterns[0].content
        assert patterns[0].priority == PatternPriority.MULTILINE_BLOCK

    def test_code_block_no_language(self):
        """Test code block without language specifier."""
        text = """Code:
```
some code
```
Done."""
        patterns = self.detector.find_all_technical_content(text)

        assert len(patterns) == 1
        assert patterns[0].pattern_name == "code_block"

    def test_latex_display_formula(self):
        """Test LaTeX display math ($$...$$)."""
        text = "The formula is $$E = mc^2$$ which shows energy."
        patterns = self.detector.find_all_technical_content(text)

        assert len(patterns) == 1
        assert patterns[0].pattern_name == "latex_display"
        assert patterns[0].content == "$$E = mc^2$$"
        assert patterns[0].priority == PatternPriority.MULTILINE_BLOCK

    def test_inline_code_backticks(self):
        """Test inline code with backticks."""
        text = "Use the `MAX1482` chip for communication."
        patterns = self.detector.find_all_technical_content(text)

        assert len(patterns) == 1
        assert patterns[0].pattern_name == "inline_code"
        assert patterns[0].content == "`MAX1482`"

    def test_latex_inline_with_subscript(self):
        """Test LaTeX inline formula with subscript."""
        text = "The voltage $V_{cm}$ is measured."
        patterns = self.detector.find_all_technical_content(text)

        assert len(patterns) == 1
        assert patterns[0].pattern_name == "latex_inline"
        assert patterns[0].content == "$V_{cm}$"

    def test_latex_inline_with_superscript(self):
        """Test LaTeX inline formula with superscript."""
        text = "Calculate $x^2 + y^2$ for the result."
        patterns = self.detector.find_all_technical_content(text)

        assert len(patterns) == 1
        assert patterns[0].pattern_name == "latex_inline"
        assert patterns[0].content == "$x^2 + y^2$"

    def test_measurement_mbps(self):
        """Test measurement detection (Mbps)."""
        text = "The speed is 10 Mbps which is fast."
        patterns = self.detector.find_all_technical_content(text)

        assert len(patterns) == 1
        assert patterns[0].pattern_name == "measurement"
        assert patterns[0].content == "10 Mbps"

    def test_measurement_units_voltage(self):
        """Test measurement with voltage units."""
        text = "Supply voltage: 5V to 12V range."
        patterns = self.detector.find_all_technical_content(text)

        assert len(patterns) == 2
        assert patterns[0].content == "5V"
        assert patterns[1].content == "12V"

    def test_measurement_units_current(self):
        """Test measurement with current units."""
        text = "Current draw: 100 mA typical."
        patterns = self.detector.find_all_technical_content(text)

        assert len(patterns) == 1
        assert patterns[0].content == "100 mA"

    def test_measurement_range(self):
        """Test measurement range detection."""
        text = "Operating range: +12 to -7 V supply."
        patterns = self.detector.find_all_technical_content(text)

        # Should detect the range
        assert len(patterns) >= 1
        range_patterns = [p for p in patterns if p.pattern_name == "measurement_range"]
        assert len(range_patterns) == 1

    def test_technical_id_standard(self):
        """Test technical identifier (standard)."""
        text = "Complies with TIA/EIA-485-A specification."
        patterns = self.detector.find_all_technical_content(text)

        assert len(patterns) == 1
        assert patterns[0].pattern_name == "technical_id"
        assert patterns[0].content == "TIA/EIA-485-A"

    def test_technical_id_chip(self):
        """Test technical identifier (chip part number)."""
        text = "The DS1487 chip provides RS-485 interface."
        patterns = self.detector.find_all_technical_content(text)

        # Should find DS1487 and RS-485
        assert len(patterns) >= 2
        ids = [p.content for p in patterns if p.pattern_name == "technical_id"]
        assert "DS1487" in ids
        assert "RS-485" in ids

    def test_multiple_patterns_in_text(self):
        """Test text with multiple different patterns."""
        text = "The $V_{cm}$ voltage is 10 Mbps using `MAX1482` chip."
        patterns = self.detector.find_all_technical_content(text)

        assert len(patterns) == 3
        pattern_names = {p.pattern_name for p in patterns}
        assert "latex_inline" in pattern_names
        assert "measurement" in pattern_names
        assert "inline_code" in pattern_names


class TestLatexHeuristic:
    """Tests for LaTeX vs currency heuristic."""

    def setup_method(self):
        """Initialize detector for each test."""
        self.detector = TechnicalContentDetector()

    def test_currency_simple(self):
        """Test currency detection (not LaTeX)."""
        text = "Price: $5 or $10 total."
        patterns = self.detector.find_all_technical_content(text)

        # Should NOT detect $5 or $10 as LaTeX
        latex_patterns = [p for p in patterns if p.pattern_name == "latex_inline"]
        assert len(latex_patterns) == 0

    def test_currency_with_cents(self):
        """Test currency with cents (not LaTeX)."""
        text = "Cost is $10.50 per unit."
        patterns = self.detector.find_all_technical_content(text)

        # Should NOT detect $10.50 as LaTeX
        latex_patterns = [p for p in patterns if p.pattern_name == "latex_inline"]
        assert len(latex_patterns) == 0

    def test_variable_name(self):
        """Test simple variable name (not LaTeX)."""
        text = "The $price variable is used."
        patterns = self.detector.find_all_technical_content(text)

        # Should NOT detect $price as LaTeX
        latex_patterns = [p for p in patterns if p.pattern_name == "latex_inline"]
        assert len(latex_patterns) == 0

    def test_latex_with_subscript_detected(self):
        """Test LaTeX formula with subscript IS detected."""
        text = "The $V_{cm}$ formula is used."
        patterns = self.detector.find_all_technical_content(text)

        # SHOULD detect $V_{cm}$ as LaTeX
        latex_patterns = [p for p in patterns if p.pattern_name == "latex_inline"]
        assert len(latex_patterns) == 1
        assert latex_patterns[0].content == "$V_{cm}$"

    def test_latex_with_superscript_detected(self):
        """Test LaTeX formula with superscript IS detected."""
        text = "Calculate $x^2$ value."
        patterns = self.detector.find_all_technical_content(text)

        # SHOULD detect $x^2$ as LaTeX
        latex_patterns = [p for p in patterns if p.pattern_name == "latex_inline"]
        assert len(latex_patterns) == 1
        assert latex_patterns[0].content == "$x^2$"

    def test_latex_with_backslash_detected(self):
        """Test LaTeX formula with backslash IS detected."""
        text = "The $\\alpha$ parameter is key."
        patterns = self.detector.find_all_technical_content(text)

        # SHOULD detect $\alpha$ as LaTeX
        latex_patterns = [p for p in patterns if p.pattern_name == "latex_inline"]
        assert len(latex_patterns) == 1

    def test_latex_with_braces_detected(self):
        """Test LaTeX formula with braces IS detected."""
        text = "The ${x + y}$ expression works."
        patterns = self.detector.find_all_technical_content(text)

        # SHOULD detect ${x + y}$ as LaTeX
        latex_patterns = [p for p in patterns if p.pattern_name == "latex_inline"]
        assert len(latex_patterns) == 1

    def test_mixed_currency_and_latex(self):
        """Test text with both currency and LaTeX."""
        text = "The $V_{cm}$ costs $50 to measure."
        patterns = self.detector.find_all_technical_content(text)

        # Should only detect $V_{cm}$ as LaTeX, not $50
        latex_patterns = [p for p in patterns if p.pattern_name == "latex_inline"]
        assert len(latex_patterns) == 1
        assert latex_patterns[0].content == "$V_{cm}$"


class TestOverlapResolution:
    """Tests for pattern overlap resolution."""

    def setup_method(self):
        """Initialize detector for each test."""
        self.detector = TechnicalContentDetector()

    def test_code_block_contains_latex(self):
        """Test code block containing LaTeX formula."""
        text = """Example:
```
The formula $x^2$ is used
```
Done."""
        patterns = self.detector.find_all_technical_content(text)

        # Code block (priority 10) should win over latex_inline (priority 5)
        assert len(patterns) == 1
        assert patterns[0].pattern_name == "code_block"
        assert "$x^2$" in patterns[0].content

    def test_code_block_contains_measurement(self):
        """Test code block containing measurement."""
        text = """Config:
```
speed = 10 Mbps
```
End."""
        patterns = self.detector.find_all_technical_content(text)

        # Code block (priority 10) should win over measurement (priority 3)
        assert len(patterns) == 1
        assert patterns[0].pattern_name == "code_block"

    def test_inline_code_contains_id(self):
        """Test inline code containing technical ID."""
        text = "Use the `TIA/EIA-485-A` standard."
        patterns = self.detector.find_all_technical_content(text)

        # Inline code (priority 5) should win over technical_id (priority 2)
        assert len(patterns) == 1
        assert patterns[0].pattern_name == "inline_code"

    def test_no_overlap_separate_patterns(self):
        """Test separate patterns don't interfere."""
        text = "The $V_{cm}$ is 10 Mbps using `chip`."
        patterns = self.detector.find_all_technical_content(text)

        # All three should be detected (no overlap)
        assert len(patterns) == 3
        assert patterns[0].pattern_name == "latex_inline"
        assert patterns[1].pattern_name == "measurement"
        assert patterns[2].pattern_name == "inline_code"

    def test_adjacent_patterns(self):
        """Test adjacent patterns are both detected."""
        text = "`code`$x^2$"  # Use valid LaTeX with superscript
        patterns = self.detector.find_all_technical_content(text)

        # Both should be detected
        assert len(patterns) == 2
        assert patterns[0].content == "`code`"
        assert patterns[1].content == "$x^2$"


class TestEdgeCases:
    """Tests for edge cases and corner scenarios."""

    def setup_method(self):
        """Initialize detector for each test."""
        self.detector = TechnicalContentDetector()

    def test_empty_text(self):
        """Test empty text returns no patterns."""
        patterns = self.detector.find_all_technical_content("")
        assert len(patterns) == 0

    def test_no_technical_content(self):
        """Test text with no technical content."""
        text = "This is a normal sentence with no technical content."
        patterns = self.detector.find_all_technical_content(text)
        assert len(patterns) == 0

    def test_unclosed_backtick(self):
        """Test unclosed backtick is not matched."""
        text = "The `code without closing"
        patterns = self.detector.find_all_technical_content(text)

        # Should not match incomplete pattern
        code_patterns = [p for p in patterns if p.pattern_name == "inline_code"]
        assert len(code_patterns) == 0

    def test_unclosed_latex(self):
        """Test unclosed LaTeX is not matched."""
        text = "The $formula without closing"
        patterns = self.detector.find_all_technical_content(text)

        # Should not match incomplete pattern
        latex_patterns = [p for p in patterns if p.pattern_name == "latex_inline"]
        assert len(latex_patterns) == 0

    def test_dollar_sign_alone(self):
        """Test single dollar sign is not matched."""
        text = "This costs $ to implement."
        patterns = self.detector.find_all_technical_content(text)

        # Should not match single $
        latex_patterns = [p for p in patterns if p.pattern_name == "latex_inline"]
        assert len(latex_patterns) == 0

    def test_newline_in_inline_latex(self):
        """Test newline breaks inline LaTeX pattern."""
        text = "The $formula\nwith newline$ is here."
        patterns = self.detector.find_all_technical_content(text)

        # Should not match (inline pattern excludes newlines)
        latex_patterns = [p for p in patterns if p.pattern_name == "latex_inline"]
        assert len(latex_patterns) == 0

    def test_multiple_code_blocks(self):
        """Test multiple code blocks in text."""
        text = """First:
```
code1
```
Second:
```
code2
```
Done."""
        patterns = self.detector.find_all_technical_content(text)

        # Should find both blocks
        code_blocks = [p for p in patterns if p.pattern_name == "code_block"]
        assert len(code_blocks) == 2

    def test_nested_latex(self):
        """Test nested LaTeX-like patterns."""
        text = "The formula $outer_{inner_{deep}}$ is complex."
        patterns = self.detector.find_all_technical_content(text)

        # Should match the outermost formula
        assert len(patterns) == 1
        assert patterns[0].content == "$outer_{inner_{deep}}$"

    def test_measurement_without_space(self):
        """Test measurement without space between number and unit."""
        text = "Speed is 10Mbps today."
        patterns = self.detector.find_all_technical_content(text)

        # Pattern allows optional space
        measurements = [p for p in patterns if p.pattern_name == "measurement"]
        assert len(measurements) >= 1

    def test_decimal_measurement(self):
        """Test measurement with decimal number."""
        text = "Frequency is 3.5 MHz exactly."
        patterns = self.detector.find_all_technical_content(text)

        measurements = [p for p in patterns if p.pattern_name == "measurement"]
        assert len(measurements) == 1
        assert "3.5" in measurements[0].content


class TestStatistics:
    """Tests for statistics generation."""

    def setup_method(self):
        """Initialize detector for each test."""
        self.detector = TechnicalContentDetector()

    def test_statistics_empty(self):
        """Test statistics for empty pattern list."""
        stats = self.detector.get_statistics([])
        assert stats['total'] == 0

    def test_statistics_single_type(self):
        """Test statistics with single pattern type."""
        text = "Use `code1` and `code2` here."
        patterns = self.detector.find_all_technical_content(text)
        stats = self.detector.get_statistics(patterns)

        assert stats['total'] == 2
        assert stats['inline_code'] == 2

    def test_statistics_multiple_types(self):
        """Test statistics with multiple pattern types."""
        text = "The $V_{cm}$ is 10 Mbps using `chip`."
        patterns = self.detector.find_all_technical_content(text)
        stats = self.detector.get_statistics(patterns)

        assert stats['total'] == 3
        assert stats['latex_inline'] == 1
        assert stats['measurement'] == 1
        assert stats['inline_code'] == 1


class TestPerformance:
    """Performance benchmarks for the detector."""

    def setup_method(self):
        """Initialize detector for each test."""
        self.detector = TechnicalContentDetector()

    def test_performance_small_text(self):
        """Benchmark detection on small text (< 1KB)."""
        text = "The $V_{cm}$ is 10 Mbps using `chip` from TIA/EIA-485-A." * 10

        start = time.perf_counter()
        for _ in range(100):
            self.detector.find_all_technical_content(text)
        elapsed = time.perf_counter() - start

        # Should process 100 iterations in < 1 second
        assert elapsed < 1.0, f"Performance issue: {elapsed:.3f}s for 100 iterations"
        print(f"\nSmall text (100 iterations): {elapsed*1000:.2f}ms total, {elapsed*10:.2f}ms per iteration")

    def test_performance_medium_text(self):
        """Benchmark detection on medium text (~ 10KB)."""
        text = """
        Technical document with mixed content.
        The formula $V_{cm}$ is important.
        Speed: 10 Mbps using `MAX1482` chip.
        Standard: TIA/EIA-485-A compliant.

        Code example:
        ```python
        def calculate():
            voltage = 5V
            return voltage * 2
        ```

        More text with $x^2 + y^2$ formulas.
        """ * 100  # ~10KB

        start = time.perf_counter()
        patterns = self.detector.find_all_technical_content(text)
        elapsed = time.perf_counter() - start

        # Should process 10KB in < 100ms
        assert elapsed < 0.1, f"Performance issue: {elapsed*1000:.1f}ms for 10KB"
        print(f"\nMedium text (~10KB): {elapsed*1000:.2f}ms, found {len(patterns)} patterns")

    def test_performance_large_text(self):
        """Benchmark detection on large text (~ 100KB)."""
        text = """
        Chapter with extensive technical content.
        The voltage $V_{cm}$ varies from +12 to -7 V range.
        Data rate: 10 Mbps maximum throughput.
        Uses `DS1487` transceiver chip.
        Complies with TIA/EIA-485-A standard.

        Implementation:
        ```python
        class Driver:
            def __init__(self):
                self.voltage = 5V
                self.current = 100 mA

            def calculate(self):
                return self.voltage * self.current
        ```

        Additional formulas: $E = mc^2$ and $F = ma$.
        Frequency: 3.5 MHz to 10 MHz operating range.
        """ * 500  # ~100KB

        start = time.perf_counter()
        patterns = self.detector.find_all_technical_content(text)
        elapsed = time.perf_counter() - start

        # Should process 100KB in < 5s (more realistic given regex complexity)
        assert elapsed < 5.0, f"Performance issue: {elapsed*1000:.1f}ms for 100KB"
        print(f"\nLarge text (~100KB): {elapsed*1000:.2f}ms, found {len(patterns)} patterns")

    def test_performance_worst_case(self):
        """Benchmark worst case: dense technical content."""
        # Text with technical pattern in nearly every word
        text = " ".join([f"$V_{i}$" for i in range(1000)])

        start = time.perf_counter()
        patterns = self.detector.find_all_technical_content(text)
        elapsed = time.perf_counter() - start

        assert len(patterns) == 1000
        # Should handle 1000 patterns in < 200ms
        assert elapsed < 0.2, f"Performance issue: {elapsed*1000:.1f}ms for 1000 patterns"
        print(f"\nWorst case (1000 dense patterns): {elapsed*1000:.2f}ms")


class TestRealWorldExamples:
    """Tests with real-world technical text examples."""

    def setup_method(self):
        """Initialize detector for each test."""
        self.detector = TechnicalContentDetector()

    def test_electronics_datasheet(self):
        """Test electronics datasheet excerpt."""
        text = """The MAX1482 is a low-power transceiver for TIA/EIA-485-A communication.

Operating Conditions:
- Supply voltage: +5V ±10%
- Current consumption: 300 µA typical
- Data rate: up to 10 Mbps
- Common-mode voltage range: +12 to -7 V

The differential output voltage $V_{OD}$ is calculated as:
$$V_{OD} = V_A - V_B$$

Usage example:
```python
driver = MAX1482(voltage=5V, rate=10Mbps)
driver.transmit(data)
```
"""

        patterns = self.detector.find_all_technical_content(text)
        stats = self.detector.get_statistics(patterns)

        # Should detect multiple types
        assert stats['total'] > 5
        assert 'technical_id' in stats  # MAX1482, TIA/EIA-485-A
        assert 'measurement' in stats   # 5V, 300 µA, 10 Mbps
        assert 'latex_inline' in stats  # $V_{OD}$
        assert 'latex_display' in stats # $$V_{OD} = V_A - V_B$$
        assert 'code_block' in stats    # Python code

        print(f"\nElectronics datasheet: {stats}")

    def test_math_textbook(self):
        """Test math textbook excerpt."""
        text = """The Pythagorean theorem states that $a^2 + b^2 = c^2$ for right triangles.

For complex calculations, we use:
$$\\int_{0}^{\\infty} e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}$$

The implementation in Python:
```python
import math

def pythagorean(a, b):
    return math.sqrt(a**2 + b**2)
```
"""

        patterns = self.detector.find_all_technical_content(text)
        stats = self.detector.get_statistics(patterns)

        assert 'latex_inline' in stats
        assert 'latex_display' in stats
        assert 'code_block' in stats

        print(f"\nMath textbook: {stats}")

    def test_programming_tutorial(self):
        """Test programming tutorial excerpt."""
        text = """To optimize performance, use the `cache` decorator:

```python
from functools import cache

@cache
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

This reduces time complexity from `O(2^n)` to `O(n)`.
"""

        patterns = self.detector.find_all_technical_content(text)
        stats = self.detector.get_statistics(patterns)

        assert 'inline_code' in stats  # `cache`, `O(2^n)`, `O(n)`
        assert 'code_block' in stats   # Python code

        print(f"\nProgramming tutorial: {stats}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
