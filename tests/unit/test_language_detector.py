"""Regression tests for source-language auto-detection (src/utils/language_detector.py)."""
from src.utils.language_detector import LanguageDetector


def test_detection_is_deterministic():
    # Short, mixed input: unseeded langdetect returns different probabilities
    # from one call to the next.
    sample = "Bonjour hello ciao hola guten Tag the le la der die das el " * 2
    results = {LanguageDetector.detect_language_from_text(sample, 0.0) for _ in range(15)}

    assert len(results) == 1
