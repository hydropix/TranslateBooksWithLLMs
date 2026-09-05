"""
Unit tests for get_default_pricing() model matching.

Regression tests for item 1 of issue #231: the last-resort substring lookup
returned the first insertion-order entry whose name overlapped the queried
model, so a dated snapshot such as "gpt-4.1-nano-2025-04-14" was priced as
"gpt-4.1" simply because the broader key is listed first.

https://github.com/hydropix/TranslateBooksWithLLMs/issues/231
"""
import pytest
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.pricing.pricing_data import DEFAULT_PRICING, get_default_pricing


class TestGetDefaultPricingSpecificity:
    """The most specific known model name must win the substring fallback."""

    @pytest.mark.parametrize("provider,model,expected", [
        # Dated OpenAI snapshots: the "-mini"/"-nano" variants used to resolve
        # to their much more expensive parent.
        ("openai", "gpt-4o-mini-2024-07-18", {"input": 0.15, "output": 0.60}),
        ("openai", "gpt-4.1-mini-2025-04-14", {"input": 0.40, "output": 1.60}),
        ("openai", "gpt-4.1-nano-2025-04-14", {"input": 0.10, "output": 0.40}),
        ("openai", "o1-mini-2024-09-12", {"input": 3.00, "output": 12.00}),
        # Gemini preview ids reach the model dropdown, so this is a real input.
        ("gemini", "gemini-2.5-flash-lite-preview-06-17",
         {"input": 0.10, "output": 0.40}),
    ])
    def test_dated_variant_uses_its_own_price(self, provider, model, expected):
        """A dated/preview id must match the longest known name it contains."""
        assert get_default_pricing(provider, model) == expected

    @pytest.mark.parametrize("provider,model,expected", [
        ("openai", "gpt-4o-2024-11-20", {"input": 2.50, "output": 10.00}),
        ("openai", "gpt-4-turbo-2024-04-09", {"input": 10.00, "output": 30.00}),
        ("openai", "gpt-4o-mini", {"input": 0.15, "output": 0.60}),
        ("openai", "GPT-4O-MINI", {"input": 0.15, "output": 0.60}),
        ("mistral", "mistral-large-2411", {"input": 2.00, "output": 6.00}),
        ("gemini", "gemini-2.5-pro", {"input": 1.25, "output": 10.00}),
    ])
    def test_existing_matches_are_unchanged(self, provider, model, expected):
        """Exact, case-insensitive and parent-model lookups keep their price."""
        assert get_default_pricing(provider, model) == expected

    @pytest.mark.parametrize("provider,model", [
        ("openai", "totally-unknown-model"),
        ("unknown-provider", "gpt-4o"),
    ])
    def test_unknown_stays_unknown(self, provider, model):
        """No pricing is invented for unknown providers or models."""
        assert get_default_pricing(provider, model) is None

    def test_every_table_key_resolves_to_its_own_entry(self):
        """No key in DEFAULT_PRICING may be shadowed by another one."""
        mismatches = [
            (provider, model)
            for provider, models in DEFAULT_PRICING.items()
            for model, entry in models.items()
            if get_default_pricing(provider, model) != {
                "input": entry["input"],
                "output": entry["output"],
            }
        ]
        assert mismatches == [], f"Shadowed pricing entries: {mismatches}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
