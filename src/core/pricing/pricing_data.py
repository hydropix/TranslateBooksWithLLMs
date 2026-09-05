"""
Default pricing data for providers without a public pricing API.

Prices are in USD per 1 million tokens.
Users can override these values via the UI (sent in /api/cost/estimate payload).

Last updated: 2026-05-09
Sources: official provider documentation pages.
"""

LAST_UPDATED = "2026-05-09"

DEFAULT_PRICING = {
    "gemini": {
        "gemini-2.5-pro":        {"input": 1.25, "output": 10.00, "note": "Standard tier (<=200K context)"},
        "gemini-2.5-pro-large":  {"input": 2.50, "output": 15.00, "note": "Extended tier (>200K context)"},
        "gemini-2.5-flash":      {"input": 0.30, "output": 2.50},
        "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
        "gemini-2.0-flash":      {"input": 0.10, "output": 0.40, "note": "Deprecated 2026-06-01"},
        "gemini-1.5-pro":        {"input": 1.25, "output": 5.00, "note": "Legacy"},
        "gemini-1.5-flash":      {"input": 0.075, "output": 0.30, "note": "Legacy"},
    },
    "openai": {
        "gpt-4o":           {"input": 2.50,  "output": 10.00},
        "gpt-4o-mini":      {"input": 0.15,  "output": 0.60},
        "gpt-4.1":          {"input": 2.00,  "output": 8.00},
        "gpt-4.1-mini":     {"input": 0.40,  "output": 1.60},
        "gpt-4.1-nano":     {"input": 0.10,  "output": 0.40},
        "gpt-4-turbo":      {"input": 10.00, "output": 30.00, "note": "Legacy"},
        "gpt-4":            {"input": 30.00, "output": 60.00, "note": "Legacy"},
        "gpt-3.5-turbo":    {"input": 0.50,  "output": 1.50,  "note": "Legacy"},
        "o1":               {"input": 15.00, "output": 60.00, "note": "Reasoning model"},
        "o1-mini":          {"input": 3.00,  "output": 12.00, "note": "Reasoning model"},
        "o3-mini":          {"input": 1.10,  "output": 4.40,  "note": "Reasoning model"},
    },
    "deepseek": {
        "deepseek-chat":     {"input": 0.28, "output": 0.42, "note": "V3.2, cache miss pricing"},
        "deepseek-reasoner": {"input": 0.28, "output": 0.42, "note": "V3.2, cache miss pricing"},
        "deepseek-v4-flash": {"input": 0.14, "output": 0.28},
        "deepseek-v4-pro":   {"input": 1.74, "output": 3.48, "note": "75% discount until 2026-05-31"},
    },
    "mistral": {
        "mistral-large-latest":  {"input": 2.00, "output": 6.00},
        "mistral-large-2411":    {"input": 2.00, "output": 6.00},
        "mistral-medium-latest": {"input": 0.40, "output": 2.00},
        "mistral-medium-3":      {"input": 0.40, "output": 2.00},
        "mistral-small-latest":  {"input": 0.20, "output": 0.60},
        "mistral-small-3":       {"input": 0.10, "output": 0.30},
        "ministral-8b-latest":   {"input": 0.10, "output": 0.10},
        "ministral-3b-latest":   {"input": 0.04, "output": 0.04},
        "codestral-latest":      {"input": 0.30, "output": 0.90},
        "pixtral-large-latest":  {"input": 2.00, "output": 6.00},
    },
    "nim": {
        # NVIDIA NIM is mostly free-credits via build.nvidia.com.
        # These values are pay-as-you-go reference prices.
        "meta/llama-3.1-8b-instruct":   {"input": 0.04, "output": 0.04},
        "meta/llama-3.1-70b-instruct":  {"input": 0.40, "output": 0.40},
        "meta/llama-3.1-405b-instruct": {"input": 1.20, "output": 1.20},
        "deepseek-ai/deepseek-v3":      {"input": 0.27, "output": 1.10},
        "deepseek-ai/deepseek-r1":      {"input": 0.55, "output": 2.19},
    },
}


def get_default_pricing(provider: str, model: str) -> dict | None:
    """
    Return {input, output} prices per 1M tokens for the given provider/model.

    Returns None if no default pricing is known.
    Lookup is case-insensitive and tolerates minor variants in model names
    (dated snapshots, preview suffixes); the most specific known name wins.
    """
    provider_data = DEFAULT_PRICING.get(provider.lower())
    if not provider_data:
        return None

    if model in provider_data:
        return _strip_note(provider_data[model])

    model_lower = model.lower()
    for known_model, pricing in provider_data.items():
        if known_model.lower() == model_lower:
            return _strip_note(pricing)

    # Last resort: substring match. Prefer the most specific key, otherwise
    # dict insertion order decides and "gpt-4.1-nano-2025-04-14" is priced as
    # "gpt-4.1" (20x too high) simply because it is listed first.
    contained = [k for k in provider_data if k.lower() in model_lower]
    if contained:
        return _strip_note(provider_data[max(contained, key=len)])

    containing = [k for k in provider_data if model_lower in k.lower()]
    if containing:
        return _strip_note(provider_data[min(containing, key=len)])

    return None


def _strip_note(entry: dict) -> dict:
    return {"input": entry["input"], "output": entry["output"]}
