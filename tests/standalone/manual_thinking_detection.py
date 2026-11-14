#!/usr/bin/env python3
"""
Test script to evaluate thinking model detection on Ollama.

This script tests all available models on your Ollama server to determine:
1. Which models produce <think> blocks or thinking field when think=false
2. Which models respect the think=false parameter
3. Which models CANNOT be prevented from thinking (need warning)

Usage:
    python test_thinking_detection.py

Configure OLLAMA_ENDPOINT below to match your server.
"""

import httpx
import asyncio
import json
import re
import sys
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# =============================================================================
# CONFIGURATION
# =============================================================================
OLLAMA_ENDPOINT = "http://ai_server.mds.com:11434"
TEST_TIMEOUT = 120  # seconds per test
VERBOSE = True  # Show detailed output

# Test prompts - simple questions that shouldn't require thinking
TEST_PROMPTS = [
    "What is 2+2? Reply with just the number.",
    "Say hello in one word.",
    "What color is the sky? One word answer.",
]


class ThinkingBehavior(Enum):
    """Classification of model thinking behavior"""
    STANDARD = "standard"           # No thinking, respects think=false
    CONTROLLABLE = "controllable"   # Thinks by default, but think=false works
    UNCONTROLLABLE = "uncontrollable"  # CANNOT stop thinking (needs warning)
    NO_THINK_SUPPORT = "no_think_support"  # Model doesn't support think param
    ERROR = "error"                 # Failed to test


@dataclass
class ModelTestResult:
    """Result of testing a single model"""
    model_name: str
    behavior: ThinkingBehavior

    # Test with think=true
    thinks_when_enabled: bool = False
    thinking_field_when_enabled: bool = False
    think_tags_when_enabled: bool = False

    # Test with think=false
    thinks_when_disabled: bool = False
    thinking_field_when_disabled: bool = False
    think_tags_when_disabled: bool = False

    # Test without think param
    thinks_without_param: bool = False
    thinking_field_without_param: bool = False
    think_tags_without_param: bool = False

    # Raw responses for debugging
    response_think_true: str = ""
    response_think_false: str = ""
    response_no_param: str = ""
    thinking_content: str = ""

    error_message: str = ""
    supports_think_param: bool = True

    def __str__(self):
        if self.behavior == ThinkingBehavior.ERROR:
            return f"{self.model_name}: ERROR - {self.error_message}"

        status_emoji = {
            ThinkingBehavior.STANDARD: "[OK]",
            ThinkingBehavior.CONTROLLABLE: "[CTRL]",
            ThinkingBehavior.UNCONTROLLABLE: "[WARN]",
            ThinkingBehavior.NO_THINK_SUPPORT: "[NO-THINK]",
        }

        emoji = status_emoji.get(self.behavior, "[?]")

        details = []
        if self.thinks_when_enabled:
            parts = []
            if self.thinking_field_when_enabled:
                parts.append("field")
            if self.think_tags_when_enabled:
                parts.append("tags")
            details.append(f"think=true: {'+'.join(parts) if parts else 'yes'}")
        if self.thinks_when_disabled:
            details.append("think=false: STILL THINKS!")
        if self.thinks_without_param:
            details.append("no param: thinks")

        detail_str = f" ({', '.join(details)})" if details else ""
        return f"{emoji} {self.model_name}: {self.behavior.value}{detail_str}"


async def get_available_models(endpoint: str) -> List[str]:
    """Fetch list of available models from Ollama"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{endpoint}/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            return sorted(models)
        except Exception as e:
            print(f"Error fetching models: {e}")
            return []


async def test_model_thinking(
    endpoint: str,
    model: str,
    prompt: str,
    think_param: Optional[bool] = None
) -> tuple[str, str, bool, bool, bool]:
    """
    Test a model with specific think parameter.

    Args:
        endpoint: Ollama API endpoint
        model: Model name
        prompt: Test prompt
        think_param: True/False/None (None = don't include param)

    Returns:
        (content, thinking_field, has_think_tags, has_thinking_field, success)
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_ctx": 2048},
    }

    # Only add think param if specified
    if think_param is not None:
        payload["think"] = think_param

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{endpoint}/api/chat",
            json=payload,
            timeout=TEST_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()

        message = data.get("message", {})
        content = message.get("content", "")
        thinking_field = message.get("thinking", "")

        # Check for <think> tags in content
        has_think_tags = bool(re.search(r'<think>|</think>', content, re.IGNORECASE))
        has_thinking_field = bool(thinking_field)

        return content, thinking_field, has_think_tags, has_thinking_field, True


async def test_single_model(endpoint: str, model: str) -> ModelTestResult:
    """Run complete thinking detection test on a single model"""
    result = ModelTestResult(model_name=model, behavior=ThinkingBehavior.ERROR)

    prompt = TEST_PROMPTS[0]

    # Test 1: Without think parameter (baseline)
    try:
        if VERBOSE:
            print(f"  [1/3] Testing without think param...")

        content_none, thinking_none, tags_none, field_none, _ = await test_model_thinking(
            endpoint, model, prompt, think_param=None
        )

        result.response_no_param = content_none
        result.thinking_field_without_param = field_none
        result.think_tags_without_param = tags_none
        result.thinks_without_param = field_none or tags_none

    except Exception as e:
        result.error_message = f"Baseline test failed: {str(e)[:50]}"
        return result

    # Test 2: With think=true
    try:
        if VERBOSE:
            print(f"  [2/3] Testing with think=true...")

        content_true, thinking_true, tags_true, field_true, _ = await test_model_thinking(
            endpoint, model, prompt, think_param=True
        )

        result.response_think_true = content_true
        result.thinking_field_when_enabled = field_true
        result.think_tags_when_enabled = tags_true
        result.thinks_when_enabled = field_true or tags_true

        if thinking_true:
            result.thinking_content = thinking_true[:500]

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 400:
            # Model doesn't support think param
            result.supports_think_param = False
            if VERBOSE:
                print(f"  [!] Model doesn't support think parameter")
        else:
            result.error_message = f"think=true test: HTTP {e.response.status_code}"
            return result
    except Exception as e:
        result.error_message = f"think=true test: {str(e)[:50]}"
        return result

    # Test 3: With think=false (only if model supports think param)
    if result.supports_think_param:
        try:
            if VERBOSE:
                print(f"  [3/3] Testing with think=false...")

            content_false, thinking_false, tags_false, field_false, _ = await test_model_thinking(
                endpoint, model, prompt, think_param=False
            )

            result.response_think_false = content_false
            result.thinking_field_when_disabled = field_false
            result.think_tags_when_disabled = tags_false
            result.thinks_when_disabled = field_false or tags_false

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                result.supports_think_param = False
            else:
                result.error_message = f"think=false test: HTTP {e.response.status_code}"
                return result
        except Exception as e:
            result.error_message = f"think=false test: {str(e)[:50]}"
            return result

    # Classify behavior
    if not result.supports_think_param:
        # Model doesn't support think parameter
        if result.thinks_without_param:
            # Has <think> tags in baseline - it's a thinking model we can't control
            result.behavior = ThinkingBehavior.UNCONTROLLABLE
        else:
            # Standard model that doesn't need think param
            result.behavior = ThinkingBehavior.NO_THINK_SUPPORT
    elif not result.thinks_when_enabled and not result.thinks_when_disabled and not result.thinks_without_param:
        # Model never thinks - standard model
        result.behavior = ThinkingBehavior.STANDARD
    elif result.thinks_when_disabled:
        # Model thinks even with think=false - UNCONTROLLABLE!
        result.behavior = ThinkingBehavior.UNCONTROLLABLE
    elif result.thinks_when_enabled and not result.thinks_when_disabled:
        # Model thinks when asked, but respects think=false
        result.behavior = ThinkingBehavior.CONTROLLABLE
    elif result.thinks_without_param and not result.thinks_when_disabled:
        # Model thinks by default but think=false works
        result.behavior = ThinkingBehavior.CONTROLLABLE
    else:
        # Standard model
        result.behavior = ThinkingBehavior.STANDARD

    return result


async def run_all_tests(endpoint: str, models: Optional[List[str]] = None):
    """Run thinking detection tests on all models"""

    print("=" * 70)
    print("THINKING MODEL DETECTION TEST")
    print(f"Endpoint: {endpoint}")
    print("=" * 70)
    print()

    # Get models if not provided
    if models is None:
        print("Fetching available models...")
        models = await get_available_models(endpoint)

    if not models:
        print("No models found!")
        return

    print(f"Found {len(models)} models to test:")
    for m in models:
        print(f"  - {m}")
    print()

    # Run tests
    results: List[ModelTestResult] = []

    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] Testing: {model}")
        result = await test_single_model(endpoint, model)
        results.append(result)
        print(f"  Result: {result}")

    # Summary
    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    standard = [r for r in results if r.behavior == ThinkingBehavior.STANDARD]
    controllable = [r for r in results if r.behavior == ThinkingBehavior.CONTROLLABLE]
    uncontrollable = [r for r in results if r.behavior == ThinkingBehavior.UNCONTROLLABLE]
    no_think = [r for r in results if r.behavior == ThinkingBehavior.NO_THINK_SUPPORT]
    errors = [r for r in results if r.behavior == ThinkingBehavior.ERROR]

    print(f"\n[OK] STANDARD MODELS ({len(standard)}) - No thinking, supports think param:")
    for r in standard:
        print(f"   {r.model_name}")

    print(f"\n[NO-THINK] MODELS WITHOUT THINK SUPPORT ({len(no_think)}) - No think param, but no thinking:")
    for r in no_think:
        print(f"   {r.model_name}")

    print(f"\n[CTRL] CONTROLLABLE THINKING MODELS ({len(controllable)}) - think=false works:")
    for r in controllable:
        print(f"   {r.model_name}")
        if r.thinks_when_enabled:
            method = []
            if r.thinking_field_when_enabled:
                method.append("thinking field")
            if r.think_tags_when_enabled:
                method.append("<think> tags")
            print(f"      -> Thinking method: {', '.join(method)}")

    print(f"\n[WARN] UNCONTROLLABLE THINKING MODELS ({len(uncontrollable)}) - NEED WARNING:")
    for r in uncontrollable:
        print(f"   {r.model_name}")
        reasons = []
        if r.thinking_field_when_disabled:
            reasons.append("Still produces 'thinking' field with think=false")
        if r.think_tags_when_disabled:
            reasons.append("Still produces <think> tags with think=false")
        if r.thinks_without_param and not r.supports_think_param:
            reasons.append("Produces <think> tags and doesn't support think param")
        for reason in reasons:
            print(f"      -> {reason}")

    if errors:
        print(f"\n[ERR] ERRORS ({len(errors)}):")
        for r in errors:
            print(f"   {r.model_name}: {r.error_message}")

    # Generate detection patterns
    print("\n")
    print("=" * 70)
    print("RECOMMENDED DETECTION STRATEGY")
    print("=" * 70)

    # Extract model family patterns
    uncontrollable_patterns = set()
    for r in uncontrollable:
        model_lower = r.model_name.lower()
        # Extract base name (before :size)
        base = model_lower.split(":")[0]
        uncontrollable_patterns.add(base)

    controllable_patterns = set()
    for r in controllable:
        model_lower = r.model_name.lower()
        base = model_lower.split(":")[0]
        if base not in uncontrollable_patterns:
            controllable_patterns.add(base)

    print("\n# Models that CANNOT be prevented from thinking (show warning):")
    print(f"UNCONTROLLABLE_THINKING_MODELS = {sorted(uncontrollable_patterns)}")

    print("\n# Models that CAN be controlled with think=false:")
    print(f"CONTROLLABLE_THINKING_MODELS = {sorted(controllable_patterns)}")

    # Generate code suggestion
    print("\n")
    print("=" * 70)
    print("SUGGESTED CODE FOR config.py")
    print("=" * 70)
    print("""
# Models that produce <think> blocks even with think=false
# These need a WARNING displayed to the user
UNCONTROLLABLE_THINKING_MODELS = [""")
    for p in sorted(uncontrollable_patterns):
        print(f'    "{p}",')
    print("""]

# Models that support think=false to disable thinking
# No warning needed - just use think=false
CONTROLLABLE_THINKING_MODELS = [""")
    for p in sorted(controllable_patterns):
        print(f'    "{p}",')
    print("""]
""")

    # Detailed results table
    print("\n")
    print("=" * 70)
    print("DETAILED RESULTS TABLE")
    print("=" * 70)
    print(f"\n{'Model':<35} {'Behavior':<15} {'think=T':<10} {'think=F':<10} {'NoParam':<10}")
    print("-" * 80)
    for r in results:
        think_t = "thinks" if r.thinks_when_enabled else "-"
        think_f = "THINKS!" if r.thinks_when_disabled else "-"
        no_param = "thinks" if r.thinks_without_param else "-"
        if r.behavior == ThinkingBehavior.ERROR:
            print(f"{r.model_name:<35} {'ERROR':<15} {r.error_message[:30]}")
        else:
            print(f"{r.model_name:<35} {r.behavior.value:<15} {think_t:<10} {think_f:<10} {no_param:<10}")

    # Return results for programmatic use
    return {
        "standard": [r.model_name for r in standard],
        "controllable": [r.model_name for r in controllable],
        "uncontrollable": [r.model_name for r in uncontrollable],
        "no_think_support": [r.model_name for r in no_think],
        "errors": [(r.model_name, r.error_message) for r in errors],
        "patterns": {
            "uncontrollable": list(uncontrollable_patterns),
            "controllable": list(controllable_patterns),
        },
        "detailed_results": [
            {
                "model": r.model_name,
                "behavior": r.behavior.value,
                "thinks_when_enabled": r.thinks_when_enabled,
                "thinks_when_disabled": r.thinks_when_disabled,
                "thinks_without_param": r.thinks_without_param,
                "supports_think_param": r.supports_think_param,
                "thinking_method": {
                    "field_enabled": r.thinking_field_when_enabled,
                    "tags_enabled": r.think_tags_when_enabled,
                    "field_disabled": r.thinking_field_when_disabled,
                    "tags_disabled": r.think_tags_when_disabled,
                }
            }
            for r in results
        ]
    }


async def test_with_actual_provider():
    """Test the actual OllamaProvider detection logic"""
    print("\n")
    print("=" * 70)
    print("TESTING ACTUAL PROVIDER DETECTION")
    print("=" * 70)

    from src.core.llm import OllamaProvider

    # Test a few key models
    test_models = ["qwen3:14b", "qwen3:30b", "gemma3:12b"]

    for model in test_models:
        print(f"\n[PROVIDER TEST] {model}")
        provider = OllamaProvider(
            api_endpoint=f"{OLLAMA_ENDPOINT}/api/generate",
            model=model,
            context_window=2048,
            log_callback=lambda t, m: print(f"  {t}: {m}")
        )

        try:
            behavior = await provider._detect_thinking_behavior()
            print(f"  -> Behavior: {behavior.value}")
            print(f"  -> Supports think param: {provider._supports_think_param}")
        except Exception as e:
            print(f"  -> ERROR: {e}")
        finally:
            await provider.close()


async def main():
    """Main entry point"""
    # You can specify specific models to test, or leave None to test all
    specific_models = None  # e.g., ["qwen3:14b", "llama3.2:latest"]

    results = await run_all_tests(OLLAMA_ENDPOINT, specific_models)

    # Also test the actual provider logic
    await test_with_actual_provider()

    # Save results to JSON for later analysis
    if results:
        with open("thinking_detection_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("\n[INFO] Results saved to thinking_detection_results.json")


if __name__ == "__main__":
    asyncio.run(main())
