"""
Smoke test for the LiteLLM provider.

Verifies that the project's factory wiring can route a real translation through
LiteLLM end to end, using a provider-prefixed model and the matching native
credential from .env. This is the test that proves the feature actually works
against a live API, beyond the stubbed unit tests.

Requires the optional dependency:
    pip install "litellm>=1.65,<1.85"

Run from repo root:
    python tests/standalone/manual_litellm_smoke.py

By default it routes to gemini/gemini-2.5-flash (cheap, uses GEMINI_API_KEY).
Override the model via the LITELLM_MODEL env var to test another provider.
"""

import asyncio
import os
import sys
from pathlib import Path

# Force UTF-8 on stdout/stderr so library logs containing emoji don't crash on
# Windows consoles that default to cp1252.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# Make `src` importable when invoked directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

# Importing src.config calls load_dotenv() on the repo's .env, so the native
# provider keys (GEMINI_API_KEY, OPENAI_API_KEY, ...) become visible.
from src import config  # noqa: F401
from src.core.llm.factory import create_llm_provider

DEFAULT_MODEL = "gemini/gemini-2.5-flash"

SOURCE_TEXT = "The quick brown fox jumps over the lazy dog."
SYSTEM_PROMPT = (
    "You are a professional translator. Translate the user's text from English "
    "into French. Wrap the translation between <TRANSLATION> and </TRANSLATION> "
    "tags and output nothing else."
)
USER_PROMPT = f"<TRANSLATION>\n{SOURCE_TEXT}\n</TRANSLATION>"

# Map a model prefix to the native env var LiteLLM expects, so we can give a
# clear precondition error instead of a cryptic auth failure.
_PREFIX_TO_ENV = {
    "gemini/": "GEMINI_API_KEY",
    "openai/": "OPENAI_API_KEY",
    "anthropic/": "ANTHROPIC_API_KEY",
    "mistral/": "MISTRAL_API_KEY",
    "deepseek/": "DEEPSEEK_API_KEY",
}


async def run() -> int:
    try:
        import litellm  # noqa: F401
    except ImportError:
        print('FAIL: litellm not installed. Run: pip install "litellm>=1.65,<1.85"')
        return 1

    model = os.getenv("LITELLM_MODEL", "").strip() or DEFAULT_MODEL

    required_env = next(
        (env for prefix, env in _PREFIX_TO_ENV.items() if model.startswith(prefix)),
        None,
    )
    if required_env and not os.getenv(required_env, "").strip():
        print(f"FAIL: {required_env} is not set in .env (required for model '{model}')")
        return 1

    print("Provider: litellm")
    print(f"Model:    {model}")
    print(f"Source:   {SOURCE_TEXT}")
    print()

    provider = create_llm_provider("litellm", model=model)
    try:
        response = await provider.generate(USER_PROMPT, system_prompt=SYSTEM_PROMPT)
    finally:
        await provider.close()

    if response is None:
        print("FAIL: provider returned None (request failed after retries)")
        return 1

    print(f"Raw response:\n{response.content}\n")

    translation = provider.extract_translation(response.content)
    if not translation or not translation.strip():
        print("FAIL: could not extract translation from <TRANSLATION> tags")
        return 1

    print(f"Translation: {translation.strip()}")
    print(f"Tokens:      {response.prompt_tokens}+{response.completion_tokens}")
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
