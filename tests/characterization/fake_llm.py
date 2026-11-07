"""Deterministic echo LLM provider for characterization tests.

The provider parses the source/draft text out of the prompt (the block the
real prompts wrap between ``INPUT_TAG_IN`` / ``INPUT_TAG_OUT``) and returns it
verbatim, wrapped in the translation tags. Echoing the source means:

* placeholders (``[[0]]`` / ``[0]``) survive untouched, so EPUB/DOCX
  placeholder validation passes and the happy-path is exercised;
* subtitle index markers (``[3]``) survive, so SRT remapping succeeds;
* the run is fully deterministic — no randomness, no time dependence.

Token counts are derived from string length so cost/token fields are stable
too (the recorder strips them anyway).
"""

from typing import Optional

from src.config import (
    INPUT_TAG_IN,
    INPUT_TAG_OUT,
    TRANSLATE_TAG_IN,
    TRANSLATE_TAG_OUT,
)
from src.core.llm.base import LLMProvider, LLMResponse


def _extract_source_block(prompt: str) -> str:
    """Return the content of the LAST INPUT_TAG block in the prompt.

    The user prompt always contains exactly one such block (the chunk to
    translate or the draft to refine). Taking the last occurrence is robust
    even if an example block ever leaks into the same string.
    """
    start = prompt.rfind(INPUT_TAG_IN)
    if start == -1:
        return prompt.strip()
    start += len(INPUT_TAG_IN)
    end = prompt.find(INPUT_TAG_OUT, start)
    if end == -1:
        return prompt[start:].strip()
    return prompt[start:end].strip("\n")


class FakeEchoProvider(LLMProvider):
    """An LLMProvider that echoes the source text back as the translation."""

    def __init__(self, model: str = "fake-echo", **_kwargs):
        super().__init__(model=model, api_keys=None, provider_name="fake")
        # Attributes various callers read directly on the provider/client.
        self.context_window = _kwargs.get("context_window") or 4096
        self._is_thinking_model = False

    async def generate(
        self,
        prompt: str,
        timeout: int = 0,
        system_prompt: Optional[str] = None,
    ) -> Optional[LLMResponse]:
        source = _extract_source_block(prompt)
        content = f"{TRANSLATE_TAG_IN}{source}{TRANSLATE_TAG_OUT}"
        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = max(1, len(source) // 4)
        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            context_used=prompt_tokens + completion_tokens,
            context_limit=self.context_window,
            was_truncated=False,
            was_fallback=False,
        )

    async def _detect_thinking_model(self) -> bool:
        return False


def _fake_create_llm_provider(provider_type: str = "ollama", **kwargs) -> FakeEchoProvider:
    """Drop-in replacement for ``create_llm_provider`` used by monkeypatching."""
    return FakeEchoProvider(model=kwargs.get("model", "fake-echo"), **{
        k: v for k, v in kwargs.items() if k == "context_window"
    })


# Every module that resolves ``create_llm_provider`` at call time or holds a
# module-level binding to it. Patching all of them guarantees the fake is used
# regardless of the format path taken (txt/srt via LLMClient, epub via
# LLMClient, docx via the raw factory, refine via _create_llm_client).
_PATCH_TARGETS = (
    "src.core.llm.factory.create_llm_provider",
    "src.core.llm.create_llm_provider",
    "src.core.llm_client.create_llm_provider",
)


def install(monkeypatch) -> None:
    """Patch all ``create_llm_provider`` lookup sites to the echo provider.

    Echoing is by design an untranslated answer, which the source-residue check
    (issue #270) would flag and retry; the goldens pin the happy path, so the
    check is switched off for them.
    """
    import src.config
    monkeypatch.setattr(src.config, "SOURCE_RESIDUE_CHECK_ENABLED", False)
    for target in _PATCH_TARGETS:
        monkeypatch.setattr(target, _fake_create_llm_provider, raising=True)
