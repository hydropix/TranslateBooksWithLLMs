"""Reproduce issue #194 — translated EPUB3 keeps an untranslated Table of Contents.

The bug: when an EPUB3's reader-facing TOC lives in the navigation document
(``nav.xhtml`` with ``<nav epub:type="toc">``), the translation pipeline never
updates its link labels. Body headings get translated, the legacy ``toc.ncx``
gets synced (fix #172), but the ``nav.xhtml`` TOC is left in the source
language. EPUB3 readers that build their TOC from the nav document therefore
show stale / placeholder entries.

This script proves it WITHOUT a real LLM or API key: a fake provider that
UPPERCASES the source stands in for "translation", so anything still in mixed
case after the run was demonstrably not translated.

Run:
    python tests/standalone/repro_issue_194_toc.py
"""

import asyncio
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.config import INPUT_TAG_IN, INPUT_TAG_OUT, TRANSLATE_TAG_IN, TRANSLATE_TAG_OUT
from src.core.llm.base import LLMProvider, LLMResponse
from scripts.build_test_epub import build_epub
from src.core.epub.translator import translate_epub_file


def _extract_source_block(prompt: str) -> str:
    start = prompt.rfind(INPUT_TAG_IN)
    if start == -1:
        return prompt.strip()
    start += len(INPUT_TAG_IN)
    end = prompt.find(INPUT_TAG_OUT, start)
    if end == -1:
        return prompt[start:].strip()
    return prompt[start:end].strip("\n")


class UpperCaseProvider(LLMProvider):
    """Fake translator: returns the source UPPERCASED.

    Uppercasing only touches ASCII/Unicode letters, leaving ``[[0]]``-style
    inline placeholders intact so EPUB placeholder validation still passes.
    Any lowercase letters left in the output were therefore NOT translated.
    """

    def __init__(self, model: str = "fake-upper", **_kwargs):
        super().__init__(model=model, api_keys=None, provider_name="fake")
        self.context_window = _kwargs.get("context_window") or 8192
        self._is_thinking_model = False

    async def generate(self, prompt: str, timeout: int = 0,
                       system_prompt: Optional[str] = None) -> Optional[LLMResponse]:
        source = _extract_source_block(prompt)
        translated = source.upper()
        content = f"{TRANSLATE_TAG_IN}{translated}{TRANSLATE_TAG_OUT}"
        return LLMResponse(
            content=content,
            prompt_tokens=max(1, len(prompt) // 4),
            completion_tokens=max(1, len(source) // 4),
            context_used=max(1, len(prompt) // 4),
            context_limit=self.context_window,
            was_truncated=False,
            was_fallback=False,
        )

    async def _detect_thinking_model(self) -> bool:
        return False


def _fake_factory(provider_type: str = "ollama", **kwargs) -> UpperCaseProvider:
    return UpperCaseProvider(model=kwargs.get("model", "fake-upper"),
                             context_window=kwargs.get("context_window"))


def _install_fake():
    import src.core.llm.factory as factory_mod
    import src.core.llm as llm_pkg
    import src.core.llm_client as client_mod
    for mod in (factory_mod, llm_pkg, client_mod):
        if hasattr(mod, "create_llm_provider"):
            mod.create_llm_provider = _fake_factory


def _read_zip_text(epub_path: Path, suffix: str) -> str:
    with zipfile.ZipFile(epub_path) as z:
        name = next(n for n in z.namelist() if n.endswith(suffix))
        return z.read(name).decode("utf-8")


def _strip_tags(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


async def _run():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        src_epub = tmp / "sampler.epub"
        out_epub = tmp / "sampler.fr.epub"
        build_epub(src_epub)

        _install_fake()
        await translate_epub_file(
            input_filepath=str(src_epub),
            output_filepath=str(out_epub),
            source_language="English",
            target_language="French",
            model_name="fake-upper",
            llm_provider="ollama",
            context_window=8192,
            auto_adjust_context=False,
            max_tokens_per_chunk=120,
            log_callback=lambda code, msg="", **_: print(f"  [log] {code}: {msg}"),
        )

        ch01 = _read_zip_text(out_epub, "ch01.xhtml")
        nav = _read_zip_text(out_epub, "nav.xhtml")
        ncx = _read_zip_text(out_epub, "toc.ncx")

        h1 = _strip_tags(re.search(r"<h1[^>]*>(.*?)</h1>", ch01, re.S).group(1))
        nav_links = [_strip_tags(m) for m in re.findall(r"<a [^>]*>(.*?)</a>", nav, re.S)]
        ncx_labels = re.findall(r"<text>(.*?)</text>", ncx, re.S)

        print("=" * 70)
        print("ISSUE #194 - EPUB3 Table of Contents not translated")
        print("=" * 70)
        print(f"\nChapter 1 <h1> in body  : {h1!r}")
        print("  -> uppercase = TRANSLATED by the pipeline (OK)\n")

        print("nav.xhtml TOC links (EPUB3 reader TOC):")
        for link in nav_links:
            state = "translated" if link.isupper() else "NOT translated  <-- BUG"
            print(f"  {link!r:50} [{state}]")

        print("\ntoc.ncx navLabels (legacy EPUB2 TOC, fix #172):")
        for lbl in ncx_labels:
            if lbl == "The Translator's Sampler":
                continue
            state = "translated" if lbl.isupper() else "not translated"
            print(f"  {lbl!r:50} [{state}]")

        nav_translated = any(l.isupper() for l in nav_links if l)
        ncx_translated = any(l.isupper() for l in ncx_labels if l and l != "The Translator's Sampler")

        print("\n" + "-" * 70)
        print(f"NCX TOC translated?  {ncx_translated}")
        print(f"NAV TOC translated?  {nav_translated}")
        if ncx_translated and not nav_translated:
            print("\n>>> BUG REPRODUCED: body + NCX are translated, but the EPUB3")
            print(">>> nav.xhtml TOC still shows the original source-language titles.")
            return 1
        print("\n>>> Bug not reproduced.")
        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_run()))
