"""
Detection of source-language text left behind in a translation (issue #270).

A chunk used to count as translated as soon as the LLM answered and, in the
placeholder pipeline, kept its tags intact. Nothing checked that the answer was
actually in the target language, so a model that skipped one paragraph of a
long chunk, or echoed the whole chunk back, produced a "successful" chunk that
the retry machinery of issue #261 could never see.

This module answers one question, cheaply and without any LLM call: does this
translation still contain a meaningful amount of the source text?

Two independent signals:

- **Source script residue.** When the source language writes in a script the
  target language does not use (Korean -> English, English -> Japanese,
  Russian -> French, ...), any long run of that script in the output is text
  the model did not translate. Short runs are tolerated: a name, an acronym or
  a brand kept in its original form is legitimate.
- **Echo.** For every language pair, including the ones that share a script
  (English -> French), an output identical to a non-trivial input is an
  untranslated chunk.

Same-script pairs therefore only get the echo check. A per-paragraph language
detector would cover them too, but it is too noisy on short paragraphs and
dialogue to drive retries.

Not every finding is equally reliable. An echo is often legitimate (a
bibliography entry, a list of names, a code block), and so is a Latin
quotation kept inside a Japanese or Russian text. Those findings are
`retry_only`: they earn one corrective retry and nothing more, so a book that
legitimately keeps such text is never left 'partial' with a chunk that every
repair pass would retranslate in vain. Only a leftover run of a non-Latin
source script (Korean, Japanese kana, Chinese, Cyrillic, ...) is `blocking`:
the chunk is then recorded as untranslated.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Sequence, Tuple


# Unicode ranges per script. Only letters matter: digits and punctuation are
# shared by every script and never count as residue.
_SCRIPT_RANGES: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "latin": ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x00D6),
              (0x00D8, 0x00F6), (0x00F8, 0x024F), (0x1E00, 0x1EFF)),
    "cyrillic": ((0x0400, 0x04FF), (0x0500, 0x052F)),
    "greek": ((0x0370, 0x03FF), (0x1F00, 0x1FFF)),
    "arabic": ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF)),
    "hebrew": ((0x0590, 0x05FF),),
    "armenian": ((0x0530, 0x058F),),
    "georgian": ((0x10A0, 0x10FF),),
    "devanagari": ((0x0900, 0x097F),),
    "bengali": ((0x0980, 0x09FF),),
    "gurmukhi": ((0x0A00, 0x0A7F),),
    "gujarati": ((0x0A80, 0x0AFF),),
    "tamil": ((0x0B80, 0x0BFF),),
    "telugu": ((0x0C00, 0x0C7F),),
    "kannada": ((0x0C80, 0x0CFF),),
    "malayalam": ((0x0D00, 0x0D7F),),
    "sinhala": ((0x0D80, 0x0DFF),),
    "thai": ((0x0E00, 0x0E7F),),
    "lao": ((0x0E80, 0x0EFF),),
    "myanmar": ((0x1000, 0x109F),),
    "khmer": ((0x1780, 0x17FF),),
    "ethiopic": ((0x1200, 0x137F),),
    "hangul": ((0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7A3)),
    "kana": ((0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF)),
    "han": ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)),
}

# Scripts written without spaces between words: a run is measured in
# characters there, and in words everywhere else.
_CHARACTER_SCRIPTS = frozenset({"han", "kana", "hangul", "thai", "lao", "myanmar", "khmer"})

# Scripts each language is written in, keyed by primary language subtag.
# Languages absent from this table are assumed to use the Latin script; an
# unknown language name (no code at all) disables the script check.
_LANGUAGE_SCRIPTS: Dict[str, FrozenSet[str]] = {
    "zh": frozenset({"han"}),
    "ja": frozenset({"han", "kana"}),
    "ko": frozenset({"hangul", "han"}),
    "ru": frozenset({"cyrillic"}), "uk": frozenset({"cyrillic"}),
    "be": frozenset({"cyrillic"}), "bg": frozenset({"cyrillic"}),
    "mk": frozenset({"cyrillic"}), "kk": frozenset({"cyrillic"}),
    "ky": frozenset({"cyrillic"}), "mn": frozenset({"cyrillic"}),
    "tg": frozenset({"cyrillic"}),
    # Serbian is routinely written in both alphabets.
    "sr": frozenset({"cyrillic", "latin"}),
    "el": frozenset({"greek"}),
    "ar": frozenset({"arabic"}), "fa": frozenset({"arabic"}),
    "ur": frozenset({"arabic"}), "ps": frozenset({"arabic"}),
    "ku": frozenset({"arabic", "latin"}), "sd": frozenset({"arabic"}),
    "ug": frozenset({"arabic"}),
    "he": frozenset({"hebrew"}), "yi": frozenset({"hebrew"}),
    "hy": frozenset({"armenian"}),
    "ka": frozenset({"georgian"}),
    "hi": frozenset({"devanagari"}), "mr": frozenset({"devanagari"}),
    "ne": frozenset({"devanagari"}), "sa": frozenset({"devanagari"}),
    "bn": frozenset({"bengali"}), "as": frozenset({"bengali"}),
    "pa": frozenset({"gurmukhi"}),
    "gu": frozenset({"gujarati"}),
    "ta": frozenset({"tamil"}),
    "te": frozenset({"telugu"}),
    "kn": frozenset({"kannada"}),
    "ml": frozenset({"malayalam"}),
    "si": frozenset({"sinhala"}),
    "th": frozenset({"thai"}),
    "lo": frozenset({"lao"}),
    "my": frozenset({"myanmar"}),
    "km": frozenset({"khmer"}),
    "am": frozenset({"ethiopic"}), "ti": frozenset({"ethiopic"}),
}

# A run at least this long is untranslated text, not a kept name. Latin gets a
# higher bar: non-Latin target texts legitimately keep Latin brand names,
# acronyms and short quotations.
MIN_RUN_CHARACTERS = 6
MIN_RUN_WORDS = 4
MIN_RUN_WORDS_LATIN = 6

# Scattered short leftovers also count once they add up to this share of the
# source-script text of the input.
MAX_RESIDUE_RATIO = 0.3

# Below this many letters an identical output is not suspicious: titles, names
# and one-word lines are often the same in both languages.
MIN_ECHO_LETTERS = 20

# Tag placeholders ([id0], see PLACEHOLDER_PATTERN in src/config.py, plus the
# legacy [[0]] form still accepted by older checkpoints).
_PLACEHOLDER_RE = re.compile(r"\[id\d+\]|\[\[\d+\]\]")


@dataclass(frozen=True)
class ResidueReport:
    """What the detector found in one translation.

    Attributes:
        kind: 'echo' when the output repeats the input, 'script' when it keeps
            long runs of a script only the source language uses.
        sample: A short excerpt of the leftover text, for the log.
        residue_units: Characters (or words) of source script left in the
            output. Lower is better when comparing attempts.
        retry_only: True when the finding may well be legitimate (an echo, or
            leftover Latin text): worth one corrective retry, never enough to
            record the chunk as untranslated.
    """

    kind: str
    sample: str
    residue_units: int
    retry_only: bool = False

    @property
    def blocking(self) -> bool:
        """True when the chunk must be recorded as untranslated."""
        return not self.retry_only


def get_language_code(language: Optional[str]) -> Optional[str]:
    """Language name -> ISO code, imported lazily.

    `src.core.epub` imports the translator, which imports this module, so a
    module-level import of `src.core.epub.lang_support` would be circular.
    """
    from src.core.epub.lang_support import get_language_code as _resolve
    return _resolve(language)


def _script_of(char: str) -> Optional[str]:
    cp = ord(char)
    for script, ranges in _SCRIPT_RANGES.items():
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return script
    return None


def language_scripts(language: Optional[str]) -> FrozenSet[str]:
    """Scripts a language is written in, or an empty set when it is unknown."""
    code = get_language_code(language)
    if not code:
        return frozenset()
    return _LANGUAGE_SCRIPTS.get(code.split("-")[0].lower(), frozenset({"latin"}))


def distinctive_source_scripts(source_language: str, target_language: str) -> FrozenSet[str]:
    """Scripts the source language uses and the target language does not."""
    source = language_scripts(source_language)
    target = language_scripts(target_language)
    if not source or not target:
        return frozenset()
    return source - target


def _strip_placeholders(text: str) -> str:
    return _PLACEHOLDER_RE.sub(" ", text or "")


def _runs(text: str, scripts: FrozenSet[str]):
    """Yield (run_text, units) for each maximal run of `scripts` letters.

    A run may span whitespace, digits and punctuation, but any letter of
    another script ends it, so "Kim Min-jun" between two English words never
    joins them into one run.
    """
    start = None
    last_letter = None
    units = 0
    in_word = False
    character_based = bool(scripts & _CHARACTER_SCRIPTS)
    for i, ch in enumerate(text):
        if ch.isalpha() or _script_of(ch) is not None:
            script = _script_of(ch)
            if script in scripts:
                if start is None:
                    start = i
                    units = 0
                    in_word = False
                last_letter = i
                if character_based and script in _CHARACTER_SCRIPTS:
                    units += 1
                elif not in_word:
                    units += 1
                in_word = True
                continue
            # A letter of another script closes the current run.
            if start is not None:
                yield text[start:last_letter + 1], units
                start = None
            in_word = False
            continue
        in_word = False
    if start is not None:
        yield text[start:last_letter + 1], units


def _run_threshold(scripts: FrozenSet[str]) -> int:
    if scripts & _CHARACTER_SCRIPTS:
        return MIN_RUN_CHARACTERS
    if scripts == frozenset({"latin"}):
        return MIN_RUN_WORDS_LATIN
    return MIN_RUN_WORDS


def _normalize_for_echo(text: str) -> str:
    return " ".join(_strip_placeholders(text).split()).casefold()


def _letter_count(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha())


def detect_source_residue(
    source_text: str,
    translated_text: str,
    source_language: str,
    target_language: str,
) -> Optional[ResidueReport]:
    """Return a report when `translated_text` still holds source text, else None.

    Args:
        source_text: The text sent to the LLM (placeholders allowed).
        translated_text: The LLM's answer (placeholders allowed).
        source_language: Source language name or code.
        target_language: Target language name or code.

    Returns:
        A ResidueReport, or None when the translation looks complete or when
        the check does not apply (unknown or identical languages, empty text).
    """
    from src import config
    if not config.SOURCE_RESIDUE_CHECK_ENABLED:
        return None
    if not source_text or not translated_text:
        return None
    src_code = get_language_code(source_language)
    tgt_code = get_language_code(target_language)
    if not src_code or not tgt_code or src_code.split("-")[0] == tgt_code.split("-")[0]:
        return None

    source_clean = _strip_placeholders(source_text)
    output_clean = _strip_placeholders(translated_text)

    scripts = distinctive_source_scripts(source_language, target_language)
    if scripts:
        threshold = _run_threshold(scripts)
        source_units = sum(units for _, units in _runs(source_clean, scripts))
        if source_units:
            runs = list(_runs(output_clean, scripts))
            total = sum(units for _, units in runs)
            longest = max(runs, key=lambda r: r[1], default=("", 0))
            if longest[1] >= threshold or (
                total >= threshold and total >= MAX_RESIDUE_RATIO * source_units
            ):
                return ResidueReport("script", _excerpt(longest[0]), total,
                                     retry_only=scripts == frozenset({"latin"}))

    if (_letter_count(source_clean) >= MIN_ECHO_LETTERS
            and _normalize_for_echo(source_text) == _normalize_for_echo(translated_text)):
        return ResidueReport("echo", _excerpt(source_clean.strip()),
                             _letter_count(source_clean), retry_only=True)

    return None


def detect_blocking_residue(
    source_text: str,
    translated_text: str,
    source_language: str,
    target_language: str,
) -> Optional[ResidueReport]:
    """Like `detect_source_residue`, keeping only findings that block a chunk.

    This is what the pipelines use, after the request already retried once, to
    decide whether a chunk is recorded as untranslated.
    """
    report = detect_source_residue(source_text, translated_text, source_language, target_language)
    return report if report is not None and report.blocking else None


def residue_score(report: Optional[ResidueReport]) -> int:
    """Ordering key for picking the best of several attempts (lower is better)."""
    return 0 if report is None else max(1, report.residue_units)


def _excerpt(text: str, limit: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def residue_log_message(report: ResidueReport, source_language: str) -> str:
    """One log line describing a report, shared by every pipeline."""
    if report.kind == "echo":
        return f"the model returned the {source_language} text unchanged: \"{report.sample}\""
    return f"{source_language} text left untranslated: \"{report.sample}\""


__all__: Sequence[str] = (
    "ResidueReport",
    "detect_blocking_residue",
    "detect_source_residue",
    "distinctive_source_scripts",
    "language_scripts",
    "residue_log_message",
    "residue_score",
)
