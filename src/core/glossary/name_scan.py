"""
Whole-document scan for recurring proper names (no LLM involved).

The NER pass only ever shows the model a few thousand characters of a book.
Evenly spaced excerpts catch the main cast, who appear everywhere, but rarely a
secondary character who lives in a handful of chapters — and that character's
name then drifts from one chunk to the next. This module reads the whole text
cheaply, finds capitalized words that recur in mid-sentence position, and
builds the excerpts around them, so a single NER call covers the names of the
entire book for the same prompt size.

Only cased scripts benefit. A text without letter case (CJK, Thai, Arabic, ...)
yields no candidates, and a text where capitals are everywhere (German
capitalizes every noun) is detected and skipped; both fall back to the evenly
spaced excerpts of `document_sampler.take_distributed_samples`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.utils import document_sampler

# A word: letters, optionally joined by apostrophes or hyphens (O'Brien, Jean-Luc).
_WORD_RE = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*")
_POSSESSIVE_RE = re.compile(r"['’]s$", re.IGNORECASE)

# Skipped when looking backwards for the previous meaningful character.
_OPENERS = "\"'“”‘’«»„‚‹›([{*_—–-"
_SENTENCE_END = ".!?…"

# Abbreviations whose trailing period does not end a sentence. Also never
# candidates themselves: a title is not a name.
_TITLE_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "mx", "dr", "st", "mt", "prof", "sgt", "capt", "lt",
    "col", "gen", "rev", "fr", "sr", "jr", "mme", "mlle", "messrs",
})

#: Minimum occurrences for a word to count as recurring.
MIN_OCCURRENCES = 3
#: Minimum share of those occurrences that must sit mid-sentence. Words such as
#: "But" or "Then" are capitalized almost only at sentence starts; names are not.
MIN_MID_SENTENCE_SHARE = 0.3
#: A word written in lowercase more than this many times as often as it is
#: capitalized is a common word in titles ("The", "How"), not a name. A name
#: that is also a frequent word ("Will") is lost; the evenly spaced excerpts
#: still give it a chance.
MAX_LOWERCASE_RATIO = 2
#: Above this share of mid-sentence capitalized words, capitals carry no signal
#: (German nouns, title-cased text) and the scan gives up.
MAX_CAPITALIZED_SHARE = 0.12
#: Characters of context kept around each targeted name. Shorter windows fit
#: more names, but below a few sentences the sample reads like a list of names
#: and models stop translating them: with 180-char windows gemma4:31b returned
#: ~60 English-to-Chinese candidates with nearly every target copied from the
#: source (dropped as identity pairs); at 400 it translated them again.
WINDOW_CHARS = 400
#: Part of the budget kept for evenly spaced excerpts: they carry the general
#: context and the uncapitalized terms (techniques, invented words) the scan
#: cannot see.
BACKGROUND_SHARE = 0.25
#: Occurrences tried per name when choosing its window.
_OCCURRENCES_TRIED = 8


@dataclass
class NameCandidate:
    """A capitalized word that recurs in mid-sentence position."""

    name: str
    count: int = 0
    mid_sentence: int = 0
    positions: list[int] = field(default_factory=list)


def find_recurring_names(text: str) -> list[NameCandidate]:
    """Return recurring capitalized words, most frequent first.

    Returns [] when the text has no usable case signal (uncased script, or
    capitals on most nouns).
    """
    if not text:
        return []

    stats: dict[str, NameCandidate] = {}
    total_words = 0
    capitalized_mid = 0

    lowercase: dict[str, int] = {}

    for match in _WORD_RE.finditer(text):
        total_words += 1
        word = _POSSESSIVE_RE.sub("", match.group())
        if word.islower():
            lowercase[word] = lowercase.get(word, 0) + 1
            continue
        if len(word) < 2 or not word[0].isupper() or word.isupper():
            continue
        if word.lower() in _TITLE_ABBREVIATIONS or word.startswith(("I'", "I’")):
            continue

        initial = _is_sentence_initial(text, match.start())
        if not initial:
            capitalized_mid += 1

        candidate = stats.get(word)
        if candidate is None:
            candidate = stats[word] = NameCandidate(word)
        candidate.count += 1
        candidate.positions.append(match.start())
        if not initial:
            candidate.mid_sentence += 1

    if not total_words or capitalized_mid / total_words > MAX_CAPITALIZED_SHARE:
        return []

    names = [
        c for c in stats.values()
        if c.count >= MIN_OCCURRENCES
        and c.mid_sentence >= 2
        and c.mid_sentence / c.count >= MIN_MID_SENTENCE_SHARE
        and lowercase.get(c.name.lower(), 0) <= MAX_LOWERCASE_RATIO * c.count
    ]
    names.sort(key=lambda c: (-c.count, c.name))
    return names


def take_glossary_samples(
    text: str,
    total_budget: int,
    num_samples: int,
    min_sample_size: int = 500,
    separator: str = document_sampler.EXCERPT_SEPARATOR,
) -> tuple[str, int, int]:
    """Build the NER sample: a few evenly spaced excerpts plus one short window
    around each recurring name they missed.

    Returns ``(joined_excerpts, excerpt_count, names_targeted)``. When the scan
    finds no names, this is exactly ``take_distributed_samples`` with
    ``names_targeted == 0``. Otherwise the joined text, separators included,
    stays within ``total_budget``: `suggest_terms` truncates anything longer,
    which would silently drop the last windows.
    """
    if len(text) <= total_budget:
        return text, 1, 0

    names = find_recurring_names(text)
    if not names:
        joined, count = document_sampler.take_distributed_samples(
            text, total_budget, num_samples, min_sample_size=min_sample_size,
            separator=separator,
        )
        return joined, count, 0

    background_budget = max(min_sample_size, int(total_budget * BACKGROUND_SHARE))
    background_count = max(1, min(num_samples, background_budget // min_sample_size))
    background, bg_count = document_sampler.take_distributed_samples(
        text, background_budget, background_count, min_sample_size=min_sample_size,
        separator=separator,
    )

    uncovered = {c.name for c in names} - _words_in(background)
    budget = total_budget - len(background)
    windows = _pick_windows(text, names, uncovered, budget, len(separator))
    if not windows:
        return background, bg_count, 0

    excerpts = [text[start:end].strip() for start, end in windows]
    targeted = len(uncovered & _words_in(" ".join(excerpts)))
    return separator.join([background, *excerpts]), bg_count + len(windows), targeted


def _pick_windows(
    text: str,
    names: list[NameCandidate],
    uncovered: set[str],
    budget: int,
    separator_len: int,
) -> list[tuple[int, int]]:
    """Greedily choose non-overlapping windows, most frequent name first, each
    placed where it also covers the most other uncovered names."""
    chosen: list[tuple[int, int]] = []
    remaining = set(uncovered)

    for candidate in names:
        if candidate.name not in remaining:
            continue
        if budget < WINDOW_CHARS // 2 + separator_len:
            break

        best: tuple[int, int] | None = None
        best_covered: set[str] = set()
        for pos in _spread(candidate.positions, _OCCURRENCES_TRIED):
            span = _window_around(text, pos, min(WINDOW_CHARS, budget - separator_len))
            if any(span[0] < end and start < span[1] for start, end in chosen):
                continue
            covered = _words_in(text[span[0]:span[1]]) & remaining
            if len(covered) > len(best_covered):
                best, best_covered = span, covered

        if best is None:
            continue
        chosen.append(best)
        remaining -= best_covered
        budget -= (best[1] - best[0]) + separator_len

    chosen.sort()
    return chosen


def _window_around(text: str, pos: int, size: int) -> tuple[int, int]:
    """A span of about `size` chars centered on `pos`, snapped to whitespace."""
    start = max(0, pos - size // 2)
    end = min(len(text), start + size)
    start = max(0, end - size)
    if start > 0:
        ws = text.find(" ", start, pos)
        if ws != -1:
            start = ws + 1
    if end < len(text):
        ws = text.rfind(" ", pos, end)
        if ws != -1:
            end = ws
    return start, end


def _spread(positions: list[int], limit: int) -> list[int]:
    """Up to `limit` positions evenly spread over the list (book-wide variety)."""
    if len(positions) <= limit:
        return positions
    step = len(positions) / limit
    return [positions[int(i * step)] for i in range(limit)]


def _words_in(text: str) -> set[str]:
    return {_POSSESSIVE_RE.sub("", m.group()) for m in _WORD_RE.finditer(text)}


def _is_sentence_initial(text: str, start: int) -> bool:
    """True when the word at `start` opens a sentence, a line or the text."""
    i = start - 1
    while i >= 0 and (text[i].isspace() or text[i] in _OPENERS):
        if text[i] == "\n":
            return True
        i -= 1
    if i < 0:
        return True
    if text[i] not in _SENTENCE_END:
        return False
    if text[i] == ".":
        word_end = i
        j = i - 1
        while j >= 0 and text[j].isalpha():
            j -= 1
        if text[j + 1:word_end].lower() in _TITLE_ABBREVIATIONS:
            return False
    return True
