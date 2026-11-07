"""
Locate the wrong renderings named by correction rules.

Matching follows the glossary filter (src/core/glossary/filter.py): Latin
alternatives are anchored on word boundaries so "Fan" does not match inside
"Fantasy", CJK alternatives are plain substrings. Unlike the filter, this
returns positions, because the term-fix pass edits the text in place.
"""
import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from src.core.glossary.filter import _is_cjk
from src.core.term_fix.rules import Alternative, TermRule

_WORD_CHAR_RE = re.compile(r"\w")


@dataclass(frozen=True)
class Span:
    """One occurrence of a wrong rendering: text[start:end] == matched."""
    start: int
    end: int
    rule_index: int
    alternative_index: int
    matched: str


def _alternative_pattern(alt: Alternative) -> str:
    """Regex for one alternative. Boundaries only apply on word-character edges,
    so an alternative ending in punctuation (e.g. "Mr.") still matches."""
    body = re.escape(alt.text)
    if _is_cjk(alt.text):
        # No word boundaries in CJK scripts: a prefix wildcard adds nothing.
        return body
    left = r"(?<!\w)" if _WORD_CHAR_RE.match(alt.text[0]) else ""
    if alt.prefix:
        right = r"\w*"
    else:
        right = r"(?!\w)" if _WORD_CHAR_RE.match(alt.text[-1]) else ""
    return f"{left}{body}{right}"


def find_spans(
    text: str,
    rules: Sequence[TermRule],
    case_sensitive: bool = False,
) -> List[Span]:
    """Return non-overlapping occurrences of every rule's alternatives, in text order.

    On overlap, the occurrence that starts first wins; at the same start, the
    longest one wins (so "Lord Ashen" beats "Lord Ashe").
    """
    if not text or not rules:
        return []
    flags = 0 if case_sensitive else re.IGNORECASE

    candidates: List[Tuple[int, int, int, int]] = []  # (start, end, rule, alt)
    for rule_index, rule in enumerate(rules):
        for alt_index, alt in enumerate(rule.alternatives):
            for m in re.finditer(_alternative_pattern(alt), text, flags):
                if m.end() > m.start():
                    candidates.append((m.start(), m.end(), rule_index, alt_index))

    candidates.sort(key=lambda c: (c[0], -(c[1] - c[0])))
    spans: List[Span] = []
    last_end = -1
    for start, end, rule_index, alt_index in candidates:
        if start < last_end:
            continue
        spans.append(Span(start, end, rule_index, alt_index, text[start:end]))
        last_end = end
    return spans
