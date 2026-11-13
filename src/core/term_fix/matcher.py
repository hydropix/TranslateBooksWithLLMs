"""
Locate the wrong renderings named by correction rules.

Matching shares the glossary filter's logic (src/core/glossary/filter.py):
Latin alternatives are anchored on word boundaries so "Fan" does not match
inside "Fantasy", CJK alternatives are plain substrings, and overlaps resolve
to the longest match. Unlike the filter, this returns positions, because the
term-fix pass edits the text in place.
"""
import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from src.core.glossary.filter import longest_non_overlapping, term_pattern
from src.core.term_fix.rules import Alternative, TermRule


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
    so an alternative ending in punctuation (e.g. "Mr.") still matches. In CJK
    scripts there are no word boundaries, so a prefix wildcard adds nothing."""
    return term_pattern(alt.text, prefix=alt.prefix)


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

    return [
        Span(start, end, rule_index, alt_index, text[start:end])
        for start, end, rule_index, alt_index in longest_non_overlapping(candidates)
    ]
