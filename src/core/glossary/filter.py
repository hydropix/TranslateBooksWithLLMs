"""
Per-chunk glossary filtering.

Latin terms are matched with word boundaries (so "Fan" does not match "Fantasy").
Boundaries apply per edge, and only on an edge that is a word character, so
terms like "C++" or "Ph.D." still match. CJK terms are matched as substrings
(no word boundary concept in CJK scripts). Overlapping occurrences resolve to
the longest one, so a one-character CJK name does not fire inside a longer
glossary term that contains it (the term-fix matcher shares this logic).
The filter returns only the subset of glossary entries that actually appear in
the chunk, sorted by source-term length (longest first).

When the per-chunk cap is hit, the kept subset is selected by occurrence count
(most frequent first, length as tiebreaker), then re-sorted by length for
output stability.

Source terms may declare alternative forms separated by '|' to handle inflected
languages (e.g. "Москва|Москве|Москвы|Москвой -> Moscou"). The filter matches
the entry if ANY of the alternatives appears in the chunk; occurrence counts
are summed across alternatives.
"""
import re
from typing import Dict, List, Tuple

from src.core.glossary.models import GlossaryConfig

_CJK_RE = re.compile(r'[぀-ゟ゠-ヿ一-鿿가-힯㐀-䶿]')


def _is_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


_WORD_CHAR_RE = re.compile(r'\w')


def term_pattern(term: str, prefix: bool = False) -> str:
    """Regex for one term, shared by the glossary filter and the term-fix matcher.

    CJK terms are plain substrings. Otherwise each edge gets its own boundary,
    and only when that edge is a word character: "C++" is anchored on the
    left only, "Ph.D." too, "(x)" on neither side. With `prefix`, the term
    may be followed by more word characters (a "term*" wildcard).
    """
    body = re.escape(term)
    if _is_cjk(term):
        return body
    left = r"(?<!\w)" if _WORD_CHAR_RE.match(term[0]) else ""
    if prefix:
        right = r"\w*"
    else:
        right = r"(?!\w)" if _WORD_CHAR_RE.match(term[-1]) else ""
    return f"{left}{body}{right}"


def longest_non_overlapping(candidates: List[Tuple]) -> List[Tuple]:
    """Keep non-overlapping (start, end, ...) matches.

    On overlap the match that starts first wins; at the same start the longest
    one wins. Returned in text order.
    """
    ordered = sorted(candidates, key=lambda c: (c[0], -(c[1] - c[0])))
    kept: List[Tuple] = []
    last_end = -1
    for cand in ordered:
        if cand[0] < last_end:
            continue
        kept.append(cand)
        last_end = cand[1]
    return kept


def _split_alternatives(source: str) -> List[str]:
    """Split a source term on '|' into non-empty stripped alternatives."""
    if "|" not in source:
        stripped = source.strip()
        return [stripped] if stripped else []
    return [a.strip() for a in source.split("|") if a.strip()]


def _max_alt_length(source: str) -> int:
    """Length used for sort: the longest alternative wins (overlap handling)."""
    alts = _split_alternatives(source)
    return max((len(a) for a in alts), default=0)


def filter_glossary(
    chunk: str,
    glossary_terms: Dict[str, str],
    config: GlossaryConfig = None,
) -> Tuple[Dict[str, str], bool]:
    """
    Return only the glossary entries that appear in the chunk.

    Args:
        chunk: The source text to scan.
        glossary_terms: {source_term: translated_term}. A source_term may
            declare alternative inflected forms separated by '|'.
        config: GlossaryConfig (max_entries cap, case sensitivity).

    Returns:
        (filtered_terms, capped) where filtered_terms preserves order
        (longest source first) and capped is True if the cap was hit.
    """
    if not chunk or not glossary_terms:
        return {}, False

    config = config or GlossaryConfig()
    flags = 0 if config.case_sensitive else re.IGNORECASE

    # Sort by longest alternative descending so longer terms (e.g.
    # "Li Fanqing") are checked before shorter prefixes (e.g. "Li Fan").
    sorted_terms = sorted(
        glossary_terms.items(),
        key=lambda kv: _max_alt_length(kv[0]),
        reverse=True,
    )

    # Every occurrence of every alternative, then overlaps resolved to the
    # longest match: an occurrence inside a longer matched term does not
    # count for the shorter one.
    candidates: List[Tuple[int, int, int]] = []  # (start, end, entry_index)
    for entry_index, (source, _target) in enumerate(sorted_terms):
        for alt in _split_alternatives(source):
            for m in re.finditer(term_pattern(alt), chunk, flags):
                if m.end() > m.start():
                    candidates.append((m.start(), m.end(), entry_index))

    counts: Dict[int, int] = {}
    for _start, _end, entry_index in longest_non_overlapping(candidates):
        counts[entry_index] = counts.get(entry_index, 0) + 1

    matched: List[Tuple[str, str, int]] = [  # (source, target, occurrence_count)
        (source, target, counts[entry_index])
        for entry_index, (source, target) in enumerate(sorted_terms)
        if counts.get(entry_index)
    ]

    capped = False
    if config.max_entries and len(matched) > config.max_entries:
        capped = True
        # When capping, keep the most frequent terms first (length as
        # tiebreaker so longer-and-rarer beats shorter-and-rarer). This
        # is more useful than the previous length-only cut, which could
        # drop a high-frequency 2-char CJK name in favor of 50 longer
        # but rarer entries.
        kept = set(
            (s, t) for s, t, _ in
            sorted(matched, key=lambda x: (x[2], _max_alt_length(x[0])), reverse=True)[: config.max_entries]
        )
        # Preserve the original length-desc order in the output so the
        # rendered block stays predictable for the LLM.
        matched = [(s, t, c) for s, t, c in matched if (s, t) in kept]

    return {s: t for s, t, _ in matched}, capped
