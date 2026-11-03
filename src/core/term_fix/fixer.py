"""
Apply correction rules to a list of text units.

A unit is whatever a format adapter can edit independently: the whole file
for TXT/SRT, one text node for EPUB, one paragraph for DOCX. The fixer finds
the matches, cuts sentence snippets, hands every snippet of every unit to a
corrector in one go (so the LLM batches span the whole book), then turns each
accepted correction into minimal character edits on its unit.
"""
import bisect
from typing import Callable, List, Optional, Sequence, Tuple

from src.core.term_fix.diffing import minimal_edits
from src.core.term_fix.literal import LiteralCorrector, match_case
from src.core.term_fix.matcher import find_spans
from src.core.term_fix.models import (
    STATUS_LITERAL,
    Correction,
    Occurrence,
    SnippetItem,
    TermFixResult,
)
from src.core.term_fix.rules import TermRule
from src.core.term_fix.snippets import Snippet, build_snippets, splice

#: locate(unit_index, offset_in_unit) -> human-readable location for the report.
Locator = Callable[[int, int], str]

_Planned = Tuple[int, Snippet, SnippetItem]


def line_locator(text: str) -> Locator:
    """Locator for a single-unit text: 'line N'."""
    line_starts = [0] + [i + 1 for i, c in enumerate(text) if c == "\n"]
    return lambda _unit, pos: f"line {bisect.bisect_right(line_starts, pos)}"


def plan_snippets(
    units: Sequence[str],
    rules: Sequence[TermRule],
    case_sensitive: bool = False,
) -> List[_Planned]:
    """Every snippet of every unit, in document order."""
    planned: List[_Planned] = []
    for unit_index, text in enumerate(units):
        spans = find_spans(text, rules, case_sensitive=case_sensitive)
        for snip in build_snippets(text, spans):
            item = SnippetItem(
                text=text[snip.start:snip.end],
                spans=tuple(
                    (s.start - snip.start, s.end - snip.start, s.rule_index)
                    for s in snip.spans
                ),
            )
            planned.append((unit_index, snip, item))
    return planned


def assemble(
    units: Sequence[str],
    rules: Sequence[TermRule],
    planned: Sequence[_Planned],
    corrections: Sequence[Correction],
    locate: Locator,
) -> TermFixResult:
    """Splice accepted corrections into their units and build the occurrences."""
    edits: List[List[Tuple[int, int, str]]] = [[] for _ in units]
    occurrences: List[Occurrence] = []
    for (unit_index, snip, item), corr in zip(planned, corrections):
        after = item.text if corr.text is None else corr.text
        if corr.text is not None:
            edits[unit_index].extend(minimal_edits(item.text, corr.text, base=snip.start))
        for span in snip.spans:
            replacement = rules[span.rule_index].replacement
            if corr.status == STATUS_LITERAL:
                replacement = match_case(span.matched, replacement)
            occurrences.append(Occurrence(
                rule_index=span.rule_index,
                alternative_index=span.alternative_index,
                location=locate(unit_index, span.start),
                matched=span.matched,
                replacement=replacement,
                before=item.text,
                after=after,
                status=corr.status,
                reason=corr.reason,
            ))
    texts = [splice(text, unit_edits) if unit_edits else text
             for text, unit_edits in zip(units, edits)]
    return TermFixResult(texts=texts, edits=edits, occurrences=occurrences)


async def fix_units(
    units: Sequence[str],
    rules: Sequence[TermRule],
    locate: Locator,
    corrector=None,
    case_sensitive: bool = False,
) -> TermFixResult:
    """Correct every unit. `corrector` defaults to literal replacement; pass a
    SmartCorrector for LLM corrections."""
    corrector = corrector or LiteralCorrector(rules)
    planned = plan_snippets(units, rules, case_sensitive=case_sensitive)
    corrections = await corrector.correct([item for _u, _s, item in planned]) if planned else []
    return assemble(units, rules, planned, corrections, locate)


def apply_literal_fix(
    text: str,
    rules: Sequence[TermRule],
    case_sensitive: bool = False,
) -> TermFixResult:
    """Synchronous literal fix of a single text. Bytes outside the matches are
    left exactly as they were."""
    corrector = LiteralCorrector(rules)
    planned = plan_snippets([text], rules, case_sensitive=case_sensitive)
    corrections = corrector.correct_sync([item for _u, _s, item in planned])
    return assemble([text], rules, planned, corrections, line_locator(text))
