"""
Deterministic (no LLM) term replacement with case preservation.

Literal mode cannot inflect the replacement or fix agreement, so it refuses
rules with a prefix wildcard: "Drakbrand*" would match "Drakbrandens" and the
genitive ending would be lost.
"""
import bisect
from dataclasses import dataclass, field
from typing import List, Sequence

from src.core.term_fix.matcher import Span, find_spans
from src.core.term_fix.rules import TermRule
from src.core.term_fix.snippets import build_snippets, splice

STATUS_LITERAL = "literal"


@dataclass(frozen=True)
class Occurrence:
    """One replaced occurrence, with its sentence before and after the change."""
    rule_index: int
    alternative_index: int
    line: int
    matched: str
    replacement: str
    before: str
    after: str
    status: str


@dataclass
class TermFixResult:
    text: str
    occurrences: List[Occurrence] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.occurrences)


def match_case(matched: str, replacement: str) -> str:
    """Carry the casing of the matched text over to the replacement.

    The replacement as the user wrote it is authoritative, except for two
    signals the original text carries: an all-caps occurrence (heading, shout)
    is upper-cased, and a capitalized occurrence (sentence start) gets a
    capital first letter.
    """
    letters = [c for c in matched if c.isalpha()]
    if len(letters) > 1 and all(c.isupper() for c in letters):
        return replacement.upper()
    if letters and matched.lstrip()[:1].isupper() and replacement[:1].islower():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def check_literal_rules(rules: Sequence[TermRule]) -> None:
    """Raise ValueError when a rule cannot be applied without an LLM."""
    for rule in rules:
        if rule.has_prefix_alternative:
            raise ValueError(
                f"Rule '{rule}' uses a '*' prefix wildcard, which literal "
                f"replacement cannot inflect. List the exact forms instead "
                f"(e.g. 'Drakbrand|Drakbranden|Drakbrandens -> ...')."
            )


def _line_of(pos: int, line_starts: List[int]) -> int:
    """1-based line number of offset `pos`."""
    return bisect.bisect_right(line_starts, pos)


def apply_literal_fix(
    text: str,
    rules: Sequence[TermRule],
    case_sensitive: bool = False,
) -> TermFixResult:
    """Replace every occurrence of every rule in `text`. Bytes outside the
    matches are left exactly as they were."""
    check_literal_rules(rules)
    spans = find_spans(text, rules, case_sensitive=case_sensitive)
    if not spans:
        return TermFixResult(text=text)

    def replacement_for(span: Span) -> str:
        return match_case(span.matched, rules[span.rule_index].replacement)

    line_starts = [0] + [i + 1 for i, c in enumerate(text) if c == "\n"]
    occurrences: List[Occurrence] = []
    for snippet in build_snippets(text, spans):
        before = text[snippet.start:snippet.end]
        after = splice(before, [
            (s.start - snippet.start, s.end - snippet.start, replacement_for(s))
            for s in snippet.spans
        ])
        for span in snippet.spans:
            occurrences.append(Occurrence(
                rule_index=span.rule_index,
                alternative_index=span.alternative_index,
                line=_line_of(span.start, line_starts),
                matched=span.matched,
                replacement=replacement_for(span),
                before=before,
                after=after,
                status=STATUS_LITERAL,
            ))

    new_text = splice(text, [(s.start, s.end, replacement_for(s)) for s in spans])
    return TermFixResult(text=new_text, occurrences=occurrences)
