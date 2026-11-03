"""
Deterministic (no LLM) term replacement with case preservation.

Literal mode cannot inflect the replacement or fix agreement, so it refuses
rules with a prefix wildcard: "Drakbrand*" would match "Drakbrandens" and the
genitive ending would be lost.
"""
from typing import List, Sequence

from src.core.term_fix.models import STATUS_LITERAL, Correction, SnippetItem
from src.core.term_fix.rules import TermRule
from src.core.term_fix.snippets import splice


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
                f"(e.g. 'Drakbrand|Drakbranden|Drakbrandens -> ...'), or use "
                f"the smart (LLM) mode."
            )


class LiteralCorrector:
    """Corrector that swaps each match for the rule's replacement."""

    def __init__(self, rules: Sequence[TermRule]):
        check_literal_rules(rules)
        self.rules = list(rules)

    def correct_sync(self, items: Sequence[SnippetItem]) -> List[Correction]:
        out: List[Correction] = []
        for item in items:
            edits = [
                (s, e, match_case(item.text[s:e], self.rules[r].replacement))
                for s, e, r in item.spans
            ]
            out.append(Correction(text=splice(item.text, edits), status=STATUS_LITERAL))
        return out

    async def correct(self, items: Sequence[SnippetItem]) -> List[Correction]:
        return self.correct_sync(items)
