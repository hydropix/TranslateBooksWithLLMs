"""
Scan summary and change report for the term-fix pass.

The scan runs before any change: per rule and per alternative it counts the
matches and keeps a few sample sentences, so the user can spot a variant the
rules forgot. The change report lists every replaced occurrence with its
sentence before and after, for review once the run is done.
"""
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Sequence

from src.core.term_fix.models import STATUS_NEEDS_REVIEW, Occurrence
from src.core.term_fix.matcher import find_spans
from src.core.term_fix.rules import TermRule
from src.core.term_fix.snippets import build_snippets

DEFAULT_SAMPLES_PER_RULE = 5


@dataclass
class RuleScan:
    rule: TermRule
    counts: List[int]                      # per alternative, same order
    samples: List[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(self.counts)


def scan_units(
    units: Sequence[str],
    rules: Sequence[TermRule],
    samples_per_rule: int = DEFAULT_SAMPLES_PER_RULE,
    case_sensitive: bool = False,
) -> List[RuleScan]:
    """Count matches per alternative and collect sample sentences per rule,
    over every text unit of a document."""
    results = [RuleScan(rule=r, counts=[0] * len(r.alternatives)) for r in rules]
    for text in units:
        spans = find_spans(text, rules, case_sensitive=case_sensitive)
        for span in spans:
            results[span.rule_index].counts[span.alternative_index] += 1
        for snippet in build_snippets(text, spans):
            sentence = " ".join(text[snippet.start:snippet.end].split())
            for rule_index in dict.fromkeys(s.rule_index for s in snippet.spans):
                if len(results[rule_index].samples) < samples_per_rule:
                    results[rule_index].samples.append(sentence)
    return results


def scan_text(
    text: str,
    rules: Sequence[TermRule],
    samples_per_rule: int = DEFAULT_SAMPLES_PER_RULE,
    case_sensitive: bool = False,
) -> List[RuleScan]:
    """`scan_units` for a single text."""
    return scan_units([text], rules, samples_per_rule, case_sensitive)


def scan_to_dict(results: Sequence[RuleScan]) -> List[dict]:
    """JSON-friendly scan results (for the web API)."""
    return [
        {
            "rule": str(res.rule),
            "replacement": res.rule.replacement,
            "total": res.total,
            "alternatives": [
                {"text": str(alt), "count": count}
                for alt, count in zip(res.rule.alternatives, res.counts)
            ],
            "samples": list(res.samples),
        }
        for res in results
    ]


def format_scan(results: Sequence[RuleScan]) -> List[str]:
    """Human-readable scan lines, one block per rule."""
    lines: List[str] = []
    for idx, res in enumerate(results, start=1):
        lines.append(f"Rule {idx}: {res.rule}  ({res.total} match(es))")
        for alt, count in zip(res.rule.alternatives, res.counts):
            lines.append(f"    {alt}: {count}")
        for sample in res.samples:
            lines.append(f"    > {sample}")
    return lines


def _cell(text: str) -> str:
    """Make arbitrary text safe inside a Markdown table cell."""
    return " ".join(text.split()).replace("|", "\\|")


def render_report(
    source_name: str,
    rules: Sequence[TermRule],
    occurrences: Sequence[Occurrence],
) -> str:
    """Markdown change report: totals per rule, then one row per occurrence."""
    by_status: Counter = Counter(o.status for o in occurrences)
    changed: Counter = Counter(o.rule_index for o in occurrences if o.after != o.before)
    found: Counter = Counter(o.rule_index for o in occurrences)
    summary = ", ".join(f"{count} {status}" for status, count in sorted(by_status.items()))
    out: List[str] = [
        f"# Term-fix report: {source_name}",
        "",
        f"{len(occurrences)} occurrence(s) found" + (f" ({summary})." if summary else "."),
    ]
    if by_status.get(STATUS_NEEDS_REVIEW):
        out += ["", "Occurrences marked needs_review were left unchanged: fix them by hand."]
    out += [
        "",
        "| Rule | Found | Changed |",
        "| --- | --- | --- |",
    ]
    for idx, rule in enumerate(rules):
        out.append(f"| {_cell(str(rule))} | {found.get(idx, 0)} | {changed.get(idx, 0)} |")

    if occurrences:
        out += [
            "",
            "## Occurrences",
            "",
            "| Location | Found | Before | After | Status |",
            "| --- | --- | --- | --- | --- |",
        ]
        for o in occurrences:
            status = f"{o.status}: {o.reason}" if o.reason else o.status
            out.append(
                f"| {_cell(o.location)} | {_cell(o.matched)} | {_cell(o.before)} "
                f"| {_cell(o.after)} | {_cell(status)} |"
            )
    out.append("")
    return "\n".join(out)


def report_path_for(output_filepath: str) -> str:
    """`<output stem>.term-fix.md`, next to the output file."""
    stem, _ext = os.path.splitext(output_filepath)
    return f"{stem}.term-fix.md"
