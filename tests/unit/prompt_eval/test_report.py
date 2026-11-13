"""Aggregation, verdict and Markdown rendering checks on hand-made records."""

import pytest

from tests.prompt_eval.model import RunRecord
from tests.prompt_eval.report import (
    IMPROVEMENT,
    INCONCLUSIVE,
    NO_PER_CASE_DIFFERENCE,
    NO_REGRESSION,
    REGRESSION,
    SECTION_TITLES,
    WITHIN_NOISE,
    aggregate,
    classify,
    inconclusive_targets,
    overall_verdict,
    render_diff_markdown,
    render_markdown,
)

T = "ollama:small"


def rec(arm, *, case="C1", target=T, fmt="pass", raw="x", judge=None, **metrics):
    values = {"format": fmt, **metrics}
    return RunRecord(case_id=case, arm=arm, target=target, repeat=0, raw=raw,
                     extracted="x" if raw else None, prompt_tokens=None, completion_tokens=None,
                     elapsed_s=0.1, metrics=values if raw else {k: "n/a" for k in values},
                     judge=judge)


def pair(n, baseline_fails, candidate_fails, metric="paragraphs", target=T):
    records = []
    for i in range(n):
        records.append(rec("baseline", target=target, case=f"C{i}",
                           **{metric: "fail" if i < baseline_fails else "pass"}))
        records.append(rec("candidate", target=target, case=f"C{i}",
                           **{metric: "fail" if i < candidate_fails else "pass"}))
    return records


@pytest.mark.parametrize("b, c, applicable, expected", [
    (0, 2, 10, REGRESSION),        # threshold 2 met exactly
    (0, 1, 10, WITHIN_NOISE),      # one below
    (2, 0, 10, IMPROVEMENT),
    (1, 0, 10, WITHIN_NOISE),
    (0, 5, 50, REGRESSION),        # threshold ceil(5.0) = 5 met exactly
    (0, 4, 50, WITHIN_NOISE),
    (0, 6, 51, REGRESSION),        # threshold ceil(5.1) = 6
    (0, 5, 51, WITHIN_NOISE),
    (10, 4, 51, IMPROVEMENT),
])
def test_classify_boundaries(b, c, applicable, expected):
    assert classify(b, c, applicable) == expected


def test_aggregate_counts_applicable_without_na_or_request_failures():
    records = [
        rec("baseline", paragraphs="fail"),
        rec("baseline", paragraphs="pass"),
        rec("baseline", paragraphs="n/a"),
        rec("baseline", raw=None, paragraphs="fail"),
        rec("candidate", paragraphs="pass"),
    ]
    table = aggregate(records)
    assert table[(T, "paragraphs")] == {"baseline": (1, 2), "candidate": (0, 1)}


def _with_failures(calls, failures):
    records = []
    for i in range(calls):
        failed = i < failures
        records.append(rec("baseline", case=f"C{i}", raw=None if failed else "x"))
        records.append(rec("candidate", case=f"C{i}"))
    return records


def test_inconclusive_at_twenty_percent_is_not_triggered():
    assert inconclusive_targets(_with_failures(10, 2)) == set()


def test_inconclusive_just_above_twenty_percent():
    records = _with_failures(10, 3)
    assert inconclusive_targets(records) == {T}
    assert overall_verdict(records) == INCONCLUSIVE


def test_verdict_precedence():
    assert overall_verdict(pair(10, 0, 0)) == NO_REGRESSION
    assert overall_verdict(pair(10, 0, 3)) == REGRESSION
    # REGRESSION on one target wins over INCONCLUSIVE on another.
    mixed = pair(10, 0, 3) + _with_failures(10, 5)
    for r in mixed[20:]:
        r.target = "ollama:other"
    assert overall_verdict(mixed) == REGRESSION


def _meta():
    return {"baseline_sha": "a" * 40, "baseline_ref": "HEAD", "head_sha": "b" * 40, "dirty": True,
            "targets": [T], "repeat": 3, "date": "2026-11-07 10:00:00"}


SIZES = [{"case_id": "C0", "builder": "refine",
          "baseline": {"system": {"chars": 100, "words": 20}, "user": {"chars": 50, "words": 10}},
          "candidate": {"system": {"chars": 130, "words": 25}, "user": {"chars": 50, "words": 10}}}]
SKIPPED = {"changed": ["C0"], "unchanged": ["U1"], "render_errors": {"E1": "TypeError: x"},
           "dropped_kwargs": {"C0": {"baseline": ["glossary_block"]}}}


def test_markdown_contains_every_section_in_order():
    records = pair(10, 0, 3)
    records[0].judge = {"accuracy": 8, "fluency": 7, "style": 6, "overall": 7, "feedback": "ok"}
    md = render_markdown(_meta(), records, SIZES, SKIPPED)
    positions = [md.index(title + "\n") for title in SECTION_TITLES]
    assert positions == sorted(positions)
    assert "**Verdict: REGRESSION**" in md
    assert "100 → 130 (+30)" in md
    assert "`E1`: TypeError: x" in md and "glossary_block" in md
    assert "| paragraphs | 0/10 | 3/10 | **REGRESSION** |" in md
    assert "No regression detected means none was detected at 3 repeats" in md


def test_markdown_without_differences_or_judge():
    md = render_markdown(_meta(), pair(5, 1, 1), SIZES, SKIPPED)
    assert NO_PER_CASE_DIFFERENCE in md
    assert SECTION_TITLES[6] not in md
    assert "**Verdict: NO REGRESSION DETECTED**" in md


def test_diff_markdown_has_sections_two_to_four_only():
    md = render_diff_markdown({"baseline_sha": "a" * 40, "head_sha": "b" * 40}, SIZES, SKIPPED)
    assert md.startswith(SECTION_TITLES[1])
    assert SECTION_TITLES[2] in md and SECTION_TITLES[3] in md
    assert SECTION_TITLES[4] not in md and "Verdict" not in md
