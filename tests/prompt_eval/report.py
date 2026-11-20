"""Aggregation, verdict and the Markdown report.

The report must never contain an API key, an endpoint URL or a ``.env`` value:
targets are rendered as ``provider:model`` only.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Optional

from tests.prompt_eval.arms import RenderResult
from tests.prompt_eval.metrics import FAIL, METRIC_NAMES, PASS
from tests.prompt_eval.model import Case, RenderedPrompt, RunRecord

ARMS = ("baseline", "candidate")
REGRESSION = "REGRESSION"
IMPROVEMENT = "IMPROVEMENT"
WITHIN_NOISE = "within noise"
INCONCLUSIVE = "INCONCLUSIVE"
NO_REGRESSION = "NO REGRESSION DETECTED"

MAX_REQUEST_FAILURE_RATE = 0.2
JUDGE_FIELDS = ("accuracy", "fluency", "style", "overall")

SECTION_TITLES = (
    "# Prompt regression report",
    "## Run metadata",
    "## Prompt size delta",
    "## Skipped cases",
    "## Metric results",
    "## Per-case differences",
    "## LLM judge (informative, not part of the verdict)",
)
NO_PER_CASE_DIFFERENCE = "No per-case difference."
FOOTER = ("No regression detected means none was detected at {repeat} repeats on this corpus, "
          "not that none exists.")


# ---------------------------------------------------------------------------
# Aggregation and verdict
# ---------------------------------------------------------------------------

def _empty_counts() -> dict:
    return {arm: (0, 0) for arm in ARMS}


def _add(counts: dict, arm: str, value: str) -> None:
    fails, applicable = counts[arm]
    if value == FAIL:
        counts[arm] = (fails + 1, applicable + 1)
    elif value == PASS:
        counts[arm] = (fails, applicable + 1)


def aggregate(records: Iterable[RunRecord]) -> dict:
    """(target, metric) -> {"baseline": (fails, applicable), "candidate": (...)}."""
    table: dict = defaultdict(_empty_counts)
    for record in records:
        if record.request_failed:
            continue
        for metric, value in record.metrics.items():
            _add(table[(record.target, metric)], record.arm, value)
    return dict(table)


def aggregate_by_case(records: Iterable[RunRecord]) -> dict:
    """(target, case_id, metric) -> {"baseline": (fails, applicable), ...}."""
    table: dict = defaultdict(_empty_counts)
    for record in records:
        if record.request_failed:
            continue
        for metric, value in record.metrics.items():
            _add(table[(record.target, record.case_id, metric)], record.arm, value)
    return dict(table)


def classify(baseline_fails: int, candidate_fails: int, applicable: int) -> str:
    threshold = max(2, math.ceil(0.1 * applicable))
    if candidate_fails - baseline_fails >= threshold:
        return REGRESSION
    if baseline_fails - candidate_fails >= threshold:
        return IMPROVEMENT
    return WITHIN_NOISE


def request_failure_stats(records: Iterable[RunRecord]) -> dict:
    """target -> arm -> (request failures, calls)."""
    stats: dict = defaultdict(lambda: {arm: (0, 0) for arm in ARMS})
    for record in records:
        failed, calls = stats[record.target][record.arm]
        stats[record.target][record.arm] = (failed + int(record.request_failed), calls + 1)
    return dict(stats)


def inconclusive_targets(records: Iterable[RunRecord]) -> set[str]:
    result = set()
    for target, arms in request_failure_stats(records).items():
        for failed, calls in arms.values():
            if calls and failed / calls > MAX_REQUEST_FAILURE_RATE:
                result.add(target)
    return result


def classifications(records: list[RunRecord]) -> dict:
    """(target, metric) -> classification, INCONCLUSIVE for inconclusive targets."""
    inconclusive = inconclusive_targets(records)
    result = {}
    for (target, metric), counts in aggregate(records).items():
        if target in inconclusive:
            result[(target, metric)] = INCONCLUSIVE
            continue
        (b_fails, b_app), (c_fails, c_app) = counts["baseline"], counts["candidate"]
        result[(target, metric)] = classify(b_fails, c_fails, max(b_app, c_app))
    for target in inconclusive:
        for metric in METRIC_NAMES:
            result.setdefault((target, metric), INCONCLUSIVE)
    return result


def overall_verdict(records: list[RunRecord]) -> str:
    classes = classifications(records)
    if any(c == REGRESSION for c in classes.values()):
        return REGRESSION
    if inconclusive_targets(records):
        return INCONCLUSIVE
    return NO_REGRESSION


# ---------------------------------------------------------------------------
# Prompt sizes
# ---------------------------------------------------------------------------

def _size(text: str) -> dict:
    return {"chars": len(text), "words": len(text.split())}


def compute_prompt_sizes(
    cases: list[Case],
    baseline: dict[str, RenderResult],
    candidate: dict[str, RenderResult],
    case_ids: Iterable[str],
) -> list[dict]:
    wanted = set(case_ids)
    sizes = []
    for case in cases:
        if case.id not in wanted:
            continue
        b, c = baseline.get(case.id), candidate.get(case.id)
        if not isinstance(b, RenderedPrompt) or not isinstance(c, RenderedPrompt):
            continue
        sizes.append({
            "case_id": case.id,
            "builder": case.builder,
            "baseline": {"system": _size(b.system), "user": _size(b.user)},
            "candidate": {"system": _size(c.system), "user": _size(c.user)},
        })
    return sizes


def _delta(before: float, after: float) -> str:
    diff = after - before
    if isinstance(diff, float):
        diff = round(diff)
    return f"{before:.0f} → {after:.0f} ({diff:+d})"


# ---------------------------------------------------------------------------
# Markdown sections
# ---------------------------------------------------------------------------

def _short(sha: Optional[str]) -> str:
    return (sha or "?")[:8]


def section_metadata(meta: dict, records: Optional[list[RunRecord]] = None) -> str:
    lines = [SECTION_TITLES[1], ""]
    lines.append(f"- Baseline: `{_short(meta.get('baseline_sha'))}`"
                 + (f" (`{meta['baseline_ref']}`)" if meta.get("baseline_ref") else ""))
    dirty = " (dirty)" if meta.get("dirty") else ""
    lines.append(f"- Candidate: working tree @ `{_short(meta.get('head_sha'))}`{dirty}")
    if meta.get("targets"):
        lines.append("- Targets: " + ", ".join(f"`{t}`" for t in meta["targets"]))
    if meta.get("repeat") is not None:
        lines.append(f"- Repeat: {meta['repeat']}")
    if meta.get("date"):
        lines.append(f"- Date: {meta['date']}")
    if records is not None:
        failures = sum(1 for r in records if r.request_failed)
        lines.append(f"- Total calls: {len(records)}")
        lines.append(f"- Request failures: {failures}")
    return "\n".join(lines)


def section_prompt_sizes(prompt_sizes: list[dict]) -> str:
    lines = [SECTION_TITLES[2], ""]
    if not prompt_sizes:
        lines.append("No changed case.")
        return "\n".join(lines)

    lines += ["Per builder (mean over changed cases, characters / words):", "",
              "| builder | cases | system chars | system words | user chars | user words |",
              "|---|---|---|---|---|---|"]
    by_builder: dict = defaultdict(list)
    for entry in prompt_sizes:
        by_builder[entry["builder"]].append(entry)
    for builder, entries in by_builder.items():
        cells = []
        for part in ("system", "user"):
            for unit in ("chars", "words"):
                before = sum(e["baseline"][part][unit] for e in entries) / len(entries)
                after = sum(e["candidate"][part][unit] for e in entries) / len(entries)
                cells.append(_delta(before, after))
        lines.append(f"| {builder} | {len(entries)} | " + " | ".join(cells) + " |")

    lines += ["", "Per changed case:", "",
              "| case | system chars | system words | user chars | user words |",
              "|---|---|---|---|---|"]
    for entry in prompt_sizes:
        cells = [_delta(entry["baseline"][part][unit], entry["candidate"][part][unit])
                 for part in ("system", "user") for unit in ("chars", "words")]
        lines.append(f"| `{entry['case_id']}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def section_skipped(skipped: dict) -> str:
    lines = [SECTION_TITLES[3], ""]
    changed = skipped.get("changed")
    if changed is not None:
        lines.append(f"- Changed: {len(changed)}"
                     + (" - " + ", ".join(f"`{i}`" for i in changed) if changed else ""))
    unchanged = skipped.get("unchanged") or []
    included = skipped.get("include_unchanged")
    label = "Unchanged (run anyway, A/A noise floor)" if included else "Unchanged (not run)"
    lines.append(f"- {label}: {len(unchanged)}"
                 + (" - " + ", ".join(f"`{i}`" for i in unchanged) if unchanged else ""))
    errors = skipped.get("render_errors") or {}
    lines.append(f"- Render errors: {len(errors)}")
    for case_id, message in errors.items():
        lines.append(f"  - `{case_id}`: {message}")
    dropped = skipped.get("dropped_kwargs") or {}
    lines.append(f"- Dropped kwargs (signature drift): {len(dropped)}")
    for case_id, arms in dropped.items():
        parts = [f"{arm}: {', '.join(names)}" for arm, names in arms.items() if names]
        lines.append(f"  - `{case_id}`: " + "; ".join(parts))
    return "\n".join(lines)


def _fmt(counts: tuple[int, int]) -> str:
    fails, applicable = counts
    return f"{fails}/{applicable}"


def section_metrics(records: list[RunRecord]) -> str:
    lines = [SECTION_TITLES[4], ""]
    table = aggregate(records)
    classes = classifications(records)
    stats = request_failure_stats(records)
    targets = list(dict.fromkeys(r.target for r in records))
    if not targets:
        lines.append("No call was made.")
        return "\n".join(lines)
    for target in targets:
        b_fail, b_calls = stats[target]["baseline"]
        c_fail, c_calls = stats[target]["candidate"]
        lines += [f"### `{target}`", "",
                  f"Request failures: baseline {b_fail}/{b_calls}, candidate {c_fail}/{c_calls}.", "",
                  "| metric | baseline fail/applicable | candidate fail/applicable | classification |",
                  "|---|---|---|---|"]
        for metric in METRIC_NAMES:
            counts = table.get((target, metric), _empty_counts())
            if counts["baseline"][1] == 0 and counts["candidate"][1] == 0 \
                    and classes.get((target, metric)) != INCONCLUSIVE:
                continue
            classification = classes.get((target, metric), WITHIN_NOISE)
            if classification in (REGRESSION, IMPROVEMENT, INCONCLUSIVE):
                classification = f"**{classification}**"
            lines.append(f"| {metric} | {_fmt(counts['baseline'])} | {_fmt(counts['candidate'])} "
                         f"| {classification} |")
        lines.append("")
    return "\n".join(lines).rstrip()


def section_per_case(records: list[RunRecord]) -> str:
    lines = [SECTION_TITLES[5], ""]
    rows = []
    for (target, case_id, metric), counts in aggregate_by_case(records).items():
        if counts["baseline"][0] != counts["candidate"][0]:
            rows.append(f"| `{target}` | `{case_id}` | {metric} | {_fmt(counts['baseline'])} "
                        f"| {_fmt(counts['candidate'])} |")
    if not rows:
        lines.append(NO_PER_CASE_DIFFERENCE)
        return "\n".join(lines)
    lines += ["| target | case | metric | baseline fail/applicable | candidate fail/applicable |",
              "|---|---|---|---|---|"]
    lines += sorted(rows)
    return "\n".join(lines)


def section_judge(records: list[RunRecord], judge_failures: int = 0) -> Optional[str]:
    judged = [r for r in records if r.judge]
    if not judged:
        return None
    lines = [SECTION_TITLES[6], "",
             "| target | arm | judged | " + " | ".join(JUDGE_FIELDS) + " |",
             "|---|---|---|" + "---|" * len(JUDGE_FIELDS)]
    groups: dict = defaultdict(list)
    for record in judged:
        groups[(record.target, record.arm)].append(record.judge)
    for (target, arm), scores in sorted(groups.items()):
        means = []
        for name in JUDGE_FIELDS:
            values = [s[name] for s in scores if isinstance(s.get(name), (int, float))]
            means.append(f"{sum(values) / len(values):.2f}" if values else "n/a")
        lines.append(f"| `{target}` | {arm} | {len(scores)} | " + " | ".join(means) + " |")
    lines += ["", f"Judge failures: {judge_failures}."]
    return "\n".join(lines)


def render_markdown(meta: dict, records: list[RunRecord], prompt_sizes: list[dict], skipped: dict) -> str:
    verdict = overall_verdict(records)
    parts = [
        f"{SECTION_TITLES[0]}\n\n**Verdict: {verdict}**",
        section_metadata(meta, records),
        section_prompt_sizes(prompt_sizes),
        section_skipped(skipped),
        section_metrics(records),
        section_per_case(records),
    ]
    judge = section_judge(records, meta.get("judge_failures", 0))
    if judge:
        parts.append(judge)
    parts.append("---\n\n" + FOOTER.format(repeat=meta.get("repeat", "?")))
    return "\n\n".join(parts) + "\n"


def render_diff_markdown(meta: dict, prompt_sizes: list[dict], skipped: dict) -> str:
    """Sections 2 to 4 only, for the ``diff`` dry run."""
    return "\n\n".join([
        section_metadata(meta),
        section_prompt_sizes(prompt_sizes),
        section_skipped(skipped),
    ]) + "\n"
