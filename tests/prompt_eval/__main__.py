"""CLI: ``python -m tests.prompt_eval {run,diff}``. See README.md."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

EXIT_OK, EXIT_REGRESSION, EXIT_INCONCLUSIVE, EXIT_SETUP = 0, 1, 2, 3
RUNS_DIR = Path(__file__).resolve().parent / "runs"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tests.prompt_eval",
        description="Before/after regression check for LLM prompt changes.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--baseline", default=None,
                       help="Git ref of the baseline prompts (default: merge-base of HEAD and main)")
        p.add_argument("--builders", default="translate,refine",
                       help="Comma-separated builders to include (default: translate,refine)")
        p.add_argument("--cases", default=None, help="Regex matched against case ids")

    run = sub.add_parser("run", help="Run both arms against real models and print the report")
    add_common(run)
    run.add_argument("--target", action="append", default=None, metavar="PROVIDER:MODEL",
                     help="Model to test, repeatable (default: ollama:<DEFAULT_MODEL>)")
    run.add_argument("--repeat", type=int, default=3, help="Repeats per case and arm (default: 3)")
    run.add_argument("--include-unchanged", action="store_true",
                     help="Also run cases whose prompts are identical in both arms (A/A noise floor)")
    run.add_argument("--judge", choices=("openrouter", "poe"), default=None,
                     help="Add informative LLM-judge scores (never part of the verdict)")
    run.add_argument("--out", default=None, help="Output folder (default: tests/prompt_eval/runs/<timestamp>/)")

    diff = sub.add_parser("diff", help="Dry run: which cases changed and prompt size delta, no LLM call")
    add_common(diff)
    return parser


def _setup_error(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return EXIT_SETUP


def _render(args):
    """Load the corpus and render both arms. Returns (cases, sha, baseline, candidate)."""
    from tests.prompt_eval.arms import render_both
    from tests.prompt_eval.corpus import load_corpus

    builders = [b.strip() for b in args.builders.split(",") if b.strip()]
    cases = load_corpus(builders=builders, case_filter=args.cases)
    if not cases:
        raise ValueError("no corpus case matches --builders/--cases")
    sha, baseline, candidate = render_both(cases, args.baseline)
    return cases, sha, baseline, candidate


def _skipped(cases, selection, baseline, candidate, include_unchanged: bool) -> dict:
    from tests.prompt_eval.model import RenderedPrompt

    dropped = {}
    for case in cases:
        arms = {arm: list(r.dropped_kwargs) for arm, r in
                (("baseline", baseline.get(case.id)), ("candidate", candidate.get(case.id)))
                if isinstance(r, RenderedPrompt) and r.dropped_kwargs}
        if arms:
            dropped[case.id] = arms
    unchanged = set(selection.unchanged)
    return {
        "changed": [c.id for c in selection.to_run if c.id not in unchanged],
        "unchanged": selection.unchanged,
        "include_unchanged": include_unchanged,
        "render_errors": selection.render_errors,
        "dropped_kwargs": dropped,
    }


def _meta(args, sha: str) -> dict:
    from tests.prompt_eval.arms import head_sha, working_tree_dirty
    return {
        "baseline_sha": sha,
        "baseline_ref": args.baseline or "merge-base HEAD main",
        "head_sha": head_sha(),
        "dirty": working_tree_dirty(),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _changed_ids(selection) -> list[str]:
    unchanged = set(selection.unchanged)
    return [c.id for c in selection.to_run if c.id not in unchanged]


def cmd_diff(args) -> int:
    from tests.prompt_eval.report import compute_prompt_sizes, render_diff_markdown
    from tests.prompt_eval.runner import select_cases

    try:
        cases, sha, baseline, candidate = _render(args)
    except (RuntimeError, ValueError) as exc:
        return _setup_error(str(exc))
    selection = select_cases(cases, baseline, candidate, include_unchanged=False)
    sizes = compute_prompt_sizes(cases, baseline, candidate, _changed_ids(selection))
    print(render_diff_markdown(_meta(args, sha), sizes, _skipped(cases, selection, baseline, candidate, False)))
    return EXIT_OK


def _write_prompts(out_dir: Path, cases, baseline, candidate, case_ids) -> None:
    prompts_dir = out_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    for case_id in case_ids:
        for arm, rendered in (("baseline", baseline[case_id]), ("candidate", candidate[case_id])):
            (prompts_dir / f"{case_id}.{arm}.txt").write_text(
                f"{rendered.system}\n===== USER =====\n{rendered.user}", encoding="utf-8")


def cmd_run(args, provider_factory=None) -> int:
    from src import config
    from tests.prompt_eval.report import (
        INCONCLUSIVE, REGRESSION, compute_prompt_sizes, overall_verdict, render_markdown,
    )
    from tests.prompt_eval.runner import parse_target, run_matrix, select_cases

    if args.repeat < 1:
        return _setup_error("--repeat must be at least 1")
    try:
        targets = [parse_target(t) for t in (args.target or [f"ollama:{config.DEFAULT_MODEL}"])]
    except ValueError as exc:
        return _setup_error(str(exc))

    try:
        cases, sha, baseline, candidate = _render(args)
    except (RuntimeError, ValueError) as exc:
        return _setup_error(str(exc))

    selection = select_cases(cases, baseline, candidate, args.include_unchanged)
    if not selection.to_run:
        return _setup_error("no changed case to run (use --include-unchanged for an A/A run)")

    # Fail fast on missing credentials, before any call.
    factory = provider_factory
    if factory is None:
        from src.core.llm.factory import create_llm_provider as factory
    try:
        for target in targets:
            probe = factory(target.provider, model=target.model)
            close = getattr(probe, "close", None)
            if close is not None:
                try:
                    asyncio.run(close())
                except Exception:
                    pass
    except ValueError as exc:
        return _setup_error(str(exc))

    out_dir = Path(args.out) if args.out else RUNS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    changed = _changed_ids(selection)
    _write_prompts(out_dir, cases, baseline, candidate, [c.id for c in selection.to_run])

    records = asyncio.run(run_matrix(
        cases, baseline, candidate, targets, args.repeat, args.include_unchanged, out_dir,
        provider_factory=factory,
    ))

    meta = _meta(args, sha)
    meta.update(targets=[str(t) for t in targets], repeat=args.repeat)

    if args.judge:
        from tests.prompt_eval.judge import count_judge_failures, judge_records
        try:
            asyncio.run(judge_records(records, cases, args.judge))
        except Exception as exc:
            print(f"warning: judge unavailable ({type(exc).__name__}); report has no judge scores",
                  file=sys.stderr)
        meta["judge_failures"] = count_judge_failures(records, cases)
        with (out_dir / "records.jsonl").open("w", encoding="utf-8") as sink:
            for record in records:
                sink.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    sizes = compute_prompt_sizes(cases, baseline, candidate, changed)
    skipped = _skipped(cases, selection, baseline, candidate, args.include_unchanged)
    report = render_markdown(meta, records, sizes, skipped)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Run saved to {out_dir}", file=sys.stderr)

    verdict = overall_verdict(records)
    if verdict == REGRESSION:
        return EXIT_REGRESSION
    if verdict == INCONCLUSIVE:
        return EXIT_INCONCLUSIVE
    return EXIT_OK


def main(argv=None, provider_factory=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    args = _parser().parse_args(argv)
    if args.command == "diff":
        return cmd_diff(args)
    return cmd_run(args, provider_factory=provider_factory)


if __name__ == "__main__":
    sys.exit(main())
