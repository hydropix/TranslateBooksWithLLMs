"""
Apply manual evaluations (JSON array of {eval_id, scores}) back to a run.

Reads the JSON reply produced by an evaluator (Claude Code session, manual
review, anything else) and patches the corresponding `TranslationResult`
entries in the run JSON. Idempotent: re-applying the same JSON is a no-op.

Usage:
    python scripts/apply_evaluations.py <run_id> <evaluations.json> [--judge-id NAME] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark.config import BenchmarkConfig  # noqa: E402
from benchmark.models import EvaluationScores  # noqa: E402
from benchmark.results.storage import ResultsStorage  # noqa: E402


def make_eval_id(text_id: str, target_lang: str, model: str) -> str:
    raw = f"{text_id}|{target_lang}|{model}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:10]


def _load_evaluations(path: Path) -> list[dict]:
    """Accept either a top-level JSON array, or a code-fenced JSON block, or a list field."""
    raw = path.read_text(encoding="utf-8").strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        first_nl = raw.find("\n")
        if first_nl != -1:
            raw = raw[first_nl + 1:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    data = json.loads(raw)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("evaluations", "results", "scores"):
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError(f"Could not extract a JSON array from {path}")


def _validate_scores(scores: dict) -> tuple[bool, str]:
    required = {"accuracy", "fluency", "style", "overall"}
    missing = required - scores.keys()
    if missing:
        return False, f"missing keys: {sorted(missing)}"
    for key in required:
        v = scores[key]
        if not isinstance(v, (int, float)) or not (1.0 <= float(v) <= 10.0):
            return False, f"{key} must be in [1, 10], got {v}"
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="Benchmark run ID to update")
    parser.add_argument("evaluations", type=Path, help="Path to the JSON file with evaluator scores")
    parser.add_argument("--judge-id", help="Judge identity to record in run.evaluator_model")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change, don't write")
    parser.add_argument("--strict", action="store_true", help="Fail on unmatched eval_ids")
    args = parser.parse_args()

    config = BenchmarkConfig()
    storage = ResultsStorage(config)
    run = storage.load_run(args.run_id)
    if run is None:
        print(f"Run {args.run_id} not found.", file=sys.stderr)
        return 1

    if not args.evaluations.is_file():
        print(f"Evaluations file not found: {args.evaluations}", file=sys.stderr)
        return 1

    try:
        evals = _load_evaluations(args.evaluations)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"Failed to parse {args.evaluations}: {exc}", file=sys.stderr)
        return 1

    # Index existing results by eval_id for fast patching.
    by_id: dict[str, int] = {}
    for idx, r in enumerate(run.results):
        eid = make_eval_id(r.source_text_id, r.target_language, r.model)
        by_id[eid] = idx

    applied = 0
    skipped_invalid = 0
    skipped_missing = 0

    for entry in evals:
        eid = entry.get("eval_id")
        scores = entry.get("scores")
        if not eid or not isinstance(scores, dict):
            skipped_invalid += 1
            print(f"  [skip] malformed entry: {entry}", file=sys.stderr)
            continue

        ok, why = _validate_scores(scores)
        if not ok:
            skipped_invalid += 1
            print(f"  [skip] {eid}: {why}", file=sys.stderr)
            continue

        if eid not in by_id:
            skipped_missing += 1
            msg = f"  [skip] no result with eval_id={eid}"
            if args.strict:
                print(msg + " (strict mode → failing)", file=sys.stderr)
                return 1
            print(msg, file=sys.stderr)
            continue

        result = run.results[by_id[eid]]
        result.scores = EvaluationScores(
            accuracy=float(scores["accuracy"]),
            fluency=float(scores["fluency"]),
            style=float(scores["style"]),
            overall=float(scores["overall"]),
            feedback=str(scores.get("feedback") or "").strip() or None,
        )
        applied += 1

    if args.judge_id:
        run.evaluator_model = args.judge_id

    print(f"Applied: {applied}")
    if skipped_invalid:
        print(f"Skipped (invalid scores): {skipped_invalid}")
    if skipped_missing:
        print(f"Skipped (eval_id not found): {skipped_missing}")

    if args.dry_run:
        print("[dry-run] not writing changes.")
        return 0

    storage.save_run(run)
    print(f"Run {run.run_id} saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
