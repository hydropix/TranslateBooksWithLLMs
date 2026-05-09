"""
Reconstruct a `benchmark_results/<RUN_ID>.json` from an existing submission.

Use case: re-evaluate a historical submission with a stronger / newer judge,
without re-running the translations. The script preserves the model outputs
verbatim and clears the scores so the standard `dump_for_evaluation.py` →
score → `apply_evaluations.py` → `submit` pipeline applies as usual, but
produces a new submission with a different `judge_id`.

Usage:
    python scripts/submission_to_run.py benchmark/data/submissions/<file>.json
    python scripts/submission_to_run.py <file>.json --out-id mycustom2026

The reconstructed run is written under `benchmark_results/<RUN_ID>.json`. The
script prints the chosen RUN_ID on the last line so it can be captured by a
wrapper or a Claude skill.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark.config import BenchmarkConfig  # noqa: E402
from benchmark.models import (  # noqa: E402
    BenchmarkRun,
    TranslationResult,
)


def _validate_submission(submission: dict) -> list[str]:
    """Best-effort schema check; returns a list of error strings."""
    schema_path = REPO_ROOT / "benchmark" / "schemas" / "submission.schema.json"
    if not schema_path.is_file():
        return [f"Schema file missing: {schema_path}"]

    try:
        import jsonschema
    except ImportError:
        return ["WARN: jsonschema not installed; skipping schema validation"]

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(submission), key=lambda e: list(e.absolute_path))
    return [
        f"{'/'.join(str(p) for p in err.absolute_path) or '<root>'}: {err.message}"
        for err in errors
    ]


def submission_to_run(
    submission: dict,
    *,
    run_id: str,
) -> BenchmarkRun:
    """Build a BenchmarkRun whose results carry the submission's outputs but no scores."""
    model_id = submission["model"]["id"]
    provider = submission["model"]["provider"]
    submitted_by = submission.get("submission", {}).get("submitted_by", "unknown")
    original_judge = submission.get("environment", {}).get("judge_id", "unknown")

    target_codes = sorted({r["target_lang"] for r in submission["results"]})

    now = datetime.now(timezone.utc).isoformat()
    run = BenchmarkRun(
        run_id=run_id,
        started_at=now,
        completed_at=now,
        models=[model_id],
        languages=target_codes,
        evaluator_model="skipped",
        results=[],
        status="completed",
    )

    note = (
        f"reconstructed from submission by {submitted_by} "
        f"(original judge: {original_judge})"
    )

    for r in submission["results"]:
        run.results.append(TranslationResult(
            source_text_id=r["text_id"],
            target_language=r["target_lang"],
            model=model_id,
            translated_text=r["output"],
            scores=None,
            translation_time_ms=int(r.get("translation_latency_ms") or 0),
            evaluation_time_ms=0,
            timestamp=now,
            error=None,
            n_obs=1,
            verified=(provider != "ollama"),
            contributors=[submitted_by],
        ))

    return run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path, help="Path to the submission JSON to rescore")
    parser.add_argument("--out-id", help="Custom run_id for the reconstructed run (default: random 8-hex)")
    parser.add_argument("--allow-invalid", action="store_true", help="Skip schema validation of the source submission")
    args = parser.parse_args()

    if not args.submission.is_file():
        print(f"Submission file not found: {args.submission}", file=sys.stderr)
        return 1

    try:
        with args.submission.open("r", encoding="utf-8") as f:
            submission = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"Failed to parse submission JSON: {exc}", file=sys.stderr)
        return 1

    if not args.allow_invalid:
        errors = _validate_submission(submission)
        # The first item may be a "WARN:" line meaning jsonschema isn't installed.
        if errors and not errors[0].startswith("WARN:"):
            print(f"Submission failed schema validation ({len(errors)} error(s)):", file=sys.stderr)
            for err in errors[:10]:
                print(f"  - {err}", file=sys.stderr)
            print("Use --allow-invalid to bypass.", file=sys.stderr)
            return 1
        elif errors:
            print(errors[0], file=sys.stderr)

    run_id = args.out_id or uuid.uuid4().hex[:8]
    run = submission_to_run(submission, run_id=run_id)

    config = BenchmarkConfig()
    out_path = config.paths.results_dir / f"{run.run_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(run.to_json(), encoding="utf-8")

    print(f"Reconstructed {len(run.results)} result(s) from {args.submission.name}")
    print(f"Original judge:    {submission.get('environment', {}).get('judge_id', 'unknown')}")
    print(f"Original model:    {submission['model']['id']} ({submission['model']['provider']})")
    print(f"Languages covered: {', '.join(run.languages)}")
    print(f"Run JSON:          {out_path}")
    print()
    print(f"RUN_ID={run.run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
