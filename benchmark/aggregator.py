"""
Aggregate community-submitted benchmark results.

Reads all `*.json` files from `benchmark/data/submissions/` (after they've been
validated against `submission.schema.json`) and produces a synthetic
`BenchmarkRun` whose results carry:

- The **median** of `accuracy/fluency/style/overall` when several contributors
  have tested the same `(model, text_id, target_lang)` triple.
- `n_obs`: how many observations went into that median.
- `verified`: True when all observations come from cloud providers (replayable),
  False when at least one observation is `self_reported` (local model).
- `contributors`: list of distinct GitHub identities that submitted that triple.

The wiki generator can then consume this `BenchmarkRun` exactly like a real one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Iterable, Optional

from .models import (
    BenchmarkRun,
    EvaluationScores,
    Submission,
    TranslationResult,
)


# Cloud providers whose results are replayable in CI. Anything outside this set
# is treated as "self-reported".
CLOUD_PROVIDERS = {"openai", "openrouter", "gemini", "mistral", "deepseek", "poe", "nim"}


@dataclass
class AggregationStats:
    n_submissions: int = 0
    n_results_in: int = 0
    n_results_out: int = 0
    n_conflicts: int = 0  # how many keys had >1 observation


class SubmissionAggregator:
    """Merge per-contributor submissions into a single BenchmarkRun."""

    def __init__(self, submissions_dir: Path):
        self.submissions_dir = submissions_dir
        self.stats = AggregationStats()

    def discover(self) -> list[Path]:
        if not self.submissions_dir.is_dir():
            return []
        return sorted(p for p in self.submissions_dir.glob("*.json") if p.is_file())

    def load(self) -> list[Submission]:
        submissions: list[Submission] = []
        for path in self.discover():
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                submissions.append(Submission.from_dict(data))
            except (OSError, json.JSONDecodeError, KeyError) as exc:
                raise RuntimeError(f"Failed to load submission {path}: {exc}") from exc
        self.stats.n_submissions = len(submissions)
        return submissions

    def aggregate(
        self,
        submissions: Optional[Iterable[Submission]] = None,
        run_id: Optional[str] = None,
    ) -> BenchmarkRun:
        if submissions is None:
            submissions = self.load()
        submissions = list(submissions)

        # Bucket observations by (model_id, text_id, target_lang).
        buckets: dict[tuple[str, str, str], list[dict]] = {}
        models: set[str] = set()
        languages: set[str] = set()
        judges: set[str] = set()

        for sub in submissions:
            models.add(sub.model_id)
            judges.add(sub.judge_id)
            verified = sub.model_provider in CLOUD_PROVIDERS

            for r in sub.results:
                key = (sub.model_id, r.text_id, r.target_lang)
                self.stats.n_results_in += 1
                languages.add(r.target_lang)
                buckets.setdefault(key, []).append({
                    "submission": sub,
                    "result": r,
                    "verified": verified,
                })

        # Reduce each bucket to a single TranslationResult with median scores.
        merged: list[TranslationResult] = []

        for key, observations in buckets.items():
            model_id, text_id, target_lang = key

            if len(observations) > 1:
                self.stats.n_conflicts += 1

            accs = [o["result"].scores.accuracy for o in observations]
            flus = [o["result"].scores.fluency for o in observations]
            stys = [o["result"].scores.style for o in observations]
            ovrs = [o["result"].scores.overall for o in observations]

            # Pick the output text from the observation closest to the median
            # overall score, so the example shown is representative.
            target_overall = median(ovrs)
            representative = min(
                observations,
                key=lambda o: abs(o["result"].scores.overall - target_overall),
            )
            rep_result = representative["result"]
            rep_sub = representative["submission"]

            scores = EvaluationScores(
                accuracy=median(accs),
                fluency=median(flus),
                style=median(stys),
                overall=target_overall,
                feedback=rep_result.scores.feedback,
            )

            merged.append(TranslationResult(
                source_text_id=text_id,
                target_language=target_lang,
                model=model_id,
                translated_text=rep_result.output,
                scores=scores,
                translation_time_ms=int(rep_result.translation_latency_ms or 0),
                evaluation_time_ms=0,
                timestamp=rep_sub.submitted_at or datetime.now(timezone.utc).isoformat(),
                error=None,
                n_obs=len(observations),
                verified=all(o["verified"] for o in observations),
                contributors=sorted({o["submission"].submitted_by for o in observations}),
            ))

        self.stats.n_results_out = len(merged)

        # Pick the most-used judge as the run's evaluator label.
        judge_label = ", ".join(sorted(judges)) if judges else ""

        run = BenchmarkRun(
            run_id=run_id or "aggregated",
            started_at=datetime.now(timezone.utc).isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat(),
            models=sorted(models),
            languages=sorted(languages),
            evaluator_model=judge_label,
            results=merged,
            status="completed",
        )
        return run

    def write_run(self, run: BenchmarkRun, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            f.write(run.to_json())
