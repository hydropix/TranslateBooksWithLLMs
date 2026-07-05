"""
Aggregate translations + judgments from the v2 split layout into a BenchmarkRun.

Reads:
    benchmark/data/translations/<model-slug>.json    — TranslationsFile per model
    benchmark/data/judgments/<judge-id>.json         — JudgmentsFile per judge

Joins translations to judgment scores by `(model_id, text_id, target_lang)`,
validating `output_hash` matches as an integrity check. Produces a
`BenchmarkRun` the wiki generator can consume.

If a translation has no matching score from the active judge, it is included
with `scores=None` and `error="not judged"`. The aggregator logs how many
unjudged translations were found.

Single-judge model: pass `--judge-id` (or `active_judge_id` in code). Multi-
judge support (cross-judge comparison) is a future extension.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .models import (
    BenchmarkRun,
    EvaluationScores,
    JudgmentsFile,
    TranslationResult,
    TranslationsFile,
)


CLOUD_PROVIDERS = {"openai", "openrouter", "requesty", "gemini", "mistral", "deepseek", "poe", "nim"}


@dataclass
class AggregationStats:
    n_translation_files: int = 0
    n_translations: int = 0
    n_judgment_files: int = 0
    n_scores: int = 0
    n_matched: int = 0
    n_unjudged: int = 0
    n_hash_mismatches: int = 0
    n_orphan_scores: int = 0  # scores whose translation is not present


class BenchmarkAggregator:
    """Join translations/ + judgments/ → BenchmarkRun for wiki generation."""

    def __init__(
        self,
        translations_dir: Path,
        judgments_dir: Path,
        active_judge_id: Optional[str] = None,
    ):
        self.translations_dir = translations_dir
        self.judgments_dir = judgments_dir
        self.active_judge_id = active_judge_id
        self.stats = AggregationStats()

    # ─── loading ──────────────────────────────────────────────────────────

    def load_translations(self) -> dict[str, TranslationsFile]:
        """{model_id: TranslationsFile}"""
        if not self.translations_dir.is_dir():
            return {}
        out: dict[str, TranslationsFile] = {}
        for path in sorted(self.translations_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                tf = TranslationsFile.from_dict(data)
            except (OSError, json.JSONDecodeError, KeyError) as exc:
                raise RuntimeError(f"Failed to load translations {path}: {exc}") from exc
            out[tf.model_id] = tf
            self.stats.n_translations += len(tf.translations)
        self.stats.n_translation_files = len(out)
        return out

    def load_judgments(self) -> dict[str, JudgmentsFile]:
        """{judge_id: JudgmentsFile}"""
        if not self.judgments_dir.is_dir():
            return {}
        out: dict[str, JudgmentsFile] = {}
        for path in sorted(self.judgments_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                jf = JudgmentsFile.from_dict(data)
            except (OSError, json.JSONDecodeError, KeyError) as exc:
                raise RuntimeError(f"Failed to load judgments {path}: {exc}") from exc
            out[jf.judge_id] = jf
            self.stats.n_scores += len(jf.scores)
        self.stats.n_judgment_files = len(out)
        return out

    # ─── aggregation ──────────────────────────────────────────────────────

    def aggregate(self, run_id: Optional[str] = None) -> BenchmarkRun:
        translations = self.load_translations()
        judgments = self.load_judgments()

        if not translations:
            raise RuntimeError(
                f"No translation files in {self.translations_dir}. "
                "Run scripts/migrate_to_split_layout.py first?"
            )

        # Resolve active judge
        if self.active_judge_id is None:
            if len(judgments) == 1:
                self.active_judge_id = next(iter(judgments))
            elif len(judgments) == 0:
                # No scores yet — emit translations with null scores
                self.active_judge_id = "(no-judge)"
            else:
                raise RuntimeError(
                    f"Multiple judges available ({list(judgments)}); pass active_judge_id explicitly."
                )

        active_judge = judgments.get(self.active_judge_id)
        scores_by_key: dict[tuple[str, str, str], object] = (
            active_judge.by_key() if active_judge else {}
        )

        results: list[TranslationResult] = []
        models_seen: set[str] = set()
        languages_seen: set[str] = set()

        for mid, tf in translations.items():
            verified = tf.model_provider in CLOUD_PROVIDERS
            contributors = sorted({c["by"] for c in tf.contributors})
            for t in tf.translations:
                models_seen.add(mid)
                languages_seen.add(t.target_lang)
                key = (mid, t.text_id, t.target_lang)
                score = scores_by_key.get(key)
                scores_obj: Optional[EvaluationScores] = None
                error: Optional[str] = None
                if score is not None:
                    if score.output_hash != t.output_hash:
                        self.stats.n_hash_mismatches += 1
                        error = (
                            f"hash mismatch: translation={t.output_hash[:16]} "
                            f"score={score.output_hash[:16]}"
                        )
                    else:
                        scores_obj = EvaluationScores(
                            accuracy=score.accuracy,
                            fluency=score.fluency,
                            style=score.style,
                            overall=score.overall,
                            feedback=score.feedback,
                        )
                        self.stats.n_matched += 1
                else:
                    self.stats.n_unjudged += 1
                    error = "not judged"

                results.append(TranslationResult(
                    source_text_id=t.text_id,
                    target_language=t.target_lang,
                    model=mid,
                    translated_text=t.output,
                    scores=scores_obj,
                    translation_time_ms=t.translation_latency_ms,
                    evaluation_time_ms=(score.evaluation_time_ms if score else 0),
                    timestamp=t.produced_at or "",
                    error=error,
                    n_obs=1,
                    verified=verified,
                    contributors=contributors,
                ))

        # Count orphan scores (scored but no translation)
        if active_judge is not None:
            translation_keys = {
                (mid, t.text_id, t.target_lang)
                for mid, tf in translations.items()
                for t in tf.translations
            }
            for s in active_judge.scores:
                if s.key not in translation_keys:
                    self.stats.n_orphan_scores += 1

        run = BenchmarkRun(
            run_id=run_id or f"aggregated_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
            started_at=datetime.now(timezone.utc).isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat(),
            models=sorted(models_seen),
            languages=sorted(languages_seen),
            evaluator_model=self.active_judge_id,
            results=results,
            status="completed",
        )
        return run

    def write_run(self, run: BenchmarkRun, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(run.to_json(), encoding="utf-8")

    def print_stats(self) -> None:
        s = self.stats
        print(f"Aggregation:")
        print(f"  translation files: {s.n_translation_files}")
        print(f"  translations:      {s.n_translations}")
        print(f"  judgment files:    {s.n_judgment_files}")
        print(f"  scores:            {s.n_scores}")
        print(f"  matched:           {s.n_matched}")
        print(f"  unjudged:          {s.n_unjudged}")
        if s.n_hash_mismatches:
            print(f"  HASH MISMATCHES:   {s.n_hash_mismatches}  ← integrity error")
        if s.n_orphan_scores:
            print(f"  orphan scores:     {s.n_orphan_scores}  (judged but translation missing)")
