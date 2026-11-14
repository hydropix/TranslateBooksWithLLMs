"""
JSON-based storage for benchmark results.

Provides:
- Save/load benchmark runs to/from JSON files
- List available runs
- Resume interrupted runs
- Incremental saving (after each translation)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import BenchmarkConfig
from ..models import BenchmarkRun, TranslationResult


class ResultsStorage:
    """Handles persistence of benchmark results to JSON files."""

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """
        Initialize storage with configuration.

        Args:
            config: Benchmark configuration. If None, uses default.
        """
        self.config = config or BenchmarkConfig()
        self.results_dir = self.config.paths.results_dir

    def _ensure_results_dir(self) -> None:
        """Create results directory if it doesn't exist."""
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _get_run_path(self, run_id: str) -> Path:
        """Get the file path for a benchmark run."""
        return self.results_dir / f"{run_id}.json"

    def _generate_run_id(self) -> str:
        """Generate a unique run ID based on timestamp."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def create_run(
        self,
        models: list[str],
        languages: list[str],
        evaluator_model: str,
        run_id: Optional[str] = None,
    ) -> BenchmarkRun:
        """
        Create a new benchmark run.

        Args:
            models: List of Ollama model names to test
            languages: List of language codes to translate to
            evaluator_model: OpenRouter model for evaluation
            run_id: Optional custom run ID. If None, auto-generated.

        Returns:
            New BenchmarkRun instance (not yet saved)
        """
        if run_id is None:
            run_id = self._generate_run_id()

        run = BenchmarkRun(
            run_id=run_id,
            started_at=datetime.now().isoformat(),
            models=models,
            languages=languages,
            evaluator_model=evaluator_model,
            status="running",
        )

        return run

    def save_run(self, run: BenchmarkRun) -> Path:
        """
        Save a benchmark run to JSON file.

        Args:
            run: The benchmark run to save

        Returns:
            Path to the saved file
        """
        self._ensure_results_dir()
        path = self._get_run_path(run.run_id)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(run.to_dict(), f, indent=2, ensure_ascii=False)

        return path

    def load_run(self, run_id: str) -> Optional[BenchmarkRun]:
        """
        Load a benchmark run from JSON file.

        Args:
            run_id: The run ID to load

        Returns:
            BenchmarkRun instance or None if not found
        """
        path = self._get_run_path(run_id)

        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return BenchmarkRun.from_dict(data)

    def add_result(self, run: BenchmarkRun, result: TranslationResult) -> None:
        """
        Add a translation result to a run and save immediately.

        This provides incremental saving for resumability.

        Args:
            run: The benchmark run to update
            result: The translation result to add
        """
        run.add_result(result)
        self.save_run(run)

    def complete_run(
        self, run: BenchmarkRun, error: Optional[str] = None
    ) -> None:
        """
        Mark a run as completed and save.

        Args:
            run: The benchmark run to complete
            error: Optional error message if run failed
        """
        run.completed_at = datetime.now().isoformat()
        run.status = "failed" if error else "completed"
        run.error = error
        self.save_run(run)

    def list_runs(self) -> list[dict]:
        """
        List all available benchmark runs.

        Returns:
            List of run summaries (id, status, date, progress)
        """
        self._ensure_results_dir()
        runs = []

        for path in sorted(self.results_dir.glob("*.json"), reverse=True):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                runs.append({
                    "run_id": data.get("run_id", path.stem),
                    "status": data.get("status", "unknown"),
                    "started_at": data.get("started_at", ""),
                    "completed_at": data.get("completed_at"),
                    "models": data.get("models", []),
                    "languages": data.get("languages", []),
                    "total_results": len(data.get("results", [])),
                    "evaluator_model": data.get("evaluator_model", ""),
                })
            except (json.JSONDecodeError, OSError):
                continue

        return runs

    def get_latest_run(self) -> Optional[BenchmarkRun]:
        """
        Get the most recent benchmark run.

        Returns:
            Most recent BenchmarkRun or None if no runs exist
        """
        runs = self.list_runs()
        if not runs:
            return None
        return self.load_run(runs[0]["run_id"])

    def get_resumable_run(
        self,
        models: list[str],
        languages: list[str],
    ) -> Optional[BenchmarkRun]:
        """
        Find an incomplete run that matches the given parameters.

        Useful for resuming interrupted benchmarks.

        Args:
            models: List of models that should match
            languages: List of languages that should match

        Returns:
            Matching incomplete BenchmarkRun or None
        """
        for run_info in self.list_runs():
            if run_info["status"] != "running":
                continue

            # Check if parameters match
            if set(run_info["models"]) == set(models) and \
               set(run_info["languages"]) == set(languages):
                return self.load_run(run_info["run_id"])

        return None

    def get_completed_translations(self, run: BenchmarkRun) -> set[tuple[str, str, str]]:
        """
        Get set of completed translation keys for a run.

        Useful for skipping already-done translations when resuming.

        Args:
            run: The benchmark run to check

        Returns:
            Set of (source_text_id, target_language, model) tuples
        """
        return {
            (r.source_text_id, r.target_language, r.model)
            for r in run.results
        }

    def delete_run(self, run_id: str) -> bool:
        """
        Delete a benchmark run.

        Args:
            run_id: The run ID to delete

        Returns:
            True if deleted, False if not found
        """
        path = self._get_run_path(run_id)

        if not path.exists():
            return False

        path.unlink()
        return True

    def merge_runs(self, run_ids: list[str], new_run_id: Optional[str] = None) -> Optional[BenchmarkRun]:
        """
        Merge multiple runs into a single run.

        Useful for combining partial runs or runs with different models.

        Args:
            run_ids: List of run IDs to merge
            new_run_id: Optional ID for merged run

        Returns:
            Merged BenchmarkRun or None if no valid runs
        """
        all_results = []
        all_models = set()
        all_languages = set()
        earliest_start = None
        latest_end = None
        evaluator_model = None

        for run_id in run_ids:
            run = self.load_run(run_id)
            if run is None:
                continue

            all_results.extend(run.results)
            all_models.update(run.models)
            all_languages.update(run.languages)

            if earliest_start is None or run.started_at < earliest_start:
                earliest_start = run.started_at

            if run.completed_at:
                if latest_end is None or run.completed_at > latest_end:
                    latest_end = run.completed_at

            if evaluator_model is None:
                evaluator_model = run.evaluator_model

        if not all_results:
            return None

        merged = BenchmarkRun(
            run_id=new_run_id or self._generate_run_id(),
            started_at=earliest_start or datetime.now().isoformat(),
            completed_at=latest_end,
            models=sorted(all_models),
            languages=sorted(all_languages),
            evaluator_model=evaluator_model or "",
            status="completed" if latest_end else "running",
        )
        merged.results = all_results

        self.save_run(merged)
        return merged

    def export_csv(self, run_id: str, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Export a run's results to CSV format.

        Args:
            run_id: The run ID to export
            output_path: Optional output path. If None, uses results_dir.

        Returns:
            Path to CSV file or None if run not found
        """
        run = self.load_run(run_id)
        if run is None:
            return None

        if output_path is None:
            output_path = self.results_dir / f"{run_id}.csv"

        import csv

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "source_text_id",
                "target_language",
                "model",
                "accuracy",
                "fluency",
                "style",
                "overall",
                "translation_time_ms",
                "evaluation_time_ms",
                "timestamp",
                "error",
            ])

            # Data rows
            for result in run.results:
                scores = result.scores
                writer.writerow([
                    result.source_text_id,
                    result.target_language,
                    result.model,
                    scores.accuracy if scores else "",
                    scores.fluency if scores else "",
                    scores.style if scores else "",
                    scores.overall if scores else "",
                    result.translation_time_ms,
                    result.evaluation_time_ms,
                    result.timestamp,
                    result.error or "",
                ])

        return output_path

    def get_aggregated_stats(self, run_id: str) -> Optional[dict]:
        """
        Get aggregated statistics for a run.

        Args:
            run_id: The run ID to analyze

        Returns:
            Dictionary with model_stats, language_stats, and summary
        """
        run = self.load_run(run_id)
        if run is None:
            return None

        model_stats = run.get_model_stats()
        language_stats = run.get_language_stats()

        # Calculate best/worst for each
        for stats in model_stats:
            lang_scores = {}
            for result in run.results:
                if result.model == stats.model and result.scores:
                    lang = result.target_language
                    if lang not in lang_scores:
                        lang_scores[lang] = []
                    lang_scores[lang].append(result.scores.overall)

            if lang_scores:
                avg_by_lang = {
                    lang: sum(scores) / len(scores)
                    for lang, scores in lang_scores.items()
                }
                stats.best_language = max(avg_by_lang, key=avg_by_lang.get)
                stats.worst_language = min(avg_by_lang, key=avg_by_lang.get)

        for stats in language_stats:
            model_scores = {}
            for result in run.results:
                if result.target_language == stats.language_code and result.scores:
                    model = result.model
                    if model not in model_scores:
                        model_scores[model] = []
                    model_scores[model].append(result.scores.overall)

            if model_scores:
                avg_by_model = {
                    model: sum(scores) / len(scores)
                    for model, scores in model_scores.items()
                }
                stats.best_model = max(avg_by_model, key=avg_by_model.get)
                stats.worst_model = min(avg_by_model, key=avg_by_model.get)

        # Summary stats
        all_scores = [
            r.scores.overall
            for r in run.results
            if r.scores
        ]

        return {
            "model_stats": [s.to_dict() for s in model_stats],
            "language_stats": [s.to_dict() for s in language_stats],
            "summary": {
                "total_translations": len(run.results),
                "successful_translations": len(all_scores),
                "failed_translations": len(run.results) - len(all_scores),
                "avg_overall_score": sum(all_scores) / len(all_scores) if all_scores else 0,
                "min_score": min(all_scores) if all_scores else 0,
                "max_score": max(all_scores) if all_scores else 0,
            },
        }
