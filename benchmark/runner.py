"""
Benchmark orchestrator.

Coordinates the complete benchmark workflow:
1. Load languages and reference texts
2. Run translations with specified provider models
3. Evaluate translations with OpenRouter
4. Track progress and handle resumption
5. Generate results
"""

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Generator

import yaml

from benchmark.config import BenchmarkConfig
from benchmark.data_loader import load_languages, load_reference_texts
from benchmark.models import (
    Language, LanguageCategory, ReferenceText, TranslationResult,
    BenchmarkRun, EvaluationScores
)
from benchmark.translator import (
    BenchmarkTranslator, TranslationRequest,
    code_to_language_name,
    test_ollama_connection, get_available_ollama_models,
    test_openai_translation_connection, get_available_openai_models,
    test_openrouter_translation_connection, get_available_openrouter_models
)
from benchmark.evaluator import (
    TranslationEvaluator, test_openrouter_connection, test_poe_connection
)


class BenchmarkRunner:
    """
    Main orchestrator for benchmark runs.

    Handles:
    - Loading configuration and data
    - Running translation + evaluation pipeline
    - Progress tracking and callbacks
    - Error handling and resumption
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        log_callback: Optional[Callable[[str, str], None]] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ):
        """
        Initialize the benchmark runner.

        Args:
            config: Benchmark configuration
            log_callback: Optional callback for logging (level, message)
            progress_callback: Optional callback for progress (stage, current, total)
        """
        self.config = config
        self.log_callback = log_callback
        self.progress_callback = progress_callback

        self._languages: dict[str, Language] = {}
        self._texts: dict[str, ReferenceText] = {}
        self._translator: Optional[BenchmarkTranslator] = None
        self._evaluator: Optional[TranslationEvaluator] = None

    def _log(self, level: str, message: str) -> None:
        """Log a message."""
        if self.log_callback:
            self.log_callback(level, message)
        else:
            print(f"[{level.upper()}] {message}")

    def _progress(self, stage: str, current: int, total: int) -> None:
        """Report progress."""
        if self.progress_callback:
            self.progress_callback(stage, current, total)
        else:
            percent = (current / total * 100) if total > 0 else 0
            self._log("info", f"{stage}: {current}/{total} ({percent:.1f}%)")

    def load_languages(self) -> dict[str, Language]:
        """Load languages from the split layout (or legacy YAML)."""
        languages = load_languages(
            base_dir=self.config.paths.base_dir,
            legacy_file=self.config.paths.languages_file,
        )
        self._languages = languages
        self._log("info", f"Loaded {len(languages)} languages")
        return languages

    def load_reference_texts(self) -> dict[str, ReferenceText]:
        """Load reference texts from the split layout (or legacy YAML)."""
        texts = load_reference_texts(
            base_dir=self.config.paths.base_dir,
            legacy_file=self.config.paths.reference_texts_file,
        )
        self._texts = texts
        self._log("info", f"Loaded {len(texts)} reference texts")
        return texts

    def get_language(self, code: str) -> Optional[Language]:
        """Get a language by code."""
        return self._languages.get(code)

    def get_text(self, text_id: str) -> Optional[ReferenceText]:
        """Get a reference text by ID."""
        return self._texts.get(text_id)

    def filter_languages(self, codes: Optional[list[str]] = None) -> list[Language]:
        """
        Filter languages by codes.

        Args:
            codes: List of language codes to include. If None, returns all.

        Returns:
            List of Language objects
        """
        if codes is None:
            return list(self._languages.values())

        return [
            self._languages[code]
            for code in codes
            if code in self._languages
        ]

    async def validate_setup(self, evaluate: bool = True) -> tuple[bool, list[str]]:
        """
        Validate the benchmark setup.

        Args:
            evaluate: When False, skip evaluator connection checks.
        """
        errors = []

        config_errors = self.config.validate()
        errors.extend(config_errors)

        if self.config.translation_provider == "openrouter":
            or_trans_ok, or_trans_msg = await test_openrouter_translation_connection(self.config)
            if not or_trans_ok:
                errors.append(f"OpenRouter (translation): {or_trans_msg}")
            else:
                self._log("info", f"OpenRouter (translation): {or_trans_msg}")
        elif self.config.translation_provider == "openai":
            openai_ok, openai_msg = await test_openai_translation_connection(self.config)
            if not openai_ok:
                errors.append(f"OpenAI-compatible (translation): {openai_msg}")
            else:
                self._log("info", f"OpenAI-compatible (translation): {openai_msg}")
        elif self.config.translation_provider == "poe":
            if not self.config.poe.api_key:
                errors.append("Poe API key not configured. Set POE_API_KEY in .env or use --poe-key.")
            else:
                self._log("info", "Poe (translation): API key present")
        else:
            ollama_ok, ollama_msg = await test_ollama_connection(self.config)
            if not ollama_ok:
                errors.append(f"Ollama: {ollama_msg}")
            else:
                self._log("info", f"Ollama: {ollama_msg}")

        if evaluate:
            if self.config.evaluator_provider == "poe":
                poe_ok, poe_msg = await test_poe_connection(self.config)
                if not poe_ok:
                    errors.append(f"Poe (evaluation): {poe_msg}")
                else:
                    self._log("info", f"Poe (evaluation): {poe_msg}")
            else:
                openrouter_ok, openrouter_msg = await test_openrouter_connection(self.config)
                if not openrouter_ok:
                    errors.append(f"OpenRouter (evaluation): {openrouter_msg}")
                else:
                    self._log("info", f"OpenRouter (evaluation): {openrouter_msg}")
        else:
            self._log("info", "Evaluator check skipped (--no-evaluate)")

        return len(errors) == 0, errors

    def _resolve_lang_name(self, code: str) -> str:
        """Resolve a language code to its display name (loaded data, then fallback map)."""
        lang = self._languages.get(code)
        if lang is not None:
            return lang.name
        return code_to_language_name(code)

    def _generate_jobs(
        self,
        models: list[str],
        pairs: list[tuple[str, str]],
        texts: list[ReferenceText],
        existing_results: Optional[list[TranslationResult]] = None,
    ) -> Generator[TranslationRequest, None, None]:
        """
        Generate translation jobs for the given (source, target) pairs.

        For each pair, only reference texts whose `source_language` matches the
        pair's source code are emitted.
        """
        completed = set()
        if existing_results:
            for result in existing_results:
                if result.success:
                    key = (result.source_text_id, result.target_language, result.model)
                    completed.add(key)

        for src_code, tgt_code in pairs:
            src_name = self._resolve_lang_name(src_code)
            tgt_name = self._resolve_lang_name(tgt_code)
            src_texts = [t for t in texts if t.source_language == src_code]
            if not src_texts:
                self._log("warning", f"No reference texts for source language '{src_code}', skipping pair {src_code}->{tgt_code}")
                continue

            for model in models:
                for text in src_texts:
                    key = (text.id, tgt_code, model)
                    if key in completed:
                        continue
                    yield TranslationRequest(
                        text=text,
                        target_language=tgt_code,
                        target_language_name=tgt_name,
                        source_language=src_code,
                        source_language_name=src_name,
                        model=model,
                    )

    async def run(
        self,
        models: list[str],
        language_codes: Optional[list[str]] = None,
        pairs: Optional[list[tuple[str, str]]] = None,
        resume_run: Optional[BenchmarkRun] = None,
        evaluate: bool = True,
    ) -> BenchmarkRun:
        """
        Execute a complete benchmark run.

        Args:
            models: List of provider model names to benchmark
            language_codes: Target language codes (English source assumed). Used
                only when `pairs` is None.
            pairs: Explicit (source_code, target_code) pairs. Overrides `language_codes`.
            resume_run: Optional previous run to resume
            evaluate: When False, skip the LLM judge step entirely (translations
                are produced with `scores=None`).
        """
        # Load data if not already loaded
        if not self._languages:
            self.load_languages()
        if not self._texts:
            self.load_reference_texts()

        # Resolve pairs: explicit list wins, else build from language_codes assuming English source.
        if pairs is None:
            if language_codes is None:
                language_codes = self.config.quick_languages
            pairs = [("en", code) for code in language_codes]

        # Drop pairs whose target language is unknown.
        validated_pairs: list[tuple[str, str]] = []
        for src_code, tgt_code in pairs:
            if tgt_code not in self._languages:
                self._log("warning", f"Unknown target language '{tgt_code}', skipping {src_code}->{tgt_code}")
                continue
            validated_pairs.append((src_code, tgt_code))

        if not validated_pairs:
            raise ValueError("No valid (source, target) pairs to run.")
        if not models:
            raise ValueError("No models specified")

        texts = list(self._texts.values())
        target_codes = sorted({tgt for _, tgt in validated_pairs})

        # Determine evaluator model based on provider (label only — actual call gated by `evaluate`).
        if self.config.evaluator_provider == "poe":
            evaluator_model = self.config.poe.default_model
        else:
            evaluator_model = self.config.openrouter.default_model
        if not evaluate:
            evaluator_model = "skipped"

        # Create or resume run
        if resume_run:
            run = resume_run
            run.status = "running"
            self._log("info", f"Resuming run {run.run_id} ({run.total_completed}/{run.total_expected} completed)")
        else:
            run = BenchmarkRun(
                run_id=str(uuid.uuid4())[:8],
                started_at=datetime.now().isoformat(),
                models=models,
                languages=target_codes,
                evaluator_model=evaluator_model,
            )
            self._log("info", f"Starting new run {run.run_id}")

        # Log run parameters
        pair_strs = [f"{s}->{t}" for s, t in validated_pairs]
        self._log("info", f"Models: {', '.join(models)}")
        self._log("info", f"Pairs: {', '.join(pair_strs)}")
        self._log("info", f"Texts: {len(texts)}")
        if not evaluate:
            self._log("info", "Auto-evaluation disabled (--no-evaluate). Translations will have scores=None.")

        # Initialize translator (and evaluator only if needed)
        self._translator = BenchmarkTranslator(
            self.config,
            self.log_callback,
            provider_type=self.config.translation_provider
        )
        if evaluate:
            self._evaluator = TranslationEvaluator(
                self.config,
                self.log_callback,
                provider=self.config.evaluator_provider
            )
        else:
            self._evaluator = None

        try:
            jobs = list(self._generate_jobs(
                models=models,
                pairs=validated_pairs,
                texts=texts,
                existing_results=run.results if resume_run else None
            ))

            self._log("info", f"Jobs to process: {len(jobs)}")

            for i, job in enumerate(jobs):
                self._progress("translation", i + 1, len(jobs))

                result = await self._translator.translate(job)

                if evaluate and result.success and result.translated_text and self._evaluator is not None:
                    source_text = self.get_text(result.source_text_id)
                    if source_text:
                        scores, eval_time = await self._evaluator.evaluate(
                            source_text=source_text,
                            translated_text=result.translated_text,
                            target_language=result.target_language,
                            target_language_name=job.target_language_name
                        )
                        result.scores = scores
                        result.evaluation_time_ms = eval_time

                        self._log(
                            "info",
                            f"  Score: {scores.overall:.1f}/10 "
                            f"(acc={scores.accuracy:.1f}, flu={scores.fluency:.1f}, sty={scores.style:.1f})"
                        )

                # Add to run
                run.add_result(result)

                # Progress update
                self._progress("overall", run.total_completed, run.total_expected)

            # Complete run
            run.status = "completed"
            run.completed_at = datetime.now().isoformat()

            # Log summary
            self._log("info", "=" * 50)
            self._log("info", f"Run {run.run_id} completed")
            self._log("info", f"Total translations: {run.total_completed}")
            if run.results:
                success_rate = sum(1 for r in run.results if r.success) / len(run.results) * 100
                self._log("info", f"Success rate: {success_rate:.1f}%")

            if self._evaluator is not None:
                cost_summary = self._evaluator.get_cost_summary()
                self._log("info", f"Evaluation cost: ${cost_summary['total_cost_usd']:.4f}")

            return run

        except Exception as e:
            run.status = "failed"
            run.error = str(e)
            run.completed_at = datetime.now().isoformat()
            self._log("error", f"Run failed: {e}")
            raise

        finally:
            await self.close()

    async def close(self) -> None:
        """Clean up resources."""
        if self._translator:
            await self._translator.close()
        if self._evaluator:
            await self._evaluator.close()


async def quick_benchmark(
    config: BenchmarkConfig,
    models: Optional[list[str]] = None,
    log_callback: Optional[Callable[[str, str], None]] = None
) -> BenchmarkRun:
    """
    Run a quick benchmark with default settings.

    Args:
        config: Benchmark configuration
        models: Optional list of models (defaults to auto-detected provider models)
        log_callback: Optional logging callback

    Returns:
        BenchmarkRun with results
    """
    runner = BenchmarkRunner(config, log_callback)

    # Validate setup
    valid, errors = await runner.validate_setup()
    if not valid:
        raise RuntimeError(f"Setup validation failed: {'; '.join(errors)}")

    # Get models if not specified
    if models is None:
        if config.translation_provider == "openrouter":
            provider_models = await get_available_openrouter_models(config)
            models = [m["id"] if isinstance(m, dict) else m for m in provider_models]
        elif config.translation_provider == "openai":
            provider_models = await get_available_openai_models(config)
            models = [m["id"] if isinstance(m, dict) else m for m in provider_models]
        else:
            models = await get_available_ollama_models(config)
        if not models:
            raise RuntimeError(f"No {config.translation_provider} models available")
        # Limit to first 3 models for quick benchmark
        models = models[:3]

    return await runner.run(models=models)


async def full_benchmark(
    config: BenchmarkConfig,
    models: list[str],
    log_callback: Optional[Callable[[str, str], None]] = None
) -> BenchmarkRun:
    """
    Run a full benchmark with all languages.

    Args:
        config: Benchmark configuration
        models: List of provider models to benchmark
        log_callback: Optional logging callback

    Returns:
        BenchmarkRun with results
    """
    runner = BenchmarkRunner(config, log_callback)

    # Load all languages
    runner.load_languages()
    all_language_codes = list(runner._languages.keys())

    return await runner.run(models=models, language_codes=all_language_codes)
