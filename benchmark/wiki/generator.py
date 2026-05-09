"""
Wiki page generator for benchmark results.

Generates GitHub wiki pages from benchmark data using Jinja2 templates.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader

from ..config import BenchmarkConfig, get_score_indicator
from ..data_loader import load_languages, load_reference_texts
from ..models import BenchmarkRun, LanguageCategory
from ..results.storage import ResultsStorage


def _visual_len(text: str) -> int:
    """
    Calculate visual length of a string, accounting for wide characters.

    CJK characters and emojis take 2 columns in monospace fonts.
    """
    import unicodedata
    length = 0
    for char in text:
        # East Asian Width
        if unicodedata.east_asian_width(char) in ('F', 'W'):
            length += 2
        # Emojis (simplified check for common emoji ranges)
        elif ord(char) >= 0x1F300:
            length += 2
        else:
            length += 1
    return length


def _pad_to_width(text: str, width: int) -> str:
    """Pad a string to a specific visual width."""
    current_width = _visual_len(text)
    if current_width >= width:
        return text
    return text + " " * (width - current_width)


def format_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """
    Format a Markdown table with aligned columns.

    Args:
        headers: List of column headers
        rows: List of rows, where each row is a list of cell values

    Returns:
        Formatted Markdown table string
    """
    if not headers or not rows:
        return ""

    # Calculate max width for each column (using visual length)
    col_widths = [_visual_len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], _visual_len(str(cell)))

    # Build header row
    header_cells = [_pad_to_width(h, col_widths[i]) for i, h in enumerate(headers)]
    header_line = "| " + " | ".join(header_cells) + " |"

    # Build separator row
    separator_cells = ["-" * col_widths[i] for i in range(len(headers))]
    separator_line = "|-" + "-|-".join(separator_cells) + "-|"

    # Build data rows
    data_lines = []
    for row in rows:
        # Pad row if it has fewer cells than headers
        padded_row = list(row) + [""] * (len(headers) - len(row))
        cells = [_pad_to_width(str(padded_row[i]), col_widths[i]) for i in range(len(headers))]
        data_lines.append("| " + " | ".join(cells) + " |")

    return "\n".join([header_line, separator_line] + data_lines)


class WikiGenerator:
    """Generates GitHub wiki pages from benchmark results."""

    # Score indicators for templates
    INDICATORS = {
        "excellent": "\U0001F7E2",  # Green circle
        "good": "\U0001F7E1",       # Yellow circle
        "acceptable": "\U0001F7E0",  # Orange circle
        "poor": "\U0001F534",        # Red circle
        "failed": "\u26AB",          # Black circle
    }

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """
        Initialize wiki generator.

        Args:
            config: Benchmark configuration. If None, uses default.
        """
        self.config = config or BenchmarkConfig()
        self.storage = ResultsStorage(config)
        self.output_dir = self.config.paths.wiki_output_dir
        self.templates_dir = self.config.paths.templates_dir

        # Load Jinja2 environment
        self._env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Load language data via the unified loader (split or legacy YAML).
        self._languages = load_languages(
            base_dir=self.config.paths.base_dir,
            legacy_file=self.config.paths.languages_file,
        )

    _CATEGORY_LABELS = {
        LanguageCategory.EUROPEAN_MAJOR: "European Major Languages",
        LanguageCategory.ASIAN: "Asian Languages",
        LanguageCategory.SEMITIC: "Semitic Languages",
        LanguageCategory.CYRILLIC: "Cyrillic Languages",
        LanguageCategory.CLASSICAL: "Classical/Historical Languages",
        LanguageCategory.MINORITY: "Minority/Regional Languages",
    }

    def _get_language_info(self, code: str) -> dict:
        """Get language info by code."""
        lang = self._languages.get(code)
        if lang is not None:
            return {
                "code": lang.code,
                "name": lang.name,
                "native_name": lang.native_name,
                "script": lang.script,
                "category": self._CATEGORY_LABELS.get(lang.category, lang.category.value),
                "is_rtl": lang.is_rtl,
            }
        return {
            "code": code,
            "name": code,
            "native_name": code,
            "script": "Unknown",
            "category": "Unknown",
            "is_rtl": False,
        }

    def _slugify(self, text: str) -> str:
        """Convert text to URL-safe slug for GitHub wiki page names."""
        slug = text.lower()
        slug = re.sub(r"[^a-z0-9\-_]", "-", slug)
        slug = re.sub(r"-+", "-", slug)
        slug = slug.strip("-")
        return slug

    def _language_page_name(self, language_name: str) -> str:
        """Generate wiki page name for a language (flat structure for GitHub wiki)."""
        return f"Language-{self._slugify(language_name)}"

    def _model_page_name(self, model_name: str) -> str:
        """Generate wiki page name for a model (flat structure for GitHub wiki)."""
        return f"Model-{self._slugify(model_name)}"

    def _calculate_score_distribution(self, scores: list[float]) -> dict:
        """Calculate score distribution buckets."""
        dist = {"excellent": 0, "good": 0, "acceptable": 0, "poor": 0, "failed": 0}
        for score in scores:
            if score >= 9:
                dist["excellent"] += 1
            elif score >= 7:
                dist["good"] += 1
            elif score >= 5:
                dist["acceptable"] += 1
            elif score >= 3:
                dist["poor"] += 1
            else:
                dist["failed"] += 1
        return dist

    def generate_all(self, run_id: Optional[str] = None) -> Path:
        """
        Generate all wiki pages for a benchmark run.

        Args:
            run_id: Run ID to generate pages for. If None, uses latest run.

        Returns:
            Path to output directory
        """
        # Load benchmark run
        if run_id:
            run = self.storage.load_run(run_id)
        else:
            run = self.storage.get_latest_run()

        if run is None:
            raise ValueError("No benchmark run found")

        # Ensure output directory exists (flat structure for GitHub wiki)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate all pages
        self._generate_home(run)
        self._generate_all_languages_page(run)
        self._generate_all_models_page(run)
        self._generate_language_pages(run)
        self._generate_model_pages(run)

        return self.output_dir

    def _aggregation_summary_for_results(self, results: list) -> dict:
        """Compute n_obs and verified ratio for a slice of results."""
        n_obs_total = sum(getattr(r, "n_obs", 1) for r in results)
        n_results = len(results)
        verified = sum(1 for r in results if getattr(r, "verified", True))
        return {
            "n_obs_total": n_obs_total,
            "verified_results": verified,
            "self_reported_results": n_results - verified,
            "verified_badge": (
                "verified" if n_results > 0 and verified == n_results
                else ("self-reported" if verified == 0 else "mixed")
            ),
        }

    def _generate_home(self, run: BenchmarkRun) -> None:
        """Generate Home.md page."""
        template = self._env.get_template("home.md.j2")

        # Calculate model rankings
        model_stats = run.get_model_stats()
        results_by_model: dict = {}
        for r in run.results:
            results_by_model.setdefault(r.model, []).append(r)

        model_rankings = []
        for stats in sorted(model_stats, key=lambda x: x.avg_overall, reverse=True):
            agg = self._aggregation_summary_for_results(results_by_model.get(stats.model, []))
            model_rankings.append({
                "name": stats.model,
                "page_name": self._model_page_name(stats.model),
                "avg_overall": stats.avg_overall,
                "avg_accuracy": stats.avg_accuracy,
                "avg_fluency": stats.avg_fluency,
                "avg_style": stats.avg_style,
                "indicator": get_score_indicator(stats.avg_overall),
                "languages_tested": stats.successful_translations,
                **agg,
            })

        # Calculate language rankings
        language_stats = run.get_language_stats()
        results_by_lang: dict = {}
        for r in run.results:
            results_by_lang.setdefault(r.target_language, []).append(r)

        language_rankings = []
        for stats in sorted(language_stats, key=lambda x: x.avg_overall, reverse=True):
            lang_info = self._get_language_info(stats.language_code)
            agg = self._aggregation_summary_for_results(results_by_lang.get(stats.language_code, []))
            language_rankings.append({
                "name": lang_info["name"],
                "native_name": lang_info["native_name"],
                "page_name": self._language_page_name(lang_info["name"]),
                "avg_overall": stats.avg_overall,
                "indicator": get_score_indicator(stats.avg_overall),
                "best_model": stats.best_model or "N/A",
                "total_translations": stats.total_translations,
                **agg,
            })

        # Group languages by category
        categories = self._group_languages_by_category(language_rankings)

        content = template.render(
            last_updated=datetime.now().strftime("%Y-%m-%d %H:%M"),
            indicators=self.INDICATORS,
            model_rankings=model_rankings,
            language_rankings=language_rankings,
            categories=categories,
            total_models=len(run.models),
            total_languages=len(run.languages),
            total_translations=len(run.results),
            evaluator_model=run.evaluator_model,
        )

        (self.output_dir / "Home.md").write_text(content, encoding="utf-8")

    def _generate_all_languages_page(self, run: BenchmarkRun) -> None:
        """Generate All-Languages.md page."""
        language_stats = run.get_language_stats()
        results_by_lang: dict = {}
        for r in run.results:
            results_by_lang.setdefault(r.target_language, []).append(r)

        headers = ["Language", "Native Name", "Category", "Avg Score", "Best Model", "Obs", "Verified"]
        rows = []

        for stats in sorted(language_stats, key=lambda x: x.avg_overall, reverse=True):
            lang_info = self._get_language_info(stats.language_code)
            indicator = get_score_indicator(stats.avg_overall)
            page_name = self._language_page_name(lang_info['name'])
            agg = self._aggregation_summary_for_results(results_by_lang.get(stats.language_code, []))
            rows.append([
                f"[{lang_info['name']}]({page_name})",
                lang_info['native_name'],
                lang_info['category'],
                f"{indicator} {stats.avg_overall:.1f}",
                stats.best_model or "N/A",
                str(agg["n_obs_total"]),
                agg["verified_badge"],
            ])

        table = format_markdown_table(headers, rows)
        content = f"# All Languages\n\n{table}\n\n---\n\n[← Back to Home](Home)\n"

        (self.output_dir / "All-Languages.md").write_text(content, encoding="utf-8")

    def _generate_all_models_page(self, run: BenchmarkRun) -> None:
        """Generate All-Models.md page."""
        model_stats = run.get_model_stats()
        results_by_model: dict = {}
        for r in run.results:
            results_by_model.setdefault(r.model, []).append(r)

        headers = ["Model", "Avg Score", "Accuracy", "Fluency", "Style", "Languages", "Obs", "Verified"]
        rows = []

        for stats in sorted(model_stats, key=lambda x: x.avg_overall, reverse=True):
            indicator = get_score_indicator(stats.avg_overall)
            page_name = self._model_page_name(stats.model)
            agg = self._aggregation_summary_for_results(results_by_model.get(stats.model, []))
            rows.append([
                f"[{stats.model}]({page_name})",
                f"{indicator} {stats.avg_overall:.1f}",
                f"{stats.avg_accuracy:.1f}",
                f"{stats.avg_fluency:.1f}",
                f"{stats.avg_style:.1f}",
                str(stats.successful_translations),
                str(agg["n_obs_total"]),
                agg["verified_badge"],
            ])

        table = format_markdown_table(headers, rows)
        content = f"# All Models\n\n{table}\n\n---\n\n[← Back to Home](Home)\n"

        (self.output_dir / "All-Models.md").write_text(content, encoding="utf-8")

    def _generate_language_pages(self, run: BenchmarkRun) -> None:
        """Generate individual language pages."""
        template = self._env.get_template("language.md.j2")

        # Group results by language
        results_by_lang: dict = {}
        for result in run.results:
            lang = result.target_language
            if lang not in results_by_lang:
                results_by_lang[lang] = []
            results_by_lang[lang].append(result)

        for lang_code, results in results_by_lang.items():
            lang_info = self._get_language_info(lang_code)

            # Calculate stats for this language
            scores = [r.scores.overall for r in results if r.scores]
            avg_overall = sum(scores) / len(scores) if scores else 0
            avg_accuracy = sum(r.scores.accuracy for r in results if r.scores) / len(scores) if scores else 0
            avg_fluency = sum(r.scores.fluency for r in results if r.scores) / len(scores) if scores else 0
            avg_style = sum(r.scores.style for r in results if r.scores) / len(scores) if scores else 0

            # Group by model
            by_model: dict = {}
            for r in results:
                if r.model not in by_model:
                    by_model[r.model] = []
                by_model[r.model].append(r)

            model_results = []
            for model, model_results_list in by_model.items():
                m_scores = [r.scores for r in model_results_list if r.scores]
                if m_scores:
                    model_results.append({
                        "model": model,
                        "model_page_name": self._model_page_name(model),
                        "avg_overall": sum(s.overall for s in m_scores) / len(m_scores),
                        "avg_accuracy": sum(s.accuracy for s in m_scores) / len(m_scores),
                        "avg_fluency": sum(s.fluency for s in m_scores) / len(m_scores),
                        "avg_style": sum(s.style for s in m_scores) / len(m_scores),
                        "indicator": get_score_indicator(sum(s.overall for s in m_scores) / len(m_scores)),
                    })

            model_results.sort(key=lambda x: x["avg_overall"], reverse=True)

            # Get examples (best translations)
            examples = self._get_translation_examples(results, run)

            # Best/worst model
            best_model = model_results[0]["model"] if model_results else "N/A"
            worst_model = model_results[-1]["model"] if model_results else "N/A"

            content = template.render(
                language=lang_info,
                indicator=get_score_indicator(avg_overall),
                avg_overall=avg_overall,
                avg_accuracy=avg_accuracy,
                avg_fluency=avg_fluency,
                avg_style=avg_style,
                total_translations=len(results),
                model_results=model_results,
                best_model=best_model,
                best_model_page_name=self._model_page_name(best_model),
                worst_model=worst_model,
                worst_model_page_name=self._model_page_name(worst_model),
                examples=examples,
                score_dist=self._calculate_score_distribution(scores),
                indicators=self.INDICATORS,
            )

            # Write to flat directory structure (GitHub wiki doesn't support subdirectories)
            filename = f"{self._language_page_name(lang_info['name'])}.md"
            (self.output_dir / filename).write_text(content, encoding="utf-8")

    def _generate_model_pages(self, run: BenchmarkRun) -> None:
        """Generate individual model pages."""
        template = self._env.get_template("model.md.j2")

        # Group results by model
        results_by_model: dict = {}
        for result in run.results:
            model = result.model
            if model not in results_by_model:
                results_by_model[model] = []
            results_by_model[model].append(result)

        for model_name, results in results_by_model.items():
            # Calculate stats
            scores = [r.scores.overall for r in results if r.scores]
            avg_overall = sum(scores) / len(scores) if scores else 0
            avg_accuracy = sum(r.scores.accuracy for r in results if r.scores) / len(scores) if scores else 0
            avg_fluency = sum(r.scores.fluency for r in results if r.scores) / len(scores) if scores else 0
            avg_style = sum(r.scores.style for r in results if r.scores) / len(scores) if scores else 0

            # Group by language
            by_lang: dict = {}
            for r in results:
                if r.target_language not in by_lang:
                    by_lang[r.target_language] = []
                by_lang[r.target_language].append(r)

            language_results = []
            for lang_code, lang_results_list in by_lang.items():
                lang_info = self._get_language_info(lang_code)
                l_scores = [r.scores for r in lang_results_list if r.scores]
                if l_scores:
                    lang_avg = sum(s.overall for s in l_scores) / len(l_scores)
                    language_results.append({
                        "code": lang_code,
                        "name": lang_info["name"],
                        "page_name": self._language_page_name(lang_info["name"]),
                        "category": lang_info["category"],
                        "avg_overall": lang_avg,
                        "avg_accuracy": sum(s.accuracy for s in l_scores) / len(l_scores),
                        "avg_fluency": sum(s.fluency for s in l_scores) / len(l_scores),
                        "avg_style": sum(s.style for s in l_scores) / len(l_scores),
                        "indicator": get_score_indicator(lang_avg),
                    })

            language_results.sort(key=lambda x: x["avg_overall"], reverse=True)

            # Best/worst language
            best_lang = language_results[0] if language_results else None
            worst_lang = language_results[-1] if language_results else None

            # Group by category
            categories = self._group_results_by_category(language_results)

            # Get best/worst examples
            best_example = self._get_best_example(results, run)
            worst_example = self._get_worst_example(results, run)

            # Calculate timing stats
            translation_times = [r.translation_time_ms for r in results if r.translation_time_ms > 0]
            avg_translation_time = sum(translation_times) / len(translation_times) if translation_times else 0

            content = template.render(
                model={"name": model_name, "id": model_name},
                indicator=get_score_indicator(avg_overall),
                avg_overall=avg_overall,
                avg_accuracy=avg_accuracy,
                avg_fluency=avg_fluency,
                avg_style=avg_style,
                total_languages=len(set(r.target_language for r in results)),
                total_translations=len(results),
                successful_translations=len([r for r in results if r.success and r.scores]),
                language_results=language_results,
                categories=categories,
                best_language=best_lang["name"] if best_lang else "N/A",
                best_language_page_name=self._language_page_name(best_lang["name"]) if best_lang else "",
                best_language_score=best_lang["avg_overall"] if best_lang else 0,
                worst_language=worst_lang["name"] if worst_lang else "N/A",
                worst_language_page_name=self._language_page_name(worst_lang["name"]) if worst_lang else "",
                worst_language_score=worst_lang["avg_overall"] if worst_lang else 0,
                best_example=best_example,
                worst_example=worst_example,
                score_dist=self._calculate_score_distribution(scores),
                indicators=self.INDICATORS,
                avg_translation_time_ms=avg_translation_time,
            )

            # Write to flat directory structure (GitHub wiki doesn't support subdirectories)
            filename = f"{self._model_page_name(model_name)}.md"
            (self.output_dir / filename).write_text(content, encoding="utf-8")

    def _group_languages_by_category(self, language_rankings: list[dict]) -> list[dict]:
        """Group language rankings by category."""
        categories_map: dict = {}

        # Build name -> category-label map from the loaded languages.
        name_to_category = {
            lang.name: self._CATEGORY_LABELS.get(lang.category, lang.category.value)
            for lang in self._languages.values()
        }

        for lang in language_rankings:
            category_name = name_to_category.get(lang["name"], "Other")
            if category_name not in categories_map:
                categories_map[category_name] = {
                    "name": category_name,
                    "languages": [],
                }
            categories_map[category_name]["languages"].append(lang)

        return list(categories_map.values())

    def _group_results_by_category(self, language_results: list[dict]) -> list[dict]:
        """Group model's language results by category."""
        categories_map: dict = {}

        for lang in language_results:
            category = lang.get("category", "Other")
            if category not in categories_map:
                categories_map[category] = {
                    "name": category,
                    "languages": [],
                    "avg_overall": 0,
                }
            categories_map[category]["languages"].append(lang)

        # Calculate category averages
        for cat in categories_map.values():
            if cat["languages"]:
                cat["avg_overall"] = sum(l["avg_overall"] for l in cat["languages"]) / len(cat["languages"])
                cat["indicator"] = get_score_indicator(cat["avg_overall"])

        return list(categories_map.values())

    def _get_translation_examples(self, results: list, run: BenchmarkRun) -> list[dict]:
        """Get best translation examples for a language."""
        # Load reference texts for source text lookup
        ref_texts = self._load_reference_texts()

        examples = []
        sorted_results = sorted(
            [r for r in results if r.scores],
            key=lambda x: x.scores.overall,
            reverse=True,
        )

        for result in sorted_results[:3]:
            ref = ref_texts.get(result.source_text_id, {})
            examples.append({
                "model": result.model,
                "text_title": ref.get("title", result.source_text_id),
                "author": ref.get("author", "Unknown"),
                "year": ref.get("year", ""),
                "source_text": ref.get("content", "")[:500],
                "translated_text": result.translated_text[:500],
                "overall": result.scores.overall if result.scores else 0,
                "indicator": get_score_indicator(result.scores.overall) if result.scores else self.INDICATORS["failed"],
                "feedback": result.scores.feedback if result.scores else None,
            })

        return examples

    def _get_best_example(self, results: list, run: BenchmarkRun) -> Optional[dict]:
        """Get the best translation example for a model."""
        ref_texts = self._load_reference_texts()

        best = max(
            [r for r in results if r.scores],
            key=lambda x: x.scores.overall,
            default=None,
        )

        if not best:
            return None

        lang_info = self._get_language_info(best.target_language)
        ref = ref_texts.get(best.source_text_id, {})

        return {
            "language": lang_info["name"],
            "text_title": ref.get("title", best.source_text_id),
            "source_text": ref.get("content", "")[:500],
            "translated_text": best.translated_text[:500],
            "overall": best.scores.overall,
            "indicator": get_score_indicator(best.scores.overall),
            "feedback": best.scores.feedback,
        }

    def _get_worst_example(self, results: list, run: BenchmarkRun) -> Optional[dict]:
        """Get the worst translation example for a model."""
        ref_texts = self._load_reference_texts()

        worst = min(
            [r for r in results if r.scores],
            key=lambda x: x.scores.overall,
            default=None,
        )

        if not worst:
            return None

        lang_info = self._get_language_info(worst.target_language)
        ref = ref_texts.get(worst.source_text_id, {})

        return {
            "language": lang_info["name"],
            "text_title": ref.get("title", worst.source_text_id),
            "source_text": ref.get("content", "")[:500],
            "translated_text": worst.translated_text[:500],
            "overall": worst.scores.overall,
            "indicator": get_score_indicator(worst.scores.overall),
            "feedback": worst.scores.feedback,
        }

    def _load_reference_texts(self) -> dict:
        """Load reference texts as a flat {id: dict} map for template usage."""
        texts = load_reference_texts(
            base_dir=self.config.paths.base_dir,
            legacy_file=self.config.paths.reference_texts_file,
        )
        return {t.id: t.to_dict() for t in texts.values()}
