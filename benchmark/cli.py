"""
Benchmark CLI - Command line interface for the benchmark system.

Provides commands for:
- Running benchmarks (quick or full)
- Generating wiki pages
- Listing and managing runs
"""

import argparse
import asyncio
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Force UTF-8 stdio so emoji prints (e.g. 💬, ❌, ⚠️) don't crash on Windows
# consoles using cp1252 (charmap codec). errors='replace' keeps the process
# alive even on terminals that can't render some glyphs.
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from benchmark.aggregator import CLOUD_PROVIDERS, SubmissionAggregator
from benchmark.config import BenchmarkConfig, DEFAULT_EVALUATOR_MODEL, DEFAULT_EVALUATOR_PROVIDER, DEFAULT_POE_EVALUATOR_MODEL
from benchmark.models import (
    BenchmarkRun,
    EvaluationScores,
    Submission,
    SubmissionResult,
)
from benchmark.runner import BenchmarkRunner, quick_benchmark, full_benchmark
from benchmark.results.storage import ResultsStorage
from benchmark.wiki.generator import WikiGenerator
from benchmark.translator import (
    get_available_ollama_models,
    get_available_openrouter_models,
    get_available_openai_models,
)


# ANSI color codes for terminal output
class Colors:
    """Terminal color codes."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def colored(text: str, color: str) -> str:
    """Apply color to text if terminal supports it."""
    if sys.stdout.isatty():
        return f"{color}{text}{Colors.ENDC}"
    return text


def log_callback(level: str, message: str) -> None:
    """Colored logging callback for CLI output."""
    level_colors = {
        "info": Colors.CYAN,
        "warning": Colors.YELLOW,
        "error": Colors.RED,
        "debug": Colors.BLUE,
    }
    color = level_colors.get(level.lower(), Colors.ENDC)
    prefix = colored(f"[{level.upper()}]", color)
    print(f"{prefix} {message}")


def print_banner() -> None:
    """Print CLI banner."""
    banner = """
+---------------------------------------------------------------+
|          TranslateBookWithLLM - Benchmark System              |
|                                                               |
|  Test translation quality across 40+ languages and models     |
+---------------------------------------------------------------+
"""
    print(colored(banner, Colors.HEADER))


def cmd_run(args: argparse.Namespace) -> int:
    """Execute benchmark run command."""
    print_banner()

    # Determine provider
    provider = getattr(args, 'provider', 'ollama') or 'ollama'

    # Build configuration
    evaluator_provider = getattr(args, 'evaluator_provider', DEFAULT_EVALUATOR_PROVIDER)
    config = BenchmarkConfig.from_cli_args(
        openrouter_key=args.openrouter_key,
        openai_key=args.openai_key,
        openai_endpoint=args.openai_endpoint,
        poe_key=args.poe_key,
        evaluator_model=args.evaluator,
        ollama_endpoint=args.ollama_endpoint,
        translation_provider=provider,
        evaluator_provider=evaluator_provider,
    )

    # Validate configuration
    errors = config.validate()
    if errors:
        for error in errors:
            log_callback("error", error)
        return 1

    # Get models based on provider
    models = args.models
    if not models:
        if provider == "poe":
            log_callback("error", "Poe provider requires explicit --models (e.g. 'gemini-3-flash-preview', 'mistral-medium-3.1', 'gpt-5-mini').")
            return 1
        if provider == "openrouter":
            print(colored("Fetching available OpenRouter models...", Colors.CYAN))
            models_data = asyncio.run(get_available_openrouter_models(config))
            if not models_data:
                log_callback("error", "No OpenRouter models available.")
                return 1
            # Extract model IDs
            models = [m["id"] if isinstance(m, dict) else m for m in models_data[:10]]
            print(colored(f"Found {len(models_data)} models. Using top 10: {', '.join(models[:3])}...", Colors.GREEN))
        elif provider == "openai":
            print(colored("Fetching available OpenAI-compatible models...", Colors.CYAN))
            models_data = asyncio.run(get_available_openai_models(config))
            if not models_data:
                log_callback("error", "No OpenAI-compatible models available.")
                return 1
            models = [m["id"] if isinstance(m, dict) else m for m in models_data[:10]]
            print(colored(f"Found {len(models_data)} models. Using top 10: {', '.join(models[:3])}...", Colors.GREEN))
        else:
            print(colored("Detecting available Ollama models...", Colors.CYAN))
            models = asyncio.run(get_available_ollama_models(config))
            if not models:
                log_callback("error", "No Ollama models found. Run 'ollama pull <model>' first.")
                return 1
            print(colored(f"Found {len(models)} models: {', '.join(models[:5])}...", Colors.GREEN))

    # Show provider info
    print(colored(f"Translation provider: {provider.upper()}", Colors.YELLOW))

    # Determine pairs / language codes
    pairs: Optional[list[tuple[str, str]]] = None
    language_codes: Optional[list[str]] = None
    if getattr(args, "pairs", None):
        pairs = []
        for spec in args.pairs:
            if ":" not in spec:
                log_callback("error", f"--pairs entries must be 'src:tgt' (got: {spec})")
                return 1
            src, tgt = spec.split(":", 1)
            pairs.append((src.strip(), tgt.strip()))
        print(colored(f"Running benchmark on {len(pairs)} pair(s): {', '.join(args.pairs)}", Colors.CYAN))
    elif args.full:
        language_codes = None
        print(colored("Running FULL benchmark with all 40+ languages (English source)", Colors.YELLOW))
    elif args.languages:
        language_codes = args.languages
        print(colored(f"Running benchmark with languages: {', '.join(language_codes)} (English source)", Colors.CYAN))
    else:
        language_codes = config.quick_languages
        print(colored(f"Running QUICK benchmark with {len(language_codes)} languages (English source)", Colors.CYAN))

    evaluate = not getattr(args, "no_evaluate", False)
    if not evaluate:
        print(colored("Auto-evaluation DISABLED — translations only.", Colors.YELLOW))

    # Check for resumable run
    storage = ResultsStorage(config)
    resume_run = None

    if args.resume:
        resume_run = storage.load_run(args.resume)
        if resume_run:
            print(colored(f"Resuming run {args.resume}...", Colors.YELLOW))
        else:
            log_callback("warning", f"Run {args.resume} not found, starting fresh")

    # Create runner
    runner = BenchmarkRunner(
        config=config,
        log_callback=log_callback,
    )

    # Run benchmark
    try:
        print(colored("\nStarting benchmark...\n", Colors.BOLD))

        run = asyncio.run(runner.run(
            models=models,
            language_codes=language_codes,
            pairs=pairs,
            resume_run=resume_run,
            evaluate=evaluate,
        ))

        # Save results
        storage.save_run(run)
        print(colored(f"\nResults saved to: {storage._get_run_path(run.run_id)}", Colors.GREEN))

        # Print summary
        print_run_summary(run)

        return 0

    except KeyboardInterrupt:
        print(colored("\nBenchmark interrupted by user", Colors.YELLOW))
        return 130
    except Exception as e:
        log_callback("error", f"Benchmark failed: {e}")
        return 1


def cmd_wiki(args: argparse.Namespace) -> int:
    """Generate wiki pages from benchmark results."""
    print_banner()

    config = BenchmarkConfig()
    generator = WikiGenerator(config)

    run_id = args.run_id

    try:
        print(colored("Generating wiki pages...", Colors.CYAN))

        output_dir = generator.generate_all(run_id)

        print(colored(f"\nWiki pages generated successfully!", Colors.GREEN))
        print(colored(f"Output directory: {output_dir}", Colors.CYAN))
        print()
        print("Generated pages:")
        print(f"  - Home.md")
        print(f"  - All-Languages.md")
        print(f"  - All-Models.md")
        print(f"  - languages/*.md")
        print(f"  - models/*.md")
        print()
        print(colored("Copy the contents of the 'wiki' directory to your GitHub wiki repository.", Colors.YELLOW))

        return 0

    except ValueError as e:
        log_callback("error", str(e))
        return 1
    except Exception as e:
        log_callback("error", f"Wiki generation failed: {e}")
        return 1


def cmd_list(args: argparse.Namespace) -> int:
    """List available benchmark runs."""
    config = BenchmarkConfig()
    storage = ResultsStorage(config)

    runs = storage.list_runs()

    if not runs:
        print(colored("No benchmark runs found.", Colors.YELLOW))
        return 0

    print(colored("\nAvailable benchmark runs:\n", Colors.BOLD))

    # Table header
    print(f"{'Run ID':<20} {'Status':<12} {'Started':<20} {'Models':<30} {'Results'}")
    print("-" * 100)

    for run in runs:
        status_color = {
            "completed": Colors.GREEN,
            "running": Colors.YELLOW,
            "failed": Colors.RED,
        }.get(run["status"], Colors.ENDC)

        status = colored(run["status"], status_color)
        models_str = ", ".join(run["models"][:2])
        if len(run["models"]) > 2:
            models_str += f" (+{len(run['models']) - 2})"

        started = run["started_at"][:19] if run["started_at"] else "N/A"

        print(f"{run['run_id']:<20} {status:<22} {started:<20} {models_str:<30} {run['total_results']}")

    print()
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Show details of a specific benchmark run."""
    config = BenchmarkConfig()
    storage = ResultsStorage(config)

    run = storage.load_run(args.run_id)
    if not run:
        log_callback("error", f"Run {args.run_id} not found")
        return 1

    print_run_summary(run)

    # Show detailed stats if requested
    if args.detailed:
        stats = storage.get_aggregated_stats(args.run_id)
        if stats:
            print(colored("\nModel Statistics:", Colors.BOLD))
            for model_stat in stats["model_stats"]:
                print(f"  {model_stat['model']}: avg={model_stat['avg_overall']:.1f}, "
                      f"best_lang={model_stat.get('best_language', 'N/A')}")

            print(colored("\nLanguage Statistics:", Colors.BOLD))
            for lang_stat in stats["language_stats"]:
                print(f"  {lang_stat['language_code']}: avg={lang_stat['avg_overall']:.1f}, "
                      f"best_model={lang_stat.get('best_model', 'N/A')}")

    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export benchmark run to CSV."""
    config = BenchmarkConfig()
    storage = ResultsStorage(config)

    output_path = storage.export_csv(args.run_id, args.output)

    if output_path:
        print(colored(f"Exported to: {output_path}", Colors.GREEN))
        return 0
    else:
        log_callback("error", f"Run {args.run_id} not found")
        return 1


def _fetch_model_ids(provider: str, config: BenchmarkConfig) -> list[str]:
    """Return a flat list of model IDs available on the given provider."""
    if provider == "ollama":
        return asyncio.run(get_available_ollama_models(config)) or []
    if provider == "openrouter":
        models = asyncio.run(get_available_openrouter_models(config)) or []
        return [m.get("id") if isinstance(m, dict) else m for m in models]
    if provider == "openai":
        models = asyncio.run(get_available_openai_models(config)) or []
        return [m.get("id") if isinstance(m, dict) else m for m in models]
    if provider == "poe":
        from src.core.llm.providers.poe import PoeProvider
        if not config.poe.api_key:
            return []
        poe = PoeProvider(api_key=config.poe.api_key, model="placeholder")
        models = asyncio.run(poe.get_available_models()) or []
        return [m.get("id") for m in models if m.get("id")]
    return []


def _close_matches(target: str, candidates: list[str], limit: int = 10) -> list[str]:
    """Return up to `limit` close matches to `target` from `candidates`.

    Combines difflib similarity with substring/token overlap so single-token
    misspellings like 'gemini-3-flash-preview' surface 'gemini-3-flash'.
    """
    import difflib
    target_l = target.lower()
    target_tokens = set(re.split(r"[^a-z0-9]+", target_l))
    target_tokens.discard("")

    scored: list[tuple[float, str]] = []
    for c in candidates:
        if not c:
            continue
        cl = c.lower()
        ratio = difflib.SequenceMatcher(None, target_l, cl).ratio()
        if target_l in cl or cl in target_l:
            ratio += 0.2
        c_tokens = set(re.split(r"[^a-z0-9]+", cl))
        if target_tokens & c_tokens:
            ratio += 0.1 * len(target_tokens & c_tokens)
        scored.append((ratio, c))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return [c for _, c in scored[:limit]]


def cmd_models(args: argparse.Namespace) -> int:
    """List available models for benchmarking, or validate a specific id."""
    config = BenchmarkConfig.from_cli_args(
        openrouter_key=args.openrouter_key,
        openai_key=args.openai_key,
        openai_endpoint=args.openai_endpoint,
        poe_key=getattr(args, "poe_key", None),
        translation_provider=args.provider,
    )
    provider = args.provider

    # --check: validate a single model id and exit 0/1.
    if getattr(args, "check", None):
        target = args.check
        ids = _fetch_model_ids(provider, config)
        if not ids:
            print(f"NOT FOUND: could not fetch model list from {provider} "
                  f"(missing API key, network error, or service down)")
            return 1
        if target in ids:
            print(f"OK: '{target}' is available on {provider}")
            return 0
        matches = _close_matches(target, ids)
        print(f"NOT FOUND: '{target}' is not a valid {provider} model id.")
        if matches:
            print("Closest matches:")
            for m in matches:
                print(f"  - {m}")
        return 1

    print_banner()

    if provider == "openrouter":
        print(colored("Fetching OpenRouter models...\n", Colors.CYAN))
        models = asyncio.run(get_available_openrouter_models(config))

        if not models:
            log_callback("error", "Failed to fetch OpenRouter models")
            return 1

        print(colored(f"Available OpenRouter Models ({len(models)} text-only models):\n", Colors.BOLD))

        # Table header
        print(f"{'Model ID':<50} {'Price (per 1M tokens)':<25}")
        print("-" * 75)

        for model in models[:50]:  # Limit to 50 for readability
            if isinstance(model, dict):
                model_id = model.get("id", "unknown")
                pricing = model.get("pricing", {})
                prompt_price = pricing.get("prompt_per_million", 0)
                completion_price = pricing.get("completion_per_million", 0)
                price_str = f"${prompt_price:.2f} / ${completion_price:.2f}"
            else:
                model_id = model
                price_str = "N/A"

            print(f"{model_id:<50} {price_str:<25}")

        print()
        print(colored("Tip: Use -m to specify models, e.g.:", Colors.YELLOW))
        print("  python -m benchmark.cli run -p openrouter -m anthropic/claude-sonnet-4 openai/gpt-4o")

    elif provider == "openai":
        print(colored("Fetching OpenAI-compatible models...\n", Colors.CYAN))
        models = asyncio.run(get_available_openai_models(config))

        if not models:
            log_callback("error", "Failed to fetch OpenAI-compatible models")
            return 1

        print(colored(f"Available OpenAI-Compatible Models ({len(models)}):\n", Colors.BOLD))
        print(f"{'Model ID':<50} {'Owner':<20}")
        print("-" * 72)

        for model in models[:50]:
            if isinstance(model, dict):
                model_id = model.get("id", "unknown")
                owned_by = model.get("owned_by", "unknown")
            else:
                model_id = model
                owned_by = "unknown"

            print(f"{model_id:<50} {owned_by:<20}")

        print()
        print(colored("Tip: Use -m and --openai-endpoint to specify a backend, e.g.:", Colors.YELLOW))
        print("  python -m benchmark.cli run -p openai --openai-endpoint http://localhost:8080/v1 -m your-model")

    elif provider == "poe":
        print(colored("Fetching Poe models...\n", Colors.CYAN))
        ids = _fetch_model_ids("poe", config)
        if not ids:
            log_callback("error", "Failed to fetch Poe models. Check POE_API_KEY or --poe-key.")
            return 1

        print(colored(f"Available Poe Models ({len(ids)}):\n", Colors.BOLD))
        for model_id in ids[:80]:
            print(f"  - {model_id}")
        if len(ids) > 80:
            print(f"  ... ({len(ids) - 80} more)")

        print()
        print(colored("Tip: Use -m to specify a model, e.g.:", Colors.YELLOW))
        print("  python -m benchmark.cli run -p poe -m gemini-3-flash")

    else:
        print(colored("Detecting Ollama models...\n", Colors.CYAN))
        models = asyncio.run(get_available_ollama_models(config))

        if not models:
            log_callback("error", "No Ollama models found. Is Ollama running? Try 'ollama pull <model>'")
            return 1

        print(colored(f"Available Ollama Models ({len(models)}):\n", Colors.BOLD))
        for model in models:
            print(f"  - {model}")

        print()
        print(colored("Tip: Use -m to specify models, e.g.:", Colors.YELLOW))
        print("  python -m benchmark.cli run -m llama3:8b qwen2.5:14b")

    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    """Delete a benchmark run."""
    config = BenchmarkConfig()
    storage = ResultsStorage(config)

    if not args.force:
        confirm = input(f"Delete run {args.run_id}? [y/N]: ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return 0

    if storage.delete_run(args.run_id):
        print(colored(f"Deleted run {args.run_id}", Colors.GREEN))
        return 0
    else:
        log_callback("error", f"Run {args.run_id} not found")
        return 1


def cmd_merge(args: argparse.Namespace) -> int:
    """Merge multiple benchmark runs into one."""
    print_banner()

    config = BenchmarkConfig()
    storage = ResultsStorage(config)

    run_ids = args.run_ids

    # Validate all runs exist
    for run_id in run_ids:
        run = storage.load_run(run_id)
        if run is None:
            log_callback("error", f"Run {run_id} not found")
            return 1

    print(colored(f"Merging {len(run_ids)} runs...", Colors.CYAN))

    merged = storage.merge_runs(run_ids, new_run_id=args.output)

    if merged is None:
        log_callback("error", "No valid results to merge")
        return 1

    print(colored(f"\nMerged run created: {merged.run_id}", Colors.GREEN))
    print(f"  Models: {', '.join(merged.models)}")
    print(f"  Languages: {len(merged.languages)}")
    print(f"  Total results: {len(merged.results)}")

    # Optionally regenerate wiki
    if args.publish:
        print(colored("\nPublishing merged results to wiki...", Colors.CYAN))
        from benchmark.wiki.generator import WikiGenerator
        generator = WikiGenerator(config)
        generator.generate_all(merged.run_id)
        print(colored("Wiki updated.", Colors.GREEN))

    return 0


def cmd_wiki_publish(args: argparse.Namespace) -> int:
    """Generate wiki pages and publish to GitHub wiki repository."""
    import shutil
    import subprocess

    print_banner()

    config = BenchmarkConfig()
    generator = WikiGenerator(config)

    wiki_clone_dir = config.paths.wiki_clone_dir
    wiki_output_dir = config.paths.wiki_output_dir
    wiki_repo_url = config.paths.wiki_repo_url

    run_id = args.run_id

    try:
        # Step 1: Generate wiki pages
        print(colored("Step 1/4: Generating wiki pages...", Colors.CYAN))
        generator.generate_all(run_id)
        print(colored("Wiki pages generated.", Colors.GREEN))

        # Step 2: Clone or update wiki repo
        print(colored("Step 2/4: Cloning/updating wiki repository...", Colors.CYAN))

        if wiki_clone_dir.exists():
            # Pull latest changes
            result = subprocess.run(
                ["git", "-C", str(wiki_clone_dir), "pull", "--rebase"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                # If pull fails, delete and re-clone
                shutil.rmtree(wiki_clone_dir)
                subprocess.run(
                    ["git", "clone", wiki_repo_url, str(wiki_clone_dir)],
                    check=True,
                    capture_output=True
                )
        else:
            # Clone fresh
            result = subprocess.run(
                ["git", "clone", wiki_repo_url, str(wiki_clone_dir)],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                log_callback("error", f"Failed to clone wiki repo: {result.stderr}")
                log_callback("error", "Make sure you have created at least one wiki page on GitHub first.")
                return 1

        print(colored("Wiki repository ready.", Colors.GREEN))

        # Step 3: Copy generated files to wiki repo
        print(colored("Step 3/4: Copying files to wiki repository...", Colors.CYAN))

        # Remove old subdirectories (now using flat structure)
        for old_subdir in ["languages", "models"]:
            old_dir = wiki_clone_dir / old_subdir
            if old_dir.exists():
                shutil.rmtree(old_dir)

        # Copy all markdown files (flat structure)
        for md_file in wiki_output_dir.glob("*.md"):
            shutil.copy2(md_file, wiki_clone_dir / md_file.name)

        print(colored("Files copied.", Colors.GREEN))

        # Step 4: Commit and push
        print(colored("Step 4/4: Committing and pushing changes...", Colors.CYAN))

        # Add all changes
        subprocess.run(
            ["git", "-C", str(wiki_clone_dir), "add", "-A"],
            check=True,
            capture_output=True
        )

        # Check if there are changes to commit
        result = subprocess.run(
            ["git", "-C", str(wiki_clone_dir), "status", "--porcelain"],
            capture_output=True,
            text=True
        )

        if not result.stdout.strip():
            print(colored("No changes to commit.", Colors.YELLOW))
            return 0

        # Commit
        commit_msg = f"Update benchmark results ({run_id or 'latest'})"
        subprocess.run(
            ["git", "-C", str(wiki_clone_dir), "commit", "-m", commit_msg],
            check=True,
            capture_output=True
        )

        # Push
        result = subprocess.run(
            ["git", "-C", str(wiki_clone_dir), "push"],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            log_callback("error", f"Failed to push: {result.stderr}")
            return 1

        print(colored("\nWiki published successfully!", Colors.GREEN))
        print(colored(f"View at: https://github.com/hydropix/TranslateBookWithLLM/wiki", Colors.CYAN))

        return 0

    except subprocess.CalledProcessError as e:
        log_callback("error", f"Git command failed: {e}")
        return 1
    except Exception as e:
        log_callback("error", f"Wiki publish failed: {e}")
        return 1


_SUBMISSION_SCHEMA_VERSION = "1.0"
_GITHUB_LOGIN_RE = re.compile(r"^github:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,38})$")


def _slugify_for_filename(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", text).strip("-").lower()
    return slug or "model"


def _sha256_text(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _detect_tbl_version() -> str:
    """Best-effort TBL version: env var > git short SHA > 'dev'."""
    import os
    env = os.getenv("TBL_VERSION")
    if env:
        return env
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return f"git-{result.stdout.strip()}"
    except Exception:
        pass
    return "dev"


def _validate_submission_schema(submission_dict: dict) -> list[str]:
    """Validate against the JSON schema. Returns [] on success, [errors] on failure.

    Returns a single warning entry instead of failing if `jsonschema` is missing.
    """
    schema_path = Path(__file__).parent / "schemas" / "submission.schema.json"
    if not schema_path.exists():
        return [f"Schema file not found: {schema_path}"]

    try:
        import jsonschema
    except ImportError:
        return ["WARN: jsonschema not installed; skipping schema validation. Install with `pip install jsonschema`."]

    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(submission_dict), key=lambda e: e.path)
    return [f"{'/'.join(str(p) for p in err.path) or '<root>'}: {err.message}" for err in errors]


def _build_submission_from_run(
    run: BenchmarkRun,
    submitted_by: str,
    *,
    provider: str,
    judge_id: str,
    tbl_version: str,
    prompt_version: str,
    notes: Optional[str] = None,
    judge_temperature: Optional[float] = None,
    judge_seed: Optional[int] = None,
    source_lang_default: str = "en",
) -> tuple[Submission, str]:
    """
    Convert an existing BenchmarkRun into a Submission, computing output hashes.

    Returns (submission, single_model_id). `single_model_id` is the dominant
    model — submissions in this layout cover one model at a time.
    """
    if not run.results:
        raise ValueError(f"Run {run.run_id} has no results to submit.")

    # The submission schema covers a single model. Pick the most-used model
    # in the run and warn if there are others.
    model_counts: dict[str, int] = {}
    for r in run.results:
        model_counts[r.model] = model_counts.get(r.model, 0) + 1
    primary_model = max(model_counts.items(), key=lambda kv: kv[1])[0]

    sub_results: list[SubmissionResult] = []
    skipped_other_models = 0
    skipped_no_scores = 0
    skipped_failed = 0

    for r in run.results:
        if r.model != primary_model:
            skipped_other_models += 1
            continue
        if not r.success:
            skipped_failed += 1
            continue
        if r.scores is None:
            skipped_no_scores += 1
            continue

        sub_results.append(SubmissionResult(
            text_id=r.source_text_id,
            source_lang=source_lang_default,
            target_lang=r.target_language,
            output=r.translated_text,
            output_hash=_sha256_text(r.translated_text),
            translation_latency_ms=int(r.translation_time_ms or 0),
            scores=EvaluationScores(
                accuracy=float(r.scores.accuracy),
                fluency=float(r.scores.fluency),
                style=float(r.scores.style),
                overall=float(r.scores.overall),
                feedback=r.scores.feedback,
            ),
        ))

    if not sub_results:
        raise ValueError(
            f"No usable results in run {run.run_id} for model {primary_model}."
        )

    if skipped_other_models:
        log_callback(
            "warning",
            f"Skipped {skipped_other_models} results for non-primary models. "
            "Run `submit` once per model.",
        )
    if skipped_failed or skipped_no_scores:
        log_callback(
            "warning",
            f"Skipped {skipped_failed} failed and {skipped_no_scores} unscored results.",
        )

    submission = Submission(
        schema_version=_SUBMISSION_SCHEMA_VERSION,
        submitted_by=submitted_by,
        submitted_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        notes=notes,
        tbl_version=tbl_version,
        prompt_version=prompt_version,
        judge_id=judge_id,
        judge_seed=judge_seed,
        judge_temperature=judge_temperature,
        model_provider=provider,
        model_id=primary_model,
        results=sub_results,
    )
    return submission, primary_model


def cmd_submit(args: argparse.Namespace) -> int:
    """Convert a benchmark run JSON into a community submission file."""
    print_banner()

    submitted_by = args.by
    if not _GITHUB_LOGIN_RE.match(submitted_by):
        log_callback(
            "error",
            f"--by must be 'github:<username>' (got: {submitted_by})",
        )
        return 1

    source_path = Path(args.input).expanduser()
    if not source_path.is_file():
        log_callback("error", f"Input file not found: {source_path}")
        return 1

    try:
        with source_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        log_callback("error", f"Failed to parse {source_path}: {exc}")
        return 1

    # Accept either a raw BenchmarkRun JSON or a Submission already.
    if "results" in payload and "submission" in payload:
        log_callback("error", "Input already looks like a submission. Aborting.")
        return 1

    run = BenchmarkRun.from_dict(payload)

    config = BenchmarkConfig()
    output_dir = Path(args.output).expanduser() if args.output else config.paths.base_dir / "data" / "submissions"
    output_dir.mkdir(parents=True, exist_ok=True)

    judge_id = args.judge_id or run.evaluator_model or "unknown"
    tbl_version = args.tbl_version or _detect_tbl_version()
    prompt_version = args.prompt_version or "v1"
    provider = args.provider

    if provider not in CLOUD_PROVIDERS and provider != "ollama":
        log_callback("warning", f"Unknown provider '{provider}' — submission will be marked self-reported by aggregator.")

    try:
        submission, primary_model = _build_submission_from_run(
            run,
            submitted_by=submitted_by,
            provider=provider,
            judge_id=judge_id,
            tbl_version=tbl_version,
            prompt_version=prompt_version,
            notes=args.notes,
            judge_temperature=args.judge_temperature,
            judge_seed=args.judge_seed,
            source_lang_default=args.source_lang,
        )
    except ValueError as exc:
        log_callback("error", str(exc))
        return 1

    # Validate against the JSON schema before writing.
    submission_dict = submission.to_dict()
    validation_errors = _validate_submission_schema(submission_dict)
    if validation_errors and not validation_errors[0].startswith("WARN:"):
        log_callback("error", "Submission failed schema validation:")
        for err in validation_errors[:10]:
            print(f"  - {err}")
        return 1
    elif validation_errors:
        log_callback("warning", validation_errors[0])

    # Compose filename: <date>_<username>_<model-slug>.json
    date_part = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    user_part = submitted_by.split(":", 1)[1]
    model_slug = _slugify_for_filename(primary_model)
    filename = f"{date_part}_{user_part}_{model_slug}.json"
    output_path = output_dir / filename

    with output_path.open("w", encoding="utf-8") as f:
        f.write(submission.to_json())

    print(colored(f"\nSubmission written: {output_path}", Colors.GREEN))
    print(colored(f"  Model:    {primary_model} ({provider})", Colors.CYAN))
    print(colored(f"  Results:  {len(submission.results)}", Colors.CYAN))
    print(colored(f"  Judge:    {judge_id}", Colors.CYAN))
    print()
    print(colored("Next steps to open a Pull Request:", Colors.BOLD))
    branch = f"submit/{model_slug}"
    rel_path = output_path.relative_to(Path.cwd()) if output_path.is_absolute() else output_path
    print(f"  git checkout -b {branch}")
    print(f"  git add {rel_path}")
    print(f"  git commit -s -m \"submit: {primary_model} benchmark results\"")
    print(f"  gh pr create --title \"Submit {primary_model} benchmark results\" --body \"by {submitted_by}\"")
    return 0


def cmd_aggregate_submissions(args: argparse.Namespace) -> int:
    """Aggregate all submissions into a synthetic BenchmarkRun JSON."""
    print_banner()

    config = BenchmarkConfig()
    submissions_dir = (
        Path(args.submissions_dir).expanduser()
        if args.submissions_dir
        else config.paths.base_dir / "data" / "submissions"
    )

    aggregator = SubmissionAggregator(submissions_dir)
    submissions = aggregator.load()
    if not submissions:
        log_callback("warning", f"No submissions found in {submissions_dir}.")
        return 0 if args.allow_empty else 1

    run = aggregator.aggregate(submissions, run_id=args.run_id or "aggregated")

    output_path = (
        Path(args.output).expanduser()
        if args.output
        else config.paths.results_dir / f"{run.run_id}.json"
    )
    aggregator.write_run(run, output_path)

    print(colored(f"\nAggregated run written: {output_path}", Colors.GREEN))
    print(f"  Submissions:       {aggregator.stats.n_submissions}")
    print(f"  Raw results:       {aggregator.stats.n_results_in}")
    print(f"  Aggregated results:{aggregator.stats.n_results_out}")
    print(f"  Conflicts (>=2 obs): {aggregator.stats.n_conflicts}")
    print(f"  Models:            {len(run.models)}")
    print(f"  Languages:         {len(run.languages)}")
    return 0


def print_run_summary(run) -> None:
    """Print a summary of a benchmark run."""
    print(colored("\n" + "=" * 60, Colors.BOLD))
    print(colored(f"Benchmark Run: {run.run_id}", Colors.BOLD))
    print("=" * 60)

    print(f"Status: {colored(run.status, Colors.GREEN if run.status == 'completed' else Colors.YELLOW)}")
    print(f"Started: {run.started_at}")
    if run.completed_at:
        print(f"Completed: {run.completed_at}")
    print(f"Evaluator: {run.evaluator_model}")
    print()

    print(f"Models: {', '.join(run.models)}")
    print(f"Languages: {len(run.languages)} ({', '.join(run.languages[:7])}...)")
    print()

    print(f"Total translations: {run.total_completed}/{run.total_expected}")
    success_count = sum(1 for r in run.results if r.success)
    success_rate = (success_count / len(run.results) * 100) if run.results else 0
    print(f"Success rate: {success_rate:.1f}%")

    # Calculate average scores
    scores = [r.scores.overall for r in run.results if r.scores]
    if scores:
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        print(f"Scores: avg={avg_score:.1f}, min={min_score:.1f}, max={max_score:.1f}")

    print()


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="benchmark",
        description="TranslateBookWithLLM Benchmark System - Test translation quality across languages and models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick benchmark with Ollama (local models)
  python -m benchmark.cli run --openrouter-key YOUR_KEY

    # Quick benchmark with an OpenAI-compatible backend
    python -m benchmark.cli run --provider openai --openai-endpoint http://localhost:8080/v1 -m your-model

  # Quick benchmark with OpenRouter (cloud models)
  python -m benchmark.cli run --provider openrouter --openrouter-key YOUR_KEY

  # Full benchmark (all 40+ languages)
  python -m benchmark.cli run --full --openrouter-key YOUR_KEY

  # Specific Ollama models and languages
  python -m benchmark.cli run -m llama3:8b qwen2.5:14b -l fr de ja zh

  # Specific OpenRouter models
  python -m benchmark.cli run -p openrouter -m anthropic/claude-sonnet-4 openai/gpt-4o -l fr de ja

    # Specific OpenAI-compatible backend and models
    python -m benchmark.cli run -p openai --openai-endpoint http://localhost:8080/v1 -m qwen2.5-14b-instruct

  # Generate wiki pages
  python -m benchmark.cli wiki

  # List all runs
  python -m benchmark.cli list
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a benchmark")
    run_parser.add_argument(
        "-m", "--models",
        nargs="+",
        help="Models to benchmark. For Ollama: model names (e.g., llama3:8b). "
             "For OpenAI-compatible backends: model IDs (e.g., gpt-4o or local server model names). "
             "For OpenRouter: model IDs (e.g., anthropic/claude-sonnet-4). "
             "If not specified, auto-detects available models."
    )
    run_parser.add_argument(
        "-l", "--languages",
        nargs="+",
        help="Language codes to test (e.g., fr de ja zh). If not specified, uses quick test set."
    )
    run_parser.add_argument(
        "--full",
        action="store_true",
        help="Run full benchmark with all 40+ languages"
    )
    run_parser.add_argument(
        "-p", "--provider",
        choices=["ollama", "openai", "openrouter", "poe"],
        default="ollama",
        help="Translation provider: 'ollama' (local), 'openai' (OpenAI-compatible), "
             "'openrouter' (cloud), or 'poe' (Poe.com unified API)."
    )
    run_parser.add_argument(
        "--pairs",
        nargs="+",
        metavar="SRC:TGT",
        help="Explicit (source:target) language pairs, e.g. 'en:zh-Hans en:fr ja:en'. "
             "Overrides --languages and --full. Texts are filtered by `source_language`.",
    )
    run_parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Skip automatic LLM judge. Translations are saved with scores=None, "
             "ready for manual evaluation via scripts/dump_for_evaluation.py.",
    )
    run_parser.add_argument(
        "--openai-key",
        help="API key for OpenAI-compatible translation backends. Can also be set via OPENAI_API_KEY env var."
    )
    run_parser.add_argument(
        "--openai-endpoint",
        help="OpenAI-compatible chat completions endpoint or /v1 base URL. Can also be set via OPENAI_API_ENDPOINT env var."
    )
    run_parser.add_argument(
        "--openrouter-key",
        help="OpenRouter API key (for evaluation, and translation if using --provider openrouter). "
             "Can also be set via OPENROUTER_API_KEY env var."
    )
    run_parser.add_argument(
        "--evaluator-provider",
        choices=["openrouter", "poe"],
        default=DEFAULT_EVALUATOR_PROVIDER,
        help=f"Provider for evaluation (default: {DEFAULT_EVALUATOR_PROVIDER})"
    )
    run_parser.add_argument(
        "--evaluator",
        default=None,
        help=f"Model for evaluation (default: {DEFAULT_EVALUATOR_MODEL} for OpenRouter, "
             f"{DEFAULT_POE_EVALUATOR_MODEL} for Poe)"
    )
    run_parser.add_argument(
        "--poe-key",
        help="Poe API key (for evaluation if using --evaluator-provider poe). "
             "Can also be set via POE_API_KEY env var."
    )
    run_parser.add_argument(
        "--ollama-endpoint",
        help="Custom Ollama API endpoint"
    )
    run_parser.add_argument(
        "--resume",
        metavar="RUN_ID",
        help="Resume an interrupted run by ID"
    )
    run_parser.set_defaults(func=cmd_run)

    # Wiki command
    wiki_parser = subparsers.add_parser("wiki", help="Generate wiki pages from results")
    wiki_parser.add_argument(
        "run_id",
        nargs="?",
        help="Run ID to generate pages for. If not specified, uses latest run."
    )
    wiki_parser.set_defaults(func=cmd_wiki)

    # Wiki-publish command
    wiki_publish_parser = subparsers.add_parser(
        "wiki-publish",
        help="Generate and publish wiki pages to GitHub"
    )
    wiki_publish_parser.add_argument(
        "run_id",
        nargs="?",
        help="Run ID to publish. If not specified, uses latest run."
    )
    wiki_publish_parser.set_defaults(func=cmd_wiki_publish)

    # Merge command
    merge_parser = subparsers.add_parser(
        "merge",
        help="Merge multiple benchmark runs into one"
    )
    merge_parser.add_argument(
        "run_ids",
        nargs="+",
        help="Run IDs to merge (at least 2)"
    )
    merge_parser.add_argument(
        "-o", "--output",
        help="Custom ID for the merged run"
    )
    merge_parser.add_argument(
        "--publish",
        action="store_true",
        help="Regenerate and publish wiki after merging"
    )
    merge_parser.set_defaults(func=cmd_merge)

    # List command
    list_parser = subparsers.add_parser("list", help="List available benchmark runs")
    list_parser.set_defaults(func=cmd_list)

    # Models command
    models_parser = subparsers.add_parser("models", help="List available models for benchmarking")
    models_parser.add_argument(
        "-p", "--provider",
        choices=["ollama", "openai", "openrouter", "poe"],
        default="ollama",
        help="Provider to list models for (default: ollama)"
    )
    models_parser.add_argument(
        "--openai-key",
        help="API key for listing models from an OpenAI-compatible endpoint"
    )
    models_parser.add_argument(
        "--openai-endpoint",
        help="OpenAI-compatible endpoint to query for available models"
    )
    models_parser.add_argument(
        "--openrouter-key",
        help="OpenRouter API key (required for listing OpenRouter models)"
    )
    models_parser.add_argument(
        "--poe-key",
        help="Poe API key (overrides POE_API_KEY env)"
    )
    models_parser.add_argument(
        "--check",
        metavar="MODEL_ID",
        help="Validate that MODEL_ID exists on the provider. Exits 0 if found, "
             "1 if not. On miss, prints up to 10 close matches to stdout."
    )
    models_parser.set_defaults(func=cmd_models)

    # Show command
    show_parser = subparsers.add_parser("show", help="Show details of a benchmark run")
    show_parser.add_argument("run_id", help="Run ID to show")
    show_parser.add_argument(
        "-d", "--detailed",
        action="store_true",
        help="Show detailed statistics"
    )
    show_parser.set_defaults(func=cmd_show)

    # Export command
    export_parser = subparsers.add_parser("export", help="Export run results to CSV")
    export_parser.add_argument("run_id", help="Run ID to export")
    export_parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file path (default: benchmark_results/<run_id>.csv)"
    )
    export_parser.set_defaults(func=cmd_export)

    # Submit command
    submit_parser = subparsers.add_parser(
        "submit",
        help="Convert a benchmark run JSON into a community submission file",
    )
    submit_parser.add_argument(
        "input",
        help="Path to the benchmark run JSON to convert (e.g. benchmark_results/<run_id>.json)",
    )
    submit_parser.add_argument(
        "--by",
        required=True,
        help="GitHub identity, e.g. github:hydropix",
    )
    submit_parser.add_argument(
        "--provider",
        required=True,
        choices=sorted(CLOUD_PROVIDERS | {"ollama"}),
        help="Provider used to produce the translations.",
    )
    submit_parser.add_argument(
        "--judge-id",
        help="Judge model ID (defaults to the run's evaluator_model field).",
    )
    submit_parser.add_argument(
        "--judge-seed",
        type=int,
        help="Seed used by the judge, if any.",
    )
    submit_parser.add_argument(
        "--judge-temperature",
        type=float,
        help="Temperature used by the judge, if any.",
    )
    submit_parser.add_argument(
        "--tbl-version",
        help="TBL version label (defaults to git short SHA or 'dev').",
    )
    submit_parser.add_argument(
        "--prompt-version",
        default="v1",
        help="Prompt version label.",
    )
    submit_parser.add_argument(
        "--source-lang",
        default="en",
        help="Source language code for the texts (default: en).",
    )
    submit_parser.add_argument(
        "--notes",
        help="Optional free-text notes attached to the submission (<=2000 chars).",
    )
    submit_parser.add_argument(
        "-o", "--output",
        help="Destination directory (default: benchmark/data/submissions/).",
    )
    submit_parser.set_defaults(func=cmd_submit)

    # Aggregate-submissions command
    aggregate_parser = subparsers.add_parser(
        "aggregate-submissions",
        help="Merge all benchmark/data/submissions/*.json into a single run JSON",
    )
    aggregate_parser.add_argument(
        "--submissions-dir",
        help="Where to read submissions from (default: benchmark/data/submissions).",
    )
    aggregate_parser.add_argument(
        "--output",
        help="Where to write the aggregated run JSON (default: benchmark_results/<run_id>.json).",
    )
    aggregate_parser.add_argument(
        "--run-id",
        help="Run ID for the aggregated run (default: 'aggregated').",
    )
    aggregate_parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Exit 0 instead of 1 when no submissions are found.",
    )
    aggregate_parser.set_defaults(func=cmd_aggregate_submissions)

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a benchmark run")
    delete_parser.add_argument("run_id", help="Run ID to delete")
    delete_parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Delete without confirmation"
    )
    delete_parser.set_defaults(func=cmd_delete)

    return parser


def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
