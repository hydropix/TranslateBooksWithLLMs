"""
Command-line interface for text translation.

A thin adapter over the shared job runner (``src/core/job/runner.py``): it
parses the arguments into a ``JobConfig`` and supplies console sinks. The
verdict, glossary snapshot, refine chaining, checkpoints and resume are the
runner's, exactly as for the web server.
"""
import os
import argparse
import asyncio
import logging
import uuid

# Force UTF-8 stdio before anything prints, so emoji log lines (💬, ✅, ❌, ...)
# don't crash on Windows cp1252 consoles. See issue #184.
from src.utils.console import ensure_utf8_stdio
ensure_utf8_stdio()

# Reduce verbosity of httpx (avoid showing 400 errors during model detection)
logging.getLogger('httpx').setLevel(logging.WARNING)

from src.config import DEFAULT_MODEL, API_ENDPOINT, LLM_PROVIDER, OPENAI_API_ENDPOINT, NIM_API_ENDPOINT, GEMINI_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY, MISTRAL_API_KEY, DEEPSEEK_API_KEY, POE_API_KEY, NIM_API_KEY, DEFAULT_SOURCE_LANGUAGE, DEFAULT_TARGET_LANGUAGE, PARALLEL_TRANSLATIONS
from src.utils.unified_logger import setup_cli_logger, LogType
from src.tts.tts_config import TTS_ENABLED, TTS_VOICE, TTS_RATE, TTS_BITRATE, TTS_OUTPUT_FORMAT
from src.persistence.checkpoint_manager import CheckpointManager
from src.core.job import JobConfig, JobConfigError, JobResult, JobStatus
from src.core.job.runner import JobRunner, JobSinks
from src.core.llm.credentials import KEY_PROVIDERS


class CliJobSinks(JobSinks):
    """Runner sinks for the console: the unified CLI logger and its progress bar."""

    def __init__(self, logger):
        self.logger = logger
        self._log = logger.create_legacy_callback()

    def log(self, key, message="", data=None):
        self._log(key, message, data)

    def emit_stats(self, stats):
        total = stats.get('total_chunks', 0) or 0
        if total > 0:
            self.logger.update_progress(stats.get('completed_chunks', 0), total)

    def job_started(self, job, is_resume):
        details = {
            'target_lang': job.target_language,
            'file_type': job.file_type.upper(),
            'model': job.model,
            'input_file': job.input_path,
            'output_file': job.output_path,
            'api_endpoint': job.endpoint,
            'llm_provider': job.llm_provider,
        }
        if job.is_refine_only:
            self.logger.info("Refine-Only Started", LogType.TRANSLATION_START,
                             {**details, 'mode': 'refine-only'})
        else:
            self.logger.info("Translation Started", LogType.TRANSLATION_START,
                             {'source_lang': job.source_language, **details})

    def job_finished(self, job, result: JobResult, elapsed):
        if job.is_term_fix and result.detail is not None and not getattr(result.detail, 'changed', True):
            self.logger.warning("⚠️ Term fix: no occurrence matched; the output is identical to the input.")
        if result.status is JobStatus.COMPLETED:
            label = "Refine-Only Completed Successfully" if job.is_refine_only else "Translation Completed Successfully"
            details = {'output_file': result.output_path}
            if job.is_refine_only:
                details['mode'] = 'refine-only'
            self.logger.info(label, LogType.TRANSLATION_END, details)
            return
        label = "Refine-only" if job.is_refine_only else ("Term fix" if job.is_term_fix else "Translation")
        reason = (f"{label} did not complete ({result.status.value}): "
                  f"{result.error or 'one or more units failed or the output could not be written'}. "
                  f"Check the log above; '{result.output_path}' may be missing or partial.")
        self.logger.error(reason, LogType.ERROR_DETAIL, {'details': reason, 'input_file': job.input_path})

    def job_rate_limited(self, job, message):
        self.logger.error(message, LogType.ERROR_DETAIL, {'details': message, 'input_file': job.input_path})

    def job_failed(self, job, error):
        self.logger.error(f"Translation failed: {error}", LogType.ERROR_DETAIL,
                          {'details': error, 'input_file': job.input_path})


def _run_term_fix_scan(args, parser, logger) -> int:
    """`--fix-terms-scan`: count the matches, write nothing, call no LLM."""
    from src.core.adapters import scan_file
    from src.core.adapters.exceptions import UnsupportedFormatError
    from src.core.term_fix import RuleParseError, format_scan, load_rules_file

    try:
        rules = load_rules_file(args.fix_terms)
    except FileNotFoundError:
        parser.error(f"Rules file not found: {args.fix_terms}")
    except RuleParseError as e:
        parser.error(f"Invalid rules in {args.fix_terms}: {e}")
    logger.info(f"Term fix: {len(rules)} rule(s) loaded from {args.fix_terms}")
    try:
        for line in format_scan(asyncio.run(scan_file(args.input, rules))):
            logger.info(line)
    except (UnsupportedFormatError, FileNotFoundError) as e:
        logger.error(f"Term fix failed: {e}", LogType.ERROR_DETAIL)
        return 1
    return 0


def _check_term_fix_rules(job, parser, logger) -> None:
    """Fail fast on an unreadable rules file or a rule literal mode cannot apply."""
    from src.core.term_fix import RuleParseError, check_literal_rules, load_rules_file
    rules_file = job.term_fix['rules_file']
    try:
        rules = load_rules_file(rules_file)
    except FileNotFoundError:
        parser.error(f"Rules file not found: {rules_file}")
    except RuleParseError as e:
        parser.error(f"Invalid rules in {rules_file}: {e}")
    if job.term_fix.get('mode') == 'literal':
        try:
            check_literal_rules(rules)
        except ValueError as e:
            parser.error(f"Term fix: {e}")
    logger.info(f"Term fix: {len(rules)} rule(s) loaded from {rules_file}")


def _resume_job(args, checkpoint_manager) -> JobConfig:
    """The stored job `--resume ID` continues, with this run's credentials."""
    data = checkpoint_manager.load_checkpoint(args.resume)
    if not data:
        raise JobConfigError(f"No checkpoint found for job '{args.resume}'.")
    stored = dict(data['job']['config'] or {})
    # Configs stored before v1.6.0 lack file_type; the job row always has it.
    if not stored.get('file_type') and data['job'].get('file_type'):
        stored['file_type'] = data['job']['file_type']
    input_path = (stored.get('preserved_input_path') or stored.get('file_path')
                  or stored.get('input_file_path'))
    if not input_path or not os.path.exists(input_path):
        raise JobConfigError(f"The input file of job '{args.resume}' is no longer available.")
    job = JobConfig.from_persisted(stored, input_path=input_path)
    provider = job.llm_provider
    cli_key = getattr(args, f'{provider}_api_key', None) if provider in KEY_PROVIDERS else None
    if cli_key:
        job = job.with_changes(provider=job.provider.with_changes(api_key=cli_key))
    return job


def _cli_glossary_store():
    """Open the glossary database only when a job selects --glossary-id."""
    from src.core.glossary import GlossaryStore
    return GlossaryStore()


def run_cli_job(job: JobConfig, logger, *, translation_id=None, checkpoint_manager=None,
                is_resume=False, engines=None) -> JobResult:
    """Run a job from the command line and return its result."""
    checkpoint_manager = checkpoint_manager or CheckpointManager()
    translation_id = translation_id or f"cli_{uuid.uuid4().hex[:8]}"
    runner = JobRunner(job, translation_id=translation_id, checkpoint_manager=checkpoint_manager,
                       sinks=CliJobSinks(logger), is_resume=is_resume, glossary_store=_cli_glossary_store,
                       temp_dir=job.output_dir or None, engines=engines)
    result = asyncio.run(runner.run())
    nothing_translated = bool(result.verdict and result.verdict.nothing_translated)
    if (result.status in (JobStatus.RATE_LIMITED, JobStatus.INTERRUPTED, JobStatus.PARTIAL)
            or nothing_translated):
        logger.info(f"Resume this job later with: python translate.py --resume {translation_id}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Translate a text, Markdown, EPUB, SRT, PGN, DOCX or PDF file using an LLM.",
        epilog="Tip: any --*_api_key flag also accepts comma-separated keys "
               "(e.g. --gemini_api_key key1,key2,key3) for automatic rotation "
               "on HTTP 429 — useful to chain free-tier accounts.",
    )
    parser.add_argument("-i", "--input", default=None, help="Path to the input file (text, Markdown, EPUB, SRT, PGN, DOCX or PDF). Required unless --resume is given.")
    parser.add_argument("-o", "--output", default=None, help="Path to the output file. If not specified, uses input filename with suffix.")
    parser.add_argument("--resume", default=None, metavar="JOB_ID", help="Resume an interrupted, rate-limited or partial CLI job from its checkpoint (the id is printed when the job stops).")
    parser.add_argument("-sl", "--source_lang", default=DEFAULT_SOURCE_LANGUAGE, help=f"Source language (default: {DEFAULT_SOURCE_LANGUAGE}).")
    parser.add_argument("-tl", "--target_lang", default=DEFAULT_TARGET_LANGUAGE, help=f"Target language (default: {DEFAULT_TARGET_LANGUAGE}).")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"LLM model (default: {DEFAULT_MODEL}).")
    parser.add_argument("--api_endpoint", default=None, help=f"API endpoint for Ollama or OpenAI-compatible servers (llama.cpp, LM Studio, vLLM, etc.). Overrides --openai_api_endpoint / --nim_api_endpoint when given (default: {API_ENDPOINT} for Ollama).")
    parser.add_argument("--provider", default=LLM_PROVIDER, choices=["ollama", "gemini", "openai", "openrouter", "mistral", "deepseek", "poe", "nim", "litellm"], help=f"LLM provider (default: {LLM_PROVIDER}). Use 'openai' for any OpenAI-compatible server. Use 'litellm' to reach 100+ providers via a provider-prefixed model name (e.g. anthropic/claude-sonnet-4-6); keys are read from each provider's native env var (OPENAI_API_KEY, ANTHROPIC_API_KEY, ...).")
    parser.add_argument("--gemini_api_key", default=GEMINI_API_KEY, help="Google Gemini API key (required if using gemini provider).")
    parser.add_argument("--openai_api_key", default=OPENAI_API_KEY, help="OpenAI API key (required for OpenAI cloud, not needed for local servers).")
    parser.add_argument("--openrouter_api_key", default=OPENROUTER_API_KEY, help="OpenRouter API key (required if using openrouter provider).")
    parser.add_argument("--mistral_api_key", default=MISTRAL_API_KEY, help="Mistral API key (required if using mistral provider).")
    parser.add_argument("--deepseek_api_key", default=DEEPSEEK_API_KEY, help="DeepSeek API key (required if using deepseek provider).")
    parser.add_argument("--poe_api_key", default=POE_API_KEY, help="Poe API key (required if using poe provider). Get your key at https://poe.com/api_key")
    parser.add_argument("--nim_api_key", default=NIM_API_KEY, help="NVIDIA NIM API key (required if using nim provider). Get your key at https://build.nvidia.com/")
    parser.add_argument("--openai_api_endpoint", default=OPENAI_API_ENDPOINT, help="Endpoint used by the openai provider when --api_endpoint is not given (default: OPENAI_API_ENDPOINT from .env).")
    parser.add_argument("--nim_api_endpoint", default=NIM_API_ENDPOINT, help="Endpoint used by the nim provider when --api_endpoint is not given (default: NIM_API_ENDPOINT from .env).")
    parser.add_argument("--parallel", type=int, default=PARALLEL_TRANSLATIONS, metavar="N", help=f"Number of chunks translated concurrently (default: {PARALLEL_TRANSLATIONS}). Only cloud providers benefit; local providers (Ollama) are forced to 1. Values > 1 drop cross-chunk context chaining.")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output.")

    prompt_group = parser.add_argument_group('Prompt Options', 'Optional instructions to include in the translation prompt')
    prompt_group.add_argument("--text-cleanup", action="store_true", help="Enable OCR/typographic cleanup (fix broken lines, spacing, punctuation).")
    prompt_group.add_argument("--refine", action="store_true", help="Run a refinement pass on the translated output to polish quality and literary style (TXT, SRT, EPUB, DOCX).")
    prompt_group.add_argument("--refine-only", action="store_true", dest="refine_only", help="Run ONLY a refinement pass on an already-translated file (skips the translation phase). The input file is assumed to already be in the target language.")
    prompt_group.add_argument("--glossary", default=None, help="Path to a glossary file (.json or .csv) injected per-chunk to keep entity translations consistent.")
    prompt_group.add_argument("--glossary-id", type=int, default=None, dest="glossary_id", help="ID of a glossary saved from the web interface (glossaries.db), used like --glossary. Cannot be combined with --glossary.")
    prompt_group.add_argument("--custom-instruction", default=None, dest="custom_instruction", metavar="FILE", help="File name (with its .yaml, .yml or .txt extension) of a custom instruction preset in the Custom_Instructions folder, as managed from the web interface.")
    prompt_group.add_argument(
        "--auto-glossary", action="store_true", dest="auto_glossary",
        help="Derive a throwaway glossary from the input document with one extra LLM call "
             "before translating (no file, nothing saved). Ignored when --glossary is given "
             "or with --refine-only.")
    prompt_group.add_argument(
        "--auto-style", action="store_true", dest="auto_style",
        help="Derive style instructions from the input document with one extra LLM call "
             "before translating (no preset file, nothing saved).")

    term_fix_group = parser.add_argument_group(
        'Term Fix', 'Replace wrong renderings of a term in an already-translated file')
    term_fix_group.add_argument(
        "--fix-terms", default=None, dest="fix_terms", metavar="RULES",
        help="Path to correction rules (.txt: one 'wrong|other -> correct' per line, "
             "or .json). Rewrites only the matched occurrences of the input file "
             "(TXT, SRT, EPUB, DOCX) and writes a '.term-fix.md' change report next to "
             "the output. -tl names the language the file is written in.")
    term_fix_group.add_argument(
        "--fix-terms-mode", choices=["smart", "literal"], default="smart", dest="fix_terms_mode",
        help="smart (default): the LLM rewrites only the sentences holding a match, so the "
             "correct term is inflected and agreement fixed. literal: plain case-preserving "
             "replacement, no LLM call ('*' wildcards are refused).")
    term_fix_group.add_argument(
        "--fix-terms-scan", action="store_true", dest="fix_terms_scan",
        help="With --fix-terms: only count the matches per rule and show sample "
             "sentences. Nothing is written, no LLM call.")

    tts_group = parser.add_argument_group('TTS Options', 'Text-to-Speech audio generation')
    tts_group.add_argument("--tts", action="store_true", default=TTS_ENABLED, help="Generate audio from translated text using Edge-TTS.")
    tts_group.add_argument("--tts-voice", default=TTS_VOICE, help="TTS voice name (auto-selected based on target language if not specified).")
    tts_group.add_argument("--tts-rate", default=TTS_RATE, help="TTS speech rate adjustment, e.g. '+10%%' or '-20%%' (default: %(default)s).")
    tts_group.add_argument("--tts-bitrate", default=TTS_BITRATE, help="Audio bitrate for encoding, e.g. '64k', '96k' (default: %(default)s).")
    tts_group.add_argument("--tts-format", default=TTS_OUTPUT_FORMAT, choices=["opus", "mp3"], help="Audio output format (default: %(default)s).")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = setup_cli_logger(enable_colors=not args.no_color)

    if args.resume:
        checkpoint_manager = CheckpointManager()
        try:
            job = _resume_job(args, checkpoint_manager)
        except JobConfigError as e:
            parser.error(str(e))
        return run_cli_job(job, logger, translation_id=args.resume,
                           checkpoint_manager=checkpoint_manager, is_resume=True).exit_code

    if not args.input:
        parser.error("the following arguments are required: -i/--input (or --resume JOB_ID)")
    if args.fix_terms_scan:
        if not args.fix_terms:
            parser.error("--fix-terms-scan requires --fix-terms RULES")
        return _run_term_fix_scan(args, parser, logger)

    try:
        job = JobConfig.from_cli_args(args)
    except JobConfigError as e:
        parser.error(str(e))

    # If no .env was found, show the *effective* settings (issue #187).
    from src.config import warn_env_config_missing, PORT
    warn_env_config_missing(provider=job.llm_provider, api_endpoint=job.endpoint,
                            model=job.model, port=PORT)
    for notice in job.notices:
        if notice.startswith("⚠️"):
            logger.warning(notice)
        else:
            logger.info(notice)
    if job.is_term_fix:
        _check_term_fix_rules(job, parser, logger)
    return run_cli_job(job, logger).exit_code


if __name__ == "__main__":
    raise SystemExit(main())
