"""
Command-line interface for text translation
"""
import os
import argparse
import asyncio
import logging

# Force UTF-8 stdio before anything prints, so emoji log lines (💬, ✅, ❌, ...)
# don't crash on Windows cp1252 consoles. See issue #184.
from src.utils.console import ensure_utf8_stdio
ensure_utf8_stdio()

# Reduce verbosity of httpx (avoid showing 400 errors during model detection)
logging.getLogger('httpx').setLevel(logging.WARNING)

from src.config import DEFAULT_MODEL, API_ENDPOINT, LLM_PROVIDER, GEMINI_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY, MISTRAL_API_KEY, DEEPSEEK_API_KEY, POE_API_KEY, NIM_API_KEY, ANTHROPIC_API_KEY, XAI_API_KEY, OPENCODE_API_KEY, OPENCODE_GO_API_KEY, OLLAMA_CLOUD_API_KEY, DEFAULT_SOURCE_LANGUAGE, DEFAULT_TARGET_LANGUAGE, PARALLEL_TRANSLATIONS
from src.utils.file_utils import get_unique_output_path, generate_tts_for_translation
from src.utils.unified_logger import setup_cli_logger, LogType
from src.tts.tts_config import TTSConfig, TTS_ENABLED, TTS_VOICE, TTS_RATE, TTS_BITRATE, TTS_OUTPUT_FORMAT
from src.persistence.checkpoint_manager import CheckpointManager
from src.core.adapters import translate_file, refine_file
from src.core import auto_prep
from src.core.auto_prep import build_auto_prompt_options
from src.core.llm_client import create_llm_client
from src.utils.notifier import notify, EVENT_SUCCESS, EVENT_FAILURE
import time
import uuid


def _apply_cli_auto_prep(args, prompt_options, logger) -> None:
    """Merge auto glossary/style into `prompt_options` (in place). Never raises.

    Mirrors the web path's auto mode (see plan/PLAN_AutoGlossaryStyle.md,
    Phase 4) for the CLI: `--auto-glossary` and `--auto-style` each derive a
    throwaway glossary/style block from the input document with one extra LLM
    call, and merge the result straight into `prompt_options` — nothing is
    saved, nothing is reviewed.

    Precedence, each logged as a warning when a flag is dropped:
      - `--auto-glossary` + `--glossary`     -> auto-glossary skipped
        ("⚠️ --auto-glossary ignored: --glossary was provided.")
      - `--auto-glossary` + `--refine-only`  -> auto-glossary skipped
        ("⚠️ --auto-glossary ignored in --refine-only mode.")
    `--auto-style` is honoured in every mode, including `--refine-only`: the
    refinement block reaches `refine_file` through the same `prompt_options`.

    Language handling mirrors decision D7 of the plan: in `--refine-only`
    mode the input is already in the target language, so
    `source_language := args.target_lang` for the style pass.

    Event-loop discipline: this helper owns its own `asyncio.run(...)`, fully
    closed before the translation starts. The `LLMClient` is created *and*
    closed inside that single coroutine — an `LLMClient` built in one event
    loop and closed in another would leak httpx state.

    The whole body is wrapped in try/except so a failure here only logs a
    warning and leaves `prompt_options` untouched; it can never make the
    translation job fail.
    """
    try:
        want_glossary = bool(getattr(args, "auto_glossary", False))
        want_style = bool(getattr(args, "auto_style", False))

        if want_glossary and getattr(args, "glossary", None):
            logger.warning("⚠️ --auto-glossary ignored: --glossary was provided.")
            want_glossary = False

        if want_glossary and getattr(args, "refine_only", False):
            logger.warning("⚠️ --auto-glossary ignored in --refine-only mode.")
            want_glossary = False

        if not want_glossary and not want_style:
            return

        source_text = auto_prep.extract_source_text(file_path=args.input)
        if not source_text or not source_text.strip():
            logger.warning(
                "⚠️ Auto mode: could not read the source document — "
                "translating without an auto glossary or style."
            )
            return

        refine_only = bool(getattr(args, "refine_only", False))
        source_language = args.target_lang if refine_only else args.source_lang
        target_language = args.target_lang

        async def _run():
            client = create_llm_client(
                args.provider,
                args.gemini_api_key,
                args.api_endpoint,
                args.model,
                openai_api_key=args.openai_api_key,
                openrouter_api_key=args.openrouter_api_key,
                mistral_api_key=args.mistral_api_key,
                deepseek_api_key=args.deepseek_api_key,
                poe_api_key=args.poe_api_key,
                nim_api_key=args.nim_api_key,
                anthropic_api_key=getattr(args, 'anthropic_api_key', None),
                xai_api_key=getattr(args, 'xai_api_key', None),
                opencode_api_key=getattr(args, 'opencode_api_key', None),
                opencodego_api_key=getattr(args, 'opencodego_api_key', None),
                ollamacloud_api_key=getattr(args, 'ollamacloud_api_key', None),
                context_window=auto_prep.AUTO_PREP_CONTEXT_WINDOW,
            )
            if client is None:
                logger.warning(
                    f"⚠️ Auto mode: unknown LLM provider '{args.provider}' — "
                    "translating without an auto glossary or style."
                )
                return {}
            try:
                return await build_auto_prompt_options(
                    source_text=source_text,
                    source_language=source_language,
                    target_language=target_language,
                    want_glossary=want_glossary,
                    want_style=want_style,
                    llm_client=client,
                    log=lambda key, msg: logger.info(msg),
                )
            finally:
                try:
                    await client.close()
                except Exception as close_exc:
                    logger.debug(f"auto prep client close failed: {close_exc}")

        fragment = asyncio.run(_run())
        if fragment:
            prompt_options.update(fragment)
    except Exception as exc:
        logger.warning(f"⚠️ Auto mode failed: {exc} — translating without it.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate a text, EPUB or SRT file using an LLM.",
        epilog="Tip: any --*_api_key flag also accepts comma-separated keys "
               "(e.g. --gemini_api_key key1,key2,key3) for automatic rotation "
               "on HTTP 429 — useful to chain free-tier accounts.",
    )
    parser.add_argument("-i", "--input", required=True, help="Path to the input file (text, EPUB, or SRT).")
    parser.add_argument("-o", "--output", default=None, help="Path to the output file. If not specified, uses input filename with suffix.")
    parser.add_argument("-sl", "--source_lang", default=DEFAULT_SOURCE_LANGUAGE, help=f"Source language (default: {DEFAULT_SOURCE_LANGUAGE}).")
    parser.add_argument("-tl", "--target_lang", default=DEFAULT_TARGET_LANGUAGE, help=f"Target language (default: {DEFAULT_TARGET_LANGUAGE}).")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"LLM model (default: {DEFAULT_MODEL}).")
    parser.add_argument("--api_endpoint", default=API_ENDPOINT, help=f"API endpoint for Ollama or OpenAI-compatible servers (llama.cpp, LM Studio, vLLM, etc.) (default: {API_ENDPOINT}).")
    parser.add_argument("--provider", default=LLM_PROVIDER, choices=["ollama", "gemini", "openai", "openrouter", "mistral", "deepseek", "poe", "nim", "anthropic", "xai", "opencode", "opencodego", "ollamacloud", "chatgpt", "litellm"], help=f"LLM provider (default: {LLM_PROVIDER}). Use 'openai' for any OpenAI-compatible server. Use 'litellm' to reach 100+ providers via a provider-prefixed model name (e.g. anthropic/claude-sonnet-4-6); keys are read from each provider's native env var (OPENAI_API_KEY, ANTHROPIC_API_KEY, ...).")
    parser.add_argument("--gemini_api_key", default=GEMINI_API_KEY, help="Google Gemini API key (required if using gemini provider).")
    parser.add_argument("--openai_api_key", default=OPENAI_API_KEY, help="OpenAI API key (required for OpenAI cloud, not needed for local servers).")
    parser.add_argument("--openrouter_api_key", default=OPENROUTER_API_KEY, help="OpenRouter API key (required if using openrouter provider).")
    parser.add_argument("--mistral_api_key", default=MISTRAL_API_KEY, help="Mistral API key (required if using mistral provider).")
    parser.add_argument("--deepseek_api_key", default=DEEPSEEK_API_KEY, help="DeepSeek API key (required if using deepseek provider).")
    parser.add_argument("--poe_api_key", default=POE_API_KEY, help="Poe API key (required if using poe provider). Get your key at https://poe.com/api_key")
    parser.add_argument("--nim_api_key", default=NIM_API_KEY, help="NVIDIA NIM API key (required if using nim provider). Get your key at https://build.nvidia.com/")
    parser.add_argument("--anthropic_api_key", default=ANTHROPIC_API_KEY, help="Anthropic API key (required if using anthropic provider).")
    parser.add_argument("--xai_api_key", default=XAI_API_KEY, help="xAI API key (required if using xai provider).")
    parser.add_argument("--opencode_api_key", default=OPENCODE_API_KEY, help="OpenCode Zen API key (required if using opencode provider).")
    parser.add_argument("--opencodego_api_key", default=OPENCODE_GO_API_KEY or OPENCODE_API_KEY, help="OpenCode Go API key (falls back to --opencode_api_key / OPENCODE_API_KEY).")
    parser.add_argument("--ollamacloud_api_key", default=OLLAMA_CLOUD_API_KEY, help="Ollama Cloud API key (required if using ollamacloud provider). Falls back to OLLAMA_API_KEY.")
    parser.add_argument("--parallel", type=int, default=PARALLEL_TRANSLATIONS, metavar="N", help=f"Number of chunks translated concurrently (default: {PARALLEL_TRANSLATIONS}). Only cloud providers benefit; local providers (Ollama) are forced to 1. Values > 1 drop cross-chunk context chaining.")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output.")

    # Prompt options (optional system prompt instructions)
    prompt_group = parser.add_argument_group('Prompt Options', 'Optional instructions to include in the translation prompt')
    prompt_group.add_argument("--text-cleanup", action="store_true", help="Enable OCR/typographic cleanup (fix broken lines, spacing, punctuation).")
    prompt_group.add_argument("--refine", action="store_true", help="Enable refinement pass: runs a second pass to polish translation quality and literary style.")
    prompt_group.add_argument("--refine-only", action="store_true", dest="refine_only", help="Run ONLY a refinement pass on an already-translated file (skips the translation phase). The input file is assumed to already be in the target language.")
    prompt_group.add_argument("--glossary", default=None, help="Path to a glossary file (.json or .csv) injected per-chunk to keep entity translations consistent.")
    prompt_group.add_argument(
        "--auto-glossary", action="store_true", dest="auto_glossary",
        help="Derive a throwaway glossary from the input document with one extra LLM call "
             "before translating (no file, nothing saved). Ignored when --glossary is given "
             "or with --refine-only.")
    prompt_group.add_argument(
        "--auto-style", action="store_true", dest="auto_style",
        help="Derive style instructions from the input document with one extra LLM call "
             "before translating (no preset file, nothing saved).")

    # TTS (Text-to-Speech) arguments
    tts_group = parser.add_argument_group('TTS Options', 'Text-to-Speech audio generation')
    tts_group.add_argument("--tts", action="store_true", default=TTS_ENABLED, help="Generate audio from translated text using Edge-TTS.")
    tts_group.add_argument("--tts-voice", default=TTS_VOICE, help="TTS voice name (auto-selected based on target language if not specified).")
    tts_group.add_argument("--tts-rate", default=TTS_RATE, help="TTS speech rate adjustment, e.g. '+10%%' or '-20%%' (default: %(default)s).")
    tts_group.add_argument("--tts-bitrate", default=TTS_BITRATE, help="Audio bitrate for encoding, e.g. '64k', '96k' (default: %(default)s).")
    tts_group.add_argument("--tts-format", default=TTS_OUTPUT_FORMAT, choices=["opus", "mp3"], help="Audio output format (default: %(default)s).")

    args = parser.parse_args()

    # Auto-select default model based on provider if not explicitly set
    from src.config import NIM_MODEL, MISTRAL_MODEL, DEEPSEEK_MODEL, POE_MODEL, OPENROUTER_MODEL, GEMINI_MODEL, LITELLM_MODEL, ANTHROPIC_MODEL, XAI_MODEL, OPENCODE_MODEL, OPENCODE_GO_MODEL, OLLAMA_CLOUD_MODEL, CHATGPT_MODEL
    if args.model == DEFAULT_MODEL:
        if args.provider == "nim" and NIM_MODEL:
            args.model = NIM_MODEL
        elif args.provider == "litellm" and LITELLM_MODEL:
            args.model = LITELLM_MODEL
        elif args.provider == "anthropic" and ANTHROPIC_MODEL:
            args.model = ANTHROPIC_MODEL
        elif args.provider == "xai" and XAI_MODEL:
            args.model = XAI_MODEL
        elif args.provider == "opencode" and OPENCODE_MODEL:
            args.model = OPENCODE_MODEL
        elif args.provider == "opencodego" and OPENCODE_GO_MODEL:
            args.model = OPENCODE_GO_MODEL
        elif args.provider == "ollamacloud" and OLLAMA_CLOUD_MODEL:
            args.model = OLLAMA_CLOUD_MODEL
        elif args.provider == "chatgpt" and CHATGPT_MODEL:
            args.model = CHATGPT_MODEL
        elif args.provider == "mistral" and MISTRAL_MODEL:
            args.model = MISTRAL_MODEL
        elif args.provider == "deepseek" and DEEPSEEK_MODEL:
            args.model = DEEPSEEK_MODEL
        elif args.provider == "poe" and POE_MODEL:
            args.model = POE_MODEL
        elif args.provider == "openrouter" and OPENROUTER_MODEL:
            args.model = OPENROUTER_MODEL
        elif args.provider == "gemini" and GEMINI_MODEL:
            args.model = GEMINI_MODEL

    # If no .env was found, surface the *effective* settings now (after argparse)
    # so the warning box shows the real CLI arguments rather than the import-time
    # defaults (issue #187). No-op when a .env exists or running as executable.
    from src.config import warn_env_config_missing, PORT
    warn_env_config_missing(
        provider=args.provider,
        api_endpoint=args.api_endpoint,
        model=args.model,
        port=PORT,
    )

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        output_ext = ext
        if args.input.lower().endswith('.epub'):
            output_ext = '.epub'
        elif args.input.lower().endswith('.srt'):
            output_ext = '.srt'
        if args.refine_only:
            args.output = f"{base} (refined){output_ext}"
        else:
            args.output = f"{base} ({args.target_lang}){output_ext}"

    # Ensure output path is unique (add number suffix if file exists)
    args.output = get_unique_output_path(args.output)

    # Determine file type
    if args.input.lower().endswith('.epub'):
        file_type = "EPUB"
    elif args.input.lower().endswith('.srt'):
        file_type = "SRT"
    else:
        file_type = "TEXT"
    
    # Setup unified logger
    logger = setup_cli_logger(enable_colors=not args.no_color)
    
    # Validate API keys for providers
    if args.provider == "gemini" and not args.gemini_api_key:
        parser.error("--gemini_api_key is required when using gemini provider")
    # Note: OpenAI API key is optional for local servers (llama.cpp, LM Studio, vLLM, etc.)
    # Only required for OpenAI cloud API
    if args.provider == "openrouter" and not args.openrouter_api_key:
        parser.error("--openrouter_api_key is required when using openrouter provider")
    if args.provider == "mistral" and not args.mistral_api_key:
        parser.error("--mistral_api_key is required when using mistral provider")
    if args.provider == "deepseek" and not args.deepseek_api_key:
        parser.error("--deepseek_api_key is required when using deepseek provider")
    if args.provider == "poe" and not args.poe_api_key:
        parser.error("--poe_api_key is required when using poe provider. Get your key at https://poe.com/api_key")
    if args.provider == "nim" and not args.nim_api_key:
        parser.error("--nim_api_key is required when using nim provider. Get your key at https://build.nvidia.com/")
    if args.provider == "anthropic" and not getattr(args, 'anthropic_api_key', None):
        parser.error("--anthropic_api_key is required when using anthropic provider")
    if args.provider == "xai" and not getattr(args, 'xai_api_key', None):
        parser.error("--xai_api_key is required when using xai provider")
    if args.provider == "opencode" and not getattr(args, 'opencode_api_key', None):
        parser.error("--opencode_api_key is required when using opencode provider")
    if args.provider == "opencodego" and not (
        getattr(args, 'opencodego_api_key', None) or getattr(args, 'opencode_api_key', None)
    ):
        parser.error("--opencodego_api_key or --opencode_api_key is required when using opencodego provider")
    if args.provider == "ollamacloud" and not getattr(args, 'ollamacloud_api_key', None):
        parser.error("--ollamacloud_api_key is required when using ollamacloud provider")
    if args.provider == "chatgpt":
        from src.core.llm.chatgpt_oauth import status_payload
        if not status_payload().get("signed_in"):
            parser.error("ChatGPT is not signed in. Use Sign in with ChatGPT in the web UI first.")
    # LiteLLM needs a provider-prefixed model name; the default Ollama model
    # won't route. Keys come from each provider's native env var, so we only
    # guard the model here rather than an API key.
    if args.provider == "litellm" and args.model == DEFAULT_MODEL:
        parser.error("litellm provider requires a provider-prefixed model. "
                     "Set LITELLM_MODEL in .env or pass -m, e.g. "
                     "-m anthropic/claude-sonnet-4-6")

    # Refinement is monolingual: mismatched source/target almost always
    # means the user forgot. Warn but proceed using target_lang.
    if args.refine_only and args.source_lang != args.target_lang:
        logger.warning(
            f"⚠️ --refine-only: source language ({args.source_lang}) differs from "
            f"target language ({args.target_lang}). Refinement is monolingual; "
            f"source_lang will be ignored and the file will be polished as "
            f"{args.target_lang}."
        )

    if args.refine_only:
        logger.info("Refine-Only Started", LogType.TRANSLATION_START, {
            'target_lang': args.target_lang,
            'file_type': file_type,
            'model': args.model,
            'input_file': args.input,
            'output_file': args.output,
            'api_endpoint': args.api_endpoint,
            'llm_provider': args.provider,
            'mode': 'refine-only',
        })
    else:
        logger.info("Translation Started", LogType.TRANSLATION_START, {
            'source_lang': args.source_lang,
            'target_lang': args.target_lang,
            'file_type': file_type,
            'model': args.model,
            'input_file': args.input,
            'output_file': args.output,
            'api_endpoint': args.api_endpoint,
            'llm_provider': args.provider
        })

    # Create legacy callback for backward compatibility
    log_callback = logger.create_legacy_callback()

    # Create stats callback to update logger progress
    def stats_callback(stats: dict):
        completed = stats.get('completed_chunks', 0)
        total = stats.get('total_chunks', 0)
        if total > 0:
            logger.update_progress(completed, total)

    # Build prompt_options from CLI arguments
    # Technical content protection is now always enabled.
    # In refine-only mode the refinement pass is implicit, so we force the
    # `refine` flag off to avoid double-counting in the progress tracker.
    prompt_options = {
        'preserve_technical_content': True,
        'text_cleanup': args.text_cleanup,
        'refine': args.refine and not args.refine_only,
    }

    # Load glossary file (JSON or CSV) into prompt_options
    if args.glossary:
        try:
            from src.core.glossary.cli_loader import load_glossary_from_file
            glossary_terms, glossary_metadata = load_glossary_from_file(args.glossary)
            if glossary_terms:
                prompt_options['glossary_terms'] = glossary_terms
                if glossary_metadata:
                    prompt_options['glossary_term_metadata'] = glossary_metadata
                logger.info(f"Glossary loaded: {len(glossary_terms)} terms from {args.glossary}")
            else:
                logger.warning(f"Glossary file {args.glossary} contained no usable entries")
        except Exception as e:
            parser.error(f"Failed to load glossary {args.glossary}: {e}")

    # Auto glossary / auto style: derive throwaway prompt_options from the
    # input document itself, with one extra LLM call per requested pass.
    _apply_cli_auto_prep(args, prompt_options, logger)

    start_time = time.time()
    try:
        # Create checkpoint manager for resume capability
        checkpoint_manager = CheckpointManager()

        # Generate unique translation ID
        translation_id = f"cli_{uuid.uuid4().hex[:8]}"

        if args.refine_only:
            asyncio.run(refine_file(
                input_filepath=args.input,
                output_filepath=args.output,
                target_language=args.target_lang,
                model_name=args.model,
                llm_provider=args.provider,
                checkpoint_manager=checkpoint_manager,
                translation_id=translation_id,
                log_callback=log_callback,
                stats_callback=stats_callback,
                check_interruption_callback=None,
                llm_api_endpoint=args.api_endpoint,
                gemini_api_key=args.gemini_api_key,
                openai_api_key=args.openai_api_key,
                openrouter_api_key=args.openrouter_api_key,
                mistral_api_key=args.mistral_api_key,
                deepseek_api_key=args.deepseek_api_key,
                poe_api_key=args.poe_api_key,
                nim_api_key=args.nim_api_key,
                anthropic_api_key=getattr(args, 'anthropic_api_key', None),
                xai_api_key=getattr(args, 'xai_api_key', None),
                opencode_api_key=getattr(args, 'opencode_api_key', None),
                opencodego_api_key=getattr(args, 'opencodego_api_key', None),
                ollamacloud_api_key=getattr(args, 'ollamacloud_api_key', None),
                prompt_options=prompt_options,
            ))
            logger.info("Refine-Only Completed Successfully", LogType.TRANSLATION_END, {
                'output_file': args.output,
                'mode': 'refine-only',
            })
        else:
            asyncio.run(translate_file(
                input_filepath=args.input,
                output_filepath=args.output,
                source_language=args.source_lang,
                target_language=args.target_lang,
                model_name=args.model,
                llm_provider=args.provider,
                checkpoint_manager=checkpoint_manager,
                translation_id=translation_id,
                log_callback=log_callback,
                stats_callback=stats_callback,
                check_interruption_callback=None,
                llm_api_endpoint=args.api_endpoint,
                gemini_api_key=args.gemini_api_key,
                openai_api_key=args.openai_api_key,
                openrouter_api_key=args.openrouter_api_key,
                mistral_api_key=args.mistral_api_key,
                deepseek_api_key=args.deepseek_api_key,
                poe_api_key=args.poe_api_key,
                nim_api_key=args.nim_api_key,
                anthropic_api_key=getattr(args, 'anthropic_api_key', None),
                xai_api_key=getattr(args, 'xai_api_key', None),
                opencode_api_key=getattr(args, 'opencode_api_key', None),
                opencodego_api_key=getattr(args, 'opencodego_api_key', None),
                ollamacloud_api_key=getattr(args, 'ollamacloud_api_key', None),
                prompt_options=prompt_options,
                parallel_workers=args.parallel
            ))

            logger.info("Translation Completed Successfully", LogType.TRANSLATION_END, {
                'output_file': args.output
            })

        notify(EVENT_SUCCESS, {
            'file': args.input,
            'output': args.output,
            'duration_seconds': time.time() - start_time,
            'provider': args.provider,
            'model': args.model,
            'source_lang': None if args.refine_only else args.source_lang,
            'target_lang': args.target_lang,
            'mode': 'refine-only' if args.refine_only else 'translate',
        })

        # TTS Generation (if enabled)
        if args.tts:
            logger.info("Starting TTS Generation", LogType.INFO, {
                'voice': args.tts_voice or 'auto',
                'rate': args.tts_rate,
                'format': args.tts_format
            })

            # Create TTS config from CLI arguments
            tts_config = TTSConfig.from_cli_args(args)

            # Generate audio from translated file
            success, message, audio_path = asyncio.run(generate_tts_for_translation(
                translated_filepath=args.output,
                target_language=args.target_lang,
                tts_config=tts_config,
                log_callback=log_callback
            ))

            if success:
                logger.info("TTS Generation Completed", LogType.INFO, {
                    'audio_file': audio_path
                })
            else:
                logger.error(f"TTS generation failed: {message}", LogType.ERROR_DETAIL, {
                    'details': message
                })

    except Exception as e:
        logger.error(f"Translation failed: {str(e)}", LogType.ERROR_DETAIL, {
            'details': str(e),
            'input_file': args.input
        })

        notify(EVENT_FAILURE, {
            'file': args.input,
            'output': args.output,
            'duration_seconds': time.time() - start_time,
            'provider': args.provider,
            'model': args.model,
            'error': str(e),
        })