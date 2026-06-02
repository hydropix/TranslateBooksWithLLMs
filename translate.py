"""
Command-line interface for text translation
"""
import os
import argparse
import asyncio
import logging

# Reduce verbosity of httpx (avoid showing 400 errors during model detection)
logging.getLogger('httpx').setLevel(logging.WARNING)

from src.config import DEFAULT_MODEL, API_ENDPOINT, LLM_PROVIDER, GEMINI_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY, MISTRAL_API_KEY, DEEPSEEK_API_KEY, POE_API_KEY, NIM_API_KEY, DEFAULT_SOURCE_LANGUAGE, DEFAULT_TARGET_LANGUAGE
from src.utils.file_utils import get_unique_output_path, generate_tts_for_translation
from src.utils.unified_logger import setup_cli_logger, LogType
from src.tts.tts_config import TTSConfig, TTS_ENABLED, TTS_VOICE, TTS_RATE, TTS_BITRATE, TTS_OUTPUT_FORMAT
from src.persistence.checkpoint_manager import CheckpointManager
from src.core.adapters import translate_file, refine_file
from src.utils.notifier import notify, EVENT_SUCCESS, EVENT_FAILURE
import time
import uuid


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
    parser.add_argument("--provider", default=LLM_PROVIDER, choices=["ollama", "gemini", "openai", "openrouter", "mistral", "deepseek", "poe", "nim", "litellm"], help=f"LLM provider (default: {LLM_PROVIDER}). Use 'openai' for any OpenAI-compatible server. Use 'litellm' to reach 100+ providers via a provider-prefixed model name (e.g. anthropic/claude-sonnet-4-6); keys are read from each provider's native env var (OPENAI_API_KEY, ANTHROPIC_API_KEY, ...).")
    parser.add_argument("--gemini_api_key", default=GEMINI_API_KEY, help="Google Gemini API key (required if using gemini provider).")
    parser.add_argument("--openai_api_key", default=OPENAI_API_KEY, help="OpenAI API key (required for OpenAI cloud, not needed for local servers).")
    parser.add_argument("--openrouter_api_key", default=OPENROUTER_API_KEY, help="OpenRouter API key (required if using openrouter provider).")
    parser.add_argument("--mistral_api_key", default=MISTRAL_API_KEY, help="Mistral API key (required if using mistral provider).")
    parser.add_argument("--deepseek_api_key", default=DEEPSEEK_API_KEY, help="DeepSeek API key (required if using deepseek provider).")
    parser.add_argument("--poe_api_key", default=POE_API_KEY, help="Poe API key (required if using poe provider). Get your key at https://poe.com/api_key")
    parser.add_argument("--nim_api_key", default=NIM_API_KEY, help="NVIDIA NIM API key (required if using nim provider). Get your key at https://build.nvidia.com/")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output.")

    # Prompt options (optional system prompt instructions)
    prompt_group = parser.add_argument_group('Prompt Options', 'Optional instructions to include in the translation prompt')
    prompt_group.add_argument("--text-cleanup", action="store_true", help="Enable OCR/typographic cleanup (fix broken lines, spacing, punctuation).")
    prompt_group.add_argument("--refine", action="store_true", help="Enable refinement pass: runs a second pass to polish translation quality and literary style.")
    prompt_group.add_argument("--refine-only", action="store_true", dest="refine_only", help="Run ONLY a refinement pass on an already-translated file (skips the translation phase). The input file is assumed to already be in the target language.")
    prompt_group.add_argument("--glossary", default=None, help="Path to a glossary file (.json or .csv) injected per-chunk to keep entity translations consistent.")

    # TTS (Text-to-Speech) arguments
    tts_group = parser.add_argument_group('TTS Options', 'Text-to-Speech audio generation')
    tts_group.add_argument("--tts", action="store_true", default=TTS_ENABLED, help="Generate audio from translated text using Edge-TTS.")
    tts_group.add_argument("--tts-voice", default=TTS_VOICE, help="TTS voice name (auto-selected based on target language if not specified).")
    tts_group.add_argument("--tts-rate", default=TTS_RATE, help="TTS speech rate adjustment, e.g. '+10%%' or '-20%%' (default: %(default)s).")
    tts_group.add_argument("--tts-bitrate", default=TTS_BITRATE, help="Audio bitrate for encoding, e.g. '64k', '96k' (default: %(default)s).")
    tts_group.add_argument("--tts-format", default=TTS_OUTPUT_FORMAT, choices=["opus", "mp3"], help="Audio output format (default: %(default)s).")

    args = parser.parse_args()

    # Auto-select default model based on provider if not explicitly set
    from src.config import NIM_MODEL, MISTRAL_MODEL, DEEPSEEK_MODEL, POE_MODEL, OPENROUTER_MODEL, GEMINI_MODEL, LITELLM_MODEL
    if args.model == DEFAULT_MODEL:
        if args.provider == "nim" and NIM_MODEL:
            args.model = NIM_MODEL
        elif args.provider == "litellm" and LITELLM_MODEL:
            args.model = LITELLM_MODEL
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
                prompt_options=prompt_options
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