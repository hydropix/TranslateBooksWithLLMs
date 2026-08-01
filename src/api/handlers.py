"""
Translation job handlers and processing logic
"""
import os
import re
import time
import asyncio
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from src.utils.unified_logger import setup_web_logger, LogType
from src.utils.file_utils import get_unique_output_path, find_partial_output_paths, generate_tts_for_translation
from src.utils.custom_instructions import (
    load_custom_instructions,
    is_safe_filename,
)
from src.core.llm import OpenRouterProvider
from src.core.llm.exceptions import RateLimitError
from src.config import AUTO_PAUSE_ON_RATE_LIMIT, RATE_LIMIT_AUTO_RESUME_DELAY
from src.core.adapters import translate_file, refine_file
from src.core.progress import snapshot_from_legacy_stats
from src.tts.tts_config import TTSConfig
from src.utils.notifier import notify, EVENT_SUCCESS, EVENT_FAILURE, EVENT_INTERRUPTION
from .completion_status import classify_completion
from .websocket import emit_update


def _format_stats_summary(config, stats, verdict):
    """Build the ' | 8/10 chunks (1 failed) (1 untranslated)' log suffix.

    Shared by the three outcomes of the finalization branch so they all report
    the same counters. Returns an empty string when the job has no chunks.

    Args:
        config: Translation config (only 'file_type' is read).
        stats: The job stats payload.
        verdict: The CompletionVerdict for this job.

    Returns:
        The formatted suffix, or "" when total_chunks is 0.
    """
    total = stats.get('total_chunks', 0) or 0
    if total <= 0:
        return ""

    completed = stats.get('completed_chunks', 0)
    unit = 'subtitles' if config['file_type'] == 'srt' else 'chunks'
    summary = f" | {completed}/{total} {unit}"
    if verdict.failed_chunks > 0:
        summary += f" ({verdict.failed_chunks} failed)"
    if verdict.fallback_chunks > 0:
        summary += f" ({verdict.fallback_chunks} untranslated)"
    return summary


def _notification_context(config, translation_id, elapsed_time, error=None):
    """Build the context dict passed to webhook notifications."""
    ctx = {
        'translation_id': translation_id,
        'file': config.get('original_filename') or config.get('input_filename') or config.get('file_path'),
        'output': config.get('output_filename'),
        'duration_seconds': elapsed_time,
        'provider': config.get('llm_provider'),
        'model': config.get('model'),
        'source_lang': config.get('source_language'),
        'target_lang': config.get('target_language'),
    }
    if error:
        ctx['error'] = error
    return ctx


def run_translation_async_wrapper(translation_id, config, state_manager, output_dir, socketio):
    """
    Wrapper for running translation in async context
    
    Args:
        translation_id (str): Translation job ID
        config (dict): Translation configuration
        state_manager: State manager instance
        output_dir (str): Output directory path
        socketio: SocketIO instance
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(perform_actual_translation(translation_id, config, state_manager, output_dir, socketio))
    except Exception as e:
        error_msg = f"Uncaught major error in translation wrapper {translation_id}: {str(e)}"
        if state_manager.exists(translation_id):
            state_manager.set_translation_field(translation_id, 'status', 'error')
            state_manager.set_translation_field(translation_id, 'error', error_msg)
            logs = state_manager.get_translation_field(translation_id, 'logs')
            if logs is None:
                logs = []
            logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] CRITICAL WRAPPER ERROR: {error_msg}")
            state_manager.set_translation_field(translation_id, 'logs', logs)
            emit_update(socketio, translation_id, {'error': error_msg, 'status': 'error', 'log': f"CRITICAL WRAPPER ERROR: {error_msg}"}, state_manager)
    finally:
        loop.close()


async def perform_actual_translation(translation_id, config, state_manager, output_dir, socketio):
    """
    Perform the actual translation job
    
    Args:
        translation_id (str): Translation job ID
        config (dict): Translation configuration
        state_manager: State manager instance
        output_dir (str): Output directory path
        socketio: SocketIO instance
    """
    if not state_manager.exists(translation_id):
        return

    state_manager.set_translation_field(translation_id, 'status', 'running')
    emit_update(socketio, translation_id, {'status': 'running', 'log': 'Translation task started by worker.'}, state_manager)

    def should_interrupt_current_task():
        if state_manager.exists(translation_id) and state_manager.get_translation_field(translation_id, 'interrupted'):
            _log_message_callback("interruption_check", f"Interruption signal detected for job {translation_id}. Halting processing.")
            return True
        return False

    # Setup unified logger for web interface
    def web_callback(log_entry):
        """Callback for WebSocket emission"""
        logs = state_manager.get_translation_field(translation_id, 'logs')
        if logs is None:
            logs = []
        logs.append(log_entry)
        state_manager.set_translation_field(translation_id, 'logs', logs)
        # Send full log entry for structured processing on client side
        emit_update(socketio, translation_id, {'log': log_entry['message'], 'log_entry': log_entry}, state_manager)
    
    def storage_callback(log_entry):
        """Callback for storing logs"""
        logs = state_manager.get_translation_field(translation_id, 'logs')
        if logs is None:
            logs = []
        logs.append(log_entry)
        state_manager.set_translation_field(translation_id, 'logs', logs)
    
    logger = setup_web_logger(web_callback, storage_callback)
    
    def _log_message_callback(message_key_from_translate_module, message_content="", data=None):
        """Legacy callback wrapper for backward compatibility"""
        # Skip debug messages for web interface
        if message_key_from_translate_module in ["llm_prompt_debug", "llm_raw_response_preview"]:
            return
        
        # Handle structured data from new logging system
        if data and isinstance(data, dict):
            log_type = data.get('type')
            if log_type == 'llm_request':
                logger.debug("LLM Request", LogType.LLM_REQUEST, data)
            elif log_type == 'llm_response':
                # Use INFO level to ensure translation preview works even when DEBUG_MODE=false
                logger.info("LLM Response", LogType.LLM_RESPONSE, data)
            elif log_type == 'refinement_request':
                # Refinement uses same log type as LLM request for UI display
                logger.debug("Refinement Request", LogType.LLM_REQUEST, data)
            elif log_type == 'refinement_response':
                # Refinement uses same log type as LLM response for UI display
                # Use INFO level to ensure translation preview works even when DEBUG_MODE=false
                logger.info("Refinement Response", LogType.LLM_RESPONSE, data)
            elif log_type == 'progress':
                logger.info("Progress Update", LogType.PROGRESS, data)
            else:
                logger.info(message_content, data=data)
        else:
            # Map specific message patterns to appropriate log types
            if "error" in message_key_from_translate_module.lower():
                logger.error(message_content)
            elif "warning" in message_key_from_translate_module.lower():
                logger.warning(message_content)
            else:
                logger.info(message_content)

    # Single progress-emit seam. The translate_file → refine_file orchestration
    # still runs two independent engine-side trackers, but the *workflow phase*
    # is now owned here and passed explicitly per phase (TRANSLATING vs
    # REFINING) by whichever call is driving the emit — replacing the old
    # mutable `_workflow_meta` side-channel. A monotonic floor on the canonical
    # `percent` guarantees the bar never regresses across the phase boundary,
    # which is what previously required a manual counter reset at the
    # transition.
    _progress_floor = {'value': 0.0}

    def _emit_progress(new_stats_dict, phase_meta):
        if not state_manager.exists(translation_id):
            return
        state_manager.update_stats(translation_id, {**new_stats_dict, **phase_meta})
        current_stats = state_manager.get_translation_field(translation_id, 'stats') or {}
        current_stats['elapsed_time'] = time.time() - current_stats.get('start_time', time.time())
        # Attach the canonical progress contract alongside the legacy fields,
        # applying the monotonic floor so the global bar never moves backward.
        snapshot = snapshot_from_legacy_stats(current_stats).to_dict()
        if snapshot['percent'] < _progress_floor['value']:
            snapshot['percent'] = _progress_floor['value']
        else:
            _progress_floor['value'] = snapshot['percent']
        current_stats.update(snapshot)
        state_manager.set_translation_field(translation_id, 'stats', current_stats)
        emit_update(socketio, translation_id, {'stats': current_stats}, state_manager)

        # Update logger progress for CLI display
        completed = current_stats.get('completed_chunks', 0)
        total = current_stats.get('total_chunks', 0)
        if total > 0:
            logger.update_progress(completed, total)

    def _translate_stats_callback(new_stats_dict):
        # Phase 1. enable_refinement advertises the two-phase bar up-front when
        # a refine-after pass will follow, so phase 1 maps to the [0, 50] band.
        _emit_progress(new_stats_dict, {
            'enable_refinement': bool(config.get('refine_after')),
            'current_phase': 1,
        })

    def _refine_after_stats_callback(new_stats_dict):
        # Phase 2 of a refine-after workflow: maps to the [50, 100] band.
        _emit_progress(new_stats_dict, {'enable_refinement': True, 'current_phase': 2})

    def _refine_only_stats_callback(new_stats_dict):
        # Single-phase refine-only: the whole bar is the refinement pass.
        _emit_progress(new_stats_dict, {
            'enable_refinement': False, 'refine_only': True, 'current_phase': 1,
        })

    def _finalize_stats_callback(new_stats_dict):
        # Finalization pushes (e.g. final elapsed_time) must not re-assert a
        # phase; the stored stats already carry the terminal phase/percent.
        _emit_progress(new_stats_dict, {})

    def _openrouter_cost_callback(cost_data):
        """Update OpenRouter cost in state. No emit: this callback runs on the
        provider's HTTP response thread, and a cross-thread emit can overtake
        the main loop's stats emit on the wire (showing a stale snapshot and
        rolling the progress bar backward). The cost is picked up by the next
        chunk's stats_callback, which is the same thread that owns progress."""
        if state_manager.exists(translation_id):
            state_manager.update_stats(translation_id, {
                'openrouter_cost': cost_data['session_cost'],
                'openrouter_prompt_tokens': cost_data['total_prompt_tokens'],
                'openrouter_completion_tokens': cost_data['total_completion_tokens']
            })

    # Setup OpenRouter cost callback if using OpenRouter provider
    if config.get('llm_provider') == 'openrouter':
        OpenRouterProvider.reset_session_cost()
        OpenRouterProvider.set_cost_callback(_openrouter_cost_callback)

    # Get checkpoint manager and handle resume
    checkpoint_manager = state_manager.get_checkpoint_manager()
    resume_from_index = config.get('resume_from_index', 0)
    is_resume = config.get('is_resume', False)

    # Snapshot the active glossary into prompt_options BEFORE persisting the
    # job, so the snapshot survives resume even if the source glossary is
    # later edited or deleted. On resume, the snapshot is already in the
    # restored config and we skip the reload.
    if not is_resume:
        glossary_id = config.get('prompt_options', {}).get('glossary_id')
        if glossary_id and not config.get('prompt_options', {}).get('glossary_terms'):
            try:
                from src.api.translation_state import get_state_manager
                store = get_state_manager().get_glossary_store()
                glossary = store.get_glossary(int(glossary_id))
                if glossary and glossary.terms:
                    if 'prompt_options' not in config:
                        config['prompt_options'] = {}
                    config['prompt_options']['glossary_terms'] = glossary.terms_dict
                    config['prompt_options']['glossary_name'] = glossary.name
                    metadata = {}
                    for term in glossary.terms:
                        if term.category:
                            metadata[term.source_term] = {'category': term.category}
                    if metadata:
                        config['prompt_options']['glossary_term_metadata'] = metadata
            except Exception as e:
                # Non-fatal: log later once the logger is wired in.
                config.setdefault('prompt_options', {})['glossary_load_error'] = str(e)

    try:
        # Create checkpoint for new jobs (not for resumed jobs)
        if not is_resume:
            file_type = config['file_type']
            input_file_path = config.get('file_path')
            checkpoint_manager.start_job(
                translation_id,
                file_type,
                config,
                input_file_path
            )

        # PHASE 2: Configuration validation is now handled by AdaptiveContextManager during translation

        # Generate unique output filename to avoid overwriting
        tentative_output_path = os.path.join(output_dir, config['output_filename'])
        output_filepath_on_server = get_unique_output_path(tentative_output_path)

        # Update config with the actual filename (may have been modified)
        actual_output_filename = os.path.basename(output_filepath_on_server)
        if actual_output_filename != config['output_filename']:
            _log_message_callback("output_filename_modified",
                f"ℹ️ Output filename modified to avoid overwriting: {config['output_filename']} → {actual_output_filename}")
            config['output_filename'] = actual_output_filename

        # Log translation start with unified logger
        logger.info("Translation Started", LogType.TRANSLATION_START, {
            'source_lang': config['source_language'],
            'target_lang': config['target_language'],
            'file_type': config['file_type'].upper(),
            'model': config['model'],
            'translation_id': translation_id,
            'output_file': config['output_filename'],
            'api_endpoint': config['llm_api_endpoint'],
            'chunk_size': config.get('chunk_size', 'default')
        })

        # Make the resumed portion's model/provider explicit in the job log so a
        # resume that switched model/provider (issue #183) is auditable.
        if is_resume:
            _log_message_callback(
                "resume_model_info",
                f"↻ Resuming from chunk {resume_from_index} using "
                f"{config.get('llm_provider', 'ollama')} / {config['model']}."
            )

        input_path_for_translate_module = config.get('file_path')

        # Handle special case for TXT with inline text content (no file upload)
        temp_txt_file_path = None
        if config['file_type'] == 'txt' and 'text' in config and input_path_for_translate_module is None:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".txt", dir=output_dir) as tmp_f:
                tmp_f.write(config['text'])
                temp_txt_file_path = tmp_f.name
            input_path_for_translate_module = temp_txt_file_path

        # Validate input file path
        if not input_path_for_translate_module:
            _log_message_callback("error_no_path", f"❌ {config['file_type'].upper()} translation requires a file path from upload.")
            raise Exception(f"{config['file_type'].upper()} translation requires a file_path.")

        # Read custom instruction file if specified
        custom_instruction_file = config.get('prompt_options', {}).get('custom_instruction_file', '')

        translation_instructions = None
        refinement_instructions = None

        if custom_instruction_file:
            project_root = Path(os.getcwd())
            custom_instructions_dir = project_root / 'Custom_Instructions'

            if not is_safe_filename(custom_instruction_file):
                _log_message_callback(
                    "custom_instructions_invalid",
                    f"⚠️ Custom instructions file name '{custom_instruction_file}' is invalid "
                    f"(allowed: alphanumeric, underscore, hyphen, dot; must end in .txt, .yaml, or .yml). "
                    f"Translation will proceed without it."
                )
            else:
                try:
                    loaded = load_custom_instructions(
                        custom_instruction_file, custom_instructions_dir
                    )
                    translation_instructions = loaded.get('translation')
                    refinement_instructions = loaded.get('refinement')

                    if translation_instructions or refinement_instructions:
                        phases = []
                        if translation_instructions:
                            phases.append('translation')
                        if refinement_instructions:
                            phases.append('refinement')
                        _log_message_callback(
                            "custom_instructions",
                            f"📝 Loaded custom instructions: {custom_instruction_file} "
                            f"(phases: {', '.join(phases)})"
                        )
                    else:
                        _log_message_callback(
                            "custom_instructions_empty",
                            f"⚠️ Custom instructions file '{custom_instruction_file}' is empty. "
                            f"Translation will proceed without it."
                        )
                except FileNotFoundError:
                    _log_message_callback(
                        "custom_instructions_missing",
                        f"⚠️ Custom instructions file '{custom_instruction_file}' was selected "
                        f"but not found in {custom_instructions_dir}. Translation will proceed "
                        f"without it."
                    )
                except (ValueError, Exception) as e:
                    _log_message_callback(
                        "custom_instructions_error",
                        f"⚠️ Failed to load custom instructions '{custom_instruction_file}': {e}. "
                        f"Translation will proceed without it."
                    )

        # Inject phase-specific custom instructions into prompt_options
        if translation_instructions or refinement_instructions:
            if 'prompt_options' not in config:
                config['prompt_options'] = {}
            if translation_instructions:
                config['prompt_options']['custom_instructions'] = translation_instructions
            if refinement_instructions:
                config['prompt_options']['refinement_instructions'] = refinement_instructions

        # Surface glossary load result (snapshot was taken earlier, before start_job).
        glossary_terms_snapshot = config.get('prompt_options', {}).get('glossary_terms')
        glossary_load_error = config.get('prompt_options', {}).pop('glossary_load_error', None)
        if glossary_terms_snapshot:
            glossary_name = config.get('prompt_options', {}).get('glossary_name', '?')
            _log_message_callback(
                "glossary_loaded",
                f"📖 Loaded glossary '{glossary_name}' ({len(glossary_terms_snapshot)} terms)"
            )
        elif glossary_load_error:
            _log_message_callback(
                "glossary_error",
                f"⚠️ Could not load glossary: {glossary_load_error}"
            )

        if config.get('refine_only'):
            _log_message_callback(
                "refine_only_mode",
                "✨ Refine-only mode: skipping translation, polishing the input file as-is."
            )
            src_lang = config.get('source_language')
            tgt_lang = config.get('target_language')
            if src_lang and tgt_lang and src_lang != tgt_lang:
                _log_message_callback(
                    "refine_only_lang_mismatch",
                    f"⚠️ source_language ({src_lang}) ≠ target_language ({tgt_lang}). "
                    f"Refinement is monolingual; the file will be polished as {tgt_lang}."
                )
            await refine_file(
                input_filepath=input_path_for_translate_module,
                output_filepath=output_filepath_on_server,
                target_language=config['target_language'],
                model_name=config['model'],
                llm_provider=config.get('llm_provider', 'ollama'),
                checkpoint_manager=checkpoint_manager,
                translation_id=translation_id,
                log_callback=_log_message_callback,
                stats_callback=_refine_only_stats_callback,
                check_interruption_callback=should_interrupt_current_task,
                resume_from_index=resume_from_index,
                llm_api_endpoint=config['llm_api_endpoint'],
                gemini_api_key=config.get('gemini_api_key', ''),
                openai_api_key=config.get('openai_api_key', ''),
                openrouter_api_key=config.get('openrouter_api_key', ''),
                mistral_api_key=config.get('mistral_api_key', ''),
                deepseek_api_key=config.get('deepseek_api_key', ''),
                poe_api_key=config.get('poe_api_key', ''),
                nim_api_key=config.get('nim_api_key', ''),
                context_window=config.get('context_window', 2048),
                auto_adjust_context=config.get('auto_adjust_context', True),
                max_tokens_per_chunk=config.get('max_tokens_per_chunk'),
                prompt_options=config.get('prompt_options', {}),
            )
        else:
            # The translate-phase callback advertises the two-phase workflow
            # up-front (via enable_refinement) when a refine-after pass will
            # follow, so the UI renders the phase bar from the start of phase 1.
            await translate_file(
                input_filepath=input_path_for_translate_module,
                output_filepath=output_filepath_on_server,
                source_language=config['source_language'],
                target_language=config['target_language'],
                model_name=config['model'],
                llm_provider=config.get('llm_provider', 'ollama'),
                checkpoint_manager=checkpoint_manager,
                translation_id=translation_id,
                log_callback=_log_message_callback,
                stats_callback=_translate_stats_callback,
                check_interruption_callback=should_interrupt_current_task,
                resume_from_index=resume_from_index,
                llm_api_endpoint=config['llm_api_endpoint'],
                gemini_api_key=config.get('gemini_api_key', ''),
                openai_api_key=config.get('openai_api_key', ''),
                openrouter_api_key=config.get('openrouter_api_key', ''),
                mistral_api_key=config.get('mistral_api_key', ''),
                deepseek_api_key=config.get('deepseek_api_key', ''),
                poe_api_key=config.get('poe_api_key', ''),
                nim_api_key=config.get('nim_api_key', ''),
                context_window=config.get('context_window', 2048),
                auto_adjust_context=config.get('auto_adjust_context', True),
                min_chunk_size=config.get('min_chunk_size', 5),
                max_tokens_per_chunk=config.get('max_tokens_per_chunk'),
                prompt_options=config.get('prompt_options', {}),
                bilingual_output=config.get('bilingual_output', False),
                parallel_workers=config.get('parallel_workers', 1)
            )

            # Optional chained refinement pass on the translated output.
            should_refine_after = (
                config.get('refine_after')
                and os.path.exists(output_filepath_on_server)
                and not state_manager.get_translation_field(translation_id, 'interrupted')
                and state_manager.get_translation_field(translation_id, 'status')
                    not in ('error', 'partial', 'rate_limited')
            )
            if should_refine_after:
                # Phase 2. No manual counter reset is needed: the refine-phase
                # callback always tags emits as phase 2 with the refine engine's
                # own counters (starting at 0), so a stale phase-1 count can
                # never pair with phase 2, and the monotonic floor in
                # _emit_progress keeps the bar from regressing across the
                # transition.
                _log_message_callback(
                    "refine_after_start",
                    "✨ Translation done — running refinement pass on the output."
                )
                await refine_file(
                    input_filepath=output_filepath_on_server,
                    output_filepath=output_filepath_on_server,
                    target_language=config['target_language'],
                    model_name=config['model'],
                    llm_provider=config.get('llm_provider', 'ollama'),
                    checkpoint_manager=checkpoint_manager,
                    translation_id=translation_id,
                    log_callback=_log_message_callback,
                    stats_callback=_refine_after_stats_callback,
                    check_interruption_callback=should_interrupt_current_task,
                    resume_from_index=0,
                    llm_api_endpoint=config['llm_api_endpoint'],
                    gemini_api_key=config.get('gemini_api_key', ''),
                    openai_api_key=config.get('openai_api_key', ''),
                    openrouter_api_key=config.get('openrouter_api_key', ''),
                    mistral_api_key=config.get('mistral_api_key', ''),
                    deepseek_api_key=config.get('deepseek_api_key', ''),
                    poe_api_key=config.get('poe_api_key', ''),
                    nim_api_key=config.get('nim_api_key', ''),
                    context_window=config.get('context_window', 2048),
                    auto_adjust_context=config.get('auto_adjust_context', True),
                    max_tokens_per_chunk=config.get('max_tokens_per_chunk'),
                    prompt_options=config.get('prompt_options', {}),
                )

        # If an EPUB translation was paused, the file was saved with a `[partial NN%]`
        # prefix. Re-point the tracking variables to the actual file on disk so the
        # download endpoint and UI list the right name.
        if (config['file_type'] == 'epub'
                and state_manager.get_translation_field(translation_id, 'interrupted')
                and not os.path.exists(output_filepath_on_server)):
            candidates = find_partial_output_paths(output_filepath_on_server)
            if candidates:
                # Pick the most recently written one if several exist
                actual = max(candidates, key=lambda p: os.path.getmtime(p))
                output_filepath_on_server = actual
                config['output_filename'] = os.path.basename(actual)
                _log_message_callback("output_marked_partial",
                    f"💾 Partial EPUB saved as: {config['output_filename']}")

        # Set result message based on file type
        file_type_upper = config['file_type'].upper()
        if os.path.exists(output_filepath_on_server) and state_manager.get_translation_field(translation_id, 'status') not in ['error', 'interrupted_before_save']:
            state_manager.set_translation_field(translation_id, 'result', f"[{file_type_upper} file translated - download to view]")
        elif not os.path.exists(output_filepath_on_server):
            state_manager.set_translation_field(translation_id, 'result', f"[{file_type_upper} file (partially) translated - content not loaded for preview or write failed]")

        # Clean up temporary text file if created
        if temp_txt_file_path and os.path.exists(temp_txt_file_path):
            os.remove(temp_txt_file_path)

        state_manager.set_translation_field(translation_id, 'output_filepath', output_filepath_on_server)

        stats = state_manager.get_translation_field(translation_id, 'stats') or {}
        elapsed_time = time.time() - stats.get('start_time', time.time())
        _finalize_stats_callback({'elapsed_time': elapsed_time})

        final_status_payload = {
            'result': state_manager.get_translation_field(translation_id, 'result'),
            'output_filename': config['output_filename'],
            'output_dir': os.path.dirname(os.path.abspath(output_filepath_on_server)),
            'file_type': config['file_type']
        }

        if state_manager.get_translation_field(translation_id, 'interrupted'):
            state_manager.set_translation_field(translation_id, 'status', 'interrupted')
            _log_message_callback("summary_interrupted", f"🛑 Translation interrupted - partial result saved ({elapsed_time:.2f}s)")
            final_status_payload['status'] = 'interrupted'
            await asyncio.to_thread(notify, EVENT_INTERRUPTION,
                _notification_context(config, translation_id, elapsed_time))

            # Mark checkpoint as interrupted in database
            checkpoint_manager.mark_interrupted(translation_id)

            # Emit checkpoint_created event to trigger UI update
            socketio.emit('checkpoint_created', {
                'translation_id': translation_id,
                'status': 'interrupted',
                'message': 'Translation paused - checkpoint created'
            }, namespace='/')

            # DON'T clean up uploaded file on interruption - keep it for resume capability
            # The file will be preserved in the job-specific directory by checkpoint_manager
            # Only clean up if the preserved file exists (meaning backup was successful)
            preserved_path = config.get('preserved_input_path')
            if preserved_path and Path(preserved_path).exists():
                # Preserved file exists, we can safely delete the original upload
                if 'file_path' in config and config['file_path']:
                    uploaded_file_path = config['file_path']
                    upload_path = Path(uploaded_file_path)

                    if upload_path.exists() and upload_path != Path(preserved_path):
                        try:
                            # Only delete if it's in the uploads directory root (not in a job subdirectory)
                            uploads_dir = Path(output_dir) / 'uploads'
                            resolved_path = upload_path.resolve()

                            # Check if file is directly in uploads/ (not in a job subdirectory)
                            if resolved_path.parent.resolve() == uploads_dir.resolve():
                                upload_path.unlink()
                                _log_message_callback("cleanup_uploaded_file", f"🗑️ Cleaned up uploaded source file (preserved copy exists): {upload_path.name}")
                            else:
                                _log_message_callback("cleanup_skipped", f"ℹ️ Skipped cleanup - file is not in uploads root directory")
                        except Exception as e:
                            _log_message_callback("cleanup_error", f"⚠️ Could not delete uploaded file {upload_path.name}: {str(e)}")
                else:
                    _log_message_callback("cleanup_info", "ℹ️ Original upload file not found or already cleaned up")
            else:
                _log_message_callback("cleanup_skipped_no_preserve", "ℹ️ Skipped cleanup - preserved file not found, keeping original for resume")

        elif state_manager.get_translation_field(translation_id, 'status') != 'error':
            # Single source of truth for "did this job actually succeed?".
            # Reads the stats directly, so it is not gated by file type.
            verdict = classify_completion(stats, output_filepath_on_server)
            stats_summary = _format_stats_summary(config, stats, verdict)

            if verdict.status == 'error':
                # The translation loop ran to the end but produced no usable
                # output file. Report it honestly instead of claiming success.
                state_manager.set_translation_field(translation_id, 'status', 'error')
                state_manager.set_translation_field(translation_id, 'error', verdict.error)
                _log_message_callback("summary_error_no_output",
                    f"❌ Translation finished without a usable output file in {elapsed_time:.2f}s"
                    f"{stats_summary} — {verdict.error}")
                final_status_payload['status'] = 'error'
                final_status_payload['error'] = verdict.error
                checkpoint_manager.mark_error(translation_id)
                await asyncio.to_thread(notify, EVENT_FAILURE,
                    _notification_context(config, translation_id, elapsed_time,
                                          error=verdict.error))
                # No cleanup_completed_job — the checkpoint is deliberately kept,
                # it is the only way back for this job.
            elif verdict.status == 'partial':
                # Chunks failed outright and/or fell back to their source text.
                # Keep the job resumable instead of marking it 'completed' and
                # deleting the checkpoint, so the user can retry those chunks.
                state_manager.set_translation_field(translation_id, 'status', 'partial')
                _log_message_callback("summary_partial",
                    f"⚠️ Translation finished with {verdict.failed_chunks} failed "
                    f"and {verdict.fallback_chunks} untranslated chunk(s) in {elapsed_time:.2f}s"
                    f"{stats_summary} — checkpoint kept for retry")
                final_status_payload['status'] = 'partial'
                checkpoint_manager.mark_partial(translation_id)
                # No cleanup_completed_job — the checkpoint is deliberately kept for retry.
            else:
                state_manager.set_translation_field(translation_id, 'status', 'completed')
                _log_message_callback("summary_completed", f"✅ Translation completed in {elapsed_time:.2f}s{stats_summary}")
                final_status_payload['status'] = 'completed'
                await asyncio.to_thread(notify, EVENT_SUCCESS,
                    _notification_context(config, translation_id, elapsed_time))

                # Cleanup completed job checkpoint (automatic immediate cleanup)
                checkpoint_manager.cleanup_completed_job(translation_id)

            # Clean up uploaded file if it exists and is in the uploads directory
            # On completion (or a partial run) we can safely delete the original
            # upload. Never on the 'error' outcome: with no output file, the
            # uploaded source is the only way to resume.
            if verdict.status != 'error' and 'file_path' in config and config['file_path']:
                uploaded_file_path = config['file_path']
                # Convert to Path object for reliable path operations
                upload_path = Path(uploaded_file_path)

                # Check if file exists
                if upload_path.exists():
                    try:
                        # Only delete if it's in the uploads directory root (not in a job subdirectory)
                        uploads_dir = Path(output_dir) / 'uploads'
                        resolved_path = upload_path.resolve()

                        # Check if file is directly in uploads/ (not in a job subdirectory)
                        if resolved_path.parent.resolve() == uploads_dir.resolve():
                            upload_path.unlink()
                            # Removed verbose cleanup message - file cleanup is automatic
                    except Exception as e:
                        _log_message_callback("cleanup_error", f"⚠️ Could not delete uploaded file {upload_path.name}: {str(e)}")
        else:
            _log_message_callback("summary_error_final", f"❌ Translation finished with errors ({elapsed_time:.2f}s)")
            final_status_payload['status'] = 'error'
            final_status_payload['error'] = state_manager.get_translation_field(translation_id, 'error') or 'Unknown error during finalization.'
            await asyncio.to_thread(notify, EVENT_FAILURE,
                _notification_context(config, translation_id, elapsed_time,
                                      error=final_status_payload['error']))

        # Stats are now included in the consolidated completion message above

        # Log OpenRouter cost summary if applicable
        if config.get('llm_provider') == 'openrouter':
            cost = stats.get('openrouter_cost', 0.0)
            prompt_tokens = stats.get('openrouter_prompt_tokens', 0)
            completion_tokens = stats.get('openrouter_completion_tokens', 0)
            total_tokens = prompt_tokens + completion_tokens
            if cost > 0 or total_tokens > 0:
                _log_message_callback("openrouter_cost_final",
                    f"💰 OpenRouter Cost: ${cost:.4f} | Tokens: {total_tokens:,} ({prompt_tokens:,} prompt + {completion_tokens:,} completion)")
            # Clear the callback to avoid memory leaks
            OpenRouterProvider.set_cost_callback(None)

        # TTS Generation (if enabled and translation completed successfully)
        if config.get('tts_enabled') and final_status_payload.get('status') == 'completed':
            await _perform_tts_generation(
                translation_id,
                config,
                output_filepath_on_server,
                state_manager,
                socketio,
                _log_message_callback
            )

        # Attach final stats so the completion card can render its summary
        # (cost, tokens, failed chunks…). emit_update no longer auto-attaches.
        final_status_payload['stats'] = state_manager.get_translation_field(translation_id, 'stats') or {}
        emit_update(socketio, translation_id, final_status_payload, state_manager)

        # Trigger file list refresh in the frontend if a file was saved
        if os.path.exists(output_filepath_on_server) and final_status_payload['status'] in ['completed', 'interrupted', 'partial']:
            socketio.emit('file_list_changed', {
                'reason': final_status_payload['status'],
                'filename': config.get('output_filename', 'unknown')
            }, namespace='/')

    except RateLimitError as e:
        auto_pause = config.get('auto_pause_on_rate_limit', AUTO_PAUSE_ON_RATE_LIMIT)
        retry_msg = f" Retry suggested after ~{e.retry_after}s." if e.retry_after else ""
        provider_name = e.provider or config.get('llm_provider', 'API')

        if not state_manager.exists(translation_id):
            return

        # Auto-resume mode keeps the job running: wait, then re-enter from the checkpoint.
        if not auto_pause:
            wait_seconds = e.retry_after or RATE_LIMIT_AUTO_RESUME_DELAY
            wait_msg = (f"⏳ Rate limited by {provider_name}.{retry_msg} "
                        f"Auto-resume in {wait_seconds}s (auto-pause disabled).")
            _log_message_callback("rate_limit_auto_resume", wait_msg)

            # Surface 'rate_limited' transiently so the UI shows what's happening.
            state_manager.set_translation_field(translation_id, 'status', 'rate_limited')
            emit_update(socketio, translation_id, {
                'status': 'rate_limited',
                'log': wait_msg
            }, state_manager)

            await asyncio.sleep(wait_seconds)

            # Honor an interrupt that arrived during the wait by falling through to pause.
            if state_manager.get_translation_field(translation_id, 'interrupted'):
                _log_message_callback("rate_limit_auto_resume_cancelled",
                    "🛑 Auto-resume cancelled by user, pausing instead.")
            else:
                cp_data = checkpoint_manager.load_checkpoint(translation_id)
                if cp_data:
                    # Track consecutive auto-resume cycles that fail without
                    # translating any chunk past the checkpoint. After three
                    # such cycles the daily quota is probably exhausted and
                    # the loop is just burning time, so we warn the user.
                    # We keep auto-resuming because they explicitly opted in,
                    # but the warning lets them choose to pause manually.
                    resume_index = cp_data['resume_from_index']
                    last_resume_index = config.get('_auto_resume_last_index')
                    stuck_count = config.get('_auto_resume_stuck_count', 0)
                    if last_resume_index == resume_index:
                        stuck_count += 1
                    else:
                        stuck_count = 1
                    if stuck_count >= 3:
                        _log_message_callback(
                            "rate_limit_auto_resume_stuck",
                            f"⚠️ Auto-resume has looped {stuck_count} times without "
                            f"progress (still stuck at chunk {resume_index}). Your "
                            f"daily quota may be exhausted or the provider is "
                            f"throttling this account; consider interrupting now "
                            f"and checking your provider's quota dashboard. "
                            f"Auto-resume will keep trying but may be wasting time."
                        )

                    new_config = dict(config)
                    new_config['is_resume'] = True
                    new_config['resume_from_index'] = resume_index
                    new_config['_auto_resume_last_index'] = resume_index
                    new_config['_auto_resume_stuck_count'] = stuck_count
                    checkpoint_manager.mark_running(translation_id)
                    state_manager.set_translation_field(translation_id, 'status', 'running')
                    emit_update(socketio, translation_id, {
                        'status': 'running',
                        'log': f"▶️ Auto-resuming from chunk {resume_index}..."
                    }, state_manager)
                    await perform_actual_translation(
                        translation_id, new_config, state_manager, output_dir, socketio
                    )
                    return
                # No checkpoint available, fall through to the pause path below.
                _log_message_callback("rate_limit_no_checkpoint",
                    "⚠️ Auto-resume requested but no checkpoint found, falling back to pause.")

        pause_msg = f"⏸️ Rate limited by {provider_name}.{retry_msg} Translation auto-paused, you can resume when ready."
        _log_message_callback("rate_limit_auto_pause", pause_msg)

        state_manager.set_translation_field(translation_id, 'status', 'rate_limited')
        state_manager.set_translation_field(translation_id, 'interrupted', True)
        checkpoint_manager.mark_interrupted(translation_id)

        stats = state_manager.get_translation_field(translation_id, 'stats') or {}
        elapsed_time = time.time() - stats.get('start_time', time.time())
        _finalize_stats_callback({'elapsed_time': elapsed_time})

        emit_update(socketio, translation_id, {
            'status': 'rate_limited',
            'log': pause_msg,
            'result': state_manager.get_translation_field(translation_id, 'result') or f"Translation paused (rate limited)"
        }, state_manager)

        socketio.emit('checkpoint_created', {
            'translation_id': translation_id,
            'status': 'rate_limited',
            'message': pause_msg
        }, namespace='/')

        output_filepath = state_manager.get_translation_field(translation_id, 'output_filepath')
        if output_filepath and os.path.exists(output_filepath):
            socketio.emit('file_list_changed', {
                'reason': 'rate_limited',
                'filename': config.get('output_filename', 'unknown')
            }, namespace='/')

    except Exception as e:
        critical_error_msg = f"Critical error during translation task ({translation_id}): {str(e)}"
        _log_message_callback("critical_error_perform_task", critical_error_msg)
        import traceback
        tb_str = traceback.format_exc()
        _log_message_callback("critical_error_perform_task_traceback", tb_str)

        if state_manager.exists(translation_id):
            state_manager.set_translation_field(translation_id, 'status', 'error')
            state_manager.set_translation_field(translation_id, 'error', critical_error_msg)

            emit_update(socketio, translation_id, {
                'error': critical_error_msg,
                'status': 'error',
                'result': state_manager.get_translation_field(translation_id, 'result') or f"Translation failed: {critical_error_msg}"
            }, state_manager)

            stats = state_manager.get_translation_field(translation_id, 'stats') or {}
            elapsed_time = time.time() - stats.get('start_time', time.time())
            await asyncio.to_thread(notify, EVENT_FAILURE,
                _notification_context(config, translation_id, elapsed_time,
                                      error=critical_error_msg))


async def _perform_tts_generation(translation_id, config, output_filepath, state_manager, socketio, log_callback):
    """
    Perform TTS generation after successful translation.

    Args:
        translation_id: Translation job ID
        config: Translation configuration dict
        output_filepath: Path to the translated file
        state_manager: State manager instance
        socketio: SocketIO instance for WebSocket events
        log_callback: Logging callback function
    """
    try:
        log_callback("tts_phase_start", "🔊 Starting TTS audio generation...")

        # Emit TTS started event
        socketio.emit('tts_update', {
            'translation_id': translation_id,
            'status': 'started',
            'message': 'TTS generation started'
        }, namespace='/')

        # Reconstruct TTSConfig from dict
        tts_config_dict = config.get('tts_config', {})
        tts_config = TTSConfig(
            enabled=True,
            provider=tts_config_dict.get('provider', 'edge-tts'),
            voice=tts_config_dict.get('voice', ''),
            rate=tts_config_dict.get('rate', '+0%'),
            volume=tts_config_dict.get('volume', '+0%'),
            pitch=tts_config_dict.get('pitch', '+0Hz'),
            output_format=tts_config_dict.get('output_format', 'opus'),
            bitrate=tts_config_dict.get('bitrate', '64k'),
            sample_rate=tts_config_dict.get('sample_rate', 24000),
            chunk_size=tts_config_dict.get('chunk_size', 5000),
            pause_between_chunks=tts_config_dict.get('pause_between_chunks', 0.5)
        )

        target_language = config.get('target_language', '')

        # Create TTS progress callback
        def tts_progress_callback(current, total, message):
            progress_pct = int((current / total) * 100) if total > 0 else 0
            log_callback("tts_chunk_progress", f"🔊 TTS: {message}")
            socketio.emit('tts_update', {
                'translation_id': translation_id,
                'status': 'processing',
                'progress': progress_pct,
                'current_chunk': current,
                'total_chunks': total,
                'message': message
            }, namespace='/')

        # Generate TTS
        success, message, audio_path = await generate_tts_for_translation(
            translated_filepath=output_filepath,
            target_language=target_language,
            tts_config=tts_config,
            log_callback=log_callback,
            progress_callback=tts_progress_callback
        )

        if success:
            log_callback("tts_complete", f"✅ TTS audio generated: {os.path.basename(audio_path)}")

            # Store audio file path in state
            state_manager.set_translation_field(translation_id, 'audio_filepath', audio_path)
            state_manager.set_translation_field(translation_id, 'audio_filename', os.path.basename(audio_path))

            # Emit success event
            socketio.emit('tts_update', {
                'translation_id': translation_id,
                'status': 'completed',
                'progress': 100,
                'audio_filename': os.path.basename(audio_path),
                'message': 'TTS generation completed successfully'
            }, namespace='/')

            # Trigger file list refresh
            socketio.emit('file_list_changed', {
                'reason': 'tts_completed',
                'filename': os.path.basename(audio_path)
            }, namespace='/')

        else:
            log_callback("tts_failed", f"❌ TTS generation failed: {message}")
            socketio.emit('tts_update', {
                'translation_id': translation_id,
                'status': 'failed',
                'error': message,
                'message': f'TTS generation failed: {message}'
            }, namespace='/')

    except Exception as e:
        error_msg = f"TTS generation error: {str(e)}"
        log_callback("tts_error", f"❌ {error_msg}")
        socketio.emit('tts_update', {
            'translation_id': translation_id,
            'status': 'failed',
            'error': error_msg,
            'message': error_msg
        }, namespace='/')


def start_translation_job(translation_id, config, state_manager, output_dir, socketio):
    """
    Start a translation job in a separate thread

    Args:
        translation_id (str): Translation job ID
        config (dict): Translation configuration
        state_manager: State manager instance
        output_dir (str): Output directory path
        socketio: SocketIO instance
    """
    thread = threading.Thread(
        target=run_translation_async_wrapper,
        args=(translation_id, config, state_manager, output_dir, socketio)
    )
    thread.daemon = True
    thread.start()