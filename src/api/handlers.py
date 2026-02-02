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

from src.utils.unified_logger import setup_web_logger, LogType
from src.utils.file_utils import get_unique_output_path, generate_tts_for_translation
from src.core.llm import OpenRouterProvider
from src.core.adapters import translate_file
from src.tts.tts_config import TTSConfig
from .websocket import emit_update


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

    def _update_translation_stats_callback(new_stats_dict):
        if state_manager.exists(translation_id):
            state_manager.update_stats(translation_id, new_stats_dict)
            current_stats = state_manager.get_translation_field(translation_id, 'stats') or {}
            current_stats['elapsed_time'] = time.time() - current_stats.get('start_time', time.time())
            state_manager.set_translation_field(translation_id, 'stats', current_stats)
            emit_update(socketio, translation_id, {'stats': current_stats}, state_manager)

            # Update logger progress for CLI display
            completed = current_stats.get('completed_chunks', 0)
            total = current_stats.get('total_chunks', 0)
            if total > 0:
                logger.update_progress(completed, total)

    def _openrouter_cost_callback(cost_data):
        """Callback to update OpenRouter cost information in real-time"""
        if state_manager.exists(translation_id):
            state_manager.update_stats(translation_id, {
                'openrouter_cost': cost_data['session_cost'],
                'openrouter_prompt_tokens': cost_data['total_prompt_tokens'],
                'openrouter_completion_tokens': cost_data['total_completion_tokens']
            })
            current_stats = state_manager.get_translation_field(translation_id, 'stats') or {}
            current_stats['elapsed_time'] = time.time() - current_stats.get('start_time', time.time())
            emit_update(socketio, translation_id, {'stats': current_stats}, state_manager)

    # Setup OpenRouter cost callback if using OpenRouter provider
    if config.get('llm_provider') == 'openrouter':
        OpenRouterProvider.reset_session_cost()
        OpenRouterProvider.set_cost_callback(_openrouter_cost_callback)

    # Get checkpoint manager and handle resume
    checkpoint_manager = state_manager.get_checkpoint_manager()
    resume_from_index = config.get('resume_from_index', 0)
    is_resume = config.get('is_resume', False)

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
        custom_instructions_content = ""
        custom_instruction_file = config.get('prompt_options', {}).get('custom_instruction_file', '')

        if custom_instruction_file:
            try:
                project_root = Path(os.getcwd())
                custom_instructions_dir = project_root / 'Custom_Instructions'

                # Security: validate filename pattern (prevent directory traversal)
                safe_filename_pattern = r'^[a-zA-Z0-9_\-\.]+\.txt$'
                if re.match(safe_filename_pattern, custom_instruction_file):
                    file_path = custom_instructions_dir / custom_instruction_file

                    # Security: ensure file is within Custom_Instructions folder
                    try:
                        file_path.resolve().relative_to(custom_instructions_dir.resolve())

                        if file_path.exists() and file_path.is_file():
                            with open(file_path, 'r', encoding='utf-8') as f:
                                custom_instructions_content = f.read().strip()

                            if custom_instructions_content:
                                _log_message_callback("custom_instructions", f"📝 Loaded custom instructions: {custom_instruction_file}")
                    except ValueError:
                        pass  # Silently ignore invalid paths (security requirement)
            except Exception as e:
                pass  # Silently ignore errors (requirement from plan)

        # Inject custom instructions into prompt_options
        if custom_instructions_content:
            # Ensure prompt_options exists
            if 'prompt_options' not in config:
                config['prompt_options'] = {}

            # For refinement prompts (existing mechanism - already fully supported)
            config['prompt_options']['refinement_instructions'] = custom_instructions_content
            # For translation prompts (new)
            config['prompt_options']['custom_instructions'] = custom_instructions_content

        # Use unified adapter-based translation
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
            stats_callback=_update_translation_stats_callback,
            check_interruption_callback=should_interrupt_current_task,
            resume_from_index=resume_from_index,
            llm_api_endpoint=config['llm_api_endpoint'],
            gemini_api_key=config.get('gemini_api_key', ''),
            openai_api_key=config.get('openai_api_key', ''),
            openrouter_api_key=config.get('openrouter_api_key', ''),
            mistral_api_key=config.get('mistral_api_key', ''),
            deepseek_api_key=config.get('deepseek_api_key', ''),
            poe_api_key=config.get('poe_api_key', ''),
            context_window=config.get('context_window', 2048),
            auto_adjust_context=config.get('auto_adjust_context', True),
            min_chunk_size=config.get('min_chunk_size', 5),
            prompt_options=config.get('prompt_options', {}),
            bilingual_output=config.get('bilingual_output', False)
        )

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
        _update_translation_stats_callback({'elapsed_time': elapsed_time})

        final_status_payload = {
            'result': state_manager.get_translation_field(translation_id, 'result'),
            'output_filename': config['output_filename'],
            'file_type': config['file_type']
        }

        if state_manager.get_translation_field(translation_id, 'interrupted'):
            state_manager.set_translation_field(translation_id, 'status', 'interrupted')
            _log_message_callback("summary_interrupted", f"🛑 Translation interrupted - partial result saved ({elapsed_time:.2f}s)")
            final_status_payload['status'] = 'interrupted'

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
            state_manager.set_translation_field(translation_id, 'status', 'completed')

            # Get stats for consolidated message
            final_stats = stats
            stats_summary = ""
            if config['file_type'] == 'txt' or (config['file_type'] == 'epub' and stats.get('total_chunks', 0) > 0):
                completed = final_stats.get('completed_chunks', 0)
                failed = final_stats.get('failed_chunks', 0)
                total = final_stats.get('total_chunks', 0)
                stats_summary = f" | {completed}/{total} chunks"
                if failed > 0:
                    stats_summary += f" ({failed} failed)"
            elif config['file_type'] == 'srt' and stats.get('total_subtitles', 0) > 0:
                completed = final_stats.get('completed_subtitles', 0)
                failed = final_stats.get('failed_subtitles', 0)
                total = final_stats.get('total_subtitles', 0)
                stats_summary = f" | {completed}/{total} subtitles"
                if failed > 0:
                    stats_summary += f" ({failed} failed)"

            # Single consolidated completion message
            _log_message_callback("summary_completed", f"✅ Translation completed in {elapsed_time:.2f}s{stats_summary}")

            final_status_payload['status'] = 'completed'

            # Cleanup completed job checkpoint (automatic immediate cleanup)
            checkpoint_manager.cleanup_completed_job(translation_id)

            # Clean up uploaded file if it exists and is in the uploads directory
            # On completion, we can safely delete the original upload file
            if 'file_path' in config and config['file_path']:
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

        emit_update(socketio, translation_id, final_status_payload, state_manager)

        # Trigger file list refresh in the frontend if a file was saved
        if os.path.exists(output_filepath_on_server) and final_status_payload['status'] in ['completed', 'interrupted']:
            socketio.emit('file_list_changed', {
                'reason': final_status_payload['status'],
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