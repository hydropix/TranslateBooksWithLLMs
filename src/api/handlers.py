"""
Web adapter of the job runner.

``perform_actual_translation`` hands a job to :class:`src.core.job.runner.JobRunner`
with web sinks: the state manager (in-memory job registry) and Socket.IO. All
orchestration (prepare, translate / refine / term fix, verdict, checkpoint
status, rate-limit pause and auto-resume, cleanup, notifications) lives in the
runner and is shared with the CLI.
"""
import asyncio
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path

from src.core.job import JobConfig, JobResult, JobStatus
from src.core.job.runner import JobRunner, JobSinks
from src.core.progress import snapshot_from_legacy_stats
from src.utils.file_utils import generate_tts_for_translation, get_unique_output_path
from src.utils.unified_logger import LogType, create_job_logger
from .services.path_validator import PathValidator
from .websocket import emit_update

# Diagnostics that must never reach the user-facing job log.
_logger = logging.getLogger("api.handlers")

# Longest string kept per field of a stored log entry's data. LLM request and
# response entries carry whole prompts and responses; the live view already
# received them in full over WebSocket, the stored copy only feeds the status
# endpoint.
_STORED_LOG_FIELD_MAX_CHARS = 2000

# Engine log keys that are debug noise in the web log.
_SKIPPED_LOG_KEYS = ("llm_prompt_debug", "llm_raw_response_preview")


def resolve_output_path(output_dir, output_filename, is_resume):
    """The file a web run writes to, inside ``output_dir``.

    A fresh job never overwrites (a " (1)" suffix); a resume rewrites the file
    the previous pass produced. The name is reduced to a safe bare filename
    and must resolve inside ``output_dir`` (ValueError otherwise).
    """
    path = str(PathValidator.resolve_output_file(output_dir, output_filename))
    return path if is_resume else get_unique_output_path(path)


def _compact_log_entry(log_entry):
    """Return a copy of ``log_entry`` with long data strings truncated."""
    data = log_entry.get('data') if isinstance(log_entry, dict) else None
    if not isinstance(data, dict):
        return log_entry
    compact = {}
    for key, value in data.items():
        if isinstance(value, str) and len(value) > _STORED_LOG_FIELD_MAX_CHARS:
            value = value[:_STORED_LOG_FIELD_MAX_CHARS] + ' [...]'
        compact[key] = value
    return {**log_entry, 'data': compact}


def _store_log_entry(state_manager, translation_id, entry):
    """Append to the job's in-memory log (bounded by the state manager)."""
    append = getattr(state_manager, 'append_log', None)
    if append is not None:
        append(translation_id, entry)
        return
    logs = state_manager.get_translation_field(translation_id, 'logs') or []
    logs.append(entry)
    state_manager.set_translation_field(translation_id, 'logs', logs)


def _mark_job_error_in_db(state_manager, translation_id, checkpoint_manager=None):
    """Record a terminal 'error' status in SQLite, never raising (a row left
    at 'running' is excluded from the resumable list until a restart)."""
    try:
        manager = checkpoint_manager or state_manager.get_checkpoint_manager()
        if manager is not None:
            manager.mark_error(translation_id)
    except Exception as exc:  # pragma: no cover - best effort only
        _logger.warning("Could not mark job %s as error in the database: %s", translation_id, exc)


class WebJobSinks(JobSinks):
    """Runner sinks for a web job: state manager + Socket.IO + a per-job logger."""

    def __init__(self, translation_id, state_manager, socketio):
        self.translation_id = translation_id
        self.state = state_manager
        self.socketio = socketio
        # Monotonic floor on the canonical percent: the bar never moves back,
        # across the translate -> refine boundary included.
        self._progress_floor = 0.0
        # The logger calls both callbacks for every entry: the web callback
        # only emits (full entry), the storage callback keeps a compact copy.
        # A logger of its own: concurrent jobs never cross-log.
        self.logger = create_job_logger(self._web_log, self._store_log)

    # -- logging -----------------------------------------------------------

    def _web_log(self, log_entry):
        self._emit({'log': log_entry['message'], 'log_entry': log_entry})

    def _store_log(self, log_entry):
        _store_log_entry(self.state, self.translation_id, _compact_log_entry(log_entry))

    def log(self, key, message="", data=None):
        if key in _SKIPPED_LOG_KEYS:
            return
        logger = self.logger
        if data and isinstance(data, dict):
            log_type = data.get('type')
            if log_type == 'llm_request':
                logger.debug("LLM Request", LogType.LLM_REQUEST, data)
            elif log_type == 'llm_response':
                # INFO so the translation preview works with DEBUG_MODE off.
                logger.info("LLM Response", LogType.LLM_RESPONSE, data)
            elif log_type == 'refinement_request':
                logger.debug("Refinement Request", LogType.LLM_REQUEST, data)
            elif log_type == 'refinement_response':
                logger.info("Refinement Response", LogType.LLM_RESPONSE, data)
            elif log_type == 'progress':
                logger.info("Progress Update", LogType.PROGRESS, data)
            else:
                logger.info(message, data=data)
        elif "error" in key.lower():
            logger.error(message)
        elif "warning" in key.lower():
            logger.warning(message)
        else:
            logger.info(message)

    # -- state -------------------------------------------------------------

    def _emit(self, payload):
        emit_update(self.socketio, self.translation_id, payload, self.state)

    def _socket_event(self, name, payload):
        if self.socketio is not None:
            self.socketio.emit(name, payload, namespace='/')

    def _exists(self):
        return self.state.exists(self.translation_id)

    def emit_stats(self, stats):
        """Merge, derive the canonical progress fields (the one place the
        legacy stats and the unified snapshot meet), emit."""
        if not self._exists():
            return

        def finalize(current):
            current['elapsed_time'] = time.time() - current.get('start_time', time.time())
            snapshot = snapshot_from_legacy_stats(current).to_dict()
            if snapshot['percent'] < self._progress_floor:
                snapshot['percent'] = self._progress_floor
            else:
                self._progress_floor = snapshot['percent']
            current.update(snapshot)

        merged = self._merge_stats(stats, finalize)
        if merged is None:
            return
        self._emit({'stats': merged})
        total = merged.get('total_chunks', 0)
        if total > 0:
            self.logger.update_progress(merged.get('completed_chunks', 0), total)

    def _merge_stats(self, stats, finalize):
        merge = getattr(self.state, 'merge_stats', None)
        if merge is not None:
            return merge(self.translation_id, stats, finalize)
        # Minimal state managers (tests).
        self.state.update_stats(self.translation_id, stats)
        current = dict(self.state.get_translation_field(self.translation_id, 'stats') or {})
        finalize(current)
        self.state.set_translation_field(self.translation_id, 'stats', current)
        return current

    def current_stats(self):
        get = getattr(self.state, 'get_stats', None)
        if get is not None:
            return get(self.translation_id)
        return dict(self.state.get_translation_field(self.translation_id, 'stats') or {})

    def is_interrupted(self):
        return bool(self._exists() and self.state.get_translation_field(self.translation_id, 'interrupted'))

    def set_status(self, status, message=None):
        if not self._exists():
            return
        self.state.set_translation_field(self.translation_id, 'status', status)
        payload = {'status': status}
        if message:
            payload['log'] = message
        self._emit(payload)

    def job_changed(self, job):
        if self._exists():
            self.state.set_translation_field(self.translation_id, 'config', job.to_persisted())

    def job_started(self, job, is_resume):
        self.logger.info("Translation Started", LogType.TRANSLATION_START, {
            'source_lang': job.source_language,
            'target_lang': job.target_language,
            'file_type': job.file_type.upper(),
            'model': job.model,
            'translation_id': self.translation_id,
            'output_file': job.output_filename,
            'api_endpoint': job.endpoint,
            'chunk_size': job.max_tokens_per_chunk or 'default',
        })

    def on_cost(self, cost_data):
        """No emit here: this runs on the provider call, and an emit could
        overtake the stats emit on the wire. The next stats emit carries it."""
        if self._exists():
            self.state.update_stats(self.translation_id, {
                'openrouter_cost': cost_data['session_cost'],
                'openrouter_prompt_tokens': cost_data['total_prompt_tokens'],
                'openrouter_completion_tokens': cost_data['total_completion_tokens'],
                'openrouter_cost_estimated': cost_data.get('cost_is_estimated', False),
            })

    # -- terminal outcomes -------------------------------------------------

    def job_finished(self, job, result: JobResult, elapsed):
        if not self._exists():
            return
        tid = self.translation_id
        output_path = result.output_path or job.output_path
        status = result.status.web_status
        file_type_upper = job.file_type.upper()
        if os.path.exists(output_path) and status != 'error':
            self.state.set_translation_field(tid, 'result', f"[{file_type_upper} file translated - download to view]")
        elif not os.path.exists(output_path):
            self.state.set_translation_field(
                tid, 'result',
                f"[{file_type_upper} file (partially) translated - content not loaded for preview or write failed]")
        self.state.set_translation_field(tid, 'output_filepath', output_path)
        self.state.set_translation_field(tid, 'status', status)
        if result.error and status == 'error':
            self.state.set_translation_field(tid, 'error', result.error)

        payload = {
            'result': self.state.get_translation_field(tid, 'result'),
            'output_filename': job.output_filename,
            'output_dir': os.path.dirname(os.path.abspath(output_path)),
            'file_type': job.file_type,
            # A partial job keeps its checkpoint; the completion card offers a
            # retry on another model, seeded with this. Never an API key.
            'translation_id': tid,
            'model': job.model,
            'llm_provider': job.llm_provider,
            'status': status,
        }
        if job.endpoint:
            payload['llm_api_endpoint'] = job.endpoint
        if status == 'error':
            payload['error'] = result.error
        if result.status is JobStatus.INTERRUPTED:
            self._socket_event('checkpoint_created', {
                'translation_id': tid,
                'status': 'interrupted',
                'message': 'Translation paused - checkpoint created',
            })
        payload['stats'] = self.current_stats()
        self._emit(payload)
        if os.path.exists(output_path) and status in ('completed', 'interrupted', 'partial'):
            self._socket_event('file_list_changed', {'reason': status, 'filename': job.output_filename})

    def job_rate_limited(self, job, message):
        if not self._exists():
            return
        tid = self.translation_id
        self.state.set_translation_field(tid, 'status', 'rate_limited')
        self.state.set_translation_field(tid, 'interrupted', True)
        self._emit({
            'status': 'rate_limited',
            'log': message,
            'result': self.state.get_translation_field(tid, 'result') or "Translation paused (rate limited)",
        })
        self._socket_event('checkpoint_created', {
            'translation_id': tid, 'status': 'rate_limited', 'message': message,
        })
        output_path = self.state.get_translation_field(tid, 'output_filepath') or job.output_path
        if output_path and os.path.exists(output_path):
            self._socket_event('file_list_changed', {'reason': 'rate_limited', 'filename': job.output_filename})

    def job_failed(self, job, error):
        if not self._exists():
            return
        tid = self.translation_id
        self.state.set_translation_field(tid, 'status', 'error')
        self.state.set_translation_field(tid, 'error', error)
        self._emit({
            'error': error,
            'status': 'error',
            'result': self.state.get_translation_field(tid, 'result') or f"Translation failed: {error}",
        })

    async def run_tts(self, job, output_path):
        await _perform_tts_generation(self.translation_id, job, output_path, self.state,
                                      self.socketio, self.log)


def run_translation_async_wrapper(translation_id, job, state_manager, output_dir, socketio,
                                  is_resume=False):
    """Thread body of a web job: its own event loop, one worker per job."""
    # The worker registry keeps a second worker off this job for as long as
    # this thread lives (auto-resume waits included). The resume route claims
    # the slot before starting us; a fresh job is recorded here.
    mark_active = getattr(state_manager, 'mark_worker_active', None)
    if mark_active is not None:
        mark_active(translation_id)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        _run_worker_loop(loop, translation_id, job, state_manager, output_dir, socketio, is_resume)
    finally:
        release = getattr(state_manager, 'release_worker', None)
        if release is not None:
            release(translation_id)


def _run_worker_loop(loop, translation_id, job, state_manager, output_dir, socketio, is_resume=False):
    try:
        loop.run_until_complete(perform_actual_translation(
            translation_id, job, state_manager, output_dir, socketio, is_resume=is_resume))
    except Exception as e:
        error_msg = f"Uncaught major error in translation wrapper {translation_id}: {str(e)}"
        if state_manager.exists(translation_id):
            state_manager.set_translation_field(translation_id, 'status', 'error')
            state_manager.set_translation_field(translation_id, 'error', error_msg)
            _store_log_entry(
                state_manager, translation_id,
                f"[{datetime.now().strftime('%H:%M:%S')}] CRITICAL WRAPPER ERROR: {error_msg}")
            emit_update(socketio, translation_id, {'error': error_msg, 'status': 'error',
                                                   'log': f"CRITICAL WRAPPER ERROR: {error_msg}"},
                        state_manager)
            _mark_job_error_in_db(state_manager, translation_id)
    finally:
        loop.close()


async def perform_actual_translation(translation_id, job, state_manager, output_dir, socketio,
                                     is_resume=False):
    """Run one web job through the shared :class:`JobRunner`.

    Args:
        translation_id: Job id.
        job: The :class:`JobConfig` (or its persisted dict form, keys from .env).
        state_manager: The in-memory job registry.
        output_dir: Server output directory (uploads live in ``<output_dir>/uploads``).
        socketio: Socket.IO server, or None.
        is_resume: Continue the stored job instead of starting it.
    """
    if not state_manager.exists(translation_id):
        return None
    if not isinstance(job, JobConfig):
        job = JobConfig.from_persisted(job, output_dir=str(output_dir))

    state_manager.set_translation_field(translation_id, 'status', 'running')
    emit_update(socketio, translation_id, {'status': 'running', 'log': 'Translation task started by worker.'},
                state_manager)

    runner = JobRunner(
        job,
        translation_id=translation_id,
        checkpoint_manager=state_manager.get_checkpoint_manager(),
        sinks=WebJobSinks(translation_id, state_manager, socketio),
        is_resume=is_resume,
        glossary_store=getattr(state_manager, 'get_glossary_store', None),
        custom_instructions_dir=Path(os.getcwd()) / 'Custom_Instructions',
        temp_dir=str(output_dir),
    )
    return await runner.run()


async def _perform_tts_generation(translation_id, job, output_filepath, state_manager, socketio, log_callback):
    """Generate audio for a finished web job, reporting over Socket.IO."""
    from src.tts.tts_config import TTSConfig

    def emit(payload):
        if socketio is not None:
            socketio.emit('tts_update', {'translation_id': translation_id, **payload}, namespace='/')

    try:
        log_callback("tts_phase_start", "🔊 Starting TTS audio generation...")
        emit({'status': 'started', 'message': 'TTS generation started'})
        tts_config = TTSConfig.from_dict(dict(job.tts or {}))

        def tts_progress_callback(current, total, message):
            progress_pct = int((current / total) * 100) if total > 0 else 0
            log_callback("tts_chunk_progress", f"🔊 TTS: {message}")
            emit({'status': 'processing', 'progress': progress_pct, 'current_chunk': current,
                  'total_chunks': total, 'message': message})

        success, message, audio_path = await generate_tts_for_translation(
            translated_filepath=output_filepath,
            target_language=job.target_language or '',
            tts_config=tts_config,
            log_callback=log_callback,
            progress_callback=tts_progress_callback,
        )
        if success:
            log_callback("tts_complete", f"✅ TTS audio generated: {os.path.basename(audio_path)}")
            state_manager.set_translation_field(translation_id, 'audio_filepath', audio_path)
            state_manager.set_translation_field(translation_id, 'audio_filename', os.path.basename(audio_path))
            emit({'status': 'completed', 'progress': 100, 'audio_filename': os.path.basename(audio_path),
                  'message': 'TTS generation completed successfully'})
            if socketio is not None:
                socketio.emit('file_list_changed', {'reason': 'tts_completed',
                                                    'filename': os.path.basename(audio_path)}, namespace='/')
        else:
            log_callback("tts_failed", f"❌ TTS generation failed: {message}")
            emit({'status': 'failed', 'error': message, 'message': f'TTS generation failed: {message}'})
    except Exception as e:
        error_msg = f"TTS generation error: {str(e)}"
        log_callback("tts_error", f"❌ {error_msg}")
        emit({'status': 'failed', 'error': error_msg, 'message': error_msg})


def start_translation_job(translation_id, job, state_manager, output_dir, socketio, is_resume=False):
    """Start a web job in its own daemon thread."""
    thread = threading.Thread(
        target=run_translation_async_wrapper,
        args=(translation_id, job, state_manager, output_dir, socketio),
        kwargs={'is_resume': is_resume},
    )
    thread.daemon = True
    thread.start()
