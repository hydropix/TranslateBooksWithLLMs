"""
Translation job management routes
"""
import logging
import time
from pathlib import Path
from flask import Blueprint, request, jsonify

from src.api.services.path_validator import PathValidator
from src.api.translation_state import new_job_id
from src.core.job import JobConfig, JobConfigError
from src.core.job.config import parse_term_fix_request
from src.core.job.output_name import provisional_name

logger = logging.getLogger('translation_routes')


def _strip_api_keys(config):
    """Remove every API key from a config dict in place (for API responses).

    Persisted configs and the in-memory job registry hold no keys any more
    (JobConfig.to_persisted strips them), so this is a defensive no-op on
    current data; it still guards configs written by older versions.
    """
    if isinstance(config, dict):
        for key in [k for k in config if k == 'api_key' or k.endswith('_api_key')]:
            config.pop(key, None)
    return config


def _config_error_response(error: JobConfigError):
    if error.status == 400 and 'endpoint' in str(error).lower():
        logger.warning("Rejected client-supplied endpoint: %s", error)
    return jsonify(error.payload), error.status


_HISTORY_FIELDS = (
    'translation_id',
    'status',
    'file_type',
    'input_filename',
    'output_filename',
    'created_at',
    'updated_at',
    'completed_at',
    'total_chunks',
    'completed_chunks',
)


#: Upper bound on the IDs one /api/history/clear call may name. The history
#: listing returns at most 200 rows, and this keeps the IN (...) clause well
#: below SQLite's bound-parameter limit.
_HISTORY_CLEAR_MAX_IDS = 500


def _history_item(job):
    """Project a completed job onto the fields the browser is allowed to see.

    Built as an allowlist, never by deleting keys from a copy of the job: the
    stored config carries the server's filesystem layout ('file_path'), the
    provider endpoints and the API keys resolved from .env, and a denylist
    would silently leak whatever field gets added to it next (issue #271).
    """
    return {field: job.get(field) for field in _HISTORY_FIELDS}


def create_translation_blueprint(state_manager, start_translation_job, output_dir):
    """
    Create and configure the translation blueprint

    Args:
        state_manager: Translation state manager instance
        start_translation_job: Function to start translation jobs
        output_dir: Base directory for file operations; uploaded source files
            live in '<output_dir>/uploads' and a client-supplied file_path must
            resolve inside it.
    """
    bp = Blueprint('translation', __name__)

    uploads_dir = Path(output_dir) / 'uploads'

    @bp.route('/api/translate', methods=['POST'])
    def start_translation_request():
        """Start a new translation job"""
        data = request.get_json(silent=True)
        try:
            job = JobConfig.from_web_request(data, output_dir=output_dir, uploads_dir=uploads_dir)
        except JobConfigError as e:
            return _config_error_response(e)

        translation_id = new_job_id()
        # The registry holds the persisted (secret-free) form; the worker gets
        # the JobConfig, whose credentials only live in memory.
        config = job.to_persisted()
        # A {translatedName} placeholder is resolved by the worker; until
        # then the job is listed under its original name, never the token.
        config['output_filename'] = provisional_name(job)
        state_manager.create_translation(translation_id, config)
        start_translation_job(translation_id, job)

        return jsonify({
            "translation_id": translation_id,
            "message": "Translation queued.",
            "config_received": _strip_api_keys(dict(config)),
        })

    @bp.route('/api/term-fix/scan', methods=['POST'])
    def scan_term_fix():
        """Count the matches of term-fix rules in an uploaded file (no LLM)."""
        import asyncio
        from src.core.adapters import scan_file
        from src.core.adapters.exceptions import UnsupportedFormatError
        from src.core.term_fix import scan_to_dict

        data = request.json or {}
        term_fix, term_fix_error = parse_term_fix_request(
            {'rules_text': data.get('rules_text'), 'mode': 'literal'}, check_mode=False)
        if term_fix_error is not None:
            return jsonify({"error": term_fix_error}), 400
        if not data.get('file_path'):
            return jsonify({"error": "Missing or empty field: file_path"}), 400
        safe_path, path_error = PathValidator.validate_upload_path(
            data['file_path'], uploads_dir
        )
        if path_error is not None:
            return jsonify({"error": path_error}), 403

        from src.core.term_fix import parse_rules_text
        rules = parse_rules_text(term_fix['rules_text'])
        try:
            results = asyncio.run(scan_file(str(safe_path), rules))
        except (UnsupportedFormatError, FileNotFoundError) as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"rules": scan_to_dict(results)})

    @bp.route('/api/translation/<translation_id>', methods=['GET'])
    def get_translation_job_status(translation_id):
        """Get status of a translation job"""
        job_data = state_manager.get_translation(translation_id)
        if not job_data:
            return jsonify({"error": "Translation not found"}), 404

        stats = job_data.get('stats', {
            'start_time': time.time(),
            'total_chunks': 0,
            'completed_chunks': 0,
            'failed_chunks': 0
        })

        # Calculate elapsed time
        if job_data.get('status') == 'running' or job_data.get('status') == 'queued':
            elapsed = time.time() - stats.get('start_time', time.time())
        else:
            elapsed = stats.get('elapsed_time', time.time() - stats.get('start_time', time.time()))

        return jsonify({
            "translation_id": translation_id,
            "status": job_data.get('status'),
            "progress": job_data.get('progress'),
            # Every stat the Socket.IO updates carry, so a reconnecting client
            # can resync its progress panel and replay a completion card.
            "stats": {
                **stats,
                'total_chunks': stats.get('total_chunks', 0),
                'completed_chunks': stats.get('completed_chunks', 0),
                'failed_chunks': stats.get('failed_chunks', 0),
                'start_time': stats.get('start_time'),
                'elapsed_time': elapsed
            },
            "logs": job_data.get('logs', [])[-100:],
            "result_preview": "[Preview functionality removed. Download file to view content.]" if job_data.get('status') in ['completed', 'interrupted', 'partial'] else None,
            "error": job_data.get('error'),
            "config": _strip_api_keys(dict(job_data['config'])) if job_data.get('config') else None,
            "output_filepath": job_data.get('output_filepath')
        })

    @bp.route('/api/translation/<translation_id>/interrupt', methods=['POST'])
    def interrupt_translation_job(translation_id):
        """Interrupt a running translation job"""
        if not state_manager.exists(translation_id):
            return jsonify({"error": "Translation not found"}), 404

        job_data = state_manager.get_translation(translation_id)
        status = job_data.get('status')
        if status in ('running', 'queued'):
            state_manager.set_interrupted(translation_id, True)
            return jsonify({
                "message": "Interruption signal sent. Translation will stop after the current segment."
            }), 200

        if status == 'rate_limited':
            # Cancels any in-flight auto-resume sleep and stops the UI from treating
            # the job as still-active.
            state_manager.set_interrupted(translation_id, True)
            state_manager.set_translation_field(translation_id, 'status', 'interrupted')
            return jsonify({
                "message": "Auto-resume cancelled. Translation marked interrupted; you can resume manually later."
            }), 200

        return jsonify({
            "message": "The translation is not in an interruptible state (e.g., already completed or failed)."
        }), 400

    @bp.route('/api/translations', methods=['GET'])
    def list_all_translations():
        """List all translation jobs"""
        summary_list = state_manager.get_translation_summaries()
        return jsonify({"translations": summary_list})

    @bp.route('/api/resumable', methods=['GET'])
    def list_resumable_jobs():
        """List all jobs that can be resumed.

        Persisted checkpoints no longer hold API keys (issue #213), but strip
        defensively anyway — the resume endpoint resolves keys server-side from
        .env or the request body, so the client never needs them.
        """
        resumable_jobs = state_manager.get_resumable_jobs()
        for job in resumable_jobs:
            _strip_api_keys(job.get('config'))
        return jsonify({"resumable_jobs": resumable_jobs})

    @bp.route('/api/history', methods=['GET'])
    def list_job_history():
        """List the most recently completed jobs, newest first.

        Returns a compact projection only (see _history_item): the history is
        a read-only display surface, so the browser gets no config, no server
        paths and no API keys.
        """
        limit = request.args.get('limit', '50')
        try:
            limit = int(limit, 10)
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid limit"}), 400

        jobs = state_manager.checkpoint_manager.get_job_history(limit)
        return jsonify({"history": [_history_item(job) for job in jobs]})

    @bp.route('/api/history/<translation_id>', methods=['DELETE'])
    def delete_history_entry(translation_id):
        """Remove one completed job from the history.

        Only the history row goes; the translated file stays in
        translated_files/. A paused or running job is not a history entry and
        answers 404, so this route can never drop a resumable checkpoint.
        """
        deleted = state_manager.checkpoint_manager.db.delete_completed_jobs([translation_id])
        if deleted < 0:
            return jsonify({"error": "Failed to delete history entry"}), 500
        if deleted == 0:
            return jsonify({"error": "History entry not found"}), 404
        return jsonify({"deleted": deleted}), 200

    @bp.route('/api/history/clear', methods=['POST'])
    def clear_history():
        """Remove several completed jobs from the history, or all of them.

        Body: {"translation_ids": [...]} to remove those entries, or no
        "translation_ids" key to clear the whole history. Translated files are
        never touched.
        """
        data = request.get_json(silent=True) or {}
        ids = data.get('translation_ids')
        if ids is not None:
            if (not isinstance(ids, list)
                    or len(ids) > _HISTORY_CLEAR_MAX_IDS
                    or not all(isinstance(i, str) and i for i in ids)):
                return jsonify({"error": "Invalid translation_ids"}), 400

        deleted = state_manager.checkpoint_manager.db.delete_completed_jobs(ids)
        if deleted < 0:
            return jsonify({"error": "Failed to clear history"}), 500
        return jsonify({"deleted": deleted}), 200

    @bp.route('/api/resume/<translation_id>', methods=['POST'])
    def resume_translation_job_endpoint(translation_id):
        """Resume a paused or interrupted translation job.

        The job is reserved for one worker before anything else happens, and
        released again on every path that does not start the worker: without
        it, a double click (or a manual resume while the job sleeps before an
        auto-resume) ran two workers on one checkpoint.
        """
        claim = getattr(state_manager, 'try_claim_worker', None)
        if claim is not None and not claim(translation_id):
            return jsonify({
                "error": "Cannot resume: this translation is already running",
                "message": "A worker is already processing this translation "
                           "(possibly waiting to auto-resume after a rate limit).",
            }), 409

        outcome = {'started': False}
        try:
            return _resume_claimed_job(translation_id, outcome)
        finally:
            if not outcome['started'] and claim is not None:
                state_manager.release_worker(translation_id)

    def _resume_claimed_job(translation_id, outcome):
        """Body of the resume route. Sets outcome['started'] once the worker
        thread has been started (it then owns the claim)."""
        # Check if there are any active translations. A 'rate_limited' job
        # that is not interrupted still has a worker asleep before its
        # auto-resume, so it counts as active.
        all_translations = state_manager.get_all_translations()
        active_translations = []
        for tid, tdata in all_translations.items():
            status = tdata.get('status')
            if status in ['running', 'queued'] or (
                    status == 'rate_limited' and not tdata.get('interrupted')):
                active_translations.append({
                    'id': tid,
                    'status': status,
                    'output_filename': tdata.get('config', {}).get('output_filename', 'unknown')
                })

        if active_translations:
            active_info = ', '.join([f"{t['output_filename']} ({t['status']})" for t in active_translations])
            return jsonify({
                "error": "Cannot resume: active translation in progress",
                "message": f"Please wait for active translation(s) to complete or interrupt them before resuming. Active: {active_info}",
                "active_translations": active_translations
            }), 409  # 409 Conflict status code

        # Check if checkpoint exists
        checkpoint_data = state_manager.checkpoint_manager.load_checkpoint(translation_id)
        if not checkpoint_data:
            return jsonify({"error": "No checkpoint found for this translation"}), 404

        # Restore job into state manager
        restored = state_manager.restore_job_from_checkpoint(translation_id)
        if not restored:
            return jsonify({"error": "Failed to restore job from checkpoint"}), 500

        stored_config = checkpoint_data['job']['config']

        # The preserved copy taken at job creation is the input of every
        # resume cycle.
        preserved_path = stored_config.get('preserved_input_path')
        if preserved_path:
            if not Path(preserved_path).exists():
                return jsonify({
                    "error": "Preserved input file not found",
                    "message": f"The preserved input file for this job no longer exists: {preserved_path}",
                    "suggestion": "This job cannot be resumed. Please delete this checkpoint and start a new translation."
                }), 404
            input_path = preserved_path
        else:
            input_path = state_manager.checkpoint_manager.get_preserved_input_path(translation_id)
            if not input_path:
                return jsonify({
                    "error": "No preserved input file",
                    "message": "This job has no preserved input file and cannot be resumed.",
                    "suggestion": "Please delete this checkpoint and start a new translation."
                }), 404

        # Optional model/provider overrides for the remaining work (issue
        # #183); credentials re-resolved and validated (issue #213).
        overrides = request.get_json(silent=True) or {}
        try:
            job = JobConfig.from_checkpoint(stored_config, output_dir=output_dir,
                                            input_path=input_path, overrides=overrides)
        except JobConfigError as e:
            return _config_error_response(e)

        state_manager.checkpoint_manager.mark_running(translation_id)
        start_translation_job(translation_id, job, is_resume=True)
        outcome['started'] = True

        resume_index = checkpoint_data['resume_from_index']
        if checkpoint_data.get('job_phase') == 'refine':
            resume_index = checkpoint_data.get('refine_resume_index', 0)
        return jsonify({
            "translation_id": translation_id,
            "message": "Translation resumed successfully",
            "resume_from_chunk": resume_index,
            "model": job.model,
            "llm_provider": job.llm_provider
        }), 200

    @bp.route('/api/checkpoint/<translation_id>', methods=['DELETE'])
    def delete_checkpoint_endpoint(translation_id):
        """Delete a checkpoint (manual cleanup by user)"""
        success = state_manager.delete_checkpoint(translation_id)

        if success:
            return jsonify({
                "message": "Checkpoint deleted successfully",
                "translation_id": translation_id
            }), 200
        else:
            return jsonify({"error": "Failed to delete checkpoint or checkpoint not found"}), 404

    return bp
