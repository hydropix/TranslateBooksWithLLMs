"""
Translation job management routes
"""
import os
import time
import copy
from pathlib import Path
from flask import Blueprint, request, jsonify

from src.api.services.path_validator import PathValidator
from src.config import (
    REQUEST_TIMEOUT,
    OLLAMA_NUM_CTX,
    AUTO_PAUSE_ON_RATE_LIMIT,
    PARALLEL_TRANSLATIONS,
    MAX_PARALLEL_TRANSLATIONS,
)
from src.tts.tts_config import TTSConfig
from src.api.api_keys import resolve_api_key as _resolve_api_key


def _clamp_parallel_workers(value):
    """Clamp the requested worker count to [1, MAX_PARALLEL_TRANSLATIONS].

    Falls back to the PARALLEL_TRANSLATIONS default when absent or malformed.
    Local-provider gating happens later in resolve_parallel_workers().
    """
    if value is None:
        return PARALLEL_TRANSLATIONS
    try:
        return max(1, min(MAX_PARALLEL_TRANSLATIONS, int(value)))
    except (TypeError, ValueError):
        return PARALLEL_TRANSLATIONS


# Cloud providers whose key lives in config['<provider>_api_key'] and env var
# '<PROVIDER>_API_KEY'. The mapping is mechanical, so supporting a new provider
# in the resume-override path requires only adding it here (and nowhere else in
# this file).
_KEY_PROVIDERS = ('gemini', 'openai', 'openrouter', 'mistral', 'deepseek', 'poe', 'nim')

# Providers that talk to a user-supplied endpoint; the others use a built-in one.
_ENDPOINT_PROVIDERS = ('ollama', 'openai')


def _strip_api_keys(config):
    """Remove every API key from a config dict in place (for API responses).

    Persisted checkpoints no longer hold keys (issue #213), but the in-memory
    config of a live job does — it must never be echoed back to the browser,
    since '__USE_ENV__' requests get their key resolved from .env server-side.
    """
    if isinstance(config, dict):
        for key in [k for k in config if k == 'api_key' or k.endswith('_api_key')]:
            config.pop(key, None)
    return config


def _validate_provider_credentials(config):
    """Check that the provider in `config` has what it needs to run.

    A cloud provider needs a key from the config (resume override or live
    request) or from .env; an endpoint-driven provider needs an endpoint.
    Persisted checkpoints carry no keys (issue #213), so for a resumed job
    .env is the normal source — this guard turns a missing key into an
    immediate 400 instead of a mid-translation failure.

    Returns a Flask (response, status) tuple on failure, or None on success.
    """
    provider = (config.get('llm_provider') or 'ollama').lower()

    if provider in _KEY_PROVIDERS:
        env_var = f"{provider.upper()}_API_KEY"
        # 'openai' also covers OpenAI-compatible local endpoints (llama.cpp,
        # LM Studio, vLLM) where a key is legitimately absent — only require
        # one for the official API, mirroring the factory's heuristic.
        key_required = (provider != 'openai'
                        or 'api.openai.com' in (config.get('llm_api_endpoint') or ''))
        if key_required and not (config.get(f"{provider}_api_key") or os.getenv(env_var)):
            return jsonify({
                "error": "Missing API key for provider",
                "message": (f"Resuming with '{provider}' requires an API key. "
                            f"Set {env_var} in .env or include it in the request."),
            }), 400

    if provider in _ENDPOINT_PROVIDERS and not config.get('llm_api_endpoint'):
        return jsonify({
            "error": "Missing API endpoint for provider",
            "message": f"Resuming with '{provider}' requires an API endpoint.",
        }), 400

    return None


def _apply_resume_overrides(config, overrides):
    """Merge optional model/provider override fields into a resume config in place.

    Lets the resume request switch model/provider for the remaining chunks
    (issue #183). An empty/absent body leaves `config` untouched. API keys flow
    through `_resolve_api_key` exactly like the start endpoint, and a multi-key
    string is passed through unchanged so the key-rotation pool still works.

    Credentials are validated even with an empty body: checkpoints no longer
    persist API keys (issue #213), so every resume must find its key in .env
    or in the request.

    Returns a Flask (response, status) tuple to abort with on validation failure,
    or None on success.
    """
    if isinstance(overrides, dict) and overrides:
        if overrides.get('model'):
            config['model'] = overrides['model']
        if overrides.get('llm_provider'):
            config['llm_provider'] = str(overrides['llm_provider']).lower()
        if overrides.get('llm_api_endpoint'):
            config['llm_api_endpoint'] = overrides['llm_api_endpoint']
        if overrides.get('context_window') is not None:
            try:
                config['context_window'] = int(overrides['context_window'])
            except (TypeError, ValueError):
                return jsonify({"error": "context_window must be an integer"}), 400

        # A single generic api_key override maps to the chosen provider's key
        # field, resolved through .env like every other entry point.
        provider = (config.get('llm_provider') or 'ollama').lower()
        raw_key = overrides.get('api_key')
        if provider in _KEY_PROVIDERS and raw_key not in (None, ''):
            env_var = f"{provider.upper()}_API_KEY"
            config[f"{provider}_api_key"] = _resolve_api_key(raw_key, env_var)

    return _validate_provider_credentials(config)


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
        data = request.json

        # Validate required fields
        if 'file_path' in data:
            required_fields = ['file_path', 'source_language', 'target_language',
                             'model', 'llm_api_endpoint', 'output_filename', 'file_type']
        else:
            required_fields = ['text', 'source_language', 'target_language',
                             'model', 'llm_api_endpoint', 'output_filename']

        for field in required_fields:
            if field not in data or (isinstance(data[field], str) and not data[field].strip()) or (not isinstance(data[field], str) and data[field] is None):
                if field == 'text' and data.get('file_type') == 'txt' and data.get('text') == "":
                    pass
                else:
                    return jsonify({"error": f"Missing or empty field: {field}"}), 400

        # Generate unique translation ID
        translation_id = f"trans_{int(time.time() * 1000)}"

        # Build configuration
        config = {
            'source_language': data['source_language'],
            'target_language': data['target_language'],
            'model': data['model'],
            'llm_api_endpoint': data['llm_api_endpoint'],
            'request_timeout': int(data.get('timeout', REQUEST_TIMEOUT)),
            'context_window': int(data.get('context_window', OLLAMA_NUM_CTX)),
            'max_attempts': int(data.get('max_attempts', 2)),
            'retry_delay': int(data.get('retry_delay', 2)),
            'parallel_workers': _clamp_parallel_workers(data.get('parallel_workers')),
            'output_filename': data['output_filename'],
            'llm_provider': data.get('llm_provider', 'ollama'),
            'gemini_api_key': _resolve_api_key(data.get('gemini_api_key'), 'GEMINI_API_KEY'),
            'openai_api_key': _resolve_api_key(data.get('openai_api_key'), 'OPENAI_API_KEY'),
            'openrouter_api_key': _resolve_api_key(data.get('openrouter_api_key'), 'OPENROUTER_API_KEY'),
            # Prompt options (optional instructions to include in the system prompt)
            'prompt_options': data.get('prompt_options', {}),
            # Auto-pause on rate limit toggle (request overrides .env default)
            'auto_pause_on_rate_limit': data.get('auto_pause_on_rate_limit', AUTO_PAUSE_ON_RATE_LIMIT),
            # Bilingual output (original + translation interleaved)
            'bilingual_output': data.get('bilingual_output', False),
            # Refine-only mode (skip translation, run only refinement on input)
            'refine_only': data.get('refine_only', False),
            # Chained refinement pass after translation
            'refine_after': data.get('refine_after', False),
            # TTS configuration
            'tts_enabled': data.get('tts_enabled', False),
            'tts_config': TTSConfig.from_web_request(data).to_dict() if data.get('tts_enabled') else None
        }

        # Add file-specific or text-specific configuration
        if 'file_path' in data:
            # The client supplies this path, so it must be confined to the
            # uploads directory — otherwise any server-readable file (.env, SSH
            # keys, /etc/passwd) could be "translated" into a downloadable
            # output. See issue #209.
            safe_path, path_error = PathValidator.validate_upload_path(
                data['file_path'], uploads_dir
            )
            if path_error is not None:
                return jsonify({"error": path_error}), 403
            config['file_path'] = str(safe_path)
            config['file_type'] = data['file_type']
        else:
            config['text'] = data['text']
            config['file_type'] = data.get('file_type', 'txt')

        # Create translation in state manager
        state_manager.create_translation(translation_id, config)

        # Start translation job
        start_translation_job(translation_id, config)

        return jsonify({
            "translation_id": translation_id,
            "message": "Translation queued.",
            # Strip keys from a copy — `config` is the live job's dict, and a
            # '__USE_ENV__' request must not get the resolved .env key back.
            "config_received": _strip_api_keys(dict(config))
        })

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
            "stats": {
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

    @bp.route('/api/resume/<translation_id>', methods=['POST'])
    def resume_translation_job_endpoint(translation_id):
        """Resume a paused or interrupted translation job"""
        # Check if there are any active translations
        all_translations = state_manager.get_all_translations()
        active_translations = []
        for tid, tdata in all_translations.items():
            status = tdata.get('status')
            if status in ['running', 'queued']:
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

        # Get job config and add resume parameters
        job = checkpoint_data['job']
        config = copy.deepcopy(job['config'])  # Create a deep copy to avoid mutating the stored config

        # Get preserved input file path if exists
        # Always use preserved_input_path from config (stored during job creation)
        # This ensures consistent file path across multiple resume cycles
        preserved_path = config.get('preserved_input_path')
        if preserved_path:
            # Verify that the preserved file actually exists
            from pathlib import Path
            if Path(preserved_path).exists():
                config['file_path'] = preserved_path
            else:
                return jsonify({
                    "error": "Preserved input file not found",
                    "message": f"The preserved input file for this job no longer exists: {preserved_path}",
                    "suggestion": "This job cannot be resumed. Please delete this checkpoint and start a new translation."
                }), 404
        else:
            # Fallback: try to get it from checkpoint manager
            preserved_path_fallback = state_manager.checkpoint_manager.get_preserved_input_path(translation_id)
            if preserved_path_fallback:
                config['file_path'] = preserved_path_fallback
            else:
                return jsonify({
                    "error": "No preserved input file",
                    "message": "This job has no preserved input file and cannot be resumed.",
                    "suggestion": "Please delete this checkpoint and start a new translation."
                }), 404

        # Add resume parameters to config
        config['resume_from_index'] = checkpoint_data['resume_from_index']
        config['is_resume'] = True

        # Optional model/provider overrides for the remaining chunks (issue #183).
        # No body = unchanged behavior.
        overrides = request.get_json(silent=True) or {}
        override_error = _apply_resume_overrides(config, overrides)
        if override_error is not None:
            return override_error

        # Mark as running in database
        state_manager.checkpoint_manager.mark_running(translation_id)

        # Start the translation job (the wrapper will inject dependencies)
        start_translation_job(translation_id, config)

        return jsonify({
            "translation_id": translation_id,
            "message": "Translation resumed successfully",
            "resume_from_chunk": checkpoint_data['resume_from_index'],
            "model": config.get('model'),
            "llm_provider": config.get('llm_provider')
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
