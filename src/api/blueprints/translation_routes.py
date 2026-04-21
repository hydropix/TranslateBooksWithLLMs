"""
Translation job management routes
"""
import os
import time
import copy
from flask import Blueprint, request, jsonify

from src.config import (
    REQUEST_TIMEOUT,
    OLLAMA_NUM_CTX
)
from src.tts.tts_config import TTSConfig


def _resolve_api_key(value, env_var_name):
    """
    Resolve API key value from request or environment.

    Args:
        value: Value from request (can be actual key, '__USE_ENV__', or empty)
        env_var_name: Name of environment variable to fall back to

    Returns:
        Resolved API key string
    """
    if value == '__USE_ENV__' or not value:
        # Use environment variable
        return os.getenv(env_var_name, '')
    return value


def create_translation_blueprint(state_manager, start_translation_job):
    """
    Create and configure the translation blueprint

    Args:
        state_manager: Translation state manager instance
        start_translation_job: Function to start translation jobs
    """
    bp = Blueprint('translation', __name__)

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
            'output_filename': data['output_filename'],
            'llm_provider': data.get('llm_provider', 'ollama'),
            'gemini_api_key': _resolve_api_key(data.get('gemini_api_key'), 'GEMINI_API_KEY'),
            'openai_api_key': _resolve_api_key(data.get('openai_api_key'), 'OPENAI_API_KEY'),
            'openrouter_api_key': _resolve_api_key(data.get('openrouter_api_key'), 'OPENROUTER_API_KEY'),
            'mistral_api_key': _resolve_api_key(data.get('mistral_api_key'), 'MISTRAL_API_KEY'),
            'deepseek_api_key': _resolve_api_key(data.get('deepseek_api_key'), 'DEEPSEEK_API_KEY'),
            'poe_api_key': _resolve_api_key(data.get('poe_api_key'), 'POE_API_KEY'),
            'nim_api_key': _resolve_api_key(data.get('nim_api_key'), 'NIM_API_KEY'),
            'fireworks_api_key': _resolve_api_key(data.get('fireworks_api_key'), 'FIREWORKS_API_KEY'),
            # Request pacing for rate-limited/free APIs (optional, mainly for CLI advanced users)
            'min_request_interval': float(data.get('min_request_interval', 0)),
            'adaptive_request_backoff': bool(data.get('adaptive_request_backoff', True)),
            'max_request_interval': float(data.get('max_request_interval', 5.0)),
            # Prompt options (optional instructions to include in the system prompt)
            'prompt_options': data.get('prompt_options', {}),
            # Bilingual output (original + translation interleaved)
            'bilingual_output': data.get('bilingual_output', False),
            # TTS configuration
            'tts_enabled': data.get('tts_enabled', False),
            'tts_config': TTSConfig.from_web_request(data).to_dict() if data.get('tts_enabled') else None
        }

        # Add file-specific or text-specific configuration
        if 'file_path' in data:
            config['file_path'] = data['file_path']
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
            "config_received": config
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
            "result_preview": "[Preview functionality removed. Download file to view content.]" if job_data.get('status') in ['completed', 'interrupted'] else None,
            "error": job_data.get('error'),
            "config": job_data.get('config'),
            "output_filepath": job_data.get('output_filepath')
        })

    @bp.route('/api/translation/<translation_id>/interrupt', methods=['POST'])
    def interrupt_translation_job(translation_id):
        """Interrupt a running translation job"""
        if not state_manager.exists(translation_id):
            return jsonify({"error": "Translation not found"}), 404

        job_data = state_manager.get_translation(translation_id)
        if job_data.get('status') == 'running' or job_data.get('status') == 'queued':
            state_manager.set_interrupted(translation_id, True)
            return jsonify({
                "message": "Interruption signal sent. Translation will stop after the current segment."
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
        """List all jobs that can be resumed"""
        resumable_jobs = state_manager.get_resumable_jobs()
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

        # Mark as running in database
        state_manager.checkpoint_manager.mark_running(translation_id)

        # Start the translation job (the wrapper will inject dependencies)
        start_translation_job(translation_id, config)

        return jsonify({
            "translation_id": translation_id,
            "message": "Translation resumed successfully",
            "resume_from_chunk": checkpoint_data['resume_from_index']
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
