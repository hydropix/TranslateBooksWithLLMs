"""
Configuration and health check routes
"""
import os
import sys
import asyncio
import logging
import requests
import re
import time
from flask import Blueprint, request, jsonify, send_from_directory
from pathlib import Path


def get_base_path():
    """Get base path for resources (templates, static files)"""
    # In PyInstaller bundle, use the temporary extraction directory
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.getcwd()


def get_config_path():
    """Get base path for configuration files (.env)"""
    return os.getcwd()

from src.config import (
    API_ENDPOINT as DEFAULT_OLLAMA_API_ENDPOINT,
    DEFAULT_MODEL,
    REQUEST_TIMEOUT,
    OLLAMA_NUM_CTX,
    MAX_TRANSLATION_ATTEMPTS,
    DEFAULT_SOURCE_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
    DEBUG_MODE,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OPENAI_API_KEY,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    MISTRAL_API_KEY,
    MISTRAL_MODEL,
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    MAX_TOKENS_PER_CHUNK
)

# Setup logger for this module
logger = logging.getLogger('config_routes')
if DEBUG_MODE:
    logger.setLevel(logging.DEBUG)


def create_config_blueprint(server_session_id=None):
    """Create and configure the config blueprint

    Args:
        server_session_id: Server session ID from state manager (optional, generates new if not provided)
    """
    bp = Blueprint('config', __name__)

    # Store server startup time/session ID to detect restarts
    # Use provided session_id from state_manager if available, otherwise generate new
    # Ensure it's an integer for consistency with health check response
    startup_time = int(server_session_id) if server_session_id else int(time.time())

    @bp.route('/')
    def serve_interface():
        """Serve the main translation interface"""
        base_path = get_base_path()
        templates_dir = os.path.join(base_path, 'src', 'web', 'templates')
        interface_path = os.path.join(templates_dir, 'translation_interface.html')
        if os.path.exists(interface_path):
            return send_from_directory(templates_dir, 'translation_interface.html')
        return f"<h1>Error: Interface not found</h1><p>Looked in: {interface_path}</p>", 404

    @bp.route('/api/health', methods=['GET'])
    def health_check():
        """API health check endpoint"""
        return jsonify({
            "status": "ok",
            "message": "Translation API is running",
            "translate_module": "loaded",
            "ollama_default_endpoint": DEFAULT_OLLAMA_API_ENDPOINT,
            "supported_formats": ["txt", "epub", "srt"],
            "startup_time": startup_time,  # Used to detect server restarts
            "session_id": startup_time  # Alias for compatibility with LifecycleManager
        })

    @bp.route('/api/models', methods=['GET', 'POST'])
    def get_available_models():
        """Get available models from Ollama, Gemini, or OpenRouter

        Supports both GET and POST methods:
        - GET: For Ollama (no API key needed) or legacy calls
        - POST: For providers requiring API keys (Gemini, OpenRouter) - more secure
        """
        if request.method == 'POST':
            data = request.get_json() or {}
            provider = data.get('provider', 'ollama')
            api_key = data.get('api_key')
        else:
            # GET method - for Ollama or legacy compatibility
            provider = request.args.get('provider', 'ollama')
            api_key = request.args.get('api_key')

        if provider == 'gemini':
            return _get_gemini_models(api_key)
        elif provider == 'openrouter':
            return _get_openrouter_models(api_key)
        elif provider == 'mistral':
            return _get_mistral_models(api_key)
        elif provider == 'deepseek':
            return _get_deepseek_models(api_key)
        elif provider == 'openai':
            # Get endpoint from request for LM Studio support
            if request.method == 'POST':
                api_endpoint = data.get('api_endpoint', 'https://api.openai.com/v1/chat/completions')
            else:
                api_endpoint = request.args.get('api_endpoint', 'https://api.openai.com/v1/chat/completions')
            return _get_openai_models(api_key, api_endpoint)
        else:
            return _get_ollama_models()

    @bp.route('/api/config', methods=['GET'])
    def get_default_config():
        """Get default configuration values"""
        # For API keys, send a masked indicator if configured, empty string if not
        # This prevents browser autocomplete from filling in random values
        def mask_api_key(key):
            """Return masked indicator if key exists, empty string otherwise"""
            if key and len(key) > 4:
                return "***" + key[-4:]  # Show last 4 chars as indicator
            return ""  # Empty = not configured

        config_response = {
            "api_endpoint": DEFAULT_OLLAMA_API_ENDPOINT,
            "default_model": DEFAULT_MODEL,
            "timeout": REQUEST_TIMEOUT,
            "context_window": OLLAMA_NUM_CTX,
            "max_attempts": MAX_TRANSLATION_ATTEMPTS,
            "retry_delay": 2,
            "supported_formats": ["txt", "epub", "srt"],
            "gemini_api_key": mask_api_key(GEMINI_API_KEY),
            "openai_api_key": mask_api_key(OPENAI_API_KEY),
            "openrouter_api_key": mask_api_key(OPENROUTER_API_KEY),
            "mistral_api_key": mask_api_key(MISTRAL_API_KEY),
            "deepseek_api_key": mask_api_key(DEEPSEEK_API_KEY),
            "gemini_api_key_configured": bool(GEMINI_API_KEY),
            "openai_api_key_configured": bool(OPENAI_API_KEY),
            "openrouter_api_key_configured": bool(OPENROUTER_API_KEY),
            "mistral_api_key_configured": bool(MISTRAL_API_KEY),
            "deepseek_api_key_configured": bool(DEEPSEEK_API_KEY)
        }

        return jsonify(config_response)

    @bp.route('/api/config/max-tokens', methods=['GET'])
    def get_max_tokens():
        """Get MAX_TOKENS_PER_CHUNK configuration value for UI preview height adjustment"""
        return jsonify({
            "max_tokens_per_chunk": MAX_TOKENS_PER_CHUNK
        })

    def _resolve_api_key(provided_key, env_var_name, config_default):
        """Resolve API key from provided value, .env marker, or config default

        Args:
            provided_key: Key from request (could be actual key or '__USE_ENV__')
            env_var_name: Environment variable name to check
            config_default: Default value from config

        Returns:
            Resolved API key or None
        """
        if provided_key and provided_key != '__USE_ENV__':
            return provided_key
        # Use .env value if marker provided or no key given
        return os.getenv(env_var_name, config_default)

    def _get_openrouter_models(provided_api_key=None):
        """Get available text-only models from OpenRouter API"""
        api_key = _resolve_api_key(provided_api_key, 'OPENROUTER_API_KEY', OPENROUTER_API_KEY)

        # Use OPENROUTER_MODEL from .env, fallback to claude-sonnet-4
        default_model = OPENROUTER_MODEL if OPENROUTER_MODEL else "anthropic/claude-sonnet-4"

        if not api_key:
            return jsonify({
                "models": [],
                "model_names": [],
                "default": default_model,
                "status": "api_key_missing",
                "count": 0,
                "error": "OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or pass api_key parameter."
            })

        try:
            from src.core.llm import OpenRouterProvider

            openrouter_provider = OpenRouterProvider(api_key=api_key)
            models = asyncio.run(openrouter_provider.get_available_models(text_only=True))

            if models:
                model_names = [m['id'] for m in models]
                # Check if default model exists in available models
                if default_model not in model_names and model_names:
                    default_model = model_names[0]
                return jsonify({
                    "models": models,
                    "model_names": model_names,
                    "default": default_model,
                    "status": "openrouter_connected",
                    "count": len(models)
                })
            else:
                return jsonify({
                    "models": [],
                    "model_names": [],
                    "default": default_model,
                    "status": "openrouter_error",
                    "count": 0,
                    "error": "Failed to retrieve OpenRouter models"
                })

        except Exception as e:
            return jsonify({
                "models": [],
                "model_names": [],
                "default": default_model,
                "status": "openrouter_error",
                "count": 0,
                "error": f"Error connecting to OpenRouter API: {str(e)}"
            })

    def _get_mistral_models(provided_api_key=None):
        """Get available models from Mistral API"""
        api_key = _resolve_api_key(provided_api_key, 'MISTRAL_API_KEY', MISTRAL_API_KEY)

        # Use MISTRAL_MODEL from .env, fallback to mistral-large-latest
        default_model = MISTRAL_MODEL if MISTRAL_MODEL else "mistral-large-latest"

        if not api_key:
            return jsonify({
                "models": [],
                "model_names": [],
                "default": default_model,
                "status": "api_key_missing",
                "count": 0,
                "error": "Mistral API key is required. Set MISTRAL_API_KEY environment variable or pass api_key parameter."
            })

        try:
            from src.core.llm import MistralProvider

            mistral_provider = MistralProvider(api_key=api_key)
            models = asyncio.run(mistral_provider.get_available_models())

            if models:
                model_names = [m['id'] for m in models]
                # Check if default model exists in available models
                if default_model not in model_names and model_names:
                    default_model = model_names[0]
                return jsonify({
                    "models": models,
                    "model_names": model_names,
                    "default": default_model,
                    "status": "mistral_connected",
                    "count": len(models)
                })
            else:
                return jsonify({
                    "models": [],
                    "model_names": [],
                    "default": default_model,
                    "status": "mistral_error",
                    "count": 0,
                    "error": "Failed to retrieve Mistral models"
                })

        except Exception as e:
            return jsonify({
                "models": [],
                "model_names": [],
                "default": default_model,
                "status": "mistral_error",
                "count": 0,
                "error": f"Error connecting to Mistral API: {str(e)}"
            })

    def _get_deepseek_models(provided_api_key=None):
        """Get available models from DeepSeek API"""
        api_key = _resolve_api_key(provided_api_key, 'DEEPSEEK_API_KEY', DEEPSEEK_API_KEY)

        # Use DEEPSEEK_MODEL from .env, fallback to deepseek-chat
        default_model = DEEPSEEK_MODEL if DEEPSEEK_MODEL else "deepseek-chat"

        if not api_key:
            return jsonify({
                "models": [],
                "model_names": [],
                "default": default_model,
                "status": "api_key_missing",
                "count": 0,
                "error": "DeepSeek API key is required. Set DEEPSEEK_API_KEY environment variable or pass api_key parameter."
            })

        try:
            from src.core.llm import DeepSeekProvider

            deepseek_provider = DeepSeekProvider(api_key=api_key)
            models = asyncio.run(deepseek_provider.get_available_models())

            if models:
                model_names = [m['id'] for m in models]
                # Check if default model exists in available models
                if default_model not in model_names and model_names:
                    default_model = model_names[0]
                return jsonify({
                    "models": models,
                    "model_names": model_names,
                    "default": default_model,
                    "status": "deepseek_connected",
                    "count": len(models)
                })
            else:
                return jsonify({
                    "models": [],
                    "model_names": [],
                    "default": default_model,
                    "status": "deepseek_error",
                    "count": 0,
                    "error": "Failed to retrieve DeepSeek models"
                })

        except Exception as e:
            return jsonify({
                "models": [],
                "model_names": [],
                "default": default_model,
                "status": "deepseek_error",
                "count": 0,
                "error": f"Error connecting to DeepSeek API: {str(e)}"
            })

    def _get_openai_models(provided_api_key=None, api_endpoint=None):
        """Get available models from OpenAI-compatible API

        Always tries to fetch models dynamically from any OpenAI-compatible endpoint.
        Falls back to static list if dynamic fetch fails.
        """
        api_key = _resolve_api_key(provided_api_key, 'OPENAI_API_KEY', OPENAI_API_KEY)

        # Determine base URL from endpoint
        if api_endpoint:
            # Extract base URL (remove /chat/completions if present)
            base_url = api_endpoint.replace('/chat/completions', '').rstrip('/')
        else:
            base_url = 'https://api.openai.com/v1'

        # Static list of OpenAI models (fallback)
        openai_static_models = [
            {'id': 'gpt-4o', 'name': 'GPT-4o (Latest)'},
            {'id': 'gpt-4o-mini', 'name': 'GPT-4o Mini'},
            {'id': 'gpt-4-turbo', 'name': 'GPT-4 Turbo'},
            {'id': 'gpt-4', 'name': 'GPT-4'},
            {'id': 'gpt-3.5-turbo', 'name': 'GPT-3.5 Turbo'}
        ]

        try:
            models_url = f"{base_url}/models"
            headers = {}
            if api_key:
                headers['Authorization'] = f'Bearer {api_key}'

            response = requests.get(models_url, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                models_data = data.get('data', [])

                if models_data:
                    # Filter and format models
                    models = []
                    for m in models_data:
                        model_id = m.get('id', '')
                        # Skip embedding models and other non-chat models
                        if 'embedding' in model_id.lower() or 'whisper' in model_id.lower():
                            continue
                        models.append({
                            'id': model_id,
                            'name': model_id,
                            'owned_by': m.get('owned_by', 'unknown')
                        })

                    # Sort models by name
                    models.sort(key=lambda x: x['name'].lower())

                    if models:
                        model_ids = [m['id'] for m in models]
                        default_model = model_ids[0] if model_ids else 'gpt-4o'

                        return jsonify({
                            "models": models,
                            "model_names": model_ids,
                            "default": default_model,
                            "status": "openai_connected",
                            "count": len(models)
                        })

        except requests.exceptions.ConnectionError as e:
            pass

        except Exception as e:
            pass

        # Fallback: return static OpenAI models
        model_ids = [m['id'] for m in openai_static_models]
        return jsonify({
            "models": openai_static_models,
            "model_names": model_ids,
            "default": "gpt-4o",
            "status": "openai_static",
            "count": len(openai_static_models)
        })

    def _get_gemini_models(provided_api_key=None):
        """Get available models from Gemini API"""
        api_key = _resolve_api_key(provided_api_key, 'GEMINI_API_KEY', GEMINI_API_KEY)

        # Use GEMINI_MODEL from .env, fallback to gemini-2.0-flash
        default_model = GEMINI_MODEL if GEMINI_MODEL else "gemini-2.0-flash"

        if not api_key:
            return jsonify({
                "models": [],
                "default": default_model,
                "status": "api_key_missing",
                "count": 0,
                "error": "Gemini API key is required. Set GEMINI_API_KEY environment variable or pass api_key parameter."
            })

        try:
            from src.core.llm import GeminiProvider

            gemini_provider = GeminiProvider(api_key=api_key)
            models = asyncio.run(gemini_provider.get_available_models())

            if models:
                model_names = [m['name'] for m in models]
                # Check if default model exists in available models
                if default_model not in model_names and model_names:
                    default_model = model_names[0]
                return jsonify({
                    "models": models,
                    "model_names": model_names,
                    "default": default_model,
                    "status": "gemini_connected",
                    "count": len(models)
                })
            else:
                return jsonify({
                    "models": [],
                    "default": default_model,
                    "status": "gemini_error",
                    "count": 0,
                    "error": "Failed to retrieve Gemini models"
                })

        except Exception as e:
            return jsonify({
                "models": [],
                "default": default_model,
                "status": "gemini_error",
                "count": 0,
                "error": f"Error connecting to Gemini API: {str(e)}"
            })

    def _get_ollama_models():
        """Get available models from Ollama API"""
        ollama_base_from_ui = request.args.get('api_endpoint', DEFAULT_OLLAMA_API_ENDPOINT)

        try:
            base_url = ollama_base_from_ui.split('/api/')[0]
            tags_url = f"{base_url}/api/tags"

            response = requests.get(tags_url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                models_data = data.get('models', [])
                model_names = [m.get('name') for m in models_data if m.get('name')]

                return jsonify({
                    "models": model_names,
                    "default": DEFAULT_MODEL if DEFAULT_MODEL in model_names else (model_names[0] if model_names else DEFAULT_MODEL),
                    "status": "ollama_connected",
                    "count": len(model_names)
                })

        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection refused to {tags_url}. Is Ollama running?"
            print(f"❌ {error_msg}")
        except requests.exceptions.Timeout as e:
            error_msg = f"Timeout connecting to {tags_url} (10s)"
            print(f"❌ {error_msg}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Could not connect to Ollama at {ollama_base_from_ui}: {e}")
        except Exception as e:
            print(f"❌ Error retrieving models from {ollama_base_from_ui}: {e}")

        return jsonify({
            "models": [],
            "default": DEFAULT_MODEL,
            "status": "ollama_offline_or_error",
            "count": 0,
            "error": f"Ollama is not accessible at {ollama_base_from_ui} or an error occurred. Verify that Ollama is running ('ollama serve') and the endpoint is correct."
        })

    @bp.route('/api/model/warning', methods=['GET'])
    def get_model_warning():
        """
        Get thinking model warning for a specific model (instant lookup).

        This endpoint checks if a model is an uncontrollable thinking model
        and returns an appropriate warning message for the UI.

        Query params:
            model: Model name (e.g., "qwen3:30b")
            endpoint: Optional API endpoint (for cache differentiation)

        Returns:
            JSON with warning message if applicable, or null if no warning
        """
        model = request.args.get('model', '')
        endpoint = request.args.get('endpoint', '')

        if not model:
            return jsonify({"warning": None, "behavior": None})

        try:
            from src.core.llm import (
                get_model_warning_message,
                get_thinking_behavior_sync,
                ThinkingBehavior
            )

            warning = get_model_warning_message(model, endpoint)
            behavior = get_thinking_behavior_sync(model, endpoint)

            return jsonify({
                "warning": warning,
                "behavior": behavior.value if behavior else None,
                "is_uncontrollable": behavior == ThinkingBehavior.UNCONTROLLABLE if behavior else False,
                "is_thinking_model": behavior in [ThinkingBehavior.CONTROLLABLE, ThinkingBehavior.UNCONTROLLABLE] if behavior else False
            })

        except Exception as e:
            return jsonify({"warning": None, "behavior": None, "error": str(e)})

    def _get_env_file_path():
        """Get the path to the .env file"""
        config_path = get_config_path()
        return Path(config_path) / '.env'

    def _update_env_file(updates: dict) -> bool:
        """
        Update specific keys in the .env file.
        Creates the file if it doesn't exist.

        Args:
            updates: Dictionary of key-value pairs to update

        Returns:
            True if successful, False otherwise
        """
        env_path = _get_env_file_path()

        # Read existing content or start fresh
        existing_lines = []
        file_is_new = not env_path.exists()

        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                existing_lines = f.readlines()
        else:
            # Create file with header if it doesn't exist
            existing_lines = [
                "# Translation API Configuration\n",
                "# This file was automatically created by the web interface\n",
                "# You can edit these values manually or via the web UI\n",
                "\n"
            ]

        # Track which keys we've updated
        updated_keys = set()
        new_lines = []

        for line in existing_lines:
            stripped = line.strip()

            # Skip empty lines and comments, keep them as-is
            if not stripped or stripped.startswith('#'):
                new_lines.append(line)
                continue

            # Check if this line has a key we want to update
            match = re.match(r'^([A-Z_][A-Z0-9_]*)=', stripped)
            if match:
                key = match.group(1)
                if key in updates:
                    # Replace this line with new value
                    new_lines.append(f"{key}={updates[key]}\n")
                    updated_keys.add(key)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        # Add any keys that weren't in the file
        for key, value in updates.items():
            if key not in updated_keys:
                new_lines.append(f"{key}={value}\n")

        # Write back
        with open(env_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

        return True

    @bp.route('/api/settings', methods=['POST'])
    def save_settings():
        """
        Save user settings to .env file.

        Accepts JSON with settings to save. Only specific keys are allowed
        for security reasons.
        """
        allowed_keys = {
            'GEMINI_API_KEY',
            'GEMINI_MODEL',
            'OPENAI_API_KEY',
            'OPENROUTER_API_KEY',
            'OPENROUTER_MODEL',
            'MISTRAL_API_KEY',
            'MISTRAL_MODEL',
            'DEEPSEEK_API_KEY',
            'DEEPSEEK_MODEL',
            'DEFAULT_MODEL',
            'LLM_PROVIDER',
            'API_ENDPOINT'
        }

        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400

            # Filter to only allowed keys
            updates = {}
            for key, value in data.items():
                if key in allowed_keys:
                    # Sanitize value - remove newlines and dangerous characters
                    safe_value = str(value).replace('\n', '').replace('\r', '')
                    updates[key] = safe_value

            if not updates:
                return jsonify({"error": "No valid settings to save"}), 400

            # Update the .env file
            _update_env_file(updates)

            logger.info(f"Settings saved: {list(updates.keys())}")

            return jsonify({
                "success": True,
                "message": f"Saved {len(updates)} setting(s)",
                "saved_keys": list(updates.keys())
            })

        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            return jsonify({"error": f"Failed to save settings: {str(e)}"}), 500

    @bp.route('/api/settings', methods=['GET'])
    def get_settings():
        """
        Get current settings that can be modified via the UI.
        Returns only the keys that are user-configurable.
        API keys are masked for security - only indicates if configured.
        """
        return jsonify({
            "gemini_api_key_configured": bool(GEMINI_API_KEY),
            "openai_api_key_configured": bool(OPENAI_API_KEY),
            "openrouter_api_key_configured": bool(OPENROUTER_API_KEY),
            "mistral_api_key_configured": bool(MISTRAL_API_KEY),
            "deepseek_api_key_configured": bool(DEEPSEEK_API_KEY),
            "default_model": DEFAULT_MODEL or "",
            "llm_provider": os.getenv('LLM_PROVIDER', 'ollama'),
            "api_endpoint": DEFAULT_OLLAMA_API_ENDPOINT or ""
        })

    return bp
