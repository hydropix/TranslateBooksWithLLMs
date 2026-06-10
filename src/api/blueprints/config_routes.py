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
from urllib.parse import urlparse
from flask import Blueprint, request, jsonify, send_from_directory, render_template, make_response
from pathlib import Path

# UI locales served by /static/locales/<code>/*.json. Keep in sync with the
# SUPPORTED_LOCALES constant in src/web/static/js/i18n/i18n.js. The list is
# small enough that duplication beats a separate config file for now.
SUPPORTED_UI_LOCALES = ['en', 'fr', 'es', 'de', 'zh-CN', 'ja', 'ko']
UI_LOCALE_COOKIE = 'ui_locale'
UI_LOCALE_COOKIE_MAX_AGE = 60 * 60 * 24 * 365  # 1 year


def resolve_ui_locale(req):
    """Pick the UI locale to render the page with.

    Order: explicit cookie (user's last choice) → Accept-Language best match
    → 'en'. Returning the code is enough; i18next-http-backend loads the
    JSON files asynchronously from /static/locales/<code>/.
    """
    cookie_locale = req.cookies.get(UI_LOCALE_COOKIE)
    if cookie_locale in SUPPORTED_UI_LOCALES:
        return cookie_locale
    best = req.accept_languages.best_match(SUPPORTED_UI_LOCALES)
    return best or 'en'


def get_base_path():
    """Get base path for resources (templates, static files)"""
    # In PyInstaller bundle, use the temporary extraction directory
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.getcwd()


def get_config_path():
    """Get base path for configuration files (.env)"""
    return os.getcwd()

import src.config as _config
from src.config import reload_config
from src import __version__
from src.core.llm.base import normalize_api_keys

# Setup logger for this module
logger = logging.getLogger('config_routes')
if _config.DEBUG_MODE:
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
        """Serve the main translation interface.

        Switched from send_from_directory to render_template so Jinja can
        inject the initial UI locale into the HTML — that avoids the
        English flash on first paint when the user prefers another locale.
        """
        base_path = get_base_path()
        templates_dir = os.path.join(base_path, 'src', 'web', 'templates')
        interface_path = os.path.join(templates_dir, 'translation_interface.html')
        if not os.path.exists(interface_path):
            return f"<h1>Error: Interface not found</h1><p>Looked in: {interface_path}</p>", 404

        from src.api.auth import API_TOKEN
        return render_template(
            'translation_interface.html',
            initial_locale=resolve_ui_locale(request),
            supported_locales=SUPPORTED_UI_LOCALES,
            app_version=__version__,
            # Per-session token handed to the SPA so it can authenticate every
            # /api/ call and the Socket.IO handshake (issue #210). Same-origin
            # delivery means a foreign page cannot read it.
            api_token=API_TOKEN,
        )

    @bp.route('/api/ui-locale', methods=['POST'])
    def set_ui_locale():
        """Persist the user's UI locale choice in a long-lived cookie.

        The client also calls i18next.changeLanguage() to swap the running
        UI; this route exists so the next full page load renders the right
        locale server-side (no English flash).
        """
        data = request.get_json(silent=True) or {}
        locale = data.get('locale')
        if locale not in SUPPORTED_UI_LOCALES:
            return jsonify({"success": False, "error": f"Unsupported locale: {locale}"}), 400

        response = make_response(jsonify({"success": True, "locale": locale}))
        response.set_cookie(
            UI_LOCALE_COOKIE,
            locale,
            max_age=UI_LOCALE_COOKIE_MAX_AGE,
            samesite='Lax',
            httponly=True,
        )
        return response

    @bp.route('/api/ui-locale', methods=['GET'])
    def get_ui_locale():
        """Return the locale the server would pick for this request."""
        return jsonify({
            "locale": resolve_ui_locale(request),
            "supported": SUPPORTED_UI_LOCALES,
        })

    @bp.route('/api/health', methods=['GET'])
    def health_check():
        """API health check endpoint"""
        return jsonify({
            "status": "ok",
            "message": "Translation API is running",
            "translate_module": "loaded",
            "ollama_default_endpoint": _config.API_ENDPOINT,
            "supported_formats": ["txt", "epub", "srt"],
            "version": __version__,
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
        elif provider == 'poe':
            return _get_poe_models(api_key)
        elif provider == 'nim':
            return _get_nim_models(api_key)
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
        # For API keys, send a masked indicator if configured, empty string if not.
        # Also expose the pool size so the UI can signal active multi-key rotation.
        def mask_api_key(raw):
            """Return (masked_last_key, key_count). Empty/0 means not configured."""
            keys = normalize_api_keys(raw)
            if not keys:
                return "", 0
            last = keys[-1]
            masked = "***" + last[-4:] if len(last) > 4 else "***"
            return masked, len(keys)

        gemini_mask, gemini_count = mask_api_key(_config.GEMINI_API_KEY)
        openai_mask, openai_count = mask_api_key(_config.OPENAI_API_KEY)
        openrouter_mask, openrouter_count = mask_api_key(_config.OPENROUTER_API_KEY)
        mistral_mask, mistral_count = mask_api_key(_config.MISTRAL_API_KEY)
        deepseek_mask, deepseek_count = mask_api_key(_config.DEEPSEEK_API_KEY)
        poe_mask, poe_count = mask_api_key(_config.POE_API_KEY)
        nim_mask, nim_count = mask_api_key(_config.NIM_API_KEY)

        config_response = {
            "api_endpoint": _config.API_ENDPOINT,
            "ollama_api_endpoint": _config.OLLAMA_API_ENDPOINT,
            "openai_api_endpoint": _config.OPENAI_API_ENDPOINT,
            "default_model": _config.DEFAULT_MODEL,
            "default_source_language": _config.DEFAULT_SOURCE_LANGUAGE,
            "default_target_language": _config.DEFAULT_TARGET_LANGUAGE,
            "timeout": _config.REQUEST_TIMEOUT,
            "context_window": _config.OLLAMA_NUM_CTX,
            "max_attempts": _config.MAX_TRANSLATION_ATTEMPTS,
            "retry_delay": 2,
            "supported_formats": ["txt", "epub", "srt"],
            "gemini_api_key": gemini_mask,
            "openai_api_key": openai_mask,
            "openrouter_api_key": openrouter_mask,
            "mistral_api_key": mistral_mask,
            "deepseek_api_key": deepseek_mask,
            "poe_api_key": poe_mask,
            "nim_api_key": nim_mask,
            "gemini_api_key_count": gemini_count,
            "openai_api_key_count": openai_count,
            "openrouter_api_key_count": openrouter_count,
            "mistral_api_key_count": mistral_count,
            "deepseek_api_key_count": deepseek_count,
            "poe_api_key_count": poe_count,
            "nim_api_key_count": nim_count,
            "gemini_api_key_configured": gemini_count > 0,
            "openai_api_key_configured": openai_count > 0,
            "openrouter_api_key_configured": openrouter_count > 0,
            "mistral_api_key_configured": mistral_count > 0,
            "deepseek_api_key_configured": deepseek_count > 0,
            "poe_api_key_configured": poe_count > 0,
            "nim_api_key_configured": nim_count > 0,
            "output_filename_pattern": _config.OUTPUT_FILENAME_PATTERN,
            "max_tokens_per_chunk": int(_config.MAX_TOKENS_PER_CHUNK),
            "parallel_translations": int(_config.PARALLEL_TRANSLATIONS),
            "max_parallel_translations": int(_config.MAX_PARALLEL_TRANSLATIONS),
            "disable_auto_pause": str(_config.DISABLE_AUTO_PAUSE).strip().lower() == 'true',
            # Webhook notifications — returned as-is for editing. URLs and tokens
            # only ever travel between this server and the same-origin browser
            # session that already controls the .env on disk.
            "notify_webhook_url": _config.NOTIFY_WEBHOOK_URL,
            "notify_webhook_method": _config.NOTIFY_WEBHOOK_METHOD,
            "notify_webhook_headers": _config.NOTIFY_WEBHOOK_HEADERS,
            "notify_webhook_payload": _config.NOTIFY_WEBHOOK_PAYLOAD,
            "notify_on_success": bool(_config.NOTIFY_ON_SUCCESS),
            "notify_on_failure": bool(_config.NOTIFY_ON_FAILURE),
            "notify_on_interruption": bool(_config.NOTIFY_ON_INTERRUPTION),
            "notify_timeout_seconds": int(_config.NOTIFY_TIMEOUT_SECONDS),
            "notify_configured": bool(_config.NOTIFY_WEBHOOK_URL)
        }

        return jsonify(config_response)

    @bp.route('/api/config/max-tokens', methods=['GET'])
    def get_max_tokens():
        """Get MAX_TOKENS_PER_CHUNK configuration value for UI preview height adjustment"""
        return jsonify({
            "max_tokens_per_chunk": _config.MAX_TOKENS_PER_CHUNK
        })

    def _resolve_api_key(provided_key, env_var_name, config_default):
        """Resolve API key from provided value, .env marker, or config default.

        Returns the raw (possibly comma-separated) value — provider constructors
        normalize via base.py. Callers that bypass providers (e.g. listing
        endpoints using requests.get directly) must call _first_key() themselves.
        """
        if provided_key and provided_key != '__USE_ENV__':
            return provided_key
        return os.getenv(env_var_name, config_default)

    def _first_key(raw):
        """Pick the first usable key from a (possibly comma-separated) value.

        Use for HTTP endpoints called directly (model listings) where only a
        single valid key is needed for the read — rotation only matters for the
        translation path.
        """
        keys = normalize_api_keys(raw)
        return keys[0] if keys else None

    def _fetch_provider_models(
        *,
        provided_api_key,
        env_var,
        config_api_key,
        config_default_model,
        provider_class,
        fallback_model,
        status_prefix,
        display_name,
        api_key_missing_message,
        get_models_kwargs=None,
        model_name_field='id',
        include_model_names_on_error=True,
    ):
        """Shared listing for cloud providers exposing `get_available_models()`.

        Factors out the 5 nearly-identical model-listing flows (openrouter,
        mistral, deepseek, poe, gemini). Each wrapper supplies its provider
        class, config values, and a few small quirks (Gemini reads model 'name'
        instead of 'id' and historically omits model_names from error bodies;
        OpenRouter passes text_only=True).

        Behavior is identical to the previous per-provider functions — kwargs
        let each caller preserve its exact response shape and messages.
        """
        api_key = _resolve_api_key(provided_api_key, env_var, config_api_key)
        default_model = config_default_model if config_default_model else fallback_model

        def _error_body(status, message):
            body = {
                "models": [],
                "default": default_model,
                "status": status,
                "count": 0,
                "error": message,
            }
            if include_model_names_on_error:
                body["model_names"] = []
            return body

        if not api_key:
            return jsonify(_error_body("api_key_missing", api_key_missing_message))

        try:
            provider = provider_class(api_key=api_key)
            models = asyncio.run(provider.get_available_models(**(get_models_kwargs or {})))

            if not models:
                return jsonify(_error_body(
                    f"{status_prefix}_error",
                    f"Failed to retrieve {display_name} models"
                ))

            model_names = [m[model_name_field] for m in models]
            resolved_default = default_model
            if resolved_default not in model_names and model_names:
                resolved_default = model_names[0]

            return jsonify({
                "models": models,
                "model_names": model_names,
                "default": resolved_default,
                "status": f"{status_prefix}_connected",
                "count": len(models)
            })

        except Exception as e:
            return jsonify(_error_body(
                f"{status_prefix}_error",
                f"Error connecting to {display_name} API: {str(e)}"
            ))

    def _get_openrouter_models(provided_api_key=None):
        """Get available text-only models from OpenRouter API"""
        from src.core.llm import OpenRouterProvider
        return _fetch_provider_models(
            provided_api_key=provided_api_key,
            env_var='OPENROUTER_API_KEY',
            config_api_key=_config.OPENROUTER_API_KEY,
            config_default_model=_config.OPENROUTER_MODEL,
            provider_class=OpenRouterProvider,
            fallback_model="anthropic/claude-sonnet-4",
            status_prefix="openrouter",
            display_name="OpenRouter",
            api_key_missing_message=(
                "OpenRouter API key is required. Set OPENROUTER_API_KEY "
                "environment variable or pass api_key parameter."
            ),
            get_models_kwargs={"text_only": True},
        )

    def _get_mistral_models(provided_api_key=None):
        """Get available models from Mistral API"""
        from src.core.llm import MistralProvider
        return _fetch_provider_models(
            provided_api_key=provided_api_key,
            env_var='MISTRAL_API_KEY',
            config_api_key=_config.MISTRAL_API_KEY,
            config_default_model=_config.MISTRAL_MODEL,
            provider_class=MistralProvider,
            fallback_model="mistral-large-latest",
            status_prefix="mistral",
            display_name="Mistral",
            api_key_missing_message=(
                "Mistral API key is required. Set MISTRAL_API_KEY "
                "environment variable or pass api_key parameter."
            ),
        )

    def _get_deepseek_models(provided_api_key=None):
        """Get available models from DeepSeek API"""
        from src.core.llm import DeepSeekProvider
        return _fetch_provider_models(
            provided_api_key=provided_api_key,
            env_var='DEEPSEEK_API_KEY',
            config_api_key=_config.DEEPSEEK_API_KEY,
            config_default_model=_config.DEEPSEEK_MODEL,
            provider_class=DeepSeekProvider,
            fallback_model="deepseek-v4-pro",
            status_prefix="deepseek",
            display_name="DeepSeek",
            api_key_missing_message=(
                "DeepSeek API key is required. Set DEEPSEEK_API_KEY "
                "environment variable or pass api_key parameter."
            ),
        )

    def _get_poe_models(provided_api_key=None):
        """Get available models from Poe API"""
        from src.core.llm.providers.poe import PoeProvider
        return _fetch_provider_models(
            provided_api_key=provided_api_key,
            env_var='POE_API_KEY',
            config_api_key=_config.POE_API_KEY,
            config_default_model=_config.POE_MODEL,
            provider_class=PoeProvider,
            fallback_model="Claude-Sonnet-4",
            status_prefix="poe",
            display_name="Poe",
            api_key_missing_message="Poe API key is required. Get your key at https://poe.com/api_key",
        )

    def _get_nim_models(provided_api_key=None):
        """Get available models from NVIDIA NIM API"""
        api_key = _first_key(_resolve_api_key(provided_api_key, 'NIM_API_KEY', _config.NIM_API_KEY))

        # Use NIM_MODEL from .env, fallback to meta/llama-3.1-8b-instruct
        default_model = _config.NIM_MODEL if _config.NIM_MODEL else "meta/llama-3.1-8b-instruct"

        if not api_key:
            return jsonify({
                "models": [],
                "model_names": [],
                "default": default_model,
                "status": "api_key_missing",
                "count": 0,
                "error": "NVIDIA NIM API key is required. Get your key at https://build.nvidia.com/"
            })

        try:
            # Determine base URL from endpoint
            base_url = _config.NIM_API_ENDPOINT.replace('/chat/completions', '').rstrip('/')
            models_url = f"{base_url}/models"
            headers = {'Authorization': f'Bearer {api_key}'}

            response = requests.get(models_url, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                models_data = data.get('data', [])

                if models_data:
                    # Keywords indicating non-chat models
                    non_chat_keywords = [
                        # Embeddings & retrieval
                        'embed', 'rerank', 'bge', 'arctic-embed',
                        # Vision & multimodal
                        'vision', 'vlm', '-vl-', '-vl', 'clip', 'neva', 'vila', 'fuyu',
                        'deplot', 'paligemma', 'kosmos', 'multimodal',
                        'cosmos', 'streampetr',
                        # Code-specific
                        'starcoder', 'codellama', 'codegemma', 'usdcode',
                        'coder', 'codestral', 'code-instruct',
                        # Safety & moderation
                        'guard', 'safety', 'shield',
                        # Audio/speech
                        'whisper', 'parakeet', 'canary', 'fastpitch',
                        # Other non-chat
                        'gliner', 'parse', 'reward', 'mathstral',
                    ]
                    # Known base models (not instruct/chat)
                    base_models = {
                        'google/gemma-2b', 'google/gemma-7b', 'google/recurrentgemma-2b',
                        'nvidia/mistral-nemo-minitron-8b-base', 'mistralai/mixtral-8x22b-v0.1',
                    }

                    models = []
                    for m in models_data:
                        model_id = m.get('id', '')
                        model_lower = model_id.lower()
                        if any(kw in model_lower for kw in non_chat_keywords):
                            continue
                        if model_id in base_models:
                            continue
                        models.append({
                            'id': model_id,
                            'name': model_id,
                            'owned_by': m.get('owned_by', 'nvidia')
                        })

                    # Sort models by name
                    models.sort(key=lambda x: x['name'].lower())

                    if models:
                        model_ids = [m['id'] for m in models]
                        if default_model not in model_ids and model_ids:
                            default_model = model_ids[0]
                        return jsonify({
                            "models": models,
                            "model_names": model_ids,
                            "default": default_model,
                            "status": "nim_connected",
                            "count": len(models)
                        })

            # If API call failed, return empty with error
            return jsonify({
                "models": [],
                "model_names": [],
                "default": default_model,
                "status": "nim_error",
                "count": 0,
                "error": f"Failed to retrieve NVIDIA NIM models (HTTP {response.status_code})"
            })

        except requests.exceptions.ConnectionError:
            return jsonify({
                "models": [],
                "model_names": [],
                "default": default_model,
                "status": "nim_error",
                "count": 0,
                "error": "Could not connect to NVIDIA NIM API. Check your internet connection."
            })
        except Exception as e:
            return jsonify({
                "models": [],
                "model_names": [],
                "default": default_model,
                "status": "nim_error",
                "count": 0,
                "error": f"Error connecting to NVIDIA NIM API: {str(e)}"
            })

    def _get_openai_models(provided_api_key=None, api_endpoint=None):
        """Get available models from OpenAI-compatible API.

        - For api.openai.com: fall back to a static OpenAI cloud list if the live
          fetch fails (keyless/offline scenarios).
        - For any other endpoint (llama.cpp, LM Studio, vLLM, etc.): never fall
          back to the OpenAI cloud list — selecting a non-existent model name
          would cause HTTP 400 at translation time. Return an explicit error so
          the UI can surface it.
        """
        api_key = _first_key(_resolve_api_key(provided_api_key, 'OPENAI_API_KEY', _config.OPENAI_API_KEY))

        if api_endpoint:
            base_url = api_endpoint.replace('/chat/completions', '').rstrip('/')
        else:
            base_url = 'https://api.openai.com/v1'

        parsed_base = urlparse(base_url)
        is_official_openai = parsed_base.hostname == 'api.openai.com'

        openai_static_models = [
            {'id': 'gpt-4o', 'name': 'GPT-4o (Latest)'},
            {'id': 'gpt-4o-mini', 'name': 'GPT-4o Mini'},
            {'id': 'gpt-4-turbo', 'name': 'GPT-4 Turbo'},
            {'id': 'gpt-4', 'name': 'GPT-4'},
            {'id': 'gpt-3.5-turbo', 'name': 'GPT-3.5 Turbo'}
        ]

        fetch_error = None
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
                    models = []
                    for m in models_data:
                        model_id = m.get('id', '')
                        if 'embedding' in model_id.lower() or 'whisper' in model_id.lower():
                            continue
                        models.append({
                            'id': model_id,
                            'name': model_id,
                            'owned_by': m.get('owned_by', 'unknown')
                        })

                    models.sort(key=lambda x: x['name'].lower())

                    if models:
                        model_ids = [m['id'] for m in models]
                        if _config.DEFAULT_MODEL and _config.DEFAULT_MODEL in model_ids:
                            default_model = _config.DEFAULT_MODEL
                        else:
                            default_model = model_ids[0] if model_ids else 'gpt-4o'

                        return jsonify({
                            "models": models,
                            "model_names": model_ids,
                            "default": default_model,
                            "status": "openai_connected",
                            "count": len(models)
                        })

                fetch_error = "Endpoint returned no models (HTTP 200, empty data)"
            else:
                fetch_error = f"HTTP {response.status_code} from {models_url}"
                if response.text:
                    fetch_error += f": {response.text[:200]}"

        except requests.exceptions.SSLError as e:
            fetch_error = f"SSL error ({e}). If this is a local server, use http:// instead of https://"
            logger.warning(f"OpenAI-compatible models fetch SSL error at {base_url}: {e}")
        except requests.exceptions.ConnectionError as e:
            fetch_error = f"Could not connect to {base_url} ({e})"
            logger.warning(f"OpenAI-compatible models fetch connection error at {base_url}: {e}")
        except Exception as e:
            fetch_error = f"{type(e).__name__}: {e}"
            logger.warning(f"OpenAI-compatible models fetch failed at {base_url}: {e}")

        # Custom endpoints get an error rather than the cloud list, because
        # picking a gpt-4o id and sending it to llama.cpp would 400 at trad time.
        if not is_official_openai:
            return jsonify({
                "models": [],
                "model_names": [],
                "default": None,
                "status": "openai_error",
                "count": 0,
                "endpoint": base_url,
                "error": fetch_error or f"Could not list models at {base_url}/models"
            })

        model_ids = [m['id'] for m in openai_static_models]
        if _config.DEFAULT_MODEL and _config.DEFAULT_MODEL in model_ids:
            fallback_default = _config.DEFAULT_MODEL
        else:
            fallback_default = "gpt-4o"
        return jsonify({
            "models": openai_static_models,
            "model_names": model_ids,
            "default": fallback_default,
            "status": "openai_static",
            "count": len(openai_static_models),
            "error": fetch_error
        })

    def _get_gemini_models(provided_api_key=None):
        """Get available models from Gemini API"""
        from src.core.llm import GeminiProvider
        # Gemini's model dicts use 'name', not 'id'; error bodies historically
        # omit model_names (preserved for response-shape compatibility).
        return _fetch_provider_models(
            provided_api_key=provided_api_key,
            env_var='GEMINI_API_KEY',
            config_api_key=_config.GEMINI_API_KEY,
            config_default_model=_config.GEMINI_MODEL,
            provider_class=GeminiProvider,
            fallback_model="gemini-2.0-flash",
            status_prefix="gemini",
            display_name="Gemini",
            api_key_missing_message=(
                "Gemini API key is required. Set GEMINI_API_KEY "
                "environment variable or pass api_key parameter."
            ),
            model_name_field='name',
            include_model_names_on_error=False,
        )

    def _get_ollama_models():
        """Get available models from Ollama API"""
        ollama_base_from_ui = request.args.get('api_endpoint', _config.API_ENDPOINT)

        try:
            parsed = urlparse(ollama_base_from_ui)
            path = parsed.path or '/'
            if '/api/' in path:
                base_path = path.split('/api/')[0]
                base_url = f"{parsed.scheme}://{parsed.netloc}{base_path}"
            else:
                base_url = f"{parsed.scheme}://{parsed.netloc}"
            tags_url = f"{base_url}/api/tags"

            response = requests.get(tags_url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                models_data = data.get('models', [])
                model_names = [m.get('name') for m in models_data if m.get('name')]

                return jsonify({
                    "models": model_names,
                    "default": _config.DEFAULT_MODEL if _config.DEFAULT_MODEL in model_names else (model_names[0] if model_names else _config.DEFAULT_MODEL),
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
            "default": _config.DEFAULT_MODEL,
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

    @bp.route('/api/custom-instructions', methods=['GET'])
    def get_custom_instructions():
        """List available custom instruction files from Custom_Instructions/ folder.

        Each entry carries `has_translation` / `has_refinement` so the UI can
        filter presets per phase. `.txt` files (legacy) apply to both phases;
        `.yaml`/`.yml` files report the phases actually present in the file.
        """
        from src.utils.custom_instructions import list_custom_instructions

        try:
            project_root = Path(get_config_path())
            custom_instructions_dir = project_root / 'Custom_Instructions'

            if not custom_instructions_dir.exists():
                return jsonify({"files": [], "count": 0, "status": "folder_not_found"})

            files = list_custom_instructions(custom_instructions_dir)
            return jsonify({"files": files, "count": len(files), "status": "ok"})

        except Exception as e:
            logger.error(f"Error listing custom instructions: {e}")
            return jsonify({"files": [], "count": 0, "status": "error", "error": str(e)})

    @bp.route('/api/custom-instructions/open-folder', methods=['POST'])
    def open_custom_instructions_folder():
        """Open the Custom_Instructions folder in the system file explorer"""
        import subprocess
        import platform

        try:
            project_root = Path(get_config_path())
            custom_instructions_dir = project_root / 'Custom_Instructions'

            # Create folder if it doesn't exist
            if not custom_instructions_dir.exists():
                custom_instructions_dir.mkdir(parents=True, exist_ok=True)

            abs_path = str(custom_instructions_dir.resolve())
            system = platform.system()

            if system == 'Windows':
                os.startfile(abs_path)
            elif system == 'Darwin':  # macOS
                subprocess.run(['open', abs_path], check=True)
            else:  # Linux and others
                subprocess.run(['xdg-open', abs_path], check=True)

            return jsonify({"success": True, "path": abs_path})

        except Exception as e:
            logger.error(f"Error opening custom instructions folder: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    def _get_env_file_path():
        """Get the path to the .env file"""
        config_path = get_config_path()
        return Path(config_path) / '.env'

    # Keys whose values may contain spaces, '#', or JSON braces. python-dotenv
    # parses unquoted values up to a '#' (treated as inline comment), so a raw
    # JSON payload like {"text":"hi #1"} would be silently truncated. Wrap in
    # single quotes (JSON never produces single quotes, so no escape needed).
    _QUOTED_ENV_KEYS = {
        'NOTIFY_WEBHOOK_URL',
        'NOTIFY_WEBHOOK_HEADERS',
        'NOTIFY_WEBHOOK_PAYLOAD',
        'OUTPUT_FILENAME_PATTERN',
    }

    def _format_env_value(key: str, value: str) -> str:
        if not value:
            return ''
        if key not in _QUOTED_ENV_KEYS:
            return value
        if "'" not in value:
            return f"'{value}'"
        # Fallback if the user did inject single quotes — use double quotes
        # with the minimal escaping python-dotenv understands.
        escaped = value.replace('\\', '\\\\').replace('"', '\\"')
        return f'"{escaped}"'

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
                    formatted = _format_env_value(key, updates[key])
                    new_lines.append(f"{key}={formatted}\n")
                    updated_keys.add(key)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        # Add any keys that weren't in the file
        for key, value in updates.items():
            if key not in updated_keys:
                formatted = _format_env_value(key, value)
                new_lines.append(f"{key}={formatted}\n")

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
            'POE_API_KEY',
            'POE_MODEL',
            'NIM_API_KEY',
            'NIM_MODEL',
            'DEFAULT_MODEL',
            'LLM_PROVIDER',
            'OLLAMA_API_ENDPOINT',
            'OPENAI_API_ENDPOINT',
            'OUTPUT_FILENAME_PATTERN',
            'MAX_TOKENS_PER_CHUNK',
            'PARALLEL_TRANSLATIONS',
            'DISABLE_AUTO_PAUSE',
            'NOTIFY_WEBHOOK_URL',
            'NOTIFY_WEBHOOK_METHOD',
            'NOTIFY_WEBHOOK_HEADERS',
            'NOTIFY_WEBHOOK_PAYLOAD',
            'NOTIFY_ON_SUCCESS',
            'NOTIFY_ON_FAILURE',
            'NOTIFY_ON_INTERRUPTION',
            'NOTIFY_TIMEOUT_SECONDS'
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
                    # Clamp MAX_TOKENS_PER_CHUNK to the same range the UI enforces
                    # so a hand-crafted POST can't break the chunker.
                    if key == 'MAX_TOKENS_PER_CHUNK':
                        try:
                            n = int(safe_value)
                        except (TypeError, ValueError):
                            continue
                        safe_value = str(max(50, min(1000, n)))
                    # Clamp parallel workers to [1, MAX_PARALLEL_TRANSLATIONS].
                    elif key == 'PARALLEL_TRANSLATIONS':
                        try:
                            n = int(safe_value)
                        except (TypeError, ValueError):
                            continue
                        safe_value = str(max(1, min(_config.MAX_PARALLEL_TRANSLATIONS, n)))
                    updates[key] = safe_value

            if not updates:
                return jsonify({"error": "No valid settings to save"}), 400

            # Update the .env file
            _update_env_file(updates)

            # Refresh module-level config so subsequent reads see the new values
            # without requiring a server restart.
            reload_config()

            logger.info(f"Settings saved and reloaded: {list(updates.keys())}")

            return jsonify({
                "success": True,
                "message": f"Saved {len(updates)} setting(s)",
                "saved_keys": list(updates.keys())
            })

        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            return jsonify({"error": f"Failed to save settings: {str(e)}"}), 500

    @bp.route('/api/notifications/test', methods=['POST'])
    def test_notification():
        """Send a test webhook using the current saved configuration.

        The notifier reads from src.config (post-reload), so the test always
        reflects what's actually on disk. Returns success/failure with a hint
        when the URL is empty or the event flag is off.
        """
        from src.utils import notifier

        if not _config.NOTIFY_WEBHOOK_URL:
            return jsonify({
                "success": False,
                "error": "NOTIFY_WEBHOOK_URL is empty. Set a URL and save before testing."
            }), 400

        data = request.get_json(silent=True) or {}
        event = data.get('event', notifier.EVENT_SUCCESS)
        if event not in notifier.known_events():
            return jsonify({
                "success": False,
                "error": f"Unknown event '{event}'. Use one of: {', '.join(notifier.known_events())}"
            }), 400

        flag_map = {
            notifier.EVENT_SUCCESS: 'NOTIFY_ON_SUCCESS',
            notifier.EVENT_FAILURE: 'NOTIFY_ON_FAILURE',
            notifier.EVENT_INTERRUPTION: 'NOTIFY_ON_INTERRUPTION',
        }
        if not bool(getattr(_config, flag_map[event], False)):
            return jsonify({
                "success": False,
                "error": f"Event '{event}' is disabled. Enable it and save before testing."
            }), 400

        ctx = {
            "file": "test-file.epub",
            "output": "test-file (French).epub",
            "duration_seconds": 12.3,
            "provider": _config.LLM_PROVIDER,
            "model": _config.DEFAULT_MODEL or "test-model",
            "source_lang": "English",
            "target_lang": "French",
            "error": "Sample error for failure event" if event == notifier.EVENT_FAILURE else None,
            "translation_id": "test-job-id",
        }

        try:
            sent = notifier.notify(event, ctx)
        except Exception as exc:
            logger.exception("Test webhook raised unexpectedly")
            return jsonify({
                "success": False,
                "error": f"Unexpected error: {exc}"
            }), 500

        if sent:
            return jsonify({
                "success": True,
                "message": f"Test {event} notification sent successfully."
            })
        return jsonify({
            "success": False,
            "error": "Webhook call failed. Check the server logs (enable DEBUG_MODE for details), the URL, headers and payload format."
        }), 502

    @bp.route('/api/settings', methods=['GET'])
    def get_settings():
        """
        Get current settings that can be modified via the UI.
        Returns only the keys that are user-configurable.
        API keys are masked for security - only indicates if configured.
        """
        return jsonify({
            "gemini_api_key_configured": bool(_config.GEMINI_API_KEY),
            "openai_api_key_configured": bool(_config.OPENAI_API_KEY),
            "openrouter_api_key_configured": bool(_config.OPENROUTER_API_KEY),
            "mistral_api_key_configured": bool(_config.MISTRAL_API_KEY),
            "deepseek_api_key_configured": bool(_config.DEEPSEEK_API_KEY),
            "poe_api_key_configured": bool(_config.POE_API_KEY),
            "nim_api_key_configured": bool(_config.NIM_API_KEY),
            "default_model": _config.DEFAULT_MODEL or "",
            "llm_provider": _config.LLM_PROVIDER,
            "api_endpoint": _config.API_ENDPOINT or "",
            "ollama_api_endpoint": _config.OLLAMA_API_ENDPOINT or "",
            "openai_api_endpoint": _config.OPENAI_API_ENDPOINT or ""
        })

    return bp
