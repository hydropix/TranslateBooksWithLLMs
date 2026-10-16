"""
Flask routes orchestrator for the translation API

This module serves as a lightweight coordinator that registers
all route blueprints. The actual route implementations are organized
into separate modules for better maintainability:

- blueprints/config_routes.py: Health checks, models, and configuration
- blueprints/translation_routes.py: Translation job management
- blueprints/file_routes.py: File listing, download, delete operations
- blueprints/security_routes.py: File upload and security endpoints
- blueprints/tts_routes.py: TTS audio generation from existing files
"""
from flask import jsonify

from .blueprints import (
    create_config_blueprint,
    create_translation_blueprint,
    create_file_blueprint,
    create_security_blueprint,
    create_tts_blueprint,
    create_glossary_blueprint,
    create_custom_instruction_blueprint,
    create_cost_blueprint,
    create_version_blueprint,
    create_sample_blueprint,
    create_ask_blueprint,
)
from .sample_state import SampleStateManager


def configure_routes(app, state_manager, output_dir, start_translation_job, socketio=None):
    """
    Configure Flask routes by registering all blueprints

    Args:
        app: Flask application instance
        state_manager: Translation state manager
        output_dir: Base directory for file operations
        start_translation_job: Function to start translation jobs
        socketio: SocketIO instance for real-time updates (optional)
    """

    # Register config and health check routes
    # Pass server_session_id from state_manager to ensure consistency
    config_bp = create_config_blueprint(server_session_id=state_manager.server_session_id)
    app.register_blueprint(config_bp)

    # Register translation management routes
    translation_bp = create_translation_blueprint(state_manager, start_translation_job, output_dir)
    app.register_blueprint(translation_bp)

    # Register file management routes
    file_bp = create_file_blueprint(output_dir)
    app.register_blueprint(file_bp)

    # Register security and upload routes
    security_bp = create_security_blueprint(output_dir)
    app.register_blueprint(security_bp)

    # Register cost estimation routes
    cost_bp = create_cost_blueprint(output_dir)
    app.register_blueprint(cost_bp)

    # Register glossary management routes
    # Reuse the process-wide GlossaryStore so we don't multiply SQLite
    # connections across the blueprint and the translation handler.
    glossary_bp = create_glossary_blueprint(store=state_manager.get_glossary_store())
    app.register_blueprint(glossary_bp)

    # Register custom-instruction (style preset) CRUD + extraction routes
    custom_instruction_bp = create_custom_instruction_blueprint()
    app.register_blueprint(custom_instruction_bp)

    # Register TTS routes (requires socketio for progress updates)
    if socketio:
        tts_bp = create_tts_blueprint(output_dir, socketio)
        app.register_blueprint(tts_bp)

    # Register version check & self-update routes
    version_bp = create_version_blueprint(state_manager)
    app.register_blueprint(version_bp)

    # Register Sample & Compare routes (ephemeral, in-memory).
    # The sample state manager is owned by this blueprint — sample runs are
    # short-lived and not persisted, so they don't share storage with the
    # main translation state.
    sample_state_manager = SampleStateManager()
    sample_bp = create_sample_blueprint(sample_state_manager, socketio, output_dir)
    app.register_blueprint(sample_bp)

    # Register Ask the Book routes (question answering grounded in an uploaded book)
    ask_bp = create_ask_blueprint(output_dir)
    app.register_blueprint(ask_bp)

    # Register error handlers
    _register_error_handlers(app)


def _register_error_handlers(app):
    """Register global error handlers"""

    @app.errorhandler(404)
    def route_not_found(error):
        return jsonify({"error": "API Endpoint not found"}), 404

    @app.errorhandler(500)
    def internal_server_error(error):
        import traceback
        tb_str = traceback.format_exc()
        print(f"INTERNAL SERVER ERROR: {error}\nTRACEBACK:\n{tb_str}")
        return jsonify({"error": "Internal server error", "details": str(error)}), 500