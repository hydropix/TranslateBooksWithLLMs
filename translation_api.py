"""
Flask web server for translation API with WebSocket support
"""
import os
import sys
import logging
import webbrowser
import threading
from datetime import datetime
from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce verbosity of werkzeug (Flask HTTP server logs)
logging.getLogger('werkzeug').setLevel(logging.WARNING)

# Reduce verbosity of httpx (avoid showing 400 errors during model detection)
logging.getLogger('httpx').setLevel(logging.WARNING)

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')

from src.config import (
    API_ENDPOINT as DEFAULT_OLLAMA_API_ENDPOINT,
    DEFAULT_MODEL,
    PORT,
    HOST,
    OUTPUT_DIR
)
from src.api.routes import configure_routes
from src.api.websocket import configure_websocket_handlers
from src.api.handlers import start_translation_job
from src.api.translation_state import get_state_manager


# Initialize Flask app with static folder configuration
# Handle PyInstaller bundle paths
if getattr(sys, 'frozen', False):
    # Running as compiled executable - files are in _MEIPASS
    bundle_dir = sys._MEIPASS
    static_folder_path = os.path.join(bundle_dir, 'src', 'web', 'static')
    template_folder_path = os.path.join(bundle_dir, 'src', 'web', 'templates')

    # Debug: print paths to verify
    print(f"🔍 PyInstaller bundle detected")
    print(f"   Bundle dir: {bundle_dir}")
    print(f"   Static folder: {static_folder_path}")
    print(f"   Template folder: {template_folder_path}")
    print(f"   Static folder exists: {os.path.exists(static_folder_path)}")
    print(f"   Template folder exists: {os.path.exists(template_folder_path)}")

    if os.path.exists(template_folder_path):
        print(f"   Templates: {os.listdir(template_folder_path)}")
    if os.path.exists(bundle_dir):
        print(f"   Bundle contents: {os.listdir(bundle_dir)}")
else:
    # Running as normal Python script
    base_path = os.getcwd()
    static_folder_path = os.path.join(base_path, 'src', 'web', 'static')
    template_folder_path = os.path.join(base_path, 'src', 'web', 'templates')

app = Flask(__name__,
            static_folder=static_folder_path,
            template_folder=template_folder_path,
            static_url_path='/static')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Thread-safe state manager (generates unique session ID for this server instance)
state_manager = get_state_manager()
logger.info(f"🔑 Server session ID: {state_manager.server_session_id}")

def validate_configuration():
    """Validate required configuration before starting server"""
    issues = []

    if not PORT or not isinstance(PORT, int):
        issues.append("PORT must be a valid integer")
    if not DEFAULT_MODEL:
        issues.append("DEFAULT_MODEL must be configured")
    if not DEFAULT_OLLAMA_API_ENDPOINT:
        issues.append("API_ENDPOINT must be configured")

    if issues:
        logger.error("\n" + "="*70)
        logger.error("❌ CONFIGURATION ERROR")
        logger.error("="*70)
        for issue in issues:
            logger.error(f"   • {issue}")
        logger.error("\n💡 SOLUTION:")
        logger.error("   1. Create a .env file from .env.example")
        logger.error("   2. Configure the required settings")
        logger.error("   3. Restart the application")
        logger.error("\n   Quick setup:")
        logger.error("   python -m src.utils.env_helper setup")
        logger.error("="*70 + "\n")
        raise ValueError("Configuration validation failed. See errors above.")

    logger.info("✅ Configuration validated successfully")

# Ensure output directory exists
try:
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    logger.info(f"Output folder '{OUTPUT_DIR}' is ready")
except OSError as e:
    logger.error(f"Critical error: Unable to create output folder '{OUTPUT_DIR}': {e}")
    sys.exit(1)

# Static files are now handled automatically by Flask

# Wrapper function for starting translation jobs
def start_job_wrapper(translation_id, config):
    """Wrapper to inject dependencies into job starter"""
    start_translation_job(translation_id, config, state_manager, OUTPUT_DIR, socketio)

# Configure routes and WebSocket handlers
configure_routes(app, state_manager, OUTPUT_DIR, start_job_wrapper, socketio)
configure_websocket_handlers(socketio, state_manager)

# Restore incomplete jobs from database on startup
def restore_incomplete_jobs():
    """Restore incomplete translation jobs from checkpoints on server startup"""
    try:
        # First, clean up old jobs (older than 30 days) to prevent database bloat
        jobs_deleted, files_cleaned = state_manager.checkpoint_manager.cleanup_old_jobs(max_age_days=30)
        if jobs_deleted > 0:
            logger.info(f"🧹 Cleaned up {jobs_deleted} old job(s) and {files_cleaned} upload folder(s)")

        # Clean up orphan upload folders (folders without corresponding jobs in DB)
        orphans_deleted = state_manager.checkpoint_manager.cleanup_orphan_uploads()
        if orphans_deleted > 0:
            logger.info(f"🧹 Cleaned up {orphans_deleted} orphan upload folder(s)")

        # Then, reset any jobs that were 'running' when the server was stopped
        # These jobs are now interrupted and should be resumable
        reset_count = state_manager.checkpoint_manager.reset_running_jobs_on_startup()
        if reset_count > 0:
            logger.info(f"🔄 Reset {reset_count} job(s) that were running when the server was stopped")

        resumable_jobs = state_manager.get_resumable_jobs()
        if resumable_jobs:
            logger.info(f"📦 Found {len(resumable_jobs)} incomplete translation job(s) from previous session:")
            for job in resumable_jobs:
                translation_id = job['translation_id']
                progress = job.get('progress', {})
                completed = progress.get('completed_chunks', 0)
                total = progress.get('total_chunks', 0)

                # Restore job into in-memory state
                state_manager.restore_job_from_checkpoint(translation_id)

                logger.info(f"   - {translation_id}: {job['file_type'].upper()} ({completed}/{total} chunks completed)")
            logger.info("   Use the web interface to resume or delete these jobs")
        else:
            logger.info("📦 No incomplete jobs to restore")
    except Exception as e:
        logger.error(f"Error restoring incomplete jobs: {e}")

restore_incomplete_jobs()

def open_browser(host, port):
    """Open the web interface in the default browser after a short delay"""
    def _open():
        # Small delay to ensure server is ready
        import time
        time.sleep(1.5)
        url = f"http://{'localhost' if host == '0.0.0.0' else host}:{port}"
        webbrowser.open(url)

    # Run in background thread to not block server startup
    thread = threading.Thread(target=_open, daemon=True)
    thread.start()


def test_ollama_connection():
    """Test Ollama connection at startup and log result"""
    import requests
    try:
        base_url = DEFAULT_OLLAMA_API_ENDPOINT.split('/api/')[0]
        tags_url = f"{base_url}/api/tags"
        logger.info(f"🔍 Testing Ollama connection at {tags_url}...")
        response = requests.get(tags_url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [m.get('name') for m in data.get('models', [])]
            logger.info(f"✅ Ollama connected! Found {len(models)} model(s)")
            if models:
                logger.info(f"   Available models:")
                for model in sorted(models):
                    logger.info(f"     - {model}")
            return True
        else:
            logger.warning(f"⚠️ Ollama returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        logger.warning(f"⚠️ Cannot connect to Ollama at {base_url}")
        logger.warning(f"   Make sure Ollama is running ('ollama serve')")
        return False
    except Exception as e:
        logger.warning(f"⚠️ Ollama connection test failed: {e}")
        return False


def start_server():
    """Start the translation server - can be called from launcher or directly"""
    try:
        # Validate configuration before starting
        validate_configuration()

        logger.info("")
        logger.info("=" * 50)
        logger.info("  TranslateBook with LLMs - Server")
        logger.info("=" * 50)
        logger.info("")
        logger.info(f"  Ollama Endpoint: {DEFAULT_OLLAMA_API_ENDPOINT}")
        logger.info(f"  Supported formats: .txt, .epub, .srt")
        logger.info("")

        # Test Ollama connection at startup
        test_ollama_connection()

        logger.info("")
        logger.info("=" * 50)
        logger.info(f"  🌐 Web Interface: http://127.0.0.1:{PORT}")
        logger.info("=" * 50)
        logger.info("")
        logger.info("  Press Ctrl+C to stop the server")
        logger.info("")

        # Production deployment note (silent for normal use)
        if HOST == '0.0.0.0' and os.environ.get('SHOW_PRODUCTION_WARNING'):
            logger.warning("⚠️  Server is binding to 0.0.0.0 (all network interfaces)")
            logger.warning("   For production, use a proper WSGI server like gunicorn")
            logger.info("")

        # Auto-open browser (especially useful for portable executable)
        open_browser(HOST, PORT)

        socketio.run(app, debug=False, host=HOST, port=PORT, allow_unsafe_werkzeug=True)
    except Exception as e:
        logger.error("="*60)
        logger.error("❌ FATAL ERROR")
        logger.error("="*60)
        logger.error(f"{e}")
        logger.error("")
        logger.error("Press any key to exit...")
        logger.error("="*60)
        import traceback
        traceback.print_exc()
        input()  # Keep console open
        sys.exit(1)


if __name__ == '__main__':
    start_server()