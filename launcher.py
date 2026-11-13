"""
Launcher wrapper for PyInstaller executable
Handles proper working directory setup and .env file management
"""
import os
import sys
import shutil
from pathlib import Path

def setup_working_directory():
    """Setup proper working directory for the executable"""

    # Determine if running as PyInstaller bundle
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        exe_dir = Path(sys.executable).parent

        # Create a data directory next to the executable
        app_data_dir = exe_dir / 'TranslateBook_Data'
        app_data_dir.mkdir(exist_ok=True)

        # Change working directory to app data folder
        os.chdir(app_data_dir)

        # Create necessary subdirectories
        (app_data_dir / 'translated_files').mkdir(exist_ok=True)
        (app_data_dir / 'checkpoints').mkdir(exist_ok=True)

        bundle_dir = Path(sys._MEIPASS)

        # Use the tiktoken encoding bundled at build time instead of
        # downloading it, so the app works offline or behind a firewall (#339).
        # An explicit TIKTOKEN_CACHE_DIR set by the user still wins.
        bundled_tiktoken_cache = bundle_dir / 'tiktoken_cache'
        if bundled_tiktoken_cache.is_dir() and not os.environ.get('TIKTOKEN_CACHE_DIR'):
            os.environ['TIKTOKEN_CACHE_DIR'] = str(bundled_tiktoken_cache)

        # Copy .env.example if it doesn't exist and is bundled
        env_example_path = app_data_dir / '.env.example'
        if not env_example_path.exists():
            bundled_env_example = bundle_dir / '.env.example'
            if bundled_env_example.exists():
                shutil.copy(bundled_env_example, env_example_path)

        # Seed Custom_Instructions folder with bundled examples on first run.
        # Skip if user already has the folder (preserves their custom files).
        custom_instructions_path = app_data_dir / 'Custom_Instructions'
        if not custom_instructions_path.exists():
            bundled_custom_instructions = bundle_dir / 'Custom_Instructions'
            if bundled_custom_instructions.exists():
                shutil.copytree(bundled_custom_instructions, custom_instructions_path)

        # Create default .env if it doesn't exist
        env_path = app_data_dir / '.env'
        if not env_path.exists():
            print("\n" + "="*70)
            print("FIRST RUN DETECTED")
            print("="*70)
            print("\nCreating default configuration file...")

            # Create a minimal .env with defaults
            default_env = """# TranslateBook with LLM Configuration
# This file was auto-generated on first run

# === LLM PROVIDER ===
LLM_PROVIDER=ollama

# === OLLAMA CONFIGURATION ===
API_ENDPOINT=http://localhost:11434/api/generate
DEFAULT_MODEL=qwen3:14b
OLLAMA_NUM_CTX=4096

# === SERVER CONFIGURATION ===
PORT=5000
HOST=127.0.0.1
OUTPUT_DIR=translated_files

# === OPTIONAL: CLOUD PROVIDERS ===
# Uncomment and add your API keys if using cloud providers
# OPENROUTER_API_KEY=sk-or-v1-...
# OPENAI_API_KEY=sk-...
# GEMINI_API_KEY=...

# === PERFORMANCE ===
REQUEST_TIMEOUT=900
MAX_TOKENS_PER_CHUNK=400
"""
            env_path.write_text(default_env, encoding='utf-8')
            print(f"[OK] Configuration file created at: {env_path}")
            print("\n[INFO] You can edit this file later to customize settings")
            print("="*70)
            print()

        print(f"[INFO] Working directory: {app_data_dir}")
        print(f"[INFO] Translated files will be saved to: {app_data_dir / 'translated_files'}")
        print()

    else:
        # Running as normal Python script
        print("[DEV] Running as Python script (development mode)")

if __name__ == '__main__':
    try:
        # Setup environment
        setup_working_directory()

        # Import and start the server
        from translation_api import start_server
        start_server()

    except KeyboardInterrupt:
        print("\n\n[STOPPED] Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print("\n" + "="*70)
        print("[ERROR] STARTUP ERROR")
        print("="*70)
        print(f"\n{e}\n")
        import traceback
        traceback.print_exc()
        print("\nPress Enter to exit...")
        input()
        sys.exit(1)
