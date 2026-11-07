"""
Utility to help users configure .env file
"""
import os
import sys
import shutil
from pathlib import Path


def _get_config_dir():
    """Get directory for configuration files"""
    return Path.cwd()


def create_env_from_template(force: bool = False) -> bool:
    """
    Create .env file from .env.example template

    Args:
        force: If True, overwrites existing .env file

    Returns:
        bool: True if file was created, False otherwise
    """
    config_dir = _get_config_dir()
    env_file = config_dir / '.env'
    env_example = config_dir / '.env.example'

    if env_file.exists() and not force:
        print(f"❌ .env file already exists at: {env_file.absolute()}")
        print("   Use force=True to overwrite")
        return False

    if not env_example.exists():
        print(f"❌ Template file .env.example not found")
        return False

    try:
        shutil.copy(env_example, env_file)
        print(f"✅ Created .env from template")
        print(f"   Location: {env_file.absolute()}")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env: {e}")
        return False


def validate_env_config(verbose: bool = True) -> dict:
    """
    Validate current environment configuration and return status

    Args:
        verbose: If True, prints detailed information

    Returns:
        dict: Status information about configuration
    """
    from dotenv import load_dotenv
    load_dotenv()

    config_dir = _get_config_dir()
    status = {
        'env_exists': (config_dir / '.env').exists(),
        'issues': [],
        'warnings': [],
        'config': {}
    }

    # Check critical configuration
    api_endpoint = os.getenv('API_ENDPOINT', 'http://localhost:11434/api/generate')
    llm_provider = os.getenv('LLM_PROVIDER', 'ollama')
    default_model = os.getenv('DEFAULT_MODEL', 'qwen3:14b')
    gemini_key = os.getenv('GEMINI_API_KEY', '')
    openai_key = os.getenv('OPENAI_API_KEY', '')

    status['config'] = {
        'api_endpoint': api_endpoint,
        'llm_provider': llm_provider,
        'model': default_model,
        'port': os.getenv('PORT', '5000'),
    }

    # Validate provider-specific requirements
    if llm_provider == 'gemini' and not gemini_key:
        status['issues'].append("Gemini provider selected but GEMINI_API_KEY is not set")

    if llm_provider == 'openai' and not openai_key:
        status['issues'].append("OpenAI provider selected but OPENAI_API_KEY is not set")

    if llm_provider == 'ollama' and 'localhost' not in api_endpoint and '127.0.0.1' not in api_endpoint:
        status['warnings'].append(f"Using remote Ollama server: {api_endpoint}")

    # Check if using defaults (likely means no .env)
    if api_endpoint == 'http://localhost:11434/api/generate' and not status['env_exists']:
        status['warnings'].append("Using default localhost configuration - may not be correct")

    if verbose:
        print("\n" + "="*70)
        print("🔍 CONFIGURATION VALIDATION")
        print("="*70)
        print(f"\n📁 .env file exists: {'✅ Yes' if status['env_exists'] else '❌ No (using defaults)'}")
        print(f"\n⚙️  Current Configuration:")
        print(f"   • LLM Provider: {llm_provider}")
        print(f"   • API Endpoint: {api_endpoint}")
        print(f"   • Model: {default_model}")
        print(f"   • Port: {status['config']['port']}")

        if status['issues']:
            print(f"\n❌ CRITICAL ISSUES:")
            for issue in status['issues']:
                print(f"   • {issue}")

        if status['warnings']:
            print(f"\n⚠️  WARNINGS:")
            for warning in status['warnings']:
                print(f"   • {warning}")

        if not status['issues'] and not status['warnings']:
            print(f"\n✅ Configuration looks good!")

        print("="*70 + "\n")

    return status


def interactive_env_setup():
    """
    Interactive setup wizard for .env configuration
    """
    print("\n" + "="*70)
    print("🛠️  INTERACTIVE .ENV SETUP WIZARD")
    print("="*70)

    config_dir = _get_config_dir()
    env_file = config_dir / '.env'

    if env_file.exists():
        response = input("\n.env file already exists. Overwrite? (yes/no): ").strip().lower()
        if response != 'yes':
            print("❌ Setup cancelled")
            return

    print("\n📋 Please provide the following information:")
    print("   (Press Enter to use default values shown in brackets)\n")

    # Collect configuration
    config = {}

    print("1️⃣  LLM Provider")
    config['LLM_PROVIDER'] = input("   Provider (ollama/gemini/openai) [ollama]: ").strip() or 'ollama'

    if config['LLM_PROVIDER'] == 'ollama':
        config['API_ENDPOINT'] = input("   Ollama API endpoint [http://localhost:11434/api/generate]: ").strip() or 'http://localhost:11434/api/generate'
        config['DEFAULT_MODEL'] = input("   Model name [qwen3:14b]: ").strip() or 'qwen3:14b'

    elif config['LLM_PROVIDER'] == 'gemini':
        config['GEMINI_API_KEY'] = input("   Gemini API Key: ").strip()
        config['GEMINI_MODEL'] = input("   Gemini Model [gemini-3.6-flash]: ").strip() or 'gemini-3.6-flash'

    elif config['LLM_PROVIDER'] == 'openai':
        config['OPENAI_API_KEY'] = input("   OpenAI API Key: ").strip()
        config['API_ENDPOINT'] = input("   API endpoint [https://api.openai.com/v1/chat/completions]: ").strip() or 'https://api.openai.com/v1/chat/completions'
        config['DEFAULT_MODEL'] = input("   Model [gpt-4o]: ").strip() or 'gpt-4o'

    config['PORT'] = input("\n2️⃣  Web server port [5000]: ").strip() or '5000'
    config['DEFAULT_SOURCE_LANGUAGE'] = input("3️⃣  Default source language [English]: ").strip() or 'English'
    config['DEFAULT_TARGET_LANGUAGE'] = input("4️⃣  Default target language [Chinese]: ").strip() or 'Chinese'

    # Write .env file
    try:
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write("# Translation API Configuration\n")

            if config['LLM_PROVIDER'] == 'ollama':
                f.write(f"API_ENDPOINT={config['API_ENDPOINT']}\n")
                f.write(f"DEFAULT_MODEL={config['DEFAULT_MODEL']}\n")

            f.write("\n# Server Configuration\n")
            f.write(f"PORT={config['PORT']}\n")
            f.write("HOST=127.0.0.1\n")
            f.write("OUTPUT_DIR=translated_files\n")

            f.write("\n# LLM Provider Settings\n")
            f.write(f"LLM_PROVIDER={config['LLM_PROVIDER']}\n")

            if config['LLM_PROVIDER'] == 'gemini':
                f.write(f"GEMINI_API_KEY={config.get('GEMINI_API_KEY', '')}\n")
                f.write(f"GEMINI_MODEL={config.get('GEMINI_MODEL', 'gemini-3.6-flash')}\n")
            else:
                f.write("GEMINI_API_KEY=\n")
                f.write("GEMINI_MODEL=gemini-3.6-flash\n")

            if config['LLM_PROVIDER'] == 'openai':
                f.write(f"OPENAI_API_KEY={config.get('OPENAI_API_KEY', '')}\n")
            else:
                f.write("OPENAI_API_KEY=\n")

            f.write("\n# Translation Settings\n")
            f.write(f"DEFAULT_SOURCE_LANGUAGE={config['DEFAULT_SOURCE_LANGUAGE']}\n")
            f.write(f"DEFAULT_TARGET_LANGUAGE={config['DEFAULT_TARGET_LANGUAGE']}\n")
            f.write("MAIN_CHUNK_SIZE=1000\n")
            f.write("REQUEST_TIMEOUT=900\n")

            f.write("\n# Context Management\n")
            f.write("OLLAMA_NUM_CTX=8192\n")
            f.write("AUTO_ADJUST_CONTEXT=true\n")
            f.write("MIN_CHUNK_SIZE=5\n")
            f.write("MAX_CHUNK_SIZE=100\n")

            f.write("\n# Advanced\n")
            f.write("MAX_TRANSLATION_ATTEMPTS=3\n")

            f.write("\n# SRT-specific configuration\n")
            f.write("# Subtitles per LLM block, shared by translate and refine.\n")
            f.write("# Lower for tiny models (e.g. 5 for 4B params), higher for big-context models.\n")
            f.write("SRT_LINES_PER_BLOCK=10\n")

        print("\n✅ .env file created successfully!")
        print(f"   Location: {env_file.absolute()}")
        print("\n💡 You can edit this file manually at any time to adjust settings.\n")

    except Exception as e:
        print(f"\n❌ Failed to create .env file: {e}\n")


if __name__ == '__main__':
    """Allow running this script standalone for configuration"""
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == 'create':
            create_env_from_template()
        elif command == 'validate':
            validate_env_config()
        elif command == 'setup':
            interactive_env_setup()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: create, validate, setup")
    else:
        print("\nUsage:")
        print("  python -m src.utils.env_helper create   - Create .env from template")
        print("  python -m src.utils.env_helper validate - Check current configuration")
        print("  python -m src.utils.env_helper setup    - Interactive setup wizard")
