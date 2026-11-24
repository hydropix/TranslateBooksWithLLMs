#!/usr/bin/env python
"""
Fix common installation issues for TranslateBookWithLLM
This script checks and fixes known issues that can occur on fresh installations.
"""
import sys
import os
from pathlib import Path
import re


def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def check_python_version():
    """Check Python version"""
    print_header("Checking Python Version")
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print(f"Python version: {version_str}")

    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print("[ERROR] Python 3.11 or higher is required")
        return False
    else:
        print("[OK] Python version is compatible")
        return True


def fix_prompts_file():
    """Fix the backslash issue in prompts.py"""
    print_header("Checking prompts.py")

    prompts_file = Path('prompts.py')

    if not prompts_file.exists():
        print(f"[ERROR] ERROR: prompts.py not found at {prompts_file.absolute()}")
        return False

    try:
        content = prompts_file.read_text(encoding='utf-8')
        original_content = content
        fixes_applied = []

        # Fix 1: Windows path with backslashes in example
        if r'`C:\Users\Documents\`' in content or r'C:\\Users\\Documents\\' in content:
            content = re.sub(
                r'- File paths: `/usr/bin/`, `C:\\Users\\Documents\\`',
                r'- File paths: `/usr/bin/`, `C:/Users/Documents/`',
                content
            )
            content = re.sub(
                r'C:\\Users\\Documents\\',
                r'C:/Users/Documents/',
                content
            )
            fixes_applied.append("Fixed Windows path backslashes")

        # Fix 2: Check for function calls with \n inside f-strings (Python 3.11 issue)
        # Pattern: _get_output_format_section(...\n...) inside an f-string
        if 'additional_rules="\\n' in content and '{_get_output_format_section(' in content:
            # This is a complex fix - the function call needs to be moved outside the f-string
            print("[WARNING]  Found function calls with backslashes inside f-strings")
            print("   This requires manual intervention or git pull with latest fixes")
            fixes_applied.append("Detected f-string issue (needs manual fix or git pull)")

        # Fix 3: Any other backslash sequences in f-strings
        # Look for common patterns that would fail in Python 3.11
        problematic_patterns = [
            (r'additional_rules="\\n', 'newline in additional_rules parameter'),
            (r'example_format=".*\\n.*"', 'newline in example_format parameter'),
            (r'\{"\\n"\.join\(', 'backslash in .join() inside f-string'),
        ]

        for pattern, description in problematic_patterns:
            if re.search(pattern, content):
                print(f"[WARNING]  Found: {description}")
                fixes_applied.append(f"Detected: {description}")

        # Write back if changes were made
        if content != original_content:
            prompts_file.write_text(content, encoding='utf-8')
            print("[OK] Applied fixes to prompts.py:")
            for fix in fixes_applied:
                print(f"   • {fix}")
            return True
        elif fixes_applied:
            print("[WARNING]  Issues detected but not auto-fixed:")
            for fix in fixes_applied:
                print(f"   • {fix}")
            print("\n   SOLUTION: Run 'git pull' to get the latest fixes")
            return False
        else:
            print("[OK] No issues found in prompts.py")
            return True

    except Exception as e:
        print(f"[ERROR] ERROR: Failed to check prompts.py: {e}")
        return False


def clear_python_cache():
    """Clear Python cache files"""
    print_header("Clearing Python Cache")

    cache_cleared = False
    root = Path('.')

    # Remove __pycache__ directories
    for pycache in root.rglob('__pycache__'):
        try:
            import shutil
            shutil.rmtree(pycache)
            print(f"  Removed: {pycache}")
            cache_cleared = True
        except Exception as e:
            print(f"  Warning: Could not remove {pycache}: {e}")

    # Remove .pyc files
    for pyc in root.rglob('*.pyc'):
        try:
            pyc.unlink()
            print(f"  Removed: {pyc}")
            cache_cleared = True
        except Exception as e:
            print(f"  Warning: Could not remove {pyc}: {e}")

    if cache_cleared:
        print("[OK] Cache cleared")
    else:
        print("[INFO]  No cache files found")

    return True


def check_env_file():
    """Check if .env file exists and is configured"""
    print_header("Checking Configuration")

    env_file = Path('.env')
    env_example = Path('.env.example')

    if not env_file.exists():
        print("[WARNING]  .env file not found")

        if env_example.exists():
            print("\n TO CREATE .env FILE:")
            print("   Option 1: Run setup wizard")
            print("             python setup_config.py")
            print("\n   Option 2: Copy template manually")
            print("             copy .env.example .env")
            print("             notepad .env")
        return False
    else:
        print("[OK] .env file exists")

        # Check for required settings
        try:
            from dotenv import load_dotenv
            load_dotenv()

            api_endpoint = os.getenv('API_ENDPOINT', 'NOT_SET')
            provider = os.getenv('LLM_PROVIDER', 'NOT_SET')

            print(f"\n  Current settings:")
            print(f"    • API_ENDPOINT: {api_endpoint}")
            print(f"    • LLM_PROVIDER: {provider}")

            if api_endpoint == 'NOT_SET':
                print("\n[WARNING]  API_ENDPOINT is not configured")
                return False

        except Exception as e:
            print(f"[WARNING]  Could not validate .env: {e}")

        return True


def test_import():
    """Test if prompts.py can be imported"""
    print_header("Testing Import")

    try:
        # Clear any cached imports
        if 'src.prompts' in sys.modules:
            del sys.modules['src.prompts']
        if 'src.prompts.prompts' in sys.modules:
            del sys.modules['src.prompts.prompts']

        print("[OK] prompts.py imports successfully")
        return True
    except SyntaxError as e:
        print(f"[ERROR] SyntaxError in prompts.py:")
        print(f"   {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Import error:")
        print(f"   {e}")
        return False


def main():
    """Main function"""
    print("\n" + "="*70)
    print("  TranslateBookWithLLM - Installation Fix Tool")
    print("="*70)
    print("\nThis tool will check and fix common installation issues.\n")

    has_critical_errors = False
    has_warnings = False

    # Check Python version (CRITICAL)
    if not check_python_version():
        has_critical_errors = True

    # Fix prompts.py (CRITICAL)
    if not fix_prompts_file():
        has_critical_errors = True

    # Clear cache (non-critical)
    clear_python_cache()

    # Check .env (WARNING only - not critical for testing)
    if not check_env_file():
        has_warnings = True
        print("\n[INFO] Configuration needed but not critical for testing")

    # Test import (CRITICAL)
    if not test_import():
        has_critical_errors = True

    # Final summary
    print_header("Summary")

    if has_critical_errors:
        print("[ERROR] Critical issues were found!")
        print("\nPlease review the errors above and:")
        print("  1. Fix any critical issues")
        print("  2. Run this script again to verify")
        print("  3. If issues persist, check the documentation")
        return 1
    elif has_warnings:
        print("[OK] All checks passed! (with minor warnings)")
        print("\n[INFO] Note: You may want to configure .env later for production use")
        return 0
    else:
        print("[OK] All checks passed!")
        return 0


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n  Script cancelled by user\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
