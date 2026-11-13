"""
Reproduction script for the stale-config bug fixed by PR #141.

Scenario: a user saves a new API key via the web UI. The endpoint writes
to .env on disk, but module-level config variables stay frozen at their
import-time value. The next GET /api/config returns the stale empty key,
and the UI shows "LLM: Error (No API Key)" until the server is restarted.

This script reproduces that lifecycle without booting Flask:
  1. Create a temp working directory with a .env that has DEEPSEEK_API_KEY=""
  2. Import src.config (simulates server startup)
  3. Modify .env on disk (simulates _update_env_file being called)
  4. Trigger reload_config() if available (simulates the fix)
  5. Read the value the way config_routes.py reads it: src.config.DEEPSEEK_API_KEY
  6. Assert it reflects the new value

The script exits 0 on success, 1 on failure. Intended to be run twice:
once before the fix (expected to fail), once after (expected to pass).
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def write_env(env_path: Path, values: dict) -> None:
    env_path.write_text(
        "\n".join(f"{k}={v}" for k, v in values.items()) + "\n",
        encoding="utf-8",
    )


def clear_env_vars(names) -> None:
    """Ensure these env vars are unset so .env values win."""
    for name in names:
        os.environ.pop(name, None)


def simulate_save_endpoint(config_module, env_path: Path, updates: dict) -> None:
    """
    Mimic the relevant part of save_settings() in config_routes.py:
    write the new values to .env, then ask the config module to reload.
    """
    existing = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                existing[k.strip()] = v.strip()
    existing.update(updates)
    write_env(env_path, existing)

    if hasattr(config_module, "reload_config"):
        config_module.reload_config()
    else:
        print("  [info] reload_config() not present — bug path active")


def run_scenario(tmp_dir: Path) -> bool:
    env_path = tmp_dir / ".env"
    tracked_keys = [
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_MODEL",
        "GEMINI_API_KEY",
        "DEFAULT_MODEL",
        "OUTPUT_FILENAME_PATTERN",
    ]
    clear_env_vars(tracked_keys)

    write_env(env_path, {
        "DEEPSEEK_API_KEY": "",
        "DEEPSEEK_MODEL": "deepseek-chat",
        "GEMINI_API_KEY": "",
        "DEFAULT_MODEL": "qwen3:14b",
        "OUTPUT_FILENAME_PATTERN": "{originalName} ({targetLang}).{ext}",
    })

    original_cwd = Path.cwd()
    os.chdir(tmp_dir)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    for mod_name in ("src.config", "src"):
        sys.modules.pop(mod_name, None)

    try:
        import src.config as config

        print("STEP 1 - Initial import:")
        print(f"  DEEPSEEK_API_KEY = {config.DEEPSEEK_API_KEY!r}")
        print(f"  DEFAULT_MODEL    = {config.DEFAULT_MODEL!r}")
        assert config.DEEPSEEK_API_KEY == "", \
            "Initial DEEPSEEK_API_KEY should be empty"
        assert config.DEFAULT_MODEL == "qwen3:14b", \
            "Initial DEFAULT_MODEL should be qwen3:14b"

        print("\nSTEP 2 - User saves new settings (writes .env, then reload):")
        new_values = {
            "DEEPSEEK_API_KEY": "sk-deepseek-new-secret-1234",
            "DEEPSEEK_MODEL": "deepseek-v4-pro",
            "GEMINI_API_KEY": "AIza-gemini-new-9999",
            "DEFAULT_MODEL": "qwen3:30b",
            "OUTPUT_FILENAME_PATTERN": "{originalName}-{model}.{ext}",
        }
        simulate_save_endpoint(config, env_path, new_values)

        print("\nSTEP 3 - Read back via attribute access (the way config_routes reads):")
        for key, expected in new_values.items():
            actual = getattr(config, key)
            status = "OK" if actual == expected else "STALE"
            print(f"  {key:<26} expected={expected!r:<40} actual={actual!r}  [{status}]")

        all_ok = all(getattr(config, k) == v for k, v in new_values.items())

        if all_ok:
            print("\nResult: FIX WORKS - all settings reflect the updated .env")
        else:
            print("\nResult: BUG REPRODUCED - module attributes are stale after .env write")
        return all_ok

    finally:
        os.chdir(original_cwd)


def main() -> int:
    tmp_dir = Path(tempfile.mkdtemp(prefix="tbl_env_reload_repro_"))
    try:
        success = run_scenario(tmp_dir)
        return 0 if success else 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
