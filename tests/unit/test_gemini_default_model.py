"""The default Gemini model must not be one Google has shut down (gemini-2.0-flash, 2026-06-01)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import config
from src.core.llm.factory import create_llm_provider
from src.core.llm.providers.gemini import GeminiProvider

RETIRED_MODEL = "gemini-2.0-flash"


def test_config_default_gemini_model_is_not_retired():
    defaults = {env_name: default for _, env_name, default in config._RELOADABLE_ENV_SETTINGS}
    assert defaults["GEMINI_MODEL"] != RETIRED_MODEL


def test_provider_default_model_is_not_retired():
    assert GeminiProvider(api_key="test-key").model != RETIRED_MODEL


def test_factory_default_model_is_not_retired():
    provider = create_llm_provider("gemini", gemini_api_key="test-key")
    assert provider.model != RETIRED_MODEL
