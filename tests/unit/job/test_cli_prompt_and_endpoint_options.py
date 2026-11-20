"""CLI parity options: provider endpoints, --glossary-id, --custom-instruction.

Goes through the real ``translate.build_parser()`` so a parser regression
(a renamed dest, a lost ``type=int``) fails here too. Fully offline.
"""
import pytest

import src.config as cfg
import translate
from src.core.job import JobConfig, JobConfigError
from src.core.job.prepare import snapshot_glossary

LOCAL = "http://localhost:1234/v1/chat/completions"
OFFICIAL = "https://api.openai.com/v1/chat/completions"
FAKE_KEY = "sk-test-not-a-real-key"


def _job(*extra, provider="ollama"):
    args = translate.build_parser().parse_args([
        "-i", "book.txt", "-sl", "English", "-tl", "French", "-m", "some-model",
        "--provider", provider, "--openai_api_key", "", "--nim_api_key", FAKE_KEY, *extra,
    ])
    return JobConfig.from_cli_args(args)


# --- endpoints -------------------------------------------------------------

def test_explicit_api_endpoint_beats_the_provider_endpoint():
    job = _job("--api_endpoint", LOCAL, "--openai_api_endpoint", OFFICIAL, provider="openai")
    assert job.endpoint == LOCAL


def test_openai_uses_its_own_endpoint_without_api_endpoint():
    job = _job("--openai_api_endpoint", LOCAL, provider="openai")
    assert job.endpoint == LOCAL


def test_openai_default_endpoint_comes_from_env_config():
    args = translate.build_parser().parse_args(["-i", "book.txt", "--provider", "openai"])
    assert args.api_endpoint is None
    assert args.openai_api_endpoint == cfg.OPENAI_API_ENDPOINT


def test_key_check_uses_the_resolved_endpoint():
    with pytest.raises(JobConfigError, match="openai_api_key"):
        _job("--openai_api_endpoint", OFFICIAL, provider="openai")
    # A local OpenAI-compatible server needs no key, even if the .env
    # endpoint points at the official API.
    _job("--api_endpoint", LOCAL, "--openai_api_endpoint", OFFICIAL, provider="openai")


def test_nim_uses_its_own_endpoint_without_api_endpoint():
    job = _job("--nim_api_endpoint", "https://nim.example.test/v1/chat/completions", provider="nim")
    assert job.endpoint == "https://nim.example.test/v1/chat/completions"


def test_ollama_falls_back_to_the_configured_endpoint():
    assert _job().endpoint == cfg.API_ENDPOINT


@pytest.mark.parametrize("flag", ["--mistral_api_endpoint", "--deepseek_api_endpoint",
                                  "--poe_api_endpoint", "--openrouter_api_endpoint"])
def test_no_endpoint_flag_for_providers_with_a_fixed_endpoint(flag):
    with pytest.raises(SystemExit):
        translate.build_parser().parse_args(["-i", "book.txt", flag, "http://x"])


# --- --glossary-id ---------------------------------------------------------

def test_glossary_id_reaches_prompt_options_as_int():
    assert _job("--glossary-id", "3").prompt_options["glossary_id"] == 3


def test_glossary_id_must_be_an_integer():
    with pytest.raises(SystemExit):
        translate.build_parser().parse_args(["-i", "book.txt", "--glossary-id", "abc"])


def test_glossary_id_and_glossary_file_are_exclusive():
    with pytest.raises(JobConfigError, match="cannot be combined"):
        _job("--glossary-id", "3", "--glossary", "g.json")


def test_glossary_id_disables_auto_glossary():
    job = _job("--glossary-id", "3", "--auto-glossary")
    assert "glossary_auto" not in job.prompt_options
    assert any("--glossary-id was provided" in n for n in job.notices)


class _Store:
    def get_glossary(self, glossary_id):
        return None


def test_unknown_glossary_id_is_reported():
    options = {"glossary_id": 42}
    snapshot_glossary(options, _Store)
    assert options["glossary_load_error"] == "glossary 42 not found"
    assert "glossary_terms" not in options


# --- --custom-instruction --------------------------------------------------

def test_custom_instruction_reaches_prompt_options():
    job = _job("--custom-instruction", "wuxia.yaml")
    assert job.prompt_options["custom_instruction_file"] == "wuxia.yaml"


def test_no_custom_instruction_by_default():
    assert "custom_instruction_file" not in _job().prompt_options
