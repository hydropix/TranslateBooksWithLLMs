"""JobConfig builders: web request, CLI arguments and the checkpoint round trip."""

import argparse
import json
from pathlib import Path

import pytest

from src.core.job import JobConfig, JobConfigError, OP_REFINE, OP_TERM_FIX, OP_TRANSLATE

FAKE_KEY = "sk-test-not-a-real-key"


@pytest.fixture
def dirs(tmp_path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    upload = uploads / "0123456789abcdef_book.txt"
    upload.write_text("Hello.", encoding="utf-8")
    return tmp_path, uploads, upload


@pytest.fixture(autouse=True)
def _no_env_keys(monkeypatch):
    import src.config as cfg
    for name in ("GEMINI_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "MISTRAL_API_KEY",
                 "DEEPSEEK_API_KEY", "POE_API_KEY", "NIM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(cfg, name, "", raising=False)


def _web_body(upload, **extra):
    body = {
        "file_path": str(upload),
        "file_type": "txt",
        "source_language": "English",
        "target_language": "French",
        "model": "m1",
        "llm_api_endpoint": "https://api.openai.com/v1/chat/completions",
        "llm_provider": "poe",
        "poe_api_key": FAKE_KEY,
        "output_filename": "book (French).txt",
        "input_filename": "book.txt",
        "prompt_options": {"refine": False, "text_cleanup": True},
    }
    body.update(extra)
    return body


def test_web_request_builds_a_translate_job(dirs):
    out_dir, uploads, upload = dirs
    job = JobConfig.from_web_request(_web_body(upload), output_dir=str(out_dir), uploads_dir=uploads)
    assert job.operation == OP_TRANSLATE
    assert job.origin == "web"
    assert job.provider.provider == "poe"
    assert job.provider.api_key == FAKE_KEY
    assert job.output_path == str(out_dir / "book (French).txt")
    assert job.input_filename == "book.txt"
    assert job.prompt_options["text_cleanup"] is True


def test_web_request_errors_carry_status(dirs, tmp_path):
    out_dir, uploads, upload = dirs
    with pytest.raises(JobConfigError) as missing:
        JobConfig.from_web_request(_web_body(upload, model=""), output_dir=str(out_dir),
                                   uploads_dir=uploads)
    assert missing.value.status == 400
    assert missing.value.payload == {"error": "Missing or empty field: model"}

    outside = tmp_path / "secret.txt"
    outside.write_text("x", encoding="utf-8")
    with pytest.raises(JobConfigError) as escaped:
        JobConfig.from_web_request(_web_body(outside), output_dir=str(out_dir), uploads_dir=uploads)
    assert escaped.value.status == 403


def test_web_term_fix_is_exclusive_with_refine(dirs):
    out_dir, uploads, upload = dirs
    job = JobConfig.from_web_request(
        _web_body(upload, refine_after=True, term_fix={"rules_text": "colour -> color",
                                                       "mode": "literal"}),
        output_dir=str(out_dir), uploads_dir=uploads)
    assert job.operation == OP_TERM_FIX
    assert job.refine_after is False
    assert not job.wants_refine_after


def test_refine_flags_mean_one_refine_pass():
    from src.core.llm.provider_config import ProviderConfig
    job = JobConfig(provider=ProviderConfig(provider="ollama"),
                    prompt_options={"refine": True}, refine_after=True)
    assert job.wants_refine_after
    assert JobConfig(prompt_options={"refine": True}).wants_refine_after
    assert not JobConfig(operation=OP_REFINE, prompt_options={"refine": True}).wants_refine_after


def test_persisted_form_has_no_secret_and_round_trips(dirs, monkeypatch):
    out_dir, uploads, upload = dirs
    job = JobConfig.from_web_request(_web_body(upload), output_dir=str(out_dir), uploads_dir=uploads)
    persisted = job.to_persisted()
    blob = json.dumps(persisted)
    assert FAKE_KEY not in blob
    assert not any(k == "api_key" or k.endswith("_api_key") for k in persisted)
    # The keys the frontend reads from a stored job config are still there.
    for key in ("output_filename", "model", "llm_provider", "llm_api_endpoint", "file_type",
                "source_language", "target_language", "input_filename", "prompt_options"):
        assert key in persisted

    persisted["preserved_input_path"] = "/data/uploads/job/book.txt"
    monkeypatch.setenv("POE_API_KEY", "env-key-for-resume")
    restored = JobConfig.from_checkpoint(persisted, output_dir=str(out_dir),
                                         input_path="/data/uploads/job/book.txt")
    assert restored.provider.api_key == "env-key-for-resume"
    assert restored.output_path == job.output_path
    assert restored.input_path == "/data/uploads/job/book.txt"
    assert restored.extras["preserved_input_path"] == "/data/uploads/job/book.txt"
    assert restored.to_persisted()["preserved_input_path"] == "/data/uploads/job/book.txt"
    assert restored.max_tokens_per_chunk == job.max_tokens_per_chunk


def test_checkpoint_resume_requires_a_key():
    stored = {"llm_provider": "poe", "model": "m", "output_filename": "o.txt", "file_type": "txt"}
    with pytest.raises(JobConfigError) as err:
        JobConfig.from_checkpoint(stored, output_dir="out")
    assert err.value.payload["error"] == "Missing API key for provider"
    job = JobConfig.from_checkpoint(stored, output_dir="out", overrides={"api_key": FAKE_KEY})
    assert job.provider.api_key == FAKE_KEY


def test_checkpoint_resume_overrides_and_flags(monkeypatch):
    stored = {"llm_provider": "ollama", "model": "a", "llm_api_endpoint": "http://localhost:11434/api/generate",
              "output_filename": "o.txt", "file_type": "txt", "retry_token_aligned": True}
    job = JobConfig.from_checkpoint(stored, output_dir="out")
    assert job.retry_token_aligned is False  # never inherited from storage
    job = JobConfig.from_checkpoint(stored, output_dir="out",
                                    overrides={"model": "b", "context_window": "8192",
                                               "retry_token_aligned": True,
                                               "auto_pause_on_rate_limit": False})
    assert job.model == "b" and job.context_window == 8192
    assert job.retry_token_aligned is True and job.auto_pause_on_rate_limit is False
    with pytest.raises(JobConfigError):
        JobConfig.from_checkpoint(stored, output_dir="out", overrides={"retry_token_aligned": "yes"})


def _cli_args(tmp_path, **extra):
    args = argparse.Namespace(
        input=str(tmp_path / "book.txt"), output=None, source_lang="English",
        target_lang="French", model="qwen3:14b", api_endpoint="http://localhost:11434/api/generate",
        provider="ollama", gemini_api_key="", openai_api_key="", openrouter_api_key="",
        mistral_api_key="", deepseek_api_key="", poe_api_key="", nim_api_key="",
        parallel=1, text_cleanup=False, refine=False, refine_only=False, glossary=None,
        auto_glossary=False, auto_style=False, fix_terms=None, fix_terms_mode="smart",
        fix_terms_scan=False, tts=False)
    for key, value in extra.items():
        setattr(args, key, value)
    return args


def test_cli_args_default_output_and_refine(tmp_path, monkeypatch):
    import src.config as cfg
    monkeypatch.setattr(cfg, "DEFAULT_MODEL", "qwen3:14b")
    job = JobConfig.from_cli_args(_cli_args(tmp_path, refine=True))
    assert job.output_path == str(Path(tmp_path) / "book (French).txt")
    assert job.wants_refine_after
    assert job.origin == "cli"


def test_cli_args_reject_missing_key_and_bad_combinations(tmp_path):
    with pytest.raises(JobConfigError, match="poe_api_key"):
        JobConfig.from_cli_args(_cli_args(tmp_path, provider="poe"))
    with pytest.raises(JobConfigError, match="fix-terms"):
        JobConfig.from_cli_args(_cli_args(tmp_path, fix_terms="r.txt", refine=True))
    with pytest.raises(JobConfigError, match="refine-only"):
        JobConfig.from_cli_args(_cli_args(tmp_path, input=str(tmp_path / "b.pdf"), refine_only=True))
    job = JobConfig.from_cli_args(_cli_args(tmp_path, input=str(tmp_path / "b.pgn"), refine=True))
    assert not job.wants_refine_after
    assert any("--refine is ignored" in n for n in job.notices)
