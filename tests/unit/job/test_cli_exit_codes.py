"""translate.py exit codes: 0 only for a completed job.

Driven through ``translate.main(argv)`` with fake engines patched onto
``src.core.adapters`` and a checkpoint database under tmp_path (the CLI's
``CheckpointManager()`` creates ``data/`` relative to the working directory).
"""

import sqlite3

import pytest

import translate
from src.core import adapters
from src.core.job import JobResult
from src.core.llm.exceptions import RateLimitError


@pytest.fixture
def cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.utils.notifier.notify", lambda *a, **k: None)
    # Without a .env the CLI prints the effective settings and sleeps 5 s.
    monkeypatch.setattr("src.config.ENV_FILE_MISSING", False)

    async def no_tts(self, job, output_path):
        return None

    # TTS_ENABLED may be on in the local .env; never generate audio here.
    monkeypatch.setattr(translate.CliJobSinks, "run_tts", no_tts)
    source = tmp_path / "book.txt"
    source.write_text("Hello.\n\nWorld.\n", encoding="utf-8")
    return tmp_path, source


def _argv(source, *extra):
    return ["-i", str(source), "-sl", "English", "-tl", "French", "-m", "llama3",
            "--provider", "ollama", "--api_endpoint", "http://localhost:11434",
            "--no-color", *extra]


def _engine(stats, *, write=True, engine_ok=True):
    async def translate_file(job, ctx):
        if write:
            with open(job.output_path, "w", encoding="utf-8") as handle:
                handle.write("Bonjour.\n\nMonde.\n")
        ctx.emit(stats)
        return JobResult.from_run(stats, job.output_path, engine_ok=engine_ok)
    return translate_file


COMPLETE = {'total_chunks': 2, 'completed_chunks': 2, 'failed_chunks': 0, 'unfinished_chunks': 0}
PARTIAL = {'total_chunks': 2, 'completed_chunks': 2, 'failed_chunks': 1, 'unfinished_chunks': 0}


def _job_ids(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "data" / "jobs.db"))
    try:
        return [row[0] for row in conn.execute(
            "SELECT translation_id FROM translation_jobs")]
    finally:
        conn.close()


def test_completed_job_exits_zero(cli, monkeypatch):
    tmp_path, source = cli
    monkeypatch.setattr(adapters, "translate_file", _engine(COMPLETE))

    assert translate.main(_argv(source)) == 0
    assert (tmp_path / "book (French).txt").read_text(encoding="utf-8").startswith("Bonjour")


def test_partial_job_exits_one(cli, monkeypatch):
    _tmp_path, source = cli
    monkeypatch.setattr(adapters, "translate_file", _engine(PARTIAL, engine_ok=False))
    assert translate.main(_argv(source)) == 1


def test_failed_job_exits_one(cli, monkeypatch):
    _tmp_path, source = cli
    monkeypatch.setattr(adapters, "translate_file", _engine(COMPLETE, write=False, engine_ok=False))
    assert translate.main(_argv(source)) == 1


def test_crashing_engine_exits_one(cli, monkeypatch):
    _tmp_path, source = cli

    async def boom(job, ctx):
        raise RuntimeError("boom")

    monkeypatch.setattr(adapters, "translate_file", boom)
    assert translate.main(_argv(source)) == 1


def test_rate_limited_job_exits_one_and_resumes_to_zero(cli, monkeypatch):
    tmp_path, source = cli

    async def rate_limited(job, ctx):
        raise RateLimitError("429", retry_after=30, provider="ollama")

    monkeypatch.setattr(adapters, "translate_file", rate_limited)
    assert translate.main(_argv(source)) == 1

    (job_id,) = _job_ids(tmp_path)
    assert job_id.startswith("cli_")

    resumed = []

    async def finish(job, ctx):
        resumed.append((ctx.translation_id, job.input_path))
        return await _engine(COMPLETE)(job, ctx)

    monkeypatch.setattr(adapters, "translate_file", finish)
    assert translate.main(["--resume", job_id, "--no-color"]) == 0
    assert resumed and resumed[0][0] == job_id
    assert (tmp_path / "book (French).txt").exists()


def test_nothing_translated_job_exits_one_prints_the_resume_hint_and_resumes(cli, monkeypatch, capsys):
    tmp_path, source = cli
    nothing = {'total_chunks': 2, 'completed_chunks': 2, 'failed_chunks': 2, 'unfinished_chunks': 2}
    monkeypatch.setattr(adapters, "translate_file", _engine(nothing, engine_ok=False))
    assert translate.main(_argv(source)) == 1

    (job_id,) = _job_ids(tmp_path)
    assert f"--resume {job_id}" in capsys.readouterr().out

    monkeypatch.setattr(adapters, "translate_file", _engine(COMPLETE))
    assert translate.main(["--resume", job_id, "--no-color"]) == 0


def test_run_cli_job_engines_argument(cli):
    """`run_cli_job(..., engines=...)` drives the same runner without patching."""
    tmp_path, source = cli
    job = translate.JobConfig.from_cli_args(translate.build_parser().parse_args(_argv(source)))
    job = job.with_changes(tts=None)

    class Logger:
        def __getattr__(self, name):
            return lambda *a, **k: None

        def create_legacy_callback(self):
            return lambda *a, **k: None

    result = translate.run_cli_job(
        job, Logger(), translation_id="cli_direct",
        checkpoint_manager=translate.CheckpointManager(db_path=str(tmp_path / "data" / "jobs.db")),
        engines={"translate_file": _engine(PARTIAL, engine_ok=False)})
    assert result.exit_code == 1


@pytest.mark.parametrize("extra", [
    ("--fix-terms", "rules.txt", "--refine"),
    ("--fix-terms", "rules.txt", "--refine-only"),
    ("--fix-terms-scan",),
])
def test_bad_argument_combination_is_a_usage_error(cli, monkeypatch, extra):
    _tmp_path, source = cli
    calls = []

    async def never(job, ctx):
        calls.append(job)

    monkeypatch.setattr(adapters, "translate_file", never)
    with pytest.raises(SystemExit) as info:
        translate.main(_argv(source, *extra))
    assert info.value.code == 2
    assert calls == []


def test_missing_input_is_a_usage_error(cli):
    with pytest.raises(SystemExit) as info:
        translate.main(["-tl", "French"])
    assert info.value.code == 2


def test_resume_with_unknown_id_is_a_usage_error(cli):
    with pytest.raises(SystemExit) as info:
        translate.main(["--resume", "cli_doesnotexist"])
    assert info.value.code == 2
