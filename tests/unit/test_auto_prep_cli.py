"""
CLI parity for auto glossary / auto style (Phase 4 of
plan/PLAN_AutoGlossaryStyle.md).

``--auto-glossary`` / ``--auto-style`` become ``prompt_options['glossary_auto']``
/ ``['style_auto']`` in ``JobConfig.from_cli_args`` (the old precedence warnings
are ``job.notices``); the passes themselves run in the job runner's prepare
phase (``src.core.job.prepare.apply_auto_prep``), exactly as for the web.

Fully offline: the provider factory is passed as ``create_provider=`` (or
``src.core.llm.factory.create_llm_provider`` is patched), and
``auto_prep.build_auto_prompt_options`` / ``extract_source_text`` are stubbed.
"""
import asyncio
import json

import pytest

import translate
from src.core import auto_prep
from src.core.job import JobResult, JobStatus
from src.core.job import prepare
from src.core.llm import factory
from src.persistence.checkpoint_manager import CheckpointManager


class FakeLogger:
    """Records log calls; enough of the unified CLI logger for ``CliJobSinks``."""

    def __init__(self):
        self.infos = []
        self.warnings = []
        self.errors = []
        self.legacy = []

    def info(self, msg, *args, **kwargs):
        self.infos.append(msg)

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(msg)

    def error(self, msg, *args, **kwargs):
        self.errors.append(msg)

    def update_progress(self, *args, **kwargs):
        pass

    def create_legacy_callback(self):
        def callback(key, message="", data=None):
            self.legacy.append((key, message))
        return callback


class FakeClient:
    """Duck-typed stand-in for an LLM provider: only `close()` is ever awaited."""

    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class Factory:
    def __init__(self):
        self.client = FakeClient()
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.client


class Log:
    def __init__(self):
        self.entries = []

    def __call__(self, key, message="", data=None):
        self.entries.append((key, message))


def _args(*extra, input_path="book.txt"):
    return translate.build_parser().parse_args([
        "-i", str(input_path), "-sl", "English", "-tl", "French", "-m", "llama3",
        "--provider", "ollama", "--api_endpoint", "http://localhost:11434", *extra,
    ])


def _job(*extra, **kwargs):
    return translate.JobConfig.from_cli_args(_args(*extra, **kwargs))


def _apply(job, fake_build, monkeypatch, create=None):
    monkeypatch.setattr(auto_prep, "build_auto_prompt_options", fake_build)
    create = create or Factory()
    options = job.engine_prompt_options()
    log = Log()
    asyncio.run(prepare.apply_auto_prep(job, options, log, create_provider=create))
    return options, log, create


@pytest.fixture(autouse=True)
def _fake_extract_source_text(monkeypatch):
    """No source file is read in this module."""
    monkeypatch.setattr(
        auto_prep, "extract_source_text",
        lambda **kwargs: "Some source text about a wizard.",
    )


@pytest.fixture
def glossary_file(tmp_path):
    path = tmp_path / "f.json"
    path.write_text(json.dumps([{"source": "Wizard", "target": "Magicien"}]), encoding="utf-8")
    return path


def _warnings(job):
    return [n for n in job.notices if n.startswith("⚠️")]


def test_auto_glossary_skipped_when_glossary_given(monkeypatch, glossary_file):
    """--auto-glossary --glossary f.json --auto-style: glossary is dropped,
    exactly one warning notice, and the still-enabled style pass receives
    want_glossary=False."""
    job = _job("--auto-glossary", "--auto-style", "--glossary", str(glossary_file))

    assert len(_warnings(job)) == 1
    assert "--auto-glossary ignored: --glossary was provided." in _warnings(job)[0]
    assert 'glossary_auto' not in job.prompt_options
    assert job.prompt_options['style_auto'] is True
    assert job.prompt_options['glossary_terms'] == {"Wizard": "Magicien"}

    captured = {}

    async def fake_build(**kwargs):
        captured.update(kwargs)
        return {"custom_instructions": "be terse"}

    options, _log, create = _apply(job, fake_build, monkeypatch)

    assert captured["want_glossary"] is False
    assert captured["want_style"] is True
    assert options["custom_instructions"] == "be terse"
    assert options["glossary_terms"] == {"Wizard": "Magicien"}
    assert create.client.closed is True


def test_auto_glossary_skipped_in_refine_only_mode(monkeypatch):
    """--auto-glossary --refine-only --auto-style: glossary is dropped with one
    warning; style still runs, with source_language forced to target_language
    (decision D7)."""
    job = _job("--auto-glossary", "--auto-style", "--refine-only")

    glossary_warnings = [n for n in _warnings(job) if "--auto-glossary" in n]
    assert glossary_warnings == ["⚠️ --auto-glossary ignored in --refine-only mode."]
    assert 'glossary_auto' not in job.prompt_options

    captured = {}

    async def fake_build(**kwargs):
        captured.update(kwargs)
        return {"refinement_instructions": "polish it"}

    options, _log, _create = _apply(job, fake_build, monkeypatch)

    assert captured["want_glossary"] is False
    assert captured["want_style"] is True
    assert captured["source_language"] == "French"
    assert captured["target_language"] == "French"
    assert options["refinement_instructions"] == "polish it"


def test_auto_style_alone_merges_fragment(monkeypatch):
    """--auto-style with no other flag merges the fake fragment into the
    prompt options and leaves the existing keys alone."""
    job = _job("--auto-style")
    assert _warnings(job) == []
    before = dict(job.prompt_options)

    async def fake_build(**kwargs):
        return {"custom_instructions": "keep it dry"}

    options, _log, create = _apply(job, fake_build, monkeypatch)

    assert options == {**before, "custom_instructions": "keep it dry"}
    assert options["text_cleanup"] is False
    assert create.client.closed is True


def test_raising_build_leaves_prompt_options_untouched(monkeypatch):
    """A raising `build_auto_prompt_options` never escapes and never mutates
    the prompt options; the client is still closed and a warning is logged."""
    job = _job("--auto-glossary")
    before = dict(job.prompt_options)

    async def fake_build(**kwargs):
        raise RuntimeError("boom")

    # Must not raise.
    options, log, create = _apply(job, fake_build, monkeypatch)

    assert options == before
    assert any("Auto mode failed" in message for _key, message in log.entries)
    assert create.client.closed is True


def test_neither_flag_never_creates_client(monkeypatch):
    """With neither --auto-glossary nor --auto-style, no client is created."""
    job = _job()
    assert 'glossary_auto' not in job.prompt_options
    assert 'style_auto' not in job.prompt_options
    calls = []

    def fake_build(**kwargs):
        calls.append("build")

    options, log, create = _apply(job, fake_build, monkeypatch)

    assert calls == []
    assert create.calls == []
    assert options == dict(job.prompt_options)
    assert log.entries == []


def test_close_awaited_even_when_the_builder_raises(monkeypatch):
    """`client.close()` runs in a `finally`, so it is awaited even though
    `build_auto_prompt_options` raised."""
    job = _job("--auto-style")

    async def fake_build(**kwargs):
        raise ValueError("kaboom")

    options, _log, create = _apply(job, fake_build, monkeypatch)

    assert create.client.closed is True
    assert "custom_instructions" not in options


def test_cli_job_runs_the_auto_passes_in_the_runner_prepare_phase(tmp_path, monkeypatch):
    """End to end through `run_cli_job`: the auto style reaches the engine and
    is persisted with the job, so a `--resume` reuses it."""
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "book.txt"
    source.write_text("Once upon a time.", encoding="utf-8")
    create = Factory()
    monkeypatch.setattr(factory, "create_llm_provider", create)

    async def fake_build(**kwargs):
        return {"custom_instructions": "auto style"}

    monkeypatch.setattr(auto_prep, "build_auto_prompt_options", fake_build)
    seen = {}

    async def translate_file(job, ctx):
        seen.update(job.prompt_options)
        with open(job.output_path, "w", encoding="utf-8") as handle:
            handle.write("Il était une fois.")
        stats = {"total_chunks": 1, "completed_chunks": 1, "failed_chunks": 0}
        ctx.emit(stats)
        return JobResult.from_run(stats, job.output_path, engine_ok=True)

    manager = CheckpointManager(db_path=str(tmp_path / "data" / "jobs.db"))
    # TTS_ENABLED may be on in the local .env; no audio in a unit test.
    job = _job("--auto-style", input_path=source).with_changes(tts=None)
    result = translate.run_cli_job(job, FakeLogger(), translation_id="cli_auto",
                                   checkpoint_manager=manager,
                                   engines={"translate_file": translate_file})

    assert result.status is JobStatus.COMPLETED
    assert len(create.calls) == 1 and create.client.closed is True
    assert seen["custom_instructions"] == "auto style"
    stored = manager.get_job("cli_auto")["config"]["prompt_options"]
    assert stored["custom_instructions"] == "auto style"
