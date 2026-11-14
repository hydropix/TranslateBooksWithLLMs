"""Resuming a TXT job paused during its refine-after pass, with the REAL engines.

The JobRunner tests drive fake engines, so they cannot see what the real TXT
refine engine does on a rate-limit pause: it saves the partially refined text
(plus the "Refined with ..." attribution footer) over the translated output,
in place. The resume then re-enters the refine phase and refines that file
again from chunk 0, so the footer ends up in the output twice and the chunks
refined before the pause are refined a second time.

Only the transport is fake (tests/characterization/fake_llm.py echoes the
source block); translate.main, JobRunner, translate_file and refine_file are
the production code.
"""

import pytest

import translate
from src.core.llm.exceptions import RateLimitError
from tests.characterization import fake_llm


PARAGRAPHS = [
    f"Paragraph {i}. The quick brown fox jumps over the lazy dog near the river bank, "
    f"and then it rests under an old oak tree for a while."
    for i in range(1, 121)
]


class _Budget:
    """Shared request counter: requests beyond `limit` hit a 429."""

    def __init__(self, limit):
        self.limit = limit
        self.calls = 0


def _install_provider(monkeypatch, budget):
    class LimitedEcho(fake_llm.FakeEchoProvider):
        async def _send(self, request):
            budget.calls += 1
            if budget.limit is not None and budget.calls > budget.limit:
                raise RateLimitError("429 Too Many Requests (test)", provider="fake")
            return await super()._send(request)

    def factory(provider_type="ollama", **kwargs):
        return LimitedEcho(model="fake-echo")

    for target in fake_llm._PATCH_TARGETS:
        monkeypatch.setattr(target, factory, raising=True)
    fake_llm.allow_echo(monkeypatch)


@pytest.fixture
def cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.utils.notifier.notify", lambda *a, **k: None)
    monkeypatch.setattr("src.config.ENV_FILE_MISSING", False)
    # Pause instead of sleeping and auto-resuming inside the test.
    monkeypatch.setattr("src.config.RATE_LIMIT_AUTO_RESUME_DELAY", 0, raising=False)

    async def no_tts(self, job, output_path):
        return None

    monkeypatch.setattr(translate.CliJobSinks, "run_tts", no_tts)
    source = tmp_path / "book.txt"
    source.write_text("\n\n".join(PARAGRAPHS) + "\n", encoding="utf-8")
    return tmp_path, source


def _argv(source, output):
    return ["-i", str(source), "-o", str(output), "-sl", "English", "-tl", "French",
            "-m", "fake-echo", "--provider", "openai",
            "--api_endpoint", "http://127.0.0.1:9/v1/chat/completions",
            "--no-color", "--refine"]


def _job_id(tmp_path):
    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "data" / "jobs.db"))
    try:
        (row,) = conn.execute(
            "SELECT translation_id FROM translation_jobs WHERE status != 'completed'").fetchall()
        return row[0]
    finally:
        conn.close()


def test_resumed_refine_pass_writes_the_attribution_footer_once(cli, monkeypatch):
    tmp_path, source = cli
    output = tmp_path / "out.txt"

    # First run without a limit only to learn how many requests a full
    # translate + refine pass takes, then pause halfway through the refine pass.
    probe = _Budget(None)
    _install_provider(monkeypatch, probe)
    assert translate.main(_argv(source, tmp_path / "probe.txt")) == 0
    full_run_calls = probe.calls
    translate_calls = full_run_calls // 2
    assert translate_calls >= 3, "the input must split into several chunks"

    budget = _Budget(translate_calls + 2)   # pause after 2 refined chunks
    _install_provider(monkeypatch, budget)
    assert translate.main(_argv(source, output)) == 1
    job_id = _job_id(tmp_path)

    budget.limit = None
    assert translate.main(["--resume", job_id, "--no-color"]) == 0

    text = output.read_text(encoding="utf-8")
    assert text.count("Refined with") == 1, (
        "the resumed refine pass re-read its own partial output, footer included")
    reference = (tmp_path / "probe.txt").read_text(encoding="utf-8")
    assert text == reference
