"""A failed or partial translation must be reported as a failure.

`translate_file` used to return True unconditionally for EPUB ("Legacy
function doesn't return success status"), and the CLI ignored the adapters'
return value altogether: it logged "Translation Completed Successfully",
sent the success notification and exited 0 even when chunks had failed or the
output was never written. A raised exception also exited 0.
"""

import argparse

import pytest

import translate
from src.core.adapters.translate_file import translate_file as adapter_translate_file


class _FakeLogger:
    def __init__(self):
        self.infos = []
        self.errors = []

    def info(self, msg, *args, **kwargs):
        self.infos.append(msg)

    def warning(self, msg, *args, **kwargs):
        pass

    def error(self, msg, *args, **kwargs):
        self.errors.append(msg)


def _args(**overrides):
    base = dict(
        input="book.txt", output="book (French).txt",
        source_lang="English", target_lang="French",
        model="m", provider="ollama", api_endpoint="http://localhost:11434",
        gemini_api_key=None, openai_api_key=None, openrouter_api_key=None,
        mistral_api_key=None, deepseek_api_key=None, poe_api_key=None,
        nim_api_key=None, parallel=1, refine_only=False, tts=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture
def cli(monkeypatch):
    notified = []
    monkeypatch.setattr(translate, "notify", lambda event, payload: notified.append(event))
    monkeypatch.setattr(translate, "CheckpointManager", lambda: object())
    return notified


def _stub_job(monkeypatch, name, result=None, exc=None):
    async def fake(**kwargs):
        if exc:
            raise exc
        return result
    monkeypatch.setattr(translate, name, fake)


@pytest.mark.parametrize("name,refine_only", [
    ("translate_file", False),
    ("refine_file", True),
])
def test_cli_reports_a_false_result_as_failure(monkeypatch, cli, name, refine_only):
    _stub_job(monkeypatch, name, result=False)
    logger = _FakeLogger()

    code = translate._run_job(_args(refine_only=refine_only), logger, {}, None, None)

    assert code != 0
    assert translate.EVENT_FAILURE in cli and translate.EVENT_SUCCESS not in cli
    assert not any("Completed Successfully" in m for m in logger.infos)
    assert logger.errors


def test_cli_exits_non_zero_on_exception(monkeypatch, cli):
    _stub_job(monkeypatch, "translate_file", exc=RuntimeError("boom"))
    code = translate._run_job(_args(), _FakeLogger(), {}, None, None)
    assert code != 0
    assert cli == [translate.EVENT_FAILURE]


def test_cli_success_still_exits_zero(monkeypatch, cli):
    _stub_job(monkeypatch, "translate_file", result=True)
    logger = _FakeLogger()
    code = translate._run_job(_args(), logger, {}, None, None)
    assert code == 0
    assert cli == [translate.EVENT_SUCCESS]
    assert any("Completed Successfully" in m for m in logger.infos)


@pytest.mark.asyncio
@pytest.mark.parametrize("epub_result,expected", [(True, True), (False, False), (None, False)])
async def test_translate_file_propagates_the_epub_result(tmp_path, monkeypatch, epub_result, expected):
    import src.core.epub.translator as epub_translator

    async def fake_translate_epub_file(**kwargs):
        return epub_result

    monkeypatch.setattr(epub_translator, "translate_epub_file", fake_translate_epub_file)
    src = tmp_path / "book.epub"
    src.write_bytes(b"PK\x03\x04")  # detection goes by extension

    result = await adapter_translate_file(
        input_filepath=str(src),
        output_filepath=str(tmp_path / "out.epub"),
        source_language="English",
        target_language="French",
        model_name="m",
        llm_provider="ollama",
        checkpoint_manager=None,
        translation_id=None,
    )

    assert result is expected
