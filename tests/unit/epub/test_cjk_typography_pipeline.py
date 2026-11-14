"""
Phase 5 of plan/PLAN_CjkSourceRendering.md: wire
`apply_script_normalization_to_epub_directory` into `translate_epub_file` as
step 6.6, behind `EPUB_SCRIPT_NORMALIZATION_ENABLED`, and make sure it can
never fail an otherwise-successful job.

Reuses the container-shape builder (`_build_cjk_epub_dir`) and the real
`main.css` fixture, plus the echo-LLM/zip/attribution harness, from
`tests/unit/epub/conftest.py`. The container is packaged into a real .epub zip
here, since `translate_epub_file` extracts its input itself (step 1).

The LLM is stubbed with an identity-echo client: it locates the source content
between the configured `<SOURCE_TEXT>` tags and returns it unchanged. This
phase is about the pipeline wiring, not translation quality, and echoing keeps
every placeholder byte-identical so the run never has to engage a retry or the
token-alignment fallback.
"""
import zipfile
from pathlib import Path

import pytest

import src.core.epub.translator as translator_module
from src.core.epub.translator import translate_epub_file
from src.utils.file_utils import find_partial_output_paths

from tests.unit.epub.conftest import REAL_CSS, _disable_attribution, _echo_llm_client, _read
from tests.job_helpers import make_ctx, make_job


# `input_epub` is a fixture defined in conftest.py -- pytest injects it
# automatically, no import needed.


def _stub_create_llm_client(*args, **kwargs):
    return _echo_llm_client()


async def _translate(input_path: Path, output_path: Path, monkeypatch, **overrides) -> list:
    """Run translate_epub_file with the stubbed LLM client and no network/keys.

    Returns the list of (event, message) pairs the pipeline logged.
    """
    monkeypatch.setattr(translator_module, "create_llm_client", _stub_create_llm_client)
    _disable_attribution(monkeypatch)

    events = []

    def log_callback(event, message, **_kwargs):
        events.append((event, message))

    check_interruption = overrides.pop("check_interruption_callback", None)
    await translate_epub_file(
        make_job(str(input_path), str(output_path),
                 source_language="Chinese", target_language="French", **overrides),
        make_ctx(log_callback=log_callback, check_interruption=check_interruption),
    )
    return events


# ---------------------------------------------------------------------------
# Criterion 1 -- runs by default and normalizes the real container
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_script_normalization_runs_by_default(tmp_path, input_epub, monkeypatch):
    output_epub = tmp_path / "output.epub"
    events = await _translate(input_epub, output_epub, monkeypatch)

    css = _read(output_epub, "OEBPS/Styles/main.css")
    assert "font-family: serif;" in css
    assert "宋体" not in css

    opf = _read(output_epub, "OEBPS/content.opf")
    assert "duokan-body-font" not in opf

    # The success summary is logged, exactly like any other run.
    assert any(event == "epub_script_normalized" for event, _ in events)
    assert any(event == "epub_save_success" for event, _ in events)


# ---------------------------------------------------------------------------
# Criterion 2 -- the flag actually disables the pass
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_script_normalization_disabled_leaves_css_and_opf_untouched(
    tmp_path, input_epub, monkeypatch
):
    monkeypatch.setattr("src.core.epub.book_postprocess.EPUB_SCRIPT_NORMALIZATION_ENABLED", False)
    output_epub = tmp_path / "output_disabled.epub"
    events = await _translate(input_epub, output_epub, monkeypatch)

    css = _read(output_epub, "OEBPS/Styles/main.css")
    assert css == REAL_CSS.read_text(encoding="utf-8")

    opf = _read(output_epub, "OEBPS/content.opf")
    assert 'name="duokan-body-font"' in opf

    assert not any(event.startswith("epub_script_norm") for event, _ in events)
    assert any(event == "epub_save_success" for event, _ in events)


# ---------------------------------------------------------------------------
# Criterion 3 -- partial-output parity (the user's original question)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_partial_output_carries_the_same_normalization_as_the_completed_run(
    tmp_path, input_epub, monkeypatch
):
    complete_output = tmp_path / "complete.epub"
    await _translate(input_epub, complete_output, monkeypatch)

    # CONTENT_OPF's spine has two files (cover, intro); interrupt right before
    # the second one, i.e. "after the first file" completed translation.
    partial_target = tmp_path / "partial.epub"
    calls = {"n": 0}

    def interrupt_after_first_file():
        calls["n"] += 1
        return calls["n"] > 1

    events = await _translate(
        input_epub, partial_target, monkeypatch,
        check_interruption_callback=interrupt_after_first_file,
    )
    assert any(event == "epub_translation_interrupted" for event, _ in events)
    assert any(event == "epub_partial_output_marked" for event, _ in events)

    partial_paths = find_partial_output_paths(str(partial_target))
    assert len(partial_paths) == 1
    partial_output = Path(partial_paths[0])

    complete_css = _read(complete_output, "OEBPS/Styles/main.css")
    partial_css = _read(partial_output, "OEBPS/Styles/main.css")
    assert complete_css == partial_css
    assert "font-family: serif;" in partial_css

    complete_opf = _read(complete_output, "OEBPS/content.opf")
    partial_opf = _read(partial_output, "OEBPS/content.opf")
    assert "duokan-body-font" not in complete_opf
    assert "duokan-body-font" not in partial_opf


# ---------------------------------------------------------------------------
# Criterion 4 -- a raising normalization pass never fails the job
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_normalization_failure_never_fails_the_job(tmp_path, input_epub, monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("simulated failure inside the normalization pass")

    monkeypatch.setattr(
        "src.core.epub.book_postprocess.apply_script_normalization_to_epub_directory", _raise
    )

    output_epub = tmp_path / "output_failure.epub"
    events = await _translate(input_epub, output_epub, monkeypatch)

    assert output_epub.exists()
    with zipfile.ZipFile(output_epub) as archive:
        namelist = archive.namelist()
        assert "mimetype" in namelist
        assert "OEBPS/content.opf" in namelist

    event_names = [event for event, _ in events]
    assert "epub_script_norm_failed" in event_names
    # The rest of the pipeline's logging still happened: the job is not
    # considered failed just because this cosmetic pass raised.
    assert "epub_save_success" in event_names
