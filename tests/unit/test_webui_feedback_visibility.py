"""Static guards for web UI feedback that must stay visible and localized.

These pin three browser-verified defects:

* the LLM status pill carries a static data-i18n marker, so applyToDOM resets
  the live "LLM: Ollama (N models)" text to "Checking..." on every UI language
  switch unless the status manager drops the marker and re-renders itself;
* FileActions reports open/reveal/delete results through #messages, which
  lives in the Translate tab and is hidden while the Files tab is shown, and
  the delete success message is the server's English text;
* a batch whose file failed to start ends with the green "batch completed"
  banner, which replaces the (localized) start error.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JS = ROOT / "src" / "web" / "static" / "js"


def _read(rel):
    return (JS / rel).read_text(encoding="utf-8")


def test_status_pill_is_reactive_to_locale_switch():
    src = _read("utils/status-manager.js")
    assert "localeChanged" in src, "status pill must re-render on localeChanged"
    assert "removeAttribute('data-i18n')" in src, (
        "status pill must drop the template's data-i18n marker, otherwise "
        "applyToDOM overwrites the live status with 'Checking...'"
    )


def test_file_actions_do_not_report_through_hidden_message_area():
    src = _read("files/file-actions.js")
    body = src.split("export const FileActions", 1)[1]
    for method in ("async open(", "async reveal(", "async delete("):
        start = body.index(method)
        end = body.find("\n    },", start)
        assert "MessageLogger.showMessage" not in body[start:end], (
            f"FileActions.{method.split('(')[0].split()[-1]} reports through "
            "#messages, which is invisible from the Files tab"
        )
    assert "data.message ||" not in body, "delete must not show the server's English message"


def test_batch_end_does_not_hide_start_errors():
    src = _read("translation/batch-controller.js")
    match = re.search(
        r"(.{0,200})MessageLogger\.showMessage\(t\('translation:batch_completed'\), 'success'\)",
        src,
        re.S,
    )
    assert match, "batch_completed banner not found"
    assert "if (" in match.group(1), (
        "the success banner is shown unconditionally and replaces the error "
        "of a file that failed to start"
    )
