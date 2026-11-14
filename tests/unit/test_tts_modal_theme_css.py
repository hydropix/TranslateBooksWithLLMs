"""Static checks that the audiobook (TTS) modal's form fields follow the theme.

The modal is built in src/web/static/js/tts/tts-audiobook-modal.js and styled
by src/web/static/style.css. These tests parse the stylesheet and assert the
rules that paint the modal's fields read theme tokens (CSS custom properties
that html.dark redefines) rather than colour literals, and that the dark field
rule does not use the `background` shorthand, which would wipe the chevron that
select.form-control draws with background-image.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
STYLE_CSS = ROOT / "src" / "web" / "static" / "style.css"
MODAL_JS = ROOT / "src" / "web" / "static" / "js" / "tts" / "tts-audiobook-modal.js"

VAR_RE = re.compile(r"^var\(--[\w-]+\)$")


def _rules():
    """Map each selector (whitespace-normalised) to its merged declarations."""
    css = re.sub(r"/\*.*?\*/", "", STYLE_CSS.read_text(encoding="utf-8"), flags=re.S)
    rules = {}
    for selector_group, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        decls = {}
        for decl in body.split(";"):
            if ":" not in decl:
                continue
            prop, value = decl.split(":", 1)
            decls[prop.strip().lower()] = value.strip()
        for selector in selector_group.split(","):
            key = " ".join(selector.split())
            rules.setdefault(key, {}).update(decls)
    return rules


@pytest.fixture(scope="module")
def rules():
    return _rules()


def _decls(rules, selector):
    assert selector in rules, f"missing CSS rule for {selector!r}"
    return rules[selector]


@pytest.mark.parametrize(
    "selector",
    [
        "html.dark .form-control",
        "html.dark .neu-inset-light .form-control",
        "html.dark .form-control:focus",
    ],
)
def test_dark_field_background_is_a_token_without_shorthand(rules, selector):
    decls = _decls(rules, selector)
    assert "background" not in decls, (
        f"{selector} uses the background shorthand, which resets the select chevron"
    )
    assert VAR_RE.match(decls.get("background-color", "")), decls


def test_dark_field_text_colours_are_tokens(rules):
    assert VAR_RE.match(_decls(rules, "html.dark .form-control").get("color", ""))
    assert VAR_RE.match(
        _decls(rules, "html.dark .form-control::placeholder").get("color", "")
    )


def test_modal_native_widgets_follow_the_theme(rules):
    assert _decls(rules, "#ttsModal").get("color-scheme") == "light"
    assert _decls(rules, "html.dark #ttsModal").get("color-scheme") == "dark"


def test_voice_cloning_field_is_lifted_off_the_inset_panel(rules):
    decls = _decls(rules, "html.dark #ttsModal .tts-voice-cloning .form-control")
    assert decls.get("background-color") == "var(--background-light)"
    assert 'class="tts-voice-cloning"' in MODAL_JS.read_text(encoding="utf-8")


def test_slider_track_is_a_visible_token(rules):
    # The type selector is needed to outrank the generic input[type="range"]
    # rule; --background-light is the modal surface, so the track vanished.
    decls = _decls(rules, 'input[type="range"].tts-slider')
    background = decls.get("background", "")
    assert VAR_RE.match(background), decls
    assert background != "var(--background-light)"


def test_modal_fields_carry_the_themed_classes():
    js = MODAL_JS.read_text(encoding="utf-8")
    selects = re.findall(r"<select\b[^>]*>", js)
    text_inputs = re.findall(r'<input\b[^>]*type="text"[^>]*>', js)
    ranges = re.findall(r'<input\b[^>]*type="range"[^>]*>', js)
    assert selects and text_inputs and ranges
    for tag in selects + text_inputs:
        assert 'class="form-control"' in tag, tag
    for tag in ranges:
        assert 'class="tts-slider"' in tag, tag
