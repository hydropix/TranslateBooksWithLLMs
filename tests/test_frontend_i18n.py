"""
Tests for the frontend i18n system.

Three regressions are guarded against:

1. Adding a new English string and forgetting to mirror it in the other
   supported locales (or vice versa). Every namespace JSON must expose the
   exact same set of keys across all locales — `en` is the reference.

2. Adding markup with hardcoded user-visible copy that bypasses i18next.
   Every user-visible text node and attribute (placeholder, title,
   aria-label, alt, optgroup label) in the main template must be wired
   through `data-i18n` / `data-i18n-html` / `data-i18n-attr`.

3. Setting a user-visible DOM property/attribute from JS with a plain
   string literal instead of going through `t(...)`. The scan covers
   `.textContent`, `.innerText`, `.placeholder`, `.title`, `.alt`,
   `.ariaLabel`, and `setAttribute('placeholder'|'title'|'aria-label'|'alt', ...)`.
   Template literals (backticks) are NOT scanned — the static parts of a
   template literal can still hide hardcoded English; reviewers should
   keep an eye on backtick assignments by sight.

Inputs:
  - `src/web/static/locales/<locale>/<namespace>.json`
  - `src/web/templates/translation_interface.html`
  - `src/web/static/js/**/*.js` (excluding `vendor/`)

If a hardcoded string is intentional (e.g. language autonyms shown
literally regardless of UI locale), add it to the allowlists at the top
of the file with a one-line rationale.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from lxml import html as lhtml

ROOT = Path(__file__).resolve().parent.parent
LOCALES_DIR = ROOT / "src" / "web" / "static" / "locales"
TEMPLATE = ROOT / "src" / "web" / "templates" / "translation_interface.html"
JS_DIR = ROOT / "src" / "web" / "static" / "js"

# Must stay in sync with SUPPORTED_LOCALES / NAMESPACES in
# src/web/static/js/i18n/i18n.js — the test will fail loudly if a locale
# directory or namespace file is missing.
SUPPORTED_LOCALES = ["en", "fr", "es", "de", "zh-CN", "ja", "ko"]
NAMESPACES = ["common", "translation", "settings", "glossary", "files", "tts", "errors", "sample", "style", "ask"]
REFERENCE_LOCALE = "en"


# ---------------------------------------------------------------------------
# Key parity
# ---------------------------------------------------------------------------

def _flatten_keys(node, prefix: str = ""):
    """Yield every leaf-key path of a nested dict, dot-joined."""
    if isinstance(node, dict):
        for k, v in node.items():
            path = f"{prefix}.{k}" if prefix else k
            yield from _flatten_keys(v, path)
    else:
        yield prefix


def _load_namespace(locale: str, namespace: str) -> dict:
    path = LOCALES_DIR / locale / f"{namespace}.json"
    assert path.is_file(), f"Missing locale file: {path.relative_to(ROOT)}"
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


@pytest.mark.parametrize("namespace", NAMESPACES)
@pytest.mark.parametrize(
    "locale", [loc for loc in SUPPORTED_LOCALES if loc != REFERENCE_LOCALE]
)
def test_locale_key_parity(locale: str, namespace: str) -> None:
    """Every locale must define exactly the same keys as `en` for each namespace."""
    reference = set(_flatten_keys(_load_namespace(REFERENCE_LOCALE, namespace)))
    target = set(_flatten_keys(_load_namespace(locale, namespace)))

    missing = sorted(reference - target)
    extra = sorted(target - reference)

    parts = []
    if missing:
        parts.append(
            f"\n  Missing in {locale}/{namespace}.json ({len(missing)}):\n    "
            + "\n    ".join(missing)
        )
    if extra:
        parts.append(
            f"\n  Extra in {locale}/{namespace}.json ({len(extra)}) "
            f"(absent from {REFERENCE_LOCALE}/{namespace}.json):\n    "
            + "\n    ".join(extra)
        )

    assert not parts, (
        f"Key mismatch between {REFERENCE_LOCALE}/{namespace}.json and "
        f"{locale}/{namespace}.json — keep all locales in lockstep:"
        + "".join(parts)
    )


# ---------------------------------------------------------------------------
# Hardcoded user-visible text in template
# ---------------------------------------------------------------------------

# Selects whose <option> text is intentionally not translated.
# Either because option text is a language autonym, a brand/product name, or a
# technical token (HTTP verb, etc.) shown literally regardless of UI locale.
_EXEMPT_OPTION_SELECT_IDS = {
    # Language pickers — option text is the language autonym ("English", ...)
    "sourceLang",
    "targetLang",
    "sampleSourceLang",
    "sampleTargetLang",
    "glossaryEditorSourceLang",
    "glossaryEditorTargetLang",
    "askAnswerLanguage",
    # UI locale picker — option text is the locale autonym ("English", "简体中文", ...)
    "uiLocaleSelect",
    # AI provider picker — product / brand names with English technical hints
    "llmProvider",
    # HTTP method picker — verbs are technical tokens
    "notifyWebhookMethod",
}

# Attributes that surface text to the user and therefore must go through i18n.
_USER_VISIBLE_ATTRS = ("placeholder", "title", "aria-label", "alt", "label")

# Text that carries no translatable content: digits, units, separators,
# punctuation, common UI glyphs (sort arrows, close ×, etc.).
_TRIVIAL_TEXT_RE = re.compile(
    "^[\\s\\d%·•\\-–—|/\\\\*+.,:;!?()\\[\\]{}<>=°@#&'\"`~^$"
    "⇅×↑↓✕✖"
    "]*$"
)

# Jinja expressions/statements — replaced with spaces (preserving newlines) so
# they don't appear as text and sourceline numbers stay accurate.
_JINJA_RE = re.compile(r"\{\{.*?\}\}|\{%.*?%\}", re.DOTALL)

# Known intentional exceptions. Keep this list short and justified — every entry
# is a literal we've decided NOT to translate. Format: (tag, attribute or
# "<text>", exact literal value as it appears stripped in the template).
_HARDCODE_ALLOWLIST: set[tuple[str, str, str]] = {
    # --- brand / product identity ----------------------------------------
    ("img", "alt", "TBL Logo"),
    ("h2", "<text>", "TBL"),
    # `v` prefix in front of the Jinja-rendered app version (e.g. "v1.2.3").
    # Universal version-number convention, not user-translatable. Both
    # entries cover the visible content and the tooltip on #appVersion.
    ("span", "<text>", "v"),
    ("span", "title", "TBL v"),

    # --- dynamic placeholders replaced by JS at runtime ------------------
    ("span", "<text>", "EN"),     # #uiLocaleDisplay — short locale code
    ("h3", "<text>", "0s"),       # #elapsedTime — timer initial value
    ("span", "<text>", "ON"),     # #notifyStatusBadge — toggle badge
    ("span", "<text>", "0 MB"),   # #totalFileSize — initial size

    # --- notification preset buttons: third-party service brand names ---
    ("button", "<text>", "ntfy.sh"),
    ("button", "<text>", "gotify"),
    ("button", "<text>", "Discord"),
    ("button", "<text>", "Slack"),
    ("button", "<text>", "Healthchecks.io"),

    # --- placeholders with technical examples (URL / JSON snippets) -----
    ("input", "placeholder", "https://api.openai.com/v1/chat/completions"),
    ("textarea", "placeholder", '{"Authorization":"Bearer YOUR_TOKEN"}'),
    ("textarea", "placeholder",
     '{"title":"Translation {event}","message":"{file} in {duration_seconds:.0f}s"}'),

    # --- export buttons: file format identifiers ------------------------
    ("button", "<text>", "CSV"),
    ("button", "<text>", "JSON"),

    # --- supported-formats hint (file extensions, separator dots) -------
    ("div", "<text>", "TXT · SRT · EPUB · DOCX · PDF"),
}


def _strip_jinja_preserving_lines(src: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return re.sub(r"[^\n]", " ", match.group(0))
    return _JINJA_RE.sub(repl, src)


def _has_i18n_ancestor(el) -> bool:
    cur = el
    while cur is not None:
        if cur.get("data-i18n") is not None or cur.get("data-i18n-html") is not None:
            return True
        cur = cur.getparent()
    return False


def _is_inside_head(el) -> bool:
    for anc in el.iterancestors():
        if anc.tag == "head":
            return True
    return False


def _is_inside_exempt_option(el) -> bool:
    if el.tag != "option":
        return False
    for anc in el.iterancestors():
        if anc.tag == "select" and anc.get("id") in _EXEMPT_OPTION_SELECT_IDS:
            return True
    return False


def _is_material_icon(el) -> bool:
    classes = (el.get("class") or "").split()
    return "material-symbols-outlined" in classes


def _bound_attrs(el) -> set[str]:
    spec = el.get("data-i18n-attr") or ""
    bound: set[str] = set()
    for pair in spec.split(";"):
        if ":" in pair:
            bound.add(pair.split(":", 1)[0].strip())
    return bound


def _text_is_trivial(text: str) -> bool:
    stripped = text.strip()
    return not stripped or bool(_TRIVIAL_TEXT_RE.match(stripped))


def test_template_has_no_hardcoded_user_visible_text() -> None:
    """All user-visible copy in the main template must be routed through i18n."""
    assert TEMPLATE.is_file(), f"Template not found at {TEMPLATE}"

    src = _strip_jinja_preserving_lines(TEMPLATE.read_text(encoding="utf-8"))
    tree = lhtml.document_fromstring(src)

    offenders: list[str] = []

    for el in tree.iter():
        if not isinstance(el.tag, str):
            continue  # comment / processing instruction
        if el.tag in ("script", "style"):
            continue
        if _is_inside_head(el):
            continue

        # ---- text content directly inside this element ------------------
        text = el.text or ""
        if (
            not _text_is_trivial(text)
            and not _has_i18n_ancestor(el)
            and not _is_material_icon(el)
            and not _is_inside_exempt_option(el)
        ):
            literal = text.strip()
            key = (el.tag, "<text>", literal)
            if key not in _HARDCODE_ALLOWLIST:
                offenders.append(
                    f"  line {el.sourceline}: <{el.tag}> text not bound: {literal!r}"
                )

        # ---- tail text (text node that follows el, owned by its parent) --
        tail = el.tail or ""
        if not _text_is_trivial(tail):
            parent = el.getparent()
            if (
                parent is not None
                and not _is_inside_head(parent)
                and parent.tag not in ("script", "style")
                and not _has_i18n_ancestor(parent)
            ):
                literal = tail.strip()
                key = (parent.tag, "<text>", literal)
                if key not in _HARDCODE_ALLOWLIST:
                    offenders.append(
                        f"  line {el.sourceline}: text after <{el.tag}> "
                        f"(in <{parent.tag}>) not bound: {literal!r}"
                    )

        # ---- user-visible attributes ------------------------------------
        bound = _bound_attrs(el)
        for attr in _USER_VISIBLE_ATTRS:
            val = el.get(attr)
            if val is None:
                continue
            stripped = val.strip()
            if not stripped:
                continue
            if attr in bound:
                continue
            key = (el.tag, attr, stripped)
            if key in _HARDCODE_ALLOWLIST:
                continue
            offenders.append(
                f"  line {el.sourceline}: <{el.tag} {attr}={stripped!r}> "
                f"not bound via data-i18n-attr"
            )

    assert not offenders, (
        "Hardcoded user-visible text found in template — route via data-i18n / "
        "data-i18n-html / data-i18n-attr (or, if truly intentional, add to "
        "_HARDCODE_ALLOWLIST in this test with a comment explaining why):\n"
        + "\n".join(offenders)
    )


# ---------------------------------------------------------------------------
# Hardcoded user-visible text in JS modules
# ---------------------------------------------------------------------------

# DOM properties that surface text to the user. An assignment of a plain
# string literal to any of these must go through t(...) instead.
_JS_TEXT_SINKS = ("textContent", "innerText", "placeholder", "title", "alt", "ariaLabel")

# Attributes set via setAttribute(...) that surface text to the user.
_JS_SETATTR_VISIBLE = ("placeholder", "title", "aria-label", "alt")

# Match `obj.<sink> = '...'` or `obj.<sink> = "..."`. Captures the literal
# body. Template literals (backticks) are intentionally excluded — their
# static parts may still leak English but they trip too many false positives
# under a literal-only scan to be worth flagging here.
_JS_PROP_ASSIGN_RE = re.compile(
    r"\.(?P<sink>" + "|".join(_JS_TEXT_SINKS) + r")"
    r"\s*=\s*"
    r"(?P<q>['\"])(?P<literal>(?:\\.|(?!(?P=q))[^\\\n])*)(?P=q)"
)

# Match `setAttribute('attr', '...')` where attr is user-visible.
_JS_SETATTR_RE = re.compile(
    r"setAttribute\s*\(\s*"
    r"(?P<aq>['\"])(?P<attr>" + "|".join(re.escape(a) for a in _JS_SETATTR_VISIBLE) + r")(?P=aq)"
    r"\s*,\s*"
    r"(?P<vq>['\"])(?P<value>(?:\\.|(?!(?P=vq))[^\\\n])*)(?P=vq)"
    r"\s*\)"
)

# A single lowercase-snake_case token (typical Material Symbols icon name,
# e.g. "expand_more", "progress_activity"). Only treated as trivial when
# assigned to textContent/innerText — never for placeholder/title/alt/etc.
_MATERIAL_ICON_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# JS literals that are intentional and must NOT trigger the test.
# Format: (sink_or_attr, literal-as-it-appears-in-source).
_JS_HARDCODE_ALLOWLIST: set[tuple[str, str]] = {
    # Generic cover-image alt text used on EPUB cover thumbnails. Single
    # English word, accessibility-only — not worth a dedicated key × 7 locales.
    # If full a11y i18n is required later, replace with t('translation:cover_alt').
    ("alt", "Cover"),
}


def _strip_js_comments(src: str) -> str:
    """Blank out // ... and /* ... */ comments while preserving offsets so
    line numbers reported by the scanner stay accurate.

    Naive: doesn't recognize comment markers inside string literals. Safe
    here because any //... or /*...*/ appearing inside a quoted string would
    not match the sink regexes anyway (they require a `.sink =` prefix or a
    `setAttribute(` call outside the string)."""
    def blank(m: re.Match[str]) -> str:
        return re.sub(r"[^\n]", " ", m.group(0))
    src = re.sub(r"/\*.*?\*/", blank, src, flags=re.DOTALL)
    src = re.sub(r"//[^\n]*", blank, src)
    return src


def _unescape_js_literal(s: str) -> str:
    """Minimal unescape — handles the escapes we'd realistically encounter
    in user-facing literals (\\', \\", \\\\, \\n, \\t)."""
    return (
        s.replace("\\\\", "\x00")
         .replace("\\'", "'")
         .replace("\\\"", "\"")
         .replace("\\n", "\n")
         .replace("\\t", "\t")
         .replace("\x00", "\\")
    )


def _is_trivial_js_literal(literal: str, sink: str) -> bool:
    stripped = literal.strip()
    if not stripped:
        return True
    if _TRIVIAL_TEXT_RE.match(stripped):
        return True
    # Material Symbols icon glyph name (snake_case) — only when assigned
    # to a text-content sink. NEVER exempt for placeholder/title/alt etc.,
    # where a snake_case literal would be a real bug.
    if sink in ("textContent", "innerText") and _MATERIAL_ICON_NAME_RE.match(stripped):
        return True
    return False


def _iter_js_files():
    for path in sorted(JS_DIR.rglob("*.js")):
        if "vendor" in path.parts:
            continue
        yield path


def _line_of(src: str, offset: int) -> int:
    return src.count("\n", 0, offset) + 1


def test_js_modules_have_no_hardcoded_user_visible_text() -> None:
    """All user-visible string literals in JS modules must route through t(...)."""
    assert JS_DIR.is_dir(), f"JS directory not found at {JS_DIR}"

    offenders: list[str] = []

    for js_path in _iter_js_files():
        raw = js_path.read_text(encoding="utf-8")
        src = _strip_js_comments(raw)
        rel = js_path.relative_to(ROOT).as_posix()

        for m in _JS_PROP_ASSIGN_RE.finditer(src):
            sink = m.group("sink")
            literal = _unescape_js_literal(m.group("literal"))
            if _is_trivial_js_literal(literal, sink):
                continue
            if (sink, literal) in _JS_HARDCODE_ALLOWLIST:
                continue
            offenders.append(
                f"  {rel}:{_line_of(src, m.start())}  "
                f".{sink} = {literal!r}"
            )

        for m in _JS_SETATTR_RE.finditer(src):
            attr = m.group("attr")
            literal = _unescape_js_literal(m.group("value"))
            if _is_trivial_js_literal(literal, attr):
                continue
            if (attr, literal) in _JS_HARDCODE_ALLOWLIST:
                continue
            offenders.append(
                f"  {rel}:{_line_of(src, m.start())}  "
                f"setAttribute({attr!r}, {literal!r})"
            )

    assert not offenders, (
        "Hardcoded user-visible string literal in JS — route through t(...) "
        "from src/web/static/js/i18n/i18n.js (or, if truly intentional, add "
        "to _JS_HARDCODE_ALLOWLIST in this test with a justification):\n"
        + "\n".join(offenders)
    )
