"""
Source-script typography normalization for translated EPUBs (CJK -> Latin).

An EPUB authored for a CJK script carries its presentation with it: the
stylesheet selects SimSun/KaiTi/Gothic font stacks (sometimes an embedded font
subset that maps a handful of ideographs and no Latin letters at all), a
two-ideograph paragraph indent (`text-indent: 2em`), CJK leading (130-135%),
and, in the general case, vertical writing mode plus CJK line-breaking rules.
Translating the text alone leaves Latin prose rendered through all of it.

This module owns the *declaration-level* fix. Following the design decision in
`plan/PLAN_CjkSourceRendering.md` §2, the offending declarations are **rewritten
in place** rather than out-specified by injected `!important` overrides: no
cascade reasoning is needed, the reader's own font/leading controls stay usable,
and idempotency is free (a rewritten value no longer matches the detector).

The module has two sections:

  * a **pure section** (this file, below) — no filesystem, no lxml, fully
    unit-testable on strings:
      - `normalize_script_language` / `is_cjk_language` — the language gate
      - `collect_font_face_families` / `contains_cjk_font_token` /
        `map_cjk_font_to_generic` — CJK font-stack detection and mapping
      - `NEUTRALIZATIONS` — the closed property table, as data
      - `neutralize_css_text` / `neutralize_style_attribute` — the rewriters
      - `CJK_EVIDENCE` — the closed table of unambiguous CJK signals, as data
      - `has_cjk_typography_features` / `should_normalize_script` — the gate
  * an **apply section** that walks an extracted EPUB directory, added below
    the marker at the end of this file.

**Evidence is narrower than the transform.** Two distinct questions are asked of
a stylesheet, and they use two different tables:

  * *"Is this book CJK-authored?"* — `CJK_EVIDENCE`, consulted only when the
    source language is unknown or mislabelled. It admits only signals with no
    Latin-typography explanation: a CJK/romanized-CJK font family, vertical
    writing mode, `text-orientation`, `text-combine[-upright]`,
    `word-break: break-all|keep-all`, `line-break`, ideographic
    `text-justify`.
  * *"What must be rewritten once we know it is?"* — `NEUTRALIZATIONS`, which is
    strictly larger: it also normalizes `line-height`, `text-indent`, and font
    stacks caught by the `@font-face` suspect list (rule (c) of
    `contains_cjk_font_token`).

The asymmetry is deliberate. A tight leading (`line-height: 1.3`), a two-em
paragraph indent and an embedded font face are all perfectly ordinary in a
Latin-script book, so treating them as evidence would fire the whole pass on a
French EPUB translated to English and silently rewrite its leading, its indents
and its embedded faces. They remain in scope as *transforms* because inside a
book already established as CJK-authored they are exactly the artefacts that
have to go.

This mirrors how `lang_support.py` separates `get_language_code` /
`set_xhtml_lang_attributes` from `apply_target_language_to_xhtml_directory`.

Deliberate non-goals (see the plan §1.4 and its neutralization table):
  - `ruby-position` / `ruby-align` are left untouched; stripping ruby content is
    a separate concern.
  - `punctuation-trim`, `hanging-punctuation`, `text-spacing`, `text-autospace`
    are out of scope: no reader applies them harmfully to Latin text.
  - Only the properties in `NEUTRALIZATIONS` are considered. In particular the
    `font` shorthand (`font: 12px/1.3 "宋体"`) is NOT rewritten — the closed
    table is a decision, not an oversight.
  - Embedded font files are never deleted; only the references from text
    selectors are neutralized, so the manifest stays valid.

No CSS parser is used, and none is needed: the rewriter masks comments, quoted
strings, `url(...)` and whole `@font-face` blocks, then rewrites one declaration
value at a time with a regex. The safety net is a structural invariant asserted
by the tests: braces are never inserted, removed or moved.
"""
from __future__ import annotations

import re
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Tuple

from .lang_support import get_language_code


# ---------------------------------------------------------------------------
# CJK codepoint detection
# ---------------------------------------------------------------------------

# Character class duplicated verbatim from `_CJK_RE` in
# src/core/glossary/filter.py:23, which stays the canonical definition. It is
# copied rather than imported so that `src.core.epub` does not grow a dependency
# on `src.core.glossary` (which pulls in the glossary domain models). The plan
# forbids inventing a *third* character class, not holding a second reference to
# the same one — keep the two literals identical.
_CJK_CSS_RE = re.compile(r'[぀-ゟ゠-ヿ一-鿿가-힯㐀-䶿]')


# ---------------------------------------------------------------------------
# Language gate
# ---------------------------------------------------------------------------

CJK_SCRIPT_CODES: FrozenSet[str] = frozenset({"zh", "ja", "ko"})

# Code-shaped inputs that resolve to a CJK script. `get_language_code` returns
# None for "Chinese (Traditional)", "zh-Hans", "zh-Hant" and "zh-TW" (its
# regional map only holds the two Portuguese variants), hence this extra layer.
#
# The plan (§3.3) phrases the rules as "starts with zh / cmn / yue", etc. Prefix
# matching is deliberately narrowed here to *exact primary subtags* of
# code-shaped input, because a bare prefix test misfires on real language names
# and real ISO codes: "Javanese" and "jav" both start with "ja", "Kongo" and
# "kok" (Konkani) both start with "ko", "zha" (Zhuang) starts with "zh". Every
# tag the plan enumerates ("zh", "zh-CN", "zh-TW", "zh-Hans", "zh-Hant", "cmn",
# "yue", "ja", "jpn", "ko", "kor") is still resolved, because the region/script
# subtag is split off before the comparison.
_ZH_CODES: FrozenSet[str] = frozenset({"zh", "zho", "chi", "cmn", "yue"})
_JA_CODES: FrozenSet[str] = frozenset({"ja", "jpn"})
_KO_CODES: FrozenSet[str] = frozenset({"ko", "kor"})

# Substring markers for human-readable language names, applied to the whole
# lower-cased input ("Chinese (Traditional)", "simplified chinese", ...).
_SCRIPT_NAME_MARKERS: Tuple[Tuple[str, str], ...] = (
    ("chinese", "zh"),
    ("japanese", "ja"),
    ("korean", "ko"),
)

_SUBTAG_SPLIT_RE = re.compile(r'[-_]')


def normalize_script_language(language: Optional[str]) -> Optional[str]:
    """Resolve a language to 'zh' | 'ja' | 'ko', else delegate to lang_support.

    Rules (plan §3.3), in order:
      1. Lowercase, strip.
      2. Split off the region/script subtag; if the primary subtag is
         code-shaped (<= 3 letters) and names a CJK language, return its script
         code. See `_ZH_CODES` for why this is an exact match, not a prefix one.
      3. Otherwise, if the name contains 'chinese' / 'japanese' / 'korean',
         return the matching script code.
      4. Otherwise fall back to `get_language_code`.

    Returns None for unresolvable input.

    Examples:
        normalize_script_language("Chinese (Traditional)") -> "zh"
        normalize_script_language("zh-Hant")               -> "zh"
        normalize_script_language("yue")                   -> "zh"
        normalize_script_language("Japanese")              -> "ja"
        normalize_script_language("Javanese")              -> None
        normalize_script_language("French")                -> "fr"
        normalize_script_language("Klingon")               -> None
    """
    if not language:
        return None

    lowered = language.lower().strip()
    if not lowered:
        return None

    primary = _SUBTAG_SPLIT_RE.split(lowered, maxsplit=1)[0].strip()
    if primary.isalpha() and len(primary) <= 3:
        if primary in _ZH_CODES:
            return "zh"
        if primary in _JA_CODES:
            return "ja"
        if primary in _KO_CODES:
            return "ko"

    for marker, code in _SCRIPT_NAME_MARKERS:
        if marker in lowered:
            return code

    return get_language_code(language)


def is_cjk_language(language: Optional[str]) -> bool:
    """True if `language` resolves to a CJK script code."""
    return normalize_script_language(language) in CJK_SCRIPT_CODES


# ---------------------------------------------------------------------------
# CJK font-stack detection and mapping
# ---------------------------------------------------------------------------

# Romanized family-name fragments that identify a CJK face. Matched as
# substrings of a normalized token, which is the conservative direction: a false
# positive costs an intentional Latin face (replaced by a generic, risk R1 in
# the plan), a false negative leaves Latin prose in CJK glyphs.
#
# 'gothic' is NOT in this set: "Century Gothic" is a Latin face. It is handled
# by `_CJK_GOTHIC_RE` below. 'mono' / 'courier' are not here either — they
# classify a face (see `map_cjk_font_to_generic`) but never identify it as CJK.
_ROMANIZED_CJK_FONTS: FrozenSet[str] = frozenset({
    # sans-serif side
    "heiti", "hei", "yahei", "jhenghei", "malgun", "dotum", "gulim",
    "nanum", "nanum-gothic", "pingfang", "meiryo", "dengxian",
    "noto sans", "source han sans", "hiragino kaku",
    # serif side
    "songti", "simsun", "sung", "ming", "mincho", "batang", "gungsuh",
    "kai", "fangsong", "stsong",
    "noto serif", "source han serif", "hiragino mincho",
})

# 'gothic' counts as CJK only when preceded by one of these vendor prefixes
# (MS Gothic, Yu Gothic, IPAGothic, HGGothic, Malgun Gothic, Apple SD Gothic
# Neo, Nanum Gothic) — or when the token also carries a CJK character.
_CJK_GOTHIC_RE = re.compile(r'(?:ms|yu|ipa|hg|malgun|apple\s*sd|nanum)[\s\-]*gothic')

# Font classes for `map_cjk_font_to_generic`, checked in this order. The first
# class with a matching token wins; the default is 'serif' because book body
# text is serif far more often than not.
_MONOSPACE_MARKERS: Tuple[str, ...] = ("mono", "courier")
_SANS_MARKERS: Tuple[str, ...] = (
    "heiti", "hei", "yahei", "jhenghei", "malgun", "dotum", "gulim",
    "nanum", "nanum-gothic", "pingfang", "meiryo", "dengxian",
    "noto sans", "source han sans", "hiragino kaku",
    "黑体", "ゴシック", "고딕", "돋움", "굴림",
)
_SERIF_MARKERS: Tuple[str, ...] = (
    "songti", "simsun", "sung", "ming", "mincho", "batang", "gungsuh",
    "kai", "fangsong", "stsong",
    "noto serif", "source han serif", "hiragino mincho",
    "宋体", "明朝", "楷", "仿宋", "명조", "바탕",
)

_WHITESPACE_RE = re.compile(r'\s+')


def _split_font_tokens(font_family_value: str) -> List[str]:
    """Split a font-family value into normalized tokens.

    Comma-separated, surrounding quotes removed, lower-cased, inner whitespace
    collapsed to single spaces.
    """
    tokens: List[str] = []
    for raw in (font_family_value or "").split(","):
        token = raw.strip()
        if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
            token = token[1:-1].strip()
        token = _WHITESPACE_RE.sub(" ", token).strip().lower()
        if token:
            tokens.append(token)
    return tokens


_FONT_FACE_BLOCK_RE = re.compile(r'@font-face\s*\{([^{}]*)\}', re.IGNORECASE)
_FONT_FAMILY_DECL_RE = re.compile(r'font-family\s*:\s*([^;}]*)', re.IGNORECASE)


def collect_font_face_families(css_text: str) -> FrozenSet[str]:
    """Family names declared by this stylesheet's @font-face rules.

    Normalized like `_split_font_tokens` (lower-cased, unquoted). Used by
    `contains_cjk_font_token` as a suspect list.
    """
    if not css_text:
        return frozenset()

    # Comments are stripped so a commented-out @font-face contributes nothing.
    stripped = _COMMENT_RE.sub(" ", css_text)

    families: set = set()
    for block in _FONT_FACE_BLOCK_RE.finditer(stripped):
        for decl in _FONT_FAMILY_DECL_RE.finditer(block.group(1)):
            families.update(_split_font_tokens(decl.group(1)))
    return frozenset(families)


def contains_cjk_font_token(font_family_value: str,
                            font_face_families: FrozenSet[str] = frozenset()) -> bool:
    """True if a font-family value references a CJK face.

    A token is CJK when ANY of:
      (a) it contains a CJK character (`_CJK_CSS_RE`);
      (b) it matches `_ROMANIZED_CJK_FONTS` (or the guarded 'gothic' rule);
      (c) it is in `font_face_families`.

    Rationale for (c): a font embedded by a CJK-authored book is in practice a
    CJK face or a CJK subset — the reported book's `zdy2.ttf` maps 15 glyphs and
    no Latin letters at all. Accepted false-positive risk (plan R1): a CJK book
    embedding a Latin display face loses that face and falls back to a generic.
    """
    for token in _split_font_tokens(font_family_value):
        if _CJK_CSS_RE.search(token):
            return True
        if any(marker in token for marker in _ROMANIZED_CJK_FONTS):
            return True
        if "gothic" in token and _CJK_GOTHIC_RE.search(token):
            return True
        if token in font_face_families:
            return True
    return False


def map_cjk_font_to_generic(font_family_value: str) -> str:
    """Return exactly one of 'serif' | 'sans-serif' | 'monospace'.

    The first *class* with a matching token wins (monospace, then sans-serif,
    then serif) — not the first token's class. Default: 'serif'.
    """
    tokens = _split_font_tokens(font_family_value)

    for marker in _MONOSPACE_MARKERS:
        if any(marker in token for token in tokens):
            return "monospace"

    for marker in _SANS_MARKERS:
        if any(marker in token for token in tokens):
            return "sans-serif"
    if any("gothic" in token and _CJK_GOTHIC_RE.search(token) for token in tokens):
        return "sans-serif"

    for marker in _SERIF_MARKERS:
        if any(marker in token for token in tokens):
            return "serif"

    return "serif"


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------
#
# Regions that must never be interpreted as declarations are replaced by opaque
# tokens before rewriting, and restored in reverse order afterwards. Tokens are
# delimited by control characters that cannot appear in CSS text, and their body
# holds a single kind letter plus digits, so a leftover token inside a
# font-family value can never match a family-name marker.
#
# Masking @font-face blocks is REQUIRED: their `font-family` declares the family
# NAME, and rewriting it would break every reference to it (and the manifest's
# intent). Masking quoted strings is what protects `content: "font-family: 宋体"`.

_MASK_START = "\x00"
_MASK_END = "\x01"
_COMMENT_KIND = "c"
_STRING_KIND = "s"
_URL_KIND = "u"
_FONT_FACE_KIND = "f"

_COMMENT_RE = re.compile(r'/\*.*?\*/', re.DOTALL)
_STRING_RE = re.compile(r'"[^"\n]*"|\'[^\'\n]*\'')
_URL_RE = re.compile(r'url\(\s*[^)]*\)', re.IGNORECASE)
_FONT_FACE_RE = re.compile(r'@font-face\s*\{[^{}]*\}', re.IGNORECASE)

_COMMENT_TOKEN_PATTERN = _MASK_START + _COMMENT_KIND + r'\d+' + _MASK_END
_STRING_TOKEN_RE = re.compile(_MASK_START + _STRING_KIND + r'\d+' + _MASK_END)

# Masking order matters: comments first (so strings inside comments are already
# hidden), then strings, then url() (so url("x.ttf") masks cleanly), then whole
# @font-face blocks (whose body is brace-free by then).
_MASK_STEPS: Tuple[Tuple[re.Pattern, str], ...] = (
    (_COMMENT_RE, _COMMENT_KIND),
    (_STRING_RE, _STRING_KIND),
    (_URL_RE, _URL_KIND),
    (_FONT_FACE_RE, _FONT_FACE_KIND),
)


def _mask(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Replace protected regions with opaque tokens.

    Returns (masked_text, masks) where `masks` is in masking order; unmasking
    walks it in reverse so nested regions are restored correctly.
    """
    masks: List[Tuple[str, str]] = []

    def _replace(match: re.Match) -> str:
        token = f"{_MASK_START}{kind}{len(masks)}{_MASK_END}"
        masks.append((token, match.group(0)))
        return token

    for pattern, kind in _MASK_STEPS:
        text = pattern.sub(_replace, text)
    return text, masks


def _unmask(text: str, masks: List[Tuple[str, str]]) -> str:
    for token, original in reversed(masks):
        text = text.replace(token, original)
    return text


# ---------------------------------------------------------------------------
# Value transforms
# ---------------------------------------------------------------------------
#
# Editorial thresholds and replacement values live here as constants (plan R3:
# changing one must be a one-line edit).

MAX_TEXT_INDENT_EM = 1.5           # em/rem/ch indents above this are excessive
TEXT_INDENT_EM_REPLACEMENT = "1.5em"
MAX_TEXT_INDENT_PERCENT = 150.0    # percentage indents above this are excessive
TEXT_INDENT_PERCENT_REPLACEMENT = "150%"
MIN_LINE_HEIGHT = 1.4              # leading below this ratio is cramped in Latin
LINE_HEIGHT_REPLACEMENT = "1.5"

_RELATIVE_INDENT_RE = re.compile(r'^([+-]?(?:\d+\.?\d*|\.\d+))(em|rem|ch|%)$', re.IGNORECASE)
_LINE_HEIGHT_RE = re.compile(r'^([+-]?(?:\d+\.?\d*|\.\d+))(em|rem|%)?$', re.IGNORECASE)

_VERTICAL_WRITING_MODES = ("vertical",)
_CJK_WORD_BREAKS = frozenset({"break-all", "keep-all"})
_CJK_TEXT_JUSTIFY = frozenset({"inter-ideograph", "inter-character", "distribute"})


def _transform_font_family(value: str, font_face_families: FrozenSet[str]) -> str:
    if not contains_cjk_font_token(value, font_face_families):
        return value
    return map_cjk_font_to_generic(value)


def _transform_text_indent(value: str, font_face_families: FrozenSet[str]) -> str:
    match = _RELATIVE_INDENT_RE.match(value.strip())
    if not match:
        # Absolute units (px, pt, mm), keywords and multi-value forms express a
        # measured intent and are left alone.
        return value
    amount = float(match.group(1))
    if match.group(2) == "%":
        return TEXT_INDENT_PERCENT_REPLACEMENT if amount > MAX_TEXT_INDENT_PERCENT else value
    return TEXT_INDENT_EM_REPLACEMENT if amount > MAX_TEXT_INDENT_EM else value


def _transform_line_height(value: str, font_face_families: FrozenSet[str]) -> str:
    match = _LINE_HEIGHT_RE.match(value.strip())
    if not match:
        # 'normal', 'inherit' and absolute units (px, pt) are skipped.
        return value
    amount = float(match.group(1))
    unit = (match.group(2) or "").lower()
    ratio = amount / 100.0 if unit == "%" else amount
    return LINE_HEIGHT_REPLACEMENT if ratio < MIN_LINE_HEIGHT else value


def _transform_writing_mode(value: str, font_face_families: FrozenSet[str]) -> str:
    lowered = value.lower()
    if any(marker in lowered for marker in _VERTICAL_WRITING_MODES):
        return "horizontal-tb"
    return value


def _transform_text_orientation(value: str, font_face_families: FrozenSet[str]) -> str:
    return "mixed"


def _transform_text_combine(value: str, font_face_families: FrozenSet[str]) -> str:
    return "none"


def _transform_word_break(value: str, font_face_families: FrozenSet[str]) -> str:
    if value.strip().lower() in _CJK_WORD_BREAKS:
        return "normal"
    return value


def _transform_line_break(value: str, font_face_families: FrozenSet[str]) -> str:
    return "auto"


def _transform_text_justify(value: str, font_face_families: FrozenSet[str]) -> str:
    if value.strip().lower() in _CJK_TEXT_JUSTIFY:
        return "auto"
    return value


# The closed neutralization table (plan, Phase 3). Adding or removing an entry
# is a design decision; the "why" of each is recorded inline. Transforms return
# the value unchanged when the trigger does not fire — the rewriter substitutes
# only on a real change, which is what makes the whole pass idempotent.
NEUTRALIZATIONS: Tuple[Tuple[str, Callable[[str, FrozenSet[str]], str]], ...] = (
    # Latin text rendered through SimSun/KaiTi glyphs, or through a 15-glyph
    # embedded subset. A generic family lets the reader pick a real Latin face.
    # The whole stack is replaced; a stack with no CJK token is left untouched.
    ("font-family", _transform_font_family),
    # 2em is the CJK two-ideograph indent; 1.5em is the closest non-excessive
    # Latin equivalent.
    ("text-indent", _transform_text_indent),
    # CJK glyphs are square and need less leading; 130-135% is cramped for Latin
    # ascenders/descenders.
    ("line-height", _transform_line_height),
    # Vertical Latin prose is unreadable.
    ("writing-mode", _transform_writing_mode),
    # Only meaningful alongside vertical writing mode.
    ("text-orientation", _transform_text_orientation),
    # Tate-chu-yoko has no Latin meaning.
    ("text-combine-upright", _transform_text_combine),
    ("text-combine", _transform_text_combine),
    # break-all breaks French words mid-word with no hyphen; keep-all (Korean)
    # prevents breaking entirely, producing overflow.
    ("word-break", _transform_word_break),
    # CJK kinsoku rules.
    ("line-break", _transform_line_break),
    # Ideograph-metric justification on Latin text produces rivers.
    ("text-justify", _transform_text_justify),
)


# ---------------------------------------------------------------------------
# Declaration rewriting
# ---------------------------------------------------------------------------

# Vendor prefixes carrying the same semantics as the unprefixed property.
_VENDOR_PREFIXES = r'(?:-epub-|-webkit-)?'

# A declaration may be separated from the previous ';' or the opening '{' by
# whitespace *and by comments* — the reported book's stylesheet puts a comment
# before nearly every declaration. Comments are masked by then, so the
# separator class accepts comment tokens as whitespace.
_DECLARATION_SEPARATOR = r'(?:\s|' + _COMMENT_TOKEN_PATTERN + r')*'

_TRAILING_WS_RE = re.compile(r'^(.*?)(\s*)$', re.DOTALL)
_IMPORTANT_RE = re.compile(r'(\s*!\s*important)$', re.IGNORECASE)


def _declaration_pattern(prop: str) -> re.Pattern:
    """Compile the rewriter regex for one property.

    Groups: 1 = the ';'/'{' (or start of string), 2 = separator, 3 = property
    name as written, 4 = ':' with its surrounding whitespace, 5 = raw value.
    Masking guarantees group 5 cannot contain an unmasked ';' or '}'.
    """
    return re.compile(
        r'(^|[;{])(' + _DECLARATION_SEPARATOR + r')('
        + _VENDOR_PREFIXES + re.escape(prop) + r')(\s*:\s*)([^;}]*)',
        re.IGNORECASE,
    )


_DECLARATION_PATTERNS: Dict[str, re.Pattern] = {
    prop: _declaration_pattern(prop) for prop, _ in NEUTRALIZATIONS
}


def _split_declaration_value(raw_value: str,
                             mask_lookup: Dict[str, str]) -> Tuple[str, str, str]:
    """Split a matched declaration value into (logical, important, trailing_ws).

    `logical` is the value a transform (or an evidence predicate) reasons about:
    the `!important` flag and the trailing whitespace removed, and the string
    masks expanded back to their content. Expanding strings is required because
    a masked `"宋体"` would hide the very thing `contains_cjk_font_token` looks
    for. Only string masks are expanded — comments, url() and @font-face stay
    masked, so the property-matching regex still cannot see inside them (this is
    what keeps `content: "font-family: 宋体"` safe: the `content` declaration is
    not in the table at all, so its string never becomes matchable text).

    Shared by the rewriter and the evidence scanner so both read a declaration
    the same way.
    """
    core, trailing_ws = _TRAILING_WS_RE.match(raw_value).groups()
    important_match = _IMPORTANT_RE.search(core)
    important = ""
    if important_match:
        important = important_match.group(1)
        core = core[: important_match.start(1)]
    logical = _STRING_TOKEN_RE.sub(
        lambda m: mask_lookup.get(m.group(0), m.group(0)), core)
    return logical, important, trailing_ws


def _neutralize_declarations(text: str,
                             font_face_families: FrozenSet[str]) -> Tuple[str, Dict[str, int]]:
    """Apply every entry of NEUTRALIZATIONS to a CSS text or declaration list.

    Counts are keyed by the *unprefixed* property name, so a rewritten
    `-webkit-writing-mode` is counted under 'writing-mode'.
    """
    counts: Dict[str, int] = {}
    if not text:
        return text, counts

    masked, masks = _mask(text)
    mask_lookup = {token: original for token, original in masks}

    for prop, transform in NEUTRALIZATIONS:
        pattern = _DECLARATION_PATTERNS[prop]

        def _replace(match: re.Match, prop=prop, transform=transform) -> str:
            logical, important, trailing_ws = _split_declaration_value(
                match.group(5), mask_lookup)
            if not logical.strip():
                return match.group(0)

            rewritten = transform(logical, font_face_families)
            if rewritten == logical:
                return match.group(0)

            counts[prop] = counts.get(prop, 0) + 1
            return (match.group(1) + match.group(2) + match.group(3)
                    + match.group(4) + rewritten + important + trailing_ws)

        masked = pattern.sub(_replace, masked)

    return _unmask(masked, masks), counts


def neutralize_css_text(css_text: str) -> Tuple[str, Dict[str, int]]:
    """Rewrite offending declaration values in a stylesheet, in place.

    Returns (rewritten_css, counts_by_property).

    Procedure (plan Phase 3, exact order):
      1. Mask comments, quoted strings, url(...) and whole @font-face blocks.
      2. For each (property, transform) in NEUTRALIZATIONS, rewrite the value
         when — and only when — the transform returns something different.
      3. Unmask in reverse order.

    Braces are never inserted, removed or moved: the output has the same brace
    structure as the input. The transformation is idempotent by construction
    ('serif' has no CJK token, '1.5em' is not > 1.5, '1.5' is not < 1.4,
    'horizontal-tb' contains no 'vertical').
    """
    if not css_text:
        return css_text, {}
    return _neutralize_declarations(css_text, collect_font_face_families(css_text))


def neutralize_style_attribute(style_value: str) -> Tuple[str, Dict[str, int]]:
    """Same transforms applied to an inline `style` attribute value.

    A bare declaration list: no selectors, no @-rules, hence no @font-face
    family names to protect and no suspect list to build.
    """
    if not style_value:
        return style_value, {}
    return _neutralize_declarations(style_value, frozenset())


# ---------------------------------------------------------------------------
# Content-based evidence of a CJK source
# ---------------------------------------------------------------------------
#
# The closed table of *unambiguous* CJK signals, deliberately narrower than
# NEUTRALIZATIONS (see the module docstring). Every predicate here answers "does
# this declaration have no plausible Latin-typography explanation?".
#
# Excluded on purpose: `line-height` and `text-indent`. Cramped leading and a
# two-em indent are ordinary Latin book typography; using them as evidence made
# `should_normalize_script('French', 'English', …)` return True on a plain
# Georgia/serif stylesheet.


def _evidence_font_family(value: str) -> bool:
    # Rules (a) and (b) of `contains_cjk_font_token` only. Rule (c) — any family
    # declared by an @font-face — is NOT evidence: embedding a font is normal in
    # a Latin book. It stays in force as a transform inside a gated CJK book.
    return contains_cjk_font_token(value, frozenset())


def _evidence_vertical_writing_mode(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in _VERTICAL_WRITING_MODES)


def _evidence_any_value(value: str) -> bool:
    # The property itself only exists to serve CJK typography, so any value is
    # evidence.
    return True


def _evidence_word_break(value: str) -> bool:
    return value.strip().lower() in _CJK_WORD_BREAKS


def _evidence_text_justify(value: str) -> bool:
    return value.strip().lower() in _CJK_TEXT_JUSTIFY


CJK_EVIDENCE: Tuple[Tuple[str, Callable[[str], bool]], ...] = (
    ("font-family", _evidence_font_family),
    ("writing-mode", _evidence_vertical_writing_mode),
    ("text-orientation", _evidence_any_value),
    ("text-combine-upright", _evidence_any_value),
    ("text-combine", _evidence_any_value),
    ("word-break", _evidence_word_break),
    ("line-break", _evidence_any_value),
    ("text-justify", _evidence_text_justify),
)


def has_cjk_typography_features(css_text: str) -> bool:
    """True if this stylesheet carries an unambiguous CJK typography signal.

    The content-based half of the gate (§3.3), for books whose source language is
    unknown or mislabelled. It is intentionally NOT "would `neutralize_css_text`
    change anything": the transform table also normalizes `line-height`,
    `text-indent` and `@font-face`-declared families, none of which is evidence
    of anything — they are ordinary in Latin-script books, and treating them as
    evidence turned the pass on for e.g. a French EPUB translated to English.

    Only `CJK_EVIDENCE` counts here. Once the gate has said yes, the
    neutralization itself is unchanged and applies in full: a genuinely
    CJK-authored book still gets its leading, its indents and its embedded faces
    normalized.

    Reuses the rewriter's masking and declaration patterns, so comments, quoted
    strings, `url(...)` and `@font-face` blocks are hidden from the scan exactly
    as they are from the rewrite.
    """
    if not css_text:
        return False

    masked, masks = _mask(css_text)
    mask_lookup = {token: original for token, original in masks}

    for prop, is_evidence in CJK_EVIDENCE:
        for match in _DECLARATION_PATTERNS[prop].finditer(masked):
            logical, _important, _trailing_ws = _split_declaration_value(
                match.group(5), mask_lookup)
            if logical.strip() and is_evidence(logical):
                return True
    return False


def should_normalize_script(source_language: Optional[str],
                            target_language: Optional[str],
                            css_texts: Iterable[str]) -> bool:
    """Decide whether the normalization pass applies.

    True iff the target is not a CJK script AND (the source is a CJK script OR
    at least one stylesheet carries CJK typography features).

    A target that cannot be resolved at all (None) does NOT disable the pass —
    only an explicitly CJK target does.
    """
    if normalize_script_language(target_language) in CJK_SCRIPT_CODES:
        return False
    if is_cjk_language(source_language):
        return True
    return any(has_cjk_typography_features(text) for text in css_texts)


# ---------------------------------------------------------------------------
# Apply section (filesystem / lxml) — added by a later phase below this line.
# Everything above must stay pure and string-only: no os, no open(), no lxml.
# ---------------------------------------------------------------------------

import codecs
import os

from lxml import etree

from .archive import opf_from_container
from .rtl_support import is_rtl_language


# ---------------------------------------------------------------------------
# Stylesheet I/O
# ---------------------------------------------------------------------------
#
# CJK-authored EPUBs are not reliably UTF-8. The reported book happens to be, but
# gb18030/Big5/Shift_JIS/EUC-KR stylesheets are common enough that reading them
# as UTF-8 would raise UnicodeDecodeError and abort the whole pass.

# Matched against the raw leading bytes (a charset name is ASCII by definition).
_CHARSET_BYTES_RE = re.compile(rb'^@charset\s+"([^"\n]{1,64})"\s*;')
# Same declaration, on decoded text, for the rewrite performed by the
# encode-failure fallback.
_CHARSET_TEXT_RE = re.compile(r'^@charset\s+"[^"\n]*"\s*;')

# Tried in order once no BOM and no @charset settled the question. UTF-8 first:
# it is self-validating, so a successful UTF-8 decode is a conclusion, not a
# guess. The CJK codecs that follow decode almost any byte sequence, so they are
# guesses and are reported through 'encoding_fallbacks'.
_CSS_GUESS_ENCODINGS: Tuple[str, ...] = (
    "utf-8",
    "gb18030",     # Simplified Chinese
    "big5",        # Traditional Chinese
    "shift_jis",   # Japanese
    "euc-kr",      # Korean
)

_BOM_ENCODINGS: Tuple[Tuple[bytes, str], ...] = (
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)


def _declared_css_encodings(raw: bytes) -> List[str]:
    """The encodings a stylesheet declares about itself: BOM, then @charset."""
    declared: List[str] = []
    for bom, encoding in _BOM_ENCODINGS:
        if raw.startswith(bom):
            declared.append(encoding)
            break
    match = _CHARSET_BYTES_RE.match(raw)
    if match:
        try:
            declared.append(match.group(1).decode("ascii").strip())
        except UnicodeDecodeError:
            pass
    return declared


def _decode_css_bytes(raw: bytes) -> Tuple[str, str, bool]:
    """Decode stylesheet bytes. Returns (text, encoding_name, guessed).

    `guessed` is True when neither the BOM, the @charset declaration nor a clean
    UTF-8 decode settled it — i.e. when the encoding was inferred by trying CJK
    codecs, or when the last-resort lossy decode was used. The caller reports it
    as an encoding fallback.
    """
    for encoding in _declared_css_encodings(raw):
        try:
            return raw.decode(encoding), encoding, False
        except (UnicodeDecodeError, LookupError):
            continue

    for encoding in _CSS_GUESS_ENCODINGS:
        try:
            return raw.decode(encoding), encoding, encoding != "utf-8"
        except UnicodeDecodeError:
            continue

    # Nothing decoded cleanly. Never fail the pass over a stylesheet.
    return raw.decode("utf-8", errors="replace"), "utf-8", True


def _read_css_text(path: str) -> Tuple[str, str, bool]:
    """`read_css_text` with its diagnostic. Returns (text, encoding, guessed)."""
    with open(path, "rb") as handle:
        raw = handle.read()
    return _decode_css_bytes(raw)


def read_css_text(path: str) -> Tuple[str, str]:
    """Read a stylesheet and return (text, encoding_name).

    Decoding order, first success wins:
      1. UTF-8/UTF-16 BOM if present
      2. the charset named by a leading `@charset "X";`
      3. utf-8
      4. gb18030    (Simplified Chinese)
      5. big5       (Traditional Chinese)
      6. shift_jis  (Japanese)
      7. euc-kr     (Korean)
      8. utf-8 with errors='replace'

    The returned encoding name is what `write_css_text` should be handed back,
    so a stylesheet is rewritten in the encoding it arrived in.
    """
    text, encoding, _guessed = _read_css_text(path)
    return text, encoding


def _write_css_text(path: str, text: str, encoding: str) -> bool:
    """`write_css_text` with its diagnostic. Returns True if it fell back to UTF-8."""
    try:
        data = text.encode(encoding)
        fell_back = False
    except (UnicodeEncodeError, LookupError):
        # Should not happen — every substitution this module makes is ASCII-only,
        # so anything that decoded also re-encodes. Kept as a safety net for
        # stylesheets whose declared charset cannot represent their own content.
        data = _CHARSET_TEXT_RE.sub('@charset "utf-8";', text, count=1).encode("utf-8")
        fell_back = True

    with open(path, "wb") as handle:
        handle.write(data)
    return fell_back


def write_css_text(path: str, text: str, encoding: str) -> None:
    """Write a stylesheet back in the encoding it was read in.

    Minimal change: every substitution this module makes is ASCII-only, so the
    original encoding always encodes. On `UnicodeEncodeError` (or an unknown
    codec name), the file is written as UTF-8 and a leading `@charset` line is
    rewritten to `"utf-8"` so the stylesheet stays self-describing.
    """
    _write_css_text(path, text, encoding)


# ---------------------------------------------------------------------------
# Container walk
# ---------------------------------------------------------------------------

_XHTML_SUFFIXES: Tuple[str, ...] = (".xhtml", ".html", ".htm")
_FONT_SUFFIXES: Tuple[str, ...] = (".ttf", ".otf", ".woff", ".woff2")

# Reader-specific metadata that overrides the body font globally, which no
# stylesheet edit can defeat. The reported book carries
# `<meta name="duokan-body-font" content="DK-SONGTI"/>`.
# The set is CLOSED on purpose: `duokan-*` also names harmless hints (page
# templates, gallery markers) that must survive.
OPF_FONT_OVERRIDE_METAS: FrozenSet[str] = frozenset({
    "duokan-body-font",
    "duokan-title-font",
    "duokan-font-family",
})

_XML_LANG_ATTR = "{http://www.w3.org/XML/1998/namespace}lang"


def _local_name(tag) -> str:
    """Local name of an lxml tag, namespace and comment/PI safe."""
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _iter_files(directory: str, suffixes: Tuple[str, ...]) -> List[str]:
    """Every file under `directory` whose lower-cased name ends with a suffix."""
    found: List[str] = []
    for root_dir, _dirs, files in os.walk(directory):
        for name in files:
            if name.lower().endswith(suffixes):
                found.append(os.path.join(root_dir, name))
    return sorted(found)


def _find_opf_path(temp_dir: str) -> Optional[str]:
    """The package `META-INF/container.xml` names, else the first `*.opf`
    found while walking `temp_dir`.

    The container lookup is the one the translator and the RTL pass use
    (`archive.find_opf_file`), so every pass edits the same package; the walk
    fallback keeps this module's case-insensitive suffix match.
    """
    from_container = opf_from_container(temp_dir)
    if from_container:
        return from_container
    for root_dir, _dirs, files in os.walk(temp_dir):
        for name in files:
            if name.lower().endswith(".opf"):
                return os.path.join(root_dir, name)
    return None


def _merge_counts(total: Dict[str, int], counts: Dict[str, int]) -> None:
    for prop, count in counts.items():
        total[prop] = total.get(prop, 0) + count


def _log(log_callback: Optional[Callable], event: str, message: str) -> None:
    if log_callback:
        log_callback(event, message)


def _parse_markup_tree(path: str) -> Optional[etree._ElementTree]:
    """Parse an XHTML/HTML file, XML first, HTML only as a fallback.

    XML parsing keeps XHTML exactly as authored (self-closing tags, namespace
    prefixes such as `xlink:`, the DOCTYPE). `etree.HTMLParser` is tried only
    when XML parsing yields no root at all — never as the primary path, and the
    result is still serialized as XML (see `_normalize_markup_file`).

    The HTML fallback is pinned to UTF-8: EPUB content documents must be UTF-8
    or UTF-16, and a file broken enough to reach this branch has usually lost
    its declaration, in which case lxml would otherwise assume Latin-1 and turn
    every ideograph into mojibake before we ever look at it.
    """
    try:
        tree = etree.parse(
            path,
            etree.XMLParser(recover=True, remove_blank_text=False, huge_tree=True),
        )
        if tree is not None and tree.getroot() is not None:
            return tree
    except etree.XMLSyntaxError:
        pass

    tree = etree.parse(
        path, etree.HTMLParser(recover=True, huge_tree=True, encoding="utf-8"))
    if tree is not None and tree.getroot() is not None:
        return tree
    return None


def _collect_style_texts(paths: Iterable[str],
                         log_callback: Optional[Callable]) -> Tuple[List[str], int]:
    """Text of every <style> element in the given markup files, plus an error count.

    Gate input only (step 0), which is why the errors are returned rather than
    folded into the result: a book the gate rejects reports nothing at all. The
    trees are deliberately discarded rather than cached — a book has hundreds of
    documents, and re-parsing the handful that actually get rewritten is cheaper
    than holding every tree in memory.
    """
    texts: List[str] = []
    errors = 0
    for path in paths:
        try:
            tree = _parse_markup_tree(path)
            if tree is None:
                continue
            for element in tree.getroot().iter():
                if _local_name(element.tag) == "style" and element.text:
                    texts.append(element.text)
        except Exception as exc:
            errors += 1
            _log(log_callback, "epub_script_norm_error",
                 f"Could not read inline styles from {path}: {exc}")
    return texts, errors


def _normalize_markup_file(path: str, result: dict,
                           log_callback: Optional[Callable]) -> None:
    """Neutralize every <style> element and every style attribute in one file.

    Serialization is XML, never `method='html'`: HTML serialization expands
    XHTML self-closing tags into `<br></br>`, drops the XML declaration and
    mangles namespace prefixes. `rtl_support` used to make exactly that
    mistake; the plan (§2) forbids repeating it.

    The file is written only when something changed, so untouched documents keep
    their exact original bytes.
    """
    try:
        tree = _parse_markup_tree(path)
        if tree is None:
            result["errors"] += 1
            _log(log_callback, "epub_script_norm_error",
                 f"Could not parse {path}: no document root")
            return

        changed = False
        for element in tree.getroot().iter():
            if _local_name(element.tag) == "style" and element.text:
                new_text, counts = neutralize_css_text(element.text)
                if counts:
                    element.text = new_text
                    _merge_counts(result["changes_by_property"], counts)
                    result["style_elements_rewritten"] += 1
                    changed = True

            style_value = element.get("style") if isinstance(element.tag, str) else None
            if style_value:
                new_value, counts = neutralize_style_attribute(style_value)
                if counts:
                    element.set("style", new_value)
                    _merge_counts(result["changes_by_property"], counts)
                    result["style_attributes_rewritten"] += 1
                    changed = True

        if not changed:
            return

        with open(path, "wb") as handle:
            handle.write(etree.tostring(tree, encoding="utf-8", xml_declaration=True))
    except Exception as exc:
        result["errors"] += 1
        _log(log_callback, "epub_script_norm_error",
             f"Could not normalize inline styles in {path}: {exc}")


def _normalize_opf(opf_path: str, target_language: str, result: dict,
                   log_callback: Optional[Callable]) -> None:
    """Drop reader-specific font overrides and reset RTL page progression.

    Writes the OPF itself: this pass runs after `_update_epub_metadata`, which
    performs the pipeline's single OPF write, so its changes would otherwise be
    lost.
    """
    try:
        tree = etree.parse(
            opf_path,
            etree.XMLParser(recover=True, remove_blank_text=False, huge_tree=True),
        )
        root = tree.getroot()
        if root is None:
            result["errors"] += 1
            _log(log_callback, "epub_script_norm_error",
                 f"Could not parse OPF {opf_path}: no document root")
            return

        changed = False

        for element in list(root.iter()):
            if _local_name(element.tag) != "meta":
                continue
            if (element.get("name") or "") not in OPF_FONT_OVERRIDE_METAS:
                continue
            parent = element.getparent()
            if parent is None:
                continue
            # The element's tail (the newline + indentation that followed it)
            # goes with it; the preceding sibling's own tail keeps the block
            # indented, so the only trace is one blank line. Cosmetic, and
            # cheaper than splicing tails around.
            parent.remove(element)
            result["opf_metas_removed"] += 1
            changed = True

        # F4: a vertical CJK book sets rtl progression, and
        # `apply_rtl_to_epub_directory`'s Case 4 (both LTR) returns early
        # without resetting it, so right-to-left page turns survive into a
        # left-to-right translation.
        if not is_rtl_language(target_language):
            for element in root.iter():
                if _local_name(element.tag) != "spine":
                    continue
                if (element.get("page-progression-direction") or "").lower() == "rtl":
                    element.set("page-progression-direction", "ltr")
                    result["progression_direction_reset"] = True
                    changed = True
                break

        if changed:
            tree.write(opf_path, encoding="utf-8", xml_declaration=True,
                       pretty_print=True)
    except Exception as exc:
        result["errors"] += 1
        _log(log_callback, "epub_script_norm_error",
             f"Could not normalize OPF {opf_path}: {exc}")


def _normalize_ncx(ncx_path: str, lang_code: str, result: dict,
                   log_callback: Optional[Callable]) -> None:
    """Point the NCX root's xml:lang at the target language.

    The reported book's NCX root carries `xml:lang="zh"`. `docTitle/text` is
    left alone: translating it needs the translated title (Phase 6).
    """
    try:
        tree = etree.parse(
            ncx_path,
            etree.XMLParser(recover=True, remove_blank_text=False, huge_tree=True),
        )
        root = tree.getroot()
        if root is None:
            result["errors"] += 1
            _log(log_callback, "epub_script_norm_error",
                 f"Could not parse NCX {ncx_path}: no document root")
            return
        if root.get(_XML_LANG_ATTR) == lang_code:
            return
        root.set(_XML_LANG_ATTR, lang_code)
        tree.write(ncx_path, encoding="utf-8", xml_declaration=True,
                   pretty_print=True)
        result["ncx_lang_updated"] += 1
    except Exception as exc:
        result["errors"] += 1
        _log(log_callback, "epub_script_norm_error",
             f"Could not update NCX language in {ncx_path}: {exc}")


def apply_script_normalization_to_epub_directory(
    temp_dir: str,
    source_language: Optional[str],
    target_language: str,
    log_callback: Optional[Callable] = None,
) -> dict:
    """Neutralize source-script typography across an extracted EPUB directory.

    Applies the pure section's transforms to every style carrier (stylesheets,
    `<style>` elements, inline `style` attributes) and fixes the two structural
    CJK packaging artefacts that no stylesheet edit can reach: reader-specific
    font-override metas in the OPF, and an rtl `page-progression-direction`.

    Steps, in order:
      0. Gate. Collect every stylesheet plus every `<style>` element's text and
         ask `should_normalize_script`. If it says no, NOTHING is touched.
      1. Every `*.css`: `neutralize_css_text`, written back only if changed.
      2. Every `*.xhtml`/`*.html`/`*.htm`: `<style>` elements and `style`
         attributes, re-serialized as XML, written back only if changed.
      3. The OPF: drop `OPF_FONT_OVERRIDE_METAS`, reset rtl progression.
      4. Every `*.ncx` beside the OPF: `xml:lang` -> the target language.
      5. Report the total size of the embedded font files (never deleted — the
         manifest must stay valid; they are simply no longer referenced).

    Returns:
        {'applied': bool,                  # False when the gate says no
         'css_files_rewritten': int,
         'style_elements_rewritten': int,
         'style_attributes_rewritten': int,
         'changes_by_property': Dict[str, int],
         'opf_metas_removed': int,
         'progression_direction_reset': bool,
         'ncx_lang_updated': int,
         'embedded_font_bytes': int,       # reported, never deleted
         'encoding_fallbacks': int,        # stylesheets whose encoding was guessed
         'errors': int}

    Failure policy: every per-file operation is individually guarded. This pass
    is cosmetic relative to the translated text and must NEVER fail a job, so a
    failure increments `errors`, is logged, and the walk continues.
    """
    result = {
        "applied": False,
        "css_files_rewritten": 0,
        "style_elements_rewritten": 0,
        "style_attributes_rewritten": 0,
        "changes_by_property": {},
        "opf_metas_removed": 0,
        "progression_direction_reset": False,
        "ncx_lang_updated": 0,
        "embedded_font_bytes": 0,
        "encoding_fallbacks": 0,
        "errors": 0,
    }

    css_paths = _iter_files(temp_dir, (".css",))
    markup_paths = _iter_files(temp_dir, _XHTML_SUFFIXES)

    # --- Step 0: gate -----------------------------------------------------
    stylesheets: List[Tuple[str, str, str]] = []   # (path, text, encoding)
    encoding_fallbacks = 0
    gate_errors = 0
    for path in css_paths:
        try:
            text, encoding, guessed = _read_css_text(path)
            stylesheets.append((path, text, encoding))
            if guessed:
                encoding_fallbacks += 1
        except Exception as exc:
            gate_errors += 1
            _log(log_callback, "epub_script_norm_error",
                 f"Could not read stylesheet {path}: {exc}")

    gate_texts = [text for _path, text, _encoding in stylesheets]
    style_texts, style_errors = _collect_style_texts(markup_paths, log_callback)
    gate_texts.extend(style_texts)

    if not should_normalize_script(source_language, target_language, gate_texts):
        # Touch nothing, report nothing: the contract is `applied: False` plus
        # zeros, and diagnostics about a book this pass does not apply to would
        # only be noise.
        return result

    result["applied"] = True
    result["encoding_fallbacks"] = encoding_fallbacks
    result["errors"] = gate_errors + style_errors

    # --- Step 1: stylesheets ---------------------------------------------
    for path, text, encoding in stylesheets:
        try:
            new_text, counts = neutralize_css_text(text)
            if not counts:
                continue
            if _write_css_text(path, new_text, encoding):
                result["encoding_fallbacks"] += 1
                _log(log_callback, "epub_script_norm_encoding",
                     f"Stylesheet {path} could not be re-encoded as "
                     f"'{encoding}'; written as UTF-8 instead")
            _merge_counts(result["changes_by_property"], counts)
            result["css_files_rewritten"] += 1
        except Exception as exc:
            result["errors"] += 1
            _log(log_callback, "epub_script_norm_error",
                 f"Could not normalize stylesheet {path}: {exc}")

    # --- Step 2: markup --------------------------------------------------
    for path in markup_paths:
        _normalize_markup_file(path, result, log_callback)

    # --- Steps 3 and 4: OPF, then the NCX files beside it -----------------
    opf_path = _find_opf_path(temp_dir)
    if opf_path:
        _normalize_opf(opf_path, target_language, result, log_callback)

    lang_code = get_language_code(target_language)
    if lang_code:
        # "Beside the OPF" — the content root, where
        # `toc_labels.update_ncx_toc_labels_from_translated_docs` looks too (its tree is
        # walked rather than globbed, so a nested toc/ folder is covered).
        # Without an OPF at all, fall back to the whole container.
        ncx_root = os.path.dirname(opf_path) if opf_path else temp_dir
        for ncx_path in _iter_files(ncx_root, (".ncx",)):
            _normalize_ncx(ncx_path, lang_code, result, log_callback)

    # --- Step 5: embedded fonts, reported only ---------------------------
    for path in _iter_files(temp_dir, _FONT_SUFFIXES):
        try:
            result["embedded_font_bytes"] += os.path.getsize(path)
        except OSError:
            result["errors"] += 1

    return result
