"""
Literal emphasis markers for Plain Text Mode (issue #352).

Plain Text Mode sends each paragraph to the LLM as flat text. When the
`plain_text_keep_emphasis` option is on, the extractor keeps <i>/<em> spans as
literal markers (`<i>word</i>`, `<em>word</em>`) inside that text, and the
rebuild turns them back into real elements. Everything here is about those
markers: detecting them, validating what the LLM sent back, and rebuilding.

The markers are deliberately minimal: a lowercase-or-uppercase tag name with no
attributes, never nested. Anything else is not a marker and stays plain text.
"""
import re
from typing import List, Optional, Tuple

from lxml import etree


EMPHASIS_TAGS = ("i", "em")

# One opening or closing marker. Group 1 is "/" for a closer, group 2 the tag.
_MARKER_RE = re.compile(r"<(/?)(i|em)>", re.IGNORECASE)


def has_literal_markers(text: str) -> bool:
    """True when the text contains anything that looks like an emphasis marker."""
    return bool(text) and _MARKER_RE.search(text) is not None


def count_spans(text: str) -> int:
    """Number of opening emphasis markers in the text."""
    if not text:
        return 0
    return sum(1 for m in _MARKER_RE.finditer(text) if not m.group(1))


def strip_markers(text: str) -> str:
    """Remove the marker tags (keeping their inner text) and normalize whitespace."""
    if not text:
        return ""
    return " ".join(_MARKER_RE.sub("", text).split())


def remove_markers(text: str) -> str:
    """Remove the marker tags only, leaving every other character in place.

    For scans that must see the words without the markers (glossary matching,
    residue detection) but keep the text's layout, unlike strip_markers.
    """
    return _MARKER_RE.sub("", text) if text else (text or "")


def is_bare_marker(tag: str) -> bool:
    """True when `tag` is exactly one emphasis marker (`<i>`, `</em>`, ...)."""
    return bool(tag) and _MARKER_RE.fullmatch(tag) is not None


def _parse(text: str) -> Optional[List[Tuple[str, Optional[str]]]]:
    """Split well-formed marked text into (segment, tag) pairs.

    tag is None for text outside any span, the lowercase tag name otherwise.
    Returns None when the markers are not well formed: nesting, a closer that
    does not match its opener, a stray closer, an unclosed opener, or an empty
    / whitespace-only span.
    """
    pieces: List[Tuple[str, Optional[str]]] = []
    open_tag: Optional[str] = None
    pos = 0
    for m in _MARKER_RE.finditer(text):
        is_close, name = bool(m.group(1)), m.group(2).lower()
        segment = text[pos:m.start()]
        if not is_close:
            if open_tag is not None:
                return None
            pieces.append((segment, None))
            open_tag = name
        else:
            if open_tag is None or name != open_tag or not segment.strip():
                return None
            pieces.append((segment, open_tag))
            open_tag = None
        pos = m.end()
    if open_tag is not None:
        return None
    pieces.append((text[pos:], None))
    return pieces


def is_well_formed(text: str) -> bool:
    """True when the markers alternate open/close cleanly (see _parse)."""
    return _parse(text or "") is not None


def sanitize(translated: str, source: str) -> Tuple[str, int]:
    """Decide which markers of a translated paragraph survive into the rebuild.

    Returns (text to rebuild, number of spans kept). The markers are kept only
    when they are well formed, the source had at least one span, and the
    translation has no more spans than the source; otherwise every marker is
    stripped and the paragraph is rebuilt as plain text. Never raises.
    """
    translated = translated or ""
    try:
        source_spans = count_spans(source or "")
        spans = count_spans(translated)
        if source_spans > 0 and spans <= source_spans and is_well_formed(translated):
            return translated, spans
    except Exception:  # pragma: no cover - defensive, the helpers do not raise
        pass
    return strip_markers(translated), 0


# A Markdown emphasis pair (`*word*`, `_word_`) some models write instead of
# the marker they were given. The delimiters must hug the text inside, and are
# not part of `**`, a word_with_underscores or a lone "a * b".
_MARKDOWN_PAIR_RES = {
    "*": re.compile(r"(?<![*\w])\*(?![\s*])([^*\n<>]*?[^\s*<>])\*(?![*\w])"),
    "_": re.compile(r"(?<![_\w])_(?![\s_])([^_\n<>]*?[^\s_<>])_(?![_\w])"),
}


def _source_tag(source: str) -> str:
    """The marker tag the source uses most (ties and no span: "i")."""
    names = [m.group(2).lower() for m in _MARKER_RE.finditer(source) if not m.group(1)]
    return "em" if names.count("em") > names.count("i") else "i"


def recover_markdown_emphasis(translated: str, source: str) -> Tuple[str, int]:
    """Turn Markdown emphasis the LLM invented back into markers.

    Some models answer a marked paragraph with `*word*` where the source had
    `<i>word</i>`. Such pairs pass marker validation (they are not markers)
    and would land in the book as literal asterisks. A pair counts as invented
    only when its delimiter never appears in the source text. While the
    translation has fewer spans than the source, each invented pair outside an
    existing span becomes a marker of the source's tag, in reading order; any
    other invented pair (beyond the missing count, or inside a span, or in
    text whose markers are not well formed) just loses its delimiters.

    Returns (text, number of spans recovered). Never raises.
    """
    translated = translated or ""
    plain_source = remove_markers(source or "")
    delims = [d for d in _MARKDOWN_PAIR_RES if d in translated and d not in plain_source]
    if not delims:
        return translated, 0

    def strip_pairs(text: str) -> str:
        for d in delims:
            text = _MARKDOWN_PAIR_RES[d].sub(r"\1", text)
        return text

    pieces = _parse(translated)
    if pieces is None:
        return strip_pairs(translated), 0

    tag = _source_tag(source or "")
    missing = count_spans(source or "") - count_spans(translated)
    recovered = 0

    def convert(m: "re.Match") -> str:
        nonlocal recovered
        if recovered >= missing:
            return m.group(1)
        recovered += 1
        return f"<{tag}>{m.group(1)}</{tag}>"

    out = []
    for segment, span_tag in pieces:
        if span_tag is not None:
            out.append(f"<{span_tag}>{strip_pairs(segment)}</{span_tag}>")
            continue
        for d in delims:
            segment = _MARKDOWN_PAIR_RES[d].sub(convert, segment)
        out.append(segment)
    return "".join(out), recovered


def append_marked_text(elem: etree._Element, text: str) -> None:
    """Write marked text into an empty element as text plus <i>/<em> children.

    Children are created without an explicit namespace, the same way the
    Plain Text rebuild creates its blocks. Text whose markers are not well
    formed is written as plain text with the markers stripped.
    """
    text = text or ""
    pieces = _parse(text)
    if pieces is None:
        elem.text = strip_markers(text) or None
        return
    elem.text = pieces[0][0] or None
    last: Optional[etree._Element] = None
    for segment, tag in pieces[1:]:
        if tag is not None:
            last = etree.SubElement(elem, tag)
            last.text = segment
        elif last is not None:
            last.tail = segment or None
