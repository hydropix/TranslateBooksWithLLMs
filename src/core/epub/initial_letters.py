"""Keep drop caps and styled initials attached to their word (issue #333).

Books style the first letter of a paragraph with an inline element:

    <p><span class="dropcap">A</span>lice was beginning...</p>

Turned into placeholders as-is, the model receives ``[id0]A[id1]lice was...``:
a word cut in two. It then translates "A" and "lice" separately, which drops
the initial (``<span class="dropcap"/>``) or doubles it (``AAlice``, ``AUma``).

The fix is deterministic and never touches the prompt. Before placeholders are
created, the element's closing tag is removed and a marker carrying the tag
name is left after its opening tag, so the model sees the whole word
(``[id0]Alice was...``). The marker sits inside the placeholder's tag group and
is never shown to the model. After placeholders are restored, the marker closes
the element right after the first letter of whatever text now follows it, so
the translation's own first letter becomes the initial.
"""
import re

from .tag_classifier import TagClassifier

# Inline elements that books use for initials.
_INLINE_TAGS = r'span|b|strong|big|em|i|small|font|u'

# An inline element holding one or two letters, optionally after an opening
# quote or inverted punctuation mark (“Ah, ¿Qué).
_INITIAL_RE = re.compile(
    r'(?P<open><(?P<tag>' + _INLINE_TAGS + r')\b[^>]*>)'
    r'(?P<letters>[\"\'“‘«„¿¡]?[^\W\d_]{1,2})'
    r'</(?P=tag)\s*>'
    r'(?=(?P<next>[^\W\d_]|\s))',
    re.IGNORECASE,
)

# When the initial is a whole word ("<span>A</span> good"), only a class that
# names it as an initial makes it safe to treat that way: a bold "I" at the
# start of a paragraph is emphasis, not a drop cap.
_INITIAL_CLASS_RE = re.compile(r'class\s*=\s*["\'][^"\']*(drop|initial|cap|lettrine)', re.IGNORECASE)

_MARKER_PREFIX = '<tbl-initial data-close="'
_MARKER_RE = re.compile(
    r'<tbl-initial data-close="(?P<tag>[a-z0-9]+)"/>'
    r'(?P<initial>\s*(?:[\"\'“‘«„¿¡]\s?)?[^\W_]|)',
    re.IGNORECASE,
)

_BLOCK_OPEN_RE = re.compile(
    r'<(?:' + '|'.join(TagClassifier.BLOCK_TAGS) + r')[\s>]',
    re.IGNORECASE,
)
_TAG_RE = re.compile(r'<[^>]+>')

# How far back to look for the enclosing block's opening tag.
_LOOKBACK = 2000


def _starts_block_text(html: str, position: int) -> bool:
    """True when no text sits between the enclosing block's start and position."""
    window_start = max(0, position - _LOOKBACK)
    window = html[window_start:position]
    block_starts = list(_BLOCK_OPEN_RE.finditer(window))
    if not block_starts:
        return False
    since_block = window[block_starts[-1].start():]
    return _TAG_RE.sub('', since_block).strip() == ''


def merge_initial_letters(html: str) -> str:
    """Rejoin a styled initial with the rest of its word before translation."""
    if '<' not in html:
        return html

    def replace(match: re.Match) -> str:
        if not _starts_block_text(html, match.start()):
            return match.group(0)
        if match.group('next').isspace() and not _INITIAL_CLASS_RE.search(match.group('open')):
            return match.group(0)
        tag = match.group('tag').lower()
        return f'{match.group("open")}{_MARKER_PREFIX}{tag}"/>{match.group("letters")}'

    return _INITIAL_RE.sub(replace, html)


def close_initial_letters(html: str) -> str:
    """Close each initial element after the first letter that now follows it.

    When no letter follows (the model moved or dropped the text), the element
    is closed immediately so the document stays well-formed.
    """
    if _MARKER_PREFIX not in html:
        return html
    return _MARKER_RE.sub(
        lambda m: f'{m.group("initial")}</{m.group("tag")}>',
        html,
    )
