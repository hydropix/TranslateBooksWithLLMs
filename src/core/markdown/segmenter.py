"""
Markdown segmentation: split a document into untouched raw text and
translatable segments.

The parser is line based and deliberately conservative: anything it does
not recognise as prose stays raw, and raw text is copied to the output byte
for byte. What reaches the LLM is only the inline text of headings,
paragraphs, list items, blockquotes, footnote definitions and table cells.

Never translated (kept raw):
    front matter (YAML ``---`` / TOML ``+++``), fenced and indented code
    blocks, ``$$`` math blocks, HTML blocks and comments, horizontal rules,
    setext underlines, table delimiter rows, link reference definitions,
    and every structural marker (``#``, ``-``, ``1.``, ``>``, ``|``,
    indentation, task-list boxes, closing ``#`` sequences).

Protected inside a segment (replaced by a placeholder, never sent):
    inline code, link and image URLs (``](url "title")``, ``][ref]``),
    the brackets around link text, footnote references, autolinks, bare
    URLs, inline HTML tags and comments.

A segment keeps its soft line breaks: the text sent to the LLM has a plain
``\\n`` where the source had a line break, and the original break (trailing
spaces, the newline and the next line's prefix, e.g. ``"\\n> "``) is put
back on rebuild, so wrapped paragraphs inside blockquotes and list items
keep their prefixes.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

# --- Block-level patterns (applied to a line after its blockquote prefix) ---

_QUOTE_PREFIX_RE = re.compile(r'^(?:[ \t]{0,3}>[ \t]?)*')
_FENCE_OPEN_RE = re.compile(r'^([ \t]*)(`{3,}|~{3,})(.*)$')
_MATH_FENCE_RE = re.compile(r'^[ \t]*\$\$')
_HTML_COMMENT_RE = re.compile(r'^[ \t]{0,3}<!--')
_HTML_OPEN_RE = re.compile(r'^[ \t]{0,3}</?([A-Za-z][A-Za-z0-9-]*)(?=[\s/>]|$)')
_HTML_LONE_TAG_RE = re.compile(r'^[ \t]{0,3}</?[A-Za-z][^<>]*>[ \t]*$')
_HTML_DECL_RE = re.compile(r'^[ \t]{0,3}<[?!]')
# CommonMark HTML block start tags (types 1 and 6): a line opening with one
# of them starts a raw block that runs to the next blank line.
_HTML_BLOCK_TAGS = frozenset('''
    address article aside base basefont blockquote body caption center col
    colgroup dd details dialog dir div dl dt fieldset figcaption figure footer
    form frame frameset h1 h2 h3 h4 h5 h6 head header hr html iframe legend li
    link main menu menuitem nav noframes ol optgroup option p param pre script
    search section style summary table tbody td textarea tfoot th thead title
    tr track ul
'''.split())
_HR_RE = re.compile(r'^[ \t]{0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$')
_SETEXT_RE = re.compile(r'^[ \t]{0,3}(?:=+|-+)[ \t]*$')
_ATX_RE = re.compile(r'^([ \t]{0,3}#{1,6})(?:([ \t]+)(.*))?$')
_ATX_CLOSE_RE = re.compile(r'[ \t]+#+[ \t]*$|[ \t]+$')
_LIST_RE = re.compile(r'^([ \t]*)([-*+]|\d{1,9}[.)])([ \t]+)(\[[ xX]\][ \t]+)?(?=\S)')
_EMPTY_LIST_RE = re.compile(r'^[ \t]*(?:[-*+]|\d{1,9}[.)])[ \t]*$')
_FOOTNOTE_DEF_RE = re.compile(r'^([ \t]{0,3}\[\^[^\]\n]+\]:[ \t]*)(?=\S)')
_LINK_REF_DEF_RE = re.compile(r'^[ \t]{0,3}\[(?!\^)[^\]\n]+\]:[ \t]*\S')
_TABLE_DELIM_RE = re.compile(r'^[ \t]*\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)*\|?[ \t]*$')

# --- Inline spans that are never sent to the LLM ---

_INLINE_PROTECTED_RE = re.compile(
    r'``[^\n]*?``'                                  # double-backtick code
    r'|`[^`\n]+`'                                   # inline code
    r'|<!--.*?-->'                                  # inline HTML comment
    r'|<(?:https?|ftp|mailto):[^>\s]+>'             # autolink
    r'|<[^@\s<>]+@[^@\s<>]+\.[^@\s<>]+>'            # email autolink
    r'|</?[A-Za-z][A-Za-z0-9-]*(?:\s[^<>\n]*)?/?>'  # inline HTML tag
    r'|\[\^[^\]\n]+\]'                              # footnote reference
    r'|\[id\d+\]'                                   # literal placeholder look-alike
    r'|(?:https?|ftp)://[^\s<>\[\]()]*[^\s<>\[\]().,;:!?\'"*_]'  # bare URL
)

# [text](url "title"), ![alt](url) and [text][ref]: the brackets and the
# destination are protected, the text in between is translated.
_LINK_RE = re.compile(
    r'(!?\[)'
    r'([^\[\]\n]*)'
    r'(\]\((?:[^()\n]|\([^()\n]*\))*\)|\]\[[^\]\n]*\])'
)


@dataclass
class Segment:
    """A run of translatable inline text.

    Attributes:
        tokens: ``(protected, text)`` pairs in order. Text tokens may contain
            ``\\n`` (soft line breaks); protected tokens never do.
        joins: The original bytes of each soft line break, in order.
        default_join: Used for line breaks the translation adds.
        single_line: Headings and table cells, where a line break would
            break the structure, so added breaks become spaces.
        group: Segments sharing a group (the cells of one table) are kept in
            the same chunk when possible.
        row_start: First cell of a table row (a table larger than the chunk
            budget is only split at row starts).
    """
    tokens: List[Tuple[bool, str]]
    joins: List[str] = field(default_factory=list)
    default_join: str = '\n'
    single_line: bool = False
    group: int = 0
    row_start: bool = True

    @property
    def text(self) -> str:
        """The segment's logical source text (protected spans included)."""
        return ''.join(t for _, t in self.tokens)

    @property
    def prose(self) -> str:
        """Only the translatable text, protected spans removed."""
        return ''.join(t for protected, t in self.tokens if not protected)

    def protected_spans(self) -> List[str]:
        return [t for protected, t in self.tokens if protected]

    def restore_breaks(self, text: str) -> str:
        """Turn the ``\\n`` of a (translated) logical text back into source breaks."""
        if self.single_line:
            return re.sub(r'[ \t]*\n[ \t]*', ' ', text)
        # A blank line would end the paragraph (or the list item) early.
        text = re.sub(r'\n[ \t]*(?:\n[ \t]*)+', '\n', text)
        lines = text.split('\n')
        out = [lines[0]]
        for i, line in enumerate(lines[1:]):
            # The join carries the source's own trailing spaces and prefix.
            out[-1] = out[-1].rstrip(' \t')
            out.append(self.joins[i] if i < len(self.joins) else self.default_join)
            out.append(line.lstrip(' \t'))
        return ''.join(out)


Piece = Union[str, Segment]


def tokenize_inline(text: str) -> List[Tuple[bool, str]]:
    """Split inline text into ``(protected, text)`` tokens.

    Adjacent protected spans are merged so each becomes one placeholder.
    """
    tokens: List[Tuple[bool, str]] = []

    def add(protected: bool, value: str):
        if not value:
            return
        if tokens and tokens[-1][0] == protected:
            tokens[-1] = (protected, tokens[-1][1] + value)
        else:
            tokens.append((protected, value))

    def scan_simple(value: str):
        pos = 0
        for m in _INLINE_PROTECTED_RE.finditer(value):
            add(False, value[pos:m.start()])
            add(True, m.group(0))
            pos = m.end()
        add(False, value[pos:])

    pos = 0
    for m in _LINK_RE.finditer(text):
        # Inline code may contain brackets; a link match starting inside a
        # code span must not win over the code span.
        before = text[pos:m.start()]
        if before.count('`') % 2 == 1:
            continue
        scan_simple(before)
        add(True, m.group(1))
        scan_simple(m.group(2))
        add(True, m.group(3))
        pos = m.end()
    scan_simple(text[pos:])
    return tokens


def _has_prose(tokens: List[Tuple[bool, str]]) -> bool:
    return any(ch.isalpha() for protected, t in tokens if not protected for ch in t)


def _is_html_block(content: str) -> bool:
    """True when a line opens an HTML block (kept raw up to a blank line).

    A paragraph that merely starts with an inline tag (``<b>Note</b> ...``)
    is not one: it stays prose with the tag protected.
    """
    if _HTML_DECL_RE.match(content) or _HTML_LONE_TAG_RE.match(content):
        return True
    m = _HTML_OPEN_RE.match(content)
    return bool(m and m.group(1).lower() in _HTML_BLOCK_TAGS)


def _split_eol(line: str) -> Tuple[str, str]:
    body = line.rstrip('\r\n')
    return body, line[len(body):]


def _split_quote(body: str) -> Tuple[str, str]:
    prefix = _QUOTE_PREFIX_RE.match(body).group(0)
    return prefix, body[len(prefix):]


class _Builder:
    """Accumulates the alternating raw / segment piece list."""

    def __init__(self):
        self.pieces: List[Piece] = []
        self._raw: List[str] = []
        self._group = 0
        self._pending_row_start = False

    def raw(self, text: str):
        if text:
            self._raw.append(text)

    def new_group(self) -> int:
        self._group += 1
        return self._group

    def segment(self, logical: str, joins: List[str], default_join: str,
                single_line: bool = False, group: Optional[int] = None,
                row_start: bool = True):
        """Add a segment, moving its edge whitespace into the raw text.

        ``logical`` uses ``\\n`` for soft breaks and ``joins`` holds their
        original bytes. A segment without prose (only code, URLs, numbers or
        punctuation) is emitted raw.
        """
        stripped = logical.lstrip(' \t')
        self.raw(logical[:len(logical) - len(stripped)])
        core = stripped.rstrip(' \t')
        trailing = stripped[len(core):]

        tokens = tokenize_inline(core)
        if not core or not _has_prose(tokens):
            self.raw(_rejoin(core, joins))
            # A row whose first cells hold no prose still starts a row.
            self._pending_row_start = self._pending_row_start or row_start
        else:
            row_start = row_start or self._pending_row_start
            self._pending_row_start = False
            if self._raw:
                self.pieces.append(''.join(self._raw))
                self._raw = []
            self.pieces.append(Segment(
                tokens=tokens, joins=list(joins), default_join=default_join,
                single_line=single_line,
                group=group if group is not None else self.new_group(),
                row_start=row_start,
            ))
        self.raw(trailing)

    def finish(self) -> List[Piece]:
        if self._raw:
            self.pieces.append(''.join(self._raw))
            self._raw = []
        return self.pieces


def _rejoin(logical: str, joins: List[str]) -> str:
    """Rebuild source bytes from a logical text and its original joins."""
    lines = logical.split('\n')
    out = [lines[0]]
    for i, line in enumerate(lines[1:]):
        out.append(joins[i] if i < len(joins) else '\n')
        out.append(line)
    return ''.join(out)


def _front_matter_end(lines: List[str]) -> int:
    """Index of the first line after a front matter block, or 0 if none."""
    if not lines:
        return 0
    first, _ = _split_eol(lines[0])
    first = first.lstrip('﻿')
    if first.rstrip() not in ('---', '+++'):
        return 0
    closers = ('---', '...') if first.rstrip() == '---' else ('+++',)
    for i in range(1, len(lines)):
        body, _ = _split_eol(lines[i])
        if body.rstrip() in closers:
            return i + 1
    return 0


def _split_table_row(content: str) -> List[Tuple[bool, str]]:
    """Split a table row into ``(is_cell, text)`` parts on unescaped pipes.

    Pipes inside backtick code spans do not split. Pipes themselves are
    returned as non-cell parts.
    """
    parts: List[Tuple[bool, str]] = []
    buf = []
    in_code = False
    i = 0
    while i < len(content):
        c = content[i]
        if c == '\\' and i + 1 < len(content):
            buf.append(content[i:i + 2])
            i += 2
            continue
        if c == '`':
            in_code = not in_code
        if c == '|' and not in_code:
            parts.append((True, ''.join(buf)))
            parts.append((False, '|'))
            buf = []
        else:
            buf.append(c)
        i += 1
    parts.append((True, ''.join(buf)))
    return parts


def _is_table_start(lines: List[str], i: int) -> bool:
    if i + 1 >= len(lines):
        return False
    _, content = _split_quote(_split_eol(lines[i])[0])
    _, delim = _split_quote(_split_eol(lines[i + 1])[0])
    return ('|' in content and '|' in delim and '-' in delim
            and bool(_TABLE_DELIM_RE.match(delim)))


def _starts_block(content: str) -> bool:
    """True when a line (blockquote prefix removed) opens a new block."""
    return bool(
        not content.strip()
        or _FENCE_OPEN_RE.match(content)
        or _MATH_FENCE_RE.match(content)
        or _HTML_COMMENT_RE.match(content)
        or _is_html_block(content)
        or _HR_RE.match(content)
        or _SETEXT_RE.match(content)
        or _ATX_RE.match(content)
        or _LIST_RE.match(content)
        or _EMPTY_LIST_RE.match(content)
        or _FOOTNOTE_DEF_RE.match(content)
        or _LINK_REF_DEF_RE.match(content)
    )


def segment_markdown(text: str) -> List[Piece]:
    """Split a Markdown document into raw strings and :class:`Segment` objects.

    Concatenating the raw strings with each segment's original bytes (see
    :func:`rebuild`) gives back ``text`` exactly.
    """
    lines = text.splitlines(keepends=True)
    b = _Builder()
    n = len(lines)

    i = _front_matter_end(lines)
    b.raw(''.join(lines[:i]))

    in_list = False
    prev_blank = True

    while i < n:
        body, eol = _split_eol(lines[i])
        qprefix, content = _split_quote(body)

        # Blank line
        if not content.strip():
            b.raw(lines[i])
            i += 1
            prev_blank = True
            continue

        indent = len(content) - len(content.lstrip(' \t'))
        if indent == 0 and not _LIST_RE.match(content):
            in_list = False

        # Fenced code block: raw up to and including the closing fence.
        m = _FENCE_OPEN_RE.match(content)
        if m and not (m.group(2)[0] == '`' and '`' in m.group(3)):
            fence = m.group(2)
            close_re = re.compile(r'^[ \t]*' + re.escape(fence[0]) + '{' + str(len(fence)) + r',}[ \t]*$')
            j = i + 1
            while j < n:
                _, inner = _split_quote(_split_eol(lines[j])[0])
                if close_re.match(inner):
                    break
                j += 1
            end = min(j + 1, n)
            b.raw(''.join(lines[i:end]))
            i = end
            prev_blank = False
            continue

        # $$ math block
        if _MATH_FENCE_RE.match(content):
            j = i
            one_line = content.strip()
            if not (len(one_line) > 4 and one_line.endswith('$$')):
                j = i + 1
                while j < n and '$$' not in _split_eol(lines[j])[0]:
                    j += 1
            end = min(j + 1, n)
            b.raw(''.join(lines[i:end]))
            i = end
            prev_blank = False
            continue

        # HTML comment: raw up to the line that closes it.
        if _HTML_COMMENT_RE.match(content):
            j = i
            while j < n and '-->' not in lines[j]:
                j += 1
            end = min(j + 1, n)
            b.raw(''.join(lines[i:end]))
            i = end
            prev_blank = False
            continue

        # HTML block: raw up to the next blank line.
        if _is_html_block(content):
            j = i
            while j < n and _split_eol(lines[j])[0].strip():
                j += 1
            b.raw(''.join(lines[i:j]))
            i = j
            prev_blank = False
            continue

        # Horizontal rule, setext underline, link reference definition
        if _HR_RE.match(content) or _SETEXT_RE.match(content) or _LINK_REF_DEF_RE.match(content):
            b.raw(lines[i])
            i += 1
            prev_blank = False
            continue

        # Indented code block (not inside a list, after a blank line)
        if indent and prev_blank and not in_list and \
                (content.startswith('    ') or content.startswith('\t')):
            j = i
            while j < n:
                lb = _split_eol(lines[j])[0]
                _, lc = _split_quote(lb)
                if lc.strip() and not (lc.startswith('    ') or lc.startswith('\t')):
                    break
                j += 1
            # Trailing blank lines are left to the main loop.
            while j > i and not _split_eol(lines[j - 1])[0].strip():
                j -= 1
            b.raw(''.join(lines[i:j]))
            i = j
            prev_blank = False
            continue

        # Table
        if _is_table_start(lines, i):
            group = b.new_group()
            j = i
            while j < n:
                lb, leol = _split_eol(lines[j])
                lq, lc = _split_quote(lb)
                if not lc.strip() or '|' not in lc:
                    break
                b.raw(lq)
                if j == i + 1:
                    b.raw(lc)
                else:
                    row_start = True
                    for is_cell, part in _split_table_row(lc):
                        if is_cell:
                            b.segment(part, [], ' ', single_line=True, group=group,
                                      row_start=row_start)
                            row_start = False
                        else:
                            b.raw(part)
                b.raw(leol)
                j += 1
            i = j
            prev_blank = False
            continue

        # ATX heading
        m = _ATX_RE.match(content)
        if m:
            b.raw(qprefix + m.group(1) + (m.group(2) or ''))
            heading = m.group(3) or ''
            close = _ATX_CLOSE_RE.search(heading)
            closing = ''
            if close:
                closing = heading[close.start():]
                heading = heading[:close.start()]
            b.segment(heading, [], ' ', single_line=True)
            b.raw(closing + eol)
            i += 1
            prev_blank = False
            continue

        # Empty list item
        if _EMPTY_LIST_RE.match(content):
            b.raw(lines[i])
            in_list = True
            i += 1
            prev_blank = False
            continue

        # Paragraph start: list item, footnote definition or plain text.
        m = _LIST_RE.match(content)
        if m:
            in_list = True
            marker = m.group(0)
            cont_indent = ' ' * len(m.group(1) + m.group(2) + m.group(3))
        else:
            m = _FOOTNOTE_DEF_RE.match(content)
            if m:
                marker = m.group(0)
                cont_indent = '    '
            else:
                marker = content[:indent]
                cont_indent = content[:indent]
        b.raw(qprefix + marker)
        logical = [content[len(marker):]]
        joins: List[str] = []
        default_join = '\n' + qprefix + cont_indent

        j = i + 1
        while j < n:
            nb, _ = _split_eol(lines[j])
            nq, nc = _split_quote(nb)
            if _starts_block(nc) or _is_table_start(lines, j):
                break
            prev_line = logical[-1]
            kept = prev_line.rstrip(' \t')
            lead = nc[:len(nc) - len(nc.lstrip(' \t'))]
            _, peol = _split_eol(lines[j - 1])
            joins.append(prev_line[len(kept):] + peol + nq + lead)
            logical[-1] = kept
            logical.append(nc[len(lead):])
            j += 1

        _, last_eol = _split_eol(lines[j - 1])
        b.segment('\n'.join(logical), joins, default_join)
        b.raw(last_eol)
        i = j
        prev_blank = False

    return b.finish()


def rebuild(pieces: List[Piece], translations: Optional[Dict[int, str]] = None) -> str:
    """Rebuild the document, substituting translated segment texts.

    Args:
        pieces: Output of :func:`segment_markdown`.
        translations: Segment index (position among the segments only) ->
            translated logical text, protected spans already restored.
            Segments without an entry keep their source text.
    """
    translations = translations or {}
    out = []
    seg_index = 0
    for piece in pieces:
        if isinstance(piece, str):
            out.append(piece)
            continue
        translated = translations.get(seg_index)
        if translated is None:
            out.append(_rejoin(piece.text, piece.joins))
        else:
            out.append(piece.restore_breaks(translated))
        seg_index += 1
    return ''.join(out)


def segments_of(pieces: List[Piece]) -> List[Segment]:
    return [p for p in pieces if isinstance(p, Segment)]
