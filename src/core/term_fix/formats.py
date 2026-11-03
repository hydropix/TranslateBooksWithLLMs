"""
Format helpers for the term-fix pass: where the editable text lives.

The pass never round-trips a document through HTML or a tag-preservation
layer. EPUB content files are edited as raw markup, text segment by text
segment, so tags, attributes and untouched text keep their exact bytes. DOCX
paragraphs are edited run by run, so formatting stays on its run.
"""
import html
import re
from typing import List, Sequence, Tuple

Edit = Tuple[int, int, str]

# Markup tokens that are never text: comments, CDATA, processing instructions,
# doctype, and ordinary tags.
_MARKUP_RE = re.compile(
    r"<!--.*?-->|<!\[CDATA\[.*?\]\]>|<\?.*?\?>|<![A-Za-z][^>]*>|<[^>]+>",
    re.DOTALL,
)
_TAG_NAME_RE = re.compile(r"<\s*(/?)\s*([A-Za-z][\w:.-]*)")
_SKIPPED_ELEMENTS = frozenset({"script", "style"})


def _local_name(name: str) -> str:
    return name.split(":")[-1].lower()


def xml_text_segments(markup: str, body_only: bool) -> List[Tuple[int, int]]:
    """(start, end) of every raw text segment between tags.

    With `body_only`, text before <body> and after </body> is ignored (the
    <head> title belongs to metadata, not to the book's prose). Text inside
    <script> and <style> is always ignored. Whitespace-only segments are
    dropped since they cannot hold a term.
    """
    segments: List[Tuple[int, int]] = []
    in_body = not body_only
    skip_depth = 0
    pos = 0
    for m in _MARKUP_RE.finditer(markup):
        if m.start() > pos and in_body and not skip_depth:
            if markup[pos:m.start()].strip():
                segments.append((pos, m.start()))
        pos = m.end()
        tag = _TAG_NAME_RE.match(m.group())
        if not tag:
            continue
        closing, name = tag.group(1) == "/", _local_name(tag.group(2))
        self_closing = m.group().rstrip(">").rstrip().endswith("/")
        if name == "body" and body_only:
            in_body = not closing
        elif name in _SKIPPED_ELEMENTS and not self_closing:
            skip_depth = max(0, skip_depth + (-1 if closing else 1))
    if pos < len(markup) and in_body and not skip_depth and markup[pos:].strip():
        segments.append((pos, len(markup)))
    return segments


def decode_segment(raw: str) -> str:
    """Raw markup text -> plain text (entities resolved)."""
    return html.unescape(raw) if "&" in raw else raw


def encode_segment(text: str) -> str:
    """Plain text -> markup text. Only the three characters XML requires are
    escaped; every other character is written as itself (UTF-8)."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def distribute_edits(pieces: Sequence[str], edits: Sequence[Edit]) -> List[str]:
    """Apply edits expressed on "".join(pieces) to the pieces themselves.

    An edit inside one piece stays in that piece. An edit spanning several
    pieces puts its replacement in the first one and removes the covered text
    from the others, so the formatting of the first run wins for the new word.
    """
    result = list(pieces)
    offsets, total = [], 0
    for piece in pieces:
        offsets.append(total)
        total += len(piece)

    def piece_at(pos: int, prefer_end: bool) -> int:
        """Index of the non-empty piece holding `pos` (or ending at it)."""
        last = None
        for i, piece in enumerate(pieces):
            if not piece:
                continue
            last = i
            start, end = offsets[i], offsets[i] + len(piece)
            if start <= pos < end or (prefer_end and pos == end):
                return i
        return last if last is not None else 0

    for start, end, replacement in sorted(edits, key=lambda e: e[0], reverse=True):
        first = piece_at(start, prefer_end=False)
        last = piece_at(max(start, end - 1), prefer_end=False) if end > start else first
        if start == total:
            first = last = piece_at(start, prefer_end=True)
        a, b = start - offsets[first], end - offsets[last]
        if first == last:
            result[first] = result[first][:a] + replacement + result[first][b:]
        else:
            result[first] = result[first][:a] + replacement
            for k in range(first + 1, last):
                result[k] = ""
            result[last] = result[last][b:]
    return result
