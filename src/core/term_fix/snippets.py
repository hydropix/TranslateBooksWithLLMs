"""
Sentence-level snippets around matches, and splicing edits back into the text.

A snippet is the sentence (or line) that encloses one or more matches. The
smart mode sends only snippets to the LLM, never whole chunks; the report uses
them to show each change in context. Overlapping or touching snippets are
merged, so two matches in one sentence make one snippet.
"""
import re
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from src.core.term_fix.matcher import Span

#: Hard cap on how far a snippet extends on each side of its matches when no
#: sentence boundary is found (long run-on sentences, lists without punctuation).
MAX_SIDE_CHARS = 200

# A sentence ends after Latin terminators followed by whitespace (closing quotes
# and brackets stay with the sentence), after CJK terminators (no whitespace in
# those scripts), or at a line break.
_BOUNDARY_RE = re.compile(
    r"[.!?…]+[\"'»”’)\]]*(?=\s)"
    r"|[。！？]+[」』”’）]*"
    r"|\n"
)


@dataclass
class Snippet:
    """text[start:end], holding the given spans (absolute offsets)."""
    start: int
    end: int
    spans: List[Span] = field(default_factory=list)


def _snippet_start(text: str, pos: int) -> int:
    """Start of the sentence containing `pos`, at most MAX_SIDE_CHARS before it."""
    window_start = max(0, pos - MAX_SIDE_CHARS)
    window = text[window_start:pos]
    boundary_end = None
    for m in _BOUNDARY_RE.finditer(window):
        boundary_end = m.end()
    if boundary_end is not None:
        start = window_start + boundary_end
    elif window_start == 0:
        return 0
    else:
        # No boundary in reach: cut at the first whitespace so no word is split.
        ws = re.search(r"\s", window)
        start = window_start + ws.end() if ws else pos
    while start < pos and text[start].isspace():
        start += 1
    return start


def _snippet_end(text: str, pos: int) -> int:
    """End of the sentence containing `pos`, at most MAX_SIDE_CHARS after it."""
    window_end = min(len(text), pos + MAX_SIDE_CHARS)
    window = text[pos:window_end]
    m = _BOUNDARY_RE.search(window)
    if m:
        # Keep the terminator, leave the newline out.
        end = pos + (m.start() if m.group() == "\n" else m.end())
        return end
    if window_end == len(text):
        return len(text)
    ws = None
    for ws in re.finditer(r"\s", window):
        pass
    return pos + ws.start() if ws else pos


def build_snippets(text: str, spans: Sequence[Span]) -> List[Snippet]:
    """Group spans (text order) into merged sentence snippets."""
    snippets: List[Snippet] = []
    for span in sorted(spans, key=lambda s: s.start):
        start = min(_snippet_start(text, span.start), span.start)
        end = max(_snippet_end(text, span.end), span.end)
        if snippets and start <= snippets[-1].end:
            last = snippets[-1]
            last.end = max(last.end, end)
            last.spans.append(span)
        else:
            snippets.append(Snippet(start, end, [span]))
    return snippets


def splice(text: str, edits: Sequence[Tuple[int, int, str]]) -> str:
    """Apply non-overlapping (start, end, replacement) edits to `text`.

    Edits are applied end to start so earlier offsets stay valid. Raises
    ValueError on overlapping edits.
    """
    ordered = sorted(edits, key=lambda e: e[0])
    for (s1, e1, _), (s2, _e2, _r) in zip(ordered, ordered[1:]):
        if s2 < e1:
            raise ValueError(f"Overlapping edits at {s1}-{e1} and {s2}.")
    result = text
    for start, end, replacement in reversed(ordered):
        result = result[:start] + replacement + result[end:]
    return result
