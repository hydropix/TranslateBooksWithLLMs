"""
Token-level diff between a snippet and its corrected version.

Two uses: turning an LLM-corrected snippet into the smallest set of character
edits (so formatting runs and untouched words keep their exact bytes), and
the guard that rejects corrections reaching beyond the words around a match.
"""
import difflib
import re
from typing import List, Sequence, Tuple

# CJK characters are tokens of their own (no spaces between words in those
# scripts); everything else splits into words, punctuation and whitespace.
_TOKEN_RE = re.compile(
    r"[぀-ヿ㐀-䶿一-鿿가-힯]"
    r"|\s+|\w+|[^\w\s]"
)

Edit = Tuple[int, int, str]


def tokenize(text: str) -> List[Tuple[int, int]]:
    """(start, end) offsets of every token; together they cover the whole text."""
    return [(m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def _opcodes(a: str, b: str):
    ta, tb = tokenize(a), tokenize(b)
    sa = [a[s:e] for s, e in ta]
    sb = [b[s:e] for s, e in tb]
    matcher = difflib.SequenceMatcher(a=sa, b=sb, autojunk=False)
    return ta, tb, matcher.get_opcodes()


def minimal_edits(original: str, corrected: str, base: int = 0) -> List[Edit]:
    """Edits (offsets in `original`, shifted by `base`) turning it into `corrected`."""
    if original == corrected:
        return []
    ta, tb, ops = _opcodes(original, corrected)
    edits: List[Edit] = []
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        start = ta[i1][0] if i1 < len(ta) else len(original)
        end = ta[i2 - 1][1] if i2 > i1 else start
        new_start = tb[j1][0] if j1 < len(tb) else len(corrected)
        new_end = tb[j2 - 1][1] if j2 > j1 else new_start
        edits.append((base + start, base + end, corrected[new_start:new_end]))
    return edits


def changes_outside(
    original: str,
    corrected: str,
    allowed: Sequence[Tuple[int, int]],
    window_words: int,
) -> bool:
    """True when `corrected` changes `original` beyond `window_words` words on
    either side of the `allowed` character ranges (the matches).

    Whitespace-only differences are ignored, so a reflowed space never fails
    the guard on its own.
    """
    ta, tb, ops = _opcodes(original, corrected)
    word_idx = [i for i, (s, e) in enumerate(ta) if not original[s:e].isspace()]
    position = {tok: n for n, tok in enumerate(word_idx)}

    near = set()
    for a_start, a_end in allowed:
        hit = [position[i] for i in word_idx if ta[i][0] < a_end and ta[i][1] > a_start]
        if not hit:
            continue
        lo = max(0, min(hit) - window_words)
        hi = min(len(word_idx) - 1, max(hit) + window_words)
        near.update(word_idx[lo:hi + 1])

    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        touched = [i for i in range(i1, i2) if not original[ta[i][0]:ta[i][1]].isspace()]
        if touched:
            if any(i not in near for i in touched):
                return True
            continue
        # Only whitespace removed or text inserted: whitespace alone is fine,
        # new words must sit next to a word that is near a match.
        inserted = corrected[tb[j1][0]:tb[j2 - 1][1]] if j2 > j1 else ""
        if not inserted.strip():
            continue
        before = [i for i in word_idx if i < i1][-1:]
        after = [i for i in word_idx if i >= i2][:1]
        if not any(n in near for n in before + after):
            return True
    return False
