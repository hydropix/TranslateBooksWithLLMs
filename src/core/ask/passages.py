"""
Split a document's plain text into retrievable passages.

Markdown-style headings (``#`` .. ``####``) are tracked so every passage carries
the section it came from, which is what the UI shows as a citation. Documents
without headings (most EPUB/DOCX/PDF extractions) are grouped by paragraph.
"""
import re
from dataclasses import dataclass, asdict
from typing import Dict, List

MAX_PASSAGE_CHARS = 1800
MIN_PASSAGE_CHARS = 200

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


@dataclass
class Passage:
    id: int
    section: str
    text: str

    def to_dict(self) -> Dict:
        return asdict(self)


def _split_long(text: str, limit: int) -> List[str]:
    """Group paragraphs into pieces of at most ~limit chars (a single huge paragraph stays whole)."""
    pieces, current = [], ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if current and len(current) + len(para) + 2 > limit:
            pieces.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        pieces.append(current)
    return pieces


def _sections(text: str):
    """Yield (section_path, body) pairs, splitting on markdown headings outside code fences."""
    stack: List[tuple] = []
    buf: List[str] = []
    in_code = False

    def flush():
        body = "\n".join(buf).strip()
        buf.clear()
        if re.search(r"\w", body):
            return (" > ".join(title for _, title in stack), body)
        return None

    for line in text.splitlines():
        if _FENCE_RE.match(line):
            in_code = not in_code
        match = None if in_code else _HEADING_RE.match(line)
        if match:
            section = flush()
            if section:
                yield section
            level = len(match.group(1))
            stack = [(lvl, title) for lvl, title in stack if lvl < level] + [(level, match.group(2))]
        else:
            buf.append(line)
    section = flush()
    if section:
        yield section


def split_into_passages(text: str, max_chars: int = MAX_PASSAGE_CHARS,
                        min_chars: int = MIN_PASSAGE_CHARS) -> List[Passage]:
    """Split document text into passages of roughly ``max_chars`` characters."""
    passages: List[Passage] = []
    seen = set()  # normalised text of passages already kept (drops repeated boilerplate/duplicated pages)
    for section, body in _sections(text):
        # Fold a tiny section into the previous passage of the same top-level section.
        if (passages and len(body) < min_chars
                and passages[-1].section.split(" > ")[0] == section.split(" > ")[0]
                and len(passages[-1].text) + len(body) < max_chars):
            heading = section.split(" > ")[-1]
            passages[-1].text += f"\n\n{heading}\n{body}" if heading else f"\n\n{body}"
            continue
        for piece in _split_long(body, max_chars):
            fingerprint = " ".join(piece.lower().split())
            if re.search(r"\w", piece) and fingerprint not in seen:
                seen.add(fingerprint)
                passages.append(Passage(id=len(passages) + 1, section=section, text=piece))
    return passages
