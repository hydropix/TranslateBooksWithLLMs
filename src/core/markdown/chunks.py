"""
Chunking and the placeholder codec of Markdown translation.

A chunk is a run of consecutive segments. It is sent to the LLM as text in
which every segment opens with a boundary placeholder and every protected
inline span (inline code, URL, ...) is replaced by an inline placeholder,
numbered in order with the project's ``[idN]`` format::

    [id0]Getting started

    [id1]Install it with [id2] and read [id3]the guide[id4].

The boundary placeholders let the answer be split back into segments
without trusting blank lines; the inline ones keep the protected spans out
of the request. Placeholders are matched back by position, so each segment
of the answer is accepted only when its placeholders all came back, once,
in order; a segment that fails keeps its source text.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .segmenter import Segment

PLACEHOLDER_RE = re.compile(r'\[id(\d+)\]')


def _ph(index: int) -> str:
    return f"[id{index}]"


@dataclass
class ChunkPlan:
    """The segments of one chunk and how their placeholders are numbered.

    Attributes:
        segment_indices: Global indices of the chunk's segments, in order.
        boundaries: Placeholder number opening each segment.
        inline: Per segment, ``(placeholder number, original span)`` pairs.
    """
    segment_indices: List[int]
    boundaries: List[int]
    inline: List[List[Tuple[int, str]]]

    @property
    def placeholder_count(self) -> int:
        return len(self.boundaries) + sum(len(x) for x in self.inline)


def plan_chunk(segments: List[Segment], segment_indices: List[int]) -> Tuple[ChunkPlan, str]:
    """Number the placeholders of a chunk and render the text sent to the LLM."""
    counter = 0
    boundaries: List[int] = []
    inline: List[List[Tuple[int, str]]] = []
    rendered: List[str] = []
    for global_idx in segment_indices:
        seg = segments[global_idx]
        boundaries.append(counter)
        parts = [_ph(counter)]
        counter += 1
        spans: List[Tuple[int, str]] = []
        for protected, text in seg.tokens:
            if protected:
                spans.append((counter, text))
                parts.append(_ph(counter))
                counter += 1
            else:
                parts.append(text)
        inline.append(spans)
        rendered.append(''.join(parts))
    plan = ChunkPlan(list(segment_indices), boundaries, inline)
    return plan, '\n\n'.join(rendered)


def decode_chunk(plan: ChunkPlan, answer: str) -> Tuple[Dict[int, str], List[int]]:
    """Map an LLM answer back to translated logical texts per segment.

    Returns:
        ``(translations, failed)``: global segment index -> logical text
        with protected spans restored, and the global indices of segments
        whose placeholders did not come back intact.
    """
    matches = list(PLACEHOLDER_RE.finditer(answer))
    positions: Dict[int, List[re.Match]] = {}
    for m in matches:
        positions.setdefault(int(m.group(1)), []).append(m)

    translations: Dict[int, str] = {}
    failed: List[int] = []
    for local, global_idx in enumerate(plan.segment_indices):
        text = _decode_segment(plan, local, answer, matches, positions)
        if text is None:
            failed.append(global_idx)
        else:
            translations[global_idx] = text
    return translations, failed


def _decode_segment(plan: ChunkPlan, local: int, answer: str,
                    matches: List[re.Match],
                    positions: Dict[int, List[re.Match]]) -> Optional[str]:
    """The restored text of one segment, or None if its placeholders broke."""
    expected = [plan.boundaries[local]] + [n for n, _ in plan.inline[local]]
    if any(len(positions.get(n, [])) != 1 for n in expected):
        return None
    start = positions[plan.boundaries[local]][0]

    # The segment runs to the next segment's boundary. If that one is
    # missing or misplaced, the text may hold a merged neighbour: reject.
    end_pos = len(answer)
    if local + 1 < len(plan.boundaries):
        next_b = positions.get(plan.boundaries[local + 1], [])
        if len(next_b) != 1 or next_b[0].start() < start.end():
            return None
        end_pos = next_b[0].start()

    body_matches = [m for m in matches if start.end() <= m.start() < end_pos]
    if [int(m.group(1)) for m in body_matches] != expected[1:]:
        return None

    spans = dict(plan.inline[local])
    out = []
    pos = start.end()
    for m in body_matches:
        out.append(answer[pos:m.start()])
        out.append(spans[int(m.group(1))])
        pos = m.end()
    out.append(answer[pos:end_pos])
    text = ''.join(out).strip()
    if not text:
        return None
    return text


def group_chunks(segments: List[Segment], max_tokens: int,
                 count_tokens) -> List[List[int]]:
    """Group segment indices into chunks of at most ``max_tokens``.

    Segments are never split. The cells of one table stay in one chunk
    unless the table alone exceeds the budget, in which case it is split at
    row boundaries only. A single segment larger than the budget gets a
    chunk of its own.
    """
    # Units that must stay together: one per segment, one per table row.
    units: List[List[int]] = []
    for idx, seg in enumerate(segments):
        if units and seg.group == segments[units[-1][0]].group and not seg.row_start:
            units[-1].append(idx)
        else:
            units.append([idx])

    tokens = [count_tokens(seg.text) + 4 for seg in segments]

    chunks: List[List[int]] = []
    current: List[int] = []
    current_tokens = 0
    i = 0
    while i < len(units):
        # Collect a whole table (its consecutive row units) when it fits.
        group = segments[units[i][0]].group
        j = i
        while j < len(units) and segments[units[j][0]].group == group:
            j += 1
        block = [idx for unit in units[i:j] for idx in unit]
        block_tokens = sum(tokens[k] for k in block)

        if block_tokens <= max_tokens or j - i == 1:
            if current and current_tokens + block_tokens > max_tokens:
                chunks.append(current)
                current, current_tokens = [], 0
            current.extend(block)
            current_tokens += block_tokens
        else:
            # A table larger than the budget: split it at row boundaries.
            for unit in units[i:j]:
                unit_tokens = sum(tokens[k] for k in unit)
                if current and current_tokens + unit_tokens > max_tokens:
                    chunks.append(current)
                    current, current_tokens = [], 0
                current.extend(unit)
                current_tokens += unit_tokens
        i = j
    if current:
        chunks.append(current)
    return chunks
