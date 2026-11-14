"""
Plain-text extraction for PDF translation (PDF -> ordered paragraphs).

Reads a text PDF with PyMuPDF and produces a PdfPlainContent:
- one clean string per paragraph (never containing "\\n", never empty)
- a coarse style per paragraph ('heading1'..'heading3', 'list', 'normal', 'cell')
- a ParagraphFormat per paragraph (dominant colour, font family, paragraph-level
  bold/italic, background box)
- ruled tables, whose non-empty cells are ordinary paragraphs with style 'cell'
  (contiguous, row-major) and whose grid is recorded as a PdfTable
- background boxes (callouts, table header fills) referenced by the formats
- images anchored to the index of the paragraph they follow
- page size, document metadata and the document's dominant font family

Text blocks are read in content-stream order. Lines inside a detected table
are taken out of their blocks and the table is emitted as a unit in their
place. Running headers/footers and page numbers are dropped, blocks whose
lines change font size or colour are split (eyebrow / title / subtitle),
hyphenated line breaks are repaired (see join_lines and
_resolve_break_hyphens), and paragraphs broken by a page boundary are
reassembled when their formats match. Scanned documents (no text layer)
are rejected: OCR is out of scope.

The input is untrusted: every PyMuPDF call sits behind the exception
contract of extract_pdf_paragraphs (only PdfOpenError, PdfEncryptedError and
PdfNoTextLayerError escape), span dicts are read with safe defaults, and the
layout helpers never raise. The output is deterministic for a given input:
paragraphs_text is the persisted translation unit list of a checkpointed job.
"""
import hashlib
import logging
import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, Hashable, List, Optional, Tuple, Union

import pymupdf

from .content import (
    FONT_FAMILIES,
    ParagraphFormat,
    PdfBox,
    PdfEncryptedError,
    PdfError,
    PdfImage,
    PdfNoTextLayerError,
    PdfOpenError,
    PdfPlainContent,
    PdfTable,
)
from .layout import (
    CONTAIN_TOLERANCE,
    DetectedBox,
    DetectedTable,
    _as_bbox,
    _contained,
    detect_boxes,
    detect_tables,
)
from .styling import FontResolver, classify_family, color_int_to_hex, is_bold, is_italic, is_light, is_near_black

logger = logging.getLogger(__name__)

BBox = Tuple[float, float, float, float]


# --- Tuning constants --------------------------------------------------------

HEADER_FOOTER_BAND = 0.07        # fraction of page height at top and bottom
REPEAT_MIN_PAGES = 3             # repetition rule needs at least this many pages
REPEAT_RATIO = 0.5               # signature present on >= 50% of pages
H1_RATIO, H2_RATIO, H3_RATIO = 1.6, 1.3, 1.15
HEADING_MAX_CHARS = 200
BOLD_HEADING_MAX_CHARS = 80
MIN_IMAGE_SIDE_PT = 24.0
MIN_CHARS_PER_PAGE = 20          # scanned-document threshold (average non-whitespace chars per page)
SIZE_SPLIT_RATIO = 1.15          # consecutive lines whose sizes differ by this ratio start a new block
COLOR_SPLIT_SHARE = 0.8          # a line's dominant colour counts for a split when it covers this share
TABLE_LINE_TOLERANCE = 1.0       # pt: a line whose centre is this close to a table belongs to it
TEXT_FLAGS = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_LIGATURES
PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s+)?(?:\d{1,4}|[ivxlcdm]{1,7})(?:\s*(?:/|of)\s*\d{1,4})?\s*$", re.I)
LIST_MARKER_RE = re.compile(r"^\s*(?:[•◦▪▫‣⁃●○■□*·]|-(?=\s)|\(?\d{1,3}[.)]|\(?[A-Za-z][.)])\s+")
# Sentence-final punctuation, Latin and others: CJK full/halfwidth stops, Arabic
# question mark, Urdu full stop, Devanagari danda, Armenian, Ethiopic, Myanmar, Khmer.
TERMINAL_PUNCT_RE = re.compile(r"[.!?…:;\"»”’)\]。！？」』）．｡︒؟۔।॥։።။។]\s*$")
LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"}

# Scripts that wrap lines without spaces (Hangul is excluded: Korean wraps at spaces)
_CJK_RANGES = (
    ("　", "〿"),   # CJK symbols and punctuation
    ("぀", "ヿ"),   # Hiragana, Katakana
    ("㐀", "䶿"),   # CJK Unified Ideographs Extension A
    ("一", "鿿"),   # CJK Unified Ideographs
    ("豈", "﫿"),   # CJK Compatibility Ideographs
    ("＀", "￯"),   # Halfwidth and fullwidth forms
)

_SOFT_HYPHEN = "­"
# Private-use stand-in for a hyphen found at a line end, written by the internal
# joins while the document is read and resolved once the whole document is
# known (see _resolve_break_hyphens). Never present in the returned content.
_BREAK_HYPHEN = "\ue000"
_LETTERS = r"[^\W\d_]+"
_BREAK_WORD_RE = re.compile(rf"({_LETTERS})\ue000({_LETTERS})")
_MARKED_TOKEN_RE = re.compile(r"\S*\ue000\S*")
_WORD_RE = re.compile(rf"{_LETTERS}(?:-{_LETTERS})*")
_BODY_STYLES = ("normal", "list")   # styles whose spans decide the document's font family
_ENCRYPTED_MESSAGE = "This PDF is password-protected. Remove the password and try again."
_NO_TEXT_MESSAGE = (
    "This PDF has no extractable text layer (scanned document?). "
    "OCR is not supported: convert it with an OCR tool first."
)


# --- Text helpers ------------------------------------------------------------


def _is_cjk(char: str) -> bool:
    return any(low <= char <= high for low, high in _CJK_RANGES)


def _ends_with_letter_hyphen(text: str) -> bool:
    return len(text) >= 2 and text[-1] == "-" and text[-2].isalpha()


def join_lines(prev: str, nxt: str, break_hyphen: str = "-") -> str:
    """
    Join two consecutive lines (or paragraph fragments) of running text.

    Rules, in order: a trailing soft hyphen is removed; a letter + hyphen at
    the line end followed by a letter is joined without a space and written
    as break_hyphen ("well-" + "known" -> "well-known" with the default "-");
    CJK text is joined without a space; anything else is joined with a single
    space.

    Two lines alone cannot tell a real compound from a word split by the
    typesetter, so the extractor passes a private marker as break_hyphen and
    decides once the whole document is known (_resolve_break_hyphens): the
    hyphen is dropped unless the document shows the hyphenated spelling.
    """
    nxt_stripped = nxt.lstrip()
    if prev.endswith(_SOFT_HYPHEN):
        return prev[:-1] + nxt_stripped
    if _ends_with_letter_hyphen(prev) and nxt_stripped[:1].isalpha():
        return prev[:-1] + break_hyphen + nxt_stripped
    prev_stripped = prev.rstrip()
    if prev_stripped and nxt_stripped and _is_cjk(prev_stripped[-1]) and _is_cjk(nxt_stripped[0]):
        return prev_stripped + nxt_stripped
    return prev_stripped + " " + nxt_stripped


def _normalize_text(text: str) -> str:
    """Expand ligatures, drop soft hyphens and collapse all whitespace (incl. newlines)."""
    for ligature, expansion in LIGATURES.items():
        text = text.replace(ligature, expansion)
    text = text.replace(_SOFT_HYPHEN, "")
    return re.sub(r"\s+", " ", text).strip()


def _join_and_normalize(lines: List[str]) -> str:
    if not lines:
        return ""
    text = lines[0]
    for line in lines[1:]:
        text = join_lines(text, line, _BREAK_HYPHEN)
    return _normalize_text(text)


def _resolve_break_hyphens(texts: List[str]) -> List[str]:
    """
    Decide, for every hyphen found at a line end, whether to keep it.

    Typeset PDFs hyphenate words to justify lines, so a line-end hyphen
    followed by a lowercase letter is dropped by default ("condi-" + "tions"
    -> "conditions"). It is kept only when the document itself shows a real
    compound: the hyphenated spelling occurs elsewhere in the document, not at
    a line end ("well-known", "peut-être"). A continuation starting with an
    uppercase letter ("Jean-" + "Pierre"), a caseless script or a digit always
    keeps the hyphen. No dictionary is consulted: the decision rests on the
    document only. Soft hyphens (U+00AD) are handled by join_lines.
    """
    if not any(_BREAK_HYPHEN in text for text in texts):
        return texts
    hyphenated_words = set()
    for text in texts:
        for word in _WORD_RE.findall(_MARKED_TOKEN_RE.sub(" ", text)):
            if "-" in word:
                hyphenated_words.add(word.lower())

    def decide(match: "re.Match") -> str:
        left, right = match.group(1), match.group(2)
        hyphenated = f"{left}-{right}"
        if right[:1].islower() and hyphenated.lower() not in hyphenated_words:
            return left + right
        return hyphenated

    resolved = []
    for text in texts:
        if _BREAK_HYPHEN in text:
            text = _BREAK_WORD_RE.sub(decide, text).replace(_BREAK_HYPHEN, "-")
        resolved.append(text)
    return resolved


def _line_text(line: dict) -> str:
    return "".join(span.get("text", "") for span in line.get("spans", ()))


def _non_blank_spans(lines: List[dict]) -> List[dict]:
    return [
        span
        for line in lines
        for span in line.get("spans", ())
        if span.get("text", "").strip()
    ]


def _non_whitespace_count(text: str) -> int:
    return sum(1 for char in text if not char.isspace())


# --- Geometry helpers --------------------------------------------------------


def _union_bbox(dicts: List[dict]) -> Optional[BBox]:
    """Union of the "bbox" of the given line/span dicts (dicts without a usable bbox are ignored)."""
    boxes = [box for box in (_as_bbox(d.get("bbox")) for d in dicts) if box is not None]
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    )


def _centre_in(box: Optional[BBox], rect: BBox, tolerance: float) -> bool:
    """Whether the centre of box lies inside rect, grown by tolerance."""
    if box is None:
        return False
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return (
        rect[0] - tolerance <= cx <= rect[2] + tolerance
        and rect[1] - tolerance <= cy <= rect[3] + tolerance
    )


# --- Char-weighted statistics -------------------------------------------------


def _char_totals(spans: List[dict], value_of: Callable[[dict], Optional[Hashable]]) -> Dict[Hashable, int]:
    """Stripped character count per span value (spans whose value is None are skipped)."""
    totals: Dict[Hashable, int] = {}
    for span in spans:
        value = value_of(span)
        if value is not None:
            totals[value] = totals.get(value, 0) + len(span.get("text", "").strip())
    return totals


def _dominant(totals: Dict[Hashable, int], tie_rank: Callable[[Hashable], object]):
    """The value with the largest total; ties broken by the smallest tie_rank. None when empty."""
    if not totals:
        return None
    return min(totals, key=lambda value: (-totals[value], tie_rank(value)))


def _merge_totals(target: Dict[Hashable, int], totals: Dict[Hashable, int]) -> None:
    for value, count in totals.items():
        target[value] = target.get(value, 0) + count


def _span_size(span: dict) -> float:
    """Span size rounded to 0.5pt."""
    return round(float(span.get("size", 0.0)) * 2) / 2


def _span_color(span: dict) -> str:
    # Fixed-width lowercase hex: lexicographic order equals integer order, so a
    # colour tie broken on the string picks the smaller int.
    return color_int_to_hex(span.get("color", 0))


def _dominant_size(spans: List[dict]) -> float:
    """Char-weighted dominant span size, rounded to 0.5pt (ties -> the smaller size)."""
    size = _dominant(_char_totals(spans, _span_size), lambda value: value)
    return 0.0 if size is None else size


def _dominant_color(spans: List[dict]) -> Tuple[Optional[str], float]:
    """Char-weighted dominant colour (ties -> the smaller int) and its share of the characters."""
    totals = _char_totals(spans, _span_color)
    color = _dominant(totals, lambda value: value)
    total = sum(totals.values())
    share = totals[color] / total if color is not None and total > 0 else 0.0
    return color, share


# --- Collection --------------------------------------------------------------


@dataclass
class _TextBlock:
    """A text block (or a sub-block after the split step) with its page context."""
    page_index: int
    page_height: float
    bbox: BBox
    line_dicts: List[dict]   # PyMuPDF line dicts of the non-blank lines
    lines: List[str]         # raw line texts, parallel to line_dicts
    text: str                # normalised block text


@dataclass
class _ImageBlock:
    page_index: int
    block: dict


@dataclass
class _TableItem:
    """A detected table, with the text lines taken out of the page's blocks for it."""
    page_index: int
    table: DetectedTable
    table_lines: List[dict]


_Item = Union[_TextBlock, _ImageBlock, _TableItem]


def _text_block(page_index: int, page_height: float, bbox: BBox, line_dicts: List[dict]) -> Optional[_TextBlock]:
    """Build a text block from its line dicts (blank lines dropped); None when it has no text."""
    pairs = [(line, _line_text(line)) for line in line_dicts]
    pairs = [(line, text) for line, text in pairs if text.strip()]
    text = _join_and_normalize([line_text for _, line_text in pairs])
    if not text:
        return None
    return _TextBlock(
        page_index, page_height, bbox,
        [line for line, _ in pairs], [line_text for _, line_text in pairs], text,
    )


def _detect_layout(page, blocks: List[dict]) -> Tuple[List[DetectedTable], List[DetectedBox]]:
    """Tables and background boxes of a page (step 2, items 1-2)."""
    line_bboxes = [
        box
        for block in blocks if block.get("type") == 0
        for line in block.get("lines", ()) if _line_text(line).strip()
        for box in (_as_bbox(line.get("bbox")),) if box is not None
    ]
    tables = detect_tables(page, line_bboxes)
    boxes = detect_boxes(page, [table.bbox for table in tables])
    return tables, boxes


def _owning_table(line: dict, tables: List[DetectedTable]) -> Optional[int]:
    """Index of the first table whose bbox holds the line's centre, or None."""
    box = _as_bbox(line.get("bbox"))
    for index, table in enumerate(tables):
        if _centre_in(box, table.bbox, TABLE_LINE_TOLERANCE):
            return index
    return None


def _item_y0(item: Union[_TextBlock, _ImageBlock]) -> Optional[float]:
    box = item.bbox if isinstance(item, _TextBlock) else _as_bbox(item.block.get("bbox"))
    return None if box is None else box[1]


def _page_items(
    page_index: int,
    page_height: float,
    blocks: List[dict],
    tables: List[DetectedTable],
    include_images: bool,
) -> List[_Item]:
    """One page's items in stream order, table lines moved into table items (step 2, items 3-5)."""
    items: List[Union[_TextBlock, _ImageBlock]] = []
    table_lines: List[List[dict]] = [[] for _ in tables]
    anchors: Dict[int, int] = {}
    for block in blocks:
        block_type = block.get("type")
        if block_type == 0:
            kept = []
            for line in block.get("lines", ()):
                owner = _owning_table(line, tables)
                if owner is None:
                    kept.append(line)
                else:
                    table_lines[owner].append(line)
                    anchors.setdefault(owner, len(items))
            lost = len(kept) != len(block.get("lines", ()))
            bbox = (None if lost else _as_bbox(block.get("bbox"))) or _union_bbox(kept) or (0.0, 0.0, 0.0, 0.0)
            item = _text_block(page_index, page_height, bbox, kept)
            if item is not None:
                items.append(item)
        elif block_type == 1 and include_images:
            items.append(_ImageBlock(page_index, block))

    # A table that took no line goes before the first item below it, else at the page end
    item_tops = [_item_y0(item) for item in items]
    for index, table in enumerate(tables):
        if index not in anchors:
            anchors[index] = next(
                (position for position, y0 in enumerate(item_tops) if y0 is not None and y0 >= table.bbox[3]),
                len(items),
            )

    result: List[_Item] = []
    for position in range(len(items) + 1):
        # Tables come from detect_tables sorted by (y0, x0); keep that order on a shared anchor
        result.extend(
            _TableItem(page_index, table, table_lines[index])
            for index, table in enumerate(tables) if anchors[index] == position
        )
        if position < len(items):
            result.append(items[position])
    return result


def _collect_items(
    doc, include_images: bool, detect_layout: bool,
) -> Tuple[List[_Item], Dict[int, List[DetectedBox]]]:
    """Read every page's blocks in content-stream order, with tables and boxes (steps 2 and 3)."""
    items: List[_Item] = []
    page_boxes: Dict[int, List[DetectedBox]] = {}
    for page_index in range(doc.page_count):
        page = doc[page_index]
        blocks = page.get_text("dict", flags=TEXT_FLAGS, sort=False).get("blocks", [])
        tables, boxes = _detect_layout(page, blocks) if detect_layout else ([], [])
        if boxes:
            page_boxes[page_index] = boxes
        items.extend(_page_items(page_index, page.rect.height, blocks, tables, include_images))
    return items, page_boxes


# --- Header / footer removal -------------------------------------------------


def _is_in_band(item: _TextBlock) -> bool:
    _, y0, _, y1 = item.bbox
    band = HEADER_FOOTER_BAND * item.page_height
    return y1 <= band or y0 >= item.page_height - band


def _signature(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", text.lower())).strip()


def _repeat_threshold(page_count: int) -> int:
    return max(REPEAT_MIN_PAGES, math.ceil(REPEAT_RATIO * page_count))


def _drop_headers_footers(items: list, page_count: int) -> list:
    """Drop page numbers and running headers/footers found in the top/bottom bands (step 4)."""
    pages_by_signature: Dict[str, set] = {}
    in_band = set()
    for position, item in enumerate(items):
        if isinstance(item, _TextBlock) and _is_in_band(item):
            in_band.add(position)
            pages_by_signature.setdefault(_signature(item.text), set()).add(item.page_index)

    threshold = _repeat_threshold(page_count)
    kept = []
    for position, item in enumerate(items):
        if position in in_band:
            if PAGE_NUMBER_RE.fullmatch(item.text):
                continue
            pages = pages_by_signature.get(_signature(item.text), ())
            if page_count >= REPEAT_MIN_PAGES and len(pages) >= threshold:
                continue
        kept.append(item)
    return kept


# --- Block splitting ---------------------------------------------------------


def _starts_new_block(prev_line: dict, line: dict) -> bool:
    """Whether line opens a new sub-block after prev_line: size jump or uniform colour change."""
    prev_spans, spans = _non_blank_spans([prev_line]), _non_blank_spans([line])
    prev_size, size = _dominant_size(prev_spans), _dominant_size(spans)
    if prev_size > 0 and size > 0 and max(prev_size, size) / min(prev_size, size) >= SIZE_SPLIT_RATIO:
        return True
    prev_color, prev_share = _dominant_color(prev_spans)
    color, share = _dominant_color(spans)
    return prev_color != color and prev_share >= COLOR_SPLIT_SHARE and share >= COLOR_SPLIT_SHARE


def _split_block(item: _TextBlock) -> List[_TextBlock]:
    """Split a block where its lines change font size or colour (step 4b)."""
    groups: List[List[dict]] = [item.line_dicts[:1]]
    for prev_line, line in zip(item.line_dicts, item.line_dicts[1:]):
        if _starts_new_block(prev_line, line):
            groups.append([])
        groups[-1].append(line)
    if len(groups) == 1:
        return [item]
    blocks = (
        _text_block(item.page_index, item.page_height, _union_bbox(group) or item.bbox, group)
        for group in groups
    )
    return [block for block in blocks if block is not None]


def _split_items(items: List[_Item]) -> List[_Item]:
    result: List[_Item] = []
    for item in items:
        result.extend(_split_block(item) if isinstance(item, _TextBlock) else [item])
    return result


# --- Size statistics and style -----------------------------------------------


def _check_text_layer(items: List[_Item], page_count: int) -> None:
    """Reject documents without enough text, table cells included (step 5)."""
    char_count = sum(_non_whitespace_count(item.text) for item in items if isinstance(item, _TextBlock))
    char_count += sum(
        _non_whitespace_count(cell)
        for item in items if isinstance(item, _TableItem)
        for row in item.table.cells
        for cell in row if isinstance(cell, str)
    )
    if char_count < MIN_CHARS_PER_PAGE * page_count:
        raise PdfNoTextLayerError(_NO_TEXT_MESSAGE)


def _span_is_bold(span: dict, resolver: FontResolver) -> bool:
    return is_bold(resolver.resolve(span.get("font", "")), span.get("flags", 0))


def _span_is_italic(span: dict, resolver: FontResolver) -> bool:
    return is_italic(resolver.resolve(span.get("font", "")), span.get("flags", 0))


def _block_style(item: _TextBlock, body_size: float, resolver: FontResolver) -> str:
    """Classify a block as heading1..3 or normal (step 6)."""
    spans = _non_blank_spans(item.line_dicts)
    block_size = _dominant_size(spans)
    ratio = block_size / body_size if body_size > 0 else 1.0
    text = item.text

    if len(text) <= HEADING_MAX_CHARS:
        if ratio >= H1_RATIO:
            return "heading1"
        if ratio >= H2_RATIO:
            return "heading2"
        if ratio >= H3_RATIO:
            return "heading3"

    all_bold = bool(spans) and all(_span_is_bold(span, resolver) for span in spans)
    if (
        all_bold
        and len(text) <= BOLD_HEADING_MAX_CHARS
        and len(item.lines) <= 2
        and block_size >= body_size - 0.5
        and text[-1:] not in (".", ",", ";", ":")
    ):
        return "heading3"
    return "normal"


# --- Paragraph formats -------------------------------------------------------


class _Formatter:
    """Computes paragraph formats (step 6b), registers their boxes and the body font families."""

    def __init__(self, resolver: FontResolver, page_boxes: Dict[int, List[DetectedBox]]):
        self.resolver = resolver
        self.page_boxes = page_boxes
        self.boxes: List[PdfBox] = []
        self._box_indices: Dict[tuple, int] = {}
        self._body_families: Dict[Hashable, int] = {}

    def _register(self, key: tuple, box: PdfBox) -> int:
        """Index of the box in self.boxes, appended on first use of key."""
        if key not in self._box_indices:
            self._box_indices[key] = len(self.boxes)
            self.boxes.append(box)
        return self._box_indices[key]

    def _span_family(self, span: dict) -> Optional[str]:
        font = span.get("font", "")
        # A Type3 span carries meaningless flags; its resolved name still classifies by tokens
        return classify_family(self.resolver.resolve(font), span.get("flags", 0),
                               use_flags=not self.resolver.is_type3(font))

    def _format(self, spans: List[dict], box: Optional[int]) -> ParagraphFormat:
        color, _ = _dominant_color(spans)
        if color is not None and is_near_black(color):
            color = None
        background = self.boxes[box].background if box is not None else None
        if color is not None and is_light(color) and (background is None or is_light(background)):
            color = None
        family = _dominant(_char_totals(spans, self._span_family), FONT_FAMILIES.index)
        return ParagraphFormat(
            color=color,
            font_family=family,
            bold=bool(spans) and all(_span_is_bold(span, self.resolver) for span in spans),
            italic=bool(spans) and all(_span_is_italic(span, self.resolver) for span in spans),
            box=box,
        )

    def paragraph(self, page_index: int, line_dicts: List[dict], style: str) -> ParagraphFormat:
        """Format of a paragraph built from line_dicts; its box is the smallest one holding it."""
        spans = _non_blank_spans(line_dicts)
        if style in _BODY_STYLES:
            _merge_totals(self._body_families, _char_totals(spans, self._span_family))
        area = _union_bbox(line_dicts)
        detected = next(
            (
                box for box in self.page_boxes.get(page_index, ())
                if area is not None and _contained(area, box.rect, CONTAIN_TOLERANCE)
            ),
            None,
        )
        box = None
        if detected is not None:
            box = self._register(
                ("box", page_index, detected.rect), PdfBox(detected.background, detected.border_left),
            )
        return self._format(spans, box)

    def cell(self, page_index: int, table: DetectedTable, spans: List[dict], header: bool) -> ParagraphFormat:
        """Format of a table cell; header cells sit in the table's header fill box."""
        box = None
        if header and table.header_fill:
            box = self._register(("header", page_index, table.bbox), PdfBox(background=table.header_fill))
        return self._format(spans, box)

    def default_family(self) -> str:
        """Dominant family over the normal and list paragraphs, 'serif' when none is known."""
        return _dominant(self._body_families, FONT_FAMILIES.index) or "serif"


# --- List splitting and merging ----------------------------------------------


def _split_list_items(lines: List[str]) -> List[Tuple[str, str, List[int]]]:
    """Split a normal block at lines starting with a list marker (step 7).

    Returns (text, style, line indices) per paragraph.
    """
    groups: List[List[int]] = []
    for index, line in enumerate(lines):
        if index == 0 or LIST_MARKER_RE.match(line):
            groups.append([index])
        else:
            groups[-1].append(index)

    paragraphs = []
    for group in groups:
        text = _join_and_normalize([lines[index] for index in group])
        if text:
            style = "list" if LIST_MARKER_RE.match(text) else "normal"
            paragraphs.append((text, style, group))
    return paragraphs


# Scripts without case and without sentence-final punctuation (Thai, Lao):
# nothing tells a continuation from a new paragraph, so they never merge.
_NO_PUNCT_SCRIPTS = (("\u0e00", "\u0eff"),)


def _continues_sentence(first: str) -> bool:
    """Whether a paragraph starting with `first` may continue the previous one."""
    if first.islower():
        return True
    # A letter without case (CJK, Arabic, Hebrew, Devanagari...): rely on the
    # previous paragraph's punctuation alone.
    return (
        first.isalpha()
        and not first.isupper()
        and not any(low <= first <= high for low, high in _NO_PUNCT_SCRIPTS)
    )


def _should_merge(prev_text: str, prev_style: str, text: str) -> bool:
    """Whether a new normal paragraph continues the previous one (step 8)."""
    # A trailing "<letter>-" is not terminal punctuation, so hyphenated page
    # breaks are covered by the same rule. Headings and list items never merge.
    if prev_style != "normal" or not _continues_sentence(text[:1]):
        return False
    return TERMINAL_PUNCT_RE.search(prev_text) is None


class _ParagraphSink:
    """Accumulates paragraphs and their formats, applying the cross-block / cross-page merge."""

    def __init__(self):
        self.texts: List[str] = []
        self.styles: List[str] = []
        self.formats: List[ParagraphFormat] = []

    def append(self, text: str, style: str, fmt: ParagraphFormat) -> None:
        if not text:
            return
        if (
            style == "normal"
            and self.texts
            and self.formats[-1] == fmt
            and _should_merge(self.texts[-1], self.styles[-1], text)
        ):
            self.texts[-1] = _normalize_text(join_lines(self.texts[-1], text, _BREAK_HYPHEN))
            return
        self._add(text, style, fmt)

    def append_cell(self, text: str, fmt: ParagraphFormat) -> int:
        """Append a table cell (never merged) and return its index."""
        self._add(text, "cell", fmt)
        return len(self.texts) - 1

    def _add(self, text: str, style: str, fmt: ParagraphFormat) -> None:
        self.texts.append(text)
        self.styles.append(style)
        self.formats.append(fmt)


# --- Paragraph and table emission ----------------------------------------------


def _emit_text_block(item: _TextBlock, body_size: float, sink: _ParagraphSink, formatter: _Formatter) -> None:
    """Style, format, list-split and append one text sub-block (steps 6 to 8)."""
    style = _block_style(item, body_size, formatter.resolver)
    if style == "normal":
        for text, paragraph_style, group in _split_list_items(item.lines):
            line_dicts = [item.line_dicts[index] for index in group]
            sink.append(text, paragraph_style, formatter.paragraph(item.page_index, line_dicts, paragraph_style))
    else:
        sink.append(item.text, style, formatter.paragraph(item.page_index, item.line_dicts, style))


def _cell_text(raw) -> str:
    lines = raw.split("\n") if isinstance(raw, str) else []
    return _join_and_normalize([line for line in lines if line.strip()])


def _cell_bbox(table: DetectedTable, row: int, column: int) -> Optional[BBox]:
    try:
        return table.cell_bboxes[row][column]
    except IndexError:
        return None


def _emit_table(item: _TableItem, sink: _ParagraphSink, formatter: _Formatter) -> Optional[PdfTable]:
    """Append a table's non-empty cells as 'cell' units (row-major); None when every cell is empty."""
    table = item.table
    spans = _non_blank_spans(item.table_lines)
    first_index = len(sink.texts)
    rows: List[List[Optional[int]]] = []
    for row_index, row in enumerate(table.cells):
        indices: List[Optional[int]] = []
        for column_index, raw in enumerate(row):
            text = _cell_text(raw)
            if not text:
                indices.append(None)
                continue
            cell_box = _cell_bbox(table, row_index, column_index)
            cell_spans = [
                span for span in spans
                if cell_box is not None and _centre_in(_as_bbox(span.get("bbox")), cell_box, 0.0)
            ]
            fmt = formatter.cell(item.page_index, table, cell_spans, table.header and row_index == 0)
            indices.append(sink.append_cell(text, fmt))
        rows.append(indices)
    if len(sink.texts) == first_index:
        return None

    # Keep the PdfTable shape invariants even on an irregular grid
    width = max(len(row) for row in rows)
    rows = [row + [None] * (width - len(row)) for row in rows]
    widths = list(table.column_widths)
    if len(widths) != width:
        widths = [1.0 / width] * width
    return PdfTable(first_index=first_index, rows=rows, header=table.header, column_widths=widths)


# --- Images ------------------------------------------------------------------


def _image_hash(block: dict) -> Optional[str]:
    data = block.get("image")
    if not isinstance(data, (bytes, bytearray)) or not data:
        return None
    return hashlib.sha1(data).hexdigest()


def _count_image_pages(items: list) -> Dict[str, set]:
    """First pass: the distinct pages on which each image hash appears."""
    pages_by_hash: Dict[str, set] = {}
    for item in items:
        if isinstance(item, _ImageBlock):
            digest = _image_hash(item.block)
            if digest:
                pages_by_hash.setdefault(digest, set()).add(item.page_index)
    return pages_by_hash


def _to_pdf_image(block: dict) -> Optional[PdfImage]:
    """Normalise an image block to png/jpeg bytes; None if it cannot be converted."""
    x0, y0, x1, y1 = block["bbox"]
    data = bytes(block["image"])
    ext = str(block.get("ext", "")).lower()
    if ext == "jpg":
        ext = "jpeg"
    if ext not in ("png", "jpeg"):
        try:
            pix = pymupdf.Pixmap(data)
            if pix.alpha or pix.n - pix.alpha > 3:
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            data = pix.tobytes("png")
            ext = "png"
        except Exception as exc:
            logger.debug("Dropping a PDF image that could not be converted: %s", exc)
            return None
    return PdfImage(data=data, ext=ext, width_pt=x1 - x0, height_pt=y1 - y0)


def _keep_image(block: dict, pages_by_hash: Dict[str, set], threshold: int) -> bool:
    """Skip tiny images and images repeated on many pages (logos, backgrounds)."""
    x0, y0, x1, y1 = block["bbox"]
    if (x1 - x0) < MIN_IMAGE_SIDE_PT or (y1 - y0) < MIN_IMAGE_SIDE_PT:
        return False
    digest = _image_hash(block)
    if digest is None:
        return False
    return len(pages_by_hash.get(digest, ())) < threshold


# --- Public API --------------------------------------------------------------


def _open_document(source: Union[str, bytes]):
    """Open a PDF from a path or bytes, mapping failures to PdfOpenError (step 1)."""
    try:
        if isinstance(source, (bytes, bytearray)):
            doc = pymupdf.open(stream=bytes(source), filetype="pdf")
        else:
            doc = pymupdf.open(source)
    except Exception as exc:
        raise PdfOpenError(str(exc)) from exc
    return doc


def _extract(doc, include_images: bool, detect_layout: bool = True) -> PdfPlainContent:
    """Run steps 1 (checks) to 11 on an open document."""
    # A path is opened by extension, so a non-PDF document could get this far
    if not doc.is_pdf:
        raise PdfOpenError("The file is not a PDF.")
    if doc.needs_pass:
        raise PdfEncryptedError(_ENCRYPTED_MESSAGE)
    page_count = doc.page_count
    if page_count == 0:
        raise PdfOpenError("The PDF has no pages.")

    items, page_boxes = _collect_items(doc, include_images, detect_layout)    # steps 2-3
    items = _drop_headers_footers(items, page_count)                         # step 4
    items = _split_items(items)                                              # step 4b
    _check_text_layer(items, page_count)                                     # step 5
    body_size = _dominant_size(
        [span for item in items if isinstance(item, _TextBlock) for span in _non_blank_spans(item.line_dicts)]
    )

    pages_by_hash = _count_image_pages(items)
    image_threshold = _repeat_threshold(page_count)
    formatter = _Formatter(FontResolver(doc), page_boxes)
    sink = _ParagraphSink()
    tables: List[PdfTable] = []
    images_by_paragraph: Dict[int, List[PdfImage]] = {}

    for item in items:                                                       # steps 6-9
        if isinstance(item, _ImageBlock):
            if not _keep_image(item.block, pages_by_hash, image_threshold):
                continue
            image = _to_pdf_image(item.block)
            if image is not None:
                images_by_paragraph.setdefault(len(sink.texts) - 1, []).append(image)
        elif isinstance(item, _TableItem):
            table = _emit_table(item, sink, formatter)
            if table is not None:
                tables.append(table)
        else:
            _emit_text_block(item, body_size, sink, formatter)

    logger.debug(
        "PDF extracted: %d paragraphs, %d tables, %d boxes",
        len(sink.texts), len(tables), len(formatter.boxes),
    )
    first_page = doc[0].rect
    metadata = doc.metadata or {}
    return PdfPlainContent(                                                  # step 10
        paragraphs_text=_resolve_break_hyphens(sink.texts),
        paragraphs_style=sink.styles,
        images_by_paragraph=images_by_paragraph,
        page_size=(first_page.width, first_page.height),
        title=metadata.get("title") or "",
        author=metadata.get("author") or "",
        page_count=page_count,
        paragraphs_format=sink.formats,
        boxes=formatter.boxes,
        tables=tables,
        default_font_family=formatter.default_family(),
    )


def extract_pdf_paragraphs(
    source: Union[str, bytes], *, include_images: bool = True, detect_layout: bool = True,
) -> PdfPlainContent:
    """
    Extract ordered paragraphs, styles, formats, tables, boxes and anchored images from a PDF.

    Args:
        source: Path to the PDF, or its raw bytes.
        include_images: Collect images (skip them for text-only consumers).
        detect_layout: Detect ruled tables and background boxes. When False, table
            text stays in ordinary paragraphs, no box is recorded and tables is empty.

    Returns:
        PdfPlainContent with parallel paragraphs_text / paragraphs_style /
        paragraphs_format lists. No paragraph is empty or contains a newline.

    Raises:
        PdfOpenError: The input is not a readable PDF.
        PdfEncryptedError: The PDF requires a password.
        PdfNoTextLayerError: The PDF has no extractable text (scanned document).
    """
    doc = _open_document(source)
    try:
        return _extract(doc, include_images, detect_layout)
    except PdfError:
        raise
    except Exception as exc:
        raise PdfOpenError(f"Could not read the PDF: {exc}") from exc
    finally:
        doc.close()


def extract_pdf_text(source: Union[str, bytes], hard_cap: Optional[int] = None) -> str:
    """
    Return the PDF's paragraphs joined by blank lines, or "" on any failure.

    Never raises: meant for auxiliary features (cost estimate, samples,
    language detection) that must degrade gracefully. Layout detection is
    skipped (tables come out as ordinary paragraphs).
    """
    try:
        content = extract_pdf_paragraphs(source, include_images=False, detect_layout=False)
    except Exception as exc:
        logger.debug("PDF text extraction failed: %s", exc)
        return ""
    text = "\n\n".join(content.paragraphs_text)
    if isinstance(hard_cap, int) and not isinstance(hard_cap, bool) and hard_cap > 0:
        text = text[:hard_cap]
    return text
