"""
Plain-text extraction and rebuild for Plain Text Mode (DOCX).

Reads paragraphs directly via python-docx (skipping mammoth + HTML round-trip),
collecting:
- the textual content of each paragraph (one string)
- the structural style of each paragraph ('heading{n}', 'list', 'normal', 'quote')
- inline images anchored to their parent paragraph index (preserved as bytes
  with their original dimensions)
- page metadata (size + margins) for the rebuilt document

The body is walked in document order, tables included: every paragraph of a
table cell becomes one entry of the flat paragraph list (so the translation
pipeline stays unchanged) and remembers which cell it came from. Nested tables
are flattened into the cell that contains them.

At rebuild time, a fresh Document() is created with the same page setup, then
each translated paragraph is added with the right style, either to the body or
back into its table cell (tables are recreated with the same grid and merged
cells). Anchored images are emitted in a follow-up paragraph of the same
container (no inline-image positioning is attempted — that would require
tracking the exact run offset).
"""
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from docx import Document
from docx.shared import Inches
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph


# --- Extraction --------------------------------------------------------------


@dataclass
class _ImageRef:
    """One embedded image extracted from a paragraph."""
    blob: bytes
    width_emu: Optional[int] = None
    height_emu: Optional[int] = None


# (table index, row, grid column) of the cell a paragraph belongs to.
CellLocation = Tuple[int, int, int]


@dataclass
class TableLayout:
    """Grid shape of one top-level table, enough to recreate it."""
    rows: int
    cols: int
    # Merged regions as inclusive (row0, col0, row1, col1) rectangles.
    merges: List[Tuple[int, int, int, int]] = field(default_factory=list)


@dataclass
class DocxPlainContent:
    """Everything we need to rebuild a plain-text-mode DOCX."""
    paragraphs_text: List[str] = field(default_factory=list)
    paragraphs_style: List[str] = field(default_factory=list)  # 'heading1'..'heading6', 'list', 'normal', 'quote'
    images_by_paragraph: Dict[int, List[_ImageRef]] = field(default_factory=dict)
    # Parallel to paragraphs_text: None for a body paragraph, else the cell.
    paragraphs_location: List[Optional[CellLocation]] = field(default_factory=list)
    tables: List[TableLayout] = field(default_factory=list)
    page_size: Optional[Dict[str, float]] = None
    margins: Optional[Dict[str, float]] = None


def _classify_paragraph_style(paragraph) -> str:
    """Map python-docx style name to one of our coarse buckets."""
    style_name = (paragraph.style.name if paragraph.style else "") or ""
    name = style_name.lower()
    if name.startswith("heading "):
        try:
            level = int(name.split(" ")[1])
            level = max(1, min(level, 6))
            return f"heading{level}"
        except (ValueError, IndexError):
            return "heading1"
    if name.startswith("title"):
        return "heading1"
    if "list" in name or "bullet" in name or "number" in name:
        return "list"
    if "quote" in name:
        return "quote"
    return "normal"


def _extract_inline_images(paragraph, doc: Document) -> List[_ImageRef]:
    """
    Collect inline images embedded in a paragraph, in document order.

    python-docx represents inline images as <w:drawing> elements containing a
    <a:blip r:embed="rId"> that references a relationship to the actual image
    part. We resolve those relationships to grab the raw bytes.
    """
    images: List[_ImageRef] = []
    rels = doc.part.rels

    # Find all blip elements with r:embed attributes within the paragraph
    blips = paragraph._element.findall(
        ".//" + qn("w:drawing") + "//" + qn("a:blip")
    )
    if not blips:
        blips = paragraph._element.findall(".//" + qn("a:blip"))

    for blip in blips:
        embed_id = blip.get(qn("r:embed"))
        if not embed_id or embed_id not in rels:
            continue
        rel = rels[embed_id]
        try:
            blob = rel.target_part.blob
        except AttributeError:
            continue
        # <wp:extent> is a child of the <wp:inline>/<wp:anchor> ancestor of
        # this blip (a sibling of <a:graphic>), never an ancestor of the blip.
        width = height = None
        extent = None
        container = blip.getparent()
        while container is not None and container.tag not in (
            qn("wp:inline"), qn("wp:anchor")
        ):
            container = container.getparent()
        if container is not None:
            extent = container.find(qn("wp:extent"))
        if extent is not None:
            try:
                width = int(extent.get("cx") or 0) or None
                height = int(extent.get("cy") or 0) or None
            except (TypeError, ValueError):
                width = height = None

        images.append(_ImageRef(blob=blob, width_emu=width, height_emu=height))

    return images


def extract_plain_paragraphs(docx_path: str) -> DocxPlainContent:
    """
    Read a DOCX file as plain paragraphs (no HTML conversion).

    Skips empty paragraphs but anchors any inline image to the preceding
    paragraph (or starts a synthetic empty paragraph if none exists yet).
    """
    doc = Document(docx_path)
    content = DocxPlainContent()

    # Page setup
    if doc.sections:
        section = doc.sections[0]
        content.page_size = {
            "width": section.page_width.inches if section.page_width else None,
            "height": section.page_height.inches if section.page_height else None,
        }
        content.margins = {
            "top": section.top_margin.inches if section.top_margin else None,
            "bottom": section.bottom_margin.inches if section.bottom_margin else None,
            "left": section.left_margin.inches if section.left_margin else None,
            "right": section.right_margin.inches if section.right_margin else None,
        }

    body = doc._body
    for child in _iter_block_elements(doc.element.body):
        if child.tag == qn("w:p"):
            _add_extracted_paragraph(content, Paragraph(child, body), doc, None)
        elif child.tag == qn("w:tbl"):
            _extract_table(content, child, doc)

    return content


def _iter_block_elements(container):
    """
    Yield the <w:p> and <w:tbl> children of a block container in document
    order, looking through content controls (<w:sdt>/<w:sdtContent>).
    """
    for child in container:
        if child.tag in (qn("w:p"), qn("w:tbl")):
            yield child
        elif child.tag == qn("w:sdt"):
            sdt_content = child.find(qn("w:sdtContent"))
            if sdt_content is not None:
                yield from _iter_block_elements(sdt_content)


def _iter_cell_paragraphs(container):
    """Yield every <w:p> of a cell in order, flattening nested tables."""
    for child in _iter_block_elements(container):
        if child.tag == qn("w:p"):
            yield child
        else:
            for row in child.findall(qn("w:tr")):
                for tc in row.findall(qn("w:tc")):
                    yield from _iter_cell_paragraphs(tc)


def _int_attr(element, child_tag: str, default: int) -> int:
    if element is None:
        return default
    node = element.find(child_tag)
    if node is None:
        return default
    try:
        return int(node.get(qn("w:val")))
    except (TypeError, ValueError):
        return default


def _extract_table(content: DocxPlainContent, tbl, doc: Document) -> None:
    """Record a top-level table's grid and extract its cells row by row."""
    table_index = len(content.tables)
    layout = TableLayout(rows=0, cols=0)
    content.tables.append(layout)
    body = doc._body

    grid = tbl.find(qn("w:tblGrid"))
    max_cols = len(grid.findall(qn("w:gridCol"))) if grid is not None else 0
    # Vertical merge regions still open, keyed by grid column.
    open_vmerge: Dict[int, int] = {}

    rows = tbl.findall(qn("w:tr"))
    layout.rows = len(rows)
    for row_index, tr in enumerate(rows):
        col = _int_attr(tr.find(qn("w:trPr")), qn("w:gridBefore"), 0)
        for tc in tr.findall(qn("w:tc")):
            tc_pr = tc.find(qn("w:tcPr"))
            span = max(1, _int_attr(tc_pr, qn("w:gridSpan"), 1))
            vmerge = tc_pr.find(qn("w:vMerge")) if tc_pr is not None else None
            vmerge_val = None
            if vmerge is not None:
                vmerge_val = vmerge.get(qn("w:val")) or "continue"

            if vmerge_val == "continue" and col in open_vmerge:
                # Continuation of a vertically merged cell: extend the region.
                region_index = open_vmerge[col]
                r0, c0, _r1, c1 = layout.merges[region_index]
                layout.merges[region_index] = (r0, c0, row_index, c1)
            else:
                open_vmerge.pop(col, None)
                if vmerge_val == "restart" or span > 1:
                    layout.merges.append((row_index, col, row_index, col + span - 1))
                    if vmerge_val == "restart":
                        open_vmerge[col] = len(layout.merges) - 1
                location = (table_index, row_index, col)
                for p_element in _iter_cell_paragraphs(tc):
                    _add_extracted_paragraph(
                        content, Paragraph(p_element, body), doc, location
                    )
            col += span
        max_cols = max(max_cols, col)

    layout.cols = max_cols
    # Single-cell "regions" (a vMerge restart never continued) are no-ops.
    layout.merges = [m for m in layout.merges if m[0] != m[2] or m[1] != m[3]]


def _add_extracted_paragraph(
    content: DocxPlainContent,
    paragraph,
    doc: Document,
    location: Optional[CellLocation],
) -> None:
    """Append one source paragraph (body or cell) to the extracted content."""
    text = paragraph.text or ""
    normalized = " ".join(text.split())
    images = _extract_inline_images(paragraph, doc)

    if not normalized and not images:
        return

    if not normalized and images:
        # Image-only paragraph: anchor to the previous block of the same
        # container (body or same cell), or create a synthetic empty one.
        if content.paragraphs_text and content.paragraphs_location[-1] == location:
            anchor = len(content.paragraphs_text) - 1
            content.images_by_paragraph.setdefault(anchor, []).extend(images)
        else:
            anchor = len(content.paragraphs_text)
            content.paragraphs_text.append("")
            content.paragraphs_style.append("normal")
            content.paragraphs_location.append(location)
            content.images_by_paragraph[anchor] = images
        return

    idx = len(content.paragraphs_text)
    content.paragraphs_text.append(normalized)
    content.paragraphs_style.append(_classify_paragraph_style(paragraph))
    content.paragraphs_location.append(location)
    if images:
        content.images_by_paragraph[idx] = images


# --- Rebuild -----------------------------------------------------------------


def _apply_page_metadata(doc: Document, content: DocxPlainContent) -> None:
    if not doc.sections:
        return
    section = doc.sections[0]
    ps = content.page_size or {}
    if ps.get("width") is not None:
        section.page_width = Inches(ps["width"])
    if ps.get("height") is not None:
        section.page_height = Inches(ps["height"])
    m = content.margins or {}
    if m.get("top") is not None:
        section.top_margin = Inches(m["top"])
    if m.get("bottom") is not None:
        section.bottom_margin = Inches(m["bottom"])
    if m.get("left") is not None:
        section.left_margin = Inches(m["left"])
    if m.get("right") is not None:
        section.right_margin = Inches(m["right"])


def _new_paragraph(container, used_cells: set):
    """
    Return a fresh paragraph in `container` (the document or a table cell).

    A newly created cell already holds one empty paragraph: reuse it for the
    first content written to that cell instead of leaving a blank line.
    """
    tc = getattr(container, "_tc", None)
    if tc is not None and id(tc) not in used_cells:
        used_cells.add(id(tc))
        paragraphs = container.paragraphs
        if len(paragraphs) == 1 and not paragraphs[0].text:
            return paragraphs[0]
    return container.add_paragraph()


def _set_style(paragraph, style_name: str) -> None:
    try:
        paragraph.style = style_name
    except (KeyError, ValueError):
        pass


def _add_styled_paragraph(container, text: str, style: str, used_cells: set):
    """Add a paragraph with the right python-docx style, return it."""
    paragraph = _new_paragraph(container, used_cells)
    if style.startswith("heading"):
        try:
            level = max(1, min(int(style.replace("heading", "")), 6))
            _set_style(paragraph, f"Heading {level}")
        except ValueError:
            pass
    elif style == "list":
        _set_style(paragraph, "List Bullet")
    elif style == "quote":
        _set_style(paragraph, "Intense Quote")
    paragraph.add_run(text)
    return paragraph


def _add_image(container, image: _ImageRef, used_cells: set) -> None:
    """Insert an image as its own paragraph (no inline-positioning)."""
    paragraph = _new_paragraph(container, used_cells)
    run = paragraph.add_run()
    try:
        if image.width_emu and image.height_emu:
            # EMU -> python-docx Emu shape via direct attribute would require
            # an extra import; we rebuild via Inches approximation (914400 EMU
            # per inch) so we stay on the public API.
            width_in = image.width_emu / 914400.0 if image.width_emu else None
            height_in = image.height_emu / 914400.0 if image.height_emu else None
            run.add_picture(
                io.BytesIO(image.blob),
                width=Inches(width_in) if width_in else None,
                height=Inches(height_in) if height_in else None,
            )
        else:
            run.add_picture(io.BytesIO(image.blob))
    except Exception:
        # Unsupported format or corrupted blob - drop silently rather than
        # blocking the whole rebuild.
        return


def _build_table(doc: Document, layout: TableLayout):
    """Create an empty table with the recorded grid and merged regions."""
    table = doc.add_table(rows=max(1, layout.rows), cols=max(1, layout.cols))
    try:
        table.style = "Table Grid"
    except (KeyError, ValueError):
        pass
    for r0, c0, r1, c1 in layout.merges:
        try:
            table.cell(r0, c0).merge(table.cell(r1, c1))
        except Exception:
            # An irregular grid python-docx cannot merge: keep the cells apart.
            continue
    return table


def _container_for(doc: Document, content: DocxPlainContent, index: int, built_tables: Dict[int, object]):
    """Return the document, or the rebuilt table cell, paragraph `index` goes to."""
    location = (
        content.paragraphs_location[index]
        if index < len(content.paragraphs_location)
        else None
    )
    if location is None:
        return doc
    table_index, row, col = location
    table = built_tables.get(table_index)
    if table is None:
        table = _build_table(doc, content.tables[table_index])
        built_tables[table_index] = table
    return table.cell(row, col)


def build_minimal_docx(
    translated_paragraphs: List[str],
    content: DocxPlainContent,
    output_path: str,
    bilingual: bool = False,
) -> None:
    """
    Generate a fresh DOCX from translated paragraphs.

    Args:
        translated_paragraphs: parallel to content.paragraphs_text (same length)
        content: the result of extract_plain_paragraphs() on the source DOCX
        output_path: where to save
        bilingual: when True, emit source text before each translated paragraph
    """
    doc = Document()
    _apply_page_metadata(doc, content)

    built_tables: Dict[int, object] = {}
    used_cells: set = set()

    count = len(translated_paragraphs)
    for i in range(count):
        text = (translated_paragraphs[i] or "").strip()
        style = (
            content.paragraphs_style[i]
            if i < len(content.paragraphs_style)
            else "normal"
        )
        container = _container_for(doc, content, i, built_tables)

        if bilingual and i < len(content.paragraphs_text):
            source_text = (content.paragraphs_text[i] or "").strip()
            if source_text:
                _add_styled_paragraph(container, source_text, style, used_cells)

        if text:
            _add_styled_paragraph(container, text, style, used_cells)

        # Images anchored to this paragraph
        for image in content.images_by_paragraph.get(i, []):
            _add_image(container, image, used_cells)

    # Attribution stamp: identical for every install, disabled by the
    # ATTRIBUTION_ENABLED switch. Left untouched when attribution is off.
    import src.config as cfg
    if cfg.ATTRIBUTION_ENABLED:
        doc.core_properties.last_modified_by = cfg.GENERATOR_NAME

    doc.save(output_path)
