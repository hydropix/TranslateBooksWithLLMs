"""
Plain-text extraction and rebuild for Plain Text Mode (DOCX).

Reads paragraphs directly via python-docx (skipping mammoth + HTML round-trip),
collecting:
- the textual content of each paragraph (one string)
- the structural style of each paragraph ('heading{n}', 'list', 'normal', 'quote')
- inline images anchored to their parent paragraph index (preserved as bytes
  with their original dimensions)
- page metadata (size + margins) for the rebuilt document

At rebuild time, a fresh Document() is created with the same page setup, then
each translated paragraph is added with the right style. Anchored images are
emitted in a follow-up paragraph (no inline-image positioning is attempted —
that would require tracking the exact run offset).
"""
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from docx import Document
from docx.shared import Inches
from docx.oxml.ns import qn


# --- Extraction --------------------------------------------------------------


@dataclass
class _ImageRef:
    """One embedded image extracted from a paragraph."""
    blob: bytes
    width_emu: Optional[int] = None
    height_emu: Optional[int] = None


@dataclass
class DocxPlainContent:
    """Everything we need to rebuild a plain-text-mode DOCX."""
    paragraphs_text: List[str] = field(default_factory=list)
    paragraphs_style: List[str] = field(default_factory=list)  # 'heading1'..'heading6', 'list', 'normal', 'quote'
    images_by_paragraph: Dict[int, List[_ImageRef]] = field(default_factory=dict)
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

    for paragraph in doc.paragraphs:
        text = paragraph.text or ""
        normalized = " ".join(text.split())
        images = _extract_inline_images(paragraph, doc)

        if not normalized and not images:
            continue

        if not normalized and images:
            # Image-only paragraph: anchor to the previous block (or create one).
            if content.paragraphs_text:
                anchor = len(content.paragraphs_text) - 1
                content.images_by_paragraph.setdefault(anchor, []).extend(images)
            else:
                content.paragraphs_text.append("")
                content.paragraphs_style.append("normal")
                content.images_by_paragraph[0] = images
            continue

        idx = len(content.paragraphs_text)
        content.paragraphs_text.append(normalized)
        content.paragraphs_style.append(_classify_paragraph_style(paragraph))
        if images:
            content.images_by_paragraph[idx] = images

    return content


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


def _add_styled_paragraph(doc: Document, text: str, style: str):
    """Add a paragraph with the right python-docx style, return it."""
    if style.startswith("heading"):
        try:
            level = int(style.replace("heading", ""))
            level = max(1, min(level, 6))
            return doc.add_heading(text, level=level)
        except ValueError:
            return doc.add_paragraph(text)
    if style == "list":
        try:
            return doc.add_paragraph(text, style="List Bullet")
        except KeyError:
            return doc.add_paragraph(text)
    if style == "quote":
        try:
            return doc.add_paragraph(text, style="Intense Quote")
        except KeyError:
            return doc.add_paragraph(text)
    return doc.add_paragraph(text)


def _add_image(doc: Document, image: _ImageRef) -> None:
    """Insert an image as its own paragraph (no inline-positioning)."""
    paragraph = doc.add_paragraph()
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

    count = len(translated_paragraphs)
    for i in range(count):
        text = (translated_paragraphs[i] or "").strip()
        style = (
            content.paragraphs_style[i]
            if i < len(content.paragraphs_style)
            else "normal"
        )

        if bilingual and i < len(content.paragraphs_text):
            source_text = (content.paragraphs_text[i] or "").strip()
            if source_text:
                _add_styled_paragraph(doc, source_text, style)

        if text:
            _add_styled_paragraph(doc, text, style)

        # Images anchored to this paragraph
        for image in content.images_by_paragraph.get(i, []):
            _add_image(doc, image)

    # Attribution stamp: identical for every install, disabled by the
    # ATTRIBUTION_ENABLED switch. Left untouched when attribution is off.
    import src.config as cfg
    if cfg.ATTRIBUTION_ENABLED:
        doc.core_properties.last_modified_by = cfg.GENERATOR_NAME

    doc.save(output_path)
