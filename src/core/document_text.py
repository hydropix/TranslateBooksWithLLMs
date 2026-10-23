"""
Plain-text extraction for uploaded documents (TXT, EPUB, DOCX, PDF).

Shared by features that need a document's readable text rather than its
layout: Sample & Compare (extract sampling) and Ask the Book (retrieval index).
"""
from typing import List


def extract_plain_text(file_path: str, file_type: str) -> str:
    """
    Extract the textual content of a file as plain text.

    For TXT we read directly. For EPUB/DOCX we reuse the plain extractors used
    by Plain Text Mode in the main translate flow, and for PDF the shared
    text-layer extractor. SRT is handled by the
    caller (sampled at the cue-group level, not via this helper).
    """
    ft = file_type.lower()
    if ft == "txt":
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    if ft == "epub":
        return extract_epub_text(file_path)
    if ft == "docx":
        return extract_docx_text(file_path)
    if ft == "pdf":
        return extract_pdf_text_layer(file_path)
    raise ValueError(f"Unsupported file type for text extraction: {file_type}")


def extract_epub_text(file_path: str) -> str:
    """Concatenate the block-level text of every XHTML body in the EPUB.

    Reuses the main translate flow's `extract_plain_paragraphs`, which walks
    block elements once and returns one string per paragraph. (An earlier
    version iterated *every* element and joined `itertext()` per element, which
    re-emitted each paragraph once per ancestor — heavily duplicating content
    and corrupting the sampled extracts.)
    """
    import zipfile
    from lxml import etree

    from src.core.epub.plain_extractor import _local_name, extract_plain_paragraphs

    parts: List[str] = []
    try:
        with zipfile.ZipFile(file_path, "r") as zf:
            for name in zf.namelist():
                lower = name.lower()
                if not (lower.endswith(".xhtml") or lower.endswith(".html") or lower.endswith(".htm")):
                    continue
                try:
                    raw = zf.read(name)
                    root = etree.fromstring(raw)
                except Exception:
                    continue
                body = next(
                    (el for el in root.iter()
                     if isinstance(el.tag, str) and _local_name(el) == "body"),
                    root,
                )
                paragraphs, _tags, _images, _attrib = extract_plain_paragraphs(body)
                for text in paragraphs:
                    text = (text or "").strip()
                    if text:
                        parts.append(text)
                        parts.append("\n\n")
    except zipfile.BadZipFile:
        raise ValueError("Invalid EPUB file (not a zip archive)")
    return "".join(parts).strip()


def extract_docx_text(file_path: str) -> str:
    """Concatenate paragraph text from a DOCX using the plain extractor."""
    from docx import Document

    doc = Document(file_path)
    parts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n\n".join(parts)


def extract_pdf_text_layer(file_path: str) -> str:
    """Return the PDF's text layer ('' when unreadable or scanned)."""
    from src.core.pdf import extract_pdf_text

    return extract_pdf_text(file_path)
