"""Shared document text extraction and distributed sampling.

Pulls readable text out of uploaded documents (plain text, EPUB, DOCX, PDF) and
carves it into a bounded number of representative excerpts. Originally lived
in the glossary NER upload path; extracted here so other features (e.g.
style extraction) can reuse the same extraction/sampling logic.
"""

from __future__ import annotations

import io
import posixpath
import re
import zipfile
from pathlib import Path

from lxml import etree

TEXT_EXTS: frozenset[str] = frozenset({'.txt', '.srt', '.md', '.markdown'})
RICH_EXTS: frozenset[str] = frozenset({'.epub', '.docx', '.pdf'})
SUPPORTED_EXTS: frozenset[str] = TEXT_EXTS | RICH_EXTS

# Hard cap on text we pull from a single upload before sampling.
# 5M chars is more than the longest novels; protects memory on huge inputs.
FULL_TEXT_CAP: int = 5_000_000

# Visible separator inserted between non-contiguous excerpts. The LLM treats
# it as a discontinuity hint without us needing to change the prompt.
EXCERPT_SEPARATOR: str = '\n\n[…]\n\n'

_OPF_NS = {
    'opf': 'http://www.idpf.org/2007/opf',
    'container': 'urn:oasis:names:tc:opendocument:xmlns:container',
}


def _decode_text(data: bytes) -> str:
    try:
        return data.decode('utf-8-sig')
    except UnicodeDecodeError:
        return data.decode('utf-8', errors='replace')


def _extract_epub_full_text(file_data: bytes, hard_cap: int) -> str | None:
    """Pull readable text from an EPUB, following the spine when possible.

    Reads up to ``hard_cap`` characters total before stopping.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(file_data)) as zf:
            ordered_files: list[str] = []
            try:
                container = zf.read('META-INF/container.xml')
                container_tree = etree.fromstring(container)
                rootfile = container_tree.find('.//container:rootfile', _OPF_NS)
                opf_path = rootfile.get('full-path') if rootfile is not None else None
            except (KeyError, etree.XMLSyntaxError):
                opf_path = None

            if opf_path:
                try:
                    opf_tree = etree.fromstring(zf.read(opf_path))
                    opf_dir = posixpath.dirname(opf_path)
                    manifest = {}
                    for item in opf_tree.findall('.//opf:item', _OPF_NS):
                        media_type = item.get('media-type') or ''
                        href = item.get('href')
                        item_id = item.get('id')
                        if item_id and href and media_type in ('application/xhtml+xml', 'text/html'):
                            manifest[item_id] = href
                    for itemref in opf_tree.findall('.//opf:itemref', _OPF_NS):
                        href = manifest.get(itemref.get('idref'))
                        if not href:
                            continue
                        ordered_files.append(posixpath.join(opf_dir, href) if opf_dir else href)
                except (KeyError, etree.XMLSyntaxError):
                    ordered_files = []

            if not ordered_files:
                ordered_files = sorted(
                    n for n in zf.namelist()
                    if n.lower().endswith(('.xhtml', '.html', '.htm'))
                )

            parts: list[str] = []
            running = 0
            parser = etree.HTMLParser()
            for name in ordered_files:
                try:
                    content = zf.read(name)
                except KeyError:
                    continue
                try:
                    tree = etree.fromstring(content, parser)
                except etree.XMLSyntaxError:
                    continue
                if tree is None:
                    continue
                body = tree.find('.//body')
                if body is None:
                    continue
                text = etree.tostring(body, method='text', encoding='unicode')
                text = re.sub(r'\s+', ' ', text).strip()
                if not text:
                    continue
                parts.append(text)
                running += len(text)
                if running >= hard_cap:
                    break

            if not parts:
                return None
            return '\n\n'.join(parts)[:hard_cap]
    except (zipfile.BadZipFile, OSError):
        return None


def _extract_docx_full_text(file_data: bytes, hard_cap: int) -> str | None:
    """Pull paragraph text from a DOCX up to ``hard_cap`` characters."""
    try:
        from docx import Document
    except ImportError:
        return None
    try:
        doc = Document(io.BytesIO(file_data))
    except Exception:
        return None

    parts: list[str] = []
    running = 0
    for para in doc.paragraphs:
        text = (para.text or '').strip()
        if not text:
            continue
        parts.append(text)
        running += len(text)
        if running >= hard_cap:
            break
    if not parts:
        return None
    return '\n\n'.join(parts)[:hard_cap]


def _extract_pdf_full_text(file_data: bytes, hard_cap: int) -> str | None:
    """Pull the text layer from a PDF up to ``hard_cap`` characters.

    Returns None when the PDF has no usable text (e.g. a scanned document) or
    cannot be read. ``extract_pdf_text`` never raises.
    """
    from src.core.pdf import extract_pdf_text

    return extract_pdf_text(file_data, hard_cap=hard_cap) or None


def extract_full_text(file_data: bytes, filename: str, hard_cap: int = FULL_TEXT_CAP) -> str | None:
    """Extract the full readable text from an uploaded file (capped)."""
    if not filename or hard_cap <= 0:
        return None
    ext = Path(filename).suffix.lower()
    if ext in TEXT_EXTS:
        return _decode_text(file_data)[:hard_cap]
    if ext == '.epub':
        return _extract_epub_full_text(file_data, hard_cap)
    if ext == '.docx':
        return _extract_docx_full_text(file_data, hard_cap)
    if ext == '.pdf':
        return _extract_pdf_full_text(file_data, hard_cap)
    return None


def take_distributed_samples(
    text: str,
    total_budget: int,
    num_samples: int,
    min_sample_size: int = 500,
    separator: str = EXCERPT_SEPARATOR,
) -> tuple[str, int]:
    """Return ``(joined_excerpts, effective_sample_count)``.

    - Short texts (≤ budget) are returned untouched, count = 1.
    - ``num_samples`` is clamped so each excerpt is at least
      ``min_sample_size`` chars (otherwise NER quality collapses).
    - Sample edges snap to nearby whitespace to avoid cutting mid-word.
    """
    n = max(1, int(num_samples))
    text_len = len(text)

    if text_len <= total_budget:
        return text, 1
    if n == 1:
        return text[:total_budget], 1

    max_n_for_budget = max(1, total_budget // min_sample_size)
    if n > max_n_for_budget:
        n = max_n_for_budget
    if n <= 1:
        return text[:total_budget], 1

    sample_size = total_budget // n
    if sample_size * n >= text_len:
        return text[:total_budget], 1

    stride = (text_len - sample_size) / (n - 1)

    pieces: list[str] = []
    last_end = -1
    for i in range(n):
        start = int(round(i * stride))
        end = start + sample_size

        if start > 0 and start < text_len:
            ws = text.rfind(' ', max(0, start - 80), start + 1)
            if ws != -1:
                start = ws + 1
        if end < text_len:
            ws = text.find(' ', end, min(text_len, end + 80))
            if ws != -1:
                end = ws

        if start <= last_end:
            start = last_end + 1
        if start >= text_len or end <= start:
            continue

        chunk = text[start:end].strip()
        if chunk:
            pieces.append(chunk)
            last_end = end

    if not pieces:
        return text[:total_budget], 1
    return separator.join(pieces), len(pieces)


def extract_samples_from_upload(
    file_data: bytes,
    filename: str,
    max_chars: int,
    num_samples: int = 1,
    min_sample_size: int = 500,
) -> tuple[str | None, int, int]:
    """Extract distributed samples from an uploaded file.

    Returns ``(joined_text, effective_sample_count, full_text_chars)``.
    ``joined_text`` is None when the format is unsupported or extraction
    failed; ``full_text_chars`` is the size of the full extracted text
    (useful to tell the user how much of the document was searched).
    """
    if not filename or max_chars <= 0:
        return None, 0, 0
    full_text = extract_full_text(file_data, filename, FULL_TEXT_CAP)
    if not full_text:
        return None, 0, 0
    joined, effective_n = take_distributed_samples(
        full_text, max_chars, num_samples, min_sample_size=min_sample_size
    )
    return joined, effective_n, len(full_text)
