"""EPUB archive I/O: extraction, package discovery, spine reading, repackaging.

Everything here works on the extracted book directory and knows nothing
about translation, so the refiner and the RTL pass can share it with the
translation pipeline.
"""

import os
import zipfile
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import unquote

import aiofiles
from lxml import etree

from src.config import NAMESPACES
from src.utils.security import safe_extract_zip


def extract_epub(input_filepath: str, temp_dir: str,
                 log_callback: Optional[Callable] = None) -> None:
    """Extract the EPUB archive into ``temp_dir``."""
    if log_callback:
        log_callback("epub_extract_start", "Extracting EPUB...")

    with zipfile.ZipFile(input_filepath, 'r') as zip_ref:
        safe_extract_zip(zip_ref, temp_dir)


def find_opf_file(temp_dir: str) -> Optional[str]:
    """Find the first OPF file in an extracted EPUB (directory walk order)."""
    for root_dir, _, files in os.walk(temp_dir):
        for file in files:
            if file.endswith('.opf'):
                return os.path.join(root_dir, file)
    return None


def resolve_content_path(opf_dir: str, content_href: str) -> str:
    """Resolve a manifest href to a filesystem path.

    EPUB hrefs are URLs: spaces and non-ASCII characters are commonly
    percent-encoded ("Chapter%201.xhtml"). They must be unquoted before
    joining, otherwise the file is reported missing and ships untranslated.
    """
    return os.path.normpath(os.path.join(opf_dir, unquote(content_href)))


def content_files_from_spine(spine: etree._Element, manifest: etree._Element) -> List[str]:
    """XHTML/HTML hrefs of the spine, in reading order."""
    content_files = []
    for itemref in spine.findall('.//opf:itemref', namespaces=NAMESPACES):
        idref = itemref.get('idref')
        item = manifest.find(f'.//opf:item[@id="{idref}"]', namespaces=NAMESPACES)
        if item is not None:
            media_type = item.get('media-type')
            href = item.get('href')
            if media_type in ['application/xhtml+xml', 'text/html'] and href:
                content_files.append(href)
    return content_files


def parse_epub_manifest(temp_dir: str, log_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Parse the OPF package of an extracted EPUB.

    Returns:
        Dictionary with keys: opf_path, opf_tree, opf_dir, content_files
    """
    opf_path = find_opf_file(temp_dir)
    if not opf_path:
        raise FileNotFoundError("CRITICAL ERROR: content.opf not found in EPUB.")

    opf_tree = etree.parse(opf_path)
    opf_root = opf_tree.getroot()
    opf_dir = os.path.dirname(opf_path)

    manifest = opf_root.find('.//opf:manifest', namespaces=NAMESPACES)
    spine = opf_root.find('.//opf:spine', namespaces=NAMESPACES)
    if manifest is None or spine is None:
        raise ValueError("CRITICAL ERROR: manifest or spine missing in EPUB.")

    content_files = content_files_from_spine(spine, manifest)

    if log_callback:
        log_callback("epub_files_found", f"Found {len(content_files)} content files to translate.")

    return {
        'opf_path': opf_path,
        'opf_tree': opf_tree,
        'opf_dir': opf_dir,
        'content_files': content_files
    }


def parse_xhtml(content: str) -> etree._Element:
    """Parse XHTML text the way every book-level reader does (recovering)."""
    parser = etree.XMLParser(encoding='utf-8', recover=True, remove_blank_text=False)
    return etree.fromstring(content.encode('utf-8'), parser)


async def read_text(file_path: str) -> str:
    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
        return await f.read()


def serialize_xhtml(doc_root: etree._Element) -> bytes:
    return etree.tostring(doc_root, encoding='utf-8', xml_declaration=True,
                          pretty_print=True, method='xml')


def repackage_epub(
    temp_dir: str,
    output_filepath: str,
    log_callback: Optional[Callable] = None,
) -> None:
    """Zip the extracted book back into an EPUB (``mimetype`` first, stored)."""
    with zipfile.ZipFile(output_filepath, 'w', zipfile.ZIP_DEFLATED) as epub_zip:
        mimetype_path = os.path.join(temp_dir, 'mimetype')
        if os.path.exists(mimetype_path):
            epub_zip.write(mimetype_path, 'mimetype', compress_type=zipfile.ZIP_STORED)

        for root_path, _, files in os.walk(temp_dir):
            for file_item in files:
                if file_item != 'mimetype':
                    file_path_abs = os.path.join(root_path, file_item)
                    arcname = os.path.relpath(file_path_abs, temp_dir)
                    epub_zip.write(file_path_abs, arcname)
