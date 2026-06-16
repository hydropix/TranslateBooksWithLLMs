"""
Body serialization for simplified EPUB processing

This module handles extracting and replacing the content of <body> elements
in XHTML documents, enabling full-document translation instead of per-element processing.
"""
from lxml import etree
from typing import Tuple, Optional
import re

from src.utils.unified_logger import info, LogType
from .boilerplate_filter import strip_web_boilerplate
from .exceptions import XmlParsingError, BodyExtractionError


def normalize_whitespace(html: str) -> str:
    """
    Normalize excessive whitespace in HTML content while preserving structural breaks.

    This handles the common case where EPUB source files have arbitrary line breaks
    and indentation that are just formatting artifacts, not meaningful content.

    For example, transforms:
        "La technologie est en constante évolution.
            Au fur et à mesure que les ordinateurs"
    Into:
        "La technologie est en constante évolution. Au fur et à mesure que les ordinateurs"

    But preserves structural breaks by injecting newlines after block-level closing tags.

    Rules:
    - Normalize whitespace only WITHIN text content (inside paragraphs, etc.)
    - Inject newlines after closing block tags (</p>, </li>, </div>, etc.)
    - Preserve content inside <pre>, <code>, <script>, <style> tags unchanged
    - Preserve <br> and <br/> tags (they represent intentional line breaks)

    Args:
        html: Raw HTML string with potential excessive whitespace

    Returns:
        HTML with normalized whitespace
    """
    # Protect content in preformatted tags by replacing with placeholders
    preserved_blocks = []

    def preserve_block(match):
        preserved_blocks.append(match.group(0))
        return f"__PRESERVED_BLOCK_{len(preserved_blocks) - 1}__"

    # Preserve <pre>, <code>, <script>, <style> blocks (case insensitive)
    html = re.sub(
        r'<(pre|code|script|style)[^>]*>.*?</\1>',
        preserve_block,
        html,
        flags=re.DOTALL | re.IGNORECASE
    )

    # Preserve <br> tags (they represent intentional line breaks)
    html = re.sub(r'<br\s*/?\s*>', preserve_block, html, flags=re.IGNORECASE)

    # Step 1: Normalize line endings
    html = html.replace('\r\n', '\n').replace('\r', '\n')

    # Step 2: Inject newlines after block-level closing tags
    # This ensures structural separation is maintained even if source has no newlines
    block_tags = r'</(?:p|div|li|h[1-6]|blockquote|section|article|header|footer|nav|aside|ol|ul|table|tr|td|th|dt|dd)>'
    html = re.sub(f'({block_tags})[ \t]*', r'\1\n', html, flags=re.IGNORECASE)

    # Step 3: Replace remaining single newlines (within text content) with a single space
    # But preserve newlines that are right after a closing tag (just added in step 2)
    # Pattern: newline NOT preceded by > (closing tag)
    html = re.sub(r'(?<!>)[ \t]*\n(?!\n)[ \t]*', ' ', html)

    # Step 4: Collapse multiple spaces into one
    html = re.sub(r' {2,}', ' ', html)

    # Step 5: Clean up whitespace around newlines
    html = re.sub(r' \n', '\n', html)  # Remove space before newline
    html = re.sub(r'\n ', '\n', html)  # Remove space after newline

    # Restore preserved blocks
    for i, block in enumerate(preserved_blocks):
        html = html.replace(f"__PRESERVED_BLOCK_{i}__", block)

    return html


def extract_body_html(
    doc_root: etree._Element,
    normalize: bool = True,
    strip_boilerplate: bool = True,
    log_callback=None,
) -> Tuple[str, Optional[etree._Element]]:
    """
    Extract the HTML content of <body> as a string.

    Args:
        doc_root: Root of the parsed XHTML document
        normalize: If True, normalize excessive whitespace (default: True)
        strip_boilerplate: If True, remove web-scraping boilerplate (social
            share bars, related-post widgets, prev/next nav, hidden elements)
            before serialization so it is never translated (issue #239). The
            EPUB3 TOC navigation is preserved.
        log_callback: Optional callback for logging removed-boilerplate counts

    Returns:
        Tuple (body_inner_html, body_element)
        Returns ("", None) if no body element found

    Note: when strip_boilerplate is True the body element is mutated in place;
    both translation paths replace the body content afterwards, so the removed
    elements simply never appear in the output.
    """
    # Try XHTML namespace first, then fallback to no namespace
    body = doc_root.find('.//{http://www.w3.org/1999/xhtml}body')
    if body is None:
        body = doc_root.find('.//body')

    if body is None:
        return "", None

    if strip_boilerplate:
        strip_web_boilerplate(body, log_callback=log_callback)

    # Serialize the inner content of body (without the <body> tag itself)
    inner_html = etree.tostring(body, encoding='unicode', method='html')

    # Remove the outer <body> tags
    # <body class="x">content</body> → content
    inner_html = re.sub(r'^<body[^>]*>', '', inner_html)
    inner_html = re.sub(r'</body>$', '', inner_html)

    inner_html = inner_html.strip()

    # Normalize whitespace to remove arbitrary line breaks from source formatting
    if normalize:
        inner_html = normalize_whitespace(inner_html)

    return inner_html, body


def replace_body_content(body_element: etree._Element, new_html: str) -> None:
    """
    Replace the content of <body> with new translated HTML.

    Args:
        body_element: The <body> element to modify
        new_html: New translated HTML content
    """
    # Parse the new content FIRST before clearing the body
    # This prevents data loss if parsing fails
    # Wrap in a temp element to handle multiple root elements
    wrapped = f"<temp xmlns='http://www.w3.org/1999/xhtml'>{new_html}</temp>"

    # Try XML parser first with huge_tree option to handle large documents
    # This preserves exact XML structure without HTML normalization
    parser = etree.XMLParser(
        recover=True,           # Recover from errors
        encoding='utf-8',
        huge_tree=True,         # Allow parsing very large trees
        remove_blank_text=False # Preserve whitespace exactly as-is
    )

    parse_method = "XML"
    parse_warnings = []

    try:
        temp = etree.fromstring(wrapped.encode('utf-8'), parser)
        if parser.error_log:
            parse_warnings.append(f"recovered from {len(parser.error_log)} errors")
    except etree.XMLSyntaxError as e:
        # Fallback: try without namespace
        wrapped_no_ns = f"<temp>{new_html}</temp>"
        try:
            temp = etree.fromstring(wrapped_no_ns.encode('utf-8'), parser)
            parse_method = "XML (no namespace)"
        except Exception as e2:
            # Last resort: HTML parser (but this may alter structure!)
            temp = etree.HTML(wrapped_no_ns)
            parse_method = "HTML fallback"
            if temp is not None:
                temp_elem = temp.find('.//temp')
                if temp_elem is not None:
                    temp = temp_elem
                else:
                    raise XmlParsingError(
                        "Could not find <temp> element after HTML parsing",
                        original_error=e2,
                        content_preview=new_html[:200]
                    )
            else:
                raise XmlParsingError(
                    "All parsing methods failed",
                    original_error=e2,
                    content_preview=new_html[:200]
                )

    # NOW clear the body (only after successful parsing)
    body_element.text = None
    for child in list(body_element):
        body_element.remove(child)

    # Copy content into body
    body_element.text = temp.text
    child_count = 0
    for child in temp:
        body_element.append(child)
        child_count += 1

    # Single consolidated log message
    warnings_str = f" ({', '.join(parse_warnings)})" if parse_warnings else ""
    info(f"📄 Body reconstructed: {len(new_html)} chars → {child_count} elements [{parse_method}{warnings_str}]")
