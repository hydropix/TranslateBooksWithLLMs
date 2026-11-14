"""
Body serialization for simplified EPUB processing

This module handles extracting and replacing the content of <body> elements
in XHTML documents, enabling full-document translation instead of per-element processing.
"""
import logging
from lxml import etree
from typing import Callable, Tuple, Optional
import html as _html
import re

from .boilerplate_filter import strip_web_boilerplate
from .exceptions import XmlParsingError, BodyExtractionError
from .ruby_annotations import fold_ruby_annotations

logger = logging.getLogger(__name__)


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


def find_body(doc_root: etree._Element) -> Optional[etree._Element]:
    """The document's <body>: XHTML namespace first, then no namespace."""
    body = doc_root.find('.//{http://www.w3.org/1999/xhtml}body')
    if body is None:
        body = doc_root.find('.//body')
    return body


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

    Note: the body element is mutated in place (boilerplate removal when
    strip_boilerplate is True, and <ruby> folding unconditionally); both
    translation paths replace the body content afterwards, so the removed
    elements simply never appear in the output.
    """
    body = find_body(doc_root)
    if body is None:
        return "", None

    if strip_boilerplate:
        strip_web_boilerplate(body, log_callback=log_callback)

    # Fold <ruby> annotations before serialization, otherwise the placeholder
    # pass would split each annotated word into two fragments (issue #242).
    fold_ruby_annotations(body, log_callback=log_callback)

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


# Maximum share of visible text the re-parse of a translated body may lose.
# lxml's recover mode silently drops content after a mis-nested or broken
# tag; a parse that keeps less than this is treated as a failure instead of
# erasing part of a chapter. A parse that needed recovery is held to a much
# tighter bound than a clean one (which should lose nothing at all).
MAX_VISIBLE_TEXT_LOSS = 0.10
MAX_VISIBLE_TEXT_LOSS_WITH_RECOVERY = 0.02

_COMMENT_RE = re.compile(r'<!--.*?-->', re.DOTALL)
_ANY_TAG_RE = re.compile(r'<[^>]*>')
_WHITESPACE_RE = re.compile(r'\s+')


def _visible_text_length(markup: str) -> int:
    """Count non-whitespace characters of the text in a markup string."""
    text = _ANY_TAG_RE.sub('', _COMMENT_RE.sub('', markup))
    return len(_WHITESPACE_RE.sub('', _html.unescape(text)))


def _parsed_text_length(element: etree._Element) -> int:
    """Count non-whitespace characters of the text held by a parsed element."""
    parts = []
    for node in element.iter():
        if not isinstance(node.tag, str):
            # Comments and processing instructions: only their tail is text.
            if node.tail:
                parts.append(node.tail)
            continue
        if node.text:
            parts.append(node.text)
        if node is not element and node.tail:
            parts.append(node.tail)
    return len(_WHITESPACE_RE.sub('', ''.join(parts)))


def replace_body_content(
    body_element: etree._Element,
    new_html: str,
    log_callback: Optional[Callable] = None,
) -> None:
    """
    Replace the content of <body> with new translated HTML.

    Args:
        body_element: The <body> element to modify
        new_html: New translated HTML content
        log_callback: The job's log callback; the reconstruction summary goes
            there, or to this module's logger when there is none

    Raises:
        BodyExtractionError: if new_html is empty while the body still holds
            content. INVARIANT: a populated body is never emptied. Replacing an
            already-empty body with empty content stays a no-op.
        XmlParsingError: if new_html cannot be parsed, or if the parsed result
            lost a significant share of its visible text (see
            MAX_VISIBLE_TEXT_LOSS). The body is left untouched in both cases.
    """
    # Checked before anything is mutated, so the element is left untouched.
    if new_html.strip() == "" and (len(body_element) > 0 or (body_element.text or "").strip()):
        raise BodyExtractionError("refusing to replace populated <body> with empty content")

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
    except etree.XMLSyntaxError:
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

    # Refuse a parse that lost a significant part of the visible text: a
    # mis-nested restoration recovered by lxml can come back empty or
    # truncated, and writing it would silently erase part of the chapter.
    expected_len = _visible_text_length(new_html)
    if expected_len:
        parsed_len = _parsed_text_length(temp)
        recovered = bool(parse_warnings) or parse_method != "XML"
        max_loss = MAX_VISIBLE_TEXT_LOSS_WITH_RECOVERY if recovered else MAX_VISIBLE_TEXT_LOSS
        if parsed_len < expected_len * (1 - max_loss):
            raise XmlParsingError(
                f"Re-parsed body lost visible text: {parsed_len}/{expected_len} "
                f"characters kept [{parse_method}]",
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
    message = (f"📄 Body reconstructed: {len(new_html)} chars → {child_count} elements "
               f"[{parse_method}{warnings_str}]")
    if log_callback:
        log_callback("body_reconstructed", message)
    else:
        logger.info(message)
