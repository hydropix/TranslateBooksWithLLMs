"""
XML/HTML helper utilities for safe element manipulation

This module provides safe wrappers around lxml element operations to handle
edge cases and compatibility issues across different lxml versions.
"""
import re
from typing import Iterator, Dict, Any
from lxml import etree


def safe_iter_children(element: etree._Element) -> Iterator[etree._Element]:
    """
    Safely iterate over element children, handling different lxml/Cython versions

    Args:
        element: lxml element

    Yields:
        child elements
    """
    try:
        # Try normal iteration first
        for child in element:
            yield child
    except TypeError:
        # If that fails, try alternative methods
        try:
            # Try getchildren() (deprecated but might work)
            if hasattr(element, 'getchildren'):
                for child in element.getchildren():
                    yield child
            else:
                # Try converting to list
                children = list(element)
                for child in children:
                    yield child
        except Exception:
            # If all else fails, use xpath
            try:
                for child in element.xpath('*'):
                    yield child
            except Exception:
                # Give up - no children accessible
                pass


def safe_get_tag(element: etree._Element) -> str:
    """
    Safely get the tag of an element, handling cases where it might be a method

    Args:
        element: lxml element

    Returns:
        The tag name (including namespace if present)
    """
    try:
        # First try: direct access
        tag = element.tag
        if isinstance(tag, str):
            return tag

        # Second try: call if it's a method
        if callable(tag):
            tag_result = tag()
            if isinstance(tag_result, str):
                return tag_result

        # Third try: use etree.tostring to extract tag name
        try:
            # Get element as string
            elem_str = etree.tostring(element, encoding='unicode', method='xml')
            # Extract tag name from string (e.g., "<p class='bull'>..." -> "p")
            match = re.match(r'<([^>\s]+)', elem_str)
            if match:
                tag_with_ns = match.group(1)
                # Handle namespaced tags
                if '}' in tag_with_ns:
                    return tag_with_ns  # Return full namespaced tag
                else:
                    # Add namespace if element has one
                    if hasattr(element, 'nsmap') and None in element.nsmap:
                        return f"{{{element.nsmap[None]}}}{tag_with_ns}"
                    return tag_with_ns
        except Exception:
            pass

        # Fourth try: Use QName if element has it
        try:
            if hasattr(element, 'qname'):
                return str(element.qname)
        except Exception:
            pass

        return ""
    except Exception:
        return ""


def safe_get_attrib(element: etree._Element) -> Dict[str, Any]:
    """
    Safely get the attributes of an element

    Args:
        element: lxml element

    Returns:
        The attributes dictionary
    """
    try:
        attrib = element.attrib
        if callable(attrib):
            return attrib()
        return attrib
    except Exception:
        return {}


def get_node_text_content_with_br_as_newline(node: etree._Element, namespaces: Dict[str, str],
                                             content_block_tags: set) -> str:
    """
    Extract text content from XML/HTML node with <br> handling

    This function recursively extracts text from a node while converting
    <br> tags to newlines and respecting block-level elements.

    Args:
        node: lxml element node
        namespaces: XML namespace mappings
        content_block_tags: Set of tags that should trigger newlines

    Returns:
        Extracted text with <br> tags converted to newlines
    """
    parts = []
    if node.text:
        parts.append(node.text)

    for child in safe_iter_children(node):
        child_qname_str = safe_get_tag(child)

        # Skip if we couldn't get a valid tag
        if not child_qname_str or ' at 0x' in str(child_qname_str):
            # Try to get text content anyway
            try:
                if hasattr(child, 'text') and child.text:
                    parts.append(child.text)
                if hasattr(child, 'tail') and child.tail:
                    parts.append(child.tail)
            except Exception:
                pass
            continue

        br_xhtml_tag = etree.QName(namespaces.get('xhtml', 'http://www.w3.org/1999/xhtml'), 'br').text

        if child_qname_str == br_xhtml_tag:
            if not (parts and (parts[-1].endswith('\n') or parts[-1] == '\n')):
                parts.append('\n')
        elif child_qname_str in content_block_tags:
            if parts and parts[-1] and not parts[-1].endswith('\n'):
                parts.append('\n')
        else:
            parts.append(get_node_text_content_with_br_as_newline(child, namespaces, content_block_tags))

        if child.tail:
            parts.append(child.tail)

    return "".join(parts)


def serialize_inline_tags(node: etree._Element, preserve_tags: bool = True) -> str:
    """
    Serialize XML/HTML node content while preserving or removing inline tags

    Args:
        node: lxml element node
        preserve_tags: If True, preserve inline tags as XML strings

    Returns:
        Serialized content with tags preserved or removed
    """
    # Use lxml's built-in method first, then clean up
    try:
        if preserve_tags:
            # Get the full XML content
            content = etree.tostring(node, encoding='unicode', method='xml', pretty_print=False)
            # Remove the outer tag
            # Match opening tag
            match = re.match(r'^<[^>]+>', content)
            if match:
                opening_tag_len = len(match.group(0))
                # Find closing tag from the end
                closing_match = re.search(r'</[^>]+>$', content)
                if closing_match:
                    closing_tag_start = closing_match.start()
                    # Extract inner content
                    inner_content = content[opening_tag_len:closing_tag_start]
                    return inner_content
            return content
        else:
            # Get text content only
            return etree.tostring(node, encoding='unicode', method='text')
    except Exception as e:
        # Fallback to manual serialization
        parts = []

        if hasattr(node, 'text') and node.text:
            parts.append(node.text)

        try:
            for child in node:
                # Get child content
                child_content = etree.tostring(child, encoding='unicode', method='xml')
                if child_content and ' at 0x' not in child_content:
                    parts.append(child_content)
        except Exception:
            pass

        return "".join(parts)


def rebuild_element_from_translated_content(element: etree._Element, translated_content: str) -> None:
    """
    Rebuild element structure from translated content containing inline tags

    This function parses translated content that may contain XML/HTML tags
    and reconstructs the element tree structure.

    Args:
        element: lxml element to rebuild
        translated_content: Translated text with preserved XML tags
    """
    # Clear existing content
    element.text = None
    element.tail = None
    for child in list(element):
        element.remove(child)

    # Parse the translated content as XML fragment
    try:
        # Wrap content in a temporary root to handle mixed content
        wrapped_content = f"<temp_root>{translated_content}</temp_root>"

        # Parse with recovery mode to handle potential issues
        parser = etree.XMLParser(recover=True, encoding='utf-8')
        temp_root = etree.fromstring(wrapped_content.encode('utf-8'), parser)

        # Copy content from temp root to element
        element.text = temp_root.text

        # Add all children from temp root
        for child in safe_iter_children(temp_root):
            # Create new element with the same tag and attributes
            new_child = etree.SubElement(element, safe_get_tag(child), attrib=dict(safe_get_attrib(child)))
            new_child.text = child.text
            new_child.tail = child.tail

            # Recursively copy any nested children
            _copy_element_children(child, new_child)

    except Exception as e:
        # Fallback: if parsing fails, just set as text
        element.text = translated_content


def _copy_element_children(source: etree._Element, target: etree._Element) -> None:
    """
    Recursively copy children from source element to target element

    Args:
        source: Source lxml element
        target: Target lxml element
    """
    for child in safe_iter_children(source):
        new_child = etree.SubElement(target, safe_get_tag(child), attrib=dict(safe_get_attrib(child)))
        new_child.text = child.text
        new_child.tail = child.tail
        _copy_element_children(child, new_child)
