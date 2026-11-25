"""
FictionBook 2 (FB2) parsing, text extraction and rebuild.

An FB2 file is a single XML document. Translatable text lives in leaf blocks
that only hold inline markup: ``<p>``, ``<v>``, ``<subtitle>``,
``<text-author>``, ``<th>`` and ``<td>`` inside every ``<body>`` (titles,
epigraphs, poems, cites and the notes body are made of these), plus
``<book-title>``, ``<keywords>`` and the annotation blocks of
``<title-info>``. Everything else (ids, links, ``<image>``, ``<binary>``
blobs, author names, ``<document-info>``, attributes) is never sent and is
written back unchanged.

Each block becomes one ``<p data-b="N">...</p>`` line of an HTML-like string
so the EPUB machinery (TagPreserver placeholders + HtmlChunker) can chunk and
translate it. Inline elements stay as tags inside the line and therefore
become placeholders. Note references (``<a type="note">``) and ``<code>``
are swapped for an empty marker element first, so their text is not
translated either. After translation each line is parsed back and its
content replaces the content of the original element, which keeps the
element itself, its attributes and its tail.
"""

import codecs
import copy
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from lxml import etree

FB2_NS = 'http://www.gribuser.ru/xml/fictionbook/2.0'

#: Leaf blocks of the FB2 body whose content is translated.
TEXT_BLOCKS = frozenset({'p', 'v', 'subtitle', 'text-author', 'th', 'td'})
#: Translated children of ``<title-info>`` (annotation blocks come on top).
META_BLOCKS = frozenset({'book-title', 'keywords'})
#: Blocks duplicated (source then translation) in bilingual output.
BILINGUAL_BLOCKS = frozenset({'p', 'v', 'subtitle', 'text-author'})

#: Local name of the marker standing in for a kept inline element.
KEEP_MARKER = 'fb2-keep'

_UTF8_BOM = b'\xef\xbb\xbf'
_DECLARATION_RE = re.compile(r'^\s*<\?xml[^>]*\?>', re.S)
_ENCODING_RE = re.compile(rb'<\?xml[^>]*?encoding\s*=\s*["\']([A-Za-z0-9._:-]+)["\']')
_LINE_RE = re.compile(r'<p data-b="(\d+)">(.*?)</p>', re.S)
_BOUNDARY_RE = re.compile(r'<p data-b="\d+">|</p>')
_BARE_AMPERSAND_RE = re.compile(r'&(?!#\d+;|#x[0-9A-Fa-f]+;|[A-Za-z][\w.-]*;)')


class Fb2Error(ValueError):
    """The file is not a readable FB2 document."""


def _parser(recover: bool = False) -> etree.XMLParser:
    # No entity expansion, no network: the file comes from the user.
    # huge_tree: <binary> images routinely exceed libxml2's 10 MB text limit.
    return etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=True,
                           remove_blank_text=False, recover=recover)


def _local(element) -> str:
    tag = element.tag
    if not isinstance(tag, str):
        return ''
    return tag.rsplit('}', 1)[-1]


@dataclass
class Fb2Block:
    """One translatable leaf element."""

    element: etree._Element
    start_tag: str  # serialized start tag, with the namespace declarations
    end_tag: str
    inner: str  # serialized content, kept elements replaced by markers
    kept: List[etree._Element] = field(default_factory=list)

    @property
    def name(self) -> str:
        return _local(self.element)


@dataclass
class Fb2Document:
    """A parsed FB2 file and its translatable blocks."""

    tree: etree._ElementTree
    source_encoding: str
    blocks: List[Fb2Block]

    @property
    def root(self) -> etree._Element:
        return self.tree.getroot()


def decode_fb2(data: bytes) -> Tuple[str, str]:
    """Decode raw FB2 bytes into text without its XML declaration.

    The encoding comes from a BOM, else from the XML declaration (many FB2
    files are windows-1251), else UTF-8.

    Returns:
        ``(text, encoding)``
    """
    if data.startswith(_UTF8_BOM):
        encoding, data = 'utf-8', data[len(_UTF8_BOM):]
    elif data.startswith((b'\xff\xfe', b'\xfe\xff')):
        encoding = 'utf-16'
    else:
        match = _ENCODING_RE.search(data[:512])
        encoding = match.group(1).decode('ascii').lower() if match else 'utf-8'
        try:
            codecs.lookup(encoding)
        except LookupError:
            encoding = 'utf-8'
    try:
        text = data.decode(encoding)
    except UnicodeDecodeError:
        text = data.decode(encoding, errors='replace')
    if text.startswith('﻿'):
        text = text[1:]
    return _DECLARATION_RE.sub('', text, count=1).lstrip(), encoding


def parse_fb2(data: bytes) -> Fb2Document:
    """Parse FB2 bytes and collect their translatable blocks."""
    text, encoding = decode_fb2(data)
    try:
        root = etree.fromstring(text.encode('utf-8'), _parser())
    except etree.XMLSyntaxError as e:
        raise Fb2Error(f"not well-formed XML: {e}") from e
    if _local(root) != 'FictionBook':
        raise Fb2Error(f"root element is <{_local(root)}>, expected <FictionBook>")
    tree = root.getroottree()
    return Fb2Document(tree=tree, source_encoding=encoding, blocks=collect_blocks(root))


def _is_kept_inline(element) -> bool:
    """Inline elements whose text must not be translated."""
    name = _local(element)
    if name == 'code':
        return True
    return name == 'a' and element.get('type') == 'note'


def _in_scope(element) -> bool:
    """Whether a candidate block sits in a translated part of the book."""
    name = _local(element)
    ancestors = [_local(a) for a in element.iterancestors()]
    if 'body' in ancestors:
        return name in TEXT_BLOCKS
    if 'title-info' in ancestors:
        if name in META_BLOCKS:
            return True
        return name in TEXT_BLOCKS and 'annotation' in ancestors
    return False


def collect_blocks(root: etree._Element) -> List[Fb2Block]:
    """The translatable leaf blocks of the document, in document order."""
    blocks: List[Fb2Block] = []
    taken = set()
    for element in root.iter():
        name = _local(element)
        if name not in TEXT_BLOCKS and name not in META_BLOCKS:
            continue
        if not _in_scope(element):
            continue
        # A malformed file may nest blocks; the outer one owns the text.
        if any(a in taken for a in element.iterancestors()):
            continue
        if not ''.join(element.itertext()).strip():
            continue
        taken.add(element)
        blocks.append(_serialize_block(element))
    return blocks


def _serialize_block(element: etree._Element) -> Fb2Block:
    clone = copy.deepcopy(element)
    clone.tail = None
    namespace = etree.QName(element).namespace
    marker_tag = f'{{{namespace}}}{KEEP_MARKER}' if namespace else KEEP_MARKER

    kept: List[etree._Element] = []
    for child in list(clone.iterdescendants()):
        if not _is_kept_inline(child):
            continue
        if any(_local(a) == KEEP_MARKER for a in child.iterancestors()) \
                or child.getparent() is None:
            continue
        original = copy.deepcopy(child)
        original.tail = None
        marker = etree.Element(marker_tag, i=str(len(kept)))
        marker.tail = child.tail
        child.getparent().replace(child, marker)
        kept.append(original)

    markup = etree.tostring(clone, encoding='unicode')
    start_end = markup.index('>') + 1
    if markup[start_end - 2] == '/':
        # Self-closing: only possible without content, which is never collected.
        raise Fb2Error(f"empty block <{_local(element)}>")
    end_start = markup.rindex('</')
    return Fb2Block(
        element=element,
        start_tag=markup[:start_end],
        end_tag=markup[end_start:],
        inner=markup[start_end:end_start],
        kept=kept,
    )


def build_translation_html(blocks: List[Fb2Block]) -> str:
    """The HTML-like string the placeholder pipeline translates."""
    return '\n'.join(f'<p data-b="{i}">{block.inner}</p>' for i, block in enumerate(blocks))


def is_block_boundary(markup: str) -> bool:
    """Whether a placeholder's markup separates two blocks of
    :func:`build_translation_html` (text must never cross it)."""
    return bool(_BOUNDARY_RE.search(markup))


def split_translated_html(html: str) -> Dict[int, str]:
    """Map block index -> translated inner markup, from the restored string."""
    lines: Dict[int, str] = {}
    for match in _LINE_RE.finditer(html):
        lines.setdefault(int(match.group(1)), match.group(2))
    return lines


def _parse_block(block: Fb2Block, inner: str) -> Optional[etree._Element]:
    """Parse a translated line back into an element, or None."""
    candidates = [inner]
    fixed = _BARE_AMPERSAND_RE.sub('&amp;', inner)
    if fixed != inner:
        candidates.append(fixed)
    for candidate in candidates:
        try:
            return etree.fromstring(block.start_tag + candidate + block.end_tag, _parser())
        except etree.XMLSyntaxError:
            continue
    return None


def _restore_kept(parsed: etree._Element, block: Fb2Block) -> bool:
    """Put the kept inline elements back in place of their markers."""
    markers = [el for el in parsed.iter() if _local(el) == KEEP_MARKER]
    indices = [m.get('i') or '' for m in markers]
    if sorted(int(i) if i.isdigit() else -1 for i in indices) != list(range(len(block.kept))):
        return False
    for marker in markers:
        original = copy.deepcopy(block.kept[int(marker.get('i'))])
        original.tail = marker.tail
        marker.getparent().replace(marker, original)
    return True


def apply_block_translation(block: Fb2Block, inner: str) -> bool:
    """Replace the content of ``block.element`` with the translated markup.

    The element keeps its own tag, attributes (ids) and tail. Returns False,
    leaving the element untouched, when the translation cannot be parsed or
    lost one of the kept inline elements.
    """
    parsed = _parse_block(block, inner)
    if parsed is None or not _restore_kept(parsed, block):
        return False
    element = block.element
    for child in list(element):
        element.remove(child)
    element.text = parsed.text
    for child in list(parsed):
        element.append(child)  # lxml moves the node and reconciles namespaces
    return True


def set_document_language(root: etree._Element, language_code: str) -> bool:
    """Set ``<title-info><lang>``; False when the file has no such element."""
    for title_info in root.iter():
        if _local(title_info) != 'title-info':
            continue
        if _local(title_info.getparent()) != 'description':
            continue
        for child in title_info:
            if _local(child) == 'lang':
                child.text = language_code
                return True
    return False


def apply_translations(document: Fb2Document, translated: Dict[int, str],
                       bilingual: bool = False) -> List[int]:
    """Write the translated lines into the tree.

    Args:
        document: The parsed document.
        translated: Block index -> translated inner markup.
        bilingual: Keep each body text block in the source language and add
            its translation right after it (the translation drops ``id``
            attributes so ids stay unique).

    Returns:
        Indices of the blocks that kept their source text.
    """
    failed: List[int] = []
    for index, block in enumerate(document.blocks):
        inner = translated.get(index)
        if inner is None:
            failed.append(index)
            continue
        original = copy.deepcopy(block.element) if bilingual else None
        if not apply_block_translation(block, inner):
            failed.append(index)
            continue
        if bilingual and block.name in BILINGUAL_BLOCKS and inner != block.inner:
            translation = block.element
            original.tail = translation.tail
            translation.getparent().insert(translation.getparent().index(translation),
                                           original)
            translation.attrib.pop('id', None)
    return failed


def serialize_fb2(document: Fb2Document) -> bytes:
    """The document as UTF-8 bytes with a matching XML declaration."""
    root = document.root
    etree.cleanup_namespaces(root)
    # Comments and processing instructions around the root element, each on
    # its own line (serializing the whole tree would glue them together).
    nodes = list(reversed(list(root.itersiblings(preceding=True))))
    nodes += [root] + list(root.itersiblings())
    body = '\n'.join(etree.tostring(node, encoding='unicode', with_tail=False)
                     for node in nodes)
    return ('<?xml version="1.0" encoding="utf-8"?>\n' + body + '\n').encode('utf-8')
