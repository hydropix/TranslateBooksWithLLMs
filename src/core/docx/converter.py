"""
DOCX ↔ HTML conversion for translation.

Uses:
- mammoth: Conversion DOCX → HTML (semantic, clean)
- python-docx: Metadata extraction + DOCX reconstruction
"""

import base64
import binascii
import copy
import io
import logging
import os
import re
import tempfile
import mammoth
from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, RGBColor
from docx.text.run import Run
from typing import Tuple, Dict, Any, Optional, Set
from lxml import etree


logger = logging.getLogger(__name__)

_W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
_R_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
_M_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
_XML_NS = 'http://www.w3.org/XML/1998/namespace'

# Inline HTML formatting tags and the python-docx run property each sets.
_INLINE_FORMAT_TAGS = {
    'strong': 'bold', 'b': 'bold',
    'em': 'italic', 'i': 'italic',
    'u': 'underline',
}

# Marker injected in place of OMML equations before mammoth runs.
# Mammoth drops <m:oMath> entirely; we replace each with this text marker so
# it survives DOCX -> HTML -> DOCX, then we re-inject the original OMML.
_EQ_MARKER_PREFIX = '__TTBLLEQ'
_EQ_MARKER_SUFFIX = 'EQTTBLL__'
_EQ_MARKER_REGEX = re.compile(
    re.escape(_EQ_MARKER_PREFIX) + r'(\d+)' + re.escape(_EQ_MARKER_SUFFIX)
)

# In the HTML returned by to_html(), each equation appears as a self-closing
# <eq id="N"/> tag. Because TagPreserver groups anything matching <[^>]+>
# into [idN] placeholders that the LLM treats as opaque tokens, the equation
# never reaches the LLM as raw text. from_html() converts these tags back to
# text markers before the HTML->DOCX rebuild so _restore_equations can splice
# the original OMML back in.
_EQ_TAG_REGEX = re.compile(r'<eq id="(\d+)"\s*/>')

# Tags that start a new paragraph when found inside a <blockquote>.
_BLOCKQUOTE_BLOCK_TAGS = frozenset(
    {'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote'}
)
# Blockquote children that are rebuilt by the generic dispatch instead of
# being folded into a quoted paragraph.
_BLOCKQUOTE_DELEGATED_TAGS = frozenset({'ul', 'ol', 'table'})

_OMML_XPATH = etree.XPath(
    './/m:oMathPara | .//m:oMath[not(ancestor::m:oMathPara)]',
    namespaces={'m': _M_NS},
)
_WT_XPATH = etree.XPath('.//w:t', namespaces={'w': _W_NS})

# Header/footer slots of a python-docx Section, copied from the source
# document because mammoth never reads headers and footers.
_HEADER_FOOTER_ATTRS = (
    'header', 'first_page_header', 'even_page_header',
    'footer', 'first_page_footer', 'even_page_footer',
)

# Colour of Word's built-in "Hyperlink" character style, applied by hand
# because python-docx's default template does not define that style.
_HYPERLINK_COLOR = RGBColor(0x05, 0x63, 0xC1)


class DocxHtmlConverter:
    """Converts DOCX to/from HTML for translation."""

    def to_html(self, docx_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Convert DOCX → HTML + metadata.

        Process:
        1. Use mammoth for clean HTML conversion
        2. Extract metadata with python-docx (styles, fonts, etc.)
        3. Return HTML + metadata dict

        Args:
            docx_path: Path to input DOCX file

        Returns:
            (html_content, metadata)
            - html_content: Semantic HTML (<p>, <strong>, <em>, etc.)
            - metadata: Dict with styles, fonts, page settings
        """
        # 1a. Pre-extract OMML equations into markers (mammoth drops them).
        patched_path, equations = self._replace_equations_with_markers(docx_path)

        try:
            # 1b. Conversion via mammoth (clean semantic HTML)
            with open(patched_path, 'rb') as docx_file:
                result = mammoth.convert_to_html(docx_file)
                html_content = result.value

            # 1c. Promote raw equation markers to HTML tags so TagPreserver
            # protects them as opaque [idN] placeholders. Without this the
            # LLM sees the literal text "__TTBLLEQ0EQTTBLL__" and may
            # mangle, translate or drop it.
            if equations:
                html_content = _EQ_MARKER_REGEX.sub(
                    r'<eq id="\1"/>', html_content
                )

            # 2. Extract metadata via python-docx (use original to keep page settings)
            doc = Document(docx_path)
            metadata = self._extract_metadata(doc)
            metadata['equations'] = equations
            # Preserve original inline image dimensions in document order;
            # mammoth strips them from the HTML, so we restore at rebuild time.
            metadata['image_dimensions'] = [
                (shape.width, shape.height) for shape in doc.inline_shapes
            ]

            return html_content, metadata
        finally:
            if patched_path != docx_path and os.path.exists(patched_path):
                try:
                    os.remove(patched_path)
                except OSError:
                    pass

    def from_html(
        self,
        html_content: str,
        metadata: Dict[str, Any],
        output_path: str,
        source_docx_path: Optional[str] = None,
    ) -> None:
        """
        Reconstruct DOCX from translated HTML + metadata.

        Process:
        1. Parse HTML with lxml
        2. Create empty Document() with python-docx
        3. For each HTML element, create DOCX paragraph/run
        4. Apply styles from metadata
        5. Save DOCX

        Args:
            html_content: Translated HTML content
            metadata: Original document metadata (styles, fonts, etc.)
            output_path: Path to output DOCX file
            source_docx_path: The source DOCX. When given, its headers and
                footers are copied into the output unchanged: mammoth does
                not read them, so they never reach the translated HTML.
        """
        # Reverse the equation-tag substitution done in to_html so the rest
        # of the rebuild pipeline (which expects raw text markers) is unchanged.
        html_content = _EQ_TAG_REGEX.sub(
            lambda m: f'{_EQ_MARKER_PREFIX}{m.group(1)}{_EQ_MARKER_SUFFIX}',
            html_content,
        )

        # Parse HTML
        html_tree = etree.HTML(html_content)

        # Create DOCX document
        doc = Document()

        # Apply page metadata (page size, margins, etc.)
        self._apply_page_metadata(doc, metadata)

        # Queue of (width, height) pairs in document order, consumed by
        # _add_image_run as it embeds each <img>. mammoth drops sizing info
        # from the HTML, so we replay it from the original DOCX.
        self._image_dims_queue = list(metadata.get('image_dimensions', []))

        # Convert HTML → DOCX paragraphs
        if html_tree is not None:
            body = html_tree.find('.//body')
            if body is not None:
                for element in body:
                    self._convert_html_element_to_docx(doc, element, metadata)

        if source_docx_path:
            self._copy_headers_footers(source_docx_path, doc)

        # Attribution stamp: identical for every install, disabled by the
        # ATTRIBUTION_ENABLED switch. Left untouched when attribution is off.
        import src.config as cfg
        if cfg.ATTRIBUTION_ENABLED:
            doc.core_properties.last_modified_by = cfg.GENERATOR_NAME

        # Save
        doc.save(output_path)

        # Restore OMML equations (replace text markers with original OMML XML)
        equations = metadata.get('equations') if metadata else None
        if equations:
            self._restore_equations(output_path, equations)

    def _copy_headers_footers(self, source_docx_path: str, doc: Document) -> None:
        """
        Copy the first section's headers and footers from the source DOCX.

        The rebuilt document has a single section, so the first section's
        header/footer set is the one carried over. Their XML is copied as is
        (text untranslated, fields such as page numbers kept); images are
        re-embedded and hyperlinks re-related in the output's own parts.
        A failure here is logged and leaves the output without them rather
        than failing the whole translation.
        """
        try:
            source = Document(source_docx_path)
            if not source.sections or not doc.sections:
                return
            src_section = source.sections[0]
            dst_section = doc.sections[0]
            dst_section.different_first_page_header_footer = (
                src_section.different_first_page_header_footer
            )
            doc.settings.odd_and_even_pages_header_footer = (
                source.settings.odd_and_even_pages_header_footer
            )
            for attr in _HEADER_FOOTER_ATTRS:
                src_story = getattr(src_section, attr)
                # A first section's "linked" slot has no definition at all.
                if src_story.is_linked_to_previous:
                    continue
                dst_story = getattr(dst_section, attr)
                dst_story.is_linked_to_previous = False
                self._copy_story(src_story, dst_story)
        except Exception as exc:  # noqa: BLE001 - a header must not sink the job
            logger.warning("Could not copy DOCX headers/footers: %s", exc)

    def _copy_story(self, src_story, dst_story) -> None:
        """Replace ``dst_story``'s content with a copy of ``src_story``'s."""
        src_el = src_story._element
        dst_el = dst_story._element
        for child in list(dst_el):
            dst_el.remove(child)
        for child in src_el:
            dst_el.append(copy.deepcopy(child))
        self._remap_relationships(dst_el, src_story.part, dst_story.part)

    def _remap_relationships(self, element, src_part, dst_part) -> None:
        """
        Point every r:* reference in a copied subtree at ``dst_part``'s rels.

        Images are re-embedded and external targets (hyperlinks) re-related.
        A reference to any other internal part cannot be carried over, so the
        run holding it is removed rather than left dangling.
        """
        mapping: Dict[str, Optional[str]] = {}
        for node in list(element.iter()):
            for attr_name, value in list(node.attrib.items()):
                if not attr_name.startswith(f'{{{_R_NS}}}'):
                    continue
                if value not in mapping:
                    mapping[value] = self._copy_relationship(value, src_part, dst_part)
                new_id = mapping[value]
                if new_id is not None:
                    node.set(attr_name, new_id)
                    continue
                holder = node
                while holder is not None and holder.tag != qn('w:r'):
                    holder = holder.getparent()
                target = holder if holder is not None else node
                parent = target.getparent()
                if parent is not None:
                    parent.remove(target)
                break

    def _copy_relationship(self, r_id: str, src_part, dst_part) -> Optional[str]:
        """Recreate relationship ``r_id`` of ``src_part`` in ``dst_part``."""
        rel = src_part.rels.get(r_id)
        if rel is None:
            return None
        if rel.is_external:
            return dst_part.relate_to(rel.target_ref, rel.reltype, is_external=True)
        if rel.reltype == RT.IMAGE:
            new_id, _ = dst_part.get_or_add_image(io.BytesIO(rel.target_part.blob))
            return new_id
        return None

    def _replace_equations_with_markers(
        self, docx_path: str
    ) -> Tuple[str, Dict[int, str]]:
        """
        Pre-process a DOCX so OMML equations survive the mammoth round-trip.

        Mammoth drops <m:oMath> / <m:oMathPara> entirely. We extract each
        outermost equation, store its XML, and replace it with a <w:r><w:t>
        containing a unique text marker. Mammoth then carries the marker
        through into the HTML output, where it can later be matched by
        _restore_equations on the rebuilt DOCX.

        Returns:
            (path_to_use, equations_dict)
            - path_to_use: original path if no equations, else a temp DOCX
            - equations_dict: {idx: serialized_omml_xml}
        """
        doc = Document(docx_path)
        body = doc.element.body

        omml_elements = _OMML_XPATH(body)

        if not omml_elements:
            return docx_path, {}

        equations: Dict[int, str] = {}
        for idx, eq in enumerate(omml_elements):
            equations[idx] = etree.tostring(eq, encoding='unicode')

            new_r = etree.Element(f'{{{_W_NS}}}r')
            new_t = etree.SubElement(new_r, f'{{{_W_NS}}}t')
            new_t.text = f'{_EQ_MARKER_PREFIX}{idx}{_EQ_MARKER_SUFFIX}'
            new_t.set(f'{{{_XML_NS}}}space', 'preserve')

            parent = eq.getparent()
            eq_idx = list(parent).index(eq)
            parent.remove(eq)
            parent.insert(eq_idx, new_r)

        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.docx')
        os.close(tmp_fd)
        doc.save(tmp_path)
        return tmp_path, equations

    def _restore_equations(
        self, docx_path: str, equations: Dict[int, str]
    ) -> None:
        """
        Replace text markers in the saved DOCX with the original OMML XML.

        A marker may sit alone in a <w:t>, or share its run with surrounding
        text (mammoth often consolidates adjacent runs). In the second case
        we split the run into [text-prefix, OMML, text-suffix] within the
        same paragraph.
        """
        doc = Document(docx_path)
        body = doc.element.body

        for t_elem in list(_WT_XPATH(body)):
            text = t_elem.text
            if not text or _EQ_MARKER_PREFIX not in text:
                continue

            matches = list(_EQ_MARKER_REGEX.finditer(text))
            if not matches:
                continue

            r_elem = t_elem.getparent()
            if r_elem is None:
                continue
            p_elem = r_elem.getparent()
            if p_elem is None:
                continue

            r_idx = list(p_elem).index(r_elem)

            new_elements = []
            cursor = 0
            for m in matches:
                before = text[cursor:m.start()]
                if before:
                    rb = etree.Element(f'{{{_W_NS}}}r')
                    tb = etree.SubElement(rb, f'{{{_W_NS}}}t')
                    tb.text = before
                    tb.set(f'{{{_XML_NS}}}space', 'preserve')
                    new_elements.append(rb)

                eq_idx = int(m.group(1))
                omml_xml = equations.get(eq_idx)
                if omml_xml:
                    try:
                        new_elements.append(etree.fromstring(omml_xml))
                    except etree.XMLSyntaxError:
                        pass
                cursor = m.end()

            after = text[cursor:]
            if after:
                ra = etree.Element(f'{{{_W_NS}}}r')
                ta = etree.SubElement(ra, f'{{{_W_NS}}}t')
                ta.text = after
                ta.set(f'{{{_XML_NS}}}space', 'preserve')
                new_elements.append(ra)

            p_elem.remove(r_elem)
            for offset, elem in enumerate(new_elements):
                p_elem.insert(r_idx + offset, elem)

        doc.save(docx_path)

    def _extract_metadata(self, doc: Document) -> Dict[str, Any]:
        """
        Extract styles, fonts, page settings from DOCX.

        Args:
            doc: python-docx Document instance

        Returns:
            Dict with document metadata
        """
        metadata = {
            'styles': {},
            'default_font': None,
            'page_size': None,
            'margins': None,
        }

        # Extract page settings from first section
        if doc.sections:
            section = doc.sections[0]
            metadata['page_size'] = {
                'width': section.page_width.inches if section.page_width else None,
                'height': section.page_height.inches if section.page_height else None
            }
            metadata['margins'] = {
                'top': section.top_margin.inches if section.top_margin else None,
                'bottom': section.bottom_margin.inches if section.bottom_margin else None,
                'left': section.left_margin.inches if section.left_margin else None,
                'right': section.right_margin.inches if section.right_margin else None
            }

        # Extract default font if available
        # Note: python-docx doesn't provide easy access to default font,
        # so we'll use a common default
        metadata['default_font'] = {
            'name': 'Calibri',
            'size': 11
        }

        return metadata

    def _apply_page_metadata(self, doc: Document, metadata: Dict[str, Any]) -> None:
        """
        Apply page settings to document.

        Args:
            doc: python-docx Document instance
            metadata: Document metadata
        """
        if not doc.sections:
            return

        section = doc.sections[0]

        # Apply page size
        page_size = metadata.get('page_size', {})
        if page_size.get('width') is not None:
            section.page_width = Inches(page_size['width'])
        if page_size.get('height') is not None:
            section.page_height = Inches(page_size['height'])

        # Apply margins
        margins = metadata.get('margins', {})
        if margins.get('top') is not None:
            section.top_margin = Inches(margins['top'])
        if margins.get('bottom') is not None:
            section.bottom_margin = Inches(margins['bottom'])
        if margins.get('left') is not None:
            section.left_margin = Inches(margins['left'])
        if margins.get('right') is not None:
            section.right_margin = Inches(margins['right'])

    def _convert_html_element_to_docx(
        self,
        doc: Document,
        element: etree._Element,
        metadata: Dict[str, Any]
    ):
        """
        Convert an HTML element to appropriate DOCX element.

        Args:
            doc: python-docx Document instance
            element: lxml HTML element
            metadata: Document metadata for styling
        """
        tag = element.tag

        if tag == 'p':
            self._convert_paragraph(doc, element, metadata)
        elif tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            self._convert_heading(doc, element, tag, metadata)
        elif tag in ['ul', 'ol']:
            self._convert_list(doc, element, metadata)
        elif tag == 'table':
            self._convert_table(doc, element, metadata)
        elif tag == 'br':
            doc.add_paragraph()  # Empty paragraph for line break
        elif tag == 'img':
            self._add_image_run(doc.add_paragraph(), element)
        elif tag == 'blockquote':
            self._convert_blockquote(doc, element, metadata)
        # Skip other tags (div, span handled within paragraphs)

    def _convert_paragraph(
        self,
        doc: Document,
        element: etree._Element,
        metadata: Dict[str, Any]
    ):
        """Convert HTML <p> to DOCX paragraph."""
        p = doc.add_paragraph()
        self._add_runs_from_element(p, element, metadata)

    def _convert_heading(
        self,
        doc: Document,
        element: etree._Element,
        tag: str,
        metadata: Dict[str, Any]
    ):
        """Convert HTML heading to DOCX heading."""
        level = int(tag[1])  # h1 → 1, h2 → 2, etc.
        text = self._get_text_content(element)
        doc.add_heading(text, level=level)

    def _convert_blockquote(
        self,
        doc: Document,
        element: etree._Element,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Convert HTML <blockquote> to one or more quoted DOCX paragraphs.

        Each block-level child (<p>, <h1>-<h6>) becomes its own paragraph so
        multi-paragraph quotes are not collapsed; a nested <blockquote>
        recurses. Lists and tables inside the quote are handed back to the
        generic dispatch so they are not silently dropped.
        """
        style_name = self._quote_style_name(doc)
        block_children = [
            child for child in element
            if child.tag in _BLOCKQUOTE_BLOCK_TAGS
        ]

        if block_children:
            for child in block_children:
                if child.tag == 'blockquote':
                    self._convert_blockquote(doc, child, metadata)
                    continue
                paragraph = doc.add_paragraph()
                self._add_runs_from_element(paragraph, child, metadata)
                self._apply_quote_format(paragraph, style_name)
        elif self._get_block_text(element, exclude=_BLOCKQUOTE_DELEGATED_TAGS).strip():
            # Bare quote text with no <p> wrapper. The delegated children are
            # emitted by the loop below, so they are skipped here rather than
            # flattened into this paragraph a second time.
            paragraph = doc.add_paragraph()
            self._add_runs_from_element(
                paragraph, element, metadata, skip_tags=_BLOCKQUOTE_DELEGATED_TAGS
            )
            self._apply_quote_format(paragraph, style_name)

        for child in element:
            if child.tag in _BLOCKQUOTE_DELEGATED_TAGS:
                self._convert_html_element_to_docx(doc, child, metadata)

    def _quote_style_name(self, doc: Document) -> Optional[str]:
        """
        Return the quote style available in the document, or None.

        python-docx raises KeyError for a style name the template does not
        define, so each candidate is probed before use.
        """
        for name in ('Quote', 'Intense Quote'):
            try:
                doc.styles[name]
            except KeyError:
                continue
            return name
        return None

    def _apply_quote_format(self, paragraph, style_name: Optional[str]) -> None:
        """Mark a paragraph as quoted, falling back to a left indent."""
        if style_name:
            paragraph.style = style_name
        else:
            paragraph.paragraph_format.left_indent = Inches(0.5)

    def _convert_list(
        self,
        doc: Document,
        element: etree._Element,
        metadata: Dict[str, Any],
        level: int = 1
    ):
        """
        Convert an HTML list to DOCX list paragraphs.

        Only direct <li> children are consumed, in document order; nested
        <ul>/<ol> recurse with an incremented level. Using `.//li` here would
        emit every descendant item twice, once for the outer list and once
        for the nested one.
        """
        is_ordered = element.tag == 'ol'

        for li in [child for child in element if child.tag == 'li']:
            text = self._get_block_text(li, exclude={'ul', 'ol'})
            if text.strip():
                self._add_list_paragraph(doc, text, is_ordered, level)

            for child in li:
                if child.tag in ('ul', 'ol'):
                    self._convert_list(doc, child, metadata, level + 1)

    def _list_style(self, is_ordered: bool, level: int) -> str:
        """Return the list style name for a nesting level (1-based)."""
        base = 'List Number' if is_ordered else 'List Bullet'
        if level >= 2:
            return f'{base} {min(level, 3)}'
        return base

    def _add_list_paragraph(
        self,
        doc: Document,
        text: str,
        is_ordered: bool,
        level: int
    ):
        """
        Add a list paragraph, falling back to the base style if needed.

        The style is applied after the paragraph is created: add_paragraph()
        appends the paragraph before applying the style, so catching the
        KeyError around the whole call would leave an unstyled duplicate
        behind.
        """
        paragraph = doc.add_paragraph(text)
        try:
            paragraph.style = self._list_style(is_ordered, level)
        except KeyError:
            paragraph.style = self._list_style(is_ordered, 1)
        return paragraph

    def _convert_table(
        self,
        doc: Document,
        element: etree._Element,
        metadata: Dict[str, Any]
    ):
        """
        Convert HTML <table> to a DOCX table.

        Rows and cells are collected from direct children only, so the rows
        of a table nested inside a cell never leak into the parent table.
        A nested table's text is flattened into its parent cell, appearing
        exactly once; real nested DOCX tables are out of scope.
        """
        rows = self._collect_table_rows(element)
        if not rows:
            return

        # Direct children in document order, so a <th> after a <td> keeps
        # its position instead of being pushed to the end of the row.
        row_cells = [
            [child for child in tr if child.tag in ('td', 'th')]
            for tr in rows
        ]

        # The widest row (spans included) defines the column count; a narrow
        # header row must not truncate the rows below it.
        placements, cols = self._place_table_cells(row_cells)
        if cols == 0:
            return

        table = doc.add_table(rows=len(rows), cols=cols)
        table.style = 'Table Grid'

        for row_idx, col_idx, row_span, col_span, cell in placements:
            target = table.cell(row_idx, col_idx)
            if row_span > 1 or col_span > 1:
                try:
                    target = target.merge(
                        table.cell(row_idx + row_span - 1, col_idx + col_span - 1)
                    )
                except Exception:  # noqa: BLE001 - malformed overlapping spans
                    target = table.cell(row_idx, col_idx)
            target.text = self._get_text_content(cell)

    @staticmethod
    def _span(value: Optional[str]) -> int:
        """A colspan/rowspan attribute as a positive int (1 when invalid)."""
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 1

    def _place_table_cells(self, row_cells: list) -> Tuple[list, int]:
        """
        Lay HTML cells out on a grid, honouring colspan and rowspan.

        Returns ``(placements, column_count)``; each placement is
        ``(row, col, row_span, col_span, cell_element)``. A cell starts at
        the first column of its row not already covered by a rowspan from
        above, and a rowspan past the last row is clipped. Without spans this
        is the plain "n-th cell in the n-th column" layout.
        """
        row_count = len(row_cells)
        occupied: Set[Tuple[int, int]] = set()
        placements = []
        for row_idx, cells in enumerate(row_cells):
            col_idx = 0
            for cell in cells:
                while (row_idx, col_idx) in occupied:
                    col_idx += 1
                row_span = min(self._span(cell.get('rowspan')), row_count - row_idx)
                col_span = self._span(cell.get('colspan'))
                for dr in range(row_span):
                    for dc in range(col_span):
                        occupied.add((row_idx + dr, col_idx + dc))
                placements.append((row_idx, col_idx, row_span, col_span, cell))
                col_idx += col_span
        cols = max((col for _, col in occupied), default=-1) + 1
        return placements, cols

    def _collect_table_rows(self, element: etree._Element) -> list:
        """
        Collect the <tr> belonging to this table, in document order.

        Direct-child <tr> keep their position, and <thead>/<tbody>/<tfoot>
        sections contribute their own direct-child <tr>.
        """
        rows = []
        for child in element:
            if child.tag == 'tr':
                rows.append(child)
            elif child.tag in ('thead', 'tbody', 'tfoot'):
                rows.extend(
                    grandchild for grandchild in child
                    if grandchild.tag == 'tr'
                )
        return rows

    def _add_runs_from_element(
        self,
        paragraph,
        element: etree._Element,
        metadata: Dict[str, Any],
        skip_tags: Set[str] = frozenset()
    ):
        """
        Add runs to paragraph from HTML element, preserving inline formatting.

        Handles <strong>, <em>, <b>, <i>, etc.

        Children whose tag is in ``skip_tags`` contribute no run, but their
        tail text is kept. Callers use this when the skipped subtree is
        emitted separately, so its text is not written twice.
        """
        # Handle direct text
        if element.text:
            paragraph.add_run(element.text)

        # Handle child elements
        for child in element:
            if child.tag in skip_tags:
                # Emitted separately by the caller; only the tail belongs here.
                if child.tail:
                    paragraph.add_run(child.tail)
                continue

            if child.tag in _INLINE_FORMAT_TAGS:
                # Nested inline formatting (<strong><em>…</em></strong>) is
                # accumulated so a bold + italic run keeps both properties.
                self._add_formatted_runs(paragraph, child, frozenset())
            elif child.tag == 'a':
                self._add_hyperlink(paragraph, child, frozenset())
            elif child.tag == 'img':
                self._add_image_run(paragraph, child)
            else:
                # For other tags, just extract text
                text = self._get_text_content(child)
                paragraph.add_run(text)

            # Handle tail text (text after closing tag)
            if child.tail:
                paragraph.add_run(child.tail)

    def _add_formatted_runs(self, paragraph, element: etree._Element, inherited: frozenset):
        """Emit the runs of an inline-formatting subtree, accumulating
        bold/italic/underline down nested <strong>/<em>/<u> elements."""
        formats = inherited | {_INLINE_FORMAT_TAGS[element.tag]}
        self._emit_inline_content(paragraph, element, formats)

    def _emit_inline_content(self, paragraph, element: etree._Element, formats: frozenset):
        """Emit ``element``'s text and inline children with ``formats`` on."""
        def emit(text):
            if not text:
                return
            run = paragraph.add_run(text)
            for attr in formats:
                setattr(run, attr, True)

        emit(element.text)
        for child in element:
            if child.tag in _INLINE_FORMAT_TAGS:
                self._add_formatted_runs(paragraph, child, formats)
            elif child.tag == 'a':
                self._add_hyperlink(paragraph, child, formats)
            elif child.tag == 'img':
                self._add_image_run(paragraph, child)
            else:
                emit(self._get_text_content(child))
            emit(child.tail)

    def _add_hyperlink(self, paragraph, element: etree._Element, formats: frozenset) -> None:
        """
        Emit an <a> as a DOCX hyperlink backed by an external relationship.

        The link text keeps its inline formatting. Internal anchors
        (``#bookmark``) and bare bookmark targets (<a id> without href) are
        emitted as plain text: bookmarks are not rebuilt, so a link to one
        would dangle.
        """
        href = (element.get('href') or '').strip()
        if not href or href.startswith('#'):
            self._emit_inline_content(paragraph, element, formats)
            return

        p_elem = paragraph._p
        first_new = len(p_elem)
        self._emit_inline_content(paragraph, element, formats)
        new_runs = [
            child for child in list(p_elem)[first_new:] if child.tag == qn('w:r')
        ]
        if not new_runs:
            return

        r_id = paragraph.part.relate_to(href, RT.HYPERLINK, is_external=True)
        hyperlink = OxmlElement('w:hyperlink')
        hyperlink.set(qn('r:id'), r_id)
        p_elem.insert(list(p_elem).index(new_runs[0]), hyperlink)
        for run_elem in new_runs:
            hyperlink.append(run_elem)
            run = Run(run_elem, paragraph)
            run.font.color.rgb = _HYPERLINK_COLOR
            run.underline = True

    def _add_image_run(self, paragraph, img_element: etree._Element) -> None:
        """
        Embed a base64 data-URI <img> into the paragraph as an inline picture.

        Mammoth emits images as `<img src="data:image/<fmt>;base64,...">`.
        Without this, images would be silently dropped on DOCX reconstruction.
        Non-data-URI sources (http://, file://) are skipped — they cannot be
        reliably resolved at translation time.
        """
        src = img_element.get('src', '')
        if not src.startswith('data:'):
            return
        try:
            header, payload = src.split(',', 1)
        except ValueError:
            return
        if ';base64' not in header:
            return
        try:
            image_bytes = base64.b64decode(payload)
        except (binascii.Error, ValueError):
            return

        width = height = None
        queue = getattr(self, '_image_dims_queue', None)
        if queue:
            width, height = queue.pop(0)

        try:
            paragraph.add_run().add_picture(
                io.BytesIO(image_bytes), width=width, height=height
            )
        except Exception:
            # python-docx raises UnrecognizedImageError for unsupported formats
            return

    def _get_text_content(self, element: etree._Element) -> str:
        """Extract all text content from an element and its children."""
        return ''.join(element.itertext())

    def _get_block_text(
        self,
        element: etree._Element,
        exclude: Set[str] = frozenset()
    ) -> str:
        """
        Extract text content, skipping the subtrees of excluded children.

        The tail text of an excluded child is kept, so `<li>A<ul>…</ul>B</li>`
        with exclude={'ul'} yields "AB" rather than dropping the trailing
        text. Unlike _get_text_content, this is meant for block containers
        whose nested blocks are rebuilt separately.
        """
        parts = [element.text or '']
        for child in element:
            if child.tag not in exclude:
                parts.append(''.join(child.itertext()))
            parts.append(child.tail or '')
        return ''.join(parts)
