"""
The default (placeholder) DOCX mode keeps hyperlinks, merged table cells and
headers/footers.

The output document is rebuilt from mammoth's HTML. These used to be dropped
on the way: <a href> became plain text, colspan/rowspan were ignored (every
cell landed in its own column), and headers/footers, which mammoth never
reads, were simply missing from the output.
"""

import base64
import io
import os

import pytest
from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from src.core.docx.converter import DocxHtmlConverter
from src.core.docx.translator import translate_docx_file
from src.core.llm.base import LLMRequest
from tests.characterization.fake_llm import FakeEchoProvider
from tests.job_helpers import make_ctx, make_job

# A 1x1 transparent PNG.
_PNG_1X1 = base64.b64decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA'
    '60e6kgAAAABJRU5ErkJggg=='
)

_WORDS = {
    'Visit': 'Visitez',
    'the site': 'le site',
    'today': "aujourd'hui",
    'Merged across': 'Fusion horizontale',
    'Merged down': 'Fusion verticale',
}


class _WordSwapProvider(FakeEchoProvider):
    """Echo provider that "translates" a few known phrases.

    Placeholders are echoed untouched, so the placeholder pipeline succeeds
    and the output shows which text went through the LLM.
    """

    async def _send(self, request: LLMRequest):
        response = await super()._send(request)
        for source, target in _WORDS.items():
            response.content = response.content.replace(source, target)
        return response


def _add_hyperlink(paragraph, url, text):
    r_id = paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True)
    hyperlink = OxmlElement('w:hyperlink')
    hyperlink.set(qn('r:id'), r_id)
    run = OxmlElement('w:r')
    text_el = OxmlElement('w:t')
    text_el.text = text
    run.append(text_el)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def _build_source(path):
    doc = Document()
    section = doc.sections[0]
    section.header.paragraphs[0].text = 'Quarterly report header'
    section.header.paragraphs[0].add_run().add_picture(io.BytesIO(_PNG_1X1))
    section.footer.paragraphs[0].text = 'Confidential footer'
    section.different_first_page_header_footer = True
    section.first_page_header.paragraphs[0].text = 'Cover page header'

    paragraph = doc.add_paragraph('Visit ')
    _add_hyperlink(paragraph, 'https://example.com/page', 'the site')
    paragraph.add_run(' today.')

    table = doc.add_table(rows=3, cols=3)
    table.style = 'Table Grid'
    for i, row in enumerate(table.rows):
        for j, cell in enumerate(row.cells):
            cell.text = f'r{i}c{j}'
    table.cell(0, 0).merge(table.cell(0, 1)).text = 'Merged across'
    table.cell(1, 2).merge(table.cell(2, 2)).text = 'Merged down'

    doc.save(path)
    return path


async def _translate(source, output):
    result = await translate_docx_file(
        make_job(
            input_path=source,
            output_path=output,
            source_language='English',
            target_language='French',
            model='fake-echo',
            max_tokens_per_chunk=450,
            prompt_options=None,
        ),
        make_ctx(log_callback=None),
        llm_client=_WordSwapProvider(),
        max_retries=1,
    )
    assert result['success'] is True
    return Document(output)


@pytest.fixture
def translated(temp_dir):
    import asyncio
    source = _build_source(os.path.join(temp_dir, 'source.docx'))
    output = os.path.join(temp_dir, 'output.docx')
    return asyncio.run(_translate(source, output))


def _hyperlinks(doc):
    links = []
    for paragraph in doc.paragraphs:
        for hyperlink in paragraph._p.findall(qn('w:hyperlink')):
            rel = paragraph.part.rels[hyperlink.get(qn('r:id'))]
            text = ''.join(t.text or '' for t in hyperlink.iter(qn('w:t')))
            links.append((rel.target_ref, rel.is_external, text))
    return links


def _tc_props(table):
    """(gridSpan, vMerge) per <w:tc>, row by row, from the raw XML."""
    rows = []
    for tr in table._tbl.findall(qn('w:tr')):
        cells = []
        for tc in tr.findall(qn('w:tc')):
            tc_pr = tc.find(qn('w:tcPr'))
            span = vmerge = None
            if tc_pr is not None:
                grid_span = tc_pr.find(qn('w:gridSpan'))
                v_merge = tc_pr.find(qn('w:vMerge'))
                span = grid_span.get(qn('w:val')) if grid_span is not None else None
                if v_merge is not None:
                    vmerge = v_merge.get(qn('w:val')) or 'continue'
            cells.append((span, vmerge))
        rows.append(cells)
    return rows


class TestHyperlinks:

    def test_link_target_and_translated_text_survive(self, translated):
        assert _hyperlinks(translated) == [('https://example.com/page', True, 'le site')]
        assert translated.paragraphs[0].text == "Visitez le site aujourd'hui."

    def test_internal_anchor_is_plain_text(self, temp_dir):
        output = os.path.join(temp_dir, 'anchor.docx')
        DocxHtmlConverter().from_html(
            '<html><body><p>See <a href="#_Toc1">chapter</a> one.</p></body></html>',
            {}, output,
        )
        doc = Document(output)
        assert doc.paragraphs[0].text == 'See chapter one.'
        assert _hyperlinks(doc) == []

    def test_link_inside_bold_keeps_bold(self, temp_dir):
        output = os.path.join(temp_dir, 'bold_link.docx')
        DocxHtmlConverter().from_html(
            '<html><body><p><strong>Go <a href="https://example.org">here</a></strong></p></body></html>',
            {}, output,
        )
        doc = Document(output)
        assert _hyperlinks(doc) == [('https://example.org', True, 'here')]
        link_run = doc.paragraphs[0]._p.find(qn('w:hyperlink')).find(qn('w:r'))
        assert link_run.find(qn('w:rPr')).find(qn('w:b')) is not None


class TestMergedCells:

    def test_gridspan_and_vmerge_survive(self, translated):
        table = translated.tables[0]
        assert _tc_props(table) == [
            [('2', None), (None, None)],
            [(None, None), (None, None), (None, 'restart')],
            [(None, None), (None, None), (None, 'continue')],
        ]
        assert table.cell(0, 0).text == 'Fusion horizontale'
        assert table.cell(0, 2).text == 'r0c2'
        assert table.cell(1, 2).text == 'Fusion verticale'
        assert table.cell(2, 0).text == 'r2c0'
        assert table.cell(2, 1).text == 'r2c1'

    def test_rowspan_shifts_following_rows(self, temp_dir):
        """A cell under a rowspan starts one column further right."""
        output = os.path.join(temp_dir, 'rowspan.docx')
        DocxHtmlConverter().from_html(
            '<html><body><table>'
            '<tr><td rowspan="2">A</td><td>B</td><td>C</td></tr>'
            '<tr><td>D</td><td>E</td></tr>'
            '</table></body></html>',
            {}, output,
        )
        table = Document(output).tables[0]
        assert len(table.columns) == 3
        assert [table.cell(1, c).text for c in range(3)] == ['A', 'D', 'E']

    def test_table_without_spans_is_unchanged(self, temp_dir):
        output = os.path.join(temp_dir, 'plain.docx')
        DocxHtmlConverter().from_html(
            '<html><body><table><tr><td>1</td><td>2</td></tr>'
            '<tr><td>3</td></tr></table></body></html>',
            {}, output,
        )
        table = Document(output).tables[0]
        assert len(table.columns) == 2
        assert _tc_props(table) == [[(None, None), (None, None)], [(None, None), (None, None)]]
        assert [table.cell(1, c).text for c in range(2)] == ['3', '']


class TestHeadersFooters:

    def test_headers_and_footers_are_copied(self, translated):
        section = translated.sections[0]
        assert section.header.paragraphs[0].text == 'Quarterly report header'
        assert section.footer.paragraphs[0].text == 'Confidential footer'
        assert section.different_first_page_header_footer is True
        assert section.first_page_header.paragraphs[0].text == 'Cover page header'

    def test_header_image_is_re_embedded(self, translated):
        header = translated.sections[0].header
        blips = header._element.findall('.//' + qn('a:blip'))
        assert len(blips) == 1
        image_part = header.part.related_parts[blips[0].get(qn('r:embed'))]
        assert image_part.blob == _PNG_1X1

    def test_no_source_means_no_headers(self, temp_dir):
        output = os.path.join(temp_dir, 'bare.docx')
        DocxHtmlConverter().from_html('<html><body><p>x</p></body></html>', {}, output)
        assert Document(output).sections[0].header.is_linked_to_previous is True
