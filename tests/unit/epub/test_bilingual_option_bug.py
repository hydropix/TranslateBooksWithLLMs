"""
Regression test for the 'bilingual' option bug.

Issue: https://github.com/hydropix/TranslateBooksWithLLMs/issues/109

The 'bilingual' flag of prompt_options was not passed down to the XHTML
translation. It now travels from the run's prompt options to the settings the
file-level engine translates with.
"""

import pytest
from unittest.mock import MagicMock, patch
from lxml import etree

from src.core.epub import document_translation
from src.core.epub.run_context import EpubRun, FileTask
from tests.job_helpers import make_ctx, make_job


def _doc_root():
    doc_root = etree.Element("html")
    body = etree.SubElement(doc_root, "body")
    p = etree.SubElement(body, "p")
    p.text = "Hello world"
    return doc_root


async def _settings_seen(prompt_options):
    """The settings translate_document hands to the XHTML translation."""
    run = EpubRun(job=make_job("book.epub"), ctx=make_ctx(), llm_client=MagicMock(),
                  max_tokens_per_chunk=1000, prompt_options=prompt_options)
    task = FileTask(href="OEBPS/ch1.xhtml", file_path="/tmp/OEBPS/ch1.xhtml")
    with patch.object(document_translation, "translate_xhtml_document") as mock_translate:
        mock_translate.return_value = (True, MagicMock())
        await document_translation.translate_document(_doc_root(), run, task)
    assert mock_translate.called, "translate_xhtml_document should have been called"
    return mock_translate.call_args[0][1]


class TestBilingualOptionBug:
    """Test suite for the bilingual option bug."""

    @pytest.mark.asyncio
    async def test_bilingual_option_reaches_the_translation(self):
        """bilingual=True in prompt_options is propagated (bug #109)."""
        settings = await _settings_seen({'bilingual': True})
        assert settings.bilingual is True, \
            f"Bug #109: expected bilingual=True, got bilingual={settings.bilingual}"

    @pytest.mark.asyncio
    async def test_bilingual_false_when_not_in_prompt_options(self):
        """bilingual defaults to False when prompt_options is absent."""
        settings = await _settings_seen(None)
        assert settings.bilingual is False

    @pytest.mark.asyncio
    async def test_bilingual_false_when_explicit_in_prompt_options(self):
        """bilingual=False works when explicitly set in prompt_options."""
        settings = await _settings_seen({'bilingual': False})
        assert settings.bilingual is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
