"""
Test reproducing the 'bilingual' option bug in EpubTranslationAdapter.

Issue: https://github.com/hydropix/TranslateBooksWithLLMs/issues/109

The 'bilingual' parameter in prompt_options was not passed to translate_xhtml_simplified.
"""

import pytest
from unittest.mock import patch, MagicMock
from lxml import etree

from src.core.epub.epub_translation_adapter import EpubTranslationAdapter


class TestBilingualOptionBug:
    """Test suite for the bilingual option bug."""

    @pytest.mark.asyncio
    async def test_bilingual_option_passed_to_translate_xhtml_simplified(self):
        """
        Test that the 'bilingual' option in prompt_options is correctly passed
        to translate_xhtml_simplified.

        This test reproduces bug #109 where bilingual=True in prompt_options
        was not propagated to the translation function.
        """
        # Create an adapter
        adapter = EpubTranslationAdapter()

        # Create a minimal XHTML document
        doc_root = etree.Element("html")
        body = etree.SubElement(doc_root, "body")
        p = etree.SubElement(body, "p")
        p.text = "Hello world"

        # Mock the LLM client
        mock_llm = MagicMock()

        # Mock translate_xhtml_simplified to capture the arguments
        # Note: the import happens inside the method, so we patch the source module
        with patch('src.core.epub.xhtml_translator.translate_xhtml_simplified') as mock_translate:
            mock_translate.return_value = (True, MagicMock())

            # Call translate_content with bilingual=True in prompt_options
            prompt_options = {'bilingual': True}
            
            await adapter.translate_content(
                raw_content=doc_root,
                source_language="English",
                target_language="French",
                model_name="test-model",
                llm_client=mock_llm,
                max_tokens_per_chunk=1000,
                prompt_options=prompt_options,
            )
            
            # Verify translate_xhtml_simplified was called with bilingual=True
            assert mock_translate.called, "translate_xhtml_simplified should have been called"

            call_kwargs = mock_translate.call_args[1]

            # This test verifies that bug #109 is fixed
            # BEFORE the fix: 'bilingual' was missing from call_kwargs or was False
            # AFTER the fix: 'bilingual' should be True
            assert 'bilingual' in call_kwargs, \
                "Bug #109: 'bilingual' parameter not passed to translate_xhtml_simplified"
            assert call_kwargs['bilingual'] is True, \
                f"Bug #109: expected bilingual=True, got bilingual={call_kwargs['bilingual']}"

    @pytest.mark.asyncio
    async def test_bilingual_false_when_not_in_prompt_options(self):
        """
        Test that bilingual defaults to False when absent from prompt_options.
        """
        adapter = EpubTranslationAdapter()
        
        doc_root = etree.Element("html")
        body = etree.SubElement(doc_root, "body")
        p = etree.SubElement(body, "p")
        p.text = "Hello world"
        
        mock_llm = MagicMock()
        
        with patch('src.core.epub.xhtml_translator.translate_xhtml_simplified') as mock_translate:
            mock_translate.return_value = (True, MagicMock())
            
            # Call without prompt_options
            await adapter.translate_content(
                raw_content=doc_root,
                source_language="English",
                target_language="French",
                model_name="test-model",
                llm_client=mock_llm,
                max_tokens_per_chunk=1000,
                prompt_options=None,
            )
            
            assert mock_translate.called
            call_kwargs = mock_translate.call_args[1]
            
            # By default, bilingual should be False
            assert 'bilingual' in call_kwargs
            assert call_kwargs['bilingual'] is False

    @pytest.mark.asyncio
    async def test_bilingual_false_when_explicit_in_prompt_options(self):
        """
        Test that bilingual=False works when explicitly set in prompt_options.
        """
        adapter = EpubTranslationAdapter()
        
        doc_root = etree.Element("html")
        body = etree.SubElement(doc_root, "body")
        p = etree.SubElement(body, "p")
        p.text = "Hello world"
        
        mock_llm = MagicMock()
        
        with patch('src.core.epub.xhtml_translator.translate_xhtml_simplified') as mock_translate:
            mock_translate.return_value = (True, MagicMock())
            
            prompt_options = {'bilingual': False}
            
            await adapter.translate_content(
                raw_content=doc_root,
                source_language="English",
                target_language="French",
                model_name="test-model",
                llm_client=mock_llm,
                max_tokens_per_chunk=1000,
                prompt_options=prompt_options,
            )
            
            assert mock_translate.called
            call_kwargs = mock_translate.call_args[1]
            
            assert 'bilingual' in call_kwargs
            assert call_kwargs['bilingual'] is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
