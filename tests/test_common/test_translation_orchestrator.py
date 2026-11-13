"""
Tests for the generic translation orchestrator.

Tests the unified translation pipeline with mock adapters.
"""

import pytest
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import Mock, patch

from src.core.common.translation_orchestrator import (
    TranslationAdapter,
    GenericTranslationOrchestrator
)


class MockAdapter(TranslationAdapter[str, str]):
    """Mock adapter for testing."""

    def __init__(self):
        self.extract_called = False
        self.preserve_called = False
        self.chunk_called = False
        self.reconstruct_called = False
        self.finalize_called = False

    def extract_content(
        self,
        source: str,
        log_callback: Optional[Callable]
    ) -> Tuple[str, Any]:
        self.extract_called = True
        if log_callback:
            log_callback("extract", "Extracting content")
        return f"extracted_{source}", {"context": "test"}

    def preserve_structure(
        self,
        content: str,
        context: Any,
        log_callback: Optional[Callable]
    ) -> Tuple[str, Dict[str, str], Tuple[str, str]]:
        self.preserve_called = True
        if log_callback:
            log_callback("preserve", "Preserving structure")

        # Simple mock: replace "tag" with placeholder
        text = content.replace("<tag>", "[id0]").replace("</tag>", "[/id0]")
        structure_map = {"id0": "<tag>", "/id0": "</tag>"}
        placeholder_format = ("[", "]")

        return text, structure_map, placeholder_format

    def create_chunks(
        self,
        text: str,
        structure_map: Dict[str, str],
        max_tokens: int,
        log_callback: Optional[Callable]
    ) -> List[Dict]:
        self.chunk_called = True
        if log_callback:
            log_callback("chunk", "Creating chunks")

        # Simple mock: split by sentences
        chunks = [
            {
                'index': i,
                'text': chunk.strip(),
                'context_before': '',
                'context_after': ''
            }
            for i, chunk in enumerate(text.split('.'))
            if chunk.strip()
        ]
        return chunks

    def reconstruct_content(
        self,
        translated_chunks: List[str],
        structure_map: Dict[str, str],
        context: Any
    ) -> str:
        self.reconstruct_called = True

        # Join chunks and restore structure
        full_text = ''.join(translated_chunks)
        for placeholder, original in structure_map.items():
            full_text = full_text.replace(f"[{placeholder}]", original)

        return full_text

    def finalize_output(
        self,
        reconstructed_content: str,
        source: str,
        context: Any,
        log_callback: Optional[Callable]
    ) -> str:
        self.finalize_called = True
        if log_callback:
            log_callback("finalize", "Finalizing output")
        return f"finalized_{reconstructed_content}"


class TestGenericTranslationOrchestrator:
    """Test suite for GenericTranslationOrchestrator."""

    @pytest.mark.asyncio
    async def test_basic_pipeline(self):
        """Test basic translation pipeline."""
        adapter = MockAdapter()
        orchestrator = GenericTranslationOrchestrator(adapter)

        # Mock translation function
        with patch('src.core.epub.xhtml_translator._translate_all_chunks') as mock_translate:
            # Mock returns translated chunks and stats
            from src.core.epub.translation_metrics import TranslationMetrics
            mock_stats = TranslationMetrics()
            mock_translate.return_value = (
                ["translated_chunk1", "translated_chunk2"],
                mock_stats
            )

            # Mock LLM client
            mock_llm = Mock()

            # Run translation
            result, stats = await orchestrator.translate(
                source="test_source",
                source_language="English",
                target_language="French",
                model_name="test-model",
                llm_client=mock_llm,
                max_tokens_per_chunk=450
            )

            # Verify all adapter methods were called
            assert adapter.extract_called
            assert adapter.preserve_called
            assert adapter.chunk_called
            assert adapter.reconstruct_called
            assert adapter.finalize_called

            # Verify result
            assert result.startswith("finalized_")
            assert stats is not None

    @pytest.mark.asyncio
    async def test_empty_content(self):
        """Test handling of empty content."""
        class EmptyAdapter(MockAdapter):
            def extract_content(self, source, log_callback):
                self.extract_called = True
                return "", {"context": "empty"}

        adapter = EmptyAdapter()
        orchestrator = GenericTranslationOrchestrator(adapter)
        mock_llm = Mock()

        result, stats = await orchestrator.translate(
            source="empty_source",
            source_language="English",
            target_language="French",
            model_name="test-model",
            llm_client=mock_llm
        )

        # Should return empty result without error
        assert result.startswith("finalized_")
        assert adapter.extract_called
        assert adapter.finalize_called
        # Other methods should not be called for empty content
        assert not adapter.preserve_called
        assert not adapter.chunk_called

    @pytest.mark.asyncio
    async def test_logging_callbacks(self):
        """Test that logging callbacks are called."""
        adapter = MockAdapter()
        orchestrator = GenericTranslationOrchestrator(adapter)

        # Mock translation function
        with patch('src.core.epub.xhtml_translator._translate_all_chunks') as mock_translate:
            from src.core.epub.translation_metrics import TranslationMetrics
            mock_translate.return_value = (["translated"], TranslationMetrics())

            # Track log calls
            log_calls = []
            def log_callback(event, message):
                log_calls.append((event, message))

            mock_llm = Mock()

            await orchestrator.translate(
                source="test",
                source_language="English",
                target_language="French",
                model_name="test-model",
                llm_client=mock_llm,
                log_callback=log_callback
            )

            # Verify logs were called
            assert len(log_calls) > 0
            assert any("extract" in event.lower() for event, _ in log_calls)

    @pytest.mark.asyncio
    async def test_stats_callback(self):
        """Test that stats callback is called."""
        adapter = MockAdapter()
        orchestrator = GenericTranslationOrchestrator(adapter)

        with patch('src.core.epub.xhtml_translator._translate_all_chunks') as mock_translate:
            from src.core.epub.translation_metrics import TranslationMetrics
            mock_translate.return_value = (["translated"], TranslationMetrics())

            # Track stats calls
            stats_calls = []
            def stats_callback(stats_dict):
                stats_calls.append(stats_dict)

            mock_llm = Mock()

            await orchestrator.translate(
                source="test",
                source_language="English",
                target_language="French",
                model_name="test-model",
                llm_client=mock_llm,
                stats_callback=stats_callback
            )

            # Stats callback should be passed to _translate_all_chunks
            # Verify it was passed as a parameter
            assert mock_translate.called
            call_kwargs = mock_translate.call_args[1]
            assert 'stats_callback' in call_kwargs
            assert call_kwargs['stats_callback'] == stats_callback

    @pytest.mark.asyncio
    async def test_refinement_optional(self):
        """Test optional refinement step."""
        adapter = MockAdapter()
        orchestrator = GenericTranslationOrchestrator(adapter)

        with patch('src.core.epub.xhtml_translator._translate_all_chunks') as mock_translate:
            with patch('src.core.epub.xhtml_translator._refine_epub_chunks') as mock_refine:
                from src.core.epub.translation_metrics import TranslationMetrics
                mock_translate.return_value = (["translated"], TranslationMetrics())
                mock_refine.return_value = ["refined"]

                mock_llm = Mock()

                # Test with refinement enabled
                await orchestrator.translate(
                    source="test",
                    source_language="English",
                    target_language="French",
                    model_name="test-model",
                    llm_client=mock_llm,
                    prompt_options={'refine': True}
                )

                # Refine should be called
                assert mock_refine.called

                # Test with refinement disabled
                mock_refine.reset_mock()
                await orchestrator.translate(
                    source="test",
                    source_language="English",
                    target_language="French",
                    model_name="test-model",
                    llm_client=mock_llm,
                    prompt_options={'refine': False}
                )

                # Refine should not be called
                assert not mock_refine.called

    @pytest.mark.asyncio
    async def test_context_manager_passed(self):
        """Test that context manager is passed to translation."""
        adapter = MockAdapter()
        orchestrator = GenericTranslationOrchestrator(adapter)

        with patch('src.core.epub.xhtml_translator._translate_all_chunks') as mock_translate:
            from src.core.epub.translation_metrics import TranslationMetrics
            mock_translate.return_value = (["translated"], TranslationMetrics())

            mock_llm = Mock()
            mock_context_manager = Mock()

            await orchestrator.translate(
                source="test",
                source_language="English",
                target_language="French",
                model_name="test-model",
                llm_client=mock_llm,
                context_manager=mock_context_manager
            )

            # Verify context_manager was passed to translate_all_chunks
            call_kwargs = mock_translate.call_args[1]
            assert call_kwargs['context_manager'] == mock_context_manager

    @pytest.mark.asyncio
    async def test_max_retries_passed(self):
        """Test that max_retries is passed to translation."""
        adapter = MockAdapter()
        orchestrator = GenericTranslationOrchestrator(adapter)

        with patch('src.core.epub.xhtml_translator._translate_all_chunks') as mock_translate:
            from src.core.epub.translation_metrics import TranslationMetrics
            mock_translate.return_value = (["translated"], TranslationMetrics())

            mock_llm = Mock()

            await orchestrator.translate(
                source="test",
                source_language="English",
                target_language="French",
                model_name="test-model",
                llm_client=mock_llm,
                max_retries=3
            )

            # Verify max_retries was passed
            call_kwargs = mock_translate.call_args[1]
            assert call_kwargs['max_retries'] == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
