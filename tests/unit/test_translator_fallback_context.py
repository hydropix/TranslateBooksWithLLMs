"""
Unit tests for translator fallback context isolation (issue #170 fix)
"""
import pytest
from unittest.mock import AsyncMock, Mock

from src.core.translator import _make_llm_request_with_adaptive_context
from src.core.llm.base import LLMResponse


class TestTranslatorFallbackContext:
    """Test that raw fallback responses do not contaminate chunk context chain."""

    @pytest.fixture
    def mock_llm_client(self):
        client = Mock()
        client.extract_translation = Mock(side_effect=lambda text: None)
        return client

    @pytest.mark.asyncio
    async def test_successful_extraction_has_no_fallback_flag(self, mock_llm_client):
        """When tags are found, was_fallback must be False."""
        mock_llm_client.generate = AsyncMock(return_value=LLMResponse(
            content="<TRANSLATION>Bonjour</TRANSLATION>",
            prompt_tokens=10,
            completion_tokens=5,
            context_used=15,
            context_limit=2048,
            was_truncated=False,
        ))
        mock_llm_client.extract_translation = Mock(return_value="Bonjour")

        translated, _, response = await _make_llm_request_with_adaptive_context(
            main_content="Hello",
            previous_translation_context="",
            source_language="English",
            target_language="French",
            model="test-model",
            llm_client=mock_llm_client,
            log_callback=None,
            has_placeholders=False,
        )

        assert translated == "Bonjour"
        assert response.was_fallback is False

    @pytest.mark.asyncio
    async def test_plain_text_fallback_sets_fallback_flag(self, mock_llm_client):
        """When extraction fails for plain text, was_fallback must be True."""
        # Response must NOT contain the input text exactly, otherwise echo detection rejects it
        mock_llm_client.generate = AsyncMock(return_value=LLMResponse(
            content="Here is the translation: Bonjour le monde",
            prompt_tokens=10,
            completion_tokens=5,
            context_used=15,
            context_limit=2048,
            was_truncated=False,
        ))

        translated, _, response = await _make_llm_request_with_adaptive_context(
            main_content="Hello world",
            previous_translation_context="",
            source_language="English",
            target_language="French",
            model="test-model",
            llm_client=mock_llm_client,
            log_callback=None,
            has_placeholders=False,
        )

        # The "Here is the translation:" preamble is not part of the translation.
        assert translated == "Bonjour le monde"
        assert response.was_fallback is True

    @pytest.mark.asyncio
    async def test_epub_no_fallback_on_failure(self, mock_llm_client):
        """When has_placeholders=True, failed extraction must return None (no raw fallback)."""
        mock_llm_client.generate = AsyncMock(return_value=LLMResponse(
            content="Here is the translation: Hello world",
            prompt_tokens=10,
            completion_tokens=5,
            context_used=15,
            context_limit=2048,
            was_truncated=False,
        ))

        translated, _, response = await _make_llm_request_with_adaptive_context(
            main_content="Hello world",
            previous_translation_context="",
            source_language="English",
            target_language="French",
            model="test-model",
            llm_client=mock_llm_client,
            log_callback=None,
            has_placeholders=True,
        )

        assert translated is None
        assert response.was_fallback is False

    @pytest.mark.asyncio
    async def test_context_manager_implicit_truncation_retry(self, mock_llm_client):
        """If response starts with <TRANSLATION> but has no closing tag, retry with larger context."""
        call_count = 0
        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(
                    content="<TRANSLATION>\nPartial text without closing tag",
                    prompt_tokens=10,
                    completion_tokens=5,
                    context_used=15,
                    context_limit=2048,
                    was_truncated=False,
                )
            return LLMResponse(
                content="<TRANSLATION>Completed</TRANSLATION>",
                prompt_tokens=10,
                completion_tokens=5,
                context_used=15,
                context_limit=4096,
                was_truncated=False,
            )

        mock_llm_client.generate = side_effect
        # Second call succeeds
        mock_llm_client.extract_translation = Mock(side_effect=lambda text: None if "Partial" in text else "Completed")

        context_manager = Mock()
        context_manager.should_retry_with_larger_context = Mock(return_value=True)
        context_manager.increase_context = Mock()
        context_manager.get_context_size = Mock(return_value=4096)

        translated, _, response = await _make_llm_request_with_adaptive_context(
            main_content="Hello",
            previous_translation_context="",
            source_language="English",
            target_language="French",
            model="test-model",
            llm_client=mock_llm_client,
            log_callback=None,
            has_placeholders=False,
            context_manager=context_manager,
        )

        assert call_count == 2
        assert translated == "Completed"
        assert context_manager.increase_context.called
