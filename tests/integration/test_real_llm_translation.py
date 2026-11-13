"""
Integration tests against a real LLM (Ollama).

These tests use the .env configuration and make real requests to Ollama.
They are marked with @pytest.mark.integration so they can be excluded in CI/CD.

Usage:
    # Run all integration tests
    pytest tests/integration/test_real_llm_translation.py -v

    # Exclude integration tests (for CI/CD)
    pytest tests/ --ignore=tests/integration/test_real_llm_translation.py
"""

import pytest
import asyncio

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def ollama_provider():
    """Create a real Ollama provider using .env configuration."""
    from src.core.llm import create_llm_provider
    from src.config import API_ENDPOINT, DEFAULT_MODEL

    provider = create_llm_provider(
        provider_type="ollama",
        api_endpoint=API_ENDPOINT,
        model=DEFAULT_MODEL
    )
    yield provider

    # Cleanup
    asyncio.get_event_loop().run_until_complete(provider.close())


@pytest.fixture
def sample_texts():
    """Sample texts for translation testing."""
    return {
        "short": "Hello, how are you today?",
        "medium": "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
        "with_names": "John and Mary went to Paris last summer. They visited the Eiffel Tower.",
        "dialogue": '"What time is it?" she asked. "It\'s nearly midnight," he replied.',
    }


class TestOllamaConnection:
    """Test basic Ollama connectivity."""

    @pytest.mark.asyncio
    async def test_ollama_is_reachable(self, ollama_provider):
        """Verify Ollama server is accessible."""
        from src.config import API_ENDPOINT

        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            # Extract base URL from API_ENDPOINT
            base_url = API_ENDPOINT.rsplit('/api/', 1)[0]
            response = await client.get(f"{base_url}/api/tags")
            assert response.status_code == 200, f"Ollama not reachable at {base_url}"


class TestBasicTranslation:
    """Test basic translation functionality with real LLM."""

    @pytest.mark.asyncio
    async def test_simple_translation_en_to_fr(self, ollama_provider, sample_texts):
        """Test simple English to French translation."""
        from src.prompts.prompts import generate_translation_prompt

        prompt_pair = generate_translation_prompt(
            main_content=sample_texts["short"],
            context_before="",
            context_after="",
            previous_translation_context="",
            source_language="English",
            target_language="French",
            has_placeholders=False
        )

        response = await ollama_provider.generate(
            prompt=prompt_pair.user,
            system_prompt=prompt_pair.system
        )

        assert response is not None, "LLM returned no response"
        assert response.content, "LLM response is empty"

        # Extract translation
        translation = ollama_provider.extract_translation(response.content)
        assert translation is not None, f"Could not extract translation from: {response.content[:200]}"

        # Basic validation: should contain French words
        french_indicators = ["bonjour", "comment", "aujourd'hui", "vous", "tu", "ça", "va"]
        translation_lower = translation.lower()
        has_french = any(word in translation_lower for word in french_indicators)
        assert has_french, f"Translation doesn't appear to be French: {translation}"

    @pytest.mark.asyncio
    async def test_preserves_proper_names(self, ollama_provider, sample_texts):
        """Test that proper names are preserved in translation."""
        from src.prompts.prompts import generate_translation_prompt

        prompt_pair = generate_translation_prompt(
            main_content=sample_texts["with_names"],
            context_before="",
            context_after="",
            previous_translation_context="",
            source_language="English",
            target_language="French",
            has_placeholders=False
        )

        response = await ollama_provider.generate(
            prompt=prompt_pair.user,
            system_prompt=prompt_pair.system
        )

        translation = ollama_provider.extract_translation(response.content)
        assert translation is not None

        # Proper names should be preserved
        assert "John" in translation, f"Name 'John' not preserved in: {translation}"
        assert "Mary" in translation, f"Name 'Mary' not preserved in: {translation}"
        assert "Paris" in translation, f"Name 'Paris' not preserved in: {translation}"


class TestSRTTranslation:
    """Test SRT subtitle translation with real LLM."""

    @pytest.fixture
    def sample_srt_content(self):
        """Sample SRT content."""
        return """1
00:00:01,000 --> 00:00:03,000
Hello, welcome to the show.

2
00:00:03,500 --> 00:00:06,000
Today we will discuss technology.

3
00:00:06,500 --> 00:00:09,000
Let's get started!
"""

    @pytest.mark.asyncio
    async def test_srt_translation_preserves_structure(self, ollama_provider, sample_srt_content):
        """Test that SRT translation preserves subtitle structure."""
        from src.prompts.prompts import generate_subtitle_block_prompt

        # Parse subtitles into blocks
        subtitle_blocks = [
            (0, "Hello, welcome to the show."),
            (1, "Today we will discuss technology."),
            (2, "Let's get started!")
        ]

        prompt_pair = generate_subtitle_block_prompt(
            subtitle_blocks=subtitle_blocks,
            previous_translation_block="",
            source_language="English",
            target_language="French"
        )

        response = await ollama_provider.generate(
            prompt=prompt_pair.user,
            system_prompt=prompt_pair.system
        )

        translation = ollama_provider.extract_translation(response.content)
        assert translation is not None, f"Could not extract translation from: {response.content[:200]}"

        # Should preserve index markers
        assert "[0]" in translation, f"Index [0] not found in: {translation}"
        assert "[1]" in translation, f"Index [1] not found in: {translation}"
        assert "[2]" in translation, f"Index [2] not found in: {translation}"


class TestTXTTranslation:
    """Test plain text translation with real LLM."""

    @pytest.mark.asyncio
    async def test_txt_paragraph_translation(self, ollama_provider):
        """Test multi-paragraph text translation."""
        from src.prompts.prompts import generate_translation_prompt

        text = """This is the first paragraph. It contains multiple sentences.

This is the second paragraph. It has different content.

And here is a third paragraph to test preservation of structure."""

        prompt_pair = generate_translation_prompt(
            main_content=text,
            context_before="",
            context_after="",
            previous_translation_context="",
            source_language="English",
            target_language="French",
            has_placeholders=False
        )

        response = await ollama_provider.generate(
            prompt=prompt_pair.user,
            system_prompt=prompt_pair.system
        )

        translation = ollama_provider.extract_translation(response.content)
        assert translation is not None

        # Should preserve paragraph structure (multiple paragraphs)
        paragraphs = [p.strip() for p in translation.split('\n\n') if p.strip()]
        assert len(paragraphs) >= 2, f"Paragraph structure not preserved. Got: {translation}"


class TestEPUBTranslation:
    """Test EPUB-like HTML translation with placeholders."""

    @pytest.mark.asyncio
    async def test_html_with_placeholders(self, ollama_provider):
        """Test translation with HTML tag placeholders."""
        from src.prompts.prompts import generate_translation_prompt

        # Simulate EPUB content with placeholders replacing HTML tags
        text_with_placeholders = "[id0]This is bold text[id1] and [id2]this is italic[id3]."

        prompt_pair = generate_translation_prompt(
            main_content=text_with_placeholders,
            context_before="",
            context_after="",
            previous_translation_context="",
            source_language="English",
            target_language="French",
            has_placeholders=True
        )

        response = await ollama_provider.generate(
            prompt=prompt_pair.user,
            system_prompt=prompt_pair.system
        )

        translation = ollama_provider.extract_translation(response.content)
        assert translation is not None

        # All placeholders should be preserved
        assert "[id0]" in translation, f"Placeholder [id0] not preserved in: {translation}"
        assert "[id1]" in translation, f"Placeholder [id1] not preserved in: {translation}"
        assert "[id2]" in translation, f"Placeholder [id2] not preserved in: {translation}"
        assert "[id3]" in translation, f"Placeholder [id3] not preserved in: {translation}"


class TestOpenAICompatibility:
    """Test OpenAI-compatible API with Ollama backend."""

    @pytest.fixture
    def openai_provider(self):
        """Create an OpenAI-compatible provider pointing to Ollama."""
        from src.core.llm import create_llm_provider
        from src.config import API_ENDPOINT, DEFAULT_MODEL

        # Extract base URL and build OpenAI-compatible endpoint
        # Ollama API: http://host:11434/api/generate
        # OpenAI API: http://host:11434/v1/chat/completions
        base_url = API_ENDPOINT.rsplit('/api/', 1)[0]
        openai_endpoint = f"{base_url}/v1/chat/completions"

        provider = create_llm_provider(
            provider_type="openai",
            api_endpoint=openai_endpoint,
            model=DEFAULT_MODEL,
            api_key="ollama"  # Ollama doesn't require a real key
        )
        return provider

    @pytest.mark.asyncio
    async def test_openai_endpoint_is_reachable(self, openai_provider):
        """Verify Ollama's OpenAI-compatible endpoint is accessible."""
        from src.config import API_ENDPOINT

        import httpx
        base_url = API_ENDPOINT.rsplit('/api/', 1)[0]
        openai_models_url = f"{base_url}/v1/models"

        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(openai_models_url)
            assert response.status_code == 200, f"OpenAI endpoint not reachable at {openai_models_url}"

            # Verify response structure matches OpenAI format
            data = response.json()
            assert "data" in data or "models" in data, "Response doesn't match OpenAI format"

    @pytest.mark.asyncio
    async def test_openai_translation(self, openai_provider):
        """Test translation using OpenAI-compatible API."""
        from src.prompts.prompts import generate_translation_prompt

        prompt_pair = generate_translation_prompt(
            main_content="Good morning, how are you?",
            context_before="",
            context_after="",
            previous_translation_context="",
            source_language="English",
            target_language="French",
            has_placeholders=False
        )

        try:
            response = await openai_provider.generate(
                prompt=prompt_pair.user,
                system_prompt=prompt_pair.system
            )

            assert response is not None, "OpenAI-compatible endpoint returned no response"
            assert response.content, "Response content is empty"

            # Extract and validate translation
            translation = openai_provider.extract_translation(response.content)
            assert translation is not None, f"Could not extract translation: {response.content[:200]}"

            # Should contain French words
            french_words = ["bonjour", "comment", "allez", "vas", "matin"]
            translation_lower = translation.lower()
            has_french = any(word in translation_lower for word in french_words)
            assert has_french, f"Translation doesn't appear to be French: {translation}"

        finally:
            await openai_provider.close()


class TestEdgeCases:
    """Test edge cases and special content."""

    @pytest.mark.asyncio
    async def test_empty_string_handling(self, ollama_provider):
        """Test handling of minimal content."""
        from src.prompts.prompts import generate_translation_prompt

        prompt_pair = generate_translation_prompt(
            main_content="Yes.",
            context_before="",
            context_after="",
            previous_translation_context="",
            source_language="English",
            target_language="French",
            has_placeholders=False
        )

        response = await ollama_provider.generate(
            prompt=prompt_pair.user,
            system_prompt=prompt_pair.system
        )

        translation = ollama_provider.extract_translation(response.content)
        assert translation is not None
        assert len(translation.strip()) > 0

    @pytest.mark.asyncio
    async def test_special_characters(self, ollama_provider):
        """Test handling of special characters."""
        from src.prompts.prompts import generate_translation_prompt

        text = "The price is $100 (€90). Email: test@example.com"

        prompt_pair = generate_translation_prompt(
            main_content=text,
            context_before="",
            context_after="",
            previous_translation_context="",
            source_language="English",
            target_language="French",
            has_placeholders=False
        )

        response = await ollama_provider.generate(
            prompt=prompt_pair.user,
            system_prompt=prompt_pair.system
        )

        translation = ollama_provider.extract_translation(response.content)
        assert translation is not None

        # Special characters should be preserved
        assert "$100" in translation or "100" in translation
        assert "test@example.com" in translation or "example.com" in translation


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
