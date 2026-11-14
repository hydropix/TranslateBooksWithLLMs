"""
Integration tests for the translator's _build_chunk_glossary_block helper.

Verifies the wiring between prompt_options and the glossary filter/injector,
including respect for a custom GlossaryConfig.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.translator import _build_chunk_glossary_block
from src.core.glossary.models import GlossaryConfig


class TestBuildChunkGlossaryBlock:
    """Tests for _build_chunk_glossary_block."""

    def test_none_prompt_options_returns_empty(self):
        """None prompt_options yields an empty string."""
        result = _build_chunk_glossary_block("any text", None)
        assert result == ""

    def test_empty_prompt_options_returns_empty(self):
        """Empty prompt_options dict yields an empty string."""
        result = _build_chunk_glossary_block("any text", {})
        assert result == ""

    def test_no_glossary_terms_key_returns_empty(self):
        """prompt_options without glossary_terms yields an empty string."""
        result = _build_chunk_glossary_block("any text", {"some_other": "value"})
        assert result == ""

    def test_glossary_terms_no_match_returns_empty(self):
        """glossary_terms with no matches in the chunk yields an empty string."""
        prompt_options = {"glossary_terms": {"unicorn": "licorne"}}
        result = _build_chunk_glossary_block("a regular sentence", prompt_options)
        assert result == ""

    def test_glossary_terms_with_match_returns_block(self):
        """Matching glossary_terms produce a non-empty block with the term line."""
        prompt_options = {"glossary_terms": {"dog": "chien"}}
        result = _build_chunk_glossary_block("My dog is happy", prompt_options)
        assert result != ""
        assert "dog -> chien" in result

    def test_custom_config_case_insensitive_honored(self):
        """A custom GlossaryConfig in prompt_options['glossary_config'] is honored."""
        prompt_options = {
            "glossary_terms": {"FAN": "X"},
            "glossary_config": GlossaryConfig(case_sensitive=False),
        }
        result = _build_chunk_glossary_block("fan is here", prompt_options)
        assert result != ""
        assert "FAN -> X" in result


class TestBuildChunkGlossaryBlockEdgeCases:
    """Edge case tests for _build_chunk_glossary_block."""

    def test_empty_glossary_terms_returns_empty(self):
        """An empty glossary_terms dict yields an empty string."""
        prompt_options = {"glossary_terms": {}}
        result = _build_chunk_glossary_block("any text", prompt_options)
        assert result == ""

    def test_default_config_is_case_sensitive(self):
        """Without a custom config, matching is case-sensitive by default."""
        prompt_options = {"glossary_terms": {"FAN": "X"}}
        # Lowercase 'fan' in chunk should NOT match uppercase 'FAN' term by default
        result = _build_chunk_glossary_block("fan is here", prompt_options)
        assert result == ""


class TestCapWarning:
    """Cap warning emission semantics."""

    def test_cap_log_emitted_once(self):
        """When the cap fires and warn_on_cap=True, log exactly one entry per job.

        Dedupe state lives in the caller-owned runtime_state dict so it never
        leaks into the persisted prompt_options snapshot.
        """
        terms = {"alpha": "A", "beta": "B", "gamma": "G"}
        chunk = "alpha beta gamma"
        prompt_options = {
            "glossary_terms": terms,
            "glossary_config": GlossaryConfig(max_entries=2, warn_on_cap=True),
        }
        runtime_state: dict = {}
        seen = []
        cb = lambda key, msg: seen.append((key, msg))

        # First chunk -> warn
        _build_chunk_glossary_block(chunk, prompt_options, log_callback=cb, runtime_state=runtime_state)
        # Second chunk -> should NOT warn (deduped via runtime_state)
        _build_chunk_glossary_block(chunk, prompt_options, log_callback=cb, runtime_state=runtime_state)

        assert len(seen) == 1
        assert seen[0][0] == "glossary_capped"
        assert "max_entries" in seen[0][1] or "Glossary cap" in seen[0][1]

    def test_cap_not_logged_when_disabled(self):
        terms = {"alpha": "A", "beta": "B", "gamma": "G"}
        prompt_options = {
            "glossary_terms": terms,
            "glossary_config": GlossaryConfig(max_entries=2, warn_on_cap=False),
        }
        seen = []
        _build_chunk_glossary_block(
            "alpha beta gamma", prompt_options,
            log_callback=lambda k, m: seen.append((k, m)),
            runtime_state={},
        )
        assert seen == []

    def test_runtime_state_does_not_pollute_prompt_options(self):
        """The cap-warning flag must live in runtime_state, not prompt_options.

        prompt_options is part of the persisted job snapshot — it must remain
        clean across calls so resumed jobs don't carry stale flags.
        """
        terms = {"alpha": "A", "beta": "B", "gamma": "G"}
        prompt_options = {
            "glossary_terms": terms,
            "glossary_config": GlossaryConfig(max_entries=2, warn_on_cap=True),
        }
        snapshot_keys_before = set(prompt_options.keys())
        runtime_state: dict = {}

        _build_chunk_glossary_block(
            "alpha beta gamma", prompt_options,
            log_callback=lambda k, m: None,
            runtime_state=runtime_state,
        )

        # prompt_options must be unchanged — no internal flags injected.
        assert set(prompt_options.keys()) == snapshot_keys_before
        assert "_glossary_cap_warned" not in prompt_options
        # The flag should be in runtime_state instead.
        assert runtime_state.get("glossary_cap_warned") is True


class TestMetadataPropagation:
    """The translator helper should forward category metadata to the injector."""

    def test_metadata_appears_in_block(self):
        prompt_options = {
            "glossary_terms": {"李凡": "Li Fan"},
            "glossary_term_metadata": {"李凡": {"category": "character"}},
        }
        result = _build_chunk_glossary_block("李凡 is here", prompt_options)
        assert "李凡 -> Li Fan  [character]" in result


class TestRefinementPromptGlossary:
    """generate_refinement_prompt must inject the glossary block in the user
    prompt, just before the DRAFT TO REFINE section (chunk-dynamic content
    lives in the user prompt so the system prompt stays cacheable)."""

    def test_refinement_includes_glossary_block(self):
        from src.prompts.prompts import generate_refinement_prompt

        block = "# GLOSSARY (mandatory translations for this segment)\nfoo -> bar"
        pair = generate_refinement_prompt(
            draft_translation="some draft",
            target_language="French",
            has_placeholders=False,
            glossary_block=block,
        )
        # Block must appear in the user prompt, not the system prompt.
        assert block in pair.user
        assert block not in pair.system
        # And it must come BEFORE the DRAFT TO REFINE section.
        assert pair.user.index(block) < pair.user.index("DRAFT TO REFINE")

    def test_refinement_without_glossary_block_unchanged(self):
        from src.prompts.prompts import generate_refinement_prompt

        pair = generate_refinement_prompt(
            draft_translation="some draft",
            target_language="French",
            has_placeholders=False,
        )
        # No GLOSSARY heading when not provided.
        assert "# GLOSSARY" not in pair.system
        assert "# GLOSSARY" not in pair.user


class TestSubtitleBlockPromptGlossary:
    """generate_subtitle_block_prompt must inject the glossary block in the
    user prompt, just before the SUBTITLES TO TRANSLATE section."""

    def test_subtitle_block_includes_glossary_block(self):
        from src.prompts.prompts import generate_subtitle_block_prompt

        block = "# GLOSSARY (mandatory translations for this segment)\nfoo -> bar"
        pair = generate_subtitle_block_prompt(
            subtitle_blocks=[(0, "Hello"), (1, "World")],
            previous_translation_block="",
            source_language="English",
            target_language="French",
            glossary_block=block,
        )
        assert block in pair.user
        assert block not in pair.system
        assert pair.user.index(block) < pair.user.index("SUBTITLES TO TRANSLATE")

    def test_subtitle_block_without_glossary_block_unchanged(self):
        from src.prompts.prompts import generate_subtitle_block_prompt

        pair = generate_subtitle_block_prompt(
            subtitle_blocks=[(0, "Hello")],
            previous_translation_block="",
            source_language="English",
            target_language="French",
        )
        assert "# GLOSSARY" not in pair.system
        assert "# GLOSSARY" not in pair.user


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
