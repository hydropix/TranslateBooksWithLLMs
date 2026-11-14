"""
Tests for SrtAdapter.
"""

import pytest
import tempfile
from pathlib import Path

from src.core.adapters import SrtAdapter, TranslationUnit


@pytest.fixture
def sample_srt_content():
    """Create a sample SRT file content."""
    return """1
00:00:01,000 --> 00:00:03,000
Hello, this is the first subtitle.

2
00:00:03,000 --> 00:00:05,000
This is the second subtitle.

3
00:00:05,000 --> 00:00:07,000
And this is the third subtitle.

4
00:00:07,000 --> 00:00:09,000
Fourth subtitle here.

5
00:00:09,000 --> 00:00:11,000
Fifth and final subtitle.
"""


@pytest.fixture
def temp_srt_file(sample_srt_content):
    """Create a temporary SRT file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8') as f:
        f.write(sample_srt_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_output_file():
    """Create a temporary output file path."""
    with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as f:
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


class TestSrtAdapter:
    """Test suite for SrtAdapter."""

    def test_adapter_initialization(self, temp_srt_file, temp_output_file):
        """Test that adapter initializes correctly."""
        adapter = SrtAdapter(
            input_file_path=temp_srt_file,
            output_file_path=temp_output_file,
            config={'lines_per_block': 2}
        )

        assert adapter.input_file_path == Path(temp_srt_file)
        assert adapter.output_file_path == Path(temp_output_file)
        assert adapter.format_name == "srt"

    @pytest.mark.asyncio
    async def test_prepare_for_translation(self, temp_srt_file, temp_output_file):
        """Test SRT file preparation."""
        adapter = SrtAdapter(
            input_file_path=temp_srt_file,
            output_file_path=temp_output_file,
            config={'lines_per_block': 2}
        )

        result = await adapter.prepare_for_translation()

        assert result is True
        assert len(adapter.subtitles) == 5
        assert len(adapter.blocks) > 0

    @pytest.mark.asyncio
    async def test_get_translation_units(self, temp_srt_file, temp_output_file):
        """Test translation unit generation."""
        adapter = SrtAdapter(
            input_file_path=temp_srt_file,
            output_file_path=temp_output_file,
            config={'lines_per_block': 2}
        )

        await adapter.prepare_for_translation()
        units = adapter.get_translation_units()

        assert len(units) > 0
        assert all(isinstance(unit, TranslationUnit) for unit in units)

        # Check first unit structure
        first_unit = units[0]
        assert first_unit.unit_id == "block_0"
        assert '[0]' in first_unit.content  # Local index
        assert 'block_index' in first_unit.metadata
        assert 'local_to_global' in first_unit.metadata

    @pytest.mark.asyncio
    async def test_save_and_reconstruct(self, temp_srt_file, temp_output_file):
        """Test saving translations and reconstructing output."""
        adapter = SrtAdapter(
            input_file_path=temp_srt_file,
            output_file_path=temp_output_file,
            config={'lines_per_block': 2}
        )

        await adapter.prepare_for_translation()
        adapter.get_translation_units()

        # Simulate translation of first block
        # Original: "[0]Hello, this is the first subtitle.\n[1]This is the second subtitle."
        # Translated:
        translated_text = "[0]Bonjour, c'est le premier sous-titre.\n[1]C'est le deuxième sous-titre."

        result = await adapter.save_unit_translation("block_0", translated_text)
        assert result is True

        # Reconstruct output
        output_bytes = await adapter.reconstruct_output()
        output_text = output_bytes.decode('utf-8')

        # Strip width-zero Unicode marks (rendering normalization may insert
        # them) before comparing on visible content.
        import re as _re
        visible_text = _re.sub(r'[​-‍⁠﻿]', '', output_text)

        # Check that translations are present
        assert "Bonjour, c'est le premier sous-titre." in visible_text
        assert "C'est le deuxième sous-titre." in visible_text

        # Check that SRT structure is preserved
        assert "00:00:01,000 --> 00:00:03,000" in output_text

    @pytest.mark.asyncio
    async def test_resume_from_checkpoint(self, temp_srt_file, temp_output_file):
        """Test resuming from checkpoint."""
        adapter = SrtAdapter(
            input_file_path=temp_srt_file,
            output_file_path=temp_output_file,
            config={'lines_per_block': 2}
        )

        await adapter.prepare_for_translation()

        # Simulate checkpoint data
        checkpoint_data = {
            'resume_from_index': 1,
            'chunks': [
                {
                    'status': 'completed',
                    'translated_text': "[0]Bonjour, c'est le premier sous-titre.\n[1]C'est le deuxième sous-titre.",
                    'chunk_data': {
                        'block_index': 0,
                        'local_to_global': {0: 0, 1: 1}
                    }
                }
            ]
        }

        resume_from = await adapter.resume_from_checkpoint(checkpoint_data)

        assert resume_from == 1
        assert len(adapter.translations) > 0

        # Verify translations were restored
        assert 0 in adapter.translations
        assert 1 in adapter.translations

    @pytest.mark.asyncio
    async def test_local_index_renumbering(self, temp_srt_file, temp_output_file):
        """Test that local indices are correctly mapped to global indices."""
        adapter = SrtAdapter(
            input_file_path=temp_srt_file,
            output_file_path=temp_output_file,
            config={'lines_per_block': 2}
        )

        await adapter.prepare_for_translation()
        units = adapter.get_translation_units()

        # Check that each block has local indices starting from 0
        for unit in units:
            content_lines = unit.content.split('\n')
            local_indices = []

            for line in content_lines:
                if line.startswith('['):
                    local_idx = int(line[1:line.index(']')])
                    local_indices.append(local_idx)

            # Verify indices start from 0 and are sequential
            assert local_indices == list(range(len(local_indices)))

            # Verify local_to_global mapping exists
            assert 'local_to_global' in unit.metadata
            local_to_global = unit.metadata['local_to_global']
            assert len(local_to_global) == len(local_indices)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
