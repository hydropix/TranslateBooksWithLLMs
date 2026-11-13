"""
Integration test for SRT resume functionality with GenericTranslator and CheckpointManager.
"""

import pytest
import tempfile
from pathlib import Path

from src.core.adapters import SrtAdapter, GenericTranslator
from src.persistence.checkpoint_manager import CheckpointManager


@pytest.fixture
def sample_srt_content():
    """Create a sample SRT file content with multiple subtitles."""
    return """1
00:00:01,000 --> 00:00:03,000
First subtitle in English.

2
00:00:03,000 --> 00:00:05,000
Second subtitle in English.

3
00:00:05,000 --> 00:00:07,000
Third subtitle in English.

4
00:00:07,000 --> 00:00:09,000
Fourth subtitle in English.

5
00:00:09,000 --> 00:00:11,000
Fifth subtitle in English.

6
00:00:11,000 --> 00:00:13,000
Sixth subtitle in English.
"""


@pytest.fixture
def temp_srt_file(sample_srt_content):
    """Create a temporary SRT file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8') as f:
        f.write(sample_srt_content)
        temp_path = f.name

    yield temp_path

    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_output_file():
    """Create a temporary output file path."""
    with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as f:
        temp_path = f.name

    yield temp_path

    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_db():
    """Create a temporary database for checkpoints."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        temp_path = f.name

    yield temp_path

    # Force close any connections before cleanup
    import gc
    gc.collect()

    try:
        Path(temp_path).unlink()
    except PermissionError:
        pass  # Still in use by the process




class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, should_fail_at: int = -1):
        self.call_count = 0
        self.should_fail_at = should_fail_at

    async def generate(self, prompt: str, **kwargs):
        """Mock translation that fails at a specific call."""
        self.call_count += 1

        if self.call_count == self.should_fail_at:
            raise Exception("Simulated LLM failure")

        # Simple mock translation (reverse text)
        return prompt[::-1]


class TestSrtResumeIntegration:
    """Integration tests for SRT resume functionality."""

    @pytest.mark.asyncio
    async def test_full_translation_without_interruption(
        self, temp_srt_file, temp_output_file, temp_db
    ):
        """Test complete translation without interruption."""
        checkpoint_manager = CheckpointManager(
            db_path=temp_db
        )

        adapter = SrtAdapter(
            input_file_path=temp_srt_file,
            output_file_path=temp_output_file,
            config={'lines_per_block': 2}
        )

        GenericTranslator(
            adapter=adapter,
            checkpoint_manager=checkpoint_manager,
            translation_id='test_srt_001'
        )

        # Mock translation function
        async def mock_translate_unit(unit):
            """Simple mock translation."""
            return f"[TRANSLATED]{unit.content}"

        # Prepare
        await adapter.prepare_for_translation()
        units = adapter.get_translation_units()

        # Start job
        checkpoint_manager.start_job(
            translation_id='test_srt_001',
            file_type='srt',
            config={'lines_per_block': 2},
            input_file_path=temp_srt_file
        )

        # Translate all units - must preserve the [index]text format
        for i, unit in enumerate(units):
            # Parse original indices and create translated version preserving format
            lines = unit.content.split('\n')
            translated_lines = []
            for line in lines:
                if line.startswith('[') and ']' in line:
                    idx_end = line.index(']')
                    idx = line[0:idx_end+1]
                    text = line[idx_end+1:]
                    translated_lines.append(f"{idx}[TRANSLATED]{text}")
                else:
                    translated_lines.append(line)

            translated = '\n'.join(translated_lines)
            await adapter.save_unit_translation(unit.unit_id, translated)

            checkpoint_manager.save_checkpoint(
                translation_id='test_srt_001',
                chunk_index=i,
                original_text=unit.content,
                translated_text=translated,
                chunk_data=unit.metadata,
                total_chunks=len(units),
                completed_chunks=i + 1
            )

        # Reconstruct
        output_bytes = await adapter.reconstruct_output()

        # Save final file
        with open(temp_output_file, 'wb') as f:
            f.write(output_bytes)

        # Mark complete
        checkpoint_manager.mark_completed('test_srt_001')

        # Verify
        assert Path(temp_output_file).exists()
        output_text = output_bytes.decode('utf-8')
        assert '[TRANSLATED]' in output_text

        # Verify checkpoint
        job = checkpoint_manager.db.get_job('test_srt_001')
        assert job['status'] == 'completed'

        # Close database
        checkpoint_manager.db.close()

    @pytest.mark.asyncio
    async def test_resume_after_interruption(
        self, temp_srt_file, temp_output_file, temp_db
    ):
        """Test resuming translation after interruption."""
        checkpoint_manager = CheckpointManager(
            db_path=temp_db
        )

        # First pass: translate partially
        adapter1 = SrtAdapter(
            input_file_path=temp_srt_file,
            output_file_path=temp_output_file,
            config={'lines_per_block': 2}
        )

        await adapter1.prepare_for_translation()
        units = adapter1.get_translation_units()

        # Start job
        checkpoint_manager.start_job(
            translation_id='test_srt_resume_001',
            file_type='srt',
            config={'lines_per_block': 2},
            input_file_path=temp_srt_file
        )

        # Translate only first 2 units (simulate interruption)
        for i in range(min(2, len(units))):
            unit = units[i]
            # Preserve the [index]text format
            lines = unit.content.split('\n')
            translated_lines = []
            for line in lines:
                if line.startswith('[') and ']' in line:
                    idx_end = line.index(']')
                    idx = line[0:idx_end+1]
                    text = line[idx_end+1:]
                    translated_lines.append(f"{idx}[PASS1]{text}")
                else:
                    translated_lines.append(line)

            translated = '\n'.join(translated_lines)
            await adapter1.save_unit_translation(unit.unit_id, translated)

            checkpoint_manager.save_checkpoint(
                translation_id='test_srt_resume_001',
                chunk_index=i,
                original_text=unit.content,
                translated_text=translated,
                chunk_data=unit.metadata,
                total_chunks=len(units),
                completed_chunks=i + 1
            )

        # Cleanup adapter1
        await adapter1.cleanup()

        # Second pass: resume from checkpoint
        adapter2 = SrtAdapter(
            input_file_path=temp_srt_file,
            output_file_path=temp_output_file,
            config={'lines_per_block': 2}
        )

        await adapter2.prepare_for_translation()

        # Load checkpoint
        checkpoint_data = checkpoint_manager.load_checkpoint('test_srt_resume_001')
        assert checkpoint_data is not None

        # Resume
        resume_from = await adapter2.resume_from_checkpoint(checkpoint_data)
        assert resume_from == 2  # Should resume from 3rd unit

        # Verify that first 2 units were restored
        assert len(adapter2.translations) >= 2

        # Continue translation from resume point
        units2 = adapter2.get_translation_units()
        for i in range(resume_from, len(units2)):
            unit = units2[i]
            # Preserve the [index]text format
            lines = unit.content.split('\n')
            translated_lines = []
            for line in lines:
                if line.startswith('[') and ']' in line:
                    idx_end = line.index(']')
                    idx = line[0:idx_end+1]
                    text = line[idx_end+1:]
                    translated_lines.append(f"{idx}[PASS2]{text}")
                else:
                    translated_lines.append(line)

            translated = '\n'.join(translated_lines)
            await adapter2.save_unit_translation(unit.unit_id, translated)

            checkpoint_manager.save_checkpoint(
                translation_id='test_srt_resume_001',
                chunk_index=i,
                original_text=unit.content,
                translated_text=translated,
                chunk_data=unit.metadata,
                total_chunks=len(units2),
                completed_chunks=i + 1
            )

        # Reconstruct
        output_bytes = await adapter2.reconstruct_output()

        # Save final file
        with open(temp_output_file, 'wb') as f:
            f.write(output_bytes)

        # Mark complete
        checkpoint_manager.mark_completed('test_srt_resume_001')

        # Verify
        output_text = output_bytes.decode('utf-8')
        assert '[PASS1]' in output_text or '[PASS2]' in output_text

        # Verify all subtitles have translations
        job = checkpoint_manager.db.get_job('test_srt_resume_001')
        assert job['status'] == 'completed'

        # Close database
        checkpoint_manager.db.close()

        # Cleanup
        await adapter2.cleanup()

    @pytest.mark.asyncio
    async def test_checkpoint_preserves_local_to_global_mapping(
        self, temp_srt_file, temp_output_file, temp_db
    ):
        """Test that checkpoint correctly preserves local_to_global mapping for SRT."""
        checkpoint_manager = CheckpointManager(
            db_path=temp_db
        )

        adapter = SrtAdapter(
            input_file_path=temp_srt_file,
            output_file_path=temp_output_file,
            config={'lines_per_block': 2}
        )

        await adapter.prepare_for_translation()
        units = adapter.get_translation_units()

        # Start job
        checkpoint_manager.start_job(
            translation_id='test_srt_mapping_001',
            file_type='srt',
            config={'lines_per_block': 2},
            input_file_path=temp_srt_file
        )

        # Save first unit with metadata
        unit = units[0]
        translated = "[0]First translated subtitle.\n[1]Second translated subtitle."

        await adapter.save_unit_translation(unit.unit_id, translated)

        checkpoint_manager.save_checkpoint(
            translation_id='test_srt_mapping_001',
            chunk_index=0,
            original_text=unit.content,
            translated_text=translated,
            chunk_data=unit.metadata,
            total_chunks=len(units),
            completed_chunks=1
        )

        # Load checkpoint
        checkpoint_data = checkpoint_manager.load_checkpoint('test_srt_mapping_001')

        # Verify metadata is preserved
        first_chunk = checkpoint_data['chunks'][0]
        assert 'local_to_global' in first_chunk['chunk_data']
        assert 'block_index' in first_chunk['chunk_data']

        # Create new adapter and resume
        adapter2 = SrtAdapter(
            input_file_path=temp_srt_file,
            output_file_path=temp_output_file,
            config={'lines_per_block': 2}
        )

        await adapter2.prepare_for_translation()
        await adapter2.resume_from_checkpoint(checkpoint_data)

        # Verify translations were restored with correct global indices
        assert len(adapter2.translations) == 2
        assert 0 in adapter2.translations
        assert 1 in adapter2.translations

        # Close database
        checkpoint_manager.db.close()

        # Cleanup
        await adapter.cleanup()
        await adapter2.cleanup()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
