"""
Markdown format adapter for the generic translation system.

The document is split by :mod:`src.core.markdown` into raw text, copied to
the output byte for byte, and translatable segments (the inline text of
headings, paragraphs, list items, blockquotes and table cells). Segments are
grouped into chunks on block boundaries; protected inline spans (code, URLs,
HTML) are replaced by ``[idN]`` placeholders and never sent to the LLM. The
request uses the existing placeholder prompt mode (the one EPUB uses), so no
Markdown-specific prompt text is added.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from src.core.markdown import (
    decode_chunk, group_chunks, plan_chunk, rebuild, segment_markdown, segments_of,
)
from src.core.markdown.chunks import ChunkPlan

from .format_adapter import FormatAdapter
from .translation_unit import TranslationUnit

_BOM = '﻿'


class MarkdownAdapter(FormatAdapter):
    """Adapter for Markdown (.md, .markdown) files."""

    def __init__(self, input_file_path: str, output_file_path: str, config: Dict[str, Any]):
        super().__init__(input_file_path, output_file_path, config)
        self.text: str = ""
        self.bom: str = ""
        self.pieces: list = []
        self.segments: list = []
        self.chunks: List[List[int]] = []
        self.plans: List[ChunkPlan] = []
        self.rendered: List[str] = []
        self.translations: Dict[int, str] = {}  # segment index -> logical text

    @property
    def placeholder_format(self) -> Optional[Tuple[str, str]]:
        """Requests carry ``[idN]`` placeholders (protected inline spans)."""
        from src.config import PLACEHOLDER_PREFIX, PLACEHOLDER_SUFFIX
        return (PLACEHOLDER_PREFIX, PLACEHOLDER_SUFFIX)

    async def prepare_for_translation(self) -> bool:
        """Read the file, segment it and plan the chunks."""
        try:
            with open(self.input_file_path, 'rb') as f:
                raw = f.read()
            # Bytes are decoded without newline translation so CRLF survives.
            text = raw.decode('utf-8', errors='replace')
            if text.startswith(_BOM):
                self.bom, text = _BOM, text[1:]
            self.text = text

            self.pieces = segment_markdown(text)
            self.segments = segments_of(self.pieces)

            from src.config import MAX_TOKENS_PER_CHUNK
            from src.core.chunking.token_chunker import count_tokens
            max_tokens = self.config.get('max_tokens_per_chunk') or MAX_TOKENS_PER_CHUNK
            self.chunks = group_chunks(self.segments, max_tokens, count_tokens)

            self.plans, self.rendered = [], []
            for indices in self.chunks:
                plan, rendered = plan_chunk(self.segments, indices)
                self.plans.append(plan)
                self.rendered.append(rendered)
            self.translations = {}
            return True

        except Exception as e:
            raise RuntimeError(f"Could not read or parse the Markdown file: {e}") from e

    def get_translation_units(self) -> List[TranslationUnit]:
        """One unit per chunk of segments."""
        return [
            TranslationUnit(
                unit_id=f"chunk_{i}",
                content=rendered,
                metadata={'chunk_index': i, 'total_chunks': len(self.chunks)},
            )
            for i, rendered in enumerate(self.rendered)
        ]

    def _chunk_index(self, unit_id: str) -> Optional[int]:
        try:
            index = int(unit_id.split('_')[1])
        except (IndexError, ValueError):
            return None
        return index if 0 <= index < len(self.plans) else None

    def validate_unit_translation(self, unit_id: str, translated_content: str) -> Optional[str]:
        """Check that every placeholder came back once, in order, and that the
        translation is not far shorter than its source."""
        index = self._chunk_index(unit_id)
        if index is None:
            return None
        plan = self.plans[index]
        _, failed = decode_chunk(plan, translated_content)
        if failed:
            return (f"{len(failed)} of {len(plan.segment_indices)} passages came back "
                    f"with missing, duplicated or reordered [idN] placeholders")
        return self._length_problem(plan, translated_content)

    def _length_problem(self, plan: ChunkPlan, translated_content: str) -> Optional[str]:
        """Same guard as the TXT adapter, on prose only (placeholders removed)."""
        import src.config as cfg
        from src.core.chunking.token_chunker import count_tokens

        min_ratio = cfg.MIN_TRANSLATION_LENGTH_RATIO
        if min_ratio <= 0:
            return None
        source = ''.join(self.segments[i].prose for i in plan.segment_indices)
        source_tokens = count_tokens(source)
        if source_tokens < cfg.MIN_TOKENS_FOR_LENGTH_CHECK:
            return None
        translated_tokens = count_tokens(re.sub(r'\[id\d+\]', '', translated_content))
        if translated_tokens < source_tokens * min_ratio:
            return (f"the translation has {translated_tokens} tokens for "
                    f"{source_tokens} source tokens, so part of the text was "
                    f"probably skipped")
        return None

    def _store(self, index: int, translated_content: str):
        translations, _ = decode_chunk(self.plans[index], translated_content)
        self.translations.update(translations)

    async def save_unit_translation(self, unit_id: str, translated_content: str) -> bool:
        """Keep every segment of the chunk whose placeholders came back intact."""
        try:
            index = self._chunk_index(unit_id)
            if index is None:
                return False
            self._store(index, translated_content)
            return True
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            return False

    async def reconstruct_output(self, bilingual: bool = False) -> bytes:
        """Rebuild the document; untranslated segments keep their source text.

        Args:
            bilingual: If True, each translated segment reads
                ``original / translation`` (kept on the segment's own line
                structure, so headings and table cells stay valid).
        """
        translations = self.translations
        if bilingual:
            translations = {
                idx: f"{self.segments[idx].text} / {text}"
                for idx, text in self.translations.items()
                if text != self.segments[idx].text
            }
        return (self.bom + rebuild(self.pieces, translations)).encode('utf-8')

    async def resume_from_checkpoint(self, checkpoint_data: Dict[str, Any]) -> int:
        """Restore the chunks translated before the interruption."""
        try:
            for chunk_data in checkpoint_data.get('chunks', []):
                if chunk_data.get('status') != 'completed':
                    continue
                metadata = chunk_data.get('chunk_data', {}) or {}
                index = metadata.get('chunk_index')
                translated = chunk_data.get('translated_text')
                if index is None or not translated or not (0 <= index < len(self.plans)):
                    continue
                # Only restore onto the same source chunk: after a re-chunking
                # the index may point at different segments.
                stored_source = chunk_data.get('original_text')
                if stored_source is not None and stored_source != self.rendered[index]:
                    continue
                self._store(index, translated)
            return checkpoint_data.get('resume_from_index', 0)
        except Exception as e:
            raise RuntimeError(f"Could not restore the Markdown checkpoint: {e}") from e

    async def cleanup(self):
        """Nothing to clean up: everything is in memory."""
        pass

    @property
    def format_name(self) -> str:
        return "md"
