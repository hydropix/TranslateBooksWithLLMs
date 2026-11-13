"""
Adaptateur DOCX pour l'orchestrateur générique.

Pipeline:
- Document → HTML (mammoth)
- HTML → placeholders (TagPreserver)
- Placeholders → chunks (HtmlChunker)
- Translation chunks
- Chunks → HTML restored (TagPreserver)
- HTML → Document (python-docx)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from .converter import DocxHtmlConverter
from ..epub.container import TranslationContainer


class DocxTranslationAdapter:
    """
    Adaptateur pour traduire des documents DOCX.

    Note: Utilise le chemin du fichier DOCX (str) comme SourceT plutôt que
    Document directement car la conversion mammoth nécessite un chemin de fichier.
    """

    def __init__(self):
        """Initialise l'adaptateur DOCX."""
        self.converter = DocxHtmlConverter()
        self.container = TranslationContainer()
        self.tag_preserver = self.container.tag_preserver
        self.html_chunker = self.container.chunker

    def extract_content(
        self,
        source: str,
        log_callback: Optional[Callable]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Extrait le contenu HTML depuis le fichier DOCX.

        Args:
            source: Chemin vers le fichier DOCX
            log_callback: Callback de logging

        Returns:
            (html_content, context)
            - html_content: HTML extrait via mammoth
            - context: Dict avec metadata DOCX et tag preserver
        """
        # Convert DOCX → HTML
        html_content, metadata = self.converter.to_html(source)

        if log_callback:
            log_callback("extract_done", f"Extracted {len(html_content)} chars HTML from DOCX")

        context = {
            'metadata': metadata,
            'preserver': self.tag_preserver,
            'source_path': source
        }
        return html_content, context

    def preserve_structure(
        self,
        content: str,
        context: Dict[str, Any],
        log_callback: Optional[Callable]
    ) -> Tuple[str, Dict[str, str], Tuple[str, str]]:
        """
        Préserve les tags HTML via placeholders.

        Réutilise TagPreserver d'EPUB.

        Args:
            content: HTML content
            context: Context dict avec preserver
            log_callback: Callback de logging

        Returns:
            (text_with_placeholders, tag_map, placeholder_format)
        """
        preserver = context['preserver']
        text_with_placeholders, tag_map = preserver.preserve_tags(content)
        placeholder_format = (
            preserver.placeholder_format.prefix,
            preserver.placeholder_format.suffix
        )

        if log_callback:
            log_callback("tags_preserved", f"Preserved {len(tag_map)} tag groups")

        return text_with_placeholders, tag_map, placeholder_format

    def create_chunks(
        self,
        text: str,
        structure_map: Dict[str, str],
        max_tokens: int,
        log_callback: Optional[Callable]
    ) -> List[Dict]:
        """
        Découpe via HtmlChunker.

        Réutilise HtmlChunker d'EPUB.

        Args:
            text: Text with placeholders
            structure_map: Map of placeholders
            max_tokens: Max tokens per chunk
            log_callback: Callback de logging

        Returns:
            List of chunks
        """
        # The caller's max_tokens (the user's chunk size) wins over the
        # container's default configuration.
        chunks = self.container.chunker_for(max_tokens).chunk_html_with_placeholders(
            text, structure_map
        )

        if log_callback:
            log_callback("chunks_created", f"Created {len(chunks)} chunks")

        return chunks

    def reconstruct_content(
        self,
        translated_chunks: List[str],
        structure_map: Dict[str, str],
        context: Dict[str, Any]
    ) -> str:
        """
        Reconstruit le HTML depuis les chunks traduits.

        Réutilise TagPreserver d'EPUB.

        Args:
            translated_chunks: Translated chunks
            structure_map: Map of placeholders
            context: Context dict avec preserver

        Returns:
            Reconstructed HTML
        """
        preserver = context['preserver']
        full_translated_text = ''.join(translated_chunks)
        final_html = preserver.restore_tags(full_translated_text, structure_map)
        return final_html

    def finalize_output(
        self,
        reconstructed_content: str,
        source: str,
        context: Dict[str, Any],
        log_callback: Optional[Callable]
    ) -> bytes:
        """
        Reconstruit DOCX depuis HTML traduit.

        Args:
            reconstructed_content: Reconstructed HTML content
            source: Source file path (not used, metadata from context)
            context: Context dict avec metadata
            log_callback: Callback de logging

        Returns:
            DOCX file as bytes
        """
        # Get metadata from context
        metadata = context['metadata']

        # Convert HTML → DOCX in memory
        # Create temporary file for conversion
        # (python-docx requires a file path or file object)
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode='w', suffix='.docx', delete=False, encoding='utf-8') as tmp:
            tmp_path = tmp.name

        try:
            # Convert and save to temp file
            self.converter.from_html(reconstructed_content, metadata, tmp_path)

            # Read back as bytes
            with open(tmp_path, 'rb') as f:
                docx_bytes = f.read()
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        if log_callback:
            log_callback("docx_rebuilt", f"DOCX document reconstructed ({len(docx_bytes)} bytes)")

        return docx_bytes

    async def translate_content(
        self,
        raw_content: Any,
        source_language: str,
        target_language: str,
        model_name: str,
        llm_client: Any,
        max_tokens_per_chunk: int,
        log_callback: Optional[Callable] = None,
        context_manager: Optional[Any] = None,
        max_retries: int = 1,
        prompt_options: Optional[Dict] = None,
        stats_callback: Optional[Callable] = None,
        checkpoint_manager: Optional[Any] = None,
        translation_id: Optional[str] = None,
        file_href: Optional[str] = None,
        check_interruption_callback: Optional[Callable] = None,
        resume_state: Optional[Any] = None,
        **kwargs
    ) -> Tuple[bytes, Any]:
        """
        Translate DOCX content with checkpoint support.

        Uses _translate_all_chunks_with_checkpoint for chunk-level
        interruption and resume support. A document with no text is rebuilt
        as an empty DOCX without any LLM call.

        Args:
            raw_content: DOCX file path (str)
            source_language: Source language
            target_language: Target language
            model_name: Model name
            llm_client: LLM client
            max_tokens_per_chunk: Max tokens per chunk
            log_callback: Logging callback
            context_manager: Context manager
            max_retries: Max retries
            prompt_options: Prompt options
            stats_callback: Stats callback
            checkpoint_manager: Checkpoint manager for partial state
            translation_id: Translation ID for checkpointing
            file_href: File identifier for checkpointing (use filename for DOCX)
            check_interruption_callback: Interruption check callback
            resume_state: Resume state for partial translation
            **kwargs: Additional arguments

        Returns:
            (docx_bytes, stats)
        """
        from ..epub.xhtml_translator import _translate_all_chunks_with_checkpoint
        from ..epub.translation_metrics import TranslationMetrics

        source_path = raw_content  # DOCX file path
        parallel_workers = kwargs.get('parallel_workers', 1)

        # Use filename as file_href if not provided
        if not file_href:
            import os
            file_href = os.path.basename(source_path)

        # Plain Text Mode bypasses mammoth + tag preservation entirely.
        if prompt_options and prompt_options.get('plain_text_mode'):
            return await self._translate_plain_text(
                source_path=source_path,
                source_language=source_language,
                target_language=target_language,
                model_name=model_name,
                llm_client=llm_client,
                max_tokens_per_chunk=max_tokens_per_chunk,
                log_callback=log_callback,
                context_manager=context_manager,
                prompt_options=prompt_options,
                stats_callback=stats_callback,
                check_interruption_callback=check_interruption_callback,
                parallel_workers=parallel_workers,
                file_href=file_href,
                checkpoint_manager=checkpoint_manager,
                translation_id=translation_id,
                max_retries=max_retries,
                resume_state=resume_state,
            )

        # A Plain Text Mode checkpoint carries no tag map and an empty
        # placeholder format, so replaying it here would restore a prefix under
        # a placeholder scheme that never produced it. Mirror of the guard the
        # plain pipeline applies: each mode resumes only its own state.
        from src.core.common.plain_text_checkpoint import is_plain_text_state
        if is_plain_text_state(resume_state):
            if log_callback:
                log_callback(
                    "docx_resume_ignored",
                    "⚠️ Checkpoint was written in Plain Text Mode; restarting "
                    "this file with placeholder translation"
                )
            resume_state = None

        # === RESUME FROM PARTIAL STATE ===
        if resume_state:
            if log_callback:
                log_callback("docx_resume_partial",
                    f"📂 Resuming DOCX translation from chunk {resume_state.current_chunk_index}/{len(resume_state.chunks)}")

            # Restore state from checkpoint
            chunks = resume_state.chunks
            global_tag_map = resume_state.global_tag_map
            placeholder_format = resume_state.placeholder_format
            translated_chunks = resume_state.translated_chunks.copy()
            start_chunk_index = resume_state.current_chunk_index
            html_content = resume_state.original_body_html

            # Restore statistics
            stats = TranslationMetrics.from_dict(resume_state.stats) if resume_state.stats else TranslationMetrics()
            stats.total_chunks = len(chunks)  # Ensure total_chunks is set from restored chunks

            # Restore tag_preserver
            tag_preserver = self.tag_preserver
            tag_preserver.placeholder_format.prefix = placeholder_format[0]
            tag_preserver.placeholder_format.suffix = placeholder_format[1]

            # Restore context
            metadata = resume_state.doc_metadata
            context = {
                'metadata': metadata,
                'preserver': tag_preserver,
                'source_path': source_path
            }

        else:
            # === NORMAL INITIALIZATION (NO RESUME) ===
            # 1. Extract content
            html_content, context = self.extract_content(source_path, log_callback)
            if not html_content or not html_content.strip():
                if log_callback:
                    log_callback("no_content", "No content to translate")
                return self.finalize_output("", source_path, context, log_callback), TranslationMetrics()

            # 2. Preserve structure
            text_with_placeholders, global_tag_map, placeholder_format = \
                self.preserve_structure(html_content, context, log_callback)

            # 3. Create chunks
            chunks = self.create_chunks(
                text_with_placeholders,
                global_tag_map,
                max_tokens_per_chunk,
                log_callback
            )

            # Initialize variables for new translation
            translated_chunks = []
            start_chunk_index = 0
            stats = TranslationMetrics()
            stats.total_chunks = len(chunks)
            tag_preserver = self.tag_preserver
            metadata = context['metadata']

        # 4. Translation with checkpoint support
        # For DOCX, we have a single file so global stats = local stats
        total_chunks = len(chunks)
        completed_chunks = len(translated_chunks) if translated_chunks else 0

        translated_chunks, stats, was_interrupted = await _translate_all_chunks_with_checkpoint(
            chunks=chunks,
            source_language=source_language,
            target_language=target_language,
            model_name=model_name,
            llm_client=llm_client,
            max_retries=max_retries,
            context_manager=context_manager,
            placeholder_format=placeholder_format,
            log_callback=log_callback,
            stats_callback=stats_callback,
            checkpoint_manager=checkpoint_manager,
            translation_id=translation_id,
            file_href=file_href,
            file_path=source_path,
            check_interruption_callback=check_interruption_callback,
            start_chunk_index=start_chunk_index,
            translated_chunks=translated_chunks,
            global_tag_map=global_tag_map,
            stats=stats,
            prompt_options=prompt_options,
            # Pass global stats for DOCX (single file = global stats)
            global_total_chunks=total_chunks,
            global_completed_chunks=completed_chunks,
            parallel_workers=parallel_workers,
        )

        # If interrupted, save state and return partial result
        if was_interrupted:
            if log_callback:
                log_callback("docx_interrupted", "DOCX translation interrupted - state saved")
            # Return empty bytes to indicate incomplete translation
            return b'', stats

        # 5. Reconstruct content
        if log_callback:
            log_callback("reconstruct_start", "Reconstructing DOCX content")

        reconstructed_html = self.reconstruct_content(
            translated_chunks,
            global_tag_map,
            context
        )

        # 6. Finalize output
        docx_bytes = self.finalize_output(
            reconstructed_html,
            source_path,
            context,
            log_callback
        )

        return docx_bytes, stats

    async def _translate_plain_text(
        self,
        source_path: str,
        source_language: str,
        target_language: str,
        model_name: str,
        llm_client: Any,
        max_tokens_per_chunk: int,
        log_callback: Optional[Callable],
        context_manager: Optional[Any],
        prompt_options: Optional[Dict],
        stats_callback: Optional[Callable],
        check_interruption_callback: Optional[Callable],
        parallel_workers: int = 1,
        file_href: Optional[str] = None,
        checkpoint_manager: Optional[Any] = None,
        translation_id: Optional[str] = None,
        max_retries: int = 1,
        resume_state: Optional[Any] = None,
    ) -> Tuple[bytes, Any]:
        """
        Plain-text-mode DOCX translation: skip mammoth + placeholders.

        Read paragraphs via python-docx, translate as plain text, rebuild
        a fresh Document with the same page setup. Images are reattached
        right after their original paragraph (separate image-only paragraph).

        Segment-level checkpointing reuses the XHTML partial-state machinery:
        a rate limit or an interruption leaves a resumable checkpoint behind,
        and `resume_state` restarts at the first unattempted segment. Unlike
        EPUB there is no post-save seam to clean up after us, so the state is
        deleted here once the document has been rebuilt.
        """
        import os
        import tempfile

        from .plain_extractor import extract_plain_paragraphs, build_minimal_docx
        from src.core.common.plain_text_pipeline import translate_paragraphs_plain
        from src.core.common.plain_text_checkpoint import (
            build_plain_checkpoint_hook,
            delete_plain_checkpoint,
            resume_plain_segments,
        )
        from ..epub.translation_metrics import TranslationMetrics

        bilingual_flag = bool(prompt_options.get('bilingual')) if prompt_options else False

        content = extract_plain_paragraphs(source_path)

        if log_callback:
            log_callback(
                "plain_text_extracted",
                f"📝 Plain Text Mode (DOCX): {len(content.paragraphs_text)} paragraphs, "
                f"{sum(len(v) for v in content.images_by_paragraph.values())} images anchored"
            )

        if not content.paragraphs_text:
            if log_callback:
                log_callback("plain_text_empty_docx", "⚠️ No paragraphs found in DOCX")
            return b'', TranslationMetrics()

        paragraph_count = len(content.paragraphs_text)
        resume_segments, resume_translated = resume_plain_segments(
            resume_state, paragraph_count, log_callback
        )
        checkpoint_hook = build_plain_checkpoint_hook(
            checkpoint_manager=checkpoint_manager,
            translation_id=translation_id,
            file_href=file_href,
            source_language=source_language,
            target_language=target_language,
            model_name=model_name,
            max_tokens_per_chunk=max_tokens_per_chunk,
            max_retries=max_retries,
            paragraph_count=paragraph_count,
            prompt_options=prompt_options,
            bilingual=bilingual_flag,
        )

        translated, stats, was_interrupted = await translate_paragraphs_plain(
            paragraphs=content.paragraphs_text,
            source_language=source_language,
            target_language=target_language,
            model_name=model_name,
            llm_client=llm_client,
            max_tokens_per_chunk=max_tokens_per_chunk,
            log_callback=log_callback,
            stats_callback=stats_callback,
            context_manager=context_manager,
            check_interruption_callback=check_interruption_callback,
            prompt_options=prompt_options,
            parallel_workers=parallel_workers,
            resume_segments=resume_segments,
            resume_translated=resume_translated,
            checkpoint_hook=checkpoint_hook,
        )

        if was_interrupted:
            if log_callback:
                log_callback("docx_plain_text_interrupted", "DOCX plain-text translation interrupted")
            return b'', stats

        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            build_minimal_docx(
                translated_paragraphs=translated,
                content=content,
                output_path=tmp_path,
                bilingual=bilingual_flag,
            )
            with open(tmp_path, 'rb') as f:
                docx_bytes = f.read()
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        if log_callback:
            log_callback("docx_plain_text_rebuilt", f"📄 Plain-text DOCX rebuilt ({len(docx_bytes)} bytes)")

        # The document is complete: drop the segment checkpoint so a later
        # resume of the same job does not replay a stale state.
        delete_plain_checkpoint(checkpoint_manager, translation_id, file_href, log_callback)

        return docx_bytes, stats
