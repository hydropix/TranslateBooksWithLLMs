"""
Adaptateur EPUB pour l'orchestrateur générique.

Migre le code existant de xhtml_translator.py vers le nouveau pattern.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
from lxml import etree

from ..common.translation_orchestrator import TranslationAdapter
from .body_serializer import extract_body_html, replace_body_content
from .container import TranslationContainer
from .exceptions import XmlParsingError, BodyExtractionError


class EpubTranslationAdapter(TranslationAdapter[etree._Element, bool]):
    """
    Adaptateur pour traduire des documents XHTML/EPUB.

    Réutilise tous les modules EPUB existants via le nouveau pattern.
    """

    def __init__(self, container: Optional[TranslationContainer] = None):
        """
        Initialise l'adaptateur EPUB.

        Args:
            container: Container avec composants réutilisables
        """
        self.container = container or TranslationContainer()
        self.tag_preserver = self.container.tag_preserver
        self.html_chunker = self.container.chunker

    def extract_content(
        self,
        source: etree._Element,
        log_callback: Optional[Callable]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Extrait le body HTML depuis l'ElementTree.

        Args:
            source: etree._Element root
            log_callback: Callback de logging

        Returns:
            (body_html, context)
            - body_html: HTML content from body
            - context: Dict avec body_element et preserver
        """
        body_html, body_element = extract_body_html(source, log_callback=log_callback)

        if log_callback:
            log_callback("body_extracted", f"Extracted {len(body_html)} chars from XHTML body")

        context = {
            'body_element': body_element,
            'preserver': self.tag_preserver,
            'doc_root': source
        }
        return body_html, context

    def preserve_structure(
        self,
        content: str,
        context: Dict[str, Any],
        log_callback: Optional[Callable]
    ) -> Tuple[str, Dict[str, str], Tuple[str, str]]:
        """
        Préserve les tags HTML via placeholders.

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

        Args:
            text: Text with placeholders
            structure_map: Map of placeholders
            max_tokens: Max tokens per chunk
            log_callback: Callback de logging

        Returns:
            List of chunks
        """
        chunks = self.html_chunker.chunk_html_with_placeholders(
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
        Reconstruit le HTML.

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
        source: etree._Element,
        context: Dict[str, Any],
        log_callback: Optional[Callable]
    ) -> bool:
        """
        Replace body content in XHTML.

        Args:
            reconstructed_content: Reconstructed HTML content
            source: Source etree._Element (not used, body_element from context)
            context: Context dict avec body_element
            log_callback: Callback de logging

        Returns:
            True if successful, False otherwise
        """
        body_element = context['body_element']

        try:
            replace_body_content(body_element, reconstructed_content)

            if log_callback:
                log_callback("body_replaced", "Body content replaced successfully")

            return True
        except (XmlParsingError, BodyExtractionError) as e:
            if log_callback:
                log_callback("replace_body_error", str(e))
            return False

    async def translate_content(
        self,
        raw_content: Any,
        structure_map: Dict[str, Any],
        context: Dict[str, Any],
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
    ) -> Tuple[bool, Any]:
        """
        Translate EPUB XHTML content with checkpoint support.

        This method bypasses the generic orchestrator and calls translate_xhtml_simplified
        directly to leverage chunk-level checkpoint support.

        Args:
            raw_content: etree._Element (doc_root)
            structure_map: Not used (kept for interface compatibility)
            context: Context dict with preservation info
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
            file_href: File href for checkpointing
            check_interruption_callback: Interruption check callback
            resume_state: Resume state for partial translation
            **kwargs: Additional arguments

        Returns:
            (success, stats)
        """
        from .xhtml_translator import translate_xhtml_simplified

        doc_root = raw_content

        # Extract global_stats from kwargs if provided
        global_total_chunks = kwargs.get('global_total_chunks')
        global_completed_chunks = kwargs.get('global_completed_chunks')
        parallel_workers = kwargs.get('parallel_workers', 1)

        # Extract bilingual flag from prompt_options (bug fix #109)
        bilingual_flag = prompt_options.get('bilingual', False) if prompt_options else False

        # Plain Text Mode bypasses the placeholder pipeline entirely.
        if prompt_options and prompt_options.get('plain_text_mode'):
            return await self._translate_plain_text(
                doc_root=doc_root,
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
                bilingual_flag=bilingual_flag,
                file_href=file_href,
                parallel_workers=parallel_workers,
            )

        success, stats = await translate_xhtml_simplified(
            doc_root=doc_root,
            source_language=source_language,
            target_language=target_language,
            model_name=model_name,
            llm_client=llm_client,
            max_tokens_per_chunk=max_tokens_per_chunk,
            log_callback=log_callback,
            context_manager=context_manager,
            max_retries=max_retries,
            container=self.container,
            prompt_options=prompt_options,
            bilingual=bilingual_flag,
            checkpoint_manager=checkpoint_manager,
            translation_id=translation_id,
            file_href=file_href,
            check_interruption_callback=check_interruption_callback,
            resume_state=resume_state,
            stats_callback=stats_callback,
            global_total_chunks=global_total_chunks,
            global_completed_chunks=global_completed_chunks,
            parallel_workers=parallel_workers,
        )

        return success, stats

    async def _translate_plain_text(
        self,
        doc_root: etree._Element,
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
        bilingual_flag: bool,
        file_href: Optional[str],
        parallel_workers: int = 1,
    ) -> Tuple[bool, Any]:
        """
        Plain-text-mode translation path: skip placeholders entirely.

        Extract body as a list of plain paragraphs (anchoring images), translate
        them via the common plain-text pipeline, then rewrite body with a flat
        structure (block tags preserved, images reattached after their parent
        paragraph, inline formatting dropped).
        """
        from .plain_extractor import extract_plain_paragraphs, replace_body_with_paragraphs
        from .translation_metrics import TranslationMetrics
        from src.core.common.plain_text_pipeline import translate_paragraphs_plain

        body = doc_root.find('.//{http://www.w3.org/1999/xhtml}body')
        if body is None:
            body = doc_root.find('.//body')

        if body is None:
            if log_callback:
                log_callback("plain_text_no_body", f"⚠️ {file_href or 'document'}: no <body> found, skipping")
            return False, TranslationMetrics()

        paragraphs_text, paragraphs_tag, images_by_paragraph = extract_plain_paragraphs(body)

        if log_callback:
            log_callback(
                "plain_text_extracted",
                f"📝 Plain Text Mode: {len(paragraphs_text)} paragraphs, "
                f"{sum(len(v) for v in images_by_paragraph.values())} images anchored"
            )

        translated, stats, was_interrupted = await translate_paragraphs_plain(
            paragraphs=paragraphs_text,
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
        )

        if was_interrupted:
            # Caller (EPUB translator) treats failed translation as keeping original;
            # we leave the body untouched so the partial output keeps the source text.
            return False, stats

        replace_body_with_paragraphs(
            body_element=body,
            translated_paragraphs=translated,
            paragraphs_tag=paragraphs_tag,
            images_by_paragraph=images_by_paragraph,
            bilingual=bilingual_flag,
            source_paragraphs=paragraphs_text if bilingual_flag else None,
        )

        return True, stats
