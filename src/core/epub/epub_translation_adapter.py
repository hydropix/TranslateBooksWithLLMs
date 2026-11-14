"""Transitional keyword-argument facade over the file-level EPUB engine.

The book layer (``src/core/epub/translator.py``) still calls this class; the
file-level work lives in ``document_translation`` (``prepare_document`` /
``translate_document``). Delete this module once the book layer calls those
functions directly.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from lxml import etree

from .body_serializer import extract_body_html
from .chunk_settings import ChunkLoopSettings
from .container import TranslationContainer


class EpubTranslationAdapter:
    """Keyword-argument entry points kept for the book layer."""

    def __init__(self, container: Optional[TranslationContainer] = None):
        self.container = container or TranslationContainer()
        self.tag_preserver = self.container.tag_preserver

    def extract_content(
        self,
        source: etree._Element,
        log_callback: Optional[Callable]
    ) -> Tuple[str, Dict[str, Any]]:
        """Extract the body HTML; ``(body_html, context)`` (precount)."""
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
        """Replace the tags with placeholders (precount)."""
        preserver = context['preserver']
        text_with_placeholders, tag_map = preserver.preserve_tags(content)
        placeholder_format = (preserver.placeholder_format.prefix, preserver.placeholder_format.suffix)
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
        """Chunk the placeholder text with the caller's chunk size (precount)."""
        from .document_chunking import create_chunks
        return create_chunks(text, structure_map, max_tokens, log_callback, self.container)

    async def translate_document(
        self,
        doc_root: etree._Element,
        log_callback: Optional[Callable] = None,
        **kwargs
    ) -> Tuple[bool, Any]:
        """Translate one XHTML document in place; ``(success, stats)``."""
        from .document_translation import translate_xhtml_document

        prompt_options = kwargs.get('prompt_options')
        settings = ChunkLoopSettings(
            source_language=kwargs['source_language'],
            target_language=kwargs['target_language'],
            model_name=kwargs['model_name'],
            llm_client=kwargs['llm_client'],
            max_retries=kwargs.get('max_retries', 1),
            context_manager=kwargs.get('context_manager'),
            prompt_options=prompt_options,
            log_callback=log_callback,
            stats_callback=kwargs.get('stats_callback'),
            check_interruption=kwargs.get('check_interruption_callback'),
            parallel_workers=kwargs.get('parallel_workers', 1),
            max_tokens_per_chunk=kwargs.get('max_tokens_per_chunk'),
            bilingual=bool(prompt_options.get('bilingual', False)) if prompt_options else False,
            retry_token_aligned=bool(kwargs.get('retry_token_aligned', False)),
            checkpoint_manager=kwargs.get('checkpoint_manager'),
            translation_id=kwargs.get('translation_id'),
            file_href=kwargs.get('file_href'),
            file_path=kwargs.get('file_href'),
            global_total_chunks=kwargs.get('global_total_chunks'),
            global_completed_chunks=kwargs.get('global_completed_chunks'),
        )
        return await translate_xhtml_document(
            doc_root, settings, resume_state=kwargs.get('resume_state'), container=self.container)
