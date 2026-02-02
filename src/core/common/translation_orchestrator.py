"""
Orchestrateur générique pour la traduction avec placeholders.

Implémente un pipeline unifié réutilisable pour EPUB, PDF, DOCX, ODT, etc.
via le pattern adapter.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Generic
from abc import ABC, abstractmethod

SourceT = TypeVar('SourceT')  # etree._Element, Document, fitz.Page, etc.
ResultT = TypeVar('ResultT')  # bool, bytes, List[Span], etc.


class TranslationAdapter(ABC, Generic[SourceT, ResultT]):
    """
    Interface adaptateur pour différents formats de document.

    Chaque format (EPUB, DOCX, PDF, ODT, etc.) implémente cette interface
    pour adapter l'orchestrateur générique à ses besoins spécifiques.
    """

    @abstractmethod
    def extract_content(
        self,
        source: SourceT,
        log_callback: Optional[Callable]
    ) -> Tuple[str, Any]:
        """
        Extrait le contenu brut à traduire.

        Args:
            source: Source spécifique au format (etree._Element, Document, etc.)
            log_callback: Callback de logging

        Returns:
            (raw_content, preservation_context)
            - raw_content: Contenu brut (HTML pour EPUB/DOCX, spans pour PDF)
            - preservation_context: Contexte pour la préservation
        """
        pass

    @abstractmethod
    def preserve_structure(
        self,
        content: str,
        context: Any,
        log_callback: Optional[Callable]
    ) -> Tuple[str, Dict[str, str], Tuple[str, str]]:
        """
        Préserve la structure via placeholders.

        Args:
            content: Contenu brut
            context: Contexte de préservation
            log_callback: Callback de logging

        Returns:
            (text_with_placeholders, structure_map, placeholder_format)
            - text_with_placeholders: Texte avec placeholders
            - structure_map: Map placeholder → contenu original
            - placeholder_format: (prefix, suffix) pour les placeholders
        """
        pass

    @abstractmethod
    def create_chunks(
        self,
        text: str,
        structure_map: Dict[str, str],
        max_tokens: int,
        log_callback: Optional[Callable]
    ) -> List[Dict]:
        """
        Découpe le texte en chunks intelligents.

        Args:
            text: Texte avec placeholders
            structure_map: Map des placeholders
            max_tokens: Tokens max par chunk
            log_callback: Callback de logging

        Returns:
            Liste de chunks au format standard
        """
        pass

    @abstractmethod
    def reconstruct_content(
        self,
        translated_chunks: List[str],
        structure_map: Dict[str, str],
        context: Any
    ) -> str:
        """
        Reconstruit le contenu depuis les chunks traduits.

        Args:
            translated_chunks: Chunks traduits
            structure_map: Map des placeholders
            context: Contexte de préservation

        Returns:
            Contenu reconstruit
        """
        pass

    @abstractmethod
    def finalize_output(
        self,
        reconstructed_content: str,
        source: SourceT,
        context: Any,
        log_callback: Optional[Callable]
    ) -> ResultT:
        """
        Finalise la sortie (replace body, write file, etc.).

        Args:
            reconstructed_content: Contenu reconstruit
            source: Source originale
            context: Contexte de préservation
            log_callback: Callback de logging

        Returns:
            Résultat final (format dépend de l'adaptateur)
        """
        pass


class GenericTranslationOrchestrator(Generic[SourceT, ResultT]):
    """
    Orchestrateur générique pour la traduction avec placeholders.

    Pipeline unifié:
    1. Extract content (via adapter)
    2. Preserve structure (tags/spans → placeholders)
    3. Chunk intelligently (HTML-aware / span-aware)
    4. Translate chunks (with 3-phase fallback)
    5. (Optional) Refine translation
    6. Reconstruct content (restore structure)
    7. Finalize output (save file)

    Réutilisable pour EPUB, PDF, DOCX, ODT, etc. via des adaptateurs.
    """

    def __init__(self, adapter: TranslationAdapter[SourceT, ResultT]):
        """
        Initialise l'orchestrateur avec un adaptateur.

        Args:
            adapter: Adaptateur format-spécifique (EpubAdapter, DocxAdapter, etc.)
        """
        self.adapter = adapter

    async def translate(
        self,
        source: SourceT,
        source_language: str,
        target_language: str,
        model_name: str,
        llm_client: Any,
        max_tokens_per_chunk: int = 450,
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
    ) -> Tuple[ResultT, Any]:
        """
        Pipeline de traduction générique avec support de checkpoint optionnel.

        Args:
            source: Source spécifique au format (etree._Element, Document, etc.)
            source_language: Langue source
            target_language: Langue cible
            model_name: Modèle LLM
            llm_client: Client LLM
            max_tokens_per_chunk: Tokens max par chunk
            log_callback: Callback de logging
            context_manager: Gestionnaire de contexte adaptatif
            max_retries: Tentatives max de traduction
            prompt_options: Options de prompt (refinement, etc.)
            stats_callback: Callback pour mises à jour des statistiques en temps réel
            checkpoint_manager: Gestionnaire de checkpoint (optionnel, EPUB uniquement)
            translation_id: ID de traduction (optionnel, EPUB uniquement)
            file_href: Chemin relatif du fichier (optionnel, EPUB uniquement)
            check_interruption_callback: Callback de vérification d'interruption (optionnel)
            resume_state: État partiel pour reprise (optionnel, EPUB uniquement)
            **kwargs: Paramètres additionnels passés à l'adaptateur (e.g., global_total_chunks, global_completed_chunks)

        Returns:
            (result, stats)
            - result: Résultat final (format dépend de l'adaptateur)
            - stats: Métriques de traduction

        Note:
            Les nouveaux paramètres sont optionnels et utilisés uniquement par l'adaptateur EPUB.
            Les autres adaptateurs (TXT, SRT, DOCX) les ignorent.
        """
        from ..epub.translation_metrics import TranslationMetrics

        # Check if adapter has translate_content method (for EPUB with checkpoint support)
        if hasattr(self.adapter, 'translate_content'):
            # Use adapter's custom translation method (EPUB with checkpoint support)
            raw_content, preservation_context = self.adapter.extract_content(
                source, log_callback
            )

            if not raw_content or not raw_content.strip():
                if log_callback:
                    log_callback("no_content", "No content to translate")
                empty_result = self.adapter.finalize_output("", source, preservation_context, log_callback)
                return empty_result, TranslationMetrics()

            # Preserve structure to get structure_map (needed for translate_content signature)
            text_with_placeholders, structure_map, placeholder_format = \
                self.adapter.preserve_structure(
                    raw_content, preservation_context, log_callback
                )

            # Call adapter's translate_content with checkpoint parameters
            # Pass through any additional kwargs (e.g., global_total_chunks, global_completed_chunks)
            success, stats = await self.adapter.translate_content(
                raw_content=source,
                structure_map=structure_map,
                context=preservation_context,
                source_language=source_language,
                target_language=target_language,
                model_name=model_name,
                llm_client=llm_client,
                max_tokens_per_chunk=max_tokens_per_chunk,
                log_callback=log_callback,
                context_manager=context_manager,
                max_retries=max_retries,
                prompt_options=prompt_options,
                stats_callback=stats_callback,
                checkpoint_manager=checkpoint_manager,
                translation_id=translation_id,
                file_href=file_href,
                check_interruption_callback=check_interruption_callback,
                resume_state=resume_state,
                **kwargs
            )

            return success, stats

        # Standard pipeline (for formats without checkpoint support)
        # 1. Extract content
        if log_callback:
            log_callback("extract_start", "Extracting content")

        raw_content, preservation_context = self.adapter.extract_content(
            source, log_callback
        )

        if not raw_content or not raw_content.strip():
            if log_callback:
                log_callback("no_content", "No content to translate")
            empty_result = self.adapter.finalize_output("", source, preservation_context, log_callback)
            return empty_result, TranslationMetrics()

        # 2. Preserve structure
        if log_callback:
            log_callback("preserve_start", "Preserving structure")

        text_with_placeholders, structure_map, placeholder_format = \
            self.adapter.preserve_structure(
                raw_content, preservation_context, log_callback
            )

        # 3. Chunk
        if log_callback:
            log_callback("chunk_start", "Creating chunks")

        chunks = self.adapter.create_chunks(
            text_with_placeholders, structure_map, max_tokens_per_chunk, log_callback
        )

        if log_callback:
            log_callback("chunks_created", f"Created {len(chunks)} chunks")

        # 4. Translation (réutilise _translate_all_chunks d'EPUB)
        from ..epub.xhtml_translator import _translate_all_chunks

        translated_chunks, stats = await _translate_all_chunks(
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
            check_interruption_callback=check_interruption_callback,
            prompt_options=prompt_options
        )

        # 5. Refinement (optional)
        enable_refinement = prompt_options and prompt_options.get('refine')
        if enable_refinement and translated_chunks:
            from ..epub.xhtml_translator import _refine_epub_chunks

            if log_callback:
                log_callback("refine_start", "Refining translation")

            refined_result = await _refine_epub_chunks(
                translated_chunks=translated_chunks,
                chunks=chunks,
                target_language=target_language,
                model_name=model_name,
                llm_client=llm_client,
                context_manager=context_manager,
                placeholder_format=placeholder_format,
                log_callback=log_callback,
                prompt_options=prompt_options
            )
            if refined_result:
                translated_chunks = refined_result

        # 6. Reconstruct
        if log_callback:
            log_callback("reconstruct_start", "Reconstructing content")

        reconstructed = self.adapter.reconstruct_content(
            translated_chunks, structure_map, preservation_context
        )

        # 7. Finalize
        result = self.adapter.finalize_output(
            reconstructed, source, preservation_context, log_callback
        )

        # 8. Report stats
        if log_callback:
            stats.log_summary(log_callback)
        return result, stats
