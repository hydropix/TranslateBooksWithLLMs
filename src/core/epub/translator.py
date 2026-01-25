"""
EPUB translation orchestration using generic orchestrator

This module coordinates the translation pipeline for EPUB files using the
unified generic orchestrator approach:
1. Extract EPUB to temp directory
2. Parse each XHTML file
3. Translate each document using GenericTranslationOrchestrator
4. Save the modified EPUB

Refactored to use the same pattern as DOCX for consistency and maintainability.
"""
import os
import zipfile
import tempfile
import aiofiles
from typing import Dict, Any, Optional, Callable, Tuple, List
from pathlib import Path
from lxml import etree

from src.config import (
    NAMESPACES, DEFAULT_MODEL, API_ENDPOINT,
    MAX_TOKENS_PER_CHUNK, THINKING_MODELS, ADAPTIVE_CONTEXT_INITIAL_THINKING,
    MAX_TRANSLATION_ATTEMPTS, ATTRIBUTION_ENABLED, GENERATOR_NAME, GENERATOR_SOURCE
)
from ..common.translation_orchestrator import GenericTranslationOrchestrator
from .epub_translation_adapter import EpubTranslationAdapter
from ..post_processor import clean_residual_tag_placeholders
from ..context_optimizer import AdaptiveContextManager, INITIAL_CONTEXT_SIZE, CONTEXT_STEP, MAX_CONTEXT_SIZE


async def translate_epub_file(
    input_filepath: str,
    output_filepath: str,
    source_language: str = "English",
    target_language: str = "Chinese",
    model_name: str = DEFAULT_MODEL,
    cli_api_endpoint: str = API_ENDPOINT,
    log_callback: Optional[Callable] = None,
    stats_callback: Optional[Callable] = None,
    check_interruption_callback: Optional[Callable] = None,
    llm_provider: str = "ollama",
    gemini_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    mistral_api_key: Optional[str] = None,
    deepseek_api_key: Optional[str] = None,
    context_window: int = 2048,
    auto_adjust_context: bool = True,
    min_chunk_size: int = 5,
    checkpoint_manager=None,
    translation_id: Optional[str] = None,
    resume_from_index: int = 0,
    prompt_options: Optional[Dict] = None,
    max_tokens_per_chunk: int = MAX_TOKENS_PER_CHUNK,
    max_attempts: int = None,
    bilingual: bool = False,
) -> None:
    """
    Translate an EPUB file using LLM with generic orchestrator.

    This implementation uses the unified translation pipeline:
    1. Extract EPUB to temp directory
    2. Parse manifest and get content files
    3. For each XHTML file:
       - Create EpubTranslationAdapter
       - Create GenericTranslationOrchestrator
       - Translate using unified pipeline
    4. Save translated files
    5. Update metadata
    6. Repackage EPUB

    Args:
        input_filepath: Path to input EPUB
        output_filepath: Path to output EPUB
        source_language: Source language
        target_language: Target language
        model_name: LLM model name
        cli_api_endpoint: API endpoint
        log_callback: Logging callback
        stats_callback: Statistics callback
        check_interruption_callback: Interruption check callback
        llm_provider: LLM provider (ollama/gemini/openai/openrouter/mistral/deepseek)
        gemini_api_key: Gemini API key
        openai_api_key: OpenAI API key
        openrouter_api_key: OpenRouter API key
        mistral_api_key: Mistral API key
        deepseek_api_key: DeepSeek API key
        context_window: Context window size for LLM
        auto_adjust_context: Auto-adjust context based on model
        min_chunk_size: Minimum chunk size
        checkpoint_manager: Checkpoint manager for resume functionality
        translation_id: ID of the translation job
        resume_from_index: Index to resume from (file index)
        prompt_options: Optional dict with prompt customization options
        max_tokens_per_chunk: Maximum tokens per chunk
        max_attempts: Maximum translation attempts per chunk
        bilingual: Enable bilingual translation mode
    """
    # Validate input file
    if not os.path.exists(input_filepath):
        err_msg = f"ERROR: Input EPUB file '{input_filepath}' not found."
        if log_callback:
            log_callback("epub_input_file_not_found", err_msg)
        return

    # Use default MAX_TRANSLATION_ATTEMPTS if not provided
    if max_attempts is None:
        max_attempts = MAX_TRANSLATION_ATTEMPTS

    # Add bilingual option to prompt_options
    if bilingual:
        if prompt_options is None:
            prompt_options = {}
        prompt_options['bilingual'] = True

    # Determine initial context size based on model type
    is_known_thinking_model = any(tm in model_name.lower() for tm in THINKING_MODELS)
    if auto_adjust_context:
        if is_known_thinking_model:
            initial_context = ADAPTIVE_CONTEXT_INITIAL_THINKING
        else:
            initial_context = INITIAL_CONTEXT_SIZE
    else:
        initial_context = context_window

    # Create LLM client
    llm_client = _create_llm_client(
        llm_provider=llm_provider,
        model_name=model_name,
        gemini_api_key=gemini_api_key,
        openai_api_key=openai_api_key,
        openrouter_api_key=openrouter_api_key,
        mistral_api_key=mistral_api_key,
        deepseek_api_key=deepseek_api_key,
        cli_api_endpoint=cli_api_endpoint,
        initial_context=initial_context,
        log_callback=log_callback
    )

    if llm_client is None:
        return

    # Create adaptive context manager
    context_manager = _create_context_manager(
        llm_provider=llm_provider,
        auto_adjust_context=auto_adjust_context,
        initial_context=initial_context,
        is_thinking_model=is_known_thinking_model,
        log_callback=log_callback
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # 1. Extract EPUB
            _extract_epub(input_filepath, temp_dir, log_callback)

            # 2. Parse manifest
            manifest_data = _parse_epub_manifest(temp_dir, log_callback)

            # 2.5. Restore checkpoint if resuming
            restored_docs = {}
            if checkpoint_manager and translation_id and resume_from_index > 0:
                restored_docs = await _restore_checkpoint_files(
                    checkpoint_manager, translation_id, temp_dir,
                    resume_from_index, manifest_data['opf_dir'], log_callback
                )

            # 3. Translate all files using orchestrator
            results = await _process_all_content_files(
                content_files=manifest_data['content_files'],
                opf_dir=manifest_data['opf_dir'],
                temp_dir=temp_dir,
                source_language=source_language,
                target_language=target_language,
                model_name=model_name,
                llm_client=llm_client,
                max_tokens_per_chunk=max_tokens_per_chunk,
                max_attempts=max_attempts,
                context_manager=context_manager,
                translation_id=translation_id,
                resume_from_index=resume_from_index,
                checkpoint_manager=checkpoint_manager,
                log_callback=log_callback,
                stats_callback=stats_callback,
                check_interruption_callback=check_interruption_callback,
                prompt_options=prompt_options,
                restored_docs=restored_docs
            )

            # 4. Save translated files
            await _save_translated_files(
                parsed_xhtml_docs=results['parsed_docs'],
                log_callback=log_callback
            )

            # 5. Update metadata
            _update_epub_metadata(
                opf_tree=manifest_data['opf_tree'],
                opf_path=manifest_data['opf_path'],
                target_language=target_language
            )

            # 6. Repackage EPUB
            _repackage_epub(
                temp_dir=temp_dir,
                output_filepath=output_filepath,
                log_callback=log_callback)

            # 7. Final summary
            if log_callback:
                log_callback("epub_save_success",
                             f"✅ EPUB translation complete: {results['completed_files']} files translated, {results['failed_files']} failed")

                if 'translation_stats' in results and results['translation_stats']:
                    translation_stats = results['translation_stats']
                    if translation_stats.total_chunks > 0:
                        stats_summary = translation_stats.log_summary(log_callback=None)
                        if stats_summary:
                            log_callback("epub_translation_stats", stats_summary)

        except Exception as e_epub:
            err_msg = f"MAJOR ERROR processing EPUB '{input_filepath}': {e_epub}"
            if log_callback:
                log_callback("epub_major_error", err_msg)
                import traceback
                log_callback("epub_major_error_traceback", traceback.format_exc())


# === Private Helper Functions ===

def _extract_epub(input_filepath: str, temp_dir: str, log_callback: Optional[Callable] = None) -> None:
    """Extract EPUB to temporary directory."""
    if log_callback:
        log_callback("epub_extract_start", "Extracting EPUB...")

    with zipfile.ZipFile(input_filepath, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)


def _find_opf_file(temp_dir: str) -> Optional[str]:
    """Find OPF file in extracted EPUB."""
    for root_dir, _, files in os.walk(temp_dir):
        for file in files:
            if file.endswith('.opf'):
                return os.path.join(root_dir, file)
    return None


def _get_content_files_from_spine(spine: etree._Element, manifest: etree._Element) -> list:
    """Extract content file hrefs from spine."""
    content_files = []
    for itemref in spine.findall('.//opf:itemref', namespaces=NAMESPACES):
        idref = itemref.get('idref')
        item = manifest.find(f'.//opf:item[@id="{idref}"]', namespaces=NAMESPACES)
        if item is not None:
            media_type = item.get('media-type')
            href = item.get('href')
            if media_type in ['application/xhtml+xml', 'text/html'] and href:
                content_files.append(href)
    return content_files


def _parse_epub_manifest(temp_dir: str, log_callback: Optional[Callable] = None) -> Dict:
    """
    Parse OPF manifest and extract metadata.

    Args:
        temp_dir: Temporary extraction directory
        log_callback: Optional logging callback

    Returns:
        Dictionary with keys: opf_path, opf_tree, opf_dir, content_files
    """
    # Find OPF file
    opf_path = _find_opf_file(temp_dir)
    if not opf_path:
        raise FileNotFoundError("CRITICAL ERROR: content.opf not found in EPUB.")

    # Parse OPF to get content files
    opf_tree = etree.parse(opf_path)
    opf_root = opf_tree.getroot()
    opf_dir = os.path.dirname(opf_path)

    manifest = opf_root.find('.//opf:manifest', namespaces=NAMESPACES)
    spine = opf_root.find('.//opf:spine', namespaces=NAMESPACES)
    if manifest is None or spine is None:
        raise ValueError("CRITICAL ERROR: manifest or spine missing in EPUB.")

    # Get content files from spine
    content_files = _get_content_files_from_spine(spine, manifest)

    if log_callback:
        log_callback("epub_files_found", f"Found {len(content_files)} content files to translate.")

    return {
        'opf_path': opf_path,
        'opf_tree': opf_tree,
        'opf_dir': opf_dir,
        'content_files': content_files
    }


def _create_llm_client(
    llm_provider: str,
    model_name: str,
    gemini_api_key: Optional[str],
    openai_api_key: Optional[str],
    openrouter_api_key: Optional[str],
    mistral_api_key: Optional[str],
    deepseek_api_key: Optional[str],
    cli_api_endpoint: str,
    initial_context: int,
    log_callback: Optional[Callable] = None
) -> Any:
    """Create LLM client with specified configuration."""
    from ..llm_client import create_llm_client

    llm_client = create_llm_client(
        llm_provider, gemini_api_key, cli_api_endpoint, model_name,
        openai_api_key, openrouter_api_key, mistral_api_key, deepseek_api_key,
        context_window=initial_context,
        log_callback=log_callback
    )

    if llm_client is None:
        if log_callback:
            log_callback("llm_client_error", "ERROR: Could not create LLM client.")

    return llm_client


def _create_context_manager(
    llm_provider: str,
    auto_adjust_context: bool,
    initial_context: int,
    is_thinking_model: bool,
    log_callback: Optional[Callable] = None
) -> Optional[AdaptiveContextManager]:
    """Create adaptive context manager if applicable."""
    context_manager = None
    if llm_provider == "ollama" and auto_adjust_context:
        context_manager = AdaptiveContextManager(
            initial_context=initial_context,
            context_step=CONTEXT_STEP,
            max_context=MAX_CONTEXT_SIZE,
            log_callback=log_callback
        )
        model_type = "thinking" if is_thinking_model else "standard"
        if log_callback:
            log_callback("context_adaptive",
                f"🎯 Adaptive context enabled for EPUB ({model_type} model): starting at {initial_context} tokens, "
                f"max={MAX_CONTEXT_SIZE}, step={CONTEXT_STEP}")

    return context_manager


async def _restore_checkpoint_files(
    checkpoint_manager,
    translation_id: str,
    temp_dir: str,
    resume_from_index: int,
    opf_dir: str,
    log_callback: Optional[Callable] = None
) -> Dict[str, etree._Element]:
    """
    Restore previously translated files from checkpoint.

    Args:
        checkpoint_manager: Checkpoint manager instance
        translation_id: Translation job ID
        temp_dir: Temporary directory
        resume_from_index: Index to resume from
        opf_dir: OPF directory
        log_callback: Logging callback

    Returns:
        Dictionary of file_path → doc_root for restored files
    """
    restored_docs = {}

    if log_callback:
        log_callback("epub_restore_checkpoint",
                    f"Restoring {resume_from_index} previously translated files from checkpoint...")

    restore_success = checkpoint_manager.restore_epub_files(
        translation_id=translation_id,
        work_dir=Path(temp_dir)
    )

    if not restore_success:
        if log_callback:
            log_callback("epub_restore_warning",
                         "Warning: Could not restore all files from checkpoint. Translation will continue from scratch.")
        return restored_docs

    # Parse restored files
    checkpoint_files_dir = checkpoint_manager.uploads_dir / translation_id / "translated_files"

    if not checkpoint_files_dir.exists():
        if log_callback:
            log_callback("epub_restore_no_files", "⚠️ No translated files found in checkpoint")
        return restored_docs

    restored_count = 0
    for saved_file in checkpoint_files_dir.rglob('*'):
        if not saved_file.is_file():
            continue

        # Get relative path from checkpoint storage
        rel_path = saved_file.relative_to(checkpoint_files_dir)
        rel_path_str = str(rel_path).replace('\\', '/')

        # Calculate absolute path in temp_dir
        file_path_abs = os.path.normpath(os.path.join(temp_dir, rel_path_str))

        # Fallback for old checkpoints
        if not os.path.exists(file_path_abs):
            file_path_abs = os.path.normpath(os.path.join(opf_dir, rel_path_str))
            if log_callback:
                log_callback("epub_restore_fallback",
                           f"🔄 Using fallback path for old checkpoint: {rel_path_str}")

        try:
            async with aiofiles.open(file_path_abs, 'r', encoding='utf-8') as f:
                restored_content = await f.read()

            parser = etree.XMLParser(encoding='utf-8', recover=True, remove_blank_text=False)
            doc_root = etree.fromstring(restored_content.encode('utf-8'), parser)
            restored_docs[file_path_abs] = doc_root
            restored_count += 1

            if log_callback:
                log_callback("epub_restore_file_parsed",
                           f"📄 Restored file {restored_count}: {rel_path_str}")
        except Exception as e:
            if log_callback:
                log_callback("epub_restore_parse_error",
                             f"⚠️ Warning: Could not parse restored file {rel_path_str}: {e}")

    if log_callback:
        log_callback("epub_restore_success",
                    f"✅ Successfully restored {len(restored_docs)} files from checkpoint")

    return restored_docs


async def _translate_single_xhtml_file(
    file_path: str,
    content_href: str,
    source_language: str,
    target_language: str,
    model_name: str,
    llm_client: Any,
    max_tokens_per_chunk: int,
    max_attempts: int,
    context_manager: Optional[AdaptiveContextManager],
    log_callback: Optional[Callable],
    prompt_options: Optional[Dict],
    stats_callback: Optional[Callable] = None,
    checkpoint_manager: Optional[Any] = None,
    translation_id: Optional[str] = None,
    check_interruption_callback: Optional[Callable] = None,
    global_total_chunks: Optional[int] = None,
    global_completed_chunks: Optional[int] = None,
) -> Tuple[Optional[etree._Element], bool, Any]:
    """
    Translate a single XHTML file using GenericTranslationOrchestrator.
    Now supports resume from partial state.

    Args:
        file_path: Path to XHTML file
        content_href: Content href (for logging)
        source_language: Source language
        target_language: Target language
        model_name: Model name
        llm_client: LLM client instance
        max_tokens_per_chunk: Max tokens per chunk
        max_attempts: Max translation attempts
        context_manager: Optional context manager
        log_callback: Logging callback
        prompt_options: Prompt options
        stats_callback: Optional stats callback
        checkpoint_manager: Optional checkpoint manager for partial state
        translation_id: Optional translation ID for checkpointing
        check_interruption_callback: Optional interruption check callback

    Returns:
        (doc_root, success, stats)
    """
    if not os.path.exists(file_path):
        if log_callback:
            log_callback("epub_file_not_found", f"WARNING: File '{content_href}' not found, skipped.")
        return None, False, None

    # === VÉRIFIER SI REPRISE DEPUIS ÉTAT PARTIEL ===
    resume_state = None
    if checkpoint_manager and translation_id:
        resume_state = checkpoint_manager.load_xhtml_partial_state(
            translation_id, content_href
        )

        if resume_state:
            if log_callback:
                log_callback("xhtml_resume_detected",
                    f"📂 Resuming '{content_href}' from chunk {resume_state.current_chunk_index}/{len(resume_state.chunks)}")

    try:
        # Parse XHTML file
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()

        parser = etree.XMLParser(encoding='utf-8', recover=True, remove_blank_text=False)
        doc_root = etree.fromstring(content.encode('utf-8'), parser)

        # Create adapter and orchestrator
        adapter = EpubTranslationAdapter()
        orchestrator = GenericTranslationOrchestrator(adapter)

        # Translate using generic pipeline WITH resume support
        success, stats = await orchestrator.translate(
            source=doc_root,
            source_language=source_language,
            target_language=target_language,
            model_name=model_name,
            llm_client=llm_client,
            max_tokens_per_chunk=max_tokens_per_chunk,
            log_callback=log_callback,
            context_manager=context_manager,
            max_retries=max_attempts,
            prompt_options=prompt_options,
            stats_callback=stats_callback,
            # NOUVEAUX PARAMÈTRES
            checkpoint_manager=checkpoint_manager,
            translation_id=translation_id,
            file_href=content_href,
            check_interruption_callback=check_interruption_callback,
            resume_state=resume_state,
            global_total_chunks=global_total_chunks,
            global_completed_chunks=global_completed_chunks,
        )

        return doc_root, success, stats

    except etree.XMLSyntaxError as e:
        if log_callback:
            log_callback("epub_xml_error", f"XML error in '{content_href}': {e}")
        return None, False, None
    except Exception as e:
        if log_callback:
            log_callback("epub_file_error", f"Error processing '{content_href}': {e}")
        return None, False, None


async def _precount_chunks(
    content_files: list,
    opf_dir: str,
    max_tokens_per_chunk: int,
    log_callback: Optional[Callable] = None
) -> Tuple[int, List[int]]:
    """
    Pre-count chunks across all XHTML files for accurate progress tracking.

    Returns:
        (total_chunks, chunks_per_file)
    """
    from .epub_translation_adapter import EpubTranslationAdapter

    chunks_per_file = []
    total_chunks = 0

    if log_callback:
        log_callback("epub_precount_start", f"📊 Analyzing {len(content_files)} files for progress tracking...")

    for content_href in content_files:
        file_path = os.path.normpath(os.path.join(opf_dir, content_href))
        if not os.path.exists(file_path):
            chunks_per_file.append(0)
            continue

        try:
            # Parse file
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()

            parser = etree.XMLParser(encoding='utf-8', recover=True, remove_blank_text=False)
            doc_root = etree.fromstring(content.encode('utf-8'), parser)

            # Count chunks using adapter
            adapter = EpubTranslationAdapter()
            raw_content, context = adapter.extract_content(doc_root, None)

            if not raw_content or not raw_content.strip():
                chunks_per_file.append(0)
                continue

            text_with_placeholders, structure_map, _ = adapter.preserve_structure(
                raw_content, context, None
            )

            chunks = adapter.create_chunks(
                text_with_placeholders, structure_map, max_tokens_per_chunk, None
            )

            chunk_count = len(chunks)
            chunks_per_file.append(chunk_count)
            total_chunks += chunk_count

        except Exception:
            chunks_per_file.append(0)

    if log_callback:
        log_callback("epub_precount_complete",
                     f"📊 Found {total_chunks} total chunks across {len(content_files)} files")

    return total_chunks, chunks_per_file


async def _process_all_content_files(
    content_files: list,
    opf_dir: str,
    temp_dir: str,
    source_language: str,
    target_language: str,
    model_name: str,
    llm_client: Any,
    max_tokens_per_chunk: int,
    max_attempts: int,
    context_manager: Optional[AdaptiveContextManager],
    translation_id: Optional[str],
    resume_from_index: int = 0,
    checkpoint_manager=None,
    log_callback: Optional[Callable] = None,
    stats_callback: Optional[Callable] = None,
    check_interruption_callback: Optional[Callable] = None,
    prompt_options: Optional[Dict] = None,
    restored_docs: Optional[Dict[str, etree._Element]] = None
) -> Dict:
    """
    Process all XHTML content files using GenericTranslationOrchestrator.

    Args:
        content_files: List of content file hrefs
        opf_dir: OPF directory path
        temp_dir: Temporary directory
        source_language: Source language
        target_language: Target language
        model_name: Model name
        llm_client: LLM client instance
        max_tokens_per_chunk: Max tokens per chunk
        max_attempts: Max translation attempts
        context_manager: Optional context manager
        translation_id: Optional translation ID
        resume_from_index: Index to resume from
        checkpoint_manager: Optional checkpoint manager
        log_callback: Optional logging callback        stats_callback: Optional stats callback
        check_interruption_callback: Optional interruption check callback
        prompt_options: Optional prompt options
        restored_docs: Restored documents from checkpoint

    Returns:
        Dictionary with processing results
    """
    from .translation_metrics import TranslationMetrics

    # Pre-count chunks for accurate progress tracking
    total_chunks, chunks_per_file = await _precount_chunks(
        content_files, opf_dir, max_tokens_per_chunk, log_callback
    )

    # Start with restored documents
    parsed_xhtml_docs: Dict[str, etree._Element] = restored_docs.copy() if restored_docs else {}
    total_files = len(content_files)
    completed_files = len(parsed_xhtml_docs)
    failed_files = 0

    # Accumulate translation statistics
    accumulated_stats = TranslationMetrics()

    # Track global chunk progress
    completed_chunks_global = 0
    for idx in range(resume_from_index):
        if idx < len(chunks_per_file):
            completed_chunks_global += chunks_per_file[idx]

    # Send initial stats if resuming (to update UI immediately)
    if stats_callback and resume_from_index > 0:
        stats_callback({
            'total_chunks': total_chunks,
            'completed_chunks': completed_chunks_global,
            'failed_chunks': 0,
            'total_tokens': 0
        })

    for file_idx, content_href in enumerate(content_files):
        # Check for interruption
        if check_interruption_callback and check_interruption_callback():
            if log_callback:
                log_callback("epub_translation_interrupted",
                             f"Translation interrupted at file {file_idx + 1}/{total_files}")
            break

        # Skip if already processed (resume)
        if file_idx < resume_from_index:
            completed_files += 1
            continue

        file_path = os.path.normpath(os.path.join(opf_dir, content_href))
        chunks_in_this_file = chunks_per_file[file_idx] if file_idx < len(chunks_per_file) else 0

        if log_callback:
            log_callback("epub_file_translate_start",
                         f"Translating file {file_idx + 1}/{total_files}: {content_href} ({chunks_in_this_file} chunks)")

        # Create stats wrapper that reports global statistics
        # NOTE: completed_chunks_global represents chunks from ALL previous files (not including current)
        def file_stats_wrapper(file_stats_dict: Dict):
            """Convert file-level stats to global stats by merging with accumulated stats"""
            if not stats_callback:
                return

            # Calculate global completed chunks:
            # completed_chunks_global = chunks from previous files (already updated)
            # current_file_completed = chunks completed in current file (reported by xhtml_translator)
            current_file_completed = file_stats_dict.get('completed_chunks', 0)
            global_completed = completed_chunks_global + current_file_completed

            # Report combined stats (accumulated + current file)
            stats_callback({
                'total_chunks': total_chunks,
                'completed_chunks': global_completed,
                'failed_chunks': accumulated_stats.failed_chunks + file_stats_dict.get('failed_chunks', 0),
                'total_tokens': accumulated_stats.total_tokens_processed + accumulated_stats.total_tokens_generated + file_stats_dict.get('total_tokens_processed', 0) + file_stats_dict.get('total_tokens_generated', 0)
            })

        # Translate using orchestrator WITH checkpoint support
        doc_root, success, file_stats = await _translate_single_xhtml_file(
            file_path=file_path,
            content_href=content_href,
            source_language=source_language,
            target_language=target_language,
            model_name=model_name,
            llm_client=llm_client,
            max_tokens_per_chunk=max_tokens_per_chunk,
            max_attempts=max_attempts,
            context_manager=context_manager,
            log_callback=log_callback,
            prompt_options=prompt_options,
            stats_callback=file_stats_wrapper,
            checkpoint_manager=checkpoint_manager,
            translation_id=translation_id,
            check_interruption_callback=check_interruption_callback,
            global_total_chunks=total_chunks,
            global_completed_chunks=completed_chunks_global,
        )

        # Update global chunk counter
        completed_chunks_global += chunks_in_this_file

        # Accumulate statistics
        if file_stats:
            accumulated_stats.merge(file_stats)

        # Report stats if callback provided
        if stats_callback and file_stats:
            stats_callback({
                'total_chunks': total_chunks,
                'completed_chunks': completed_chunks_global,
                'failed_chunks': accumulated_stats.failed_chunks,
                'total_tokens': accumulated_stats.total_tokens_processed + accumulated_stats.total_tokens_generated
            })

        # Save the document if translation succeeded
        if success and doc_root is not None:
            parsed_xhtml_docs[file_path] = doc_root
            completed_files += 1
        elif not success and doc_root is not None:
            # Save original document if translation failed
            parsed_xhtml_docs[file_path] = doc_root
            failed_files += 1
            if log_callback:
                log_callback("epub_file_translate_failed",
                             f"Failed to translate file {file_idx + 1}/{total_files}: {content_href}")
        else:
            failed_files += 1

        # Save checkpoint
        if checkpoint_manager and translation_id and success and doc_root is not None:
            await _save_checkpoint(
                checkpoint_manager, translation_id, file_idx, content_href,
                doc_root, file_path, temp_dir, log_callback,
                total_chunks=total_chunks,
                completed_chunks=completed_chunks_global,
                failed_chunks=accumulated_stats.failed_chunks
            )

    # Final progress
    return {
        'parsed_docs': parsed_xhtml_docs,
        'completed_files': completed_files,
        'failed_files': failed_files,
        'total_chunks': total_chunks,
        'completed_chunks': completed_chunks_global,
        'failed_chunks': accumulated_stats.failed_chunks,
        'translation_stats': accumulated_stats
    }


async def _save_checkpoint(
    checkpoint_manager,
    translation_id: str,
    file_idx: int,
    content_href: str,
    doc_root: etree._Element,
    file_path: str,
    temp_dir: str,
    log_callback: Optional[Callable] = None,
    total_chunks: int = 0,
    completed_chunks: int = 0,
    failed_chunks: int = 0
) -> None:
    """Save checkpoint for a translated file."""
    try:
        # Serialize document
        file_content = etree.tostring(
            doc_root,
            encoding='utf-8',
            xml_declaration=True,
            pretty_print=True,
            method='xml'
        )

        # Calculate relative path from temp_dir
        file_rel_path = os.path.relpath(file_path, temp_dir).replace('\\', '/')

        # Save to checkpoint storage
        save_result = checkpoint_manager.save_epub_file(
            translation_id=translation_id,
            file_href=file_rel_path,
            file_content=file_content
        )

        if save_result:
            # Delete partial state AFTER successful file save (atomicity guarantee)
            checkpoint_manager.delete_xhtml_partial_state(translation_id, file_rel_path)
            if log_callback:
                log_callback("xhtml_partial_state_deleted_after_save",
                    f"🗑️ Partial state deleted for {file_rel_path} (file saved successfully)")

            # Update checkpoint progress with chunk statistics
            checkpoint_manager.save_checkpoint(
                translation_id=translation_id,
                chunk_index=file_idx + 1,
                original_text=content_href,
                translated_text=content_href,
                chunk_data={'last_file': content_href, 'file_type': 'epub_xhtml'},
                total_chunks=total_chunks,
                completed_chunks=completed_chunks,
                failed_chunks=failed_chunks
            )

            if log_callback:
                log_callback("epub_checkpoint_file_saved",
                           f"💾 Checkpoint saved: {file_rel_path} ({len(file_content)} bytes)")
        else:
            if log_callback:
                log_callback("epub_checkpoint_save_error",
                             f"⚠️ Warning: Could not save file to checkpoint storage: {content_href}")
    except Exception as e:
        if log_callback:
            log_callback("epub_checkpoint_save_error",
                         f"⚠️ Warning: Could not save checkpoint: {content_href}: {e}")


async def _save_translated_files(
    parsed_xhtml_docs: Dict[str, etree._Element],
    log_callback: Optional[Callable] = None
) -> None:
    """Save modified XHTML files."""
    if log_callback:
        log_callback("epub_save_files_start",
                   f"💾 Saving {len(parsed_xhtml_docs)} translated XHTML files to temp directory...")

    for file_path_abs, doc_root in parsed_xhtml_docs.items():
        try:
            # Clean residual placeholders
            for element in doc_root.iter():
                if element.text:
                    element.text = clean_residual_tag_placeholders(element.text)
                if element.tail:
                    element.tail = clean_residual_tag_placeholders(element.tail)

            async with aiofiles.open(file_path_abs, 'wb') as f_out:
                await f_out.write(
                    etree.tostring(doc_root, encoding='utf-8', xml_declaration=True,
                                   pretty_print=True, method='xml')
                )
        except Exception as e_write:
            if log_callback:
                log_callback("epub_write_error", f"Error writing '{file_path_abs}': {e_write}")


def _repackage_epub(
    temp_dir: str,
    output_filepath: str,
    log_callback: Optional[Callable] = None,
) -> None:
    """Repackage the EPUB file."""
    with zipfile.ZipFile(output_filepath, 'w', zipfile.ZIP_DEFLATED) as epub_zip:
        # Add mimetype first (uncompressed)
        mimetype_path = os.path.join(temp_dir, 'mimetype')
        if os.path.exists(mimetype_path):
            epub_zip.write(mimetype_path, 'mimetype', compress_type=zipfile.ZIP_STORED)

        # Add all other files
        for root_path, _, files in os.walk(temp_dir):
            for file_item in files:
                if file_item != 'mimetype':
                    file_path_abs = os.path.join(root_path, file_item)
                    arcname = os.path.relpath(file_path_abs, temp_dir)
                    epub_zip.write(file_path_abs, arcname)

def _update_epub_metadata(
    opf_tree: etree._ElementTree,
    opf_path: str,
    target_language: str
) -> None:
    """Update EPUB metadata with target language and translation signature."""
    opf_root = opf_tree.getroot()
    metadata = opf_root.find('.//opf:metadata', namespaces=NAMESPACES)
    if metadata is not None:
        # Update language
        lang_el = metadata.find('.//dc:language', namespaces=NAMESPACES)
        if lang_el is not None:
            lang_el.text = target_language.lower()[:2]

        # Add translation signature if enabled
        if ATTRIBUTION_ENABLED:
            # Add contributor (translator)
            contributor_el = etree.SubElement(
                metadata,
                '{http://purl.org/dc/elements/1.1/}contributor'
            )
            contributor_el.text = GENERATOR_NAME
            contributor_el.set('{http://www.idpf.org/2007/opf}role', 'trl')

            # Add or update description with signature
            desc_el = metadata.find('.//dc:description', namespaces=NAMESPACES)
            signature_text = f"\n\nTranslated using {GENERATOR_NAME}\n{GENERATOR_SOURCE}"

            if desc_el is None:
                desc_el = etree.SubElement(
                    metadata,
                    '{http://purl.org/dc/elements/1.1/}description'
                )
                desc_el.text = signature_text.strip()
            else:
                if desc_el.text:
                    desc_el.text += signature_text
                else:
                    desc_el.text = signature_text.strip()

    opf_tree.write(opf_path, encoding='utf-8', xml_declaration=True, pretty_print=True)
