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
from urllib.parse import unquote
from lxml import etree

from src.config import (
    NAMESPACES, DEFAULT_MODEL, API_ENDPOINT,
    MAX_TOKENS_PER_CHUNK, THINKING_MODELS, ADAPTIVE_CONTEXT_INITIAL_THINKING,
    MAX_TRANSLATION_ATTEMPTS, ATTRIBUTION_ENABLED, GENERATOR_NAME, GENERATOR_SOURCE,
    EPUB_SCRIPT_NORMALIZATION_ENABLED, EPUB_TRANSLATE_METADATA_ENABLED
)
from ..common.translation_orchestrator import GenericTranslationOrchestrator
from .epub_translation_adapter import EpubTranslationAdapter
from .xhtml_translation_state import (
    unfinished_chunk_indices,
    untranslated_chunk_indices,
)
from ..post_processor import clean_residual_tag_placeholders
from ..context_optimizer import AdaptiveContextManager, INITIAL_CONTEXT_SIZE, CONTEXT_STEP, MAX_CONTEXT_SIZE
from .rtl_support import apply_rtl_to_epub_directory, is_rtl_language
from .lang_support import apply_target_language_to_xhtml_directory, get_language_code
from .attribution_page import add_attribution_page
from .cjk_typography import apply_script_normalization_to_epub_directory
from .metadata_translator import translate_opf_metadata
from src.utils.security import safe_extract_zip


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
    poe_api_key: Optional[str] = None,
    nim_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    xai_api_key: Optional[str] = None,
    opencode_api_key: Optional[str] = None,
    opencodego_api_key: Optional[str] = None,
    ollamacloud_api_key: Optional[str] = None,
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
    parallel_workers: int = 1,
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
        llm_provider: LLM provider (ollama/gemini/openai/openrouter/mistral/deepseek/poe)
        gemini_api_key: Gemini API key
        openai_api_key: OpenAI API key
        openrouter_api_key: OpenRouter API key
        mistral_api_key: Mistral API key
        deepseek_api_key: DeepSeek API key
        poe_api_key: Poe API key
        nim_api_key: NVIDIA NIM API key
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
        poe_api_key=poe_api_key,
        nim_api_key=nim_api_key,
        anthropic_api_key=anthropic_api_key,
        xai_api_key=xai_api_key,
        opencode_api_key=opencode_api_key,
        opencodego_api_key=opencodego_api_key,
        ollamacloud_api_key=ollamacloud_api_key,
        cli_api_endpoint=cli_api_endpoint,
        initial_context=initial_context,
        log_callback=log_callback
    )

    if llm_client is None:
        return

    # Resolve effective parallel workers (local providers are forced back to 1).
    # translate_file() already logged the effective count; this re-resolve is
    # idempotent and covers the direct-call path (CLI EPUB goes through here).
    from src.config import resolve_parallel_workers
    parallel_workers = resolve_parallel_workers(llm_provider, parallel_workers)

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
                restored_docs=restored_docs,
                parallel_workers=parallel_workers
            )

            # 4. Save translated files
            await _save_translated_files(
                parsed_xhtml_docs=results['parsed_docs'],
                log_callback=log_callback
            )

            # 4.5. Update NCX table-of-contents labels from translated XHTML
            # headings. This preserves <content src="..."> jump targets while
            # localizing the reader's side-panel TOC for EPUB2 books.
            _update_ncx_toc_labels_from_translated_docs(
                opf_dir=manifest_data['opf_dir'],
                parsed_xhtml_docs=results['parsed_docs'],
                log_callback=log_callback
            )

            # 4.6. Update the EPUB3 navigation document (nav.xhtml) TOC links
            # the same way. EPUB3 readers build their TOC from this document
            # rather than the NCX, so without this step the side-panel keeps the
            # source-language titles even though the body is translated.
            _update_nav_toc_labels_from_translated_docs(
                opf_dir=manifest_data['opf_dir'],
                opf_tree=manifest_data['opf_tree'],
                parsed_xhtml_docs=results['parsed_docs'],
                log_callback=log_callback
            )

            # 4.7. Append the attribution page to the end of the spine. Runs before
            # the metadata write (step 5) so the manifest and spine edits are
            # persisted by that single OPF write, and before RTL (6) and the lang
            # pass (6.5) so the new page inherits RTL CSS and the target lang
            # attribute like any other document in the book.
            add_attribution_page(
                opf_tree=manifest_data['opf_tree'],
                opf_dir=manifest_data['opf_dir'],
                log_callback=log_callback
            )

            # 5. Update metadata
            _update_epub_metadata(
                opf_tree=manifest_data['opf_tree'],
                opf_path=manifest_data['opf_path'],
                target_language=target_language
            )

            # 5.5. Localize the packaging metadata (OPF dc:title/dc:description
            # and every NCX docTitle), which readers show on the library shelf
            # and in the book-information panel.
            #
            # Ordering rationale — DO NOT REORDER THIS AND STEP 6.6:
            #   - after 5 (_update_epub_metadata): that call appends the
            #     attribution signature to dc:description, and this pass has to
            #     strip it before sending and re-append it afterwards, so the
            #     signature must already be there.
            #   - BEFORE 6.6 (apply_script_normalization_to_epub_directory):
            #     both passes write the OPF, from two *different* trees — 6.6
            #     re-parses the OPF from disk. Writing our in-memory
            #     `manifest_data['opf_tree']` after 6.6 has written its own tree
            #     would silently revert 6.6's font-override meta removal and its
            #     page-progression-direction reset. Running first means 6.6
            #     re-reads a file that already carries the translated metadata,
            #     and its NCX xml:lang write preserves our docTitle for the same
            #     reason.
            #
            # Guarded like 6.6: metadata localization is a nice-to-have relative
            # to the translated text and must never fail an otherwise-successful
            # job. The module already swallows its own failures; this is the
            # belt-and-braces guard for anything it did not anticipate.
            if EPUB_TRANSLATE_METADATA_ENABLED:
                try:
                    await translate_opf_metadata(
                        opf_tree=manifest_data['opf_tree'],
                        opf_path=manifest_data['opf_path'],
                        opf_dir=manifest_data['opf_dir'],
                        source_language=source_language,
                        target_language=target_language,
                        llm_client=llm_client,
                        model_name=model_name,
                        log_callback=log_callback
                    )
                except Exception as e_meta:
                    if log_callback:
                        log_callback("epub_metadata_translation_failed",
                                     f"⚠️ Packaging metadata localization "
                                     f"failed and was skipped: {e_meta}")

            # 6. Apply RTL/LTR layout based on source and target languages
            # This handles RTL->RTL, LTR->RTL, RTL->LTR, and LTR->LTR transitions
            if log_callback:
                if is_rtl_language(target_language):
                    log_callback("epub_rtl_start", f"🔄 Applying RTL layout for {target_language}...")
                elif is_rtl_language(source_language):
                    log_callback("epub_rtl_start", f"🔄 Resetting to LTR layout (translating from {source_language})...")
            
            rtl_result = apply_rtl_to_epub_directory(temp_dir, target_language, source_language)

            if log_callback:
                if rtl_result.get('was_transition'):
                    # RTL -> LTR transition
                    log_callback("epub_ltr_applied",
                               f"✅ LTR reset applied: {rtl_result['css_removed']} files cleaned, "
                               f"text direction set to left-to-right")
                elif rtl_result['is_rtl']:
                    # Applied RTL styles
                    log_callback("epub_rtl_applied",
                               f"✅ RTL support applied: {rtl_result['css_injected']} files updated, "
                               f"OPF progression: {'RTL' if rtl_result['opf_updated'] else 'unchanged'}")

            # 6.5. Update <html lang="..."> on every XHTML to the target language
            # so that e-readers apply the correct hyphenation, dictionary and TTS.
            # Runs after RTL apply so it is the final authority on lang attributes.
            apply_target_language_to_xhtml_directory(
                temp_dir, target_language, log_callback=log_callback
            )

            # 6.6. Neutralize source-script (CJK) typography left over from the
            # original stylesheet/packaging (font stacks, indent, leading,
            # writing mode, reader-specific font-override metas).
            #
            # Ordering rationale:
            #   - after 5 (_update_epub_metadata): that call performs the
            #     pipeline's single OPF write; running later means our OPF
            #     edits are not overwritten.
            #   - after 5.5 (translate_opf_metadata), and this must not be
            #     swapped: that pass writes the *in-memory* opf_tree while this
            #     one re-parses the OPF from disk, so running it first would
            #     have its meta removal and progression reset overwritten. See
            #     the longer note at step 5.5.
            #   - after 6 (apply_rtl_to_epub_directory): that pass injects
            #     <style> blocks and may set page-progression-direction.
            #     Running later makes this pass the final authority on
            #     writing-mode and progression direction, and lets it
            #     neutralize anything the RTL pass injected.
            #   - after 6.5 (apply_target_language_to_xhtml_directory): that
            #     pass rewrites every XHTML root's lang attributes; running
            #     later avoids two passes re-serializing the same files in an
            #     interleaved way.
            #   - before 7 (_repackage_epub): obviously.
            #
            # Wrapped in its own try/except: this pass is cosmetic relative to
            # the translated text and must never fail an otherwise-successful
            # job, even on a failure its own internal per-file guards did not
            # anticipate (e.g. an OSError while walking the directory).
            if EPUB_SCRIPT_NORMALIZATION_ENABLED:
                try:
                    norm_result = apply_script_normalization_to_epub_directory(
                        temp_dir, source_language, target_language,
                        log_callback=log_callback
                    )
                    _log_script_normalization(norm_result, log_callback)
                except Exception as e_norm:
                    if log_callback:
                        log_callback("epub_script_norm_failed",
                                     f"⚠️ Source-script typography normalization "
                                     f"failed and was skipped: {e_norm}")

            # 7. Repackage EPUB. If translation was paused, write to a `[partial] `
            # filename so users can tell partial outputs from completed ones at a glance.
            from src.utils.file_utils import get_partial_output_path, find_partial_output_paths
            if results.get('was_interrupted'):
                partial_path = get_partial_output_path(output_filepath)
                if log_callback:
                    log_callback("epub_partial_output_marked",
                                 f"💾 Partial EPUB will be saved as: {os.path.basename(partial_path)}")
                output_filepath = partial_path

            _repackage_epub(
                temp_dir=temp_dir,
                output_filepath=output_filepath,
                log_callback=log_callback)

            # On successful (non-interrupted) save, remove any leftover [partial ...]
            # siblings from previous interrupted runs targeting this same output.
            if not results.get('was_interrupted'):
                for stale in find_partial_output_paths(output_filepath):
                    try:
                        os.remove(stale)
                        if log_callback:
                            log_callback("epub_partial_cleanup",
                                         f"🗑️ Removed stale partial: {os.path.basename(stale)}")
                    except OSError as e:
                        if log_callback:
                            log_callback("epub_partial_cleanup_failed",
                                         f"⚠️ Could not remove stale partial {os.path.basename(stale)}: {e}")

            # 7. Final summary
            if log_callback:
                log_callback("epub_save_success",
                             f"✅ EPUB translation complete: {results['completed_files']} files translated, {results['failed_files']} failed")

                # Log layout status
                if is_rtl_language(target_language):
                    log_callback("epub_rtl_complete", 
                               f"📖 EPUB ready for RTL reading: text direction is right-to-left")
                elif is_rtl_language(source_language):
                    log_callback("epub_ltr_complete", 
                               f"📖 EPUB ready for LTR reading: text direction reset to left-to-right")

        except Exception as e_epub:
            # Re-raise RateLimitError to trigger auto-pause
            from src.core.llm.exceptions import RateLimitError
            if isinstance(e_epub, RateLimitError):
                raise
            err_msg = f"MAJOR ERROR processing EPUB '{input_filepath}': {e_epub}"
            if log_callback:
                log_callback("epub_major_error", err_msg)
                import traceback
                log_callback("epub_major_error_traceback", traceback.format_exc())


# === Private Helper Functions ===

def _log_script_normalization(norm: dict, log_callback: Optional[Callable]) -> None:
    """Emit the user-visible log lines for step 6.6's result dict.

    Factored out of translate_epub_file (already long) rather than inlined at
    the call site. See apply_script_normalization_to_epub_directory's
    docstring for the exact keys of `norm`.
    """
    if not log_callback or not norm['applied']:
        return

    log_callback("epub_script_normalized",
                 f"🔤 Source-script typography neutralized: "
                 f"{norm['css_files_rewritten']} stylesheet(s), "
                 f"{norm['style_elements_rewritten']} <style> block(s), "
                 f"{norm['style_attributes_rewritten']} inline style(s)")

    if norm['progression_direction_reset']:
        log_callback("epub_script_norm_progression_reset",
                     "📖 Page progression reset to left-to-right")

    if norm['opf_metas_removed']:
        log_callback("epub_script_norm_opf_metas_removed",
                     f"🧹 Removed {norm['opf_metas_removed']} reader-specific "
                     f"font override(s)")

    if norm['embedded_font_bytes'] > 1_000_000:
        embedded_mb = norm['embedded_font_bytes'] / (1024 * 1024)
        log_callback("epub_script_norm_fonts_orphaned",
                     f"ℹ️ {embedded_mb:.1f} MB of source-script fonts remain "
                     f"embedded and are no longer referenced")

    if norm['encoding_fallbacks']:
        log_callback("epub_script_norm_encoding_fallback",
                     f"⚠️ {norm['encoding_fallbacks']} stylesheet(s) had their "
                     f"encoding guessed while reading")

    if norm['errors']:
        log_callback("epub_script_norm_errors",
                     f"⚠️ {norm['errors']} error(s) occurred while normalizing "
                     f"source-script typography")


def _extract_epub(input_filepath: str, temp_dir: str, log_callback: Optional[Callable] = None) -> None:
    """Extract EPUB to temporary directory."""
    if log_callback:
        log_callback("epub_extract_start", "Extracting EPUB...")

    with zipfile.ZipFile(input_filepath, 'r') as zip_ref:
        safe_extract_zip(zip_ref, temp_dir)


def _find_opf_file(temp_dir: str) -> Optional[str]:
    """Find OPF file in extracted EPUB."""
    for root_dir, _, files in os.walk(temp_dir):
        for file in files:
            if file.endswith('.opf'):
                return os.path.join(root_dir, file)
    return None


def _resolve_content_path(opf_dir: str, content_href: str) -> str:
    """Resolve a manifest href to a filesystem path.

    EPUB hrefs are URLs: spaces and non-ASCII characters are commonly
    percent-encoded ("Chapter%201.xhtml"). They must be unquoted before
    joining, otherwise the file is reported missing and ships untranslated.
    """
    return os.path.normpath(os.path.join(opf_dir, unquote(content_href)))


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
    poe_api_key: Optional[str],
    nim_api_key: Optional[str],
    cli_api_endpoint: str,
    initial_context: int,
    anthropic_api_key: Optional[str] = None,
    xai_api_key: Optional[str] = None,
    opencode_api_key: Optional[str] = None,
    opencodego_api_key: Optional[str] = None,
    ollamacloud_api_key: Optional[str] = None,
    log_callback: Optional[Callable] = None
) -> Any:
    """Create LLM client with specified configuration."""
    from ..llm_client import create_llm_client

    llm_client = create_llm_client(
        llm_provider, gemini_api_key, cli_api_endpoint, model_name,
        openai_api_key, openrouter_api_key, mistral_api_key, deepseek_api_key,
        poe_api_key=poe_api_key,
        nim_api_key=nim_api_key,
        anthropic_api_key=anthropic_api_key,
        xai_api_key=xai_api_key,
        opencode_api_key=opencode_api_key,
        opencodego_api_key=opencodego_api_key,
        ollamacloud_api_key=ollamacloud_api_key,
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
    parallel_workers: int = 1,
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
            parallel_workers=parallel_workers,
        )

        return doc_root, success, stats

    except etree.XMLSyntaxError as e:
        if log_callback:
            log_callback("epub_xml_error", f"XML error in '{content_href}': {e}")
        return None, False, None
    except Exception as e:
        # Re-raise RateLimitError to trigger auto-pause
        from src.core.llm.exceptions import RateLimitError
        if isinstance(e, RateLimitError):
            raise
        if log_callback:
            log_callback("epub_file_error", f"Error processing '{content_href}': {e}")
        return None, False, None


async def _precount_chunks(
    content_files: list,
    opf_dir: str,
    max_tokens_per_chunk: int,
    log_callback: Optional[Callable] = None,
    plain_text_mode: bool = False,
) -> Tuple[int, List[int]]:
    """
    Pre-count chunks across all XHTML files for accurate progress tracking.

    When plain_text_mode is True, counts chunks using the plain-text pipeline (paragraphs
    joined by \\n\\n then chunked by TokenChunker) instead of the HTML-aware chunker.

    Returns:
        (total_chunks, chunks_per_file)
    """
    from .epub_translation_adapter import EpubTranslationAdapter

    chunks_per_file = []
    total_chunks = 0

    if log_callback:
        log_callback("epub_precount_start", f"📊 Analyzing {len(content_files)} files for progress tracking...")

    for content_href in content_files:
        file_path = _resolve_content_path(opf_dir, content_href)
        if not os.path.exists(file_path):
            chunks_per_file.append(0)
            continue

        try:
            # Parse file
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()

            parser = etree.XMLParser(encoding='utf-8', recover=True, remove_blank_text=False)
            doc_root = etree.fromstring(content.encode('utf-8'), parser)

            if plain_text_mode:
                chunk_count = _precount_chunks_plain_text(doc_root, max_tokens_per_chunk)
                chunks_per_file.append(chunk_count)
                total_chunks += chunk_count
                continue

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


def _precount_chunks_plain_text(doc_root, max_tokens_per_chunk: int) -> int:
    """
    Count chunks for one XHTML file using the plain-text-mode pipeline.
    Returns 0 on any failure (matches the normal-path behavior).
    """
    try:
        from .plain_extractor import extract_plain_paragraphs
        from src.core.common.plain_text_pipeline import build_plain_segments

        body = doc_root.find('.//{http://www.w3.org/1999/xhtml}body')
        if body is None:
            body = doc_root.find('.//body')
        if body is None:
            return 0

        paragraphs, _, _ = extract_plain_paragraphs(body)
        if not paragraphs:
            return 0

        return len(build_plain_segments(paragraphs, max_tokens_per_chunk))
    except Exception:
        return 0


# Counters the UI's rate context (retry %, fallback %, avg placeholder errors)
# divides by or displays. They are the only ones that need a per-run twin: the
# accumulated versions are deliberately rehydrated across resume passes
# (issue #180, the Fallbacks card must not reset to zero), which makes any
# percentage derived from them a cross-pass average nobody asked for.
_RUN_RATE_COUNTERS = (
    'processed_chunks',
    'successful_after_retry',
    'token_alignment_used',
    'fallback_used',
    'placeholder_errors',
)


def _global_stats_payload(total_chunks, completed_chunks, acc, file_stats=None,
                           unfinished_units=None, run_prior_counts=None,
                           untranslated_units=None, run_total_chunks=None,
                           run_is_repair=False):
    """Build the EPUB global-stats dict emitted to the progress callback.

    Single source of the cross-file payload shape, shared by the resume-initial
    emit, the per-chunk wrapper, and the post-file emit (which previously each
    rebuilt this ~10-key dict by hand). ``completed_chunks`` / ``total_chunks``
    are computed by the caller (they differ per site); the cumulative counters
    come from ``acc`` (a TranslationMetrics) plus, when given, the current
    file's not-yet-merged ``file_stats`` dict.

    ``unfinished_units`` (issue #261, design decisions D8/D9) is the job-level
    ``{file_href: [chunk_index, ...]}`` map of chunks still owed. It is emitted
    unconditionally as ``unfinished_chunks`` (the flat count) and
    ``unfinished_files`` (the map itself, so the completion card can name the
    files - Phase 5 of the plan reuses this one field, no second source of
    truth). Callers pass a live reference; it is copied here so a later
    in-place mutation of the caller's dict never reaches back into an
    already-emitted payload. When the current file's own outcome has not yet
    been folded into the map (the per-chunk callback fires mid-file, before the
    file returns), the emitted count legitimately lags by one file's worth of
    work; the post-file emit folds it in first, so it is always exact.

    ``untranslated_units`` is the same map narrowed to the chunks that actually
    fell back to their source text (CHUNK_UNTRANSLATED only, never
    CHUNK_PENDING). Its total is emitted as ``untranslated_chunks`` and is what
    the live Fallbacks stat card counts, so a retry that heals a chunk makes the
    card go down. It is deliberately NOT the same number as
    ``unfinished_chunks``: an interrupted job owes every chunk it never reached,
    and none of those is a fallback.

    ``run_total_chunks`` / ``run_is_repair`` describe the scope of *this pass*
    so the live panel can report a repair pass as its own little job ("1 TOTAL
    / 1 COMPLETED") instead of the book it is patching ("12 TOTAL / 10
    COMPLETED, 83%"). ``run_total_chunks`` is the number of chunks the pass will
    attempt - known before the first one, so the bar has a denominator
    immediately - and ``run_processed_chunks`` below is its completed
    counterpart. ``run_is_repair`` is True only when the work set came from
    retry tickets; it is an explicit flag rather than a shape the frontend has
    to infer from the numbers. Both are live-payload only: nothing here is
    persisted, and ``total_chunks`` / ``completed_chunks`` keep their
    book-level meaning for the checkpoint and the resumable-job card.

    ``run_prior_counts`` is the part of those cumulative counters that belongs
    to *earlier* passes of a resumed job (see ``_process_all_content_files``).
    Subtracting it yields the ``run_*`` twins the UI needs to express honest
    per-run percentages. They are emitted unconditionally - equal to the
    accumulated values on a fresh run, where nothing was restored - because
    ``state_manager.update_stats`` merges key by key: a key that appears only
    sometimes would leave a stale value behind.
    """
    fs = file_stats or {}
    uu = unfinished_units or {}
    ut = untranslated_units or {}
    prior = run_prior_counts or {}
    combined = {
        name: getattr(acc, name) + fs.get(name, 0)
        for name in _RUN_RATE_COUNTERS
    }
    payload = {
        'total_chunks': total_chunks,
        'completed_chunks': completed_chunks,
        'failed_chunks': acc.failed_chunks + fs.get('failed_chunks', 0),
        'token_alignment_used': combined['token_alignment_used'],
        'fallback_used': combined['fallback_used'],
        'placeholder_errors': combined['placeholder_errors'],
        'processed_chunks': combined['processed_chunks'],
        'successful_after_retry': combined['successful_after_retry'],
        'quality_warning_fired': acc.quality_warning_fired or fs.get('quality_warning_fired', False),
        # Plain Text Mode paragraph alignment (issue #253). Emitted for every
        # format: they are 0 on any path that is not Plain Text Mode, and a
        # counter nobody can read is a counter that does not exist.
        'paragraph_count_mismatches': (acc.paragraph_count_mismatches
                                       + fs.get('paragraph_count_mismatches', 0)),
        'paragraph_retry_recovered': (acc.paragraph_retry_recovered
                                      + fs.get('paragraph_retry_recovered', 0)),
        'paragraph_repair_failed': (acc.paragraph_repair_failed
                                    + fs.get('paragraph_repair_failed', 0)),
        'total_tokens': (acc.total_tokens_processed + acc.total_tokens_generated
                         + fs.get('total_tokens_processed', 0) + fs.get('total_tokens_generated', 0)),
        # Chunk-level unfinished work (issue #261, D8/D9). Emitted
        # unconditionally (0 / {} when there is nothing) because
        # state_manager.update_stats merges key by key: once these keys exist
        # they must be refreshed on every payload, or a stale non-zero value
        # would survive to the completion verdict.
        'unfinished_chunks': sum(len(v) for v in uu.values()),
        'unfinished_files': dict(uu),
        # Chunks currently sitting in their source text (Phase 3 fallback), as
        # opposed to chunks merely not reached yet. Emitted unconditionally for
        # the same key-by-key merge reason as above: the live Fallbacks card
        # trusts this key whenever it is present, so it must always be present
        # and always be fresh.
        'untranslated_chunks': sum(len(v) for v in ut.values()),
        # Scope of this pass, for the live progress panel (see the docstring).
        # Emitted unconditionally for the same key-by-key merge reason as the
        # keys above: a key that appears only on repair passes would leave a
        # stale True (or a stale denominator) behind on the next pass.
        'run_total_chunks': (total_chunks if run_total_chunks is None
                             else max(0, int(run_total_chunks))),
        'run_is_repair': bool(run_is_repair),
    }
    # Per-run twins. Clamped at 0: a stale snapshot must never produce a
    # negative counter, whatever the arithmetic says.
    for name in _RUN_RATE_COUNTERS:
        payload[f'run_{name}'] = max(0, combined[name] - prior.get(name, 0))
    return payload


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
    restored_docs: Optional[Dict[str, etree._Element]] = None,
    parallel_workers: int = 1
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
        Dictionary with processing results, including 'unfinished_units':
        {file_href: [chunk_index, ...]} - the complete current picture of the
        chunks this job still has to translate (issue #261).
    """
    from .translation_metrics import TranslationMetrics

    # Pre-count chunks for accurate progress tracking
    plain_text_mode = bool(prompt_options and prompt_options.get('plain_text_mode'))
    total_chunks, chunks_per_file = await _precount_chunks(
        content_files, opf_dir, max_tokens_per_chunk, log_callback,
        plain_text_mode=plain_text_mode,
    )

    # The progress denominator is the translation chunk count. In-translation
    # refinement (CLI --refine) is a per-file polish pass reported via logs; it
    # no longer doubles the total.
    effective_total_chunks = total_chunks

    # Start with restored documents
    parsed_xhtml_docs: Dict[str, etree._Element] = restored_docs.copy() if restored_docs else {}
    total_files = len(content_files)
    completed_files = len(parsed_xhtml_docs)
    failed_files = 0
    was_interrupted = False

    # Accumulate translation statistics. On resume, rehydrate the cross-file
    # fallback counters from the checkpoint so the Fallbacks stat card does
    # not restart at zero (issue #180). Per-file metrics are still restored
    # from the partial XHTML state inside xhtml_translator.
    accumulated_stats = TranslationMetrics()
    job_progress: Dict = {}
    if (checkpoint_manager and translation_id and resume_from_index > 0):
        try:
            job = checkpoint_manager.get_job(translation_id)
        except Exception:
            job = None
        if job:
            job_progress = job.get('progress') or {}
            snapshot = job_progress.get('epub_accumulated_stats')
            _restore_accumulated_stats(snapshot, accumulated_stats)

    # Everything in the emitted cumulative counters that was NOT produced by
    # this pass. The UI divides by a per-run denominator, so it needs the
    # accumulated values minus this baseline; the accumulated values themselves
    # stay untouched (issue #180).
    #
    # It starts as the cross-file snapshot just restored, and grows by each
    # re-entered file's own restored counters: `xhtml_translator` rebuilds a
    # file's TranslationMetrics from its XHTML partial state, so a re-entered
    # file's `file_stats` replay the chunks it already did in an earlier pass -
    # chunks the snapshot above *also* counts (they were merged into
    # `accumulated_stats` before the pause). Without that second term the
    # per-run numbers would carry every re-entered file's prior work.
    run_prior_counts = {name: getattr(accumulated_stats, name)
                        for name in _RUN_RATE_COUNTERS}

    # === Re-entry tickets: files below the resume pointer that still hold
    # unfinished chunks (issue #261) ===
    #
    # `unfinished_units` is the job-level index of remaining work
    # ({file_href: [chunk_index, ...]}, design decision D8). It starts from what
    # the job already recorded, so a file that is not re-entered during this
    # pass keeps its ticket; every file this pass processes overwrites (or
    # clears) its own entry, and the whole dict is written back - never merged.
    #
    # A ticket is only granted when the per-file partial state exists, validates
    # and still reports unfinished chunks (D7). The state is the payload, the map
    # is only the index. Without a state the file must stay skipped: the copy on
    # disk was overwritten with its translated version by restore_epub_files, so
    # re-entering would re-chunk an already-translated body and translate it a
    # second time.
    #
    # The lookup is restricted to the hrefs the map lists, so a clean resume
    # touches no extra file on disk.
    unfinished_units: Dict[str, List[int]] = {}
    # In-memory projection of the very same `chunk_statuses`, narrowed to the
    # chunks that fell back to their source text. It feeds the live Fallbacks
    # stat card, which must count damage (untranslated) and not backlog
    # (pending). It is deliberately NOT persisted: it is derivable from the
    # per-file partial states, and a second stored map would be a second truth.
    untranslated_units: Dict[str, List[int]] = {}
    retry_tickets: Dict[str, List[int]] = {}
    if resume_from_index > 0:
        listed_units = job_progress.get('epub_unfinished_units')
        if isinstance(listed_units, dict):
            unfinished_units = {str(href): list(indices or [])
                                for href, indices in listed_units.items()}
        for href in unfinished_units:
            state = None
            if checkpoint_manager and translation_id:
                try:
                    state = checkpoint_manager.load_xhtml_partial_state(
                        translation_id, href)
                except Exception:
                    state = None
            # Two projections of one truth, not two sources of truth: both lists
            # come from this single read of `state.chunk_statuses`, so they can
            # never disagree. The stored `epub_unfinished_units` map merges
            # pending with untranslated and cannot yield the narrow subset - the
            # state can, and the loop has to load it anyway to grant the ticket.
            statuses = state.chunk_statuses if state else None
            indices = unfinished_chunk_indices(statuses)
            ticket_untranslated = untranslated_chunk_indices(statuses)
            if ticket_untranslated:
                untranslated_units[href] = ticket_untranslated
            if indices:
                retry_tickets[href] = indices
            elif log_callback:
                log_callback("epub_retry_state_missing",
                             f"WARNING: {href} is listed as unfinished but has no "
                             f"usable partial state - skipped (re-entering it "
                             f"without one would re-translate a translated file)")

    # Track global chunk progress. A file that is going to be re-entered must
    # NOT be pre-counted here: the loop adds its full chunk count again once it
    # has been processed, and counting it twice pushes the progress past 100%.
    completed_chunks_global = 0
    for idx in range(resume_from_index):
        if idx < len(chunks_per_file) and content_files[idx] not in retry_tickets:
            completed_chunks_global += chunks_per_file[idx]

    # === Scope of THIS pass, for the live progress panel ===
    #
    # A repair pass re-translates a handful of chunks of an otherwise finished
    # book. Reporting it against the book ("12 TOTAL / 10 COMPLETED, 83%",
    # creeping to 100% in tiny steps, ETA computed from a denominator that has
    # nothing to do with the work in flight) describes nothing the user asked
    # for, so the panel gets the pass's own numerator/denominator instead.
    #
    # The arithmetic, per file index:
    #   - below the resume pointer: only the chunks its retry ticket lists. The
    #     ticket IS the file's work set - `_translate_all_chunks_with_checkpoint`
    #     computes `unfinished_chunk_indices(statuses) | range(start, len)` and
    #     the second term is already inside the first (invariant D4: every chunk
    #     at or above `current_chunk_index` is `pending`, hence unfinished). A
    #     file with no ticket is skipped and contributes nothing.
    #   - at or above the pointer: its full pre-counted chunk count.
    # Indexing by position rather than summing the ticket dict keeps the two
    # terms disjoint, exactly like the `completed_chunks_global` loop above.
    #
    # The file *at* the pointer is the one a previous pass was interrupted
    # inside, so it carries a partial state and will attempt only its remaining
    # chunks - fewer than `chunks_per_file` says. It has no ticket (a ticket is
    # written by `_save_checkpoint`, which only runs for files that completed),
    # so its state has to be read directly. Only that one index can be in this
    # situation: files above it were never touched.
    #
    # This matters whenever a book was interrupted *and* an earlier file left a
    # fallback behind: the tickets make `run_is_repair` True, so the panel does
    # use these numbers, and an over-counted denominator would leave the repair
    # bar stalled below 100%.
    run_is_repair = bool(retry_tickets)
    run_total_chunks = 0
    for idx, href in enumerate(content_files):
        if idx < resume_from_index:
            run_total_chunks += len(retry_tickets.get(href) or ())
        elif idx >= len(chunks_per_file):
            continue
        elif idx == resume_from_index and resume_from_index > 0:
            run_total_chunks += _pending_chunk_count(
                checkpoint_manager, translation_id, href, chunks_per_file[idx])
        else:
            run_total_chunks += chunks_per_file[idx]

    # The file index written to the checkpoint must never rewind. `chunk_index`
    # is turned back into `resume_from_index = current_chunk_index + 1` by
    # load_checkpoint, so writing the index of a re-entered *early* file (say
    # file 0 of 3, resumed at 3) would drop the pointer to 1 - and the next
    # resume would re-enter files 1 and 2 with no partial state, re-chunk their
    # already-translated bodies and translate them a second time. Tracking the
    # last completed index monotonically, floored at the pointer we resumed
    # from, is what prevents that. DO NOT replace this with `file_idx`.
    last_completed_file_idx = resume_from_index - 1

    # Send initial stats if resuming (to update UI immediately). Forward the
    # restored fallback counters so the UI hydrates the Fallbacks card with
    # the work already done before the pause, instead of showing 0.
    if stats_callback and resume_from_index > 0:
        stats_callback(_global_stats_payload(
            effective_total_chunks, completed_chunks_global, accumulated_stats,
            unfinished_units=unfinished_units,
            run_prior_counts=run_prior_counts,
            untranslated_units=untranslated_units,
            run_total_chunks=run_total_chunks,
            run_is_repair=run_is_repair))

    for file_idx, content_href in enumerate(content_files):
        # Check for interruption
        if check_interruption_callback and check_interruption_callback():
            was_interrupted = True
            if log_callback:
                log_callback("epub_translation_interrupted",
                             f"Translation interrupted at file {file_idx + 1}/{total_files}")
            break

        # Skip if already processed (resume), unless the file still holds
        # unfinished chunks and has a partial state to re-enter it with.
        retry_indices = retry_tickets.get(content_href)
        if file_idx < resume_from_index and not retry_indices:
            completed_files += 1
            continue

        if retry_indices and log_callback:
            log_callback("epub_retry_file",
                         f"Re-entering {content_href} to retry "
                         f"{len(retry_indices)} unfinished chunk(s): {retry_indices}")

        file_path = _resolve_content_path(opf_dir, content_href)
        chunks_in_this_file = chunks_per_file[file_idx] if file_idx < len(chunks_per_file) else 0
        # A re-entered file is already in parsed_xhtml_docs (it came from
        # restored_docs, which seeded completed_files), so its successful
        # translation must not be counted a second time.
        already_translated = file_path in parsed_xhtml_docs

        # Fold this file's already-counted work into the per-run baseline before
        # it is translated (the chunk loop overwrites its partial state as it
        # goes, so it has to be read now). Only files that were entered in an
        # earlier pass can have one: a file the loop completed had its state
        # deleted, and files above the pointer were never touched.
        if resume_from_index > 0:
            _add_file_prior_counts(run_prior_counts, checkpoint_manager,
                                   translation_id, content_href)

        if log_callback:
            log_callback("epub_file_translate_start",
                         f"Translating file {file_idx + 1}/{total_files}: {content_href} ({chunks_in_this_file} chunks)")

        # Create stats wrapper that reports global statistics
        # NOTE: completed_chunks_global represents chunks from ALL previous files (not including current)
        def file_stats_wrapper(file_stats_dict: Dict):
            """Convert file-level stats to global stats by merging with accumulated stats"""
            if not stats_callback:
                return

            # current_file_completed = the current file's completed chunks
            # (TranslationMetrics.to_dict reports the file as fully complete once
            # its translation finishes, so refinement does not advance this).
            current_file_completed = file_stats_dict.get('completed_chunks', 0)
            global_completed = completed_chunks_global + current_file_completed

            # Report combined stats (accumulated + current file). The fallback
            # counters are included so the Fallbacks stat card updates live.
            # `unfinished_units` still reflects the picture as of the *previous*
            # file at this point (the current file's own outcome is only known
            # once it returns, folded in below) - an accepted one-file lag for
            # this live mid-file callback; the post-file emit is exact.
            stats_callback(_global_stats_payload(
                total_chunks, global_completed, accumulated_stats, file_stats_dict,
                unfinished_units=unfinished_units,
                run_prior_counts=run_prior_counts,
                untranslated_units=untranslated_units,
                run_total_chunks=run_total_chunks,
                run_is_repair=run_is_repair))

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
            parallel_workers=parallel_workers,
        )

        # Update global chunk counter. A fully-translated file contributes all
        # its chunks. On interruption the file stopped early, so count only the
        # chunks actually processed — otherwise the bar jumps to 100% at the
        # moment of pausing and then drops on resume. Clean runs are unaffected
        # (processed == chunks_in_this_file when the file completes).
        interrupted_now = bool(check_interruption_callback and check_interruption_callback())
        if interrupted_now and file_stats is not None:
            completed_chunks_global += min(chunks_in_this_file, file_stats.processed_chunks)
        else:
            completed_chunks_global += chunks_in_this_file

        # Accumulate statistics
        if file_stats:
            accumulated_stats.merge(file_stats)

        # What this file still owes, read from the partial state the chunk loop
        # just wrote. This has to happen BEFORE the post-file stats report below
        # (so 'unfinished_chunks'/'unfinished_files' reflect this file's own
        # outcome instead of lagging by one file) and BEFORE _save_checkpoint,
        # which deletes that state when the file comes back clean.
        file_unfinished: List[int] = []
        file_untranslated: List[int] = []
        if checkpoint_manager and translation_id:
            try:
                state_after = checkpoint_manager.load_xhtml_partial_state(
                    translation_id, content_href)
            except Exception:
                state_after = None
            if state_after is not None:
                # Two projections of one truth, not two sources of truth: the
                # persisted `unfinished` set (pending + untranslated, D8) and the
                # in-memory `untranslated`-only set are both derived from this
                # single read of `chunk_statuses`, so they cannot drift apart.
                statuses_after = state_after.chunk_statuses
                file_unfinished = unfinished_chunk_indices(statuses_after)
                file_untranslated = untranslated_chunk_indices(statuses_after)

        if file_unfinished:
            unfinished_units[content_href] = file_unfinished
        else:
            unfinished_units.pop(content_href, None)

        if file_untranslated:
            untranslated_units[content_href] = file_untranslated
        else:
            untranslated_units.pop(content_href, None)

        # Report stats if callback provided
        if stats_callback and file_stats:
            stats_callback(_global_stats_payload(
                effective_total_chunks, completed_chunks_global, accumulated_stats,
                unfinished_units=unfinished_units,
                run_prior_counts=run_prior_counts,
                untranslated_units=untranslated_units,
                run_total_chunks=run_total_chunks,
                run_is_repair=run_is_repair))

        # Save the document if translation succeeded
        if success and doc_root is not None:
            parsed_xhtml_docs[file_path] = doc_root
            if not already_translated:
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
            last_completed_file_idx = max(last_completed_file_idx, file_idx)
            await _save_checkpoint(
                checkpoint_manager, translation_id, last_completed_file_idx,
                content_href, doc_root, file_path, temp_dir, log_callback,
                total_chunks=total_chunks,
                completed_chunks=completed_chunks_global,
                failed_chunks=accumulated_stats.failed_chunks,
                epub_accumulated_stats=_snapshot_accumulated_stats(accumulated_stats),
                unfinished_units=unfinished_units,
                file_unfinished=file_unfinished
            )

    # Final progress
    return {
        'parsed_docs': parsed_xhtml_docs,
        'completed_files': completed_files,
        'failed_files': failed_files,
        'total_chunks': effective_total_chunks,
        'completed_chunks': completed_chunks_global,
        'failed_chunks': accumulated_stats.failed_chunks,
        'translation_stats': accumulated_stats,
        'was_interrupted': was_interrupted,
        # Complete current picture of the chunks still to translate, so callers
        # do not have to recompute it from disk (issue #261).
        'unfinished_units': unfinished_units
    }


def _snapshot_accumulated_stats(metrics) -> Dict:
    """Capture the cross-file fallback counters we want to survive a resume.

    Only the cumulative cross-file counters need to round-trip; per-file
    metrics are already rehydrated by xhtml_translator from the partial
    state JSON. Going through dedicated fields (not TranslationMetrics.to_dict)
    avoids the doubled-total_chunks adjustment that to_dict() does for the UI.
    """
    return {
        'token_alignment_used': metrics.token_alignment_used,
        'token_alignment_success': metrics.token_alignment_success,
        'fallback_used': metrics.fallback_used,
        'failed_chunks': metrics.failed_chunks,
        'placeholder_errors': metrics.placeholder_errors,
        'processed_chunks': metrics.processed_chunks,
        'successful_first_try': metrics.successful_first_try,
        'successful_after_retry': metrics.successful_after_retry,
        'retry_attempts': metrics.retry_attempts,
        'quality_warning_fired': metrics.quality_warning_fired,
        'fallback_warning_fired': metrics.fallback_warning_fired,
        'paragraph_count_mismatches': metrics.paragraph_count_mismatches,
        'paragraph_retry_recovered': metrics.paragraph_retry_recovered,
        'paragraph_repair_failed': metrics.paragraph_repair_failed,
        'correction_attempts': metrics.correction_attempts,
        'correction_success': metrics.correction_success,
        'total_tokens_processed': metrics.total_tokens_processed,
        'total_tokens_generated': metrics.total_tokens_generated,
        'refinement_chunks_completed': metrics.refinement_chunks_completed,
    }


def _pending_chunk_count(checkpoint_manager, translation_id, content_href: str,
                         precounted: int) -> int:
    """How many chunks a file will actually attempt on this pass.

    A file that a previous pass was interrupted inside carries a partial state,
    so it resumes from `current_chunk_index` and attempts only what that state
    still reports as unfinished - fewer than the pre-counted total. Used for the
    repair-progress denominator, where over-counting would leave the bar stalled
    below 100%.

    Falls back to `precounted` whenever there is no usable state, which is the
    correct answer for a file that has never been entered.
    """
    if not (checkpoint_manager and translation_id):
        return precounted
    try:
        state = checkpoint_manager.load_xhtml_partial_state(
            translation_id, content_href)
    except Exception:
        state = None
    if state is None:
        return precounted
    return len(unfinished_chunk_indices(state.chunk_statuses))


def _add_file_prior_counts(prior_counts: Dict, checkpoint_manager, translation_id,
                           content_href: str) -> None:
    """Fold one file's already-counted work into the per-run baseline.

    A file re-entered on a resume has its TranslationMetrics rebuilt from its
    XHTML partial state, so the `file_stats` it reports include the chunks it
    translated in an earlier pass. Those same chunks are already inside the
    restored cross-file snapshot, so they appear twice in the cumulative
    counters the payload emits; counting them once more in the baseline is what
    keeps the `run_*` twins equal to this pass's own work.

    Best effort by design: no state (fresh file), an unreadable one, or a state
    the translator ends up ignoring simply means nothing is subtracted, and the
    payload's `max(0, ...)` clamp absorbs the rest.
    """
    if not (checkpoint_manager and translation_id):
        return
    try:
        state = checkpoint_manager.load_xhtml_partial_state(translation_id, content_href)
    except Exception:
        return
    stats = getattr(state, 'stats', None) if state is not None else None
    if not isinstance(stats, dict):
        return
    for name in _RUN_RATE_COUNTERS:
        try:
            prior_counts[name] = prior_counts.get(name, 0) + int(stats.get(name, 0) or 0)
        except (TypeError, ValueError):
            continue


def _restore_accumulated_stats(snapshot: Dict, metrics) -> None:
    """Restore counters captured by `_snapshot_accumulated_stats` into a fresh metrics object."""
    if not snapshot:
        return
    metrics.token_alignment_used = snapshot.get('token_alignment_used', 0)
    metrics.token_alignment_success = snapshot.get('token_alignment_success', 0)
    metrics.fallback_used = snapshot.get('fallback_used', 0)
    metrics.failed_chunks = snapshot.get('failed_chunks', 0)
    metrics.placeholder_errors = snapshot.get('placeholder_errors', 0)
    metrics.processed_chunks = snapshot.get('processed_chunks', 0)
    metrics.successful_first_try = snapshot.get('successful_first_try', 0)
    metrics.successful_after_retry = snapshot.get('successful_after_retry', 0)
    metrics.retry_attempts = snapshot.get('retry_attempts', 0)
    metrics.quality_warning_fired = snapshot.get('quality_warning_fired', False)
    metrics.fallback_warning_fired = snapshot.get('fallback_warning_fired', False)
    metrics.paragraph_count_mismatches = snapshot.get('paragraph_count_mismatches', 0)
    metrics.paragraph_retry_recovered = snapshot.get('paragraph_retry_recovered', 0)
    metrics.paragraph_repair_failed = snapshot.get('paragraph_repair_failed', 0)
    metrics.correction_attempts = snapshot.get('correction_attempts', 0)
    metrics.correction_success = snapshot.get('correction_success', 0)
    metrics.total_tokens_processed = snapshot.get('total_tokens_processed', 0)
    metrics.total_tokens_generated = snapshot.get('total_tokens_generated', 0)
    metrics.refinement_chunks_completed = snapshot.get('refinement_chunks_completed', 0)


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
    failed_chunks: int = 0,
    epub_accumulated_stats: Optional[Dict] = None,
    unfinished_units: Optional[Dict[str, List[int]]] = None,
    file_unfinished: Optional[List[int]] = None
) -> None:
    """Save checkpoint for a translated file.

    Args:
        unfinished_units: Job-level index of the chunks still to translate
            ({file_href: [chunk_index, ...]}, issue #261). Stored verbatim in
            the job progress so the next resume knows which files to re-enter.
        file_unfinished: The chunk indices this file still owes. When it is
            empty the per-file partial state is deleted (the file is done);
            when it is not, the state is KEPT, because it is the only place
            that records which chunk is still in the source language.
    """
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
            # Delete partial state AFTER successful file save (atomicity guarantee),
            # and ONLY when the file has nothing left to translate. A file with
            # unfinished chunks keeps its state: that state is what records which
            # chunk is still in the source language, and deleting it would make
            # the chunk unrecoverable all over again (issue #261).
            #
            # `content_href` is the authoritative partial-state key: it is what
            # `xhtml_translator._save_state` saves under and what
            # `_translate_single_xhtml_file` loads with. `file_rel_path` (the
            # temp-dir-relative path) is a different string and is only the key
            # used by `save_epub_file` above; deleting with it here never matched
            # the state actually written, which is what left stale "finished"
            # partial states on disk and blocked retries (issue #261).
            if file_unfinished:
                if log_callback:
                    log_callback("xhtml_partial_state_kept_unfinished",
                        f"📌 Partial state kept for {content_href}: "
                        f"chunk(s) {file_unfinished} still untranslated")
            else:
                checkpoint_manager.delete_xhtml_partial_state(translation_id, content_href)
                if log_callback:
                    log_callback("xhtml_partial_state_deleted_after_save",
                        f"🗑️ Partial state deleted for {content_href} (file saved successfully)")

            # Update checkpoint progress with chunk statistics. The
            # `epub_accumulated_stats` snapshot is what rehydrates the
            # Fallbacks stat card on resume — without it the cross-file
            # counters reset to zero after a pause (issue #180).
            checkpoint_manager.save_checkpoint(
                translation_id=translation_id,
                # Uniform convention: store the LAST COMPLETED file index
                # (resume adds +1), matching TXT/SRT. load_checkpoint maps it
                # back via the 'resume_index_semantics' marker.
                chunk_index=file_idx,
                original_text=content_href,
                translated_text=content_href,
                chunk_data={'last_file': content_href, 'file_type': 'epub_xhtml'},
                total_chunks=total_chunks,
                completed_chunks=completed_chunks,
                failed_chunks=failed_chunks,
                epub_accumulated_stats=epub_accumulated_stats,
                unfinished_units=unfinished_units
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


def _update_ncx_toc_labels_from_translated_docs(
    opf_dir: str,
    parsed_xhtml_docs: Dict[str, etree._Element],
    log_callback: Optional[Callable] = None
) -> Dict[str, int]:
    """
    Update EPUB2 NCX TOC labels using translated XHTML headings.

    The NCX side-panel TOC stores display labels separately from the XHTML body
    in ``navLabel/text`` nodes. Body translation does not touch those labels,
    so this helper maps each NCX ``content src`` target back to the already
    translated XHTML document and copies the translated heading text into the
    NCX label. The ``content src`` attribute is never modified, preserving
    reader navigation.
    """
    stats = {"updated": 0, "unchanged": 0, "errors": 0}
    opf_dir_path = Path(opf_dir)
    ncx_paths = list(opf_dir_path.glob("*.ncx"))
    if not ncx_paths:
        return stats

    docs_by_path = {
        os.path.normcase(os.path.abspath(path)): doc
        for path, doc in parsed_xhtml_docs.items()
    }
    ns = {"ncx": "http://www.daisy.org/z3986/2005/ncx/"}

    for ncx_path in ncx_paths:
        try:
            parser = etree.XMLParser(encoding="utf-8", recover=True, remove_blank_text=False)
            tree = etree.parse(str(ncx_path), parser)
            changed = False

            for nav_point in tree.findall(".//ncx:navPoint", namespaces=ns):
                text_el = nav_point.find("./ncx:navLabel/ncx:text", namespaces=ns)
                content_el = nav_point.find("./ncx:content", namespaces=ns)
                if text_el is None or content_el is None:
                    stats["unchanged"] += 1
                    continue

                src = content_el.get("src")
                if not src:
                    stats["unchanged"] += 1
                    continue

                translated_title = _get_translated_title_for_src(
                    src=src,
                    base_dir=opf_dir,
                    docs_by_path=docs_by_path
                )
                if not translated_title:
                    stats["unchanged"] += 1
                    continue

                if text_el.text != translated_title:
                    text_el.text = translated_title
                    changed = True
                    stats["updated"] += 1
                else:
                    stats["unchanged"] += 1

            if changed:
                tree.write(
                    str(ncx_path),
                    encoding="utf-8",
                    xml_declaration=True,
                    pretty_print=True
                )
        except Exception as exc:
            stats["errors"] += 1
            if log_callback:
                log_callback("epub_ncx_toc_error", f"Could not update NCX TOC '{ncx_path}': {exc}")

    if log_callback and (stats["updated"] or stats["errors"]):
        log_callback(
            "epub_ncx_toc_updated",
            f"📚 NCX TOC labels updated: {stats['updated']} updated, "
            f"{stats['unchanged']} unchanged, {stats['errors']} errors"
        )

    return stats


def _get_translated_title_for_src(
    src: str,
    base_dir: str,
    docs_by_path: Dict[str, etree._Element]
) -> Optional[str]:
    """Resolve a TOC ``src``/``href`` to the translated heading text.

    ``base_dir`` is the directory the link is relative to (the NCX file's
    directory for EPUB2, the nav document's directory for EPUB3).
    """
    href, fragment = _split_ncx_src(src)
    if not href:
        return None

    file_path = os.path.normcase(os.path.abspath(os.path.join(base_dir, href)))
    doc_root = docs_by_path.get(file_path)
    if doc_root is None:
        return None

    if fragment:
        anchor = _find_element_by_id_or_name(doc_root, fragment)
        if anchor is not None:
            title = _extract_heading_text_near_anchor(anchor)
            if title:
                return title

    return _extract_first_heading_text(doc_root)


def _find_nav_doc_href(opf_tree: etree._ElementTree) -> Optional[str]:
    """Return the href of the EPUB3 navigation document, or None.

    The nav document is the manifest item carrying ``properties="nav"``.
    """
    opf_root = opf_tree.getroot()
    manifest = opf_root.find('.//opf:manifest', namespaces=NAMESPACES)
    if manifest is None:
        return None
    for item in manifest.findall('.//opf:item', namespaces=NAMESPACES):
        props = (item.get("properties") or "").split()
        if "nav" in props:
            return item.get("href")
    return None


def _set_anchor_text(anchor: etree._Element, title: str) -> None:
    """Replace an ``<a>`` element's visible text with ``title``.

    TOC links are normally plain text, but some carry inline markup (e.g. a
    numbering ``<span>``). Clearing children and setting ``text`` guarantees
    the link displays exactly the translated heading.
    """
    for child in list(anchor):
        anchor.remove(child)
    anchor.text = title


def _update_nav_toc_labels_from_translated_docs(
    opf_dir: str,
    opf_tree: etree._ElementTree,
    parsed_xhtml_docs: Dict[str, etree._Element],
    log_callback: Optional[Callable] = None
) -> Dict[str, int]:
    """
    Update EPUB3 nav document TOC links using translated XHTML headings.

    EPUB3 readers build their table of contents from the navigation document
    (``<nav epub:type="toc">``) rather than the legacy NCX. The link labels in
    that document are stored separately from the chapter bodies, so body
    translation never touches them. This helper maps each TOC ``<a href>``
    target back to the already translated XHTML heading and copies the
    translated text into the link, leaving ``href`` untouched so navigation
    keeps working. Non-TOC navs (``landmarks``, ``page-list``) are skipped.
    """
    stats = {"updated": 0, "unchanged": 0, "errors": 0}

    nav_href = _find_nav_doc_href(opf_tree)
    if not nav_href:
        return stats

    nav_path = os.path.join(opf_dir, unquote(nav_href))
    if not os.path.exists(nav_path):
        return stats

    # nav links are relative to the nav document's own directory.
    nav_dir = os.path.dirname(nav_path)
    docs_by_path = {
        os.path.normcase(os.path.abspath(path)): doc
        for path, doc in parsed_xhtml_docs.items()
    }
    epub_type_attr = f"{{{NAMESPACES['epub']}}}type"
    skip_types = {"landmarks", "page-list"}

    try:
        parser = etree.XMLParser(encoding="utf-8", recover=True, remove_blank_text=False)
        tree = etree.parse(str(nav_path), parser)
        changed = False

        for nav_el in tree.iter():
            if _local_name(nav_el) != "nav":
                continue
            if (nav_el.get(epub_type_attr) or "").strip() in skip_types:
                continue

            for anchor in nav_el.iter():
                if _local_name(anchor) != "a":
                    continue
                src = anchor.get("href")
                if not src:
                    continue

                translated_title = _get_translated_title_for_src(
                    src=src,
                    base_dir=nav_dir,
                    docs_by_path=docs_by_path
                )
                if not translated_title:
                    stats["unchanged"] += 1
                    continue

                if _normalized_element_text(anchor) != translated_title:
                    _set_anchor_text(anchor, translated_title)
                    changed = True
                    stats["updated"] += 1
                else:
                    stats["unchanged"] += 1

        if changed:
            tree.write(
                str(nav_path),
                encoding="utf-8",
                xml_declaration=True,
                pretty_print=False
            )
    except Exception as exc:
        stats["errors"] += 1
        if log_callback:
            log_callback("epub_nav_toc_error", f"Could not update nav TOC '{nav_path}': {exc}")

    if log_callback and (stats["updated"] or stats["errors"]):
        log_callback(
            "epub_nav_toc_updated",
            f"📚 EPUB3 nav TOC labels updated: {stats['updated']} updated, "
            f"{stats['unchanged']} unchanged, {stats['errors']} errors"
        )

    return stats


def _split_ncx_src(src: str) -> Tuple[str, Optional[str]]:
    href, _, fragment = src.partition("#")
    return unquote(href), unquote(fragment) if fragment else None


def _find_element_by_id_or_name(doc_root: etree._Element, fragment: str) -> Optional[etree._Element]:
    for element in doc_root.iter():
        if element.get("id") == fragment or element.get("name") == fragment:
            return element
    return None


def _extract_heading_text_near_anchor(anchor: etree._Element) -> Optional[str]:
    current = anchor
    while current is not None:
        if _local_name(current) in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            return _normalized_element_text(current)
        current = current.getparent()

    return _normalized_element_text(anchor)


def _extract_first_heading_text(doc_root: etree._Element) -> Optional[str]:
    for element in doc_root.iter():
        if _local_name(element) in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            title = _normalized_element_text(element)
            if title:
                return title
    return None


def _normalized_element_text(element: etree._Element) -> Optional[str]:
    text = " ".join("".join(element.itertext()).split())
    return text or None


def _local_name(element: etree._Element) -> str:
    try:
        return etree.QName(element).localname.lower()
    except ValueError:
        return str(element.tag).split("}", 1)[-1].lower()


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
        # Update language. dc:language must be an ISO 639-1 code; resolve it
        # with the same helper used for the XHTML lang attributes so OPF and
        # XHTML never contradict each other. When the target cannot be
        # resolved, leave the element unchanged rather than write a bogus code.
        lang_el = metadata.find('.//dc:language', namespaces=NAMESPACES)
        if lang_el is not None:
            lang_code = get_language_code(target_language)
            if lang_code:
                lang_el.text = lang_code

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
