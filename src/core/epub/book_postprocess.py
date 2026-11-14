"""Steps that run on the extracted book once its files are translated.

Each function is one numbered step of ``translate_epub_file``; the ordering
constraints between them are documented there.
"""

import os
from typing import Callable, Dict, Optional

import aiofiles
from lxml import etree

from src.config import EPUB_SCRIPT_NORMALIZATION_ENABLED, EPUB_TRANSLATE_METADATA_ENABLED

from ..post_processor import clean_residual_tag_placeholders
from .archive import repackage_epub, serialize_xhtml
from .book_state import BookRunState
from .cjk_typography import apply_script_normalization_to_epub_directory
from .metadata_translator import translate_opf_metadata
from .rtl_support import apply_rtl_to_epub_directory, is_rtl_language
from .run_context import EpubRun


async def save_translated_files(
    parsed_xhtml_docs: Dict[str, etree._Element],
    log_callback: Optional[Callable] = None
) -> None:
    """Write every kept XHTML document back to the extracted book."""
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
                await f_out.write(serialize_xhtml(doc_root))
        except Exception as e_write:
            if log_callback:
                log_callback("epub_write_error", f"Error writing '{file_path_abs}': {e_write}")


def toc_source_docs(book: BookRunState) -> Dict[str, etree._Element]:
    """The documents TOC labels are read from: files whose translation failed
    keep the source text, so their TOC entries keep the original label."""
    return {path: doc for path, doc in book.parsed_docs.items()
            if path not in book.failed_doc_paths}


async def translate_packaging_metadata(run: EpubRun, manifest_data: Dict) -> None:
    """Step 5.5, guarded: never fails an otherwise-successful job.

    The module already swallows its own failures; this is the belt-and-braces
    guard for anything it did not anticipate.
    """
    if not EPUB_TRANSLATE_METADATA_ENABLED:
        return
    try:
        await translate_opf_metadata(
            opf_tree=manifest_data['opf_tree'],
            opf_path=manifest_data['opf_path'],
            opf_dir=manifest_data['opf_dir'],
            source_language=run.source_language,
            target_language=run.target_language,
            llm_client=run.llm_client,
            model_name=run.model_name,
            log_callback=run.log_callback
        )
    except Exception as e_meta:
        run.log("epub_metadata_translation_failed",
                f"⚠️ Packaging metadata localization "
                f"failed and was skipped: {e_meta}")


def apply_text_direction(run: EpubRun, temp_dir: str) -> None:
    """Step 6: RTL->RTL, LTR->RTL, RTL->LTR and LTR->LTR transitions."""
    target_language, source_language = run.target_language, run.source_language
    if is_rtl_language(target_language):
        run.log("epub_rtl_start", f"🔄 Applying RTL layout for {target_language}...")
    elif is_rtl_language(source_language):
        run.log("epub_rtl_start", f"🔄 Resetting to LTR layout (translating from {source_language})...")

    rtl_result = apply_rtl_to_epub_directory(temp_dir, target_language, source_language)

    if rtl_result.get('was_transition'):
        # RTL -> LTR transition
        run.log("epub_ltr_applied",
                f"✅ LTR reset applied: {rtl_result['css_removed']} files cleaned, "
                f"text direction set to left-to-right")
    elif rtl_result['is_rtl']:
        run.log("epub_rtl_applied",
                f"✅ RTL support applied: {rtl_result['css_injected']} files updated, "
                f"OPF progression: {'RTL' if rtl_result['opf_updated'] else 'unchanged'}")


def normalize_source_script(run: EpubRun, temp_dir: str) -> None:
    """Step 6.6, guarded: cosmetic relative to the translated text, so it must
    never fail an otherwise-successful job, even on a failure its own per-file
    guards did not anticipate (e.g. an OSError while walking the directory)."""
    if not EPUB_SCRIPT_NORMALIZATION_ENABLED:
        return
    try:
        norm_result = apply_script_normalization_to_epub_directory(
            temp_dir, run.source_language, run.target_language,
            log_callback=run.log_callback
        )
        _log_script_normalization(norm_result, run.log_callback)
    except Exception as e_norm:
        run.log("epub_script_norm_failed",
                f"⚠️ Source-script typography normalization "
                f"failed and was skipped: {e_norm}")


def _log_script_normalization(norm: dict, log_callback: Optional[Callable]) -> None:
    """Emit the user-visible log lines for step 6.6's result dict.

    See apply_script_normalization_to_epub_directory's docstring for the exact
    keys of `norm`.
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


def write_output_epub(run: EpubRun, temp_dir: str, output_filepath: str,
                      was_interrupted: bool) -> str:
    """Step 7: repackage; returns the path actually written.

    An interrupted pass writes to a `[partial] ` filename so users can tell
    partial outputs from completed ones at a glance. A complete one removes
    any leftover `[partial ...]` siblings from earlier interrupted runs
    targeting this same output.
    """
    from src.utils.file_utils import find_partial_output_paths, get_partial_output_path

    if was_interrupted:
        output_filepath = get_partial_output_path(output_filepath)
        run.log("epub_partial_output_marked",
                f"💾 Partial EPUB will be saved as: {os.path.basename(output_filepath)}")

    repackage_epub(temp_dir=temp_dir, output_filepath=output_filepath,
                   log_callback=run.log_callback)

    if not was_interrupted:
        for stale in find_partial_output_paths(output_filepath):
            try:
                os.remove(stale)
                run.log("epub_partial_cleanup",
                        f"🗑️ Removed stale partial: {os.path.basename(stale)}")
            except OSError as e:
                run.log("epub_partial_cleanup_failed",
                        f"⚠️ Could not remove stale partial {os.path.basename(stale)}: {e}")
    return output_filepath


def log_completion(run: EpubRun, book: BookRunState) -> None:
    """Final summary and layout status."""
    run.log("epub_save_success",
            f"✅ EPUB translation complete: {book.completed_files} files translated, {book.failed_files} failed")
    if is_rtl_language(run.target_language):
        run.log("epub_rtl_complete",
                "📖 EPUB ready for RTL reading: text direction is right-to-left")
    elif is_rtl_language(run.source_language):
        run.log("epub_ltr_complete",
                "📖 EPUB ready for LTR reading: text direction reset to left-to-right")
