"""
Subtitle-specific translation module
"""
import time
from typing import List, Dict, Optional, Tuple
from tqdm.auto import tqdm

from src.prompts.prompts import (
    generate_subtitle_block_prompt,
    generate_subtitle_refinement_block_prompt,
)
from src.config import (
    TRANSLATE_TAG_IN,
    TRANSLATE_TAG_OUT,
    SRT_LINES_PER_BLOCK,
)

# Sentinel large enough to disable the char cap when grouping subtitles:
# block sizing is now purely fixed-count (SRT_LINES_PER_BLOCK) for both
# translate and refine, so the legacy char cap is unused.
_NO_CHAR_CAP = 10 ** 12
from .llm_client import create_llm_client
from .post_processor import clean_translated_text
from .translator import generate_translation_request, _build_chunk_glossary_block
from .epub import TagPreserver


async def translate_subtitles(subtitles: List[Dict[str, str]], source_language: str,
                            target_language: str, model_name: str, api_endpoint: str,
                            log_callback=None,
                            stats_callback=None, check_interruption_callback=None, custom_instructions="",
                            llm_provider="ollama", gemini_api_key=None, openai_api_key=None,
                            openrouter_api_key=None,
                            enable_post_processing=False, post_processing_instructions="",
                            prompt_options=None) -> Dict[int, str]:
    """
    Translate subtitle entries preserving structure
    
    Args:
        subtitles (list): List of subtitle dictionaries from SRT parser
        source_language (str): Source language
        target_language (str): Target language
        model_name (str): LLM model name
        api_endpoint (str): API endpoint        log_callback (callable): Logging callback
        stats_callback (callable): Statistics update callback
        check_interruption_callback (callable): Interruption check callback
        
    Returns:
        dict: Mapping of subtitle index to translated text
    """
    total_subtitles = len(subtitles)
    translations = {}
    completed_count = 0
    failed_count = 0
    
    if log_callback:
        log_callback("srt_translation_start", f"Starting translation of {total_subtitles} subtitles...")
    
    # Create LLM client based on provider or custom endpoint
    llm_client = create_llm_client(llm_provider, gemini_api_key, api_endpoint, model_name, openai_api_key, openrouter_api_key, log_callback=log_callback)
    
    try:
        iterator = tqdm(enumerate(subtitles), total=total_subtitles, 
                       desc=f"Translating subtitles ({source_language} to {target_language})", 
                       unit="subtitle") if not log_callback else enumerate(subtitles)
        
        for idx, subtitle in iterator:
            if check_interruption_callback and check_interruption_callback():
                if log_callback:
                    log_callback("srt_translation_interrupted", 
                               f"Translation interrupted at subtitle {idx+1}/{total_subtitles}")
                else:
                    tqdm.write(f"\nTranslation interrupted at subtitle {idx+1}/{total_subtitles}")
                break

            text_to_translate = subtitle['text'].strip()
            
            if not text_to_translate:
                translations[idx] = ""
                completed_count += 1
                continue
            
            context_before = ""
            context_after = ""
            
            if idx > 0 and idx-1 in translations:
                context_before = translations[idx-1]
            elif idx > 0:
                context_before = subtitles[idx-1].get('text', '')
            
            if idx < len(subtitles) - 1:
                context_after = subtitles[idx+1].get('text', '')
            
            translated_text = await generate_translation_request(
                text_to_translate,
                context_before,
                context_after,
                "",
                source_language,
                target_language,
                model_name,
                llm_client=llm_client,
                log_callback=log_callback,
                custom_instructions=custom_instructions
            )
            
            if translated_text is not None:
                # Single point of cleaning for subtitles
                translations[idx] = clean_translated_text(translated_text)
                completed_count += 1
            else:
                # Keep original text if translation fails
                err_msg = f"Failed to translate subtitle {idx+1}"
                if log_callback:
                    log_callback("srt_subtitle_error", err_msg)
                else:
                    tqdm.write(f"\n{err_msg}")
                translations[idx] = text_to_translate  # Keep original
                failed_count += 1
            
            if stats_callback and total_subtitles > 0:
                stats_callback({
                    'completed_chunks': completed_count,
                    'failed_chunks': failed_count,
                    'total_chunks': total_subtitles,
                })
    
        if log_callback:
            log_callback("srt_translation_complete",
                        f"Completed translation: {completed_count} successful, {failed_count} failed")

        # Refinement pass (if enabled)
        enable_refinement = (prompt_options and prompt_options.get('refine')) if prompt_options else enable_post_processing

        if enable_refinement and translations:
            if log_callback:
                log_callback("srt_refinement_start", "✨ Starting SRT refinement pass to polish translation quality...")

            # Fixed-count grouping for refine (no char cap): every block
            # sent to the LLM has the same shape, which makes marker
            # accounting predictable across the whole file.
            from src.core.srt_processor import SRTProcessor
            refine_blocks = SRTProcessor().group_subtitles_for_translation(
                subtitles, SRT_LINES_PER_BLOCK, _NO_CHAR_CAP
            )

            refined_translations = await refine_subtitle_translations(
                translations=translations,
                target_language=target_language,
                model_name=model_name,
                llm_client=llm_client,
                log_callback=log_callback,
                prompt_options=prompt_options,
                post_processing_instructions=post_processing_instructions,
                stats_callback=stats_callback,
                check_interruption_callback=check_interruption_callback,
                subtitle_blocks=refine_blocks,
                subtitle_positions={id(s): i for i, s in enumerate(subtitles)},
            )

            if log_callback:
                successful_refinements = sum(1 for idx in translations if translations[idx] != refined_translations.get(idx, translations[idx]))
                log_callback("srt_refinement_complete",
                           f"✨ Refinement complete: {successful_refinements}/{len(translations)} subtitles improved")

            translations = refined_translations

    finally:
        # Clean up LLM client resources if created
        if llm_client:
            await llm_client.close()

    return translations


async def refine_subtitle_translations(
    translations: Dict[int, str],
    target_language: str,
    model_name: str,
    llm_client,
    log_callback=None,
    prompt_options=None,
    post_processing_instructions="",
    stats_callback=None,
    check_interruption_callback=None,
    subtitle_blocks: Optional[List[List[Dict[str, str]]]] = None,
    subtitle_positions: Optional[Dict[int, int]] = None,
) -> Dict[int, str]:
    """
    Refine subtitle translations using a second LLM pass.

    Mirrors the block-based translate pass: subtitles are grouped into blocks
    and refined together in a single LLM call per block, with per-subtitle
    fallback when the block response cannot be parsed.

    Args:
        translations: Dict mapping subtitle index to translated text
        target_language: Target language
        model_name: LLM model name
        llm_client: LLM client instance
        log_callback: Optional logging callback
        prompt_options: Optional prompt options dict
        post_processing_instructions: Additional refinement instructions
        stats_callback: Optional callback to report per-subtitle progress
        check_interruption_callback: Optional callback to abort the pass early
        subtitle_blocks: Optional list of subtitle blocks (each block is a
            list of subtitle dicts with a 'number' field). When provided,
            the refinement mirrors the translate-pass block structure.
            When None, blocks are derived from the translations dict using
            the configured SRT block size and char cap.
        subtitle_positions: Optional identity map (id(subtitle dict) ->
            position in the parsed subtitle list). When provided, blocks
            are resolved to the same index space the caller keyed
            `translations` with. Without it the cue number printed in the
            file is used as a fallback, which is only safe when numbering
            is exactly 1..N (issue #205).

    Returns:
        Dict mapping subtitle index to refined text
    """
    from src.core.srt_processor import SRTProcessor

    total_subtitles = len(translations)
    refined_translations: Dict[int, str] = {}
    completed_count = 0
    failed_count = 0
    # Transient per-job state (e.g. glossary cap warning dedupe) — never persisted.
    runtime_state: dict = {}

    if log_callback:
        log_callback("srt_refinement_info", f"Refining {total_subtitles} subtitles...")

    # Build the list of global-index groups we will refine together.
    # Each group is a list of global subtitle indices, ordered.
    if subtitle_blocks:
        index_groups: List[List[int]] = []
        for block in subtitle_blocks:
            group: List[int] = []
            for subtitle in block:
                if subtitle_positions is not None:
                    g_idx = subtitle_positions.get(id(subtitle))
                    if g_idx is None:
                        continue
                else:
                    # Fallback: printed cue number. Only correct for files
                    # numbered exactly 1..N — callers should pass
                    # subtitle_positions instead (issue #205).
                    try:
                        g_idx = int(subtitle['number']) - 1
                    except (KeyError, ValueError, TypeError):
                        continue
                if g_idx in translations and translations[g_idx].strip():
                    group.append(g_idx)
            if group:
                index_groups.append(group)
    else:
        # Re-group from the translations dict when no block structure is
        # supplied (e.g. refine-only path that didn't pass blocks through).
        # Fixed-count only — no char cap.
        sorted_indices = sorted(translations.keys())
        index_groups = []
        current: List[int] = []
        for g_idx in sorted_indices:
            text = translations[g_idx]
            if not text or not text.strip():
                continue
            if len(current) >= SRT_LINES_PER_BLOCK:
                index_groups.append(current)
                current = []
            current.append(g_idx)
        if current:
            index_groups.append(current)

    total_blocks = len(index_groups)
    srt_processor = SRTProcessor()
    previous_refined_block = ""

    # Preserve empty subtitles untouched so the output stays complete.
    # Count them as completed: they require no refinement, so leaving them
    # out of completed_count would prevent progress from ever reaching 100%.
    for g_idx, text in translations.items():
        if not text or not text.strip():
            refined_translations[g_idx] = text
            completed_count += 1

    max_block_attempts = 2  # initial attempt + 1 retry with reinforced reminder

    for block_idx, group in enumerate(index_groups):
        if check_interruption_callback and check_interruption_callback():
            if log_callback:
                log_callback(
                    "srt_refinement_interrupted",
                    f"Refinement interrupted at block {block_idx + 1}/{total_blocks}"
                )
            # Carry over any not-yet-refined subtitles unchanged.
            for remaining_group in index_groups[block_idx:]:
                for g_idx in remaining_group:
                    refined_translations.setdefault(g_idx, translations[g_idx])
            break

        # Build local-index tuples and the local->global mapping.
        local_subtitle_tuples: List[Tuple[int, str]] = []
        local_to_global: Dict[int, int] = {}
        for local_idx, g_idx in enumerate(group):
            local_subtitle_tuples.append((local_idx, translations[g_idx]))
            local_to_global[local_idx] = g_idx

        block_text_for_glossary = "\n".join(text for _, text in local_subtitle_tuples)
        glossary_block = _build_chunk_glossary_block(
            block_text_for_glossary, prompt_options, log_callback=log_callback,
            runtime_state=runtime_state, target_language=target_language,
        )

        block_refined: Dict[int, str] = {}
        expected_local_indices = list(range(len(local_subtitle_tuples)))

        for attempt in range(max_block_attempts):
            if check_interruption_callback and check_interruption_callback():
                break

            # On retry, reinforce the reminder with the exact missing indices.
            extra_instructions = post_processing_instructions or ''
            if attempt > 0:
                missing_local = [li for li in expected_local_indices
                                 if local_to_global[li] not in block_refined]
                missing_str = ", ".join(f"[{li}]" for li in missing_local)
                extra_instructions = (
                    (extra_instructions + "\n\n" if extra_instructions else "")
                    + f"CRITICAL: Your previous response was incomplete. "
                    f"You MUST output ALL {len(local_subtitle_tuples)} indices "
                    f"[0] through [{len(local_subtitle_tuples) - 1}] in order, "
                    f"each followed by the refined subtitle. "
                    f"Missing indices last time: {missing_str}. Do NOT stop early."
                )

            try:
                prompt_pair = generate_subtitle_refinement_block_prompt(
                    subtitle_blocks=local_subtitle_tuples,
                    previous_refined_block=previous_refined_block,
                    target_language=target_language,
                    additional_instructions=extra_instructions,
                    glossary_block=glossary_block,
                )

                if log_callback and attempt > 0:
                    log_callback("srt_refinement_retry",
                                 f"Block {block_idx + 1}: retry attempt {attempt} "
                                 f"({len(local_subtitle_tuples) - len(block_refined)} subtitles still missing)")

                llm_response = await llm_client.make_request(
                    prompt_pair.user, model_name, system_prompt=prompt_pair.system
                )

                if llm_response and llm_response.content:
                    if log_callback:
                        log_callback("refinement_response", "Refinement response received", data={
                            'type': 'refinement_response',
                            'response': llm_response.content,
                            'model': model_name,
                        })

                    refined_block_text = llm_client.extract_translation(llm_response.content)
                    if refined_block_text:
                        parsed = srt_processor.extract_block_translations_with_remapping(
                            refined_block_text, local_to_global
                        )
                        # Merge only the newly recovered (non-empty) entries.
                        for g_idx, text in parsed.items():
                            if g_idx not in block_refined and text.strip():
                                block_refined[g_idx] = text

                # All subtitles recovered? stop retrying.
                if len(block_refined) == len(group):
                    break

            except Exception as e:
                # Re-raise RateLimitError to trigger auto-pause
                from src.core.llm.exceptions import RateLimitError
                if isinstance(e, RateLimitError):
                    # Hand the caller a complete map (refined for finished
                    # blocks, the draft for this block onward) plus the first
                    # position still to refine, so the partial pass can be
                    # saved and a resume can skip what is already done.
                    for remaining_group in index_groups[block_idx:]:
                        for g_idx in remaining_group:
                            refined_translations.setdefault(g_idx, translations[g_idx])
                    e.partial_result = {
                        'translations': dict(refined_translations),
                        'next_index': group[0],
                    }
                    raise
                if log_callback:
                    log_callback("srt_refinement_error",
                                 f"Block {block_idx + 1} attempt {attempt + 1}: {e}")

        # Apply refined where available, keep original draft otherwise.
        # No per-subtitle fallback: small models corrupt single-subtitle calls
        # with hallucinated content from the surrounding context.
        # Both branches bump completed_count: the subtitle is finalized in
        # the output either way, so progress reaches 100%.
        # NOTE: we do NOT bump failed_count for kept-original. In refine
        # semantics, the subtitle already has a valid translation — refine
        # just didn't polish it further. The file is complete; reporting
        # failed_chunks > 0 here would make the backend mark the job as
        # 'partial' and hide the download UI (handlers.py:588), which is
        # the right behavior for translate but wrong for refine.
        # Per-subtitle log_callback below preserves visibility of which
        # subs fell back to original.
        # Emit stats per-subtitle so the UI ticks 1-by-1 (e.g. 781, 782, ...,
        # 786) instead of jumping by the block size all at once.
        for g_idx in group:
            if g_idx in block_refined:
                refined_translations[g_idx] = block_refined[g_idx]
            else:
                refined_translations[g_idx] = translations[g_idx]
                if log_callback:
                    log_callback("srt_refinement_fallback",
                                 f"Subtitle {g_idx + 1}: keeping original (block missed this index)")
            completed_count += 1
            if stats_callback:
                stats_callback({
                    'total_chunks': total_subtitles,
                    'completed_chunks': completed_count,
                    'failed_chunks': failed_count,
                })

        # Update previous_refined_block context (last up to 5 subtitles of this group).
        last_items = []
        for local_idx, g_idx in enumerate(group[-5:]):
            last_items.append(f"[{local_idx}]{refined_translations.get(g_idx, translations[g_idx])}")
        previous_refined_block = "\n".join(last_items)

    return refined_translations


async def translate_subtitles_in_blocks(subtitle_blocks: List[List[Dict[str, str]]],
                                      source_language: str, target_language: str,
                                      model_name: str, api_endpoint: str,
                                      log_callback=None,
                                      stats_callback=None, check_interruption_callback=None,
                                      custom_instructions="", llm_provider="ollama",
                                      gemini_api_key=None, openai_api_key=None,
                                      openrouter_api_key=None,
                                      enable_post_processing=False,
                                      post_processing_instructions="",
                                      checkpoint_manager=None, translation_id=None,
                                      resume_from_block_index=0,
                                      prompt_options=None) -> Dict[int, str]:
    """
    Translate subtitle entries in blocks for better context preservation.

    Args:
        subtitle_blocks: List of subtitle blocks (each block is a list of subtitle dicts)
        source_language: Source language
        target_language: Target language
        model_name: LLM model name
        api_endpoint: API endpoint        log_callback: Logging callback
        stats_callback: Statistics update callback
        check_interruption_callback: Interruption check callback
        custom_instructions: Additional translation instructions
        checkpoint_manager: CheckpointManager instance for saving progress
        translation_id: Job ID for checkpoint saving
        resume_from_block_index: Block index to resume from (for resumed jobs)

    Returns:
        dict: Mapping of subtitle index to translated text
    """
    from src.core.srt_processor import SRTProcessor
    from .llm_client import default_client
    
    srt_processor = SRTProcessor()
    
    total_blocks = len(subtitle_blocks)
    total_subtitles = sum(len(block) for block in subtitle_blocks)
    translations = {}
    completed_count = 0  # Number of subtitles completed
    failed_count = 0  # Number of subtitles failed
    completed_blocks_count = 0  # Number of blocks completed
    failed_blocks_count = 0  # Number of blocks failed
    previous_translation_block = ""
    # Transient per-job state (e.g. glossary cap warning dedupe) — never persisted.
    runtime_state: dict = {}

    # Handle resume: load previously translated blocks
    if checkpoint_manager and translation_id and resume_from_block_index > 0:
        checkpoint_data = checkpoint_manager.load_checkpoint(translation_id)
        if checkpoint_data:
            # Restore completed translations
            saved_chunks = checkpoint_data['chunks']
            for chunk in saved_chunks:
                if chunk['status'] == 'completed' and chunk['translated_text']:
                    # chunk_data contains the block_translations dict
                    block_translations = chunk.get('chunk_data', {}).get('block_translations', {})
                    for idx, trans_text in block_translations.items():
                        translations[int(idx)] = trans_text
                        completed_count += 1
                    completed_blocks_count += 1  # Count completed blocks
                elif chunk['status'] == 'failed':
                    # Restore original text for failed blocks
                    block_translations = chunk.get('chunk_data', {}).get('block_translations', {})
                    for idx, original_text in block_translations.items():
                        translations[int(idx)] = original_text
                        failed_count += 1
                    failed_blocks_count += 1  # Count failed blocks

            # Restore translation context for continuity
            if checkpoint_data.get('translation_context'):
                context = checkpoint_data['translation_context']
                previous_translation_block = context.get('previous_translation_block', '')

            if log_callback:
                log_callback("checkpoint_resumed",
                    f"Resumed from checkpoint: {completed_count} subtitles already completed, "
                    f"resuming from block {resume_from_block_index + 1}/{total_blocks}")

    if log_callback:
        log_callback("srt_block_translation_start",
                    f"Starting block translation: {total_subtitles} subtitles in {total_blocks} blocks...")
    
    # Create LLM client based on provider or custom endpoint
    llm_client = create_llm_client(llm_provider, gemini_api_key, api_endpoint, model_name, openai_api_key, openrouter_api_key, log_callback=log_callback)
    
    try:
        for block_idx, block in enumerate(subtitle_blocks):
            # Skip already processed blocks when resuming
            if block_idx < resume_from_block_index:
                continue

            if check_interruption_callback and check_interruption_callback():
                if log_callback:
                    log_callback("srt_translation_interrupted",
                               f"Translation interrupted at block {block_idx+1}/{total_blocks}")
                else:
                    tqdm.write(f"\nTranslation interrupted at block {block_idx+1}/{total_blocks}")
                # Mark as paused when interrupted
                if checkpoint_manager and translation_id:
                    checkpoint_manager.mark_paused(translation_id)
                break

            # Prepare subtitle blocks with indices
            subtitle_tuples = []
            block_indices = []  # Global indices (original)

            for subtitle in block:
                idx = int(subtitle['number']) - 1  # Convert to 0-based index
                text = subtitle['text'].strip()
                if text:  # Only include non-empty subtitles
                    subtitle_tuples.append((idx, text))
                    block_indices.append(idx)

            if not subtitle_tuples:
                continue

            # Renumber to local indices (0, 1, 2...) for LLM simplicity
            # Create mapping: local_index -> global_index
            local_to_global = {local_idx: global_idx for local_idx, (global_idx, _) in enumerate(subtitle_tuples)}

            # Create subtitle tuples with local indices for LLM
            local_subtitle_tuples = [(local_idx, text) for local_idx, (_, text) in enumerate(subtitle_tuples)]

            # Build glossary block from the concatenated subtitle text in this block.
            block_text_for_glossary = "\n".join(text for _, text in local_subtitle_tuples)
            glossary_block = _build_chunk_glossary_block(
                block_text_for_glossary, prompt_options, log_callback=log_callback,
                runtime_state=runtime_state, target_language=target_language,
            )

            # Generate system and user prompts for this block with local indices
            prompt_pair = generate_subtitle_block_prompt(
                local_subtitle_tuples,
                previous_translation_block,
                source_language,
                target_language,
                TRANSLATE_TAG_IN,
                TRANSLATE_TAG_OUT,
                custom_instructions,
                glossary_block=glossary_block,
            )
            
            # Make translation request using LLM client with retry mechanism
            max_retries = 3
            retry_count = 0
            translated_block_text = None

            while retry_count < max_retries:
                try:
                    if retry_count > 0 and log_callback:
                        log_callback("srt_block_retry", f"Retry attempt {retry_count} for block {block_idx+1}")

                    # Log the LLM request with structured data for web interface
                    if log_callback:
                        log_callback("llm_request", "Sending subtitle block to LLM", data={
                            'type': 'llm_request',
                            'system_prompt': prompt_pair.system,
                            'user_prompt': prompt_pair.user,
                            'model': model_name
                        })

                    # Use provided client or default - pass system and user prompts separately
                    client = llm_client or default_client
                    start_time = time.time()
                    llm_response = await client.make_request(
                        prompt_pair.user, model_name, system_prompt=prompt_pair.system
                    )
                    execution_time = time.time() - start_time

                    # Extract raw response content
                    full_raw_response = llm_response.content if llm_response else None

                    # Log the LLM response with structured data for web interface preview
                    if full_raw_response and log_callback:
                        log_callback("llm_response", "LLM Response received", data={
                            'type': 'llm_response',
                            'response': full_raw_response,
                            'execution_time': execution_time,
                            'model': model_name
                        })

                    if full_raw_response:
                        translated_block_text = client.extract_translation(full_raw_response)

                        # Validate placeholder tags if translation succeeded
                        if translated_block_text:
                            # Check if all expected LOCAL [NUMBER] tags are present (0, 1, 2...)
                            expected_local_indices = list(range(len(local_subtitle_tuples)))
                            expected_tags = set(f"[{idx}]" for idx in expected_local_indices)
                            found_tags = set()
                            import re
                            for match in re.finditer(r'\[(\d+)\]', translated_block_text):
                                found_tags.add(match.group(0))

                            missing_tags = expected_tags - found_tags

                            if missing_tags:
                                if log_callback:
                                    log_callback("srt_placeholder_validation_failed",
                                               f"Block {block_idx+1} missing tags: {missing_tags}")

                                if retry_count < max_retries - 1:
                                    # Enhance prompt with stronger instructions about preserving tags
                                    prompt_pair = generate_subtitle_block_prompt(
                                        local_subtitle_tuples,
                                        previous_translation_block,
                                        source_language,
                                        target_language,
                                        TRANSLATE_TAG_IN,
                                        TRANSLATE_TAG_OUT,
                                        custom_instructions + f"\n\nCRITICAL: You MUST preserve ALL [NUMBER] tags EXACTLY as they appear. Missing tags: {', '.join(missing_tags)}",
                                        glossary_block=glossary_block,
                                    )
                                    retry_count += 1
                                    continue
                                else:
                                    # Final retry failed, will use original text
                                    translated_block_text = None
                                    break
                            else:
                                # All tags present, translation successful
                                if retry_count > 0 and log_callback:
                                    log_callback("srt_retry_successful",
                                               f"Block {block_idx+1} translation successful after {retry_count} retries")
                                break
                        else:
                            # No translation extracted
                            if retry_count < max_retries - 1:
                                retry_count += 1
                                continue
                            else:
                                break
                    else:
                        translated_block_text = None
                        if retry_count < max_retries - 1:
                            retry_count += 1
                            continue
                        else:
                            break
                            
                except Exception as e:
                    # Re-raise RateLimitError to trigger auto-pause
                    from src.core.llm.exceptions import RateLimitError
                    if isinstance(e, RateLimitError):
                        raise
                    if log_callback:
                        log_callback("srt_block_translation_error", f"Error: {str(e)}")
                    translated_block_text = None
                    if retry_count < max_retries - 1:
                        retry_count += 1
                        continue
                    else:
                        break
            
            if translated_block_text:
                # Extract individual translations from block with local->global index remapping
                block_translations = srt_processor.extract_block_translations_with_remapping(
                    translated_block_text, local_to_global
                )

                # Update translations dictionary
                for idx, trans_text in block_translations.items():
                    translations[idx] = trans_text
                    completed_count += 1

                # Track failed translations in block (individual subtitles that couldn't be extracted)
                subtitles_failed_in_block = 0
                for idx in block_indices:
                    if idx not in block_translations:
                        # Keep original text for missing translations
                        for subtitle in block:
                            if int(subtitle['number']) - 1 == idx:
                                translations[idx] = subtitle['text']
                                failed_count += 1
                                subtitles_failed_in_block += 1
                                break

                # Increment completed blocks count (even if some subtitles failed in extraction)
                completed_blocks_count += 1

                # Store translated block for context (last 5 subtitles)
                # Use LOCAL indices (0-4) for consistency with how the LLM sees the data
                last_subtitles = []
                sorted_global_indices = sorted(block_translations.keys())[-5:]
                for local_idx, global_idx in enumerate(sorted_global_indices):
                    last_subtitles.append(f"[{local_idx}]{block_translations[global_idx]}")
                previous_translation_block = '\n'.join(last_subtitles)

                # Save checkpoint after successful block translation
                if checkpoint_manager and translation_id:
                    # Create chunk_data with block information
                    block_chunk_data = {
                        'block_translations': block_translations,
                        'block_indices': block_indices
                    }
                    translation_context = {
                        'previous_translation_block': previous_translation_block
                    }
                    # For SRT: completed_chunks = number of BLOCKS completed (not individual subtitles)
                    checkpoint_manager.save_checkpoint(
                        translation_id=translation_id,
                        chunk_index=block_idx,
                        original_text=translated_block_text,  # Store the raw LLM response
                        translated_text=translated_block_text,
                        chunk_data=block_chunk_data,
                        translation_context=translation_context,
                        total_chunks=total_blocks,
                        completed_chunks=completed_blocks_count,
                        failed_chunks=failed_blocks_count
                    )
                
            else:
                # Block translation failed - keep original text
                err_msg = f"Failed to translate block {block_idx+1}"
                if log_callback:
                    log_callback("srt_block_error", err_msg)
                else:
                    tqdm.write(f"\n{err_msg}")

                # Store original text for failed translations
                failed_block_translations = {}
                for subtitle in block:
                    idx = int(subtitle['number']) - 1
                    translations[idx] = subtitle['text']
                    failed_block_translations[idx] = subtitle['text']
                    failed_count += 1

                # Increment failed blocks count
                failed_blocks_count += 1

                previous_translation_block = ""  # Reset context on failure

                # Save checkpoint for failed block
                if checkpoint_manager and translation_id:
                    block_chunk_data = {
                        'block_translations': failed_block_translations,
                        'block_indices': block_indices
                    }
                    translation_context = {
                        'previous_translation_block': previous_translation_block
                    }
                    checkpoint_manager.save_checkpoint(
                        translation_id=translation_id,
                        chunk_index=block_idx,
                        original_text="",  # No translated text for failed blocks
                        translated_text=None,  # Mark as failed
                        chunk_data=block_chunk_data,
                        translation_context=translation_context,
                        total_chunks=total_blocks,
                        completed_chunks=completed_blocks_count,
                        failed_chunks=failed_blocks_count
                    )
            
            if stats_callback and total_subtitles > 0:
                stats_callback({
                    'completed_chunks': completed_count,
                    'failed_chunks': failed_count,
                    'total_chunks': total_subtitles,
                    'completed_blocks': block_idx + 1,
                    'total_blocks': total_blocks,
                })
        
        if log_callback:
            log_callback("srt_block_translation_complete",
                        f"Completed block translation: {completed_count} successful, {failed_count} failed")

        # Refinement pass (if enabled)
        enable_refinement = (prompt_options and prompt_options.get('refine')) or enable_post_processing

        if enable_refinement and translations:
            if log_callback:
                log_callback("srt_refinement_start", "✨ Starting SRT refinement pass to polish translation quality...")

            # Refine uses fixed-count blocks (no char cap), independent of
            # the translate block structure: refine just rewrites existing
            # target-language text, so we want predictable bloc sizes for
            # reliable [N] marker accounting.
            from src.core.srt_processor import SRTProcessor as _SRTProcessorRefine
            flat_subtitles = [s for blk in subtitle_blocks for s in blk]
            refine_blocks = _SRTProcessorRefine().group_subtitles_for_translation(
                flat_subtitles, SRT_LINES_PER_BLOCK, _NO_CHAR_CAP
            )

            refined_translations = await refine_subtitle_translations(
                translations=translations,
                target_language=target_language,
                model_name=model_name,
                llm_client=llm_client,
                log_callback=log_callback,
                prompt_options=prompt_options,
                post_processing_instructions=post_processing_instructions,
                stats_callback=stats_callback,
                check_interruption_callback=check_interruption_callback,
                subtitle_blocks=refine_blocks,
            )

            if log_callback:
                successful_refinements = sum(1 for idx in translations if translations[idx] != refined_translations.get(idx, translations[idx]))
                log_callback("srt_refinement_complete",
                           f"✨ Refinement complete: {successful_refinements}/{len(translations)} subtitles improved")

            translations = refined_translations

    finally:
        # Clean up LLM client resources if created
        if llm_client:
            await llm_client.close()

    return translations