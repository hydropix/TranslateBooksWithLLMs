"""
Sample & Compare routes.

Runs an arbitrary number of LLM configurations in parallel on N short extracts
of an uploaded book, streaming each cell back to the client over WebSocket as it
completes.
No persistence: state lives in `SampleStateManager` and is dropped after 1
hour or on server restart.
"""
import asyncio
import logging
import os
import random
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from flask import Blueprint, jsonify, request

from src.api.services.path_validator import PathValidator
from src.config import (
    MAX_TOKENS_PER_CHUNK, OLLAMA_NUM_CTX,
    REQUEST_TIMEOUT, SRT_LINES_PER_BLOCK,
)
from src.core.glossary import build_glossary_block, filter_glossary
from src.core.glossary.models import GlossaryConfig
from src.core.llm.factory import create_llm_provider
from src.core.pricing.pricing_data import get_default_pricing
from src.core.sampling import cap_chunk_text, select_sample_indices
from src.api.api_keys import provider_env_var as _provider_env_var
from src.api.api_keys import resolve_api_key as _resolve_api_key
from src.core.text_processor import split_text_into_chunks
from src.prompts.prompts import (
    generate_refinement_prompt, generate_translation_prompt,
)
from src.utils.custom_instructions import is_safe_filename, load_custom_instructions
from src.utils.file_detector import detect_file_type
from src.utils.language_detector import LanguageDetector


# Per-run concurrency cap. The product spec asks for `min(K * N, 8)` to avoid
# hammering providers; this is enforced per sample run via an asyncio.Semaphore.
SAMPLE_CONCURRENCY_CAP = 8

# Sampling defaults used when a request omits the fields (safety net — both
# front-ends always send explicit values). Must mirror the frontend single
# source in src/web/static/js/sample/sample-defaults.js.
DEFAULT_N_SAMPLES = 5
DEFAULT_MAX_CHARS = 400

logger = logging.getLogger(__name__)


def _clamp_int(value: Any, default: int, lo: int, hi: int) -> int:
    """Parse `value` as int (falling back to `default` when None) and clamp to [lo, hi].

    Raises ValueError/TypeError for non-numeric input so callers can return 400.
    """
    return max(lo, min(hi, int(default if value is None else value)))


def _small_document_warning(total: int, count: int, requested: int) -> Dict[str, Any]:
    """Structured warning for "doc has fewer interior units than requested".

    Returned as {code, params} (not a pre-formatted English string) so the
    client can translate it reactively via the `sample:warning_small_document`
    i18n key, whose {{total}}/{{count}}/{{requested}} placeholders match params.
    """
    return {
        "code": "warning_small_document",
        "params": {"total": total, "count": count, "requested": requested},
    }


def _extract_plain_text(file_path: str, file_type: str) -> str:
    """
    Extract the textual content of a file for chunking + sampling.

    For TXT we read directly. For EPUB/DOCX we reuse the plain extractors used
    by Plain Text Mode in the main translate flow. SRT is handled by the
    caller (sampled at the cue-group level, not via this helper).
    """
    ft = file_type.lower()
    if ft == "txt":
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    if ft == "epub":
        return _extract_epub_text(file_path)
    if ft == "docx":
        return _extract_docx_text(file_path)
    raise ValueError(f"Unsupported file type for sampling: {file_type}")


def _extract_epub_text(file_path: str) -> str:
    """Concatenate the block-level text of every XHTML body in the EPUB.

    Reuses the main translate flow's `extract_plain_paragraphs`, which walks
    block elements once and returns one string per paragraph. (An earlier
    version iterated *every* element and joined `itertext()` per element, which
    re-emitted each paragraph once per ancestor — heavily duplicating content
    and corrupting the sampled extracts.)
    """
    import zipfile
    from lxml import etree

    from src.core.epub.plain_extractor import _local_name, extract_plain_paragraphs

    parts: List[str] = []
    try:
        with zipfile.ZipFile(file_path, "r") as zf:
            for name in zf.namelist():
                lower = name.lower()
                if not (lower.endswith(".xhtml") or lower.endswith(".html") or lower.endswith(".htm")):
                    continue
                try:
                    raw = zf.read(name)
                    root = etree.fromstring(raw)
                except Exception:
                    continue
                body = next(
                    (el for el in root.iter()
                     if isinstance(el.tag, str) and _local_name(el) == "body"),
                    root,
                )
                paragraphs, _tags, _images = extract_plain_paragraphs(body)
                for text in paragraphs:
                    text = (text or "").strip()
                    if text:
                        parts.append(text)
                        parts.append("\n\n")
    except zipfile.BadZipFile:
        raise ValueError("Invalid EPUB file (not a zip archive)")
    return "".join(parts).strip()


def _extract_docx_text(file_path: str) -> str:
    """Concatenate paragraph text from a DOCX using the plain extractor."""
    from docx import Document

    doc = Document(file_path)
    parts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n\n".join(parts)


def _load_source_units(file_path: str, file_type: str) -> List[Dict[str, str]]:
    """
    Return a normalized list of "source units" for sampling.

    Every unit is a dict with `main_content` / `context_before` / `context_after`
    so the same item-construction code can serve TXT/EPUB/DOCX/SRT files.
    For TXT/EPUB/DOCX this is just `split_text_into_chunks`; for SRT we group
    cues into blocks of SRT_LINES_PER_BLOCK and synthesize the contexts from
    adjacent blocks.

    Deterministic: identical (file_path, file_type) always returns identical
    units, so an index produced by /initialize remains valid for /extract and
    /run later.
    """
    ft = file_type.lower()
    if ft == "srt":
        from src.core.srt_processor import SRTProcessor

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        proc = SRTProcessor()
        subtitles = proc.parse_srt(content)
        if not subtitles:
            raise ValueError("No subtitles found in SRT file")

        block_size = max(1, SRT_LINES_PER_BLOCK)
        blocks: List[str] = []
        for i in range(0, len(subtitles), block_size):
            block_text = "\n".join(
                s["text"] for s in subtitles[i:i + block_size] if s.get("text")
            )
            if block_text.strip():
                blocks.append(block_text)

        total = len(blocks)
        return [
            {
                "main_content": blocks[i],
                "context_before": blocks[i - 1] if i > 0 else "",
                "context_after": blocks[i + 1] if i + 1 < total else "",
            }
            for i in range(total)
        ]

    text = _extract_plain_text(file_path, file_type)
    if not text or not text.strip():
        raise ValueError("File is empty or unreadable")
    return split_text_into_chunks(text, max_tokens_per_chunk=MAX_TOKENS_PER_CHUNK)


def _items_for_indices(
    units: List[Dict[str, str]],
    indices: List[int],
    max_chars: int,
) -> List[Dict[str, Any]]:
    """Build sample items for the given indices, capping each main_content."""
    items: List[Dict[str, Any]] = []
    for idx in indices:
        if idx < 0 or idx >= len(units):
            continue
        unit = units[idx]
        capped, truncated = cap_chunk_text(unit.get("main_content", ""), max_chars)
        items.append({
            "index": idx,
            "source_text": capped,
            "truncated": truncated,
            "context_before": unit.get("context_before", ""),
            "context_after": unit.get("context_after", ""),
        })
    return items


def _build_srt_sample_blocks(file_path: str, n_samples: int, max_chars: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """For SRT files, sample N blocks. Returns (items, warnings)."""
    units = _load_source_units(file_path, "srt")
    total = len(units)
    if total < 3:
        raise ValueError("document too small for sampling")

    warnings: List[Dict[str, Any]] = []
    indices = select_sample_indices(total, n_samples)
    if len(indices) < n_samples:
        warnings.append(_small_document_warning(total, len(indices), n_samples))
    return _items_for_indices(units, indices, max_chars), warnings


def _build_text_sample_items(text: str, n_samples: int, max_chars: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Chunk plain text and select N representative items capped at max_chars."""
    chunks = split_text_into_chunks(text, max_tokens_per_chunk=MAX_TOKENS_PER_CHUNK)
    total = len(chunks)
    if total < 3:
        raise ValueError("document too small for sampling")

    warnings: List[Dict[str, Any]] = []
    indices = select_sample_indices(total, n_samples)
    if len(indices) < n_samples:
        warnings.append(_small_document_warning(total, len(indices), n_samples))
    return _items_for_indices(chunks, indices, max_chars), warnings


def _pick_random_unused_index(total: int, exclude: Set[int]) -> Optional[int]:
    """Pick a random interior index (1..total-2) not in `exclude`. None if all used."""
    if total < 3:
        return None
    candidates = [i for i in range(1, total - 1) if i not in exclude]
    if not candidates:
        return None
    return random.choice(candidates)


def _instantiate_provider(column: Dict[str, Any]):
    """Build an LLMProvider from a column descriptor.

    Resolves `__USE_ENV__` placeholders to the corresponding env variable.
    """
    provider = (column.get("provider") or "ollama").lower()
    env_var = _provider_env_var(provider)
    api_key = _resolve_api_key(column.get("api_key"), env_var) if env_var else None

    kwargs: Dict[str, Any] = {
        "model": column.get("model"),
        "context_window": int(column.get("context_window") or OLLAMA_NUM_CTX),
    }
    if api_key:
        kwargs["api_key"] = api_key
    endpoint = column.get("api_endpoint") or column.get("endpoint")
    if endpoint:
        kwargs["api_endpoint"] = endpoint

    return create_llm_provider(provider, **kwargs)


def _compute_cost_usd(provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    """Best-effort USD cost for one LLM call. Returns None if pricing unknown."""
    pricing = get_default_pricing(provider, model)
    if not pricing:
        return None
    input_rate = pricing.get("input", 0.0)
    output_rate = pricing.get("output", 0.0)
    return round(
        (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000,
        6,
    )


async def _execute_cell(
    *,
    sample_id: str,
    row: int,
    col: int,
    phase: str,
    prompt_pair,
    column: Dict[str, Any],
    ref_text: str,
    state: "SampleStateManager",
    socketio,
) -> Optional[str]:
    """Run one LLM call for a cell and emit its result.

    Shared by the translate and refine phases — they differ only in the prompt
    pair and the text the length ratio is measured against (`ref_text`).
    Returns the cleaned output on success, or None on error/empty/cancel. Emits
    exactly one WebSocket event (done or error), except when cancelled.
    """
    if state.is_cancelled(sample_id):
        return None

    started = time.perf_counter()
    provider = None
    try:
        provider = _instantiate_provider(column)
        response = await provider.generate(
            prompt=prompt_pair.user,
            system_prompt=prompt_pair.system,
            timeout=REQUEST_TIMEOUT,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)

        if response is None or not response.content:
            _emit_cell(
                socketio, state, sample_id, row, col, phase,
                status="error",
                output=None,
                metrics={"latency_ms": latency_ms},
                error="LLM returned an empty response",
            )
            return None

        # Strip <TRANSLATION>...</TRANSLATION> wrapper (and any <think> block)
        # like the main translation flow does. Fall back to raw content if the
        # tags are missing — same semantics as `was_fallback`.
        extracted = provider.extract_translation(response.content)
        used_fallback = response.was_fallback
        if extracted is None or not extracted.strip():
            extracted = response.content
            used_fallback = True
        output_text = extracted.strip()

        cost = _compute_cost_usd(
            column.get("provider", "ollama"),
            column.get("model", ""),
            response.prompt_tokens,
            response.completion_tokens,
        )
        src_len = max(1, len(ref_text))
        length_ratio = round(len(output_text) / src_len, 3)

        _emit_cell(
            socketio, state, sample_id, row, col, phase,
            status="done",
            output=output_text,
            metrics={
                "latency_ms": latency_ms,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "cost_usd": cost,
                "length_ratio": length_ratio,
                "was_fallback": used_fallback,
                "was_truncated": response.was_truncated,
            },
            error=None,
        )
        return output_text
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        _emit_cell(
            socketio, state, sample_id, row, col, phase,
            status="error",
            output=None,
            metrics={"latency_ms": latency_ms},
            error=str(exc),
        )
        return None
    finally:
        if provider is not None:
            try:
                await provider.close()
            except Exception:
                pass


async def _run_cell_translate(
    *,
    sample_id: str,
    row: int,
    col: int,
    item: Dict[str, Any],
    column: Dict[str, Any],
    source_language: str,
    target_language: str,
    prompt_options: Dict[str, Any],
    glossary_block: str = "",
    state: "SampleStateManager",
    socketio,
) -> Optional[str]:
    """Run a single translate call. Returns the translated text, or None."""
    prompt_pair = generate_translation_prompt(
        main_content=item["source_text"],
        context_before=item.get("context_before", ""),
        context_after=item.get("context_after", ""),
        previous_translation_context="",
        source_language=source_language,
        target_language=target_language,
        has_placeholders=False,
        prompt_options=prompt_options,
        glossary_block=glossary_block,
    )
    return await _execute_cell(
        sample_id=sample_id, row=row, col=col, phase="translate",
        prompt_pair=prompt_pair, column=column, ref_text=item["source_text"],
        state=state, socketio=socketio,
    )


async def _run_cell_refine(
    *,
    sample_id: str,
    row: int,
    col: int,
    draft_text: str,
    item: Dict[str, Any],
    column: Dict[str, Any],
    target_language: str,
    prompt_options: Dict[str, Any],
    glossary_block: str = "",
    state: "SampleStateManager",
    socketio,
) -> None:
    """Run a single refine call. Emits one WebSocket event when done."""
    prompt_pair = generate_refinement_prompt(
        draft_translation=draft_text,
        context_before=item.get("context_before", ""),
        context_after=item.get("context_after", ""),
        previous_refined_context="",
        target_language=target_language,
        has_placeholders=False,
        prompt_options=prompt_options,
        glossary_block=glossary_block,
        # A preset's refinement section reaches the refine prompt via
        # `additional_instructions` (prompt_options alone isn't read for it).
        additional_instructions=(prompt_options.get("refinement_instructions") or ""),
    )
    await _execute_cell(
        sample_id=sample_id, row=row, col=col, phase="refine",
        prompt_pair=prompt_pair, column=column, ref_text=draft_text,
        state=state, socketio=socketio,
    )


def _emit_cell(socketio, state, sample_id, row, col, phase, *, status, output, metrics, error):
    """Persist the cell result in state and emit it over WebSocket."""
    state.update_cell(
        sample_id, row, col, phase,
        status=status, output=output, metrics=metrics, error=error,
    )
    if socketio is None:
        return
    payload = {
        "sample_id": sample_id,
        "type": "cell_done" if status == "done" else "cell_error",
        "row": row,
        "col": col,
        "phase": phase,
        "output": output,
        "metrics": metrics or {},
        "error": error,
    }
    try:
        socketio.emit("sample_update", payload, namespace="/")
    except Exception as exc:
        logger.error("sample_update emit failed for %s: %s", sample_id, exc)


def _column_prompt_options(base_options: Dict[str, Any], column: Dict[str, Any]) -> Dict[str, Any]:
    """Merge the run-wide prompt_options with a column's own custom-instruction
    preset (a file in Custom_Instructions/). A per-column preset overrides the
    run-wide `custom_instructions` (translation phase) and `refinement_instructions`
    (refine phase). Best-effort: an unsafe/missing/empty file is ignored, so the
    column falls back to the run-wide options.
    """
    opts = dict(base_options or {})
    filename = (column.get("custom_instruction_file") or "").strip()
    if not filename:
        return opts
    if not is_safe_filename(filename):
        logger.warning("sample: ignoring unsafe custom instruction filename %r", filename)
        return opts
    try:
        ci_dir = Path(os.getcwd()) / "Custom_Instructions"
        loaded = load_custom_instructions(filename, ci_dir)
        translation = loaded.get("translation")
        refinement = loaded.get("refinement")
        if translation:
            opts["custom_instructions"] = translation
        if refinement:
            opts["refinement_instructions"] = refinement
    except Exception as exc:
        logger.warning("sample: failed to load custom instructions %r: %s", filename, exc)
    return opts


def _load_column_glossary(glossary_id: Any) -> Optional[Dict[str, Any]]:
    """Load a glossary by id into {terms_dict, term_metadata, target_language},
    or None when there's no/invalid glossary. Best-effort: any failure (missing
    id, store error, empty glossary) yields None so the column runs without one.
    """
    if not glossary_id:
        return None
    try:
        from src.api.translation_state import get_glossary_store  # lazy: avoid cycle
        glossary = get_glossary_store().get_glossary(int(glossary_id))
    except Exception as exc:
        logger.warning("sample: failed to load glossary %r: %s", glossary_id, exc)
        return None
    if not glossary or not glossary.terms:
        return None
    return {
        "terms_dict": glossary.terms_dict,
        "term_metadata": {
            term.source_term: {"category": term.category or ""}
            for term in glossary.terms if term.source_term
        },
        "target_language": glossary.target_language or "",
    }


def _glossary_block_for(glossary_data: Optional[Dict[str, Any]], text: str) -> str:
    """Build the per-cell glossary block: filter the glossary to terms present
    in this cell's source text, then format. Returns '' when nothing matches."""
    if not glossary_data:
        return ""
    try:
        filtered, _capped = filter_glossary(text, glossary_data["terms_dict"], GlossaryConfig())
        if not filtered:
            return ""
        return build_glossary_block(
            filtered_terms=filtered,
            target_language=glossary_data["target_language"],
            term_metadata=glossary_data["term_metadata"],
        )
    except Exception as exc:
        logger.warning("sample: failed to build glossary block: %s", exc)
        return ""


async def _run_sample_async(
    *,
    sample_id: str,
    items: List[Dict[str, Any]],
    columns: List[Dict[str, Any]],
    mode: str,
    source_language: str,
    target_language: str,
    prompt_options: Dict[str, Any],
    state: "SampleStateManager",
    socketio,
    skip_cells: Optional[set] = None,
) -> None:
    """Run all N×K cells in parallel under a concurrency semaphore.

    `skip_cells` is a set of (row, col) tuples whose LLM calls must be skipped
    — used by the cross-run cache: when the client already has a cached result
    for that cell, we avoid spending tokens on it.
    """
    skip = skip_cells or set()
    sem = asyncio.Semaphore(min(SAMPLE_CONCURRENCY_CAP, max(1, len(items) * len(columns))))

    # Resolve each column's custom-instruction preset and glossary once, so every
    # cell of that column reuses them (the glossary block is still filtered per
    # cell against that cell's source text).
    column_options = [_column_prompt_options(prompt_options, c) for c in columns]
    column_glossaries = [_load_column_glossary(c.get("glossary_id")) for c in columns]

    async def cell_task(row: int, col: int):
        async with sem:
            if state.is_cancelled(sample_id):
                return
            if (row, col) in skip:
                return
            item = items[row]
            column = columns[col]
            cell_options = column_options[col]
            glossary_block = _glossary_block_for(column_glossaries[col], item["source_text"])
            if mode == "refine":
                # Treat the source extract as the draft to refine.
                await _run_cell_refine(
                    sample_id=sample_id, row=row, col=col,
                    draft_text=item["source_text"], item=item, column=column,
                    target_language=target_language,
                    prompt_options=cell_options,
                    glossary_block=glossary_block,
                    state=state, socketio=socketio,
                )
                return

            draft = await _run_cell_translate(
                sample_id=sample_id, row=row, col=col,
                item=item, column=column,
                source_language=source_language, target_language=target_language,
                prompt_options=cell_options,
                glossary_block=glossary_block,
                state=state, socketio=socketio,
            )

            if mode == "translate_refine" and draft and not state.is_cancelled(sample_id):
                await _run_cell_refine(
                    sample_id=sample_id, row=row, col=col,
                    draft_text=draft, item=item, column=column,
                    target_language=target_language,
                    prompt_options=cell_options,
                    glossary_block=glossary_block,
                    state=state, socketio=socketio,
                )

    await asyncio.gather(
        *(cell_task(r, c) for r in range(len(items)) for c in range(len(columns))),
        return_exceptions=True,
    )

    final_status = "stopped" if state.is_cancelled(sample_id) else "completed"
    state.set_status(sample_id, final_status)
    if socketio is not None:
        try:
            socketio.emit(
                "sample_update",
                {
                    "sample_id": sample_id,
                    "type": "sample_stopped" if final_status == "stopped" else "sample_done",
                },
                namespace="/",
            )
        except Exception as exc:
            logger.error("sample_update final emit failed for %s: %s", sample_id, exc)


def _spawn_sample_thread(coro_factory):
    """Run an async coroutine in a fresh thread with its own event loop."""
    def runner():
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(coro_factory())
        finally:
            # Finalize any async generators still open (e.g. the httpx streaming
            # response behind `async for line in response.aiter_lines()` in the
            # LLM providers) before closing the loop. Without this, their cleanup
            # tasks are destroyed mid-flight and asyncio logs
            # "Task was destroyed but it is pending!". Mirrors glossary_routes.
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()
    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    return thread


def create_sample_blueprint(sample_state_manager, socketio=None, output_dir="."):
    """Create the sample blueprint.

    Args:
        sample_state_manager: Instance of SampleStateManager.
        socketio: SocketIO instance, used to emit `sample_update` events.
        output_dir: Base directory for file operations; uploaded source files
            live in '<output_dir>/uploads' and a client-supplied file_path must
            resolve inside it.
    """
    bp = Blueprint("sample", __name__)

    uploads_dir = Path(output_dir) / "uploads"

    def _validate_file(data: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[Tuple[Any, int]]]:
        """Common file_path + file_type validation. Returns (path, type, err).

        The path is confined to the uploads directory: these endpoints return
        the extracted source text directly in the JSON response, so an
        unvalidated client path would be a one-step arbitrary-file-read
        exfiltration primitive. See issue #209.
        """
        # validate_upload_path checks containment before existence, so an
        # out-of-bounds path yields 403 (not 404) and never leaks whether it
        # exists.
        safe_path, path_error = PathValidator.validate_upload_path(
            data.get("file_path"), uploads_dir
        )
        if path_error is not None:
            status = 400 if path_error.startswith("Missing") else 403
            return None, None, (jsonify({"error": path_error}), status)
        file_path = str(safe_path)
        try:
            detected = detect_file_type(file_path)
        except Exception as exc:
            return None, None, (jsonify({"error": f"Cannot detect file type: {exc}"}), 400)
        file_type = (data.get("file_type") or detected).lower()
        if file_type != detected:
            return None, None, (jsonify({
                "error": f"File type mismatch: client said {file_type!r}, server detected {detected!r}",
            }), 400)
        return file_path, file_type, None

    @bp.route("/api/sample/initialize", methods=["POST"])
    def initialize_samples():
        """Sample N initial extracts from a freshly uploaded file.

        Called by the client right after upload so the user can preview the
        selected blocks before spending any LLM tokens. Returns items with the
        same shape /api/sample/run produces, but without creating a sample_id
        and without spawning any background work.
        """
        data = request.get_json(silent=True) or {}
        file_path, file_type, err = _validate_file(data)
        if err is not None:
            return err

        try:
            n_samples = _clamp_int(data.get("n_samples"), DEFAULT_N_SAMPLES, 2, 20)
            max_chars = _clamp_int(data.get("max_chars"), DEFAULT_MAX_CHARS, 50, 2000)
        except (TypeError, ValueError):
            return jsonify({"error": "n_samples and max_chars must be integers"}), 400

        try:
            units = _load_source_units(file_path, file_type)
            total = len(units)
            if total < 3:
                return jsonify({"error": "document too small for sampling"}), 400
            indices = select_sample_indices(total, n_samples)
            items = _items_for_indices(units, indices, max_chars)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": f"Failed to initialize samples: {exc}"}), 500

        warnings: List[Dict[str, Any]] = []
        if len(indices) < n_samples:
            warnings.append(_small_document_warning(total, len(indices), n_samples))

        public_items = [
            {"index": it["index"], "source_text": it["source_text"], "truncated": it["truncated"]}
            for it in items
        ]
        return jsonify({"items": public_items, "total": total, "warnings": warnings})

    @bp.route("/api/sample/extract", methods=["POST"])
    def extract_random_sample():
        """Pick a random extract not in `exclude_indices` (server-side RNG).

        Used by the "Add a sample" button to grow the user's curated sample
        list. Returns 409 when the document has no remaining interior index.
        """
        data = request.get_json(silent=True) or {}
        file_path, file_type, err = _validate_file(data)
        if err is not None:
            return err

        try:
            max_chars = _clamp_int(data.get("max_chars"), DEFAULT_MAX_CHARS, 50, 2000)
        except (TypeError, ValueError):
            return jsonify({"error": "max_chars must be an integer"}), 400

        raw_excl = data.get("exclude_indices") or []
        exclude: Set[int] = set()
        for v in raw_excl:
            try:
                exclude.add(int(v))
            except (TypeError, ValueError):
                continue

        try:
            units = _load_source_units(file_path, file_type)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": f"Failed to load source: {exc}"}), 500

        total = len(units)
        if total < 3:
            return jsonify({"error": "document too small for sampling"}), 400

        idx = _pick_random_unused_index(total, exclude)
        if idx is None:
            return jsonify({"error": "no_more_indices", "total": total}), 409

        items = _items_for_indices(units, [idx], max_chars)
        if not items:
            return jsonify({"error": "failed to build item"}), 500
        it = items[0]
        return jsonify({
            "item": {
                "index": it["index"],
                "source_text": it["source_text"],
                "truncated": it["truncated"],
            },
            "total": total,
        })

    @bp.route("/api/sample/run", methods=["POST"])
    def start_sample_run():
        data = request.get_json(silent=True) or {}

        # Required fields. `source_language` may be empty: we auto-detect it
        # from the uploaded file's content (mirrors the Translate-tab behavior).
        for field in ("file_path", "file_type", "target_language", "columns"):
            if field not in data or data[field] in (None, "", []):
                return jsonify({"error": f"Missing or empty field: {field}"}), 400

        # File existence + type detection/mismatch (shared with initialize/extract).
        file_path, file_type, err = _validate_file(data)
        if err is not None:
            return err

        mode = (data.get("mode") or "translate").lower()
        if mode not in ("translate", "refine", "translate_refine"):
            return jsonify({"error": f"Invalid mode: {mode}"}), 400

        try:
            n_samples = _clamp_int(data.get("n_samples"), DEFAULT_N_SAMPLES, 2, 20)
            max_chars = _clamp_int(data.get("max_chars"), DEFAULT_MAX_CHARS, 50, 2000)
        except (TypeError, ValueError):
            return jsonify({"error": "n_samples and max_chars must be integers"}), 400

        columns_raw = data["columns"]
        if not isinstance(columns_raw, list) or len(columns_raw) < 1:
            return jsonify({"error": "columns must be a non-empty list"}), 400

        # Build sample items.
        #
        # Two paths:
        #  - `items` provided by the client → user already curated the sample
        #    set (initialize + add/remove). We honor the indices and the
        #    client-supplied source_text, but re-derive context_before/after
        #    server-side (deterministic given the file).
        #  - `items` missing → fall back to the legacy auto-sampling path.
        warnings: List[Dict[str, Any]] = []
        client_items = data.get("items")
        try:
            if client_items is not None:
                if not isinstance(client_items, list) or not client_items:
                    return jsonify({"error": "items must be a non-empty list"}), 400
                units = _load_source_units(file_path, file_type)
                items = []
                total = len(units)
                for raw in client_items:
                    if not isinstance(raw, dict):
                        continue
                    try:
                        idx = int(raw.get("index"))
                    except (TypeError, ValueError):
                        continue
                    if idx < 0 or idx >= total:
                        continue
                    source_text = raw.get("source_text")
                    if not isinstance(source_text, str) or not source_text.strip():
                        continue
                    items.append({
                        "index": idx,
                        "source_text": source_text,
                        "truncated": bool(raw.get("truncated")),
                        "context_before": units[idx].get("context_before", ""),
                        "context_after": units[idx].get("context_after", ""),
                    })
                if not items:
                    return jsonify({"error": "no valid items provided"}), 400
            elif file_type == "srt":
                items, warnings = _build_srt_sample_blocks(file_path, n_samples, max_chars)
            else:
                text = _extract_plain_text(file_path, file_type)
                if not text or not text.strip():
                    return jsonify({"error": "File is empty or unreadable"}), 400
                items, warnings = _build_text_sample_items(text, n_samples, max_chars)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": f"Failed to prepare samples: {exc}"}), 500

        # Normalize columns and create state entry
        columns = []
        for raw in columns_raw:
            columns.append({
                "provider": (raw.get("provider") or "ollama").lower(),
                "model": raw.get("model") or "",
                "api_key": raw.get("api_key"),
                "api_endpoint": raw.get("api_endpoint") or raw.get("endpoint"),
                "custom_instruction_file": raw.get("custom_instruction_file") or "",
                "glossary_id": raw.get("glossary_id") or None,
            })

        sample_id = f"sample_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
        sample_state_manager.create(sample_id, items, columns, mode)

        # Items exposed to the client must not leak context_before/context_after
        # — those are kept server-side only and used to enrich prompts.
        public_items = [
            {"index": it["index"], "source_text": it["source_text"], "truncated": it["truncated"]}
            for it in items
        ]
        public_columns = [
            {k: v for k, v in col.items() if k != "api_key"}
            for col in columns
        ]

        source_language = (data.get("source_language") or "").strip()
        target_language = data["target_language"]
        # Auto-detect source language from file content when the user leaves
        # the picker on "Auto-detect" (same UX as the Translate tab).
        if not source_language:
            try:
                with open(file_path, "rb") as fh:
                    file_bytes = fh.read()
                detected_name, confidence = LanguageDetector.detect_language_from_file(
                    file_bytes, os.path.basename(file_path)
                )
                if detected_name:
                    source_language = detected_name
                    warnings.append({
                        "code": "warning_lang_autodetected",
                        "params": {"lang": detected_name, "confidence": round(confidence * 100)},
                    })
            except Exception as exc:
                logger.warning("sample: language auto-detection failed: %s", exc)
        if not source_language:
            return jsonify({
                "error": "Could not auto-detect source language; please pick one manually.",
            }), 400

        if mode == "refine":
            target_language = source_language
        prompt_options = data.get("prompt_options") or {}

        # `defer_dispatch=true` lets the client read the items first, compute
        # which cells are already cached, then call /dispatch with skip_cells.
        defer_dispatch = bool(data.get("defer_dispatch"))

        # Stash the run parameters so /dispatch can pick them up. Pending state
        # entries are already created by sample_state_manager.create().
        sample_state_manager.set_run_context(sample_id, {
            "items": items,
            "columns": columns,
            "mode": mode,
            "source_language": source_language,
            "target_language": target_language,
            "prompt_options": prompt_options,
        })

        if not defer_dispatch:
            async def _runner():
                await _run_sample_async(
                    sample_id=sample_id,
                    items=items,
                    columns=columns,
                    mode=mode,
                    source_language=source_language,
                    target_language=target_language,
                    prompt_options=prompt_options,
                    state=sample_state_manager,
                    socketio=socketio,
                )

            _spawn_sample_thread(_runner)

        return jsonify({
            "sample_id": sample_id,
            "items": public_items,
            "columns": public_columns,
            "mode": mode,
            "warnings": warnings,
            "deferred": defer_dispatch,
        })

    @bp.route("/api/sample/<sample_id>/dispatch", methods=["POST"])
    def dispatch_sample_run(sample_id):
        """Start the LLM work for a previously prepared (deferred) run.

        Body: { skip_cells: [[row, col], ...] }. Cells in skip_cells are not
        sent to the LLM — the client already has them cached from an earlier
        run with identical parameters.
        """
        if not sample_state_manager.exists(sample_id):
            return jsonify({"error": "Sample run not found"}), 404

        run_ctx = sample_state_manager.get_run_context(sample_id)
        if run_ctx is None:
            return jsonify({"error": "Sample run has no pending dispatch context"}), 409

        payload = request.get_json(silent=True) or {}
        raw_skip = payload.get("skip_cells") or []
        skip: set = set()
        for pair in raw_skip:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                try:
                    skip.add((int(pair[0]), int(pair[1])))
                except (TypeError, ValueError):
                    continue

        async def _runner():
            await _run_sample_async(
                sample_id=sample_id,
                items=run_ctx["items"],
                columns=run_ctx["columns"],
                mode=run_ctx["mode"],
                source_language=run_ctx["source_language"],
                target_language=run_ctx["target_language"],
                prompt_options=run_ctx["prompt_options"],
                state=sample_state_manager,
                socketio=socketio,
                skip_cells=skip,
            )

        _spawn_sample_thread(_runner)
        return jsonify({"message": "Dispatch started", "skipped": len(skip)}), 200

    @bp.route("/api/sample/<sample_id>/stop", methods=["POST"])
    def stop_sample_run(sample_id):
        if not sample_state_manager.exists(sample_id):
            return jsonify({"error": "Sample run not found"}), 404
        sample_state_manager.cancel(sample_id)
        return jsonify({"message": "Sample run stopped"}), 200

    @bp.route("/api/sample/<sample_id>", methods=["GET"])
    def get_sample_run(sample_id):
        snapshot = sample_state_manager.get(sample_id)
        if snapshot is None:
            return jsonify({"error": "Sample run not found"}), 404
        # Strip API keys defensively before returning
        snapshot["columns"] = [
            {k: v for k, v in col.items() if k != "api_key"}
            for col in snapshot.get("columns", [])
        ]
        # Strip server-only context fields from items
        snapshot["items"] = [
            {"index": it["index"], "source_text": it["source_text"], "truncated": it["truncated"]}
            for it in snapshot.get("items", [])
        ]
        return jsonify(snapshot)

    return bp
