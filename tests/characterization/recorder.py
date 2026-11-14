"""Run a translation/refine pass and record a stable progress fingerprint.

The recorder strips every time-, cost- and machine-dependent field from each
stats callback so the captured sequence is reproducible. The produced output
file is fingerprinted in a zip-timestamp-independent way (member content
hashes) so EPUB/DOCX outputs are comparable across runs.
"""

import asyncio
import hashlib
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

# EPUB/DOCX rewrite a `dcterms:modified` (or core.xml) ISO-8601 timestamp on
# save. Neutralize it before hashing so the fingerprint reflects content only.
_ISO_TS = re.compile(rb"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?")


def _hash_member(data: bytes) -> str:
    return hashlib.sha256(_ISO_TS.sub(b"<TS>", data)).hexdigest()

from src.core.adapters import translate_file, refine_file
from src.persistence.checkpoint_manager import CheckpointManager
from tests.job_helpers import make_ctx, make_job

# Only these stats fields are kept in the golden snapshot. Everything else
# (start_time, elapsed_time, openrouter_*, ETA, token rates) is time/cost
# dependent and would make the baseline flaky.
STABLE_KEYS = (
    "total_chunks",
    "completed_chunks",
    "failed_chunks",
    "processed_chunks",
    "successful_first_try",
    "successful_after_retry",
    "fallback_used",
    "token_alignment_used",
    "placeholder_errors",
    "total_tokens",
)


def _project(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the stable keys; round floats for deterministic comparison."""
    out: Dict[str, Any] = {}
    for key in STABLE_KEYS:
        if key not in stats:
            continue
        value = stats[key]
        if isinstance(value, float):
            value = round(value, 2)
        out[key] = value
    return out


def fingerprint_output(path: Path) -> Dict[str, Any]:
    """Return a deterministic fingerprint of the output file.

    For zip-based formats (EPUB/DOCX) we hash each member's *content* (not the
    archive), which is stable regardless of the zip's embedded timestamps. For
    text formats we keep the decoded text so diffs are human-readable.
    """
    data = path.read_bytes()
    if zipfile.is_zipfile(path):
        members: Dict[str, str] = {}
        with zipfile.ZipFile(path) as zf:
            for name in sorted(zf.namelist()):
                members[name] = _hash_member(zf.read(name))
        return {"kind": "zip", "members": members}
    try:
        return {"kind": "text", "text": data.decode("utf-8")}
    except UnicodeDecodeError:
        return {"kind": "binary", "sha256": hashlib.sha256(data).hexdigest()}


async def _run(coro_factory, work_dir: Path):
    cwd_before = os.getcwd()
    os.chdir(work_dir)
    try:
        return await coro_factory()
    finally:
        os.chdir(cwd_before)


def record_translation(
    work_dir: Path,
    input_path: Path,
    output_path: Path,
    *,
    prompt_options: Dict[str, Any] | None = None,
    max_tokens_per_chunk: int = 60,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run translate_file with the echo provider and record the progress trace."""
    sequence: List[Dict[str, Any]] = []

    def stats_callback(stats: Dict[str, Any]):
        sequence.append(_project(stats))

    checkpoint_manager = CheckpointManager(db_path=str(work_dir / "jobs.db"))

    async def factory():
        await translate_file(
            make_job(
                str(input_path), str(output_path),
                provider="poe", model="fake-echo", api_key="UNUSED_FAKE_KEY",
                source_language="English", target_language="French",
                max_tokens_per_chunk=max_tokens_per_chunk,
                context_window=4096,
                auto_adjust_context=False,
                prompt_options=prompt_options or {},
            ),
            make_ctx(translation_id="char_translate",
                     checkpoint_manager=checkpoint_manager,
                     stats_callback=stats_callback),
        )

    asyncio.run(_run(factory, work_dir))
    return sequence, fingerprint_output(output_path)


def record_refine(
    work_dir: Path,
    input_path: Path,
    output_path: Path,
    *,
    prompt_options: Dict[str, Any] | None = None,
    max_tokens_per_chunk: int = 60,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run refine_file (refine-only) with the echo provider and record the trace."""
    sequence: List[Dict[str, Any]] = []

    def stats_callback(stats: Dict[str, Any]):
        sequence.append(_project(stats))

    checkpoint_manager = CheckpointManager(db_path=str(work_dir / "jobs.db"))

    async def factory():
        await refine_file(
            make_job(
                str(input_path), str(output_path), operation='refine',
                provider="poe", model="fake-echo", api_key="UNUSED_FAKE_KEY",
                source_language="French", target_language="French",
                max_tokens_per_chunk=max_tokens_per_chunk,
                context_window=4096,
                auto_adjust_context=False,
                prompt_options=prompt_options or {},
            ),
            make_ctx(translation_id="char_refine",
                     checkpoint_manager=checkpoint_manager,
                     stats_callback=stats_callback),
        )

    asyncio.run(_run(factory, work_dir))
    return sequence, fingerprint_output(output_path)
