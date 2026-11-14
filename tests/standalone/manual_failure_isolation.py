"""
T10 - Failure isolation in parallel mode.

Patches src.core.translator.generate_translation_request so that the
target chunk (chunk_index = 2) always returns None, simulating a chunk
whose translation could not be produced after retries. All other chunks
proceed normally.

Verifies that:
  * The other chunks of the batch finish (no cross-segment cancellation).
  * failed_chunks reports exactly 1.
  * The output file exists (the failed chunk falls back to original text).

Run from repo root:
    python tests/standalone/manual_failure_isolation.py
"""

import asyncio
import os
import sys
import tempfile
import uuid
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from src import config  # noqa: F401
import src.core.translator as translator_module
from src.core.adapters import translate_file
from src.persistence.checkpoint_manager import CheckpointManager
from tests.job_helpers import make_ctx, make_job


FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "sample.txt"
PARALLEL = 3
FAILING_CHUNK_INDEX = 2


def _install_failing_patch(original, fixture_chunk_text_for_fail):
    """
    Wrap generate_translation_request so that calls whose main_content matches
    the content we want to flag as failing return None instead of a translation.
    Other calls go through the real implementation.
    """
    async def wrapped(main_content, *args, **kwargs):
        if main_content == fixture_chunk_text_for_fail:
            return None
        return await original(main_content, *args, **kwargs)
    translator_module.generate_translation_request = wrapped


def _restore_patch(original):
    translator_module.generate_translation_request = original


async def run() -> int:
    api_key = os.getenv("POE_API_KEY", "").strip()
    if not api_key:
        print("FAIL: POE_API_KEY is not set in .env")
        return 1
    model = os.getenv("POE_MODEL", "Claude-Sonnet-4").strip() or "Claude-Sonnet-4"

    if not FIXTURE.exists():
        print(f"FAIL: fixture not found at {FIXTURE}")
        return 1

    # Determine which chunk's text we will force to fail. We use the same
    # chunker the production code uses so the fail target matches a real
    # chunk boundary.
    fixture_text = FIXTURE.read_text(encoding="utf-8")
    from src.core.text_processor import split_text_into_chunks
    chunks = split_text_into_chunks(fixture_text, max_tokens_per_chunk=120)
    if len(chunks) <= FAILING_CHUNK_INDEX:
        print(f"FAIL: fixture too small, only {len(chunks)} chunks")
        return 1
    failing_text = chunks[FAILING_CHUNK_INDEX]['main_content']
    print(f"Fixture:    {FIXTURE.name} ({len(chunks)} chunks)")
    print(f"Provider:   poe / {model}")
    print(f"Parallel:   parallel_requests={PARALLEL}")
    print(f"Inject:     chunk_index={FAILING_CHUNK_INDEX} will return None")

    last_stats: dict = {}

    def stats_callback(stats: dict) -> None:
        last_stats.update(stats)

    original_fn = translator_module.generate_translation_request
    _install_failing_patch(original_fn, failing_text)

    try:
        with tempfile.TemporaryDirectory(
            prefix="failure_isolation_", ignore_cleanup_errors=True
        ) as tmpdir:
            tmpdir_path = Path(tmpdir)
            cwd_before = os.getcwd()
            os.chdir(tmpdir_path)
            try:
                output_path = tmpdir_path / "sample.fr.txt"
                checkpoint_manager = CheckpointManager()
                translation_id = f"t10_{uuid.uuid4().hex[:8]}"

                ok = await translate_file(
                    make_job(
                        input_path=str(FIXTURE),
                        output_path=str(output_path),
                        source_language="English",
                        target_language="French",
                        model=model,
                        provider="poe",
                        api_key=api_key,
                        max_tokens_per_chunk=120,
                        parallel_workers=PARALLEL,
                    ),
                    make_ctx(
                        checkpoint_manager=checkpoint_manager,
                        translation_id=translation_id,
                        stats_callback=stats_callback,
                    ),
                )

                print(f"Result:     ok={ok}")
                print(f"Stats:      {last_stats}")

                failures = []

                total = last_stats.get("total_chunks", 0)
                completed = last_stats.get("completed_chunks", 0)
                failed = last_stats.get("failed_chunks", 0)

                if failed != 1:
                    failures.append(f"expected failed_chunks=1, got {failed}")
                if completed != total - 1:
                    failures.append(
                        f"expected completed_chunks={total - 1}, got {completed}"
                    )

                if not output_path.exists():
                    failures.append(f"output file missing: {output_path}")
                else:
                    translated = output_path.read_text(encoding="utf-8")
                    if not translated.strip():
                        failures.append("output file is empty")
                    else:
                        # The failed chunk falls back to original text, so the
                        # original chunk text should appear verbatim somewhere
                        # in the output.
                        if failing_text.strip() not in translated:
                            failures.append(
                                "failed chunk's original text was not present in output "
                                "(expected fallback to source)"
                            )

                if failures:
                    print("\nFAIL:")
                    for msg in failures:
                        print(f"  - {msg}")
                    return 1

                print("\nOK: all invariants passed")
                return 0
            finally:
                os.chdir(cwd_before)
    finally:
        _restore_patch(original_fn)


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
