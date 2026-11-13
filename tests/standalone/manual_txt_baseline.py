"""
T2 - TXT sequential baseline test.

Translates `tests/fixtures/sample.txt` end-to-end through the public
`translate_file()` entry point against Poe (POE_API_KEY from .env) and
checks structural invariants. Used as the reference behavior before the
parallel-requests refactor (GitHub issue #175): the same assertions must
still pass after the refactor when parallel_requests=1.

Run from repo root:
    python tests/standalone/manual_txt_baseline.py
"""

import asyncio
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

# Force UTF-8 on stdout/stderr so library logs containing emoji don't crash
# on Windows consoles that default to cp1252.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from src import config  # noqa: F401  (load_dotenv side effect)
from src.core.adapters import translate_file
from src.persistence.checkpoint_manager import CheckpointManager
from tests.job_helpers import make_ctx, make_job


FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "sample.txt"


async def run() -> int:
    api_key = os.getenv("POE_API_KEY", "").strip()
    if not api_key:
        print("FAIL: POE_API_KEY is not set in .env")
        return 1
    model = os.getenv("POE_MODEL", "Claude-Sonnet-4").strip() or "Claude-Sonnet-4"

    if not FIXTURE.exists():
        print(f"FAIL: fixture not found at {FIXTURE}")
        return 1

    source_text = FIXTURE.read_text(encoding="utf-8")
    print(f"Fixture:    {FIXTURE.name} ({len(source_text)} chars)")
    print(f"Provider:   poe / {model}")

    # Capture the last stats payload emitted by the translator.
    last_stats: dict = {}

    def stats_callback(stats: dict) -> None:
        last_stats.update(stats)

    # ignore_cleanup_errors: on Windows, the SQLite handle on data/jobs.db
    # is not released synchronously when CheckpointManager goes out of scope,
    # so the tmpdir teardown can race with an open file handle. The test result
    # is decided before cleanup, so swallowing the cleanup error is safe.
    with tempfile.TemporaryDirectory(
        prefix="txt_baseline_", ignore_cleanup_errors=True
    ) as tmpdir:
        tmpdir_path = Path(tmpdir)
        # CheckpointManager writes under the cwd ("data/jobs.db", "data/uploads/").
        # Run from a per-test directory so we don't pollute the repo.
        cwd_before = os.getcwd()
        os.chdir(tmpdir_path)
        try:
            output_path = tmpdir_path / "sample.fr.txt"
            checkpoint_manager = CheckpointManager()
            translation_id = f"t2_{uuid.uuid4().hex[:8]}"

            started = time.perf_counter()
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
                ),
                make_ctx(
                    checkpoint_manager=checkpoint_manager,
                    translation_id=translation_id,
                    stats_callback=stats_callback,
                ),
            )
            elapsed_s = time.perf_counter() - started

            print(f"Result:     ok={ok} elapsed={elapsed_s:.1f}s")
            print(f"Stats:      {last_stats}")

            failures = []

            if not ok:
                failures.append("translate_file returned False")

            total = last_stats.get("total_chunks", 0)
            completed = last_stats.get("completed_chunks", 0)
            failed = last_stats.get("failed_chunks", 0)

            if total < 2:
                failures.append(f"expected at least 2 chunks, got total_chunks={total}")
            if completed != total:
                failures.append(f"completed_chunks ({completed}) != total_chunks ({total})")
            if failed != 0:
                failures.append(f"failed_chunks={failed} (expected 0)")

            if not output_path.exists():
                failures.append(f"output file missing: {output_path}")
            else:
                translated = output_path.read_text(encoding="utf-8")
                if not translated.strip():
                    failures.append("output file is empty")
                else:
                    print(f"Output:     {len(translated)} chars")
                    if translated.strip() == source_text.strip():
                        failures.append("output is byte-identical to source (translation did not happen)")
                    # Sanity check: French translation of an English text of this length
                    # should be within a 0.5x-2.0x ratio of the source length.
                    ratio = len(translated) / max(len(source_text), 1)
                    if not (0.5 <= ratio <= 2.0):
                        failures.append(
                            f"output length ratio out of range: {ratio:.2f} "
                            f"(source={len(source_text)}, output={len(translated)})"
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


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
