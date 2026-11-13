"""
Diagnostic: record every progress callback emitted during a parallel EPUB
translation and check that completed_chunks never decreases.

Run from repo root:
    python tests/standalone/manual_epub_progress_monotonic.py
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config  # noqa: F401
from src.core.adapters import translate_file
from src.persistence.checkpoint_manager import CheckpointManager
from manual_epub_resume_parallel import _build_chunky_epub
from tests.job_helpers import make_ctx, make_job


PARALLEL = 3


async def run() -> int:
    api_key = os.getenv("POE_API_KEY", "").strip()
    if not api_key:
        print("FAIL: POE_API_KEY is not set in .env")
        return 1
    model = os.getenv("POE_MODEL", "Claude-Sonnet-4").strip() or "Claude-Sonnet-4"

    samples = []  # list of (completed, total)

    def stats_callback(stats):
        completed = stats.get("completed_chunks", 0)
        total = stats.get("total_chunks", 0)
        samples.append((completed, total))

    with tempfile.TemporaryDirectory(
        prefix="epub_progress_", ignore_cleanup_errors=True
    ) as tmpdir:
        tmpdir_path = Path(tmpdir)
        cwd_before = os.getcwd()
        os.chdir(tmpdir_path)
        try:
            input_path = tmpdir_path / "book.epub"
            output_path = tmpdir_path / "book.fr.epub"
            _build_chunky_epub(input_path)

            checkpoint_manager = CheckpointManager()
            translation_id = f"prog_{uuid.uuid4().hex[:8]}"

            await translate_file(
                make_job(
                    input_path=str(input_path),
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

            print(f"Total callbacks: {len(samples)}")
            print(f"Final: {samples[-1] if samples else None}")
            print()
            print("All values:")
            for i, (c, t) in enumerate(samples):
                print(f"  [{i:03d}] completed={c} / total={t}")

            # Check monotonicity
            regressions = []
            max_seen = -1
            for i, (c, _) in enumerate(samples):
                if c < max_seen:
                    regressions.append((i, max_seen, c))
                max_seen = max(max_seen, c)

            if regressions:
                print(f"\nREGRESSIONS ({len(regressions)}):")
                for i, prev, now in regressions:
                    print(f"  index {i}: was {prev}, dropped to {now}")
                return 1
            print("\nOK: progress is monotonic")
            return 0
        finally:
            os.chdir(cwd_before)


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
