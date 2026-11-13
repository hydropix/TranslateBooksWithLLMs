"""
T6 - TXT parallel test with concurrency observer.

Runs the same TXT fixture as T2 but with parallel_requests=3, and monkey-
patches the Poe provider to count the maximum number of in-flight LLM
requests. Asserts that:
  * The structural invariants from T2 still hold.
  * The observed max concurrency is strictly greater than 1 (proves the
    dispatch is actually parallel, independent of wall-clock variance).

The wall-clock time is logged for information but not asserted, because
cloud-API latency variance makes wall-clock comparisons flaky.

Run from repo root:
    python tests/standalone/manual_txt_parallel.py
"""

import asyncio
import os
import sys
import tempfile
import time
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
from src.core.adapters import translate_file
from src.core.llm.providers.poe import PoeProvider
from src.persistence.checkpoint_manager import CheckpointManager
from tests.job_helpers import make_ctx, make_job


FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "sample.txt"
PARALLEL = 3


class ConcurrencyObserver:
    """Tracks the maximum number of in-flight LLM requests via an async context."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._in_flight = 0
        self.max_in_flight = 0
        self.total_calls = 0

    async def __aenter__(self):
        async with self._lock:
            self._in_flight += 1
            self.total_calls += 1
            if self._in_flight > self.max_in_flight:
                self.max_in_flight = self._in_flight

    async def __aexit__(self, exc_type, exc, tb):
        async with self._lock:
            self._in_flight -= 1


def _install_observer(observer: ConcurrencyObserver):
    """Wrap PoeProvider.generate so every call is bracketed by the observer."""
    original = PoeProvider.generate

    async def wrapped(self, *args, **kwargs):
        async with observer:
            return await original(self, *args, **kwargs)

    PoeProvider.generate = wrapped
    return original


def _uninstall_observer(original):
    PoeProvider.generate = original


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
    print(f"Parallel:   parallel_requests={PARALLEL}")

    last_stats: dict = {}

    def stats_callback(stats: dict) -> None:
        last_stats.update(stats)

    observer = ConcurrencyObserver()
    original_generate = _install_observer(observer)

    try:
        with tempfile.TemporaryDirectory(
            prefix="txt_parallel_", ignore_cleanup_errors=True
        ) as tmpdir:
            tmpdir_path = Path(tmpdir)
            cwd_before = os.getcwd()
            os.chdir(tmpdir_path)
            try:
                output_path = tmpdir_path / "sample.fr.txt"
                checkpoint_manager = CheckpointManager()
                translation_id = f"t6_{uuid.uuid4().hex[:8]}"

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
                        parallel_workers=PARALLEL,
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
                print(f"Concurrency: total_calls={observer.total_calls}, "
                      f"max_in_flight={observer.max_in_flight}")

                failures = []

                if not ok:
                    failures.append("translate_file returned False")

                total = last_stats.get("total_chunks", 0)
                completed = last_stats.get("completed_chunks", 0)
                failed = last_stats.get("failed_chunks", 0)
                if total < 2:
                    failures.append(f"expected at least 2 chunks, got total_chunks={total}")
                if completed != total:
                    failures.append(
                        f"completed_chunks ({completed}) != total_chunks ({total})"
                    )
                if failed != 0:
                    failures.append(f"failed_chunks={failed} (expected 0)")

                # Core claim of T6: dispatch was parallel.
                if observer.max_in_flight <= 1:
                    failures.append(
                        f"max in-flight requests was {observer.max_in_flight} "
                        f"(expected > 1: dispatch did not run in parallel)"
                    )

                if not output_path.exists():
                    failures.append(f"output file missing: {output_path}")
                else:
                    translated = output_path.read_text(encoding="utf-8")
                    if not translated.strip():
                        failures.append("output file is empty")
                    elif translated.strip() == source_text.strip():
                        failures.append(
                            "output is byte-identical to source (translation did not happen)"
                        )
                    else:
                        ratio = len(translated) / max(len(source_text), 1)
                        print(f"Output:     {len(translated)} chars (ratio={ratio:.2f})")
                        if not (0.5 <= ratio <= 2.0):
                            failures.append(
                                f"output length ratio out of range: {ratio:.2f}"
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
        _uninstall_observer(original_generate)


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
