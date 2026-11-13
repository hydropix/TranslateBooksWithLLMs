"""
T7 - EPUB parallel test with concurrency observer.

Builds a small 3-chapter EPUB and translates it with parallel_requests=2,
monkey-patching the Poe provider to count the maximum number of in-flight
LLM requests. Asserts the structural invariants from T3 plus that the
observed max concurrency is strictly greater than 1 (proves chapter-level
parallel dispatch is actually parallel).

Run from repo root:
    python tests/standalone/manual_epub_parallel.py
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

# Reuse the EPUB-builder helpers from the sequential baseline to keep the
# fixture identical across tests.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from manual_epub_baseline import (
    _build_minimal_epub,
    _list_epub_xhtml,
    _read_epub_chapter_text,
)
from tests.job_helpers import make_ctx, make_job


PARALLEL = 2


class ConcurrencyObserver:
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

    last_stats: dict = {}

    def stats_callback(stats: dict) -> None:
        last_stats.update(stats)

    observer = ConcurrencyObserver()
    original_generate = _install_observer(observer)

    try:
        with tempfile.TemporaryDirectory(
            prefix="epub_parallel_", ignore_cleanup_errors=True
        ) as tmpdir:
            tmpdir_path = Path(tmpdir)
            cwd_before = os.getcwd()
            os.chdir(tmpdir_path)
            try:
                input_path = tmpdir_path / "voyage.epub"
                output_path = tmpdir_path / "voyage.fr.epub"
                _build_minimal_epub(input_path)

                source_chapters = _list_epub_xhtml(input_path)
                print(f"Fixture:    {input_path.name} ({input_path.stat().st_size} bytes, "
                      f"{len(source_chapters)} XHTML files)")
                print(f"Provider:   poe / {model}")
                print(f"Parallel:   parallel_requests={PARALLEL}")

                checkpoint_manager = CheckpointManager()
                translation_id = f"t7_{uuid.uuid4().hex[:8]}"

                started = time.perf_counter()
                ok = await translate_file(
                    make_job(
                        input_path=str(input_path),
                        output_path=str(output_path),
                        source_language="English",
                        target_language="French",
                        model=model,
                        provider="poe",
                        api_key=api_key,
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
                if completed != total:
                    failures.append(
                        f"completed_chunks ({completed}) != total_chunks ({total})"
                    )
                if failed != 0:
                    failures.append(f"failed_chunks={failed} (expected 0)")

                # Core claim of T7: dispatch was parallel.
                if observer.max_in_flight <= 1:
                    failures.append(
                        f"max in-flight requests was {observer.max_in_flight} "
                        f"(expected > 1)"
                    )

                if not output_path.exists():
                    failures.append(f"output EPUB missing: {output_path}")
                else:
                    out_chapters = _list_epub_xhtml(output_path)
                    print(f"Output:     {output_path.stat().st_size} bytes, "
                          f"{len(out_chapters)} XHTML files")
                    if out_chapters != source_chapters:
                        failures.append(
                            f"chapter list/order mismatch: "
                            f"source={source_chapters} output={out_chapters}"
                        )
                    else:
                        any_changed = False
                        for href in source_chapters:
                            src_xhtml = _read_epub_chapter_text(input_path, href)
                            out_xhtml = _read_epub_chapter_text(output_path, href)
                            src_p = src_xhtml.count("<p>")
                            out_p = out_xhtml.count("<p>")
                            if out_p != src_p:
                                failures.append(
                                    f"{href}: <p> count mismatch source={src_p} output={out_p}"
                                )
                            if src_xhtml != out_xhtml:
                                any_changed = True
                        if not any_changed:
                            failures.append("no chapter content changed (translation did not happen)")

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
