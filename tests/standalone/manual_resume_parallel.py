"""
T9 - Resume after a parallel-mode interruption.

Runs a TXT translation twice with the same translation_id, parallel_requests=3:
  Pass 1: the interruption callback fires once 3 chunks have completed,
          but multiple chunks may have been in flight at that moment.
  Pass 2: resume to completion (still parallel_requests=3).

Verifies that:
  * Pass 2 does not retranslate already-completed chunks (LLM call count
    on pass 2 equals total_chunks - chunks_done_in_pass_1).
  * The final output passes the same structural invariants as the
    sequential baseline.

Run from repo root:
    python tests/standalone/manual_resume_parallel.py
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
INTERRUPT_AFTER = 3  # chunks


class CallCounter:
    """Counts the number of times PoeProvider.generate is invoked."""

    def __init__(self) -> None:
        self.total = 0

    async def __aenter__(self):
        self.total += 1

    async def __aexit__(self, exc_type, exc, tb):
        pass


def _install_counter(counter: CallCounter):
    original = PoeProvider.generate

    async def wrapped(self, *args, **kwargs):
        async with counter:
            return await original(self, *args, **kwargs)

    PoeProvider.generate = wrapped
    return original


def _uninstall_counter(original):
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

    print(f"Fixture:    {FIXTURE.name}")
    print(f"Provider:   poe / {model}")
    print(f"Parallel:   parallel_requests={PARALLEL}")
    print(f"Interrupt:  after {INTERRUPT_AFTER} chunks have completed")

    pass1_stats: dict = {}
    pass2_stats: dict = {}

    counter_pass1 = CallCounter()
    counter_pass2 = CallCounter()

    def make_stats_cb(target):
        def _cb(stats):
            target.update(stats)
        return _cb

    def make_interrupt_cb():
        # Pass 1 only: fire when we've seen INTERRUPT_AFTER completed chunks.
        # Other in-flight chunks at that moment may still finish (matches the
        # documented "let other tasks finish their current unit" behavior).
        def _cb():
            return pass1_stats.get("completed_chunks", 0) >= INTERRUPT_AFTER
        return _cb

    with tempfile.TemporaryDirectory(
        prefix="resume_parallel_", ignore_cleanup_errors=True
    ) as tmpdir:
        tmpdir_path = Path(tmpdir)
        cwd_before = os.getcwd()
        os.chdir(tmpdir_path)
        try:
            output_path = tmpdir_path / "sample.fr.txt"
            translation_id = f"t9_{uuid.uuid4().hex[:8]}"

            # ---- Pass 1: parallel, interrupted ----
            print("\n[Pass 1] parallel translate, will interrupt after threshold")
            original1 = _install_counter(counter_pass1)
            try:
                ckpt_mgr_1 = CheckpointManager()
                started = time.perf_counter()
                ok1 = await translate_file(
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
                        checkpoint_manager=ckpt_mgr_1,
                        translation_id=translation_id,
                        stats_callback=make_stats_cb(pass1_stats),
                        check_interruption=make_interrupt_cb(),
                    ),
                )
                pass1_elapsed = time.perf_counter() - started
            finally:
                _uninstall_counter(original1)
            print(f"[Pass 1] ok={ok1} elapsed={pass1_elapsed:.1f}s stats={pass1_stats}")
            print(f"[Pass 1] LLM calls observed: {counter_pass1.total}")

            pass1_completed = pass1_stats.get("completed_chunks", 0)
            total_chunks = pass1_stats.get("total_chunks", 0)

            # ---- Pass 2: parallel, resume ----
            print("\n[Pass 2] parallel resume")
            original2 = _install_counter(counter_pass2)
            try:
                ckpt_mgr_2 = CheckpointManager()
                started = time.perf_counter()
                ok2 = await translate_file(
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
                        checkpoint_manager=ckpt_mgr_2,
                        translation_id=translation_id,
                        stats_callback=make_stats_cb(pass2_stats),
                        check_interruption=None,
                    ),
                )
                pass2_elapsed = time.perf_counter() - started
            finally:
                _uninstall_counter(original2)
            print(f"[Pass 2] ok={ok2} elapsed={pass2_elapsed:.1f}s stats={pass2_stats}")
            print(f"[Pass 2] LLM calls observed: {counter_pass2.total}")

            failures = []

            if ok1:
                failures.append("pass 1 should have been interrupted (ok=True returned)")
            if pass1_completed < INTERRUPT_AFTER:
                failures.append(
                    f"pass 1 completed only {pass1_completed} chunks before interrupt "
                    f"(expected >= {INTERRUPT_AFTER})"
                )

            if not ok2:
                failures.append("pass 2 (resume) returned False")
            if pass2_stats.get("completed_chunks", 0) != total_chunks:
                failures.append(
                    f"pass 2 completed_chunks ({pass2_stats.get('completed_chunks')}) "
                    f"!= total ({total_chunks})"
                )
            if pass2_stats.get("failed_chunks", 0) != 0:
                failures.append(
                    f"pass 2 failed_chunks={pass2_stats.get('failed_chunks')} (expected 0)"
                )

            # Strongest claim of T9: pass 2 retranslates exactly the chunks
            # that pass 1 did not complete.
            expected_pass2_calls = total_chunks - pass1_completed
            if counter_pass2.total != expected_pass2_calls:
                failures.append(
                    f"pass 2 issued {counter_pass2.total} LLM calls, "
                    f"expected exactly {expected_pass2_calls} "
                    f"(total={total_chunks}, done_in_pass1={pass1_completed})"
                )

            if not output_path.exists():
                failures.append(f"output file missing: {output_path}")
            else:
                translated = output_path.read_text(encoding="utf-8")
                source_text = FIXTURE.read_text(encoding="utf-8")
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
                        failures.append(f"output length ratio out of range: {ratio:.2f}")

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
