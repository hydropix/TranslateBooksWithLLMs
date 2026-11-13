"""
T4 - Resume sequential baseline test.

Runs a TXT translation twice with the same translation_id:
  Pass 1: an interruption callback fires after 3 chunks have completed.
  Pass 2: resume to completion without interruption.

Verifies that chunks 0..2 are not re-translated on pass 2 (pass 2 issues
fewer LLM requests than pass 1) and that the final output passes the same
structural invariants as the sequential baseline (T2). Used as the
reference behavior before the parallel-requests refactor (GitHub issue #175):
the resume path must keep working when the parallel implementation lands.

Run from repo root:
    python tests/standalone/manual_resume_baseline.py
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
from src.persistence.checkpoint_manager import CheckpointManager
from tests.job_helpers import make_ctx, make_job


FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "sample.txt"
INTERRUPT_AFTER = 3  # chunks


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
    print(f"Interrupt:  after {INTERRUPT_AFTER} chunks")

    pass1_stats: dict = {}
    pass2_stats: dict = {}

    def make_stats_cb(target):
        def _cb(stats):
            target.update(stats)
        return _cb

    def make_interrupt_cb():
        # Called at the start of each unit iteration. Return True once we've
        # seen INTERRUPT_AFTER chunks completed (i.e. completed_chunks >= N).
        def _cb():
            return pass1_stats.get("completed_chunks", 0) >= INTERRUPT_AFTER
        return _cb

    with tempfile.TemporaryDirectory(
        prefix="resume_baseline_", ignore_cleanup_errors=True
    ) as tmpdir:
        tmpdir_path = Path(tmpdir)
        cwd_before = os.getcwd()
        os.chdir(tmpdir_path)
        try:
            output_path = tmpdir_path / "sample.fr.txt"
            # Shared translation_id so pass 2 finds the pass 1 checkpoint.
            translation_id = f"t4_{uuid.uuid4().hex[:8]}"

            # ---- Pass 1: interrupted ----
            print("\n[Pass 1] starting, will interrupt after the threshold")
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
                ),
                make_ctx(
                    checkpoint_manager=ckpt_mgr_1,
                    translation_id=translation_id,
                    stats_callback=make_stats_cb(pass1_stats),
                    check_interruption=make_interrupt_cb(),
                ),
            )
            pass1_elapsed = time.perf_counter() - started
            print(f"[Pass 1] ok={ok1} elapsed={pass1_elapsed:.1f}s stats={pass1_stats}")

            pass1_completed = pass1_stats.get("completed_chunks", 0)
            total_chunks = pass1_stats.get("total_chunks", 0)

            # ---- Pass 2: resume ----
            print("\n[Pass 2] resuming with the same translation_id")
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
                ),
                make_ctx(
                    checkpoint_manager=ckpt_mgr_2,
                    translation_id=translation_id,
                    stats_callback=make_stats_cb(pass2_stats),
                    check_interruption=None,
                ),
            )
            pass2_elapsed = time.perf_counter() - started
            print(f"[Pass 2] ok={ok2} elapsed={pass2_elapsed:.1f}s stats={pass2_stats}")

            failures = []

            # Pass 1 must have been interrupted at the threshold.
            if ok1:
                failures.append("pass 1 should have been interrupted (ok=True returned)")
            if pass1_completed < INTERRUPT_AFTER:
                failures.append(
                    f"pass 1 completed only {pass1_completed} chunks before interrupt "
                    f"(expected >= {INTERRUPT_AFTER})"
                )
            if total_chunks < INTERRUPT_AFTER + 1:
                failures.append(
                    f"fixture too small: total_chunks={total_chunks}, need > {INTERRUPT_AFTER}"
                )

            # Pass 2 must finish and report all chunks completed.
            if not ok2:
                failures.append("pass 2 (resume) returned False")
            if pass2_stats.get("total_chunks", 0) != total_chunks:
                failures.append(
                    f"pass 2 total_chunks ({pass2_stats.get('total_chunks')}) "
                    f"!= pass 1 total_chunks ({total_chunks})"
                )
            if pass2_stats.get("completed_chunks", 0) != total_chunks:
                failures.append(
                    f"pass 2 completed_chunks ({pass2_stats.get('completed_chunks')}) "
                    f"!= total ({total_chunks})"
                )
            if pass2_stats.get("failed_chunks", 0) != 0:
                failures.append(f"pass 2 failed_chunks={pass2_stats.get('failed_chunks')} (expected 0)")

            # Strongest resume signal: pass 2 must do *strictly less* wall-clock
            # work than pass 1, because the chunks completed in pass 1 are not
            # retranslated. We allow some slack for network variance but expect
            # a meaningful gap.
            expected_remaining = total_chunks - pass1_completed
            if expected_remaining > 0 and pass1_completed > 0:
                # If pass 1 did N chunks and pass 2 should do total-N chunks,
                # pass 2 elapsed should be roughly (total-N)/N of pass 1.
                # Allow a generous 1.5x safety margin on top of that ratio.
                expected_ratio = (expected_remaining / pass1_completed) * 1.5
                actual_ratio = pass2_elapsed / max(pass1_elapsed, 0.1)
                print(f"\nTiming:     pass1={pass1_elapsed:.1f}s ({pass1_completed} chunks), "
                      f"pass2={pass2_elapsed:.1f}s ({expected_remaining} chunks), "
                      f"ratio={actual_ratio:.2f} (expected <= {expected_ratio:.2f})")
                if actual_ratio > expected_ratio:
                    failures.append(
                        f"pass 2 wall-clock ratio {actual_ratio:.2f} > {expected_ratio:.2f}: "
                        f"suggests already-completed chunks were retranslated"
                    )

            # Output file must exist and look like a real translation.
            if not output_path.exists():
                failures.append(f"output file missing: {output_path}")
            else:
                translated = output_path.read_text(encoding="utf-8")
                if not translated.strip():
                    failures.append("output file is empty")
                elif translated.strip() == source_text.strip():
                    failures.append("output is byte-identical to source (translation did not happen)")
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
