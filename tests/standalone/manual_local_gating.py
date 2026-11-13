"""
T8 - Local provider gating.

Calls translate_file() with llm_provider="ollama" and parallel_requests=4
and verifies that the entry point downgrades the parallelism to 1 via the
log_callback event "parallel_requests_forced_local". The translation
itself is allowed to fail (no Ollama daemon assumed); we only assert the
gating fired before the failure.

Run from repo root:
    python tests/standalone/manual_local_gating.py
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
from src.core.adapters import translate_file
from src.persistence.checkpoint_manager import CheckpointManager
from tests.job_helpers import make_ctx, make_job


FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "sample.txt"


async def run() -> int:
    if not FIXTURE.exists():
        print(f"FAIL: fixture not found at {FIXTURE}")
        return 1

    events = []

    def log_callback(event_type, msg):
        events.append((event_type, msg))

    print("Invoking translate_file with provider=ollama, parallel_requests=4")
    print("(Ollama daemon not required: we only check the gating log event)")

    with tempfile.TemporaryDirectory(
        prefix="local_gating_", ignore_cleanup_errors=True
    ) as tmpdir:
        tmpdir_path = Path(tmpdir)
        cwd_before = os.getcwd()
        os.chdir(tmpdir_path)
        try:
            checkpoint_manager = CheckpointManager()
            translation_id = f"t8_{uuid.uuid4().hex[:8]}"
            output_path = tmpdir_path / "out.txt"

            try:
                # We do not care whether this succeeds: most environments have no
                # Ollama daemon. We just need translate_file to reach (and log)
                # the gating step, which happens before the LLM client is built.
                await translate_file(
                    make_job(
                        input_path=str(FIXTURE),
                        output_path=str(output_path),
                        source_language="English",
                        target_language="French",
                        model="any-model",
                        provider="ollama",
                        parallel_workers=4,
                    ),
                    make_ctx(
                        checkpoint_manager=checkpoint_manager,
                        translation_id=translation_id,
                        log_callback=log_callback,
                    ),
                )
            except Exception as e:
                print(f"(translate_file raised, expected without ollama running: {type(e).__name__})")

            gating_events = [e for e in events if e[0] == "parallel_requests_forced_local"]
            if not gating_events:
                print("\nFAIL: 'parallel_requests_forced_local' event not emitted")
                print(f"All events received: {[e[0] for e in events]}")
                return 1

            print(f"OK: gating event fired -> {gating_events[0][1]}")
            return 0
        finally:
            os.chdir(cwd_before)


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
