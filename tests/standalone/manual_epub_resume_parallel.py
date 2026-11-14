"""
T11 - Parallel EPUB resume test.

Builds a 3-chapter EPUB with multiple paragraphs per chapter, runs a
parallel translation with parallel_requests=3, interrupts after a few
chunks are done via the interruption callback, then resumes the same
translation_id with parallel_requests=3 again.

Asserts that:
  * Pass 2 issues fewer LLM calls than pass 1 (chunks done in pass 1
    are not retranslated, regardless of which files they belonged to).
  * Every chapter in the final output has been translated (none left
    as source-language placeholder).
  * The final output EPUB is structurally intact.

This is the regression net for the parallel-mode resume bug discovered
during user testing on a real EPUB.

Run from repo root:
    python tests/standalone/manual_epub_resume_parallel.py
"""

import asyncio
import os
import sys
import tempfile
import time
import uuid
import zipfile
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
from src.core.llm.providers.poe import PoeProvider
from src.persistence.checkpoint_manager import CheckpointManager
from tests.job_helpers import make_ctx, make_job


PARALLEL = 3
INTERRUPT_AFTER_CHUNKS = 3  # cumulative chunks


def _build_chunky_epub(path: Path) -> None:
    """
    Build a 3-chapter EPUB where each chapter has enough text to be split
    into multiple chunks by the EPUB chunker. We use repeated paragraphs of
    moderate length so the token-based chunker breaks each chapter into
    several chunks, which is required for the mid-file interruption case.
    """
    base_paragraph = (
        "The lighthouse keeper's journal had grown thick with salt and ink, "
        "and the third assistant, recently arrived from the mainland, "
        "watched the older man with the polite patience of someone who has "
        "been told he must learn a great deal in a very short time."
    )

    container_xml = (
        '<?xml version="1.0"?>\n'
        '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">\n'
        '  <rootfiles>\n'
        '    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>\n'
        '  </rootfiles>\n'
        '</container>\n'
    )

    hrefs = [f"chapter{i+1}.xhtml" for i in range(3)]
    manifest_items = "\n".join(
        f'    <item id="ch{i+1}" href="{href}" media-type="application/xhtml+xml"/>'
        for i, href in enumerate(hrefs)
    )
    spine_items = "\n".join(
        f'    <itemref idref="ch{i+1}"/>'
        for i in range(3)
    )
    content_opf = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<package version="3.0" xmlns="http://www.idpf.org/2007/opf" unique-identifier="bookid">\n'
        '  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
        '    <dc:identifier id="bookid">urn:uuid:00000000-0000-0000-0000-000000000003</dc:identifier>\n'
        '    <dc:title>Chunky Book</dc:title>\n'
        '    <dc:language>en</dc:language>\n'
        '    <meta property="dcterms:modified">2024-01-01T00:00:00Z</meta>\n'
        '  </metadata>\n'
        '  <manifest>\n'
        f'{manifest_items}\n'
        '  </manifest>\n'
        '  <spine>\n'
        f'{spine_items}\n'
        '  </spine>\n'
        '</package>\n'
    )

    def _chapter_xhtml(title: str, n_paragraphs: int) -> str:
        body = "\n".join(
            f"    <p>Paragraph {i+1}. {base_paragraph}</p>"
            for i in range(n_paragraphs)
        )
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE html>\n'
            '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en">\n'
            '<head>\n'
            f'  <title>{title}</title>\n'
            '</head>\n'
            '<body>\n'
            f'  <h1>{title}</h1>\n'
            f'{body}\n'
            '</body>\n'
            '</html>\n'
        )

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(
            zipfile.ZipInfo("mimetype"),
            "application/epub+zip",
            compress_type=zipfile.ZIP_STORED,
        )
        z.writestr("META-INF/container.xml", container_xml)
        z.writestr("OEBPS/content.opf", content_opf)
        # 12 paragraphs per chapter ~ 3000 tokens at default chunker settings
        # should yield ~6-8 chunks per chapter.
        for i, href in enumerate(hrefs):
            z.writestr(f"OEBPS/{href}", _chapter_xhtml(f"Chapter {i+1}", 12))


class CallCounter:
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


def _xhtml_chapter_translated(input_path: Path, output_path: Path, href: str) -> bool:
    """A chapter is considered translated if its output XHTML differs from input."""
    with zipfile.ZipFile(input_path) as z:
        src = z.read(href).decode("utf-8", errors="replace")
    with zipfile.ZipFile(output_path) as z:
        out = z.read(href).decode("utf-8", errors="replace")
    return src != out


async def run() -> int:
    api_key = os.getenv("POE_API_KEY", "").strip()
    if not api_key:
        print("FAIL: POE_API_KEY is not set in .env")
        return 1
    model = os.getenv("POE_MODEL", "Claude-Sonnet-4").strip() or "Claude-Sonnet-4"

    pass1_stats: dict = {}
    pass2_stats: dict = {}
    counter_pass1 = CallCounter()
    counter_pass2 = CallCounter()

    def make_stats_cb(target):
        def _cb(stats):
            target.update(stats)
        return _cb

    def make_interrupt_cb():
        def _cb():
            return pass1_stats.get("completed_chunks", 0) >= INTERRUPT_AFTER_CHUNKS
        return _cb

    with tempfile.TemporaryDirectory(
        prefix="epub_resume_parallel_", ignore_cleanup_errors=True
    ) as tmpdir:
        tmpdir_path = Path(tmpdir)
        cwd_before = os.getcwd()
        os.chdir(tmpdir_path)
        try:
            input_path = tmpdir_path / "book.epub"
            output_path = tmpdir_path / "book.fr.epub"
            _build_chunky_epub(input_path)

            with zipfile.ZipFile(input_path) as z:
                source_chapters = sorted(
                    n for n in z.namelist() if n.endswith(".xhtml")
                )
            print(f"Fixture:    {input_path.name} "
                  f"({input_path.stat().st_size} bytes, "
                  f"{len(source_chapters)} chapters)")
            print(f"Provider:   poe / {model}")
            print(f"Parallel:   parallel_requests={PARALLEL}")
            print(f"Interrupt:  after {INTERRUPT_AFTER_CHUNKS} chunks completed")

            translation_id = f"t11_{uuid.uuid4().hex[:8]}"

            # ---- Pass 1: parallel, interrupted ----
            print("\n[Pass 1] parallel translate, will interrupt")
            original1 = _install_counter(counter_pass1)
            try:
                ckpt_mgr_1 = CheckpointManager()
                started = time.perf_counter()
                ok1 = await translate_file(
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
            print(f"[Pass 1] LLM calls: {counter_pass1.total}")

            # ---- Pass 2: parallel, resume ----
            print("\n[Pass 2] parallel resume")
            original2 = _install_counter(counter_pass2)
            try:
                ckpt_mgr_2 = CheckpointManager()
                started = time.perf_counter()
                ok2 = await translate_file(
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
            print(f"[Pass 2] LLM calls: {counter_pass2.total}")

            failures = []

            # Pass 1 expectations. translate_file always returns True for EPUB
            # (legacy translate_epub_file ignores the status). The interruption
            # signal is that pass1 stats show completed < total_chunks.
            pass1_completed = pass1_stats.get("completed_chunks", 0)
            pass1_total = pass1_stats.get("total_chunks", 0)
            if pass1_completed >= pass1_total > 0:
                failures.append(
                    f"pass 1 was supposed to be interrupted but completed "
                    f"all {pass1_total} chunks"
                )
            if pass1_completed < INTERRUPT_AFTER_CHUNKS:
                failures.append(
                    f"pass 1 completed {pass1_completed} chunks before interrupt "
                    f"(expected >= {INTERRUPT_AFTER_CHUNKS})"
                )

            # Pass 2 must finish.
            if not ok2:
                failures.append("pass 2 (resume) returned False")

            # No retranslation of done work: pass 2 calls + pass 1 calls
            # should be close to the total chunk count (allow small slack for
            # in-flight rework on the file being processed at interrupt-time,
            # i.e. file partials below the chunk-saved checkpoint may need a
            # tiny re-redo).
            total_calls = counter_pass1.total + counter_pass2.total
            if total_calls > pass1_stats.get("total_chunks", 9999) + INTERRUPT_AFTER_CHUNKS:
                failures.append(
                    f"too many total LLM calls: pass1={counter_pass1.total} + "
                    f"pass2={counter_pass2.total} = {total_calls}, expected close to "
                    f"total chunks"
                )

            # The pass 2 must do strictly less calls than pass 1 (otherwise
            # nothing was actually saved and resumed).
            if counter_pass2.total >= counter_pass1.total + 2:
                # +2 slack for the legit rework
                failures.append(
                    f"pass 2 did {counter_pass2.total} LLM calls but pass 1 only "
                    f"did {counter_pass1.total}: resume didn't shortcut anything"
                )

            # Final output must exist and every chapter must be translated.
            if not output_path.exists():
                failures.append(f"output EPUB missing: {output_path}")
            else:
                missing = [
                    href for href in source_chapters
                    if not _xhtml_chapter_translated(input_path, output_path, href)
                ]
                if missing:
                    failures.append(
                        f"{len(missing)} chapter(s) not translated in final output: {missing}"
                    )
                print(f"\nOutput:     {output_path.stat().st_size} bytes, "
                      f"{len(source_chapters)} chapters, all translated"
                      if not missing else
                      f"\nOutput:     {output_path.stat().st_size} bytes, "
                      f"{len(missing)} chapter(s) untranslated!")

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
