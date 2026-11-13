"""
T3 - EPUB sequential baseline test.

Builds a small 3-chapter EPUB on the fly, translates it end-to-end through
the public `translate_file()` entry point against Poe (POE_API_KEY from .env),
and checks structural invariants. Used as the reference behavior before the
parallel-requests refactor (GitHub issue #175): the same assertions must
still pass after the refactor when parallel_requests=1.

The EPUB is generated programmatically (not committed) so the test is fully
self-contained.

Run from repo root:
    python tests/standalone/manual_epub_baseline.py
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

from src import config  # noqa: F401
from src.core.adapters import translate_file
from src.persistence.checkpoint_manager import CheckpointManager
from tests.job_helpers import make_ctx, make_job


CHAPTERS = [
    (
        "chapter1.xhtml",
        "The Departure",
        [
            "On the morning of his departure the harbor was already alive with seabirds and the smell of pitch.",
            "He walked the long pier alone, counting his steps because counting was easier than thinking.",
            "The ship waited at the end of the pier, low in the water and patient as a tethered horse.",
        ],
    ),
    (
        "chapter2.xhtml",
        "The Crossing",
        [
            "By the second day the wind had turned and the sailors no longer sang at the rigging.",
            "He sat in the lee of the forecastle and watched the sky harden into a sheet of grey iron.",
            "The captain said nothing about the weather, which was the loudest thing the captain could have said.",
        ],
    ),
    (
        "chapter3.xhtml",
        "The Landing",
        [
            "The new coast rose out of the rain at dusk, dark and unfriendly and exactly as he had imagined.",
            "He shouldered his small bag and stepped onto the wet stones without looking back at the ship.",
            "Whatever waited for him here, he had at least the satisfaction of having arrived under his own name.",
        ],
    ),
]


def _build_minimal_epub(path: Path) -> None:
    """Build a minimal 3-chapter EPUB at the given path."""
    container_xml = (
        '<?xml version="1.0"?>\n'
        '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">\n'
        '  <rootfiles>\n'
        '    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>\n'
        '  </rootfiles>\n'
        '</container>\n'
    )

    manifest_items = "\n".join(
        f'    <item id="ch{i+1}" href="{href}" media-type="application/xhtml+xml"/>'
        for i, (href, _, _) in enumerate(CHAPTERS)
    )
    spine_items = "\n".join(
        f'    <itemref idref="ch{i+1}"/>'
        for i in range(len(CHAPTERS))
    )
    content_opf = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<package version="3.0" xmlns="http://www.idpf.org/2007/opf" unique-identifier="bookid">\n'
        '  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
        '    <dc:identifier id="bookid">urn:uuid:00000000-0000-0000-0000-000000000001</dc:identifier>\n'
        '    <dc:title>Test Voyage</dc:title>\n'
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

    def _chapter_xhtml(title: str, paragraphs: list) -> str:
        body = "\n".join(f"    <p>{p}</p>" for p in paragraphs)
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
        # mimetype must be the first entry and stored uncompressed.
        z.writestr(
            zipfile.ZipInfo("mimetype"),
            "application/epub+zip",
            compress_type=zipfile.ZIP_STORED,
        )
        z.writestr("META-INF/container.xml", container_xml)
        z.writestr("OEBPS/content.opf", content_opf)
        for href, title, paragraphs in CHAPTERS:
            z.writestr(f"OEBPS/{href}", _chapter_xhtml(title, paragraphs))


def _list_epub_xhtml(path: Path) -> list:
    """Return the list of XHTML hrefs in spine order from an EPUB."""
    with zipfile.ZipFile(path, "r") as z:
        return sorted(
            name for name in z.namelist()
            if name.startswith("OEBPS/") and name.endswith(".xhtml")
        )


def _read_epub_chapter_text(path: Path, href: str) -> str:
    """Read raw XHTML content for a chapter href."""
    with zipfile.ZipFile(path, "r") as z:
        return z.read(href).decode("utf-8")


async def run() -> int:
    api_key = os.getenv("POE_API_KEY", "").strip()
    if not api_key:
        print("FAIL: POE_API_KEY is not set in .env")
        return 1
    model = os.getenv("POE_MODEL", "Claude-Sonnet-4").strip() or "Claude-Sonnet-4"

    last_stats: dict = {}

    def stats_callback(stats: dict) -> None:
        last_stats.update(stats)

    with tempfile.TemporaryDirectory(
        prefix="epub_baseline_", ignore_cleanup_errors=True
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

            checkpoint_manager = CheckpointManager()
            translation_id = f"t3_{uuid.uuid4().hex[:8]}"

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
            if total < 1:
                failures.append(f"expected at least 1 chunk, got total_chunks={total}")
            if completed != total:
                failures.append(f"completed_chunks ({completed}) != total_chunks ({total})")
            if failed != 0:
                failures.append(f"failed_chunks={failed} (expected 0)")

            if not output_path.exists():
                failures.append(f"output EPUB missing: {output_path}")
            else:
                out_chapters = _list_epub_xhtml(output_path)
                print(f"Output:     {output_path.stat().st_size} bytes, "
                      f"{len(out_chapters)} XHTML files")
                if out_chapters != source_chapters:
                    failures.append(
                        f"chapter list/order mismatch: source={source_chapters} output={out_chapters}"
                    )
                else:
                    # Verify each chapter still has paragraphs and that at least
                    # one chapter actually changed (translation happened).
                    any_changed = False
                    for href in source_chapters:
                        src_xhtml = _read_epub_chapter_text(input_path, href)
                        out_xhtml = _read_epub_chapter_text(output_path, href)
                        src_p_count = src_xhtml.count("<p>")
                        out_p_count = out_xhtml.count("<p>")
                        if out_p_count != src_p_count:
                            failures.append(
                                f"{href}: <p> count mismatch source={src_p_count} output={out_p_count}"
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


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
