"""
Stress repro for the 'parallel EPUB, output identical to source' bug
reported by the user. T7 used 1 chunk per chapter; this builds 3
chapters with ~3-4 chunks each (multi-chunk-per-file) and runs with
parallel_requests=4, mirroring the user's web-UI scenario.

Run from repo root:
    python tests/standalone/manual_epub_parallel_big.py
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


PARALLEL = 4


def _build_big_epub(path: Path) -> None:
    """Build a 3-chapter EPUB where each chapter has many paragraphs."""
    paragraphs_per_chapter = [
        [
            "The morning light came in slow grey waves over the harbor town, and the wooden shutters of the inn clattered against their frames.",
            "Below in the kitchen the cook was already at work, banging her copper pots with the satisfied violence of a woman who has been awake longer than anyone else.",
            "The traveller, who had arrived in the dark and remembered nothing of his room, sat up and tried to assemble the previous evening out of the loose pieces in his head.",
            "He counted two glasses of brandy, the long conversation with the customs officer, the strange way the lamp had hissed when the wind came under the door.",
            "Then he remembered the letter, and remembering the letter, he reached for the inside pocket of his coat, which was on the chair where he had left it.",
            "The letter was still there, folded twice, the seal intact, the address written in the small precise handwriting of a man who could not afford to be misunderstood.",
            "He held it in his hand for a moment without opening it, the way one holds a key to a door one is not yet sure one wants to open.",
        ],
        [
            "By midmorning he had crossed the long bridge and entered the lower town, where the streets narrowed and the smell of fish gave way to the smell of leather and ink.",
            "He stopped twice to ask directions and twice received the same answer, given with a kind of weary patience that suggested the question had been asked many times before.",
            "The house, when he found it, was smaller than he had imagined, set back from the street behind a courtyard of broken paving and an apple tree that had given up on apples some years before.",
            "He stood at the gate for a long time, watching the windows for a sign of life, and finding none, finally rang the bell and waited with his hat in his hand.",
            "The woman who opened the door had a face that had once been beautiful and had now settled into something more interesting and more difficult to read.",
            "She did not invite him in, but neither did she close the door, and after a moment of careful study she stepped aside and let him pass into the dark hallway.",
        ],
        [
            "The room he was shown to was warmer than the hallway and smaller than he had expected, with a single window that looked out onto the apple tree he had noticed from the gate.",
            "There was a desk under the window, and on the desk a single oil lamp burning very low, and in the chair behind the desk a man whose silence was the loudest thing about him.",
            "The traveller introduced himself as he had been told to, naming only the city and not the man who had sent him, and waited for the silence to break or for some other thing to happen instead.",
            "The man behind the desk reached out one hand and the traveller placed the letter in it, and the silence held while the seal was broken and the paper unfolded with great care.",
            "When the reading was done the man looked up, and his expression did not so much change as become more clearly what it had already been, like a photograph slowly coming into focus.",
            "He said, in a voice that suggested he was used to being obeyed without raising it, that the traveller had done well to come, and that there was now a great deal of work for them to do together.",
        ],
    ]

    container_xml = (
        '<?xml version="1.0"?>\n'
        '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">\n'
        '  <rootfiles>\n'
        '    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>\n'
        '  </rootfiles>\n'
        '</container>\n'
    )

    hrefs = [f"chapter{i+1}.xhtml" for i in range(len(paragraphs_per_chapter))]
    manifest_items = "\n".join(
        f'    <item id="ch{i+1}" href="{href}" media-type="application/xhtml+xml"/>'
        for i, href in enumerate(hrefs)
    )
    spine_items = "\n".join(
        f'    <itemref idref="ch{i+1}"/>'
        for i in range(len(hrefs))
    )
    content_opf = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<package version="3.0" xmlns="http://www.idpf.org/2007/opf" unique-identifier="bookid">\n'
        '  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
        '    <dc:identifier id="bookid">urn:uuid:00000000-0000-0000-0000-000000000002</dc:identifier>\n'
        '    <dc:title>The Letter</dc:title>\n'
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
        z.writestr(
            zipfile.ZipInfo("mimetype"),
            "application/epub+zip",
            compress_type=zipfile.ZIP_STORED,
        )
        z.writestr("META-INF/container.xml", container_xml)
        z.writestr("OEBPS/content.opf", content_opf)
        for href, paragraphs in zip(hrefs, paragraphs_per_chapter):
            title = href.replace(".xhtml", "").replace("chapter", "Chapter ")
            z.writestr(f"OEBPS/{href}", _chapter_xhtml(title, paragraphs))


async def run() -> int:
    api_key = os.getenv("POE_API_KEY", "").strip()
    if not api_key:
        print("FAIL: POE_API_KEY is not set in .env")
        return 1
    model = os.getenv("POE_MODEL", "Claude-Sonnet-4").strip() or "Claude-Sonnet-4"

    events = []
    def log_callback(event_type, msg="", data=None):
        events.append((event_type, msg, data))

    last_stats: dict = {}
    def stats_callback(stats: dict) -> None:
        last_stats.update(stats)

    with tempfile.TemporaryDirectory(
        prefix="epub_big_", ignore_cleanup_errors=True
    ) as tmpdir:
        tmpdir_path = Path(tmpdir)
        cwd_before = os.getcwd()
        os.chdir(tmpdir_path)
        try:
            input_path = tmpdir_path / "letter.epub"
            output_path = tmpdir_path / "letter.fr.epub"
            _build_big_epub(input_path)
            print(f"Fixture:    {input_path.name} ({input_path.stat().st_size} bytes)")
            print(f"Provider:   poe / {model}")
            print(f"Parallel:   parallel_requests={PARALLEL}")

            # Force small chunks so each chapter has multiple chunks.
            checkpoint_manager = CheckpointManager()
            translation_id = f"big_{uuid.uuid4().hex[:8]}"

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
                    max_tokens_per_chunk=120,
                    parallel_workers=PARALLEL,
                ),
                make_ctx(
                    checkpoint_manager=checkpoint_manager,
                    translation_id=translation_id,
                    log_callback=log_callback,
                    stats_callback=stats_callback,
                ),
            )
            elapsed_s = time.perf_counter() - started

            print(f"Result:     ok={ok} elapsed={elapsed_s:.1f}s")
            print(f"Stats:      {last_stats}")

            # Spot-check key events.
            interesting_events = [
                e for e in events
                if any(token in e[0] for token in [
                    "translate_start", "translate_failed", "epub_file",
                    "error", "failed", "completed",
                ])
            ]
            print("\nKey events:")
            for ev in interesting_events[-30:]:
                ev_type, ev_msg = ev[0], ev[1]
                print(f"  [{ev_type}] {ev_msg[:200]}")

            # Compare source vs output: are the chapters actually translated?
            print("\nContent comparison:")
            with zipfile.ZipFile(input_path) as z_src, zipfile.ZipFile(output_path) as z_out:
                src_chapters = sorted(n for n in z_src.namelist() if n.endswith(".xhtml"))
                for href in src_chapters:
                    src = z_src.read(href).decode("utf-8", errors="replace")
                    out = z_out.read(href).decode("utf-8", errors="replace")
                    same = (src == out)
                    print(f"  {href}: source={len(src)}B, output={len(out)}B, identical={same}")

            return 0 if ok else 1
        finally:
            os.chdir(cwd_before)


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
