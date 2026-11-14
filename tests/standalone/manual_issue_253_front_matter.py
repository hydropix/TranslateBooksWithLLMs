"""Issue #253 - source paragraphs left at the end of a Plain Text Mode segment.

What this reproduces
--------------------
Plain Text Mode maps a segment's answer back to the source paragraph indices it
covered, positionally. When the model returns fewer paragraphs than the segment
holds, the reconciliation pads the tail with empty slots, and those slots fall
back to source text: a run of untranslated paragraphs at the end of the segment,
under a translation that is otherwise correct and complete.

The trigger observed on a real book (reporter's analysis on #253) is front
matter: a one-line disclaimer, an author's note of one to five paragraphs, and a
``<p><strong>Chapter N: Title</strong></p>`` heading, all at the top of a
chapter. The model treats those as metadata and folds them away, the whole
chapter slides up by that many paragraphs, and the same number of source
paragraphs reappears at the end of the segment.

This script builds an EPUB with exactly that shape - twelve chapters cycling
through four front-matter forms, forty prose paragraphs each - translates it end
to end with Plain Text Mode against a real model, and reports per chapter:

  - source vs output block count (always equal: nothing is deleted, only shifted)
  - output blocks still in the source script (the symptom)
  - the paragraph alignment counters and the plain_text_* events

A real LLM is required: the failure IS model drift, so a fake provider cannot
produce it. The deterministic half of the contract - detection, the count-stating
retry, the per-paragraph repair, the counters - is covered offline by
tests/unit/test_plain_text_paragraph_mismatch.py.

Run from the repo root (needs OPENROUTER_API_KEY in .env):

    python tests/standalone/manual_issue_253_front_matter.py

Environment overrides:
    REPRO_MODEL      OpenRouter model id (default mistralai/mistral-medium-3.1)
    REPRO_CHAPTERS   how many chapters to generate (default 12)
"""

import asyncio
import json
import os
import re
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

# A segment must hold enough paragraphs for the front matter and the prose to
# share one LLM call, which is the situation the bug lives in.
os.environ.setdefault("MAX_TOKENS_PER_CHUNK", "1800")

from src import config  # noqa: E402,F401
from src.core.adapters import translate_file  # noqa: E402
from src.persistence.checkpoint_manager import CheckpointManager  # noqa: E402
from tests.job_helpers import make_ctx, make_job


MODEL = os.getenv("REPRO_MODEL", "mistralai/mistral-medium-3.1")
CHAPTER_COUNT = int(os.getenv("REPRO_CHAPTERS", "12"))
MAX_TOKENS = 1800

DISCLAIMER = (
    "This work is a fan-written derivative. The author claims no ownership of "
    "the original characters or setting, and no profit is made from it."
)

NOTES = [
    "A/N: Thank you all so much for the comments on the last chapter, they genuinely keep me going.",
    "A/N: Updates should be weekly from here on out, assuming work stays quiet.",
    "A/N: This chapter was the hardest one to write so far, and I rewrote the middle section four times.",
    "A/N: Content warning for this chapter: descriptions of illness and one brief scene of violence.",
    "A/N: Enormous thanks to my beta reader, who caught roughly a hundred continuity errors.",
    "A/N: As always, comments and corrections are very welcome.",
]

PROSE = [
    "The rain had been falling since before dawn, and by the time the bells rang for the second hour it had settled into the kind of steady grey drizzle that makes a city look older than it is.",
    "Marek pulled his coat tighter and kept to the arcades, where the stone still held a little warmth from the day before.",
    "He had walked this route four hundred times, maybe more, and he could have named every shuttered window along it without once lifting his head.",
    "At the corner of the fish market a woman was arguing with a carter about the price of ice, and neither of them was really listening to the other.",
    "That was the thing about the lower quarter: an argument there was less a disagreement than a form of greeting.",
    "He bought bread from the stall that always opened early and ate half of it while he walked, the crust still hot enough to burn his fingers.",
    "The letter was in his inside pocket, folded twice, its wax seal broken three days ago and read so many times since that he no longer needed to look at it.",
    "It said very little. It said enough.",
    "His sister had written it in the careful hand she used for official business, which meant she had expected it to be read by someone other than him.",
    "Come home, it said, and then, in the last line, do not come alone.",
    "He had spent two of the three days trying to decide which of those instructions was the more frightening.",
    "The tram was late, as it always was when the river ran high, so he took the stairs down to the embankment and walked the rest of the way.",
    "Below the bridge the water had risen almost to the top of the old flood marks, brown and quick and carrying whole branches out toward the estuary.",
    "A boy was throwing stones at the current from the lower step, and each stone vanished without a sound.",
    "Marek stopped and watched him for a while, because he was not in a hurry and because the boy's persistence was oddly reassuring.",
    "Then the clock in the customs tower struck and he remembered that he was, in fact, in a hurry after all.",
    "The office was on the third floor of a building that had been a granary before the war and had never quite stopped smelling like one.",
    "He climbed the stairs slowly, rehearsing what he would say, and by the second landing he had abandoned the rehearsal entirely.",
    "Verel was waiting on the landing with two cups of tea she had clearly made a long time ago, and she handed him the colder one without apology.",
    "\"You took your time,\" she said, and it was not a reproach so much as a statement of the family's oldest and least interesting fact.",
    "The room behind her was full of paper: ledgers, manifests, a chart of the estuary pinned to the wall with what looked like a kitchen knife.",
    "He sat down in the only chair that did not have a stack of documents on it and waited for her to decide how much she was going to tell him.",
    "\"The Harbourmaster's office has been quietly buying up the western wharves,\" she said at last. \"Through three different companies, none of which exist.\"",
    "Marek turned the cup around in his hands. \"That is not illegal.\"",
    "\"No,\" she agreed. \"What they are doing with the warehouses is.\"",
    "Outside, the rain had thickened again, and the light through the window had gone the colour of old tin.",
    "She told him the rest quickly, the way a person does when they have rehearsed something so many times that saying it slowly has become unbearable.",
    "By the end of it he had stopped drinking the tea, which was cold anyway, and had begun to understand the last line of her letter.",
    "\"How many people know?\" he asked.",
    "\"Counting the two of us,\" she said, \"four. One of them died on Tuesday.\"",
    "There was a long silence in which the building creaked and settled and somewhere below a door was pulled shut against the wind.",
    "\"Tell me about Tuesday,\" he said.",
    "She did, and it was worse than he had prepared himself for, and he was a man who habitually prepared himself for the worse of two things.",
    "When she finished, she went to the window and stood with her back to him, looking down at the flooded quay.",
    "\"I did not want to write to you,\" she said. \"I want you to know that I considered every other option first.\"",
    "\"I know,\" he said, and he did, and that was the part that frightened him.",
    "The customs clock struck again, and the sound came in flat and muffled through the glass, as if the whole city had been wrapped in wet cloth.",
    "He stood up. \"Show me the warehouse,\" he said, and reached for his coat.",
    "The stairs down were darker than the stairs up had been, and neither of them said anything until they were out in the rain again.",
    "The western wharves lay twenty minutes' walk along the embankment, past the grain silos and the shuttered customs sheds.",
    "The boy was gone now. The river had risen another hand's breadth in the time it had taken to climb three flights of stairs.",
]

TITLES = [
    "High Water", "The Letter", "The Granary", "Do Not Come Alone",
    "The Western Wharves", "Four People", "Tuesday", "The Chart on the Wall",
    "Cold Tea", "What the River Carries", "The Harbourmaster", "Under Our Own Name",
]

# Front-matter shapes cycled across chapters: (disclaimer, author's notes, heading).
# The first is the control - a chapter with no front matter must stay clean.
SHAPES = [
    (False, 0, False),
    (False, 2, False),
    (True, 1, True),
    (True, 3, True),
]

GLOSSARY = {
    "Marek": "Марек", "Verel": "Верель", "Harbourmaster": "начальник порта",
    "the lower quarter": "нижний квартал", "the embankment": "набережная",
    "the granary": "амбар", "customs tower": "таможенная башня",
    "the western wharves": "западные причалы", "estuary": "устье",
    "manifest": "манифест", "ledger": "гроссбух", "quay": "причал",
}

CUSTOM_INSTRUCTIONS = (
    "Keep the narrative voice restrained and literary. Preserve the author's short "
    "declarative sentences; do not smooth them into longer periods. Dialogue should "
    "sound spoken, not written."
)


def build_chapters():
    """Return the chapter descriptors, each with its ordered (kind, text) blocks."""
    chapters = []
    for i in range(CHAPTER_COUNT):
        disclaimer, note_count, heading = SHAPES[i % len(SHAPES)]
        title = f"Chapter {i + 1}: {TITLES[i % len(TITLES)]}"
        rotation = (i * 7) % len(PROSE)
        prose = (PROSE[rotation:] + PROSE[:rotation])[:40]

        blocks = []
        if disclaimer:
            blocks.append(("disclaimer", DISCLAIMER))
        for k in range(note_count):
            blocks.append(("note", NOTES[(i * 2 + k) % len(NOTES)]))
        if heading:
            blocks.append(("heading", title))
        blocks.extend(("prose", p) for p in prose)

        chapters.append({
            "href": f"chapter{i + 1}.xhtml",
            "title": title,
            "blocks": blocks,
            "front_matter": len(blocks) - len(prose),
        })
    return chapters


def _chapter_xhtml(chapter):
    parts = []
    for kind, text in chapter["blocks"]:
        if kind == "heading":
            parts.append(f"  <p><strong>{text}</strong></p>")
        else:
            parts.append(f"  <p>{text}</p>")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE html>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en">\n'
        '<head>\n'
        f'  <title>{chapter["title"]}</title>\n'
        '</head>\n'
        '<body>\n'
        + "\n".join(parts) + "\n"
        '</body>\n'
        '</html>\n'
    )


def build_epub(path: Path, chapters) -> None:
    container_xml = (
        '<?xml version="1.0"?>\n'
        '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">\n'
        '  <rootfiles>\n'
        '    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>\n'
        '  </rootfiles>\n'
        '</container>\n'
    )
    manifest = "\n".join(
        f'    <item id="ch{i + 1}" href="{c["href"]}" media-type="application/xhtml+xml"/>'
        for i, c in enumerate(chapters)
    )
    spine = "\n".join(f'    <itemref idref="ch{i + 1}"/>' for i in range(len(chapters)))
    content_opf = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<package version="3.0" xmlns="http://www.idpf.org/2007/opf" unique-identifier="bookid">\n'
        '  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
        '    <dc:identifier id="bookid">urn:uuid:00000000-0000-0000-0000-000000000253</dc:identifier>\n'
        '    <dc:title>High Water</dc:title>\n'
        '    <dc:language>en</dc:language>\n'
        '    <meta property="dcterms:modified">2024-01-01T00:00:00Z</meta>\n'
        '  </metadata>\n'
        '  <manifest>\n'
        f'{manifest}\n'
        '  </manifest>\n'
        '  <spine>\n'
        f'{spine}\n'
        '  </spine>\n'
        '</package>\n'
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(
            zipfile.ZipInfo("mimetype"), "application/epub+zip",
            compress_type=zipfile.ZIP_STORED,
        )
        z.writestr("META-INF/container.xml", container_xml)
        z.writestr("OEBPS/content.opf", content_opf)
        for chapter in chapters:
            z.writestr(f"OEBPS/{chapter['href']}", _chapter_xhtml(chapter))


_TAG_RE = re.compile(r"<[^>]+>")
_BLOCK_RE = re.compile(r"<(p|h[1-6]|li|blockquote|pre)\b[^>]*>(.*?)</\1>", re.S | re.I)
_CYRILLIC_RE = re.compile("[Ѐ-ӿ]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def output_blocks(xhtml: str):
    """Return [(class attribute, text)] for every block element in the body."""
    body = xhtml.split("<body", 1)[-1]
    blocks = []
    for match in _BLOCK_RE.finditer(body):
        opening = match.group(0).split(">", 1)[0]
        class_match = re.search(r'class="([^"]*)"', opening)
        text = _TAG_RE.sub("", match.group(2))
        for entity, char in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                             ("&#160;", " "), ("&nbsp;", " ")):
            text = text.replace(entity, char)
        blocks.append((class_match.group(1) if class_match else "", " ".join(text.split())))
    return blocks


def is_source_language(text: str) -> bool:
    """True when a block came back in the source script rather than translated."""
    cyrillic = len(_CYRILLIC_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    return latin > cyrillic and latin > 0


async def run() -> int:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("FAIL: OPENROUTER_API_KEY is not set in .env")
        return 1

    chapters = build_chapters()
    events = []
    stats = {}

    def log_callback(event_type, message=None, *args, **kwargs):
        events.append((str(event_type), str(message)))

    def stats_callback(payload):
        stats.update(payload)

    with tempfile.TemporaryDirectory(prefix="issue253_", ignore_cleanup_errors=True) as tmp:
        tmp_path = Path(tmp)
        cwd_before = os.getcwd()
        os.chdir(tmp_path)
        try:
            input_path = tmp_path / "highwater.epub"
            output_path = tmp_path / "highwater.ru.epub"
            build_epub(input_path, chapters)

            print(f"Model:      openrouter / {MODEL}")
            print(f"Settings:   plain_text_mode=True, {MAX_TOKENS} tokens/chunk, parallel=1, "
                  "text_cleanup=False")
            print(f"Fixture:    {len(chapters)} chapters, blocks "
                  f"{[len(c['blocks']) for c in chapters]}")
            print()

            started = time.perf_counter()
            ok = await translate_file(
                make_job(
                    input_path=str(input_path),
                    output_path=str(output_path),
                    source_language="English",
                    target_language="Russian",
                    model=MODEL,
                    provider="openrouter",
                    api_key=api_key,
                    max_tokens_per_chunk=MAX_TOKENS,
                    parallel_workers=1,
                    prompt_options={
                    "text_cleanup": False,
                    "plain_text_mode": True,
                    "glossary_terms": GLOSSARY,
                    "custom_instructions": CUSTOM_INSTRUCTIONS,
                },
                ),
                make_ctx(
                    checkpoint_manager=CheckpointManager(),
                    translation_id=f"issue253_{uuid.uuid4().hex[:8]}",
                    log_callback=log_callback,
                    stats_callback=stats_callback,
                ),
            )
            elapsed = time.perf_counter() - started
            print(f"Result:     ok={ok} elapsed={elapsed:.0f}s")
            print(f"Stats:      {json.dumps(stats, ensure_ascii=False)}")
            print()

            failures = []
            if not ok:
                failures.append("translate_file returned False")

            llm_calls = sum(1 for key, _ in events if key == "llm_request")
            affected = 0
            for chapter in chapters:
                xhtml = zipfile.ZipFile(output_path).read(f"OEBPS/{chapter['href']}").decode("utf-8")
                blocks = output_blocks(xhtml)
                untranslated = [i for i, (_, text) in enumerate(blocks) if is_source_language(text)]
                marked = [i for i, (cls, _) in enumerate(blocks) if "plain-text-untranslated" in cls]

                if len(blocks) != len(chapter["blocks"]):
                    failures.append(
                        f"{chapter['href']}: block count {len(blocks)} != "
                        f"{len(chapter['blocks'])} (the pipeline never deletes blocks)"
                    )
                if untranslated:
                    affected += 1
                    failures.append(
                        f"{chapter['href']}: {len(untranslated)} block(s) left in the source "
                        f"language at positions {untranslated}"
                    )
                print(f"{chapter['href']:<16} front_matter={chapter['front_matter']:<2} "
                      f"blocks={len(blocks):<3} source-language={len(untranslated)} "
                      f"marked={len(marked)}")

            print()
            print(f"LLM calls:  {llm_calls} for {stats.get('total_chunks', '?')} segments")
            for key in ("paragraph_count_mismatches", "paragraph_retry_recovered",
                        "paragraph_repair_failed"):
                print(f"{key:<28} {stats.get(key, 'MISSING - not surfaced by the payload')}")
            if "paragraph_count_mismatches" not in stats:
                failures.append("the paragraph alignment counters never reached the stats payload")

            print()
            for key in ("plain_text_paragraph_mismatch", "plain_text_paragraph_retry_recovered",
                        "plain_text_paragraph_repair_started", "plain_text_paragraph_repair_failed"):
                matching = [msg for event, msg in events if event == key]
                print(f"{key:<40} {len(matching)}")

            if failures:
                print("\nFAIL:")
                for message in failures:
                    print(f"  - {message}")
                return 1

            print(f"\nOK: {len(chapters)} chapters, no source-language block in the output")
            return 0
        finally:
            os.chdir(cwd_before)


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
