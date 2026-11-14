"""Repro / proof for discussion #199 (bilingual EPUB output).

Two problems were reported on bilingual output:
  1. Layout: source and translation appeared as large separated blocks
     instead of alternating paragraph by paragraph.
  2. Data loss: when a chapter is wrapped in a single container element
     (e.g. <div class="Section13">) that spans many chunks, almost all text
     vanished because per-chunk reconstruction reparsed unbalanced fragments.

This script runs the REAL pipeline (tag preservation + chunker + bilingual
reconstruction + body reinjection) against two representative bodies:
  - a chapter wrapped in a container div (the data-loss case), and
  - a Japanese web-novel layout with top-level <p> and blank separators.

A perfect LLM is simulated (translation preserves every placeholder). The
script asserts that all source text survives, every paragraph gets a
translation, and source/translation alternate at the paragraph level.

Optionally pass a real EPUB + chapter to exercise an actual file:
  python tests/standalone/repro_bilingual_199.py [book.epub] [chapter.htm]
"""
import sys, os, re, zipfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lxml import etree
from src.core.epub.tag_preservation import TagPreserver
from src.core.epub.html_chunker import HtmlChunker
from src.core.epub.body_reassembly import reconstruct_html
from src.core.epub.placeholder_renumberer import globalize_placeholders
from src.core.epub.body_serializer import (
    extract_body_html, replace_body_content, normalize_whitespace,
)

# A chapter wrapped in a single container div spanning the whole body — this
# is the structure that caused the catastrophic data loss (#199).
CONTAINER_BODY = (
    '<span class="calibre1"><br clear="all"/></span>'
    '<div class="Section13">'
    + ''.join(f'<p class="MsoNormal3"><span> </span></p>' for _ in range(6))
    + ''.join(
        f'<p class="MsoNormal3"><span>Paragraphe source numéro {i}, '
        f'avec « guillemets » et ponctuation ; et une phrase complète.</span></p>'
        for i in range(1, 25)
    )
    + '</div>'
)

# Japanese web-novel layout: top-level <p>, many blank separators (#199 OP).
JAPANESE_BODY = (
    '<p id="p2"> </p>'
    '<p id="p3">嘘つき……ずっと一緒に居てくれると言ったのに……。</p>'
    '<p id="p4"> </p>'
    '<p id="p5">綺麗な髪色のロングヘアがよく似合う、耽美な女性。</p>'
    '<p id="p6" class="blank"><br /></p>'
    '<p id="p7">その背中を追いかけることもできたのに、足が信じられない。</p>'
    '<p id="p8"> </p>'
    '<p id="p9">私は家に一人残された。</p>'
    '<p id="p10" class="blank"><br /><br /></p>'
    '<p id="p11">私（谷口　和奏）は二十五歳になり社会人三年目。</p>'
)


def fake_translate(chunk_text):
    """Mimic a perfect LLM: keep all [idN] placeholders, translate the text."""
    n = [0]
    def repl(m):
        n[0] += 1
        return f" [EN#{n[0]}] "
    return re.sub(r'(?<=\])[^\[]+(?=\[)', repl, chunk_text)


def squeeze(s):
    """Drop ALL whitespace (incl. U+00A0/U+202F) for spacing-agnostic checks."""
    return re.sub(r'\s+', '', s)


def run_case(name, body_html):
    print("=" * 20, name, "=" * 20)
    body_html = normalize_whitespace(body_html)

    # Ground-truth source paragraphs (non-empty).
    doc = etree.fromstring(f"<body>{body_html}</body>".encode("utf-8"),
                           etree.XMLParser(recover=True, huge_tree=True))
    src_paras = [
        "".join(p.itertext())
        for p in doc.iter()
        if isinstance(p.tag, str)
        and p.tag.split('}')[-1].lower() in ('p', 'h1', 'h2', 'h3')
        and "".join(p.itertext()).strip()
    ]
    src_chars = len(squeeze("".join(src_paras)))

    tp = TagPreserver()
    text_ph, tag_map = tp.preserve_tags(body_html)
    chunks = HtmlChunker(max_tokens=450).chunk_html_with_placeholders(text_ph, tag_map)
    translated_chunks = [
        globalize_placeholders(fake_translate(c['text']), c['global_indices'])
        for c in chunks
    ]

    final_html = reconstruct_html(
        translated_chunks, tag_map, tp, original_chunks=chunks, bilingual=True,
    )
    root = etree.fromstring("<body></body>")
    replace_body_content(root, final_html)
    out = etree.tostring(root, encoding='unicode', method='xml')

    n_orig = out.count('bilingual-original')
    n_trans = out.count('bilingual-translation')

    # Source-side text volume must equal the full source (no data loss).
    orig_blocks = re.findall(
        r'bilingual-original[^>]*>(.*?)</[a-z:]*div><[a-z:]*div[^>]*bilingual-translation',
        out, re.S,
    )
    orig_side_chars = len(squeeze(re.sub(r'<[^>]+>', '', "".join(orig_blocks))))

    print(f"  paragraphs (source)   : {len(src_paras)}")
    print(f"  chunks                : {len(chunks)}")
    print(f"  bilingual-original    : {n_orig}")
    print(f"  bilingual-translation : {n_trans}")
    print(f"  source text chars     : {src_chars}")
    print(f"  original-side chars   : {orig_side_chars}")

    ok = (
        n_orig == len(src_paras)
        and n_trans == len(src_paras)
        and orig_side_chars == src_chars
    )
    print("  RESULT:", "PASS" if ok else "FAIL")
    return ok


def run_real_file(epub, chapter):
    raw = zipfile.ZipFile(epub).read(chapter).decode("utf-8", "replace")
    doc = etree.fromstring(raw.encode("utf-8"),
                           etree.XMLParser(recover=True, huge_tree=True))
    body_html, _ = extract_body_html(doc)
    return run_case(f"REAL: {os.path.basename(epub)} :: {chapter}", body_html)


def main():
    if len(sys.argv) >= 3:
        ok = run_real_file(sys.argv[1], sys.argv[2])
    else:
        ok = all([
            run_case("Container div (data-loss case)", CONTAINER_BODY),
            run_case("Japanese web-novel layout", JAPANESE_BODY),
        ])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
