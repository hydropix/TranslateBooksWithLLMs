# Term fix

Fix a term in a book that is already translated, without translating it again. You name the wrong renderings you see in the translation and the correct term. Only the sentences holding a wrong rendering change; every other byte of the file is written back as it was.

Typical case: the glossary missed a recurring name or concept, so the translation renders it in two or three different ways across a series. Re-translating five books to fix two words is not an option.

> Available in the Web UI (third drop zone) and the CLI (`--fix-terms`). Formats: TXT, SRT, EPUB, DOCX.

---

## Table of contents

- [Writing the rules](#writing-the-rules)
- [Smart and literal modes](#smart-and-literal-modes)
- [Web UI](#web-ui)
- [CLI](#cli)
- [What changes in the file](#what-changes-in-the-file)
- [The change report](#the-change-report)
- [Limits](#limits)

---

## Writing the rules

One rule per line: the wrong forms found in the translation, separated by `|`, then `->` and the correct term. Blank lines and lines starting with `#` are ignored.

```text
# wrong forms                     -> correct term
Drakbrännaren|Drakbrand*          -> Eldsdraken
Lord Ashe|Lord Ashen              -> Lord Askan
```

- Matching ignores case and respects word boundaries, so `Fan` does not match inside `Fantasy`. Chinese, Japanese and Korean forms match as substrings.
- A trailing `*` matches every word starting with that form: `Drakbrand*` covers `Drakbranden`, `Drakbrandens`, and so on. This is how you catch inflected or suffixed forms without listing each one.
- When two forms overlap, the longer one wins (`Lord Ashen` before `Lord Ashe`).
- A rule whose correct term would itself be matched by a rule (for example `Eld* -> Eldsdraken`) is refused. A second run would otherwise rewrite text that is already correct.

The same rules as JSON (CLI):

```json
{"rules": [{"find": ["Drakbrännaren", "Drakbrand*"], "replace": "Eldsdraken"}]}
```

**Check the matches first.** The scan ("Check matches" in the Web UI, `--fix-terms-scan` in the CLI) counts the matches for every form and shows a few sample sentences, without calling the LLM. A form with 0 matches usually means a variant you have not listed, or a `*` you need.

## Smart and literal modes

| | Smart (default) | Literal |
|---|---|---|
| LLM call | Yes, only for the sentences holding a match | No |
| Inflection (`Eldsdraken` → `Eldsdrakens`) | Yes | No |
| Agreement (`le Gardien est arrivé` → `la Sentinelle est arrivée`) | Yes, in the same clause | No |
| `*` wildcards | Yes | Refused (the inflected ending would be lost) |
| Cost | A few dozen tokens per occurrence | Free |

**Smart mode** sends each sentence holding a match to the LLM, in batches of about a dozen, with the wrong form marked. The model puts the correct term in the form the sentence needs and may adjust the words that agree with it. A smaller, cheaper model is usually enough for this. Every answer goes through a guard before it is accepted:

- the wrong form must be gone and the correct term present;
- placeholders must be untouched;
- nothing may change beyond a few words around the term (no style rewriting).

A rejected answer is retried once on its own. If it fails again, the sentence is left as it was and marked `needs_review` in the report.

**Literal mode** replaces the text directly and keeps the casing of each occurrence (`DRAKBRÄNNAREN` → `ELDSDRAKEN`, a capitalized occurrence at the start of a sentence stays capitalized). Use it for names in languages without inflection, or when you have listed every inflected form yourself.

## Web UI

1. Drop the translated files on **Drop files to fix a term (already translated)**. Several files can be queued; the rules apply to all of them, which suits a series.
2. Check each file's language in the queue (it is the language the file is written in).
3. Write the rules in the **Terms to fix** panel and pick the mode. The panel keeps your rules in this browser between visits.
4. Click **Check matches** to see the counts per form and sample sentences.
5. Start the batch. Each output file comes with its `.term-fix.md` report in the Files tab.

## CLI

```bash
# 1. Count the matches, write nothing
python translate.py -i "book (Swedish).epub" -tl Swedish --fix-terms rules.txt --fix-terms-scan

# 2. Apply with the LLM (smart mode, the default)
python translate.py -i "book (Swedish).epub" -tl Swedish --fix-terms rules.txt --provider openrouter -m google/gemini-2.5-flash

# Literal mode: no LLM call
python translate.py -i "book (Swedish).txt" --fix-terms rules.txt --fix-terms-mode literal
```

`-tl` names the language the file is written in. Without `-o`, the output is `<input> (term-fix).<ext>`, and the report is written next to it as `<output>.term-fix.md`. See [CLI.md](CLI.md#term-fix) for every option.

## What changes in the file

The pass never reconstructs the document. It edits the text in place:

- **TXT and SRT**: only the matched sentences change. Line endings, the byte order mark, cue numbers and timings stay byte-identical.
- **EPUB**: the raw XHTML text between tags is edited. Tags, attributes, `<head>`, `<script>` and `<style>` are never touched, and a chapter without a match is copied into the new EPUB unchanged. The table of contents (`.ncx`) is corrected too.
- **DOCX**: paragraphs (including tables, headers and footers) are edited run by run, so bold, italics and styles stay where they were. When a match spans two runs, the correct term takes the formatting of the first one. A document without a match is copied unchanged.

## The change report

Every run writes `<output>.term-fix.md`. It lists the matches found and changed per rule, then one row per occurrence with its location (line, EPUB chapter or DOCX paragraph), the sentence before and after, and its status:

| Status | Meaning |
|---|---|
| `fixed` | Corrected by the LLM and accepted by the guard |
| `literal` | Replaced in literal mode |
| `needs_review` | The LLM answer failed the guard twice; the sentence was left unchanged. Fix it by hand |
| `skipped` | The job was interrupted before this sentence was processed |

Read the report before you publish, especially with a small model. In testing, `gemma3:12b` fixed articles and past participles reliably but sometimes missed the agreement of an adjective placed after the term.

## Limits

- You have to know the wrong renderings: the pass does not compare against the source text. The scan helps you find the missing forms.
- In an EPUB, a sentence is cut at inline tags such as `<em>`, so agreement is fixed only inside the same text segment.
- Entity-encoded characters inside a term (`&#8217;` for an apostrophe, for example) are not matched by the plain form.
- PDF is not supported.
- OPF metadata (title, description) is not edited.
