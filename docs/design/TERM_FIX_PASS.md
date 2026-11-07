# Design: term-fix pass on already-translated books

Status: implemented (phases 1 to 5). See "Decisions taken during implementation" at the end for where the result differs from this proposal.
Origin: discussion hydropix/TranslateBooksWithLLMs#321 ("Third run for replacing words that was missed in glossary").

## Problem

A user translated a five-book series. The auto-extracted glossary missed at least two recurring terms, so each book renders them inconsistently. Re-translating is too expensive. They want to declare the correct rendering after the fact and have it applied throughout the translated text, in the correct inflected form, ideally by a cheap model.

### Why the existing tools do not cover it

- **Refine-only mode with a glossary.** `refine_chunk` filters the glossary against the draft, which is already in the target language (`src/core/translator.py:602`). A missed term was translated inconsistently, so its source form never appears in the draft and the entry is never injected. Even when it is injected, refinement rewrites the style of every chunk. That costs a full pass and changes far more than the user asked for.
- **Search and replace.** `docs/GLOSSARY.md` currently recommends it for already-translated chunks. It cannot inflect (Russian cases, Swedish definite suffixes, German declension) or fix agreement (French `le`/`la` when the gender of the new term differs).

## Goals

1. Fix only the occurrences of the wrong renderings. Every other byte of the book stays untouched.
2. Inflect the replacement correctly and adjust immediate agreement (articles, adjectives, pronouns in the same clause).
3. Keep the cost a small fraction of a refine pass. The LLM sees only short snippets around the matches, never whole chapters.
4. Let the user review the changes: a scan preview before the run and a change report after it.
5. Support the formats refine-only already supports: TXT, EPUB, SRT, DOCX.

## Non-goals (v1)

- Automatically finding the wrong renderings from the source text. This would need the original file plus a source-to-target alignment. The user has the translated books in front of them and can name the variants they see.
- Free-form style edits. The prompt and the guard both forbid them.

## User input: correction rules

One rule per line, reusing the glossary's `|` alternatives convention:

```text
Drakbrand*|Drakenbrännaren -> Eldsdraken
Lord Ashen|Lord Ashe -> Lord Askan
```

- The left side lists the wrong renderings found in the translation. A trailing `*` turns an alternative into a prefix match so it covers inflected or suffixed forms (`Drakbrand*` matches `Drakbranden`, `Drakbrandens`).
- The right side is the correct citation form.
- Matching is case-insensitive by default. Latin alternatives use word boundaries, and CJK alternatives use substring matching, the same as `src/core/glossary/filter.py`.
- JSON form for the API and CLI: `{"rules": [{"find": ["Drakbrand*", "Drakenbrännaren"], "replace": "Eldsdraken"}], "mode": "smart"}`.

Optional later extension: a `source` field per rule, so that after the run the term can be added to a chosen glossary and the next book in the series gets it right on the first pass.

## Pipeline

```text
load + chunk (existing refiner code)
  -> scan: find match spans per chunk (skip placeholders)
  -> no match in chunk: pass through unchanged, no LLM call
  -> match: cut a sentence-level snippet around each match, merge overlaps
  -> correct snippets (literal or smart mode), validate with the guard
  -> splice corrected snippets back into the chunk text, end to start
  -> reassemble + write (existing refiner code), write the change report
```

### Scan

New `matcher.find_spans(text, rules) -> list[Span]` returns `(start, end, rule_index, matched_text)`. It reuses the boundary and CJK logic from `filter.py` but returns positions instead of counts, and it never matches inside a placeholder token (`[[12]]` etc.). Longest alternative wins on overlap, as in the glossary filter.

### Snippets

For each span, extend to the enclosing sentence: split on `.!?…。！？` followed by whitespace, or on a newline. Cap the length at about 400 characters and clamp to word boundaries when a sentence is longer. Snippets that overlap or touch are merged, so two matches in one sentence make one request item. Placeholders inside a snippet travel with it unchanged.

Sentence snippets rather than whole chunks are the key cost lever. A term that appears once in a 1,500-token chunk costs about 60 output tokens instead of 1,500, and the model physically cannot touch the rest of the chunk.

### Correction modes

- **literal**: deterministic replacement with case preservation (`DRAKBRAND` -> `ELDSDRAKEN`, `drakbrand` -> `eldsdraken`). No LLM, no cost. It is offered only when no alternative uses `*` and the user opts in. The UI warns that agreement will not be adjusted.
- **smart** (default): snippets go to the LLM in numbered batches of about 20 per request. Each occurrence is marked in the snippet, for example `⟦Drakbranden⟧`. The prompt (new `generate_term_fix_prompt` in `src/prompts/prompts.py`) says: replace each marked span with the correct term in the grammatically correct form for this sentence; adjust only words that must agree with it in the same clause; change nothing else; keep every placeholder; return the snippets with the same ids and no markers. The target-language inflection hint from `src/core/glossary/inflection.py` is reused.

The user's normal model selector applies, so a small model can be picked for this job. Cloud providers can use `parallel_workers` across batches.

### Guard (per snippet)

An item is accepted only when all of these hold:

1. No wrong alternative from its rule still matches.
2. The placeholder multiset is identical to the input (reuse `validate_placeholders`).
3. Outside a window of ±3 words around each original match, the text is identical to the input after whitespace normalization. This allows agreement fixes and forbids rewrites.
4. The length ratio stays within 0.5 to 2.0.

A rejected item is retried once, alone. If it fails again, the original snippet is kept and the item is flagged in the report as `needs_review`. A rate-limit error propagates as in the refiners, after the partial output is persisted.

### Change report

Written next to the output as `<output_stem>.term-fix.md`. It has one row per occurrence with its location (file/chapter or cue number), before, after and status (`fixed`, `literal`, `needs_review`, `skipped`). It ends with a totals line per rule. The same totals go to the job log.

### Scan preview (no LLM)

`POST /api/term-fix/scan` with the file path and rules returns, per rule and per alternative, the match count and up to five sample contexts. The UI shows it before launch, so the user can spot a variant they forgot (for example, `Drakbranden` matched 0 times because the rule lacked `*`). It also feeds the cost estimate: number of snippets × average snippet tokens.

## Format integration

Rather than four new format modules, refactor the refiners so the per-chunk transform is injectable. Today each refiner hardcodes its LLM refine call (`refine_chunks`, `_refine_epub_chunks`, `refine_subtitle_translations`). Each refiner would accept a `transform` coroutine taking `(chunk_texts, placeholder_format) -> chunk_texts`, with the current refine call as the default. The term-fix pass supplies its own transform. Loading, chunking, tag preservation, reassembly and partial-save on rate limit are all reused.

Per-format notes:

- **TXT**: straightforward.
- **EPUB**: `refine_epub_file` rewrites every XHTML with `tree.write`. In term-fix mode, a content file with zero matches must not be parsed or rewritten, so it stays byte-identical inside the repackaged EPUB. Files with matches go through the usual tag-preservation round trip.
- **SRT**: each cue is a chunk. Timestamps and indices are already preserved verbatim.
- **DOCX**: the refiner regenerates the whole document from HTML (`DocxHtmlConverter.from_html`), which may lose formatting even in untouched paragraphs. This has to be verified first. If the round trip is lossy, v1 should edit text runs in place with python-docx for DOCX, or ship without DOCX support.

New entry point `src/core/adapters/term_fix_file.py` mirrors `refine_file()`: it detects the format and dispatches to the refiner with the term-fix transform.

## API, UI, CLI

- **API**: job config gains `term_fix: {rules, mode}` and `operation: "term_fix"`, which is mutually exclusive with `refine_only` and `refine_after`. `handlers.py` dispatches to `term_fix_file()` next to the existing `refine_only` branch. Auto-glossary and auto-style are skipped, as in refine-only.
- **Web UI**: the existing refine drop zone (`fileUploadRefine`) gains a mode selector, "Polish style" (current behavior) or "Fix terms". "Fix terms" shows the rules textarea, the literal/smart toggle and a Scan button that renders the preview. The rules apply to every queued file, which suits the five-book use case. All strings go through `data-i18n` / `t()` with keys in all 7 locales.
- **CLI**: `translate.py --fix-terms rules.txt -i book.epub -o book.fixed.epub --target-lang Swedish [--fix-terms-literal]`, plus `--fix-terms-scan` for the preview only.

## Module layout

```text
src/core/term_fix/
  __init__.py
  rules.py       parse text/JSON rules -> TermRule(alternatives, replacement)
  matcher.py     find_spans(), placeholder-aware
  snippets.py    sentence windows, merge, splice back
  literal.py     case-preserving deterministic replacement
  corrector.py   batching, LLM call, guard, retry
  report.py      markdown change report
src/core/adapters/term_fix_file.py
src/prompts/prompts.py           generate_term_fix_prompt
src/api/blueprints/term_fix_routes.py   /api/term-fix/scan
```

## Tests

- `rules`: parsing of the `|`, `*` and `->` syntax, rejection of empty sides and of a replacement that is itself matched by its own rule.
- `matcher`: word boundaries, prefix wildcard, CJK substrings, never inside placeholders, longest alternative wins.
- `snippets`: sentence cutting, overlap merge, and a splice round trip that is byte-identical when the corrections are identity.
- `literal`: case preservation.
- `corrector` with a fake LLM client: accepted item, guard rejection then retry, fallback to `needs_review`, malformed batch response.
- End to end with a fake client on TXT, EPUB and SRT. For EPUB, a content file without matches must be byte-identical after repackaging.

## Phases

1. Rules, matcher, snippets, literal mode, report, TXT adapter and CLI. This already gives a free search-and-replace with a report for uninflected targets.
2. Smart mode: prompt, batching, guard, retry.
3. Refiner transform refactor, then the EPUB and SRT adapters. DOCX after the round-trip check.
4. Scan endpoint, web UI and i18n.
5. Docs: a new section in `docs/GLOSSARY.md` replacing the search-and-replace advice, then a reply on discussion #321.

## Decisions taken during implementation

The sections above are the original proposal. Where the implementation differs, this section wins. User documentation: `docs/TERM_FIX.md`.

1. **Third drop zone instead of a mode in the refine zone.** A term fix needs its own inputs (rules, mode, scan), so mixing them with the refine options would have made both harder to read. The zone queues files with the `term_fix` operation, and a rules panel appears while such files are queued. The job reuses the refine-only plumbing (single language, single-phase progress), with a `term_fix` stats flag so the label reads "Fixing terms".
2. **No refiner refactor.** The refiners all go through tag preservation and an HTML round trip, which would rewrite untouched bytes. The pass edits each format directly instead (`src/core/adapters/term_fix_file.py`):
   - TXT and SRT: the raw file is a single unit. Cue numbers and timings cannot match a term, so SRT needs no parsing.
   - EPUB: text segments between tags in the raw XHTML/NCX entries (`src/core/term_fix/formats.py`). Untouched zip entries are copied with their original `ZipInfo`.
   - DOCX: python-docx paragraphs, with edits distributed over the runs. This answers open question 3: the HTML round trip is avoided entirely.
3. **Snippet only, no neighbor sentence.** Testing on `gemma3:12b` showed the sentence alone is enough for inflection and article agreement. A neighbor would add cost and give the model more text it might rewrite.
4. **Minimal edits.** An accepted snippet is turned into token-level edits (`src/core/term_fix/diffing.py`), so only the changed words are spliced back. This is what keeps DOCX runs and untouched words byte-identical.
5. **Item tag `<snippet>`, not `<s>`.** `</s>` is an end-of-sequence token for many models, and Ollama stopped generating at the first one. The parser also tolerates a missing closing tag, strips leftover markers, and restores the snippet's own edge whitespace.
6. **CLI default is smart.** `--fix-terms-mode literal` gives the free, no-LLM path. The scan never calls the LLM.

Known limits: a sentence in an EPUB is cut at inline tags; entity-encoded characters inside a term are not matched; OPF metadata is not edited; PDF is unsupported.
