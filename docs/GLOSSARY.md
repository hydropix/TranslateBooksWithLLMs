# Glossary

Force consistent translations of recurring terms (characters, places, organizations, items) across an entire book. The glossary block is injected into the LLM system prompt only for chunks where a term actually appears, so it costs no tokens elsewhere.

Entries may also carry a character gender, which is injected into every chunk so pronouns stay correct even where the source omits the subject. See [Character gender](#character-gender-chinese-japanese-korean-).

> Available in both the Web UI (full CRUD, auto-extract via LLM) and the CLI (`--glossary file.json`).

---

## Table of contents

- [Why use it?](#why-use-it)
- [5-minute quick start](#5-minute-quick-start)
- [End-to-end workflows](#end-to-end-workflows)
- [Handling inflected languages (Russian, German, Polish, ...)](#handling-inflected-languages-russian-german-polish-)
- [Character gender (Chinese, Japanese, Korean, ...)](#character-gender-chinese-japanese-korean-)
- [Correcting a glossary mid-translation](#correcting-a-glossary-mid-translation)
- [File formats](#file-formats)
- [Categories](#categories)
- [Per-chunk filter](#per-chunk-filter)
- [Auto-extract reference](#auto-extract-reference)
- [Auto mode](#auto-mode)
- [Storage and migration](#storage-and-migration)
- [Tuning](#tuning)
- [Troubleshooting](#troubleshooting)
- [REST API](#rest-api)

---

## Why use it?

LLMs translate one chunk at a time and have no memory between chunks. Concretely, here is the same character name across three chapters of a long Chinese novel translated **without** a glossary:

| Chapter | What the LLM produced |
| ------- | --------------------- |
| Ch. 1   | `Li Fanqing`          |
| Ch. 5   | `Li Fan Qing`         |
| Ch. 12  | `Lee Fanqing`         |
| Ch. 20  | `Li Fan-qing`         |

With the glossary entry `李凡青 -> Li Fanqing  [character]`, every chunk that contains `李凡青` gets the rule injected into its prompt, so the output is `Li Fanqing` every single time.

What the LLM actually sees in the system prompt for a matching chunk:

```text
# GLOSSARY - REQUIRED TRANSLATIONS

MANDATORY: use these EXACT translations whenever the source term appears.
Do NOT paraphrase, transliterate differently, or invent alternatives.
Apply each rule consistently every time the term occurs.

  - 李凡青 -> Li Fanqing  [character]
  - 铁宗 -> Iron Sect  [organization]
  - 玄武城 -> Xuanwu City  [location]
```

---

## 5-minute quick start

Goal: build your first glossary, run a translation that uses it, all from the Web UI.

1. **Start the server** and open <http://localhost:5000>.
   ```bash
   python translation_api.py
   ```
2. **Click the "Glossaries" tab** in the header.
3. **Click "New"**, give it a name like `My novel` and pick the source/target languages (e.g. Chinese -> English).
4. **Drag your source file onto the "Auto-extract" button** (or click it and pick a file). Accepted: `.txt`, `.srt`, `.epub`, `.docx`, `.pdf` (text PDFs only). Up to 100 MB.
5. Leave **Total chars: 6000** and **Samples: 10** (good defaults). Click **Extract**.
6. After 30-90 seconds (depending on the model), you get a table of candidates. Each row has a checkbox, the source term as detected, an editable target translation, and a category.
7. **Review and edit**. Common edits: tweak the target translation, switch a category from `other` to `character`, uncheck rows that are obviously wrong.
8. **Click "Add selected"**. The modal closes, a toast confirms `Added N terms`. The terms appear in the editor table.
9. **Switch to the Translate tab**, drop your file in, then **pick your glossary in the dropdown** above the language selector.
10. **Click "Translate"**. The glossary block is now injected into every chunk that contains a matched term.

Total time: about 5 minutes for a first pass, then 15-30 minutes of manual review for a full novel (depends on how many distinct entities the book has).

---

## End-to-end workflows

### Workflow A: translating a long novel from scratch

You have `wuxia_novel.epub`, 600 pages, Chinese to English. You want consistent character/sect/location names.

```text
1. Web UI -> Glossaries tab -> New -> "Wuxia novel", Chinese -> English
2. Auto-extract: drop the EPUB, click Extract
   -> 10 excerpts of 600 chars each, distributed across the book
   -> ~40-60 candidates (typical for the first chapters of a wuxia)
3. Review the candidates table:
   - Fix mistransliterations: 李凡青 -> "Li Fanqing" (not "Li Fan-qing")
   - Set categories: characters, organizations, locations
   - Uncheck rows where source == target (the LLM gave up)
   - Uncheck noise (common nouns, action verbs)
4. Click "Add selected"
5. Run Auto-extract a SECOND time on the same file
   -> picks up entities the first pass missed (the model sees different excerpts)
   -> rows already in the glossary are tagged "(already in glossary)" and unchecked
6. Add the new ones, repeat until extracts return mostly already-known terms
7. Translate tab -> select the glossary -> drop the EPUB -> Translate
```

The auto-extract is non-destructive, so running it 2-3 times on the same book is the recommended flow. Each run samples different parts of the document and surfaces different recurring entities.

### Workflow B: building the glossary in the UI, running the translation in the CLI

Useful when the translation will take hours and you want to run it in a terminal (e.g. on a remote server, or overnight).

```text
1. Build the glossary in the Web UI (Workflow A, steps 1-6)
2. In the editor toolbar, click "Export" -> JSON
   -> downloads "wuxia_novel.json"
3. Move the file next to your CLI script
4. Run:
```

```bash
python translate.py \
  -i wuxia_novel.epub \
  -o wuxia_novel_en.epub \
  -sl Chinese -tl English \
  --provider openrouter \
  --openrouter_api_key sk-or-... \
  -m anthropic/claude-4.5-haiku \
  --glossary wuxia_novel.json
```

The CLI logs `Glossary loaded: 47 terms from wuxia_novel.json` on startup. From then on, every chunk uses the same per-chunk filter as the Web UI.

### Workflow C: importing an existing translator's glossary

Your translation team already keeps a shared CSV (Google Sheets export, etc.).

```csv
source,target,category
Volodymyr,Vladimir,character
Kyiv,Kiev,location
Sich,Sich,organization
```

```text
1. Web UI -> Glossaries tab -> New -> name + language pair
2. Click "Import" in the editor toolbar
3. Pick the .csv file
4. The whole file is loaded as terms (existing terms in the glossary are REPLACED)
```

If you want to merge instead of replace, add the rows manually via **Add row** (one click per row), or call the bulk-add API endpoint (see [REST API](#rest-api)).

### Workflow D: from a draft translation back into a glossary

You've translated the first 3 chapters by hand or with an earlier tool, and want to harvest the entity translations you settled on. Save them as a CSV and import.

```csv
source,target,category
李凡青,Li Fanqing,character
铁宗,Iron Sect,organization
玄武城,Xuanwu City,location
```

Then run auto-extract on the rest of the book; the rows that match what you already have appear flagged `(already in glossary)`, so you only review the new ones.

### Workflow E: translating subtitles with character names

SRT files often have repeated character names across thousands of lines.

```text
1. Glossaries tab -> New -> "Show name", English -> French
2. Auto-extract: drop the .srt of episode 1
   -> typical output: 5-15 character names + a few recurring locations
3. Add selected
4. Translate tab -> drop the .srt -> select glossary -> Translate
5. For episode 2, REUSE the same glossary (no need to rebuild)
```

You can also auto-extract from a few episodes back-to-back to enrich the glossary before translating any of them.

---

## Handling inflected languages (Russian, German, Polish, ...)

Languages with grammatical case, gender, or strong agglutination produce many surface forms of the same noun. A glossary with only the nominative form would miss most occurrences. The `|` separator declares **alternatives**: any of them in the source chunk triggers the same target.

### Russian: declensions

`Москва` (Moscow) appears as `Москве`, `Москвы`, `Москвой`, `Москву` depending on case. One entry covers all of them:

```json
{
  "source": "Москва|Москве|Москвы|Москвой|Москву",
  "target": "Moscou",
  "category": "location"
}
```

What the LLM sees when any of those forms appears in a chunk:

```text
  - Москва, Москве, Москвы, Москвой, Москву -> Moscou  [location]
```

The block presents the alternatives as a comma-separated list, so the model reads them as one entity with several inflected forms.

### German: declension + gender + case

Articles and adjective endings change but the noun stem stays. `Bürgermeister` (mayor) doesn't change much, but `Schwert` (sword) becomes `Schwertes`/`Schwerter`/`Schwertern` across the genitive/nominative-plural/dative-plural cases.

```json
[
  {
    "source": "Schwert|Schwertes|Schwerter|Schwertern",
    "target": "épée",
    "category": "item"
  },
  {
    "source": "Burg|Burgen",
    "target": "château",
    "category": "location"
  }
]
```

For person names that take genitive `-s`, declare both:

```json
{ "source": "Hans|Hansens", "target": "Jean", "category": "character" }
```

### Polish: heavy declension

Polish nouns inflect through 7 cases x 2 numbers = up to 14 forms. List the common ones (or all of them if you want full coverage):

```json
{
  "source": "Warszawa|Warszawy|Warszawie|Warszawę|Warszawą",
  "target": "Varsovie",
  "category": "location"
}
```

### Tips for inflected languages

| Tip                                                | Why                                                                                  |
| -------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Put the most distinctive form first                | The block displays alternatives in declared order, helps the LLM grasp the entity.   |
| Don't worry about overlap with prefixes            | The filter sorts longest-first, so `Moskovskaya` is checked before `Moskva`.         |
| For verbs / adjectives, use `|` per stem           | `красив-` (beautiful) family: list `красивый|красивая|красивое|красивые` etc.        |
| Latin terms ignore diacritics-insensitivity        | If you want `Cafe` to also match `Café`, list both in the source.                    |
| Up to ~50 alternatives per entry is fine           | Beyond that, regex compilation cost adds up; split into multiple entries.            |

### Auto-extract behavior with inflected languages

The NER prompt asks the model to return the **base form** of each entity. After review, you typically expand the source field manually with `|` alternatives. A useful flow:

```text
1. Auto-extract on a Russian text
   -> rows like {source: "Москва", target: "Moscou", category: "location"}
2. Edit the source cell BEFORE clicking "Add selected":
   change "Москва" to "Москва|Москве|Москвы|Москвой|Москву"
3. Add selected
```

Or add the base form first, then edit it later in the editor table by clicking the source cell.

### Target-side inflection: what the model is told about the translation

Everything above is about matching *source* forms. The target you store is still a single string: the citation / dictionary form of the term. For a target language with grammatical case, agreement or derived forms, a running sentence rarely uses that exact form — a literal "use this exact translation" instruction and the target language's grammar are in tension.

For a fixed set of target languages, `build_glossary_block` adds one line right after "Do NOT paraphrase, transliterate differently, or invent alternatives." (it is a single line in the prompt; wrapped here for readability):

```text
Each target above is given in its dictionary form. Apply whatever morphological changes {target_language}
grammar requires — case, number, agreement, derived forms, and the regular stem alternations these
entail. What must never change is the choice of rendering: do not substitute a different translation,
a different transliteration, or a more familiar variant of the term.
```

`{target_language}` is filled in with the job's target language. The line grants the model the target language's *whole* morphology — not just a change of ending: this deliberately covers regular stem alternations (Russian fluid vowels, Finnish consonant gradation, Turkish final-consonant voicing, German plural umlaut, ...) and derived forms (a denominal adjective, e.g. `Muggle` -> `магловский`), which an "inflect the ending, keep the stem" rule would not permit and would in fact misdescribe for several of these languages. What the line locks down instead is the *choice of rendering*: whichever translation or transliteration you put in the glossary must not be swapped for a different one, even a more familiar or more established one the model recalls from training.

The line is only emitted when the job's target language is in `INFLECTED_TARGET_LANGUAGES`, a frozenset defined in `src/core/glossary/inflection.py` (checked via `target_language_is_inflected()`). The inclusion criterion: case/number inflection changes the *written* form of the term, so the citation form stored in the glossary is routinely not the form that appears in a running sentence. That criterion is also why some richly-inflected languages are deliberately excluded — Arabic marks case with unwritten short-vowel diacritics, and Bulgarian/Macedonian lost nominal case declension, so the written term does not change the way it does for the included languages.

If your target language is not in that set — including Chinese, Japanese, Korean, English, French, Spanish, Bulgarian and Arabic — nothing changes: the block is emitted exactly as it was before this instruction existed, with no extra line. Absence from the list is not an error condition, just the previous behavior.

This is a prompt instruction, not a guarantee: it documents what the model is told, not what it will reliably do. For a term that has a well-established competing rendering in the target language, expect the instruction to reduce, not eliminate, occasional reversion to that memorized form.

---

## Character gender (Chinese, Japanese, Korean, ...)

### The problem

Chinese, Japanese, Korean, Turkish and Finnish either don't mark gender the way English and French do, or drop the subject entirely when context makes it obvious. A passage can run for pages of dialogue without naming anyone. The model then has to produce an English pronoun from a source that contains no gender information, and it falls back on its prior: **everyone becomes "he"**.

This is not a cosmetic slip. It can invert the meaning of a whole book — the reporter of [issue #250](https://github.com/hydropix/TranslateBookWithLLM/issues/250) accidentally turned an all-female yuri novel into an all-male yaoi novel.

### How the gender field fixes it

Each glossary entry carries an optional gender: `female`, `male`, or unset. Entries with a gender are collected into a **cast block** that is injected into **every chunk**, regardless of whether the character is named in it:

```text
# CAST - CHARACTER GENDERS

MANDATORY: each entity below has a fixed gender for the entire text.
Use pronouns and gendered agreement matching the gender listed here, even when
the source passage omits the subject or does not mark gender.
Never default to masculine for an entity listed as female, and never switch an
entity's gender between passages.
This list is the reference for the whole work — an entity may be referred to
here by pronoun only in the passage you are translating.

  - 林月 (Lin Yue) — female
  - 李凡 (Li Fan) — male
```

The cast block deliberately breaks the per-chunk filtering rule that governs the rest of the glossary. That is the whole point: the chunk where the gender is *needed* is the chunk where the name is *absent*. A gender injected only when the name appears would help exactly where the model already had the answer.

The gender also appears in the normal glossary block's bracketed hint, next to the category:

```text
  - 林月 -> Lin Yue  [character, female]
```

### Cost and safety

| Aspect | Behavior |
| ------ | -------- |
| Token cost | ~10 tokens per gendered entry, on every chunk. A 40-character cast is roughly 400 tokens per chunk. |
| Cap | 80 gendered entries. Beyond that, the extra entries are dropped and a warning is logged once per job. |
| Existing glossaries | The column starts empty, so no cast block is emitted and prompts stay byte-identical until you fill a gender in. |
| Block stability | The cast block is identical for every chunk of a job, which keeps prompt caching effective and prevents a character from changing gender mid-book. |

### Auto-detection

The auto-extract pass reports a gender for each `character` it finds, based **only on evidence in the sampled text**: gendered pronouns (`她`, `彼女`), gendered forms of address or kinship terms (`姐姐`, `母亲`, `Miss`), or explicit description. It is instructed never to guess from the name itself.

When the sample carries no evidence, the model returns `unknown` and the candidate row is **flagged in the review table** with an amber border. That blank is intentional: a confidently wrong gender ships silently and corrupts the book, whereas a blank is something you notice and fix. Expect to fill in some genders by hand — a 6000-character sample simply doesn't reveal every character's gender.

Values other than `female` and `male` — including `unknown` — are stored as empty, meaning "no gender information".

### Filling genders in efficiently

1. Sort the terms table by **Category** and then filter on `character`.
2. Select several rows with the checkboxes.
3. Use **Set gender** in the bulk bar and click Apply.

Sorting by the **Gender** column groups the blanks together, which makes the remaining work visible at a glance.

### Non-binary and ungendered characters

Only `female` and `male` exist. For a deliberately ambiguous narrator, a genderless entity, or a non-binary character, leave the gender **unset** and use the style preset or custom instructions to state how that character should be referred to. Forcing one of the two values would be worse than leaving the model to the surrounding context.

---

## Correcting a glossary mid-translation

You launched a translation, watched the first few chunks come out, and noticed an entry produces a wrong target. Here is how to fix it without losing progress.

### Case 1: the target translation is wrong

Example: the glossary has `李凡青 -> Li Fan-qing` but you want `Li Fanqing` (no hyphen).

```text
1. Web UI -> Glossaries tab -> open the glossary
2. Find the row in the terms table (use the filter input above the table)
3. Click the target cell, edit "Li Fan-qing" to "Li Fanqing", press Enter
4. Done. The next chunk that contains 李凡青 will use the corrected target.
```

Note that already-translated chunks keep the old target; the fix only applies to chunks that have not been processed yet. To fix already-translated chunks, run a [term fix](TERM_FIX.md) on the output once the run completes: it rewrites only the sentences holding the wrong rendering and inflects the corrected term. You can also restart the translation from a checkpoint.

The same tool covers a term the glossary missed entirely, even across books already translated: see [TERM_FIX.md](TERM_FIX.md).

### Case 2: the source term is too greedy or not greedy enough

Example: you put `Fan -> Fan` to lock a name, but it now matches inside `Fantasy`.

The Latin word-boundary check should normally prevent this (`Fan` followed by `tasy` is not a word match). If you DO see it happening:

```text
1. Open the glossary editor
2. Edit the source: change "Fan" to "Fan Smith" (the full distinctive name)
3. Save
```

If on the contrary the source is too specific and misses inflected variants, switch to the `|` alternatives form:

```text
old: source = "Москва"
new: source = "Москва|Москве|Москвы|Москвой|Москву"
```

### Case 3: the category is wrong

Click the category dropdown in the row, pick the right one. The change is saved on selection. The injected block updates on the next chunk.

### Case 4: a term should be removed

Click the trash icon at the end of the row. Confirm. Done.

### Case 5: bulk corrections

Select multiple rows via the leftmost checkbox (or the header checkbox to select all visible). The toolbar shows bulk actions: **Delete selected**, **Set category**.

For more complex bulk edits, **Export** to JSON, edit in your text editor, then **Import** to replace.

---

## File formats

### JSON

Full glossary with metadata:

```json
{
  "name": "Wuxia novel",
  "source_lang": "Chinese",
  "target_lang": "English",
  "terms": [
    { "source": "李凡青",   "target": "Li Fanqing",  "category": "character", "gender": "male" },
    { "source": "林月",     "target": "Lin Yue",     "category": "character", "gender": "female" },
    { "source": "铁宗",     "target": "Iron Sect",   "category": "organization" },
    { "source": "玄武城",   "target": "Xuanwu City", "category": "location" }
  ]
}
```

Minimal (bare list of terms):

```json
[
  { "source": "Москва|Москве|Москвы|Москвой", "target": "Moscou", "category": "location" },
  { "source": "Иван",                         "target": "Ivan",   "category": "character" }
]
```

### CSV

Header row required. `source` and `target` are mandatory; `category` and `gender` are optional.

```csv
source,target,category,gender
李凡青,Li Fanqing,character,male
林月,Lin Yue,character,female
铁宗,Iron Sect,organization,
玄武城,Xuanwu City,location,
"Москва|Москве|Москвы|Москвой",Moscou,location,
```

A file without a `gender` column imports exactly as before.

UTF-8 BOM and quoted values are supported. Use quotes around any source that contains commas (rare with `|` separator, but possible).

### Field aliases

The parser accepts both new and legacy field names:

| Field      | Aliases                       |
| ---------- | ----------------------------- |
| `source`   | `source_term`                 |
| `target`   | `translated_term`, `translation` |
| `category` | `type`                        |
| `gender`   | `sex`                         |

So old exports from other tools should import without renaming.

`gender` values are normalized on import, so `F`, `f`, `Female`, `woman` and `feminine` all become `female`. Anything unrecognized — including `unknown`, `other` and `nonbinary` — is stored as empty.

---

## Categories

Optional metadata that appears as a bracketed hint after the arrow in the injected block:

```text
- Li Fanqing -> Li Fanqing  [character]
- Iron Sect -> Iron Sect   [organization]
```

The LLM is instructed to use the hint **only to disambiguate**, not to include it in the output. Recommended values:

| Category       | Use for                                  | Example                          |
| -------------- | ---------------------------------------- | -------------------------------- |
| `character`    | Person names                             | `Li Fanqing`, `Hans`, `Naruto`   |
| `location`     | Cities, regions, named places            | `Xuanwu City`, `Moscou`, `Konoha` |
| `organization` | Sects, factions, companies, governments  | `Iron Sect`, `Akatsuki`          |
| `item`         | Weapons, artifacts, consumables          | `Excalibur`, `Sharingan`         |
| `title`        | Honorifics, ranks, social titles         | `Sect Leader`, `Hokage`          |
| `other`        | Anything else                            |                                  |

Unknown categories are accepted but the UI may flag them. The auto-extract output is always one of the values above.

Entries in the `character` category are the ones that benefit from a [gender](#character-gender-chinese-japanese-korean-).

---

## Per-chunk filter

The filter scans each translation chunk and only injects the entries that actually appear. This keeps the prompt small and keeps the model focused.

> The cast block is the one exception: gendered entries are injected into every chunk, filter or no filter. See [Character gender](#character-gender-chinese-japanese-korean-).

| Behavior                        | Detail                                                                   |
| ------------------------------- | ------------------------------------------------------------------------ |
| Latin terms                     | Word-boundary match. `Fan` does not match `Fantasy`.                     |
| CJK terms                       | Substring match (no word-boundary concept in CJK scripts).               |
| Sort order in the block         | Longest source first, so `Li Fanqing` is checked before `Li Fan`.        |
| Cap                             | 50 entries per chunk by default. Most frequent kept on overflow.         |
| Case sensitivity                | Case-sensitive by default.                                               |
| Inflected forms                 | Sources with `|` are split into alternatives; ANY match triggers the entry. |

### Cap selection on overflow

If a chunk matches more than 50 entries (rare but possible in dense reference works), the filter:

1. Ranks matched entries by occurrence count in the chunk (most frequent first).
2. Tiebreaks by source length (longer wins).
3. Trims to 50.
4. Re-sorts the kept set by length descending for a stable injected block.

This means the most useful entries (frequent, long, specific) survive the cap.

---

## Auto-extract reference

The Web UI's **Auto-extract** button asks the configured LLM to scan a source file and propose recurring entities. Nothing is persisted until you review and click **Add selected**.

### Inputs

| Input         | Default | Cap   | Meaning                                                                  |
| ------------- | ------- | ----- | ------------------------------------------------------------------------ |
| File          |         | 100MB | `.txt`, `.srt`, `.epub`, `.docx`, `.pdf`                                 |
| Total chars   | 6000    | 6000  | How much text the LLM sees, in characters.                               |
| Samples       | 10      | 50    | Number of evenly-spaced excerpts. `1` means "first N chars".             |

### How sampling works

For long files, sending the full text would blow past the model's context. Instead:

1. Read up to 5M chars from the file (cap on memory).
2. Divide into `Samples` evenly-spaced excerpts whose total length equals `Total chars`.
3. Snap each excerpt's edges to nearby whitespace to avoid cutting mid-word.
4. Join with a `[...]` separator so the model sees a discontinuity hint.

Each excerpt must be at least 500 chars; if the budget cannot afford that many samples, the count is reduced and a warning is shown above the candidates table.

### How extraction works

The provider/model configured for translation is reused (you don't need a separate API key). It is called with a fixed 8K context window and asked to return a JSON list of `{source, target, category, gender}` candidates inside `<NER_JSON>...</NER_JSON>` tags.

The parser is permissive: if the tags are missing it falls back to markdown-fenced JSON, then balanced arrays, then balanced objects. Reasoning models' `<think>` blocks are stripped before parsing. A missing or unrecognized `gender` is treated as unknown, so smaller models that ignore the field degrade gracefully.

### Review flow

- Each candidate appears with an editable target field, a category dropdown and a gender dropdown.
- A `character` whose gender the sample did not reveal gets an amber border on its gender dropdown — set it by hand, or the model may default to "he" for that character across the whole book.
- Rows whose source equals the proposed target (the LLM gave up translating, often a proper noun kept as-is) are dimmed and unchecked by default. Check them if you want the entry anyway (useful for locking proper nouns).
- Rows whose source already exists in the glossary are tagged `(already in glossary)` and unchecked by default.
- Edit any field before clicking **Add selected**. Common edits: fix transliteration, expand `source` with `|` alternatives, switch category.
- **Add selected** sends the whole batch in one transaction. A toast reports `Added N (M already existed, skipped)`.

### Re-running on the same file

Each run samples different excerpts (the LLM sees different parts of the document each time), so running auto-extract 2-3 times on a long book is the recommended flow. The second run typically surfaces 30-50% new entities. Stop when the table is mostly `(already in glossary)`.

---

## Auto mode

Auto mode skips the whole build-review-select workflow above. Pick **Auto — build from this document** in the glossary dropdown (Translate tab) and the server derives a throwaway glossary from the document you are about to translate, right before the first chunk is sent — no separate upload, no candidates table, no "Add selected" step.

- **Where to select it.** It's a third option in the same dropdown as your saved glossaries, next to `None`: `Auto — build from this document`.
- **Cost.** Exactly one extra LLM call, on the job's own provider/model, run before the first chunk. While it runs, the progress header reads **Preparing (auto glossary/style)** and the bar stays at 0% — there is no chunk to advance yet. The log names the passes as they start and reports what each one found.
- **Nothing is reviewed, nothing is saved.** The derived terms live only inside that job (and its checkpoint): resuming an interrupted job reuses the same derived glossary rather than recomputing it, so a resumed translation never disagrees with itself mid-book.
- **An explicit glossary always wins.** Select a real, saved glossary and Auto is ignored for that job.
- **Skipped in refine-only mode.** The input is already in the target language, so a source-to-target term pass on it would produce meaningless identity pairs. (Auto-style, see [docs/STYLE_EXTRACTION.md](STYLE_EXTRACTION.md#auto-mode), still runs in refine-only mode.)
- **Not available in Quick Test or Sample & Compare.** Both are meant to stay instant; an `Auto` selection there is treated as `None`.
- **On failure or timeout,** the job proceeds with no glossary injected, and a `⚠️` line appears in the log so you know why nothing was applied.

**The quality tradeoff.** This is the same NER pass as [Auto-extract reference](#auto-extract-reference), minus the review step. Unreviewed candidates go straight into the prompt, so a wrong or garbled term translation lands in *every* chunk that contains it, not just one. Two guards limit the damage: identity pairs (source equals target) are dropped, and the injected list is capped at 60 terms — but nothing caps how wrong a kept term can be. If the log shows the `⚠️` "no usable terms" line often, or you spot a wrong name propagating across the book, turn Auto off and build a reviewed glossary instead.

---

## Storage and migration

The Web UI stores glossaries in a dedicated SQLite file:

```text
data/glossaries.db
```

Tables:

- `glossaries` (id, name, source_language, target_language, timestamps)
- `glossary_terms` (id, glossary_id, source_term, translated_term, category, gender) with `UNIQUE(glossary_id, source_term)`

`gender` is added by an idempotent additive migration on startup. Databases created before it open normally with every gender NULL, so upgrading changes no prompt until you fill a gender in.

### Migration from older builds

Earlier versions stored glossaries inside `data/jobs.db` alongside translation checkpoints, which caused write-lock contention with running translations. On first launch with the new build, glossary tables are copied from `jobs.db` into `glossaries.db` automatically and idempotently. Look for this line in the server log on first start:

```text
Migrated 4 glossaries (87 terms) from data/jobs.db to data/glossaries.db
```

The legacy rows are not deleted from `jobs.db`, so rollback is safe. The migration runs only when `glossaries.db` is empty.

### Backup

```bash
cp data/glossaries.db data/glossaries.db.bak
```

Or export each glossary to JSON via the UI for human-readable backups.

The CLI does not touch the database at all; it loads JSON/CSV files directly from disk.

---

## Tuning

| Knob                         | Default | Where                                            |
| ---------------------------- | ------- | ------------------------------------------------ |
| Per-chunk cap                | 50      | `GlossaryConfig.max_entries`                     |
| Cast block cap               | 80      | `GlossaryConfig.max_cast_entries`                |
| Case sensitivity             | True    | `GlossaryConfig.case_sensitive`                  |
| Auto-extract sample budget   | 6000    | UI field, hard cap also 6000                     |
| Auto-extract sample count    | 10      | UI field (1..50)                                 |
| NER context window           | 8192    | Fixed in the route, independent of OLLAMA_NUM_CTX |

---

## Troubleshooting

| Symptom                                              | Cause / fix                                                                                            |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `database is locked` errors during glossary writes   | Confirm only one `translation_api.py` process is running. WAL mode lets readers and writers coexist within a process, but multiple processes still serialize at the file level. Run `Get-Process python` (Windows) or `pgrep -fa translation_api.py` (macOS/Linux) to check. |
| Auto-extract returns 0 candidates                    | Try increasing **Samples** to 15-20. Some literary excerpts genuinely have no recurring named entities (descriptive passages, dialogue between unnamed characters). |
| Term in glossary but not used in output              | Word boundary issue: `Fan` does not match `Fantasy`. For inflected languages, declare alternatives with `|`. |
| Added 50 terms but only some show up in translations | The per-chunk filter only injects entries that actually appear in that chunk. The cap of 50 is per chunk, not total. A 100-term glossary works fine if only 10-20 match any given chunk. |
| Auto-extract returns identical pairs (source = target) | Some sources are not translated (proper nouns kept as-is). The UI dims them; check or uncheck case-by-case based on whether you want to lock the proper noun. |
| Glossary dropdown empty in the Translate tab         | The dropdown only lists glossaries whose `source_lang`/`target_lang` match the selected language pair. Switch the language pair in the Translate tab or relax the glossary's language fields. |
| CLI says `Glossary file ... contained no usable entries` | JSON has wrong structure: bare list of `{source, target}` or full object with `terms` field. Verify with `python -m json.tool < glossary.json`. |
| Server log: `Migrated N glossaries (M terms)`        | First-run migration from the legacy `jobs.db`. Expected; idempotent on subsequent starts. |

---

## REST API

For programmatic access, the Web UI exposes REST endpoints under `/api/glossaries`:

| Method | Path                                            | Action                                                     |
| ------ | ----------------------------------------------- | ---------------------------------------------------------- |
| GET    | `/api/glossaries`                               | List glossaries with term counts                           |
| POST   | `/api/glossaries`                               | Create a glossary                                          |
| GET    | `/api/glossaries/<gid>`                         | Read a glossary with all its terms                         |
| PUT    | `/api/glossaries/<gid>`                         | Patch glossary fields                                      |
| DELETE | `/api/glossaries/<gid>`                         | Delete (cascade to terms)                                  |
| POST   | `/api/glossaries/<gid>/duplicate`               | Clone a glossary                                           |
| POST   | `/api/glossaries/<gid>/terms`                   | Add a single term                                          |
| PUT    | `/api/glossaries/<gid>/terms/<tid>`             | Patch a term                                               |
| DELETE | `/api/glossaries/<gid>/terms/<tid>`             | Delete a term                                              |
| POST   | `/api/glossaries/<gid>/terms/bulk`              | Bulk action: `add`, `delete`, `set_category`, `set_gender` |
| POST   | `/api/glossaries/<gid>/import`                  | Replace all terms from JSON or CSV                         |
| GET    | `/api/glossaries/<gid>/export?format=json\|csv` | Download                                                   |
| POST   | `/api/glossaries/<gid>/preview-block`           | Render the injected block for a sample chunk               |
| POST   | `/api/glossaries/<gid>/suggest-terms`           | NER auto-extract                                           |

### Bulk add example

```bash
curl -X POST http://localhost:5000/api/glossaries/1/terms/bulk \
  -H 'Content-Type: application/json' \
  -d '{
    "action": "add",
    "terms": [
      {"source": "Москва|Москве|Москвы", "target": "Moscou", "category": "location"},
      {"source": "Иван",                  "target": "Ivan",   "category": "character"}
    ]
  }'
```

Response:

```json
{ "added": 2, "conflicts": 0, "skipped_empty": 0, "total_input": 2 }
```

### Preview block example

Useful when authoring a glossary and wanting to see what the LLM will actually receive for a given chunk:

```bash
curl -X POST http://localhost:5000/api/glossaries/1/preview-block \
  -H 'Content-Type: application/json' \
  -d '{"text": "Сегодня в Москве идёт снег."}'
```

Response includes the rendered block, the matched count, the total terms in the glossary, and whether the cap was hit.
