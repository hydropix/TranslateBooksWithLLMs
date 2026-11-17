# Styles

Give the model a **reusable writing style** — register, sentence rhythm, imagery, dialogue conventions — extracted from one or more sample books, or written by hand. Assign it in the Translate tab (or under Settings → Translation Options) and it applies to every chunk of a translation and/or refinement pass.

> Web UI only (Styles tab): full CRUD manager over `Custom_Instructions/`, plus an "extract from books" flow backed by an LLM call. Unlike the glossary, there is no CLI flag to select a style preset yet — `translate.py` has no `--style`/`--custom-instruction` argument, even though the loader it would need (`src.utils.custom_instructions.load_custom_instructions`) already exists and is shared with the Web UI.

---

## Table of contents

- [Why use it?](#why-use-it)
- [5-minute quick start](#5-minute-quick-start)
- [The two modes](#the-two-modes)
- [The narrative setting](#the-narrative-setting-context)
- [The dimensions](#the-dimensions)
- [Why instructions stay abstract](#why-instructions-stay-abstract)
- [Sampling reference](#sampling-reference)
- [Auto mode](#auto-mode)
- [The extended YAML format](#the-extended-yaml-format)
- [Manual authoring](#manual-authoring)
- [Editing and the manual-override state](#editing-and-the-manual-override-state)
- [REST API](#rest-api)
- [Troubleshooting](#troubleshooting)

---

## Why use it?

A translation prompt is built fresh for every chunk, so nothing about *how* earlier chunks were phrased carries over on its own. Left alone, a model's register can drift page to page — dry and clipped in chapter 1, suddenly flowery by chapter 10 — simply because nothing is anchoring the voice.

A style preset injects the same short block of instructions into every chunk's prompt: "keep a cynical, world-weary tone", "favor short declarative sentences", and so on. The instructions describe *tendencies* of the writing, never specific words, so the effect holds across a whole book without turning into a repeated tic (see [Why instructions stay abstract](#why-instructions-stay-abstract)).

Two ways to build one:

- **Extract from books.** Drop 1-5 sample files, the server excerpts them and asks the configured LLM to characterize the writing style as a list of rules, you review and check the ones you want.
- **Write by hand.** Skip extraction entirely and type the translation/refinement prose yourself, or hand-edit a generated preset's YAML file directly.

---

## 5-minute quick start

1. **Start the server** and open <http://localhost:5000>.
   ```bash
   python translation_api.py
   ```
2. **Click the "Styles" tab** in the header.
3. **Click "New from books"**. In the modal, leave mode on **"From the source text"**, drop the book you're about to translate (or a few chapters of it), and leave **Total chars: 10000** / **Samples: 6**.
4. **Click "Run extraction"**. After a short LLM call, the candidate rules appear — one block per rule, tagged with its dimension (register, sentence rhythm, imagery, ...) and showing the excerpt that justifies it.
5. **Review the rules.** Blocks flagged with a warning icon (quoting the book, naming specific words, etc.) start unchecked — leave them out or rewrite the instruction abstractly before checking them. The preview panes below update live as you check/uncheck blocks.
6. **Name the style** in the "Style name" field, then **click "Create style"**.
7. **Click the bookmark icon** on the new style's row to assign it to the translation. The icon turns green and the style dropdown on the Translate tab (and under Settings → Translation Options) follows — clicking it again unassigns. You can equally pick the style straight from that dropdown.
8. **Click "Translate"**. The assembled instructions are now injected into every chunk of the translation (and refinement, if you run one).

Total time: 2-3 minutes for the extraction, plus a couple of minutes reviewing the candidate rules.

---

## The two modes

Every preset has a `mode`, chosen once at extraction time (or set directly in the editor):

| Mode | Meaning | When to use it |
|---|---|---|
| `source` | The uploaded books **are** the text about to be translated (or a good sample of it). Rules describe how the translation must read so it stays faithful to the original's voice. | You're translating a specific book and want its own style preserved. |
| `model` | The uploaded books are a **reference author**, unrelated to what you're translating. Rules describe how to imitate that author's style on top of a different source text. | You want a translation of book A to read like author B (e.g. "translate this pulp thriller so it reads like early Hemingway"). |

The two modes change the preamble text wrapped around the rules (see [The extended YAML format](#the-extended-yaml-format)) but never the rule list itself or the abstraction constraint.

### Worked example — `source` mode

You're translating a hardboiled detective novel and want the translation to keep its voice. In the extract modal you upload the EPUB itself, pick **"From the source text"**, and run extraction. A typical accepted rule:

> `register`: Keep a cynical, world-weary tone with dry understatement.

The assembled translation block starts with:

> "Match the following writing style in the translation. These rules describe how the text must read; they never authorize adding, removing, or altering content."

— followed by the rule list. Nothing here asks the model to add or omit content; it only shapes phrasing.

### Worked example — `model` mode

You're translating a plain technical thriller into French but want it to read like a specific stylist. You upload two novels by that author (not the book being translated), pick **"From a reference author"**, and run extraction. A typical accepted rule:

> `sentence_rhythm`: Favor short declarative sentences, broken by an occasional long melancholic clause.

The assembled translation block starts with:

> "Imitate the following writing style, extracted from a reference work chosen as a stylistic model. Apply the rules to the translation while preserving the meaning of the source text exactly."

The extra "preserving the meaning ... exactly" clause exists precisely because in `model` mode the style source and the content source are two different books — the wording has to make clear that only the *voice* travels, not the plot or facts.

---

## The narrative setting (`context`)

### Why it exists

None of the 9 [dimensions](#the-dimensions) describe the *world* a text takes place in — they all describe craft (register, rhythm, imagery, ...), never the period, technology, or social frame the story is set in. That gap has a real, reported failure mode: a fantasy translation came out stylistically right — the register and rhythm rules were followed — but reached for anachronistic modern vocabulary, translating a garment as "crop top". Nothing in a style preset was ever responsible for keeping the model inside the right era's lexicon, because the setting itself was never captured anywhere.

`context` is a short, free-text field that closes that gap: one place to say what era and world the text belongs to, so the model has a reason to reject a modern-sounding word even when it's the most direct dictionary equivalent.

### What to write in it

1 to 3 sentences covering:

- **Period** (or its secondary-world equivalent, e.g. "a pre-industrial secondary world" rather than a real-world date).
- **Technological level** (what does and doesn't exist — gunpowder, electricity, printing).
- **Social and cultural frame** (the kind of social order the characters live under).

Leave out proper nouns, character names, place names, and plot summary — those belong to the story, not the setting, and `context` is never meant to substitute for a glossary or a synopsis.

| | Setting | Why |
|---|---|---|
| ✅ Good | "A late-medieval frontier town under martial law, with no gunpowder weapons and a rigid guild hierarchy." | Period, technology, and social frame in one sentence, no names. |
| ❌ Bad | "The story of young Elira, who defies the Kingdom of Thalvorn to save her village of Brenmoor." | Names a character, a kingdom, and a village, and summarizes plot instead of describing the world's era and technology. |

### The per-mode rule

| Mode | `context` behavior | Rationale |
|---|---|---|
| `source` | The extraction LLM fills it from the sampled passages (mandatory, 1-3 sentences, ≤400 characters requested in the prompt). | The sampled text *is* the book being translated, so its own setting is exactly what the translation must stay inside. |
| `model` | The extraction LLM is instructed to always return `""`, whatever the reference passages describe. | The uploaded books are only a *stylistic* model for an unrelated text (see [The two modes](#the-two-modes)). Filling `context` from them would impose the reference author's era and world onto a text that doesn't take place there — the same reason `model` mode already adds the "preserving the meaning ... exactly" clause. |

In both modes the field stays a plain, editable textarea in the style editor (`#styleEditorContext`) and in the extraction review screen (`#styleExtractContext`): a `model`-mode preset that returns an empty setting can still have one typed in by hand — the restriction is only on what the *extraction call* is allowed to auto-fill, never on what a human can type afterwards.

### How it lands in the prompt

`assemble_instructions(mode, rules, context)` (`src/core/style/assembler.py`) turns `context` into a `## Setting` section, inserted between the mode preamble and the `## Style` section that wraps the rule list, followed by a dedicated guard sentence. This is the real output of that function for `mode="source"` with a non-empty `context`:

```text
Match the following writing style in the translation. These rules describe how the text must read; they never authorize adding, removing, or altering content.

## Setting

A late-medieval frontier town under martial law, with no gunpowder weapons and a rigid guild hierarchy.

Do not use words that belong to a later era or a different technological level than this setting, even when they are the most direct equivalent.

## Style

- Keep a cynical, world-weary tone with dry understatement.
- Favor short declarative sentences, broken by an occasional long melancholic clause.

Treat these rules as tendencies of the writing, not as a vocabulary. Never reuse a fixed set of words, phrases, or images across passages: vary the wording naturally and let each sentence follow from its own content. Whenever a rule and the natural phrasing of a passage conflict, favour the natural phrasing.
```

The refinement block is identical apart from its own mode preamble. **When `context` is empty or blank, the `## Setting` section (and its guard sentence) is omitted entirely** — the output is byte-identical to a preset with no `context` at all, so every preset created before this field existed keeps working unchanged.

### Why it is not lint-checked

Rule instructions go through `lint_instruction()` precisely because naming specific vocabulary in a *rule* turns into a tic repeated every few pages (see [Why instructions stay abstract](#why-instructions-stay-abstract)). `context` is exempt from that pass on purpose: a setting is descriptive by design and is *expected* to name eras and technology levels ("late-medieval", "no gunpowder", "pre-industrial") — the exact words that would trip `lint_instruction`'s checks if they showed up in a rule. Flagging them here would be flagging the field for doing its job. `parse_style_response()` (`src/core/style/extractor.py`) reflects this directly: `context` is parsed, length-capped, and truncated with a warning if it overflows, but it is never passed through `lint_instruction` the way every rule's `instruction` is.

---

## The dimensions

Every rule is tagged with exactly one of these 9 values. The tag is UI/organizational metadata only — it is never written into the assembled prose the model sees.

| Dimension | Captures |
|---|---|
| `register` | Formality, distance, irony, emotional temperature. |
| `narrative_voice` | Person, tense, focalization, narrator presence. |
| `sentence_rhythm` | Length distribution, parataxis vs. subordination, cadence. |
| `lexicon` | Concrete vs. abstract, recurring lexical fields, archaisms. |
| `imagery` | Metaphors, similes, recurring figurative motifs. |
| `dialogue` | Speech tags, orality, idiolects, interruption handling. |
| `punctuation` | Em-dashes, semicolons, ellipses, exclamation frequency. |
| `formatting` | Paragraph length, italics usage, section breaks. |
| `other` | Anything else worth capturing, and the fallback for a dimension label the parser doesn't recognize. |

If the LLM returns a dimension outside this list, the parser coerces it to `other` and appends a warning (`unknown dimension 'X' for rule N (mapped to 'other')`) instead of dropping the rule.

---

## Why instructions stay abstract

**The failure mode this feature is built to avoid**: an instruction that names specific vocabulary doesn't nudge the model toward a *tendency* — it hands it a fixed toolbox, and the model reaches for the same tools in every chunk. A 300-page novel translated with a rule like "use rain, iron, dust, and smoke as recurring images" doesn't read as atmospheric; it reads as a translator who only knows four words, because those exact four nouns resurface every few pages regardless of what the passage is actually about. Requesting variety in the same breath does not help — a rule that enumerates the only allowed items has already deleted the variety it would need.

This is why every instruction must describe **how** the text behaves, never **which words** it uses: no quoted phrases, no example words, no proper nouns, no lexical fields to copy. This constraint is enforced at three separate points:

1. **The extraction prompt** explicitly instructs the LLM to describe properties of the writing, never the words that realize them, and forbids quotes, "for example" markers, and enumerated word lists.
2. **A lint pass** (`src/core/style/lint.py`) scans every returned instruction and flags likely violations — rows with a flag start unchecked in the review table, so an abstraction violation needs a deliberate re-check, not an accidental one.
3. **A guard clause** — the anti-tic guard — is appended verbatim to every assembled `translation`/`refinement` block, telling the model to treat the rules as tendencies, vary its wording, and favor natural phrasing whenever a rule and the passage conflict.

If you hand-author a preset instead of extracting one, the same rule applies: describe the tendency, not the vocabulary.

This abstraction requirement applies only to rule instructions. The separate `context` field is deliberately exempt from all three of the above — see [The narrative setting](#the-narrative-setting-context) for why naming an era or a technology level is exactly what that field is for.

### Rejected vs. accepted

| | Instruction | Flags |
|---|---|---|
| ❌ Rejected | `Repeat rain, iron, dust, smoke to mark tension in every chapter.` | `word_list` |
| ✅ Accepted | `Favor concrete, sensory nouns tied to weather and industry to mark tension, without repeating the same images.` | *(none)* |

Both describe the same underlying instinct — lean on concrete, physical nouns to build tension — but only the second survives contact with a whole book: the model picks *different* concrete nouns each time instead of cycling through the same four.

### The five flag codes

`lint_instruction()` returns zero or more of these codes, in this order. Detection is deliberately conservative (it favors false positives over false negatives): a flagged rule costs the reviewer one click to dismiss, while an unflagged violation bakes a tic into the whole translated book.

| Code | Review-table label | Triggers when |
|---|---|---|
| `quoted_example` | Quotes the book | The instruction contains quoted text — straight quotes, curly quotes, or guillemets — wrapping at least one letter. |
| `example_marker` | Contains a literal example | The instruction uses a marker phrase: "e.g.", "i.e.", "for example", "such as", "for instance", "words/expressions/terms/phrases like". |
| `word_list` | Lists specific words | A comma- or slash-separated run of 3+ items where every item in the run is only 1-2 words long — the shape of an enumerated vocabulary list. |
| `proper_noun` | Names a proper noun | Two or more capitalized words in a row outside a sentence's first word (e.g. an author's full name), or any single ALL-CAPS/CamelCase token. Common language names ("English", "French", ...) and the dimension labels themselves are exempt. |
| `too_specific` | Too specific to this book | The instruction is under 25 characters — too short to state a general tendency (e.g. "Write like Hemingway."). |

What to do about each: rewrite the instruction to describe the effect rather than naming the source of it (drop the quote, drop the author's name, drop the word list, expand the one-liner into an actual description), then leave the rule checked. If you can't rewrite it abstractly, leave it unchecked — an unused rule does nothing; an abstraction-violating rule that got through does damage.

---

## Sampling reference

The extract-style endpoint (`POST /api/custom-instructions/extract-style`) samples every uploaded file the same way the glossary auto-extract does, then splits the character budget evenly across files before making **one** LLM call for the combined text.

| Input | Default | Range / cap | Notes |
|---|---|---|---|
| `files` | — | 1-5 files, each ≤ 100 MB | Accepted extensions: `.txt`, `.srt`, `.epub`, `.docx`, `.pdf` (text layer only). |
| `mode` | `source` | `source` \| `model` | See [The two modes](#the-two-modes). |
| `max_chars` | `10000` | 1-12000 | Values above 12000 are silently clamped to 12000; `0` or negative is a 400 error. |
| `sample_count` | `6` | 1-20 | Outside this range → 400 error (not clamped). |
| `source_lang` / `target_lang` | `English` / `English` | — | Used only to phrase the extraction prompt; not validated against the language list. |

Multi-file budget split (deterministic, no LLM involved):

```text
n_files          = number of uploaded files
per_file_budget  = max_chars // n_files
remainder        = max_chars - per_file_budget * n_files   # added to the first file's budget
per_file_samples = max(1, sample_count // n_files)
MIN_SAMPLE_SIZE  = 1200 characters   # style needs longer passages than glossary NER
```

Each file's excerpts are pulled independently with `extract_samples_from_upload(...)`, then the per-file blocks are concatenated with a header before each one:

```text
===== EXCERPTS FROM: novel_a.epub =====

<excerpts...>

===== EXCERPTS FROM: novel_b.epub =====

<excerpts...>
```

If a file yields no usable text it is skipped with a warning; if a file's budget can't afford `sample_count` excerpts of at least 1200 characters, the excerpt count is reduced (or the whole file is sent if it's shorter than its budget) and a warning is added. If every uploaded file fails, the endpoint returns 400 instead of calling the LLM.

### Progress streaming

Send the form field `stream=1` to get the response as NDJSON (one JSON object per line) instead of a single JSON body. The web UI uses this to show its progress panel. Events, in order:

| `type` | Fields | When |
|---|---|---|
| `stage` | `stage: "sampling"`, `file`, `index`, `total` | Before each file is sampled |
| `file_sampled` | `file`, `skipped`, `sample_count`, `sample_chars`, `full_text_chars` | After each file is sampled |
| `stage` | `stage: "llm"`, `provider`, `model`, `sample_chars`, `sample_count` | Right before the LLM call |
| `log` | `event`, `message` | Provider log lines during the call (retries, rate-limit waits) |
| `stage` | `stage: "parsing"` | After the LLM answered |
| `result` | `data` (the usual JSON body) | Success, last line |
| `error` | `status`, `data` (`{"error": ...}`) | Failure, last line |

Request validation errors (bad mode, unsupported file type, too many files, rejected endpoint) are still returned as a plain JSON 4xx before any streaming starts.

The LLM call itself reuses the provider/model/key already configured for translation (context window fixed at 16384, independent of `OLLAMA_NUM_CTX`) — you don't need a separate key for extraction.

---

## Auto mode

Auto mode skips the whole extract-review-save workflow above. Pick **Auto — match this document's style** in the style dropdown (Translate tab, or Settings → Translation Options) and the server derives a throwaway style block from the document you are about to translate, right before the first chunk is sent — no file upload, no candidate-rules table, no preset saved to `Custom_Instructions/`.

- **Where to select it.** It's a third option in the same dropdown as your saved presets, next to `None`: `Auto — match this document's style`.
- **Cost.** Exactly one extra LLM call, on the job's own provider/model, run before the first chunk. While it runs, the progress header reads **Preparing (auto glossary/style)** and the bar stays at 0% — there is no chunk to advance yet.
- **Nothing is reviewed, nothing is saved.** The assembled instructions live only inside that job (and its checkpoint): resuming an interrupted job reuses the same derived style rather than recomputing it, so the voice doesn't shift partway through a resumed book.
- **An explicit preset always wins.** Select a real, saved style and Auto is ignored for that job.
- **Runs in refine-only mode too**, unlike auto-glossary (see [docs/GLOSSARY.md](GLOSSARY.md#auto-mode)): the already-translated text is sampled directly, with the source language treated as the target language, so the derived rules describe "keep this text's own voice" rather than a source-to-target change.
- **Not available in Quick Test or Sample & Compare.** Both are meant to stay instant; an `Auto` selection there is treated as `None`.
- **The abstraction lint filter still applies.** Only rules the linter leaves unflagged (see [Why instructions stay abstract](#why-instructions-stay-abstract)) are assembled — a flagged rule (quoting the book, listing specific words, too specific to be a tendency, ...) is silently dropped rather than applied, exactly the filter the manual extraction endpoint applies before showing you a preview.
- **On failure or timeout,** the job proceeds with no style instructions injected, and a `⚠️` line appears in the log so you know why nothing was applied.
- **Always `source` mode.** Auto has no way to pick a separate reference author, so it always treats the document being translated as its own style source (see [The two modes](#the-two-modes)).

**The quality tradeoff.** With no review step, a small or unreliable model can return low-quality rules on every single job, silently spending a call's worth of tokens for nothing useful. The lint filter blocks the clearest abstraction violations (quoted text, word lists, proper nouns), but it cannot tell a plausible-sounding bad rule from a good one. If the `⚠️` "no usable style rules found" line shows up often, or the translated prose picks up a stylistic tic you didn't ask for, turn Auto off and use a reviewed preset instead.

---

## The extended YAML format

A style preset is a YAML file under `Custom_Instructions/` with these top-level keys, written in this exact order:

```yaml
# Generated by the Style Extraction tab. Safe to edit by hand.
description: Hardboiled register extracted from 2 books
mode: source                      # "source" | "model"
context: A late-medieval frontier town under martial law, with no gunpowder weapons and a rigid guild hierarchy.
source_files:
  - novel_a.epub
  - novel_b.epub
generated_at: "2026-08-02T14:12:00Z"
rules:
  - dimension: register
    instruction: Keep a cynical, world-weary tone with dry understatement.
  - dimension: sentence_rhythm
    instruction: Favor short declarative sentences, broken by an occasional long melancholic clause.
translation: |-
  Match the following writing style in the translation. These rules describe how the text must read; they never authorize adding, removing, or altering content.

  ## Setting

  A late-medieval frontier town under martial law, with no gunpowder weapons and a rigid guild hierarchy.

  Do not use words that belong to a later era or a different technological level than this setting, even when they are the most direct equivalent.

  ## Style

  - Keep a cynical, world-weary tone with dry understatement.
  - Favor short declarative sentences, broken by an occasional long melancholic clause.

  Treat these rules as tendencies of the writing, not as a vocabulary. Never reuse a fixed set of words, phrases, or images across passages: vary the wording naturally and let each sentence follow from its own content. Whenever a rule and the natural phrasing of a passage conflict, favour the natural phrasing.
refinement: |-
  Polish the already-translated text so that it matches the following writing style. Rewrite phrasing, rhythm and register as needed. Do not re-translate, do not add information, and do not change the meaning.

  ## Setting

  A late-medieval frontier town under martial law, with no gunpowder weapons and a rigid guild hierarchy.

  Do not use words that belong to a later era or a different technological level than this setting, even when they are the most direct equivalent.

  ## Style

  - Keep a cynical, world-weary tone with dry understatement.
  - Favor short declarative sentences, broken by an occasional long melancholic clause.

  Treat these rules as tendencies of the writing, not as a vocabulary. Never reuse a fixed set of words, phrases, or images across passages: vary the wording naturally and let each sentence follow from its own content. Whenever a rule and the natural phrasing of a passage conflict, favour the natural phrasing.
```

What matters at translation time:

- **`translation` and `refinement` are the only keys read when a translation runs.** Everything else (`description`, `mode`, `context`, `source_files`, `generated_at`, `rules`) is metadata for the Styles tab — the loader used by the translation pipeline ignores it entirely.
- **`translation`/`refinement` are produced deterministically from `rules` and `context`**, not by the LLM. The extraction call only returns rules and a setting; a pure template function (`assemble_instructions`) turns the checked rules (and, when non-empty, the setting) into the two prose blocks — a `## Setting`/`## Style` split when `context` is present, or the flat rule list shown in [The two modes](#the-two-modes) when it's empty — wrapped in a mode-specific preamble and the anti-tic guard. This is also what `POST /api/custom-instructions/assemble` does on demand, so the editor's live preview and the saved file can never drift apart. See [The narrative setting](#the-narrative-setting-context) for the exact shape.
- **Either phase can be empty.** A preset with only `translation` set does nothing during a refine-only run, and vice versa — see [Troubleshooting](#troubleshooting).
- **`context` is omitted from the file entirely when empty**, rather than written as a noisy `context: ""` — a preset created before this field existed round-trips unchanged.
- **Unknown keys are preserved.** Saving an edit to a preset that already has extra top-level keys keeps them, appended after the known keys in their original order.

---

## Manual authoring

You don't need to run extraction at all:

- **From scratch in the editor**: click "New blank style" in the Styles tab, type the translation/refinement prose directly (no rules needed), give it a name, and save.
- **By hand outside the app**: drop a `.yaml`/`.yml` file with just a `translation:` and/or `refinement:` key into `Custom_Instructions/` (find the folder via "Open folder" in the Styles tab). It shows up in the list with an empty rule set, and opening it in the editor puts the text areas in "manual" state — see below.
- **Legacy `.txt` presets** still work exactly as before this feature existed: the whole file content is used, verbatim, for *both* `translation` and `refinement`. A `.txt` preset has no metadata (`mode`, `rules`, etc. are always empty) and the editor hides the rules table for it.

---

## Editing and the manual-override state

Opening a preset in the editor loads its description, mode, rule list, and the two assembled text areas. From there, two things can happen to the `translation`/`refinement` text:

- **Editing a rule** — its instruction text, its dimension, adding/removing a row, switching the mode, or editing the narrative setting (`#styleEditorContext`) — automatically re-runs the assembly (debounced ~300 ms) and overwrites both text areas with the freshly assembled prose. This is the default, "in sync with rules" state. (In this editor, unlike the extract modal, every rule row has no checkbox — every listed rule is always included.)
- **Typing directly into the "Translation instructions" or "Refinement instructions" text area** immediately disconnects that preset from its rules: a **"Manually edited"** badge appears, and further rule (and setting) edits stop auto-updating the text (your typed text is not overwritten). Saving in this state sends `manual: true`, which tells the server to store your prose verbatim and keep the rules and setting only as metadata — it does **not** discard them.
- **"Reassemble from rules"** asks for confirmation, then discards the manual edits, regenerates both text areas from the current rule list and setting, and clears the manual-override state.

The extract-from-books modal has its own, separate instance of the same mechanic: each candidate rule has a checkbox, and only checked, non-empty rules — together with the narrative setting field (`#styleExtractContext`) — feed the live preview and the final "Create style" call. Rows whose instruction was flagged by the lint pass start unchecked (see [Why instructions stay abstract](#why-instructions-stay-abstract)); the setting field is never lint-checked (see [The narrative setting](#the-narrative-setting-context)).

Server-side, the contract is symmetric for every write endpoint (create/update/duplicate): if `rules` is non-empty and `manual` is not `true`, the server **re-assembles** `translation`/`refinement` from the rules and ignores whatever prose the client sent — this is what makes the "in sync" state trustworthy even if a client bug sent stale text alongside fresh rules.

---

## REST API

All endpoints below are mounted at the top level (no `/api/custom-instructions`-blueprint prefix beyond the path shown), implemented in `src/api/blueprints/custom_instruction_routes.py`.

| Method | Path | Body / params | Success | Errors |
|---|---|---|---|---|
| GET | `/api/custom-instructions` | — | `{files: [...], count, status}` — each file entry carries `filename`, `display_name`, `format`, `has_translation`, `has_refinement`, `description`, `mode`, `updated_at` | 200 with `status: "error"` on failure |
| POST | `/api/custom-instructions/open-folder` | — | `{success: true, path}` | 500 |
| GET | `/api/custom-instructions/<filename>` | — | Full preset mapping (`filename`, `display_name`, `format`, `description`, `mode`, `context`, `source_files`, `generated_at`, `rules`, `translation`, `refinement`) | 400 unsafe filename, 404 missing, 422 malformed YAML |
| POST | `/api/custom-instructions` | `{name, description?, mode?, context?, source_files?, rules?, translation?, refinement?, manual?, overwrite?}` | 201 `{filename, display_name}` | 400 validation, 409 already exists (unless `overwrite: true`) |
| PUT | `/api/custom-instructions/<filename>` | Same fields as POST, minus `name` | 200 `{filename, display_name}` | 400, 404 |
| DELETE | `/api/custom-instructions/<filename>` | — | `{deleted: true}` | 400, 404 |
| POST | `/api/custom-instructions/<filename>/duplicate` | `{name?}` | 201 `{filename, display_name}` — default name is `<original>_copy`, `_copy2`, ... | 400, 404, 409 (after 50 exhausted attempts) |
| GET | `/api/custom-instructions/<filename>/export` | — | YAML file download | 400, 404 |
| POST | `/api/custom-instructions/assemble` | `{mode, rules, context?}` | `{translation, refinement, flags}` — `flags[i]` is `lint_instruction(rules[i].instruction)`; `context` is never linted | 400 |
| POST | `/api/custom-instructions/extract-style` | multipart: `files` (1-5), `mode`, `source_lang`, `target_lang`, `max_chars`, `sample_count`, `provider`/`model`/`api_key`/`api_endpoint` | 200 — see [Sampling reference](#sampling-reference) for the request shape and below for the response | 400, 413 (file too large), 429 (rate-limited, with `Retry-After`), 500 |

Write-path validation caps (create/update/duplicate), each violation returning 400 naming the offending field:

| Field | Cap |
|---|---|
| `description` | 300 characters |
| `context` | 600 characters |
| `rules` | 40 entries |
| each rule's `instruction` | 500 characters |
| `translation` / `refinement` | 20000 characters each |
| `source_files` | 10 entries, 260 characters each |

At least one of `translation`/`refinement` must be non-empty after assembly, or the write is rejected with 400.

### Extract-style response shape

```json
{
  "rules": [
    {"dimension": "register", "instruction": "...", "evidence": "...", "flags": []}
  ],
  "summary": "...",
  "suggested_name": "dry_hardboiled_noir",
  "context": "A late-medieval frontier town under martial law, with no gunpowder weapons and a rigid guild hierarchy.",
  "assembled": {"translation": "...", "refinement": "..."},
  "mode": "source",
  "warnings": [],
  "provider": "openrouter",
  "model": "anthropic/claude-4.5-haiku",
  "sample_chars": 9873,
  "per_file": [
    {"filename": "novel_a.epub", "sample_chars": 4937, "sample_count": 3, "full_text_chars": 432110}
  ]
}
```

`assembled` is computed from the **unflagged** rules only, so the first preview shown matches the initial checkbox state (flagged rows start unchecked); `context` feeds `assembled` unfiltered — it carries no `flags` because it is never lint-checked (see [The narrative setting](#the-narrative-setting-context)). In `model` mode, `context` is always `""` (see [The per-mode rule](#the-per-mode-rule)). `evidence` (a short excerpt supporting the rule, shown as a tooltip in the review table) and `flags` never end up in a saved preset — only `dimension` and `instruction` are persisted, alongside `context` itself.

### Assemble example

```bash
curl -X POST http://localhost:5000/api/custom-instructions/assemble \
  -H 'Content-Type: application/json' \
  -d '{
    "mode": "source",
    "rules": [
      {"dimension": "register", "instruction": "Keep a cynical, world-weary tone with dry understatement."}
    ]
  }'
```

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Extraction returns 0 rules | Raise **Total chars**, try more **Samples**, or the sampled passages may simply be too uniform (plain, unadorned prose has fewer distinguishable style traits to name). Check the server log for the exact LLM response if this persists. |
| Style not in the dropdown / list | The file's extension must be `.yaml`, `.yml`, or (legacy) `.txt` — anything else is silently skipped when the list is built. Check the server log for a YAML parse error if the file has the right extension but still doesn't appear. |
| Preset has no visible effect | Confirm which phase the preset actually fills. A preset with only `translation` set does nothing during a refine-only run, and one with only `refinement` does nothing during a translation-only run — check the "Phases" column in the Styles list (`T`, `R`, or `T+R`). |
| Translation uses anachronistic or modern words | The preset's narrative setting is empty, so nothing tells the model which era or technological level the text belongs to. Open the preset in the editor, fill in the "Narrative setting" field (period, technological level, social frame — see [The narrative setting](#the-narrative-setting-context)), save, and the preset re-assembles with a `## Setting` section and its guard sentence. In `source` mode you can also re-run extraction to have the model propose one from the passages; in `model` mode it must be typed by hand. |

For everything else (connection issues, provider errors, context length, etc.), see [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md).
