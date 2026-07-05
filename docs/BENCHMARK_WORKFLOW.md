# Benchmark Workflow — v2 (split layout, canonical judge)

The benchmark v2 produces translations across a fixed set of canonical
language pairs, then judges them with a single canonical evaluator
(Claude Opus 4.7 via Poe) applying [rubric v2](JUDGE_RUBRIC_V2.md).

For new contributors and PR-style submissions, see
[CONTRIBUTING_BENCHMARK.md](CONTRIBUTING_BENCHMARK.md).

---

## Architecture: split layout

```
benchmark/data/
├── translations/
│   ├── claude-haiku-4.5.json        ← one file per model, immutable outputs
│   ├── gemini-3.1-flash-lite.json
│   ├── gemma4-31b.json
│   └── ...
├── judgments/
│   └── claude-opus-4-7-rubric-v2-poe.json   ← single canonical judge file
└── submissions_v1_archive/           ← legacy v1 layout, kept for audit
```

**Why split**: translations are expensive artifacts produced once. Scores
are subjective and may need re-running (better judge, refined rubric,
calibration drift). Splitting them lets each evolve independently.

**Join key**: `(model_id, text_id, target_lang)` with `output_hash` as an
integrity check. A judgment whose hash doesn't match the corresponding
translation triggers a fatal error.

---

## Maintainer skills (`.claude/commands/benchmark-*.md`)

Four slash commands automate the v2 flow inside Claude Code:

| Command | Purpose |
|---|---|
| `/benchmark-test-model <provider> <model-id> [tier]` | Benchmark a new model end-to-end: translate → add to `translations/` → rejudge → commit → wiki |
| `/benchmark-extend-run <model-id> <tier>` | Run an existing model on more pairs (e.g. quick → full), then rejudge only the delta |
| `/benchmark-rescore-submission` | Refresh the canonical judgments file with a new rejudge pass (full or incremental) |
| `/benchmark-publish-wiki` | Aggregate `translations/` × `judgments/` and push the wiki |

Each skill confirms via `AskUserQuestion` before any user-visible action
(commit + push, rejudge, wiki publish). Stop at any "No".

---

## Phase 1 — Produce translations

```bash
python -m benchmark.cli run \
  -p <ollama|poe|openrouter|requesty|openai> \
  -m <model-id> \
  --no-evaluate \
  --pair-set <quick|standard|full>
```

**Critical flag**: `--no-evaluate` — under v2, scoring is centralized in
Phase 2, not in the runner.

Outputs `benchmark_results/<RUN_ID>.json` with `scores: null` on every
result.

Expected volume: quick → ~45 translations, standard → ~125, full → ~245.

---

## Phase 2 — Add translations to the split layout

```bash
python -m benchmark.cli add-translations benchmark_results/<RUN_ID>.json \
  --by github:<your-username> \
  --provider <provider>
```

Merges into `benchmark/data/translations/<model-slug>.json`:

- Computes `output_hash` (sha256) per translation
- Merges new entries with existing ones (most recent wins on conflict)
- Appends contributor to the file's `contributors` list
- Validates against
  [`benchmark/schemas/translations.schema.json`](../benchmark/schemas/translations.schema.json)

---

## Phase 3 — Re-judge with the canonical evaluator

```bash
python scripts/rejudge_all_via_poe.py
```

The script:

1. Loads every `benchmark/data/translations/<slug>.json`
2. Loads `benchmark/data/judgments/claude-opus-4-7-rubric-v2-poe.json`
   if present (resume default)
3. For each translation **not** already scored, calls Poe Claude-Opus-4.7
   at `temperature=0.1, max_tokens=400` with the rubric v2 system prompt
4. Writes the updated judgments file atomically every 10 calls
5. Prints a per-model `old → new` overall delta table

Cost (current Poe rates): ~$0.0068 per call. Full re-judge of ~1043
translations ≈ $8.

To force a complete re-run (e.g. after rubric tweaks):

```bash
python scripts/rejudge_all_via_poe.py --no-resume --yes
```

---

## Phase 4 — Aggregate and publish

```bash
# Build a BenchmarkRun JSON by joining translations × judgments:
python -m benchmark.cli aggregate \
  --judge-id claude-opus-4-7-rubric-v2-poe \
  --run-id aggregated \
  --output benchmark_results/aggregated.json

# Generate the wiki Markdown locally:
python -m benchmark.cli wiki aggregated

# Push to the wiki repo:
/benchmark-publish-wiki     # via Claude Code skill, or do it manually
```

The wiki shows each translation joined to its score under the active
judge. Hash mismatches or orphan scores surface as warnings.

---

## The canonical pair sets

Three fixed tiers in [benchmark/canonical_pairs.py](../benchmark/canonical_pairs.py),
addressable via `--pair-set quick|standard|full`. Unidirectional — we pick
the direction with strongest real-world demand for each language.

### Quick (8 pairs) — default

`en:zh-Hans · en:es · en:fr · en:vi · ja:en · ko:en · zh-Hans:en · ja:zh-Hans`

Rationale:

- `en→zh-Hans`: highest demand (Chinese users importing foreign content)
- `en→es`: 500M+ speakers, baseline
- `en→fr`: quality reference (DeepL excellent → comparable)
- `en→vi`: underserved by mainstream tools, growing
- `ja→en`: manga / light novel community
- `ko→en`: K-literature (+285% in 2024)
- `zh-Hans→en`: Chinese webnovel / academic flow
- `ja→zh-Hans`: documented manga industry flow

### Standard (16 pairs)

Quick + `en→{de, pt, ja, ko, ru, it, ar, hi}`. Covers major Indo-European,
East Asian, Semitic, Indo-Aryan families. The recommended tier for proper
wiki submission.

### Full (28 pairs)

Standard + `en→{nl, pl, sv, da, el, tr, th, id, bn, ta, he}` and
`zh-Hans→ja`. Adds diversity (RTL, agglutinative, rare scripts).

### Custom pairs

For one-off experiments use `--pairs SRC:TGT [SRC:TGT ...]`. Custom sets
don't contribute comparable data to the wiki.

---

## Inspecting state

```bash
# What models do we have translations for?
ls benchmark/data/translations/

# How many translations per model?
for f in benchmark/data/translations/*.json; do
  python -c "import json; d=json.load(open('$f',encoding='utf-8')); print(d['model']['id'], len(d['translations']))"
done

# What's in the current judgments file?
python -c "import json; d=json.load(open('benchmark/data/judgments/claude-opus-4-7-rubric-v2-poe.json',encoding='utf-8')); print('Scores:', len(d['scores']), 'Judge:', d['judge']['id'])"

# Any unjudged translations? (the aggregator reports this)
python -m benchmark.cli aggregate --judge-id claude-opus-4-7-rubric-v2-poe --output /tmp/agg.json
```

---

## Migration from v1

The v1 layout (`benchmark/data/submissions/`) is archived at
`benchmark/data/submissions_v1_archive/`. A one-shot migration script
extracted the translations into the v2 split layout (dropping v1 scores,
which are regenerated by the canonical Opus 4.7 re-judge):

```bash
python scripts/migrate_to_split_layout.py --apply
```

After v2 migration:
- Translations live in `translations/<slug>.json`
- Scores live in `judgments/claude-opus-4-7-rubric-v2-poe.json`
- v1 scripts (`dump_for_rerank.py`, `apply_rerank.py`,
  `submission_to_run.py`, `apply_evaluations.py`,
  `dump_for_evaluation.py`, `validate_submission.py`) are obsolete and
  archived

---

## Conventions

- **One judge in v2.** All score comparisons use the same Opus 4.7 +
  rubric v2 configuration. Multi-judge support is intentionally out of
  scope.
- **`judge_id` format**: `<model-slug>-rubric-v<n>-<provider>`. Current:
  `claude-opus-4-7-rubric-v2-poe`.
- **Bumping the rubric**: create `docs/JUDGE_RUBRIC_V<N+1>.md`, freeze the
  old one, write a new judgments file with the new `judge_id`. v1 rubric
  stays frozen.
- **Translation files are immutable artifacts.** Only `add-translations`
  writes to them. Manual edits are discouraged; if needed, document why
  in `contributors[].notes`.
- **Always confirm before** rejudge `--no-resume`, commit + push, wiki
  publish.
