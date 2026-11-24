# Contributing benchmark results

The TranslateBookWithLLM benchmark v2 is **community-driven**. You can
contribute results for any model the project doesn't already track by
opening a Pull Request that adds your translations to the split layout.

**Two-file split layout:**

- `benchmark/data/translations/<model-slug>.json` — your model's outputs
  (immutable artifact, no scores)
- `benchmark/data/judgments/claude-opus-4-7-rubric-v2-poe.json` — single
  canonical judgment file under rubric v2

A GitHub Action validates every PR against both schemas. On merge, the wiki
is regenerated automatically.

---

## 1. Run the benchmark locally (translations only)

Prereq: a working Python 3.11-3.14 env with this repo installed
(`pip install -r requirements.txt`).

Run the CLI with `--no-evaluate` — under v2, **scoring is centralized**:

```bash
# Cloud (replayable in CI):
python -m benchmark.cli run \
  --provider openrouter \
  --openrouter-key $OPENROUTER_API_KEY \
  -m anthropic/claude-haiku-4-5 \
  --no-evaluate \
  --pair-set quick

# Local Ollama:
python -m benchmark.cli run \
  --provider ollama \
  -m qwen3:14b \
  --no-evaluate \
  --pair-set quick
```

The run is saved to `benchmark_results/<run_id>.json` with translations and
no scores.

Pair sets:
- `quick`    — 8 canonical pairs (~45 translations)
- `standard` — 16 pairs (~125 translations)
- `full`     — 28 pairs (~245 translations)

---

## 2. Add translations to the split layout

```bash
python -m benchmark.cli add-translations benchmark_results/<run_id>.json \
  --by github:<your-username> \
  --provider openrouter
```

This writes/merges into
`benchmark/data/translations/<model-slug>.json`, computing `output_hash` and
validating against
[`benchmark/schemas/translations.schema.json`](../benchmark/schemas/translations.schema.json).

If the model already has a translations file, your entries are merged in
(most-recent wins on `(text_id, target_lang)` conflicts). Your GitHub
identity is appended to the `contributors` list.

---

## 3. Re-judge the new entries

The canonical judge is Opus 4.7 via Poe under
[rubric v2](JUDGE_RUBRIC_V2.md). You need a `POE_API_KEY` in your `.env`.

```bash
python scripts/rejudge_all_via_poe.py
```

The script is idempotent: it loads existing scores from
`benchmark/data/judgments/claude-opus-4-7-rubric-v2-poe.json` and only
judges new entries. Expected cost: ~$0.007 per new translation.

---

## 4. Open a Pull Request

```bash
git checkout -b submit/<model-slug>
git add benchmark/data/translations/<model-slug>.json \
        benchmark/data/judgments/claude-opus-4-7-rubric-v2-poe.json
git commit -s -m "benchmark: add <model-id> (judge: Opus 4.7 rubric v2)"
gh pr create --title "Benchmark: add <model-id>"
```

The validation workflow runs:

1. JSON schema validation on both files.
2. For cloud-provider models, optional re-translation replay (sample
   ~10%) to detect drift.
3. Markdown report posted to the PR.

Address any issues by force-pushing a corrected commit.

---

## 5. After merge

The wiki workflow runs on `main` when `benchmark/data/**` changes:

1. Joins `translations/` × `judgments/<active-judge>.json` via
   `python -m benchmark.cli aggregate`.
2. Joins by `(model_id, text_id, target_lang)` with `output_hash` integrity
   check.
3. Regenerates wiki Markdown, pushes to the GitHub wiki repo.

Each row shows `Obs` (number of contributors) and a verified /
self-reported badge based on provider.

---

## File format reference

### Translations

[`benchmark/schemas/translations.schema.json`](../benchmark/schemas/translations.schema.json).
Minimal example:

```json
{
  "schema_version": "2.0",
  "model": {"provider": "openrouter", "id": "anthropic/claude-haiku-4-5"},
  "environment": {"tbl_version": "v1.2.4", "prompt_version": "v1"},
  "contributors": [
    {"by": "github:hydropix", "at": "2026-05-10T16:01:28Z"}
  ],
  "translations": [
    {
      "text_id": "pride_prejudice",
      "source_lang": "en",
      "target_lang": "fr",
      "output": "...",
      "output_hash": "sha256:<64-hex>",
      "translation_latency_ms": 832,
      "produced_at": "2026-05-10T16:01:28Z"
    }
  ]
}
```

### Judgments

[`benchmark/schemas/judgments.schema.json`](../benchmark/schemas/judgments.schema.json).
Minimal example:

```json
{
  "schema_version": "2.0",
  "judge": {
    "id": "claude-opus-4-7-rubric-v2-poe",
    "model": "Claude-Opus-4.7",
    "rubric_version": "v2",
    "provider": "poe",
    "temperature": 0.1,
    "thinking": "disabled"
  },
  "run": {
    "id": "rejudge_20260511T220000Z",
    "started_at": "2026-05-11T22:00:00Z",
    "completed_at": "2026-05-11T22:15:00Z"
  },
  "scores": [
    {
      "model_id": "anthropic/claude-haiku-4-5",
      "text_id": "pride_prejudice",
      "target_lang": "fr",
      "output_hash": "sha256:<64-hex>",
      "accuracy": 9.0,
      "fluency": 8.5,
      "style": 8.0,
      "overall": 8.3,
      "feedback": "Faithful Austen rendering; ..."
    }
  ]
}
```

Reference texts and language codes live in
[`benchmark/data/`](../benchmark/data/) — pick `text_id` from
`reference_texts/<lang>/*.yaml` and `target_lang` from `languages/*.yaml`.

---

## FAQ

**Why is there only one judgments file?**
v2 uses a single canonical judge (Opus 4.7 rubric v2 via Poe) for
calibration stability across submissions. Multi-judge support is possible
later but not enabled.

**Two contributors test the same `(model, text, lang)` — whose translation
wins?**
The most recent submission wins on output. The contributors list grows to
record both names. Scores are taken from the canonical judge — same input
hash → same score.

**Can I submit a private/local model?**
Yes. It will be marked `self-reported` (no CI replay). The judge can still
score it from the translation text alone.

**My model produced bad translations — can I re-run them?**
Yes. Re-run `python -m benchmark.cli add-translations ...` with the same
model and provider. The merge keeps the most recent entries by date.

**Where to report bugs?**
[github.com/hydropix/TranslateBookWithLLM/issues](https://github.com/hydropix/TranslateBookWithLLM/issues).
