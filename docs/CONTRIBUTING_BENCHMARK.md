# Contributing benchmark results

The TranslateBookWithLLM benchmark is **community-driven**. You can contribute
results for any model the project doesn't already track by opening a Pull
Request that adds a single JSON file under
[`benchmark/data/submissions/`](../benchmark/data/submissions/).

A GitHub Action validates every PR against the schema and replays a sample of
the results (when the model is replayable from CI). On merge, the wiki is
regenerated automatically.

This page walks you through the full workflow. Submitting a model takes ~30
minutes, almost all of it spent waiting for translation/evaluation calls.

---

## 1. Run the benchmark locally

Prereq: a working Python 3.11+ env with this repo installed (`pip install -r requirements.txt`).

Pick the provider that matches your model and run the benchmark CLI. Examples:

```bash
# Cloud (replayable in CI):
python -m benchmark.cli run \
  --provider openrouter \
  --openrouter-key $OPENROUTER_API_KEY \
  -m anthropic/claude-haiku-4-5

# Local Ollama (will be marked self-reported):
python -m benchmark.cli run \
  --provider ollama \
  -m qwen3:14b

# Specific languages:
python -m benchmark.cli run -p openrouter -m google/gemini-3-flash-preview \
  -l fr de ja zh-Hans
```

The run is saved to `benchmark_results/<run_id>.json`.

> **Tip:** start with a small language subset (`-l fr de ja`) to confirm your
> setup works before launching a long full run.

---

## 2. Convert the run into a submission

```bash
python -m benchmark.cli submit benchmark_results/<run_id>.json \
  --by github:<your-username> \
  --provider openrouter \
  --judge-id google/gemini-3-flash-preview
```

This writes a validated submission file to
`benchmark/data/submissions/<date>_<username>_<model-slug>.json` and prints the
git commands you need next.

The CLI computes `output_hash` for each translation, fills in metadata
(`tbl_version`, `prompt_version`, `judge_id`, …), and validates the JSON
against [`benchmark/schemas/submission.schema.json`](../benchmark/schemas/submission.schema.json)
locally before writing the file.

---

## 3. Open a Pull Request

```bash
git checkout -b submit/<model-slug>
git add benchmark/data/submissions/<date>_<username>_<model-slug>.json
git commit -s -m "submit: <model-id> benchmark results"
gh pr create --title "Submit <model-id> benchmark results"
```

The `Validate Benchmark Submission` workflow will:

1. Run schema validation on every changed submission file.
2. Sample ~10% of results and **replay** them: re-translate with the same
   provider/model and re-evaluate; flag results whose chrF vs. the submitted
   output is < 0.85 **and** whose overall score differs by more than 1 point.
3. For local models (Ollama), do cheap sanity checks instead: detect output
   language, check length plausibility. The submission is then marked
   `self-reported` in the wiki.
4. Post a Markdown report on the PR.

Address any reported issues by force-pushing a corrected file.

---

## 4. After merge

The `Publish Benchmark Wiki` workflow runs on `main` whenever
`benchmark/data/**` changes. It:

1. Aggregates all submissions (`benchmark/cli aggregate-submissions`).
2. Conflicts on `(model, text, target_lang)` are resolved by **median** of the
   four scores; the representative output text is the one closest to the
   median overall score.
3. Regenerates the wiki Markdown and pushes to the GitHub wiki repo.

Each row in the wiki shows the number of observations (`Obs`) and a verified /
self-reported / mixed badge.

---

## File format reference

The full JSON schema is in
[`benchmark/schemas/submission.schema.json`](../benchmark/schemas/submission.schema.json).
A minimal example:

```json
{
  "schema_version": "1.0",
  "submission": {
    "submitted_by": "github:hydropix",
    "submitted_at": "2026-05-09T10:00:00Z"
  },
  "environment": {
    "tbl_version": "v0.1.0",
    "prompt_version": "v1",
    "judge_id": "google/gemini-3-flash-preview"
  },
  "model": {
    "provider": "openrouter",
    "id": "anthropic/claude-haiku-4-5"
  },
  "results": [
    {
      "text_id": "pride_prejudice",
      "source_lang": "en",
      "target_lang": "fr",
      "output": "...",
      "output_hash": "sha256:<64-hex>",
      "scores": {
        "accuracy": 8.5,
        "fluency": 9.0,
        "style": 7.5,
        "overall": 8.3
      }
    }
  ]
}
```

Reference texts and language codes live in
[`benchmark/data/`](../benchmark/data/) — pick `text_id` values from
`benchmark/data/reference_texts/<lang>/*.yaml` and `target_lang` codes from
`benchmark/data/languages/*.yaml`.

---

## FAQ

**Can I submit results for a private model?**
Yes, but it will be marked `self-reported` because CI cannot replay it.

**Two contributors tested the same `(model, text, lang)` — whose result wins?**
Neither: the aggregator takes the median of all observations and shows the
number of observations in the wiki.

**Do submissions expire when a new model version ships?**
Not automatically. The wiki keeps historical entries; we may add a UI filter
later.

**Where do I report bugs in the schema or workflow?**
Open an issue at
[github.com/hydropix/TranslateBookWithLLM/issues](https://github.com/hydropix/TranslateBookWithLLM/issues).
