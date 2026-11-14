# Progress-system characterization tests

Safety net for the progress system. These tests pin the observable behavior
of the translation/refine progress machinery so the implementation can change
without silently changing what the UI sees.

They are *characterization* tests: they document what the system does, so
every change to it is an explicit, reviewed golden update.

## What is captured

For each format (`txt`, `srt`, `docx`, `epub`) and each mode (`translate`,
`refine`), the golden file records:

- **`sequence`** — the ordered list of `stats` dicts passed to
  `stats_callback`, projected onto a stable key set (time/cost/ETA fields are
  stripped, floats rounded). This is the progress signal the engine emits
  (every engine builds it through `ProgressReporter`, see
  `src/core/progress/reporter.py`).
- **`output`** — a deterministic fingerprint of the produced file (decoded
  text for `txt`/`srt`; per-member content hashes with ISO timestamps
  neutralized for the `epub`/`docx` zips).

The LLM is replaced by a deterministic echo provider (`fake_llm.py`) that
returns the source text verbatim, so runs are reproducible and need no
network or API key.

## Running / updating

```bash
# compare against goldens (CI mode)
python -m pytest tests/characterization/

# regenerate goldens after an INTENTIONAL behavior change, then review the diff
UPDATE_CHARACTERIZATION=1 python -m pytest tests/characterization/
```

A missing golden is bootstrapped automatically (the test `skip`s with a
notice); re-run to compare.

## Hermetic environment

`conftest.py` runs every test with `tmp_path` as the working directory (so
nothing is written under the repository's `data/`) and pins the config knobs
`src.config` reads from a local `.env` that shape the trace:
`SRT_LINES_PER_BLOCK=20`, `PARALLEL_TRANSLATIONS=1`, and the DOCX/EPUB HTML
chunk budget at the shipped `MAX_TOKENS_PER_CHUNK` default (450). DOCX/EPUB
build their chunker from `TranslationConfig`'s import-time default and ignore
the `max_tokens_per_chunk=60` the recorder passes, so before the pin a
developer `.env` silently changed their chunk counts.

## Other tests in this directory

- `test_bridge_over_goldens.py` feeds every captured emit through
  `snapshot_from_job_stats` (the web sink's percent derivation) and pins the
  shared shape: the percent is monotonic, reaches 100% on the last emit and
  only there, the unit total never changes and every emit after the first
  advances `completed_chunks` by exactly one unit.
- `test_frontend_stats_fields.py` runs every format (TXT, SRT, PGN, DOCX,
  PDF, EPUB; translate, refine-only, translate + refine) through the job
  runner into the real `WebJobSinks` and asserts that the stats fields the
  frontend reads are still in the payload.
- `test_refine_after_phase.py` and `test_runner_end_to_end.py` cover the
  two-phase translate + refine bar.

## What the goldens show

Every engine reports through a `ProgressReporter`: `total_chunks`,
`completed_chunks` and `failed_chunks` come from its tracker, the engine's
quality fields ride along. `completed_chunks` counts every processed unit,
failed ones included; `failed_chunks` is the failed part of it.

| Observation | Captured in |
|---|---|
| TXT/SRT translate and every refine emit only the counts in the stable key set (their other fields, `unfinished_chunks` and the `run_*` twins for TXT/SRT, are not projected) | `translate_txt`, `translate_srt`, `refine_*` |
| DOCX/EPUB translate add the `TranslationMetrics` quality counters (`PROGRESS_FIELDS`); EPUB adds the book-level `total_tokens` | `translate_docx`, `translate_epub` |
| SRT **translate** counts *blocks* (50 subtitles → 3 chunks), SRT **refine** counts *subtitles* (50 → 50) | `translate_srt`, `refine_srt` |
| TXT refine is chunk-counted like the others (the token tracker and its `progress_percent` are gone) | `refine_txt` |
| EPUB translate and refine tick per chunk across files; a file boundary no longer repeats the previous tick | `translate_epub`, `refine_epub` |
