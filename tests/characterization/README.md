# Progress-system characterization tests

Safety net for the progress-system refactoring (Step 0 of the plan). These
tests pin the **current** observable behavior of the translation/refine
progress machinery so later steps can change the implementation without
silently changing what the UI sees.

They are *characterization* tests: they document what the system does today,
not what it should do. Several captured behaviors are in fact bugs (see
below) — the point is to make every future change to them explicit.

## What is captured

For each format (`txt`, `srt`, `docx`, `epub`) and each mode (`translate`,
`refine`), the golden file records:

- **`sequence`** — the ordered list of `stats` dicts passed to
  `stats_callback`, projected onto a stable key set (time/cost/ETA fields are
  stripped, floats rounded). This is the progress signal the orchestrator
  emits.
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

## Divergences this net locks in (the refactor targets)

These are the per-format inconsistencies the goldens make executable. The
refactor should *converge* them — each change here must be a conscious golden
update:

| Observation | Captured in |
|---|---|
| TXT/SRT translate emit a bare 3-key dict (`total/completed/failed`) | `translate_txt`, `translate_srt` |
| DOCX/EPUB translate emit a ~12-key `TranslationMetrics` dict — and not even the same keys as each other | `translate_docx`, `translate_epub` |
| SRT **translate** counts *blocks* (50 subtitles → 3 chunks) | `translate_srt` |
| SRT **refine** counts *subtitles* (50 → 50 chunks) — same format, different unit per mode | `refine_srt` |
| Only TXT **refine** uses the token tracker shape (`progress_percent`, `current_phase`, `*_tokens`); EPUB/SRT/DOCX refine emit bare chunk dicts | `refine_txt` vs others |
| DOCX **refine** previously emitted a single `0/N` callback and never advanced — **fixed (Step 7)**: now emits per-chunk progress | `refine_docx` |
| EPUB **refine** updates once per file, not per chunk | `refine_epub` |

See the deep-analysis report for the root causes behind each.
