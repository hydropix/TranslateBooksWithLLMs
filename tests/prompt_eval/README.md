# Prompt regression harness

Every instruction added to an LLM prompt dilutes the others, so a prompt fix can
silently degrade quality on cases it was never meant to touch. This tool answers
one question before a prompt change is merged: **did quality get worse anywhere,
not just on the case being fixed?**

It renders the **baseline** prompts (a git ref, default: the merge-base of `HEAD`
and `main`) and the **candidate** prompts (the working tree) through the real
prompt builders, sends both to one or more real models over a fixed corpus,
repeats each case N times, scores every output with deterministic metrics and
prints a Markdown before/after report you can paste into the PR description.

It is manual and on demand: not part of default `pytest`, not run in CI. Only its
deterministic parts are unit-tested (`tests/unit/prompt_eval/`).

## Commands

Dry run first. No LLM call: it shows which corpus cases the change touches and
how much each prompt grew or shrank.

```bash
python -m tests.prompt_eval diff
python -m tests.prompt_eval diff --baseline main --builders refine
```

Then the real run:

```bash
python -m tests.prompt_eval run --target ollama:qwen3:8b --repeat 3
python -m tests.prompt_eval run --baseline main --target ollama:qwen3:8b --target openrouter:mistralai/mistral-medium-3.1
```

Options:

| option | meaning |
|---|---|
| `--baseline REF` | git ref of the baseline prompts (default: `git merge-base HEAD main`) |
| `--target PROVIDER:MODEL` | repeatable; split on the first colon (default: `ollama:<DEFAULT_MODEL>`) |
| `--repeat N` | calls per case and arm (default: 3) |
| `--builders translate,refine` | prompt builders to include |
| `--cases REGEX` | filter case ids |
| `--include-unchanged` | also run cases whose prompts are identical in both arms (A/A run, shows the noise floor) |
| `--judge openrouter\|poe` | add informative LLM-judge scores (translate cases only, never part of the verdict) |
| `--out DIR` | output folder (default: `tests/prompt_eval/runs/<timestamp>/`, gitignored) |

Credentials come from `.env` through the normal provider factory. The tool never
reads or prints a key; the report names targets as `provider:model` only.

Each run writes `records.jsonl` (one line per call, written as it goes),
`report.md`, and `prompts/<case_id>.<arm>.txt` (system prompt, a
`===== USER =====` line, then the user prompt) for every case that ran.

## How it works

- The baseline tree is checked out in a temporary `git worktree` (removed at the
  end). `render.py` runs as a subprocess against each tree, so the two prompt
  versions never share a Python process.
- Kwargs a builder does not accept (signature drift between refs) are dropped
  and listed in the report, never hidden.
- A case whose system and user prompts are byte-identical in both arms is
  `unchanged` and gets no LLM call, unless `--include-unchanged`.
- Calls are sequential and interleaved (case 1 baseline, case 1 candidate,
  case 2 baseline, ...) so server load hits both arms equally. No temperature
  or seed: provider defaults are what users run; `--repeat` absorbs the noise.

## Reading the verdict

Each output gets `pass`, `fail` or `n/a` on these metrics: `format` (translation
tags found), `chatter` (text outside the tags), `placeholders`, `paragraphs`,
`residue` (source text left behind), `target_language`, `must_keep`,
`must_contain`, `must_not_contain`, `length`.

Per target and metric, failure counts are compared between arms with a
threshold of `max(2, ceil(10% of applicable outputs))`:

- **REGRESSION**: the candidate fails that many more times than the baseline.
- **IMPROVEMENT**: the reverse.
- **within noise**: anything smaller.
- **INCONCLUSIVE**: more than 20% of a target's requests failed in either arm.

The overall verdict is `REGRESSION` if any metric regressed, else `INCONCLUSIVE`
if any target is inconclusive, else `NO REGRESSION DETECTED`. Exit codes: 0 no
regression, 1 regression, 2 inconclusive, 3 usage or setup error.

"No regression detected" means none was detected at N repeats on this corpus,
not that none exists. Small regressions slip under the threshold at 3 repeats;
raise `--repeat` for changes to a system prompt.

## Cost

Calls = changed cases x 2 arms x repeat x targets. For example, 6 changed cases,
3 repeats and 2 targets make 72 calls. `diff` tells you the number of changed
cases before you spend anything.

## Rules

- **Run at least one small local model.** Dilution effects show up first on
  small models; a large API model alone can hide them.
- **The corpus is the product.** The harness only catches what the corpus
  exercises. When you fix a prompt bug, add a case that reproduces it to
  `corpus/translate.yaml` or `corpus/refine.yaml` (original prose only, 60 to
  250 words, see the field list in `model.py`).
- A new prompt builder (NER, style, subtitles, ...) is one entry in the
  `BUILDERS` registry of `render.py`, its kwargs mapping in `arms.build_request`,
  and a corpus file.
