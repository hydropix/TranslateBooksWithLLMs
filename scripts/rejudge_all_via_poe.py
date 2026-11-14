"""
Re-judge ALL benchmark translations via Poe Claude-Opus-4.7 under rubric v2.

Reads every `benchmark/data/translations/<slug>.json` (v2 split layout),
sends each (source + translation) pair to Poe at temp=0.1 with the rubric v2
prompt, and writes a single output file:

    benchmark/data/judgments/claude-opus-4-7-rubric-v2-poe.json

It NEVER touches the translation files. Re-running is idempotent: existing
scores are loaded from the judgment file and skipped, with --resume default.

Key design choices:

- **Read split layout**: this script expects `benchmark/data/translations/`
  to be populated (run `scripts/migrate_to_split_layout.py --apply` first).
- **Idempotent**: scored entries are keyed by (model_id, text_id, target_lang)
  with output_hash integrity check. Same key already scored → skipped.
- **Single output file**: one judgments file per judge config. Re-running
  appends/refreshes entries in place.
- **Concurrency**: bounded by `--concurrency` (default 8). Poe rate-limits
  to 500 req/min, 8 concurrent at ~2s/call ≈ 240 req/min.
- **Retry**: parse failures retry up to 2× with the same temperature.
  HTTP 429 → exponential backoff. Other HTTP errors → log + skip.

Usage:
    # Smoke test: 10 random translations, no writes, print comparison
    python scripts/rejudge_all_via_poe.py --dry-run 10

    # Full run (with confirmation prompt)
    python scripts/rejudge_all_via_poe.py

    # Resume after interruption (default behaviour)
    python scripts/rejudge_all_via_poe.py --resume

    # Force re-judge everything (ignore existing scores)
    python scripts/rejudge_all_via_poe.py --no-resume
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

for _name in ("stdout", "stderr"):
    s = getattr(sys, _name, None)
    if s and hasattr(s, "reconfigure"):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

load_dotenv()

from benchmark.config import BenchmarkConfig
from benchmark.data_loader import load_languages, load_reference_texts
from benchmark.models import (
    JudgmentScore,
    JudgmentsFile,
    TranslationsFile,
)


POE_ENDPOINT = "https://api.poe.com/v1/chat/completions"
POE_MODEL = "Claude-Opus-4.7"
TEMPERATURE = 0.1
MAX_TOKENS = 600
JUDGE_ID = "claude-opus-4-7-rubric-v2-poe"
RUBRIC_VERSION = "v2"

DEFAULT_CONCURRENCY = 8
CHECKPOINT_EVERY = 10
RETRY_MAX = 2

POE_INPUT_USD_PER_M = 4.29
POE_OUTPUT_USD_PER_M = 21.46


SYSTEM_PROMPT = """You are a translation judge applying a fixed rubric.

# Dimensions (1.0–10.0, decimals OK)

- **accuracy**: meaning preservation (no omissions, no contresens, no hallucinations)
- **fluency**: naturalness in target language (grammar, idiom, no untranslated source words, no script mismatches)
- **style**: register, tone, period vocabulary, literary voice, rhetorical devices
- **overall**: holistic. NOT the average.

# Anchored scale (calibrate against)

- 10: published reference literary translation (Penguin/Pléiade tier). LLMs essentially can't reach this.
- 9: excellent professional, publishable with light edit. Cap without human reference.
- 8: very good, full meaning + tone but 1–2 nuances lost.
- 7: good but 2–3 notable errors of register/idiom/specialized term.
- 6: comprehensible, clumsy, multiple problematic word choices.
- 5: usable only with editing. Real meaning errors or fluency breaks.
- 4: significantly impaired. Frequent contresens.
- 3: passages incomprehensible.
- 2: hallucinations dominant.
- 1: non-translation (wrong language, source copied, refusal, empty).

# Penalty table — start from 10, deduct

Accuracy: contresens −2.0, hallucination −2.0, source word left in target −1.5, omission −1.0, wrong specialized term −0.5 to −1.0, minor drift −0.3.
Fluency: grammar break −1.5, mixed scripts (Latin inside CJK, etc.) −1.0, awkward MT-feel −0.5 to −1.0, wrong punctuation −0.3, calque/anglicism −0.5.
Style: lost signature device (irony, parallelism) −1.0 to −1.5, period-inappropriate vocab −1.0, register flattened −0.5 to −1.0, weakened idiom −0.3.

# Hard ceilings on overall

- if accuracy < 6.0 → overall ≤ 6.0
- if fluency < 5.0 → overall ≤ 6.0
- if any dimension ≤ 3.0 → overall ≤ 4.0
- overall ≤ min(accuracy, fluency, style) + 0.5
- without human reference: overall ≤ 9.0

Be willing to use 3–5. Most LLM outputs fall in 6.5–8.5. Don't be charitable.

# Output

Respond with ONLY a JSON object, no prose around it. The "feedback" field MUST be ≤200 ASCII chars; do NOT quote source-language text verbatim — describe issues in English with English glosses or generic labels (e.g. "first Korean phrase mistranslated as 'rumors'"). Penalty totals only, no per-token enumeration.

{"accuracy": N, "fluency": N, "style": N, "overall": N, "feedback": "..."}
"""


@dataclass
class Task:
    """One judging unit identified by (model_id, text_id, target_lang)."""
    model_id: str
    text_id: str
    target_lang: str
    target_lang_name: str
    output_hash: str
    source_text: str
    source_lang: str
    text_title: str
    text_author: str
    text_year: str
    text_style: str
    translation: str
    # Original scores from existing judgments (for comparison reporting), if any
    orig_overall: Optional[float] = None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.model_id, self.text_id, self.target_lang)


@dataclass
class CallResult:
    """Outcome of one Poe call for one Task."""
    scores: Optional[dict] = None
    error: Optional[str] = None
    attempts: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


def load_tasks(translations_dir: Path) -> list[Task]:
    cfg = BenchmarkConfig()
    ref_texts = load_reference_texts(base_dir=cfg.paths.base_dir, legacy_file=cfg.paths.reference_texts_file)
    languages = load_languages(base_dir=cfg.paths.base_dir, legacy_file=cfg.paths.languages_file)

    if not translations_dir.is_dir():
        raise RuntimeError(
            f"{translations_dir} not found. Run scripts/migrate_to_split_layout.py --apply first."
        )

    tasks: list[Task] = []
    for path in sorted(translations_dir.glob("*.json")):
        tf = TranslationsFile.from_dict(json.loads(path.read_text(encoding="utf-8")))
        for t in tf.translations:
            ref = ref_texts.get(t.text_id)
            lang = languages.get(t.target_lang)
            tasks.append(Task(
                model_id=tf.model_id,
                text_id=t.text_id,
                target_lang=t.target_lang,
                target_lang_name=lang.name if lang else t.target_lang,
                output_hash=t.output_hash,
                source_text=ref.content if ref else "",
                source_lang=ref.source_language if ref else t.source_lang,
                text_title=ref.title if ref else t.text_id,
                text_author=ref.author if ref else "?",
                text_year=str(ref.year) if ref else "",
                text_style=ref.style if ref else "",
                translation=t.output,
            ))
    return tasks


def load_existing_judgments(judgments_path: Path) -> dict[tuple[str, str, str], JudgmentScore]:
    """Read the existing judge file (if any) and index by key."""
    if not judgments_path.is_file():
        return {}
    data = json.loads(judgments_path.read_text(encoding="utf-8"))
    jf = JudgmentsFile.from_dict(data)
    return jf.by_key()


def build_user_prompt(t: Task) -> str:
    return f"""# Translation evaluation

**Source language:** {t.source_lang}
**Target language:** {t.target_lang_name}
**Source text:** "{t.text_title}" by {t.text_author} ({t.text_year})
**Style:** {t.text_style}

## Source
{t.source_text}

## Translation ({t.target_lang_name})
{t.translation}

Apply the rubric. JSON only."""


def parse_response(text: str) -> Optional[dict]:
    t = text.strip()
    if t.startswith("```"):
        nl = t.find("\n")
        t = t[nl + 1:] if nl != -1 else t[3:]
        if t.endswith("```"):
            t = t[:-3]
        t = t.strip()
    try:
        d = json.loads(t)
        return _validate_scores(d)
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        c = t[i]
        if esc:
            esc = False
            continue
        if c == "\\" and in_str:
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    d = json.loads(t[start:i + 1])
                    return _validate_scores(d)
                except json.JSONDecodeError:
                    return None
    return None


def _validate_scores(d: dict) -> Optional[dict]:
    required = ("accuracy", "fluency", "style", "overall")
    if not all(k in d for k in required):
        return None
    out = {}
    for k in required:
        try:
            v = float(d[k])
        except (TypeError, ValueError):
            return None
        out[k] = max(1.0, min(10.0, v))
    out["feedback"] = str(d.get("feedback", ""))[:500]
    return out


async def call_poe(client: httpx.AsyncClient, api_key: str, task: Task) -> CallResult:
    payload = {
        "model": POE_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(task)},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    res = CallResult()

    for attempt in range(1, RETRY_MAX + 2):
        res.attempts = attempt
        try:
            r = await client.post(POE_ENDPOINT, headers=headers, json=payload, timeout=120.0)
            if r.status_code == 429:
                wait = min(2 ** attempt, 60)
                await asyncio.sleep(wait)
                continue
            if r.status_code != 200:
                res.error = f"HTTP {r.status_code}: {r.text[:200]}"
                return res
            data = r.json()
            content = data["choices"][0]["message"].get("content", "") or ""
            usage = data.get("usage", {})
            res.prompt_tokens = usage.get("prompt_tokens", 0)
            res.completion_tokens = usage.get("completion_tokens", 0)
            scores = parse_response(content)
            if scores is not None:
                res.scores = scores
                return res
            await asyncio.sleep(0.5)
            continue
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            res.error = f"network: {e}"
            await asyncio.sleep(2 ** attempt)
            continue
        except Exception as e:
            res.error = f"unexpected: {e}"
            return res

    if res.error is None:
        res.error = "parse failed after retries"
    return res


def write_judgments(scores: dict[tuple[str, str, str], JudgmentScore],
                    tasks_by_key: dict[tuple[str, str, str], Task],
                    judgments_path: Path,
                    started_at: str) -> None:
    """Atomically write the consolidated judgments file."""
    scores_list = []
    for key, s in sorted(scores.items()):
        # Sanity check: output_hash must match the translation's hash
        t = tasks_by_key.get(key)
        if t and t.output_hash != s.output_hash:
            print(f"WARN: hash mismatch on {key} — overwriting score's hash with translation's", file=sys.stderr)
            s.output_hash = t.output_hash
        scores_list.append(s)

    jf = JudgmentsFile(
        schema_version="2.0",
        judge_id=JUDGE_ID,
        judge_model=POE_MODEL,
        rubric_version=RUBRIC_VERSION,
        judge_provider="poe",
        judge_temperature=TEMPERATURE,
        judge_thinking="disabled",
        run_id=f"rejudge_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        scores=scores_list,
    )

    judgments_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = judgments_path.with_suffix(".json.tmp")
    tmp.write_text(jf.to_json() + "\n", encoding="utf-8")
    tmp.replace(judgments_path)


def print_comparison_summary(tasks: list[Task], new_scores: dict[tuple[str, str, str], JudgmentScore]) -> None:
    """Per-model average score delta old → new."""
    by_model: dict[str, dict] = defaultdict(lambda: {"old": [], "new": []})
    for t in tasks:
        s = new_scores.get(t.key)
        if s is None:
            continue
        by_model[t.model_id]["new"].append(float(s.overall))
        if t.orig_overall is not None:
            by_model[t.model_id]["old"].append(float(t.orig_overall))

    print()
    print("=" * 78)
    print(f"{'model':30s}  {'n':>4s}  {'old avg':>9s}  {'new avg':>9s}  {'Δ':>7s}")
    print("-" * 78)
    rows = []
    for m, vals in by_model.items():
        if not vals["new"]:
            continue
        new_avg = statistics.mean(vals["new"])
        old_avg = statistics.mean(vals["old"]) if vals["old"] else float("nan")
        delta = new_avg - old_avg if vals["old"] else 0
        rows.append((m, len(vals["new"]), old_avg, new_avg, delta))
    rows.sort(key=lambda r: -r[3])
    for m, n, oa, na, d in rows:
        oa_s = f"{oa:>9.2f}" if oa == oa else "        -"
        d_s = f"{d:>+7.2f}" if oa == oa else "       -"
        arrow = "↑" if d > 0.05 else ("↓" if d < -0.05 else "=")
        print(f"{m:30s}  {n:>4d}  {oa_s}  {na:>9.2f}  {d_s} {arrow}")
    print("=" * 78)


def make_judgment_score(task: Task, scores: dict) -> JudgmentScore:
    return JudgmentScore(
        model_id=task.model_id,
        text_id=task.text_id,
        target_lang=task.target_lang,
        output_hash=task.output_hash,
        accuracy=round(scores["accuracy"], 2),
        fluency=round(scores["fluency"], 2),
        style=round(scores["style"], 2),
        overall=round(scores["overall"], 2),
        feedback=scores.get("feedback") or None,
    )


async def dry_run(tasks: list[Task], api_key: str, n: int) -> None:
    sample = random.sample(tasks, min(n, len(tasks)))
    print(f"\nDry-run: judging {len(sample)} random translations, no writes.\n")
    new_scores: dict[tuple[str, str, str], JudgmentScore] = {}
    total_in = total_out = 0
    async with httpx.AsyncClient() as client:
        for i, t in enumerate(sample, 1):
            print(f"[{i}/{len(sample)}] {t.model_id:25s} {t.text_id:22s} → {t.target_lang:8s}", end="  ", flush=True)
            res = await call_poe(client, api_key, t)
            if res.scores is None:
                print(f"FAIL: {res.error}")
                continue
            total_in += res.prompt_tokens
            total_out += res.completion_tokens
            new_scores[t.key] = make_judgment_score(t, res.scores)
            orig = t.orig_overall
            arrow = ""
            if orig is not None:
                arrow = f"  (Δ={res.scores['overall'] - orig:+.1f})"
            print(f"new ovr={res.scores['overall']:.1f}{arrow}")
    print_comparison_summary(sample, new_scores)
    if total_in:
        cost = (total_in * POE_INPUT_USD_PER_M + total_out * POE_OUTPUT_USD_PER_M) / 1_000_000
        per_call = cost / max(1, len(new_scores))
        print(f"\nDry-run cost: ${cost:.4f}  ({len(new_scores)} calls @ ${per_call:.5f}/call)")
        print(f"Projected for all {len(tasks)} translations: ${per_call * len(tasks):.2f}")


async def full_run(tasks: list[Task], api_key: str, concurrency: int,
                   judgments_path: Path, resume: bool) -> int:
    existing: dict[tuple[str, str, str], JudgmentScore] = {}
    if resume:
        existing = load_existing_judgments(judgments_path)
        if existing:
            print(f"Loaded {len(existing)} existing scores from {judgments_path}")

    tasks_by_key = {t.key: t for t in tasks}
    pending = [t for t in tasks if t.key not in existing]

    print(f"\nFull run: {len(tasks)} translations, {len(existing)} cached, {len(pending)} pending.")
    print(f"Judge: {POE_MODEL} @ temp={TEMPERATURE}, concurrency={concurrency}")
    print(f"Output: {judgments_path}")

    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if not pending:
        print("All translations already judged. Re-writing file with current contents.")
        write_judgments(existing, tasks_by_key, judgments_path, started_at)
        return 0

    new_scores: dict[tuple[str, str, str], JudgmentScore] = dict(existing)
    fail_log: list[tuple[tuple, str]] = []
    total_in = total_out = 0
    done = 0
    start = time.time()
    sem = asyncio.Semaphore(concurrency)

    async def worker(task: Task, client: httpx.AsyncClient) -> None:
        nonlocal done, total_in, total_out
        async with sem:
            res = await call_poe(client, api_key, task)
            done += 1
            if res.scores is not None:
                new_scores[task.key] = make_judgment_score(task, res.scores)
                total_in += res.prompt_tokens
                total_out += res.completion_tokens
            else:
                fail_log.append((task.key, res.error or "unknown"))
            if done % CHECKPOINT_EVERY == 0 or done == len(pending):
                write_judgments(new_scores, tasks_by_key, judgments_path, started_at)
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(pending) - done) / rate if rate > 0 else 0
                print(f"  [{done}/{len(pending)}] rate={rate:.1f}/s  ETA={eta/60:.1f}m  "
                      f"in_tok={total_in:,}  out_tok={total_out:,}  fails={len(fail_log)}")

    async with httpx.AsyncClient() as client:
        await asyncio.gather(*(worker(t, client) for t in pending))

    if fail_log:
        print(f"\n{len(fail_log)} translations FAILED to judge:")
        for k, err in fail_log[:20]:
            print(f"  {k[0]:25s} {k[1]:22s} → {k[2]:8s}  {err[:120]}")

    cost = (total_in * POE_INPUT_USD_PER_M + total_out * POE_OUTPUT_USD_PER_M) / 1_000_000
    print(f"\nAPI cost this run: ${cost:.2f}  (in={total_in:,} tok, out={total_out:,} tok)")

    write_judgments(new_scores, tasks_by_key, judgments_path, started_at)
    print(f"Wrote {len(new_scores)} scores to {judgments_path}")

    print_comparison_summary(tasks, new_scores)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", type=int, metavar="N",
                        help="Judge N random translations and print comparison, no writes")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help=f"Max concurrent Poe calls (default {DEFAULT_CONCURRENCY})")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from existing judgment file (default behavior)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore existing scores and re-judge everything")
    parser.add_argument("--yes", action="store_true",
                        help="Skip confirmation prompt before full run")
    args = parser.parse_args()

    api_key = os.getenv("POE_API_KEY")
    if not api_key:
        print("ERROR: POE_API_KEY not set in environment / .env", file=sys.stderr)
        return 1

    cfg = BenchmarkConfig()
    translations_dir = cfg.paths.base_dir / "data" / "translations"
    judgments_dir = cfg.paths.base_dir / "data" / "judgments"
    judgments_path = judgments_dir / f"{JUDGE_ID}.json"

    print(f"Loading translations from {translations_dir} ...")
    try:
        tasks = load_tasks(translations_dir)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    by_model = defaultdict(int)
    for t in tasks:
        by_model[t.model_id] += 1
    print(f"  {len(tasks)} total translations across {len(by_model)} models")
    for m, n in sorted(by_model.items()):
        print(f"    {m:30s}  {n:3d}")

    if args.dry_run:
        return asyncio.run(dry_run(tasks, api_key, args.dry_run))

    resume = not args.no_resume

    if not args.yes:
        print()
        print("About to FULL RUN:")
        print(f"  - call Poe {POE_MODEL} up to {len(tasks)} times")
        print(f"  - estimated cost ~${len(tasks) * 0.0068:.2f}")
        print(f"  - write judgments to {judgments_path}")
        print(f"  - judge_id = '{JUDGE_ID}'")
        reply = input("\nProceed? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("Aborted.")
            return 1

    return asyncio.run(full_run(tasks, api_key, args.concurrency, judgments_path, resume))


if __name__ == "__main__":
    sys.exit(main())
