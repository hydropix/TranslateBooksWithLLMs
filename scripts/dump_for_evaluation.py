"""
Dump the unscored translations of a benchmark run as a Markdown brief, ready
to paste into a Claude conversation for manual evaluation.

Each translation gets:
  - a short stable `eval_id` (sha1 of source_text_id|target_lang|model, 10 chars)
  - the source text + its declared challenges
  - the translation produced by the model
  - a one-line response template at the end

The expected reply format is a single JSON array of objects, each with
`eval_id` + `scores` (accuracy, fluency, style, overall, feedback). Save the
reply to a file and feed it to `apply_evaluations.py`.

Usage:
    python scripts/dump_for_evaluation.py <run_id> [--batch-size N] [--out brief.md]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark.config import BenchmarkConfig  # noqa: E402
from benchmark.data_loader import load_reference_texts  # noqa: E402
from benchmark.results.storage import ResultsStorage  # noqa: E402


def make_eval_id(text_id: str, target_lang: str, model: str) -> str:
    raw = f"{text_id}|{target_lang}|{model}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:10]


def build_brief(run, ref_texts: dict, batch_size: int) -> list[str]:
    """Return one Markdown blob per batch."""
    pending = [r for r in run.results if r.success and r.translated_text and r.scores is None]
    if not pending:
        return []

    batches: list[str] = []
    for i in range(0, len(pending), batch_size):
        chunk = pending[i:i + batch_size]
        lines: list[str] = []
        lines.append(f"# Evaluation brief — batch {i // batch_size + 1} of {(len(pending) + batch_size - 1) // batch_size}")
        lines.append("")
        lines.append(f"Run: `{run.run_id}` — {len(chunk)} translation(s) to score")
        lines.append("")
        lines.append("**Rubric** (1.0–10.0 each, decimals allowed):")
        lines.append("- `accuracy` — preservation of meaning, no additions or omissions")
        lines.append("- `fluency` — natural target-language phrasing, grammar, idiomaticity")
        lines.append("- `style` — register, tone, literary voice, period vocabulary fidelity")
        lines.append("- `overall` — global judgement (not the average; your weighted call)")
        lines.append("- `feedback` — 1–2 sentences explaining the overall score")
        lines.append("")
        lines.append("---")
        lines.append("")

        for r in chunk:
            ref = ref_texts.get(r.source_text_id)
            eval_id = make_eval_id(r.source_text_id, r.target_language, r.model)
            lines.append(f"## eval_id: `{eval_id}`")
            lines.append("")
            lines.append(f"- **text_id**: `{r.source_text_id}` — *{ref.title if ref else '?'}* ({ref.author if ref else '?'}, {ref.year if ref else '?'})")
            lines.append(f"- **source_lang**: `{ref.source_language if ref else '?'}`")
            lines.append(f"- **target_lang**: `{r.target_language}`")
            lines.append(f"- **model**: `{r.model}`")
            if ref and ref.challenges:
                lines.append(f"- **challenges to weigh**: {'; '.join(ref.challenges)}")
            lines.append("")
            lines.append("### Source")
            lines.append("```")
            lines.append((ref.content if ref else "<missing reference>").strip())
            lines.append("```")
            lines.append("")
            lines.append("### Translation")
            lines.append("```")
            lines.append(r.translated_text.strip())
            lines.append("```")
            lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("## Reply format")
        lines.append("")
        lines.append("Respond **only** with a JSON array, one object per `eval_id`:")
        lines.append("")
        lines.append("```json")
        lines.append("[")
        for r in chunk:
            eval_id = make_eval_id(r.source_text_id, r.target_language, r.model)
            lines.append('  {"eval_id": "' + eval_id + '", "scores": {"accuracy": 0.0, "fluency": 0.0, "style": 0.0, "overall": 0.0, "feedback": ""}},')
        # Strip trailing comma on the last entry for valid JSON
        if lines[-1].endswith(","):
            lines[-1] = lines[-1][:-1]
        lines.append("]")
        lines.append("```")
        lines.append("")
        batches.append("\n".join(lines))

    return batches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="Benchmark run ID to dump")
    parser.add_argument("--batch-size", type=int, default=15, help="Translations per Markdown batch (default: 15)")
    parser.add_argument("--out", type=Path, help="Write to this file/prefix instead of stdout. With multiple batches, '_batch1.md' etc. are appended.")
    args = parser.parse_args()

    config = BenchmarkConfig()
    storage = ResultsStorage(config)
    run = storage.load_run(args.run_id)
    if run is None:
        print(f"Run {args.run_id} not found.", file=sys.stderr)
        return 1

    ref_texts = load_reference_texts(
        base_dir=config.paths.base_dir,
        legacy_file=config.paths.reference_texts_file,
    )

    batches = build_brief(run, ref_texts, args.batch_size)

    if not batches:
        print("No unscored successful translations to evaluate.")
        return 0

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if len(batches) == 1:
            args.out.write_text(batches[0], encoding="utf-8")
            print(f"Wrote {args.out} ({len(batches[0])} chars)")
        else:
            stem = args.out.with_suffix("")
            for i, b in enumerate(batches, start=1):
                p = stem.with_name(f"{stem.name}_batch{i}{args.out.suffix}")
                p.write_text(b, encoding="utf-8")
                print(f"Wrote {p}")
    else:
        for i, b in enumerate(batches):
            if i > 0:
                print("\n\n" + "=" * 80 + "\n\n")
            print(b)

    return 0


if __name__ == "__main__":
    sys.exit(main())
