"""
Replay a sample of submitted benchmark results to detect divergence.

For cloud providers (openai, openrouter, gemini, mistral, deepseek, poe, nim):
    1. Pick a random `--sample-pct` of results from each submission.
    2. Re-translate the same source text with the same provider/model.
    3. Compare the new output to the submitted output via chrF; compare the
       newly judged overall score to the submitted overall score.
    4. Flag a result as divergent if chrF < `--chrf-threshold` *and* the score
       diff exceeds `--max-divergence`.

For local providers (ollama, or anything else): full replay isn't possible from
CI, so we do cheap sanity checks instead:
    - `langdetect` says the output is in the claimed target language.
    - The output length is within [50%, 200%] of the source length.
    - The submission is annotated `self_reported: true` in the report.

Outputs a Markdown report at `--report` (default `replay_report.md`). Exits
non-zero when too many divergences are detected.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark.aggregator import CLOUD_PROVIDERS  # noqa: E402
from benchmark.data_loader import load_reference_texts  # noqa: E402


def _chrf(reference: str, candidate: str, n: int = 4) -> float:
    """A lightweight chrF score (character n-gram F1, n=1..n)."""
    if not reference or not candidate:
        return 0.0

    ref = reference.replace("\n", " ").strip()
    cand = candidate.replace("\n", " ").strip()
    if ref == cand:
        return 1.0

    f_scores = []
    for k in range(1, n + 1):
        ref_ngrams = Counter(ref[i:i + k] for i in range(len(ref) - k + 1))
        cand_ngrams = Counter(cand[i:i + k] for i in range(len(cand) - k + 1))
        if not ref_ngrams or not cand_ngrams:
            continue
        common = sum((ref_ngrams & cand_ngrams).values())
        if common == 0:
            f_scores.append(0.0)
            continue
        precision = common / sum(cand_ngrams.values())
        recall = common / sum(ref_ngrams.values())
        f_scores.append(2 * precision * recall / (precision + recall))

    return sum(f_scores) / len(f_scores) if f_scores else 0.0


def _check_language(text: str, expected_code: str) -> tuple[bool, str]:
    try:
        from langdetect import detect, DetectorFactory  # type: ignore
        DetectorFactory.seed = 0
    except ImportError:
        return True, "langdetect-missing"

    try:
        detected = detect(text)
    except Exception as exc:  # langdetect raises a generic exception on failure
        return False, f"detect-failed:{exc}"

    expected = expected_code.split("-")[0].lower()
    if detected.lower().startswith(expected):
        return True, detected
    return False, detected


def _length_plausibility(source: str, output: str) -> bool:
    if not source:
        return True
    ratio = len(output) / max(1, len(source))
    return 0.5 <= ratio <= 2.0


def _load_reference(text_id: str, ref_texts) -> Optional[object]:
    return ref_texts.get(text_id)


async def _replay_cloud_sample(
    submission: dict,
    sample: list[dict],
    ref_texts: dict,
    chrf_threshold: float,
    max_divergence: float,
) -> list[dict]:
    """Re-run translation+evaluation for cloud-provider submissions."""
    from benchmark.config import BenchmarkConfig
    from benchmark.evaluator import TranslationEvaluator
    from benchmark.translator import BenchmarkTranslator, TranslationRequest

    provider = submission["model"]["provider"]
    model_id = submission["model"]["id"]
    judge_id = submission.get("environment", {}).get("judge_id", "")

    config = BenchmarkConfig.from_cli_args(
        translation_provider=provider if provider in {"ollama", "openai", "openrouter"} else "openrouter",
    )

    translator = BenchmarkTranslator(config, provider_type=config.translation_provider)
    evaluator = TranslationEvaluator(config, provider=config.evaluator_provider)

    findings: list[dict] = []
    try:
        for r in sample:
            ref = _load_reference(r["text_id"], ref_texts)
            if ref is None:
                findings.append({
                    "text_id": r["text_id"],
                    "target_lang": r["target_lang"],
                    "status": "missing-reference",
                })
                continue

            req = TranslationRequest(
                text=ref,
                target_language=r["target_lang"],
                target_language_name=r["target_lang"],
                model=model_id,
            )

            replay = await translator.translate(req)

            if not replay.success:
                findings.append({
                    "text_id": r["text_id"],
                    "target_lang": r["target_lang"],
                    "status": "translation-failed",
                    "error": replay.error,
                })
                continue

            replay_scores, _ = await evaluator.evaluate(
                source_text=ref,
                translated_text=replay.translated_text,
                target_language=r["target_lang"],
                target_language_name=r["target_lang"],
            )

            chrf = _chrf(r["output"], replay.translated_text)
            score_diff = abs(replay_scores.overall - r["scores"]["overall"])
            divergent = chrf < chrf_threshold and score_diff > max_divergence

            findings.append({
                "text_id": r["text_id"],
                "target_lang": r["target_lang"],
                "status": "divergent" if divergent else "ok",
                "chrf": round(chrf, 3),
                "submitted_overall": r["scores"]["overall"],
                "replay_overall": round(replay_scores.overall, 2),
                "score_diff": round(score_diff, 2),
                "judge_id": judge_id,
            })
    finally:
        await translator.close()
        await evaluator.close()

    return findings


def _heuristic_sample(
    submission: dict,
    sample: list[dict],
    ref_texts: dict,
) -> list[dict]:
    """Sanity checks for non-replayable submissions (e.g. local Ollama)."""
    findings: list[dict] = []
    for r in sample:
        ref = _load_reference(r["text_id"], ref_texts)
        source = ref.content if ref is not None else ""
        ok_lang, detected = _check_language(r["output"], r["target_lang"])
        ok_len = _length_plausibility(source, r["output"])
        status = "ok" if (ok_lang and ok_len) else "suspicious"
        findings.append({
            "text_id": r["text_id"],
            "target_lang": r["target_lang"],
            "status": status,
            "lang_detected": detected,
            "lang_ok": ok_lang,
            "length_ok": ok_len,
        })
    return findings


def _render_report(submissions_report: list[dict]) -> str:
    out = ["# Replay Report", ""]
    for entry in submissions_report:
        out.append(f"## `{entry['file']}`")
        out.append("")
        out.append(f"- Submitted by: `{entry['submitted_by']}`")
        out.append(f"- Provider: `{entry['provider']}`")
        out.append(f"- Model: `{entry['model_id']}`")
        out.append(f"- Verified mode: `{entry['mode']}`")
        out.append(f"- Sampled: {entry['n_sampled']} / {entry['n_total']} results")
        out.append(f"- Divergent / suspicious: {entry['n_divergent']}")
        if entry["mode"] == "self_reported":
            out.append("- _Marked as self-reported in the wiki because the model is not replayable in CI._")

        if entry["findings"]:
            out.append("")
            out.append("| text | lang | status | details |")
            out.append("|------|------|--------|---------|")
            for f in entry["findings"]:
                details = ", ".join(f"{k}={v}" for k, v in f.items() if k not in {"text_id", "target_lang", "status"})
                out.append(f"| {f['text_id']} | {f['target_lang']} | {f['status']} | {details} |")
        out.append("")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", nargs="+", required=True, help="Submission JSON files to replay.")
    parser.add_argument("--sample-pct", type=float, default=10.0, help="Percent of results to sample per file (1-100).")
    parser.add_argument("--max-divergence", type=float, default=1.0, help="Score diff threshold above which a result is flagged.")
    parser.add_argument("--chrf-threshold", type=float, default=0.85, help="chrF threshold below which outputs differ significantly.")
    parser.add_argument("--max-divergence-rate", type=float, default=0.34, help="Fail when more than this fraction of sampled results diverge.")
    parser.add_argument("--report", default="replay_report.md", help="Where to write the Markdown report.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    sample_ratio = max(0.0, min(1.0, args.sample_pct / 100.0))

    ref_texts = load_reference_texts(
        base_dir=REPO_ROOT / "benchmark",
        legacy_file=REPO_ROOT / "benchmark" / "reference_texts.yaml",
    )

    submissions_report: list[dict] = []
    fail_overall = False

    for file_str in args.files:
        path = Path(file_str)
        if not path.is_file():
            print(f"[skip] {path}: not found", file=sys.stderr)
            continue

        with path.open("r", encoding="utf-8") as f:
            submission = json.load(f)

        results = submission.get("results", [])
        if not results:
            continue

        provider = submission["model"]["provider"]
        is_cloud = provider in CLOUD_PROVIDERS

        n_sample = max(1, int(round(len(results) * sample_ratio)))
        sample = random.sample(results, min(n_sample, len(results)))

        if is_cloud:
            mode = "verified"
            findings = asyncio.run(_replay_cloud_sample(
                submission, sample, ref_texts,
                chrf_threshold=args.chrf_threshold,
                max_divergence=args.max_divergence,
            ))
            divergent = [f for f in findings if f["status"] == "divergent"]
        else:
            mode = "self_reported"
            findings = _heuristic_sample(submission, sample, ref_texts)
            divergent = [f for f in findings if f["status"] != "ok"]

        rate = len(divergent) / len(sample) if sample else 0
        if mode == "verified" and rate > args.max_divergence_rate:
            fail_overall = True

        submissions_report.append({
            "file": str(path),
            "submitted_by": submission.get("submission", {}).get("submitted_by", "unknown"),
            "provider": provider,
            "model_id": submission["model"]["id"],
            "mode": mode,
            "n_total": len(results),
            "n_sampled": len(sample),
            "n_divergent": len(divergent),
            "findings": findings,
        })

    Path(args.report).write_text(_render_report(submissions_report), encoding="utf-8")
    print(f"Report written to {args.report}")

    if fail_overall:
        print("Replay detected too many divergences for at least one submission.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
