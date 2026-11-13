"""
Migrate benchmark/data/submissions/ to the v2 split layout.

Old layout (1 file per (date, contributor, model[, variant])):
    benchmark/data/submissions/2026-05-10_hydropix_gemma4-e2b.json
    benchmark/data/submissions/2026-05-10_hydropix_qwen3.5-35b.json
    benchmark/data/submissions/2026-05-10_hydropix_qwen3.5-35b_run2.json
    benchmark/data/submissions/2026-05-10_hydropix_qwen3.5-35b_rescore.json

New layout:
    benchmark/data/translations/<model-slug>.json   — one per model, all translations
    benchmark/data/judgments/<judge-id>.json        — one per judge (NOT created here)
    benchmark/data/submissions_v1_archive/          — old files preserved for audit

This script ONLY produces translations files. Scores are intentionally
DROPPED. The v2 series re-judges everything with Opus 4.7 rubric v2 via the
separate rejudge_all_via_poe.py script.

Conflict resolution (when the same model has multiple submission files):
    - If output_hash matches across observations for the same
      (text_id, target_lang) tuple → silently dedup.
    - If output_hash differs → KEEP THE MOST RECENT submission's output
      (by `submitted_at`), warn on stderr.

Usage:
    python scripts/migrate_to_split_layout.py            # dry-run preview
    python scripts/migrate_to_split_layout.py --apply    # actually write files
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

for _name in ("stdout", "stderr"):
    s = getattr(sys, _name, None)
    if s and hasattr(s, "reconfigure"):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

SCHEMA_VERSION = "2.0"


def model_slug(model_id: str) -> str:
    """gemma4:e2b -> gemma4-e2b ; mistral/foo -> mistral-foo"""
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", model_id).strip("-")


@dataclass
class Observation:
    """One (text_id, target_lang) translation observation from one submission file."""
    source_file: Path
    submitted_at: str  # ISO 8601
    submitted_by: str
    notes: str
    source_lang: str
    target_lang: str
    text_id: str
    output: str
    output_hash: str
    translation_latency_ms: int

    @property
    def key(self) -> tuple[str, str]:
        return (self.text_id, self.target_lang)


@dataclass
class ModelBucket:
    """All observations + metadata for one model_id."""
    model_id: str
    provider: str
    tbl_version: str
    prompt_version: str
    observations: list[Observation]
    contributors: list[dict]  # [{"by": ..., "at": ..., "notes": ...}]


def parse_submissions(submissions_dir: Path) -> dict[str, ModelBucket]:
    """Read all submissions, group by model_id."""
    buckets: dict[str, ModelBucket] = {}
    for path in sorted(submissions_dir.glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        mid = d["model"]["id"]
        if mid not in buckets:
            buckets[mid] = ModelBucket(
                model_id=mid,
                provider=d["model"]["provider"],
                tbl_version=d["environment"]["tbl_version"],
                prompt_version=d["environment"]["prompt_version"],
                observations=[],
                contributors=[],
            )
        b = buckets[mid]
        sub_meta = d["submission"]
        contrib = {
            "by": sub_meta["submitted_by"],
            "at": sub_meta["submitted_at"],
        }
        if "notes" in sub_meta and sub_meta["notes"]:
            contrib["notes"] = sub_meta["notes"]
        if contrib not in b.contributors:
            b.contributors.append(contrib)
        for r in d["results"]:
            b.observations.append(Observation(
                source_file=path,
                submitted_at=sub_meta["submitted_at"],
                submitted_by=sub_meta["submitted_by"],
                notes=sub_meta.get("notes", ""),
                source_lang=r["source_lang"],
                target_lang=r["target_lang"],
                text_id=r["text_id"],
                output=r["output"],
                output_hash=r["output_hash"],
                translation_latency_ms=int(r.get("translation_latency_ms", 0)),
            ))
    return buckets


def resolve_conflicts(observations: list[Observation]) -> tuple[list[Observation], list[str]]:
    """Return (kept_observations, warnings) — one observation per (text_id, target_lang)."""
    by_key: dict[tuple[str, str], list[Observation]] = defaultdict(list)
    for o in observations:
        by_key[o.key].append(o)

    kept: list[Observation] = []
    warnings: list[str] = []
    for key, obs_list in by_key.items():
        if len(obs_list) == 1:
            kept.append(obs_list[0])
            continue
        # Multiple observations for same (text_id, target_lang)
        hashes = {o.output_hash for o in obs_list}
        if len(hashes) == 1:
            # Same output, just dedup (pick most recent for timestamp)
            most_recent = max(obs_list, key=lambda o: o.submitted_at)
            kept.append(most_recent)
            continue
        # Hash conflict: outputs differ → most recent wins
        sorted_obs = sorted(obs_list, key=lambda o: o.submitted_at, reverse=True)
        winner = sorted_obs[0]
        losers = sorted_obs[1:]
        kept.append(winner)
        warnings.append(
            f"  CONFLICT {key[0]}→{key[1]}: {len(obs_list)} observations with different output_hash. "
            f"Keeping most recent ({winner.submitted_at} from {winner.source_file.name}). "
            f"Discarded: {', '.join(l.source_file.name for l in losers)}"
        )
    return kept, warnings


def bucket_to_doc(b: ModelBucket, kept_obs: list[Observation]) -> dict:
    translations = []
    for o in sorted(kept_obs, key=lambda x: (x.text_id, x.target_lang)):
        t = {
            "text_id": o.text_id,
            "source_lang": o.source_lang,
            "target_lang": o.target_lang,
            "output": o.output,
            "output_hash": o.output_hash,
        }
        if o.translation_latency_ms > 0:
            t["translation_latency_ms"] = o.translation_latency_ms
        if o.submitted_at:
            t["produced_at"] = o.submitted_at
        translations.append(t)
    # Sort contributors by date
    contribs_sorted = sorted(b.contributors, key=lambda c: c["at"])
    return {
        "schema_version": SCHEMA_VERSION,
        "model": {"provider": b.provider, "id": b.model_id},
        "environment": {
            "tbl_version": b.tbl_version,
            "prompt_version": b.prompt_version,
        },
        "contributors": contribs_sorted,
        "translations": translations,
    }


def write_translations(buckets: dict[str, ModelBucket],
                       conflicts_resolved: dict[str, list[Observation]],
                       output_dir: Path, dry_run: bool) -> dict[str, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, dict] = {}
    for mid, b in buckets.items():
        doc = bucket_to_doc(b, conflicts_resolved[mid])
        slug = model_slug(mid)
        out_path = output_dir / f"{slug}.json"
        if not dry_run:
            tmp = out_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            tmp.replace(out_path)
        written[mid] = {"slug": slug, "path": out_path, "n": len(doc["translations"])}
    return written


def archive_old_submissions(submissions_dir: Path, dry_run: bool) -> Path | None:
    if not submissions_dir.exists():
        return None
    archive = submissions_dir.parent / "submissions_v1_archive"
    if not dry_run:
        if archive.exists():
            print(f"WARNING: archive dir already exists at {archive}; skipping archive step", file=sys.stderr)
            return archive
        shutil.move(str(submissions_dir), str(archive))
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="Actually write files (default: dry-run preview)")
    parser.add_argument("--submissions-dir", type=Path,
                        default=REPO_ROOT / "benchmark" / "data" / "submissions")
    parser.add_argument("--translations-dir", type=Path,
                        default=REPO_ROOT / "benchmark" / "data" / "translations")
    parser.add_argument("--keep-submissions", action="store_true",
                        help="Don't archive old submissions/ folder after writing")
    args = parser.parse_args()

    if not args.submissions_dir.is_dir():
        print(f"ERROR: {args.submissions_dir} does not exist", file=sys.stderr)
        return 1

    print(f"Reading submissions from {args.submissions_dir}")
    buckets = parse_submissions(args.submissions_dir)
    if not buckets:
        print("No submission files found.")
        return 1

    print(f"  Found {len(buckets)} unique models")
    total_obs = sum(len(b.observations) for b in buckets.values())
    print(f"  Total observations across all submissions: {total_obs}")

    # Resolve conflicts per model
    print()
    print("Conflict resolution per model:")
    all_warnings: list[str] = []
    conflicts_resolved: dict[str, list[Observation]] = {}
    for mid, b in sorted(buckets.items()):
        kept, warnings = resolve_conflicts(b.observations)
        conflicts_resolved[mid] = kept
        dropped = len(b.observations) - len(kept)
        slug = model_slug(mid)
        msg = f"  {mid:30s} → {slug+'.json':35s}  obs in={len(b.observations):3d}  kept={len(kept):3d}  dropped(dup)={dropped:3d}"
        if warnings:
            msg += f"  WITH {len(warnings)} HASH-CONFLICTS"
        print(msg)
        all_warnings.extend(warnings)

    if all_warnings:
        print()
        print(f"⚠ {len(all_warnings)} hash conflicts (most-recent wins):")
        for w in all_warnings:
            print(w)

    print()
    mode = "DRY-RUN (no writes)" if not args.apply else "WRITE MODE"
    print(f"=== {mode} ===")
    written = write_translations(buckets, conflicts_resolved, args.translations_dir, dry_run=not args.apply)

    print()
    print(f"Translations written: {args.translations_dir}")
    for mid, info in sorted(written.items()):
        prefix = "[DRY]" if not args.apply else "[OK ]"
        print(f"  {prefix}  {info['slug']:32s}  {info['n']:3d} translations")

    # Archive old submissions
    if args.apply and not args.keep_submissions:
        archive = archive_old_submissions(args.submissions_dir, dry_run=False)
        if archive:
            print(f"\nOld submissions archived to: {archive}")
    elif not args.apply:
        print(f"\n[DRY] Would archive {args.submissions_dir} -> submissions_v1_archive/")

    print()
    print("Next steps:")
    print("  1. Inspect benchmark/data/translations/*.json")
    print("  2. Run: python scripts/rejudge_all_via_poe.py --dry-run 10")
    print("  3. If OK: python scripts/rejudge_all_via_poe.py")
    print("  4. Verify benchmark/data/judgments/claude-opus-4-7-rubric-v2-poe.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
