"""
One-shot archive of the v1 benchmark wiki.

Renames every existing Markdown page at the wiki root with an `Archive-`
prefix, rewrites internal cross-page links inside those pages so they keep
pointing to each other (not to the future v2 pages of the same name), and
generates an `Archive-Index.md` landing page that lists the archived content.

Run this BEFORE merging the v2 benchmark to `main`. After this commit on the
wiki, the `publish-wiki` GitHub Action regenerates the standard pages
(`Home.md`, `All-Languages.md`, etc.) which sit next to the `Archive-*` pages
without colliding.

Usage:

    python scripts/archive_v1_wiki.py [--dry-run] [--no-push] [--message MSG]

Environment:

    WIKI_REPO_URL — wiki git URL. Defaults to the project's wiki repo.
                    Use the form `https://x-access-token:<TOKEN>@github.com/<owner>/<repo>.wiki.git`
                    to push without prompting for credentials.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WIKI_URL = os.getenv(
    "WIKI_REPO_URL",
    "https://github.com/hydropix/TranslateBookWithLLM.wiki.git",
)
CLONE_DIR = REPO_ROOT / ".wiki_repo_archive"


def run_git(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    """Run a git command in `cwd`. Exits on failure unless check=False."""
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        print(f"git {' '.join(args)} failed:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    return result


def clone_wiki(url: str, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    print(f"Cloning {url} -> {dest}")
    result = subprocess.run(
        ["git", "clone", url, str(dest)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Clone failed:\n{result.stderr}", file=sys.stderr)
        print(
            "If the wiki has no pages yet, create at least one page on GitHub "
            "before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)


def list_v1_pages(wiki_dir: Path) -> list[Path]:
    """Markdown files at the wiki root that are not already archived."""
    return sorted(
        p for p in wiki_dir.glob("*.md")
        if p.is_file() and not p.name.startswith("Archive-")
    )


_LINK_RE = re.compile(r"\]\(([^)]+)\)")


def rewrite_links(content: str, renamed_pages: dict[str, str]) -> str:
    """
    Inside a markdown body, replace `](Old-Page)` with `](Archive-Old-Page)`
    when `Old-Page` is in `renamed_pages`.

    Skips external URLs (anything containing `://`), anchors (`#...`),
    and paths with slashes.
    """

    def replacer(match: re.Match[str]) -> str:
        target = match.group(1).strip()
        if not target or target.startswith("#") or "://" in target or "/" in target:
            return match.group(0)
        # Page references can have or omit `.md`. Normalize.
        bare = target.removesuffix(".md")
        if bare in renamed_pages:
            new_target = renamed_pages[bare]
            # Preserve `.md` suffix if it was there.
            if target.endswith(".md"):
                new_target = f"{new_target}.md"
            return f"]({new_target})"
        return match.group(0)

    return _LINK_RE.sub(replacer, content)


def build_index(archived: list[str]) -> str:
    """Generate Archive-Index.md content from the list of archived filenames."""
    home = [n for n in archived if n == "Archive-Home.md"]
    overview = [
        n for n in archived
        if n in ("Archive-All-Languages.md", "Archive-All-Models.md")
    ]
    languages = sorted(n for n in archived if n.startswith("Archive-Language-"))
    models = sorted(n for n in archived if n.startswith("Archive-Model-"))
    others = sorted(
        n for n in archived
        if n not in (*home, *overview, *languages, *models)
    )

    lines: list[str] = [
        "# Archived Benchmark (v1)",
        "",
        "These pages are the previous version of the TranslateBookWithLLM benchmark,",
        "kept here for historical reference.",
        "",
        "## What changed in v2",
        "",
        "- **Stronger judge.** v1 used `gemini-3-flash-preview`; v2 uses Claude Opus 4.7 applying a formal rubric (`docs/JUDGE_RUBRIC.md`).",
        "- **Multiple source languages.** v1 only translated from English. v2 covers 8 canonical pairs, including `ja:en`, `ko:en`, `zh-Hans:en`, and `ja:zh-Hans`.",
        "- **17 reference texts** across literary, scientific, philosophical, narrative and essay registers (vs 5 19th-c. English novels in v1).",
        "- **Aggregated scoring.** Multiple contributors can submit results for the same `(model, text, language)` triple; the wiki shows the median and the number of observations.",
        "",
        "For the current benchmark see [Home](Home).",
        "",
        "---",
        "",
        "## Archived pages",
        "",
    ]

    if home:
        lines.append("### Landing page")
        lines.append("")
        for n in home:
            page = n.removesuffix(".md")
            lines.append(f"- [Archived Home]({page})")
        lines.append("")

    if overview:
        lines.append("### Cross-cutting tables")
        lines.append("")
        for n in overview:
            page = n.removesuffix(".md")
            label = (
                page.replace("Archive-All-Languages", "All Languages")
                    .replace("Archive-All-Models", "All Models")
            )
            lines.append(f"- [{label}]({page})")
        lines.append("")

    if languages:
        lines.append("### Per-language pages")
        lines.append("")
        for n in languages:
            page = n.removesuffix(".md")
            label = page.replace("Archive-Language-", "").replace("-", " ").strip().title()
            lines.append(f"- [{label}]({page})")
        lines.append("")

    if models:
        lines.append("### Per-model pages")
        lines.append("")
        for n in models:
            page = n.removesuffix(".md")
            label = page.replace("Archive-Model-", "").replace("-", " ").strip()
            lines.append(f"- [{label}]({page})")
        lines.append("")

    if others:
        lines.append("### Other archived pages")
        lines.append("")
        for n in others:
            page = n.removesuffix(".md")
            lines.append(f"- [{page}]({page})")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def archive(wiki_dir: Path, *, dry_run: bool) -> int:
    pages = list_v1_pages(wiki_dir)
    if not pages:
        print("No v1 pages to archive (no *.md at root, or all already prefixed).")
        return 0

    renamed: dict[str, str] = {p.stem: f"Archive-{p.stem}" for p in pages}

    print(f"Found {len(pages)} v1 page(s) to archive:")
    for p in pages:
        print(f"  {p.name:40s} -> Archive-{p.name}")

    if dry_run:
        print()
        print("[dry-run] no changes made.")
        return len(pages)

    # 1. Rewrite cross-references in every page that's about to be archived.
    for p in pages:
        original = p.read_text(encoding="utf-8")
        rewritten = rewrite_links(original, renamed)
        if rewritten != original:
            p.write_text(rewritten, encoding="utf-8")

    # 2. Rename the files.
    archived_names: list[str] = []
    for p in pages:
        new_path = p.parent / f"Archive-{p.name}"
        p.rename(new_path)
        archived_names.append(new_path.name)

    # 3. Generate the index.
    index_content = build_index(archived_names)
    (wiki_dir / "Archive-Index.md").write_text(index_content, encoding="utf-8")
    archived_names.append("Archive-Index.md")

    print()
    print(f"Wrote Archive-Index.md and renamed {len(pages)} page(s).")
    return len(pages)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Clone, list what would be archived, but make no changes.",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Commit changes to the wiki clone but don't push.",
    )
    parser.add_argument(
        "--message",
        default="Archive v1 benchmark — renamed pages with Archive- prefix",
        help="Commit message used on the wiki.",
    )
    parser.add_argument(
        "--wiki-url",
        default=DEFAULT_WIKI_URL,
        help="Wiki git URL (defaults to project's wiki repo or $WIKI_REPO_URL).",
    )
    args = parser.parse_args()

    clone_wiki(args.wiki_url, CLONE_DIR)

    n_archived = archive(CLONE_DIR, dry_run=args.dry_run)
    if args.dry_run:
        print(f"\nClone left at {CLONE_DIR} for inspection. Re-run without --dry-run to apply.")
        return 0

    if n_archived == 0:
        return 0

    # Commit + push
    run_git(["add", "-A"], CLONE_DIR)
    status = run_git(["status", "--porcelain"], CLONE_DIR, check=False)
    if not status.stdout.strip():
        print("No changes to commit.")
        return 0

    run_git(["commit", "-m", args.message], CLONE_DIR)

    if args.no_push:
        print(f"\nCommitted in {CLONE_DIR}. Push manually when ready:")
        print(f"  git -C {CLONE_DIR} push")
        return 0

    print("Pushing to wiki remote...")
    push_result = run_git(["push"], CLONE_DIR, check=False)
    if push_result.returncode != 0:
        print("Push failed:", file=sys.stderr)
        print(push_result.stderr, file=sys.stderr)
        print(
            "\nThe wiki is committed locally. Set WIKI_REPO_URL with a token "
            "or push manually from "
            f"{CLONE_DIR}.",
            file=sys.stderr,
        )
        return 1

    print("\nDone. The wiki home is now `Home.md` (will be regenerated by the v2 publish workflow).")
    print("Archived content lives under the `Archive-*` prefix and `Archive-Index.md`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
