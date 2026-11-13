"""Baseline worktree lifecycle and the subprocess driver for render.py."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterator, Optional, Union

from tests.prompt_eval.model import Case, RenderedPrompt

REPO_ROOT = Path(__file__).resolve().parents[2]
RENDER_SCRIPT = Path(__file__).resolve().parent / "render.py"
RENDER_TIMEOUT_S = 120

RenderResult = Union[RenderedPrompt, str]


def _git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8"
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def resolve_baseline_ref(ref: Optional[str]) -> str:
    """Full sha of the baseline: ``ref``, or the merge-base of HEAD and main."""
    if ref is None:
        return _git("merge-base", "HEAD", "main")
    return _git("rev-parse", "--verify", f"{ref}^{{commit}}")


def head_sha() -> str:
    return _git("rev-parse", "HEAD")


def working_tree_dirty() -> bool:
    return bool(_git("status", "--porcelain"))


@contextlib.contextmanager
def baseline_worktree(sha: str) -> Iterator[Path]:
    """Check ``sha`` out in a temporary detached worktree, removed on exit."""
    path = Path(tempfile.mkdtemp(prefix="prompt_eval_baseline_"))
    try:
        _git("worktree", "add", "--detach", str(path), sha)
        yield path
    finally:
        try:
            _git("worktree", "remove", "--force", str(path))
        except RuntimeError:
            pass
        with contextlib.suppress(RuntimeError):
            _git("worktree", "prune")
        shutil.rmtree(path, ignore_errors=True)


def build_request(case: Case) -> dict:
    """Map a Case onto the builder kwargs (see the plan's builder registry)."""
    options = dict(case.prompt_options)
    glossary_terms = options.pop("glossary_terms", None)
    if case.builder == "translate":
        kwargs = {
            "main_content": case.text,
            "context_before": case.context_before,
            "context_after": case.context_after,
            "previous_translation_context": case.previous_context,
            "source_language": case.source_language,
            "target_language": case.target_language,
            "has_placeholders": case.has_placeholders,
            "prompt_options": options,
        }
    else:
        kwargs = {
            "draft_translation": case.text,
            "context_before": case.context_before,
            "context_after": case.context_after,
            "previous_refined_context": case.previous_context,
            "target_language": case.target_language,
            "has_placeholders": case.has_placeholders,
            "prompt_options": options,
        }
    return {
        "case_id": case.id,
        "builder": case.builder,
        "kwargs": kwargs,
        "glossary_terms": glossary_terms,
        "target_language": case.target_language,
    }


def run_renderer(root: Path, requests: list[dict]) -> dict:
    """Run render.py against ``root`` and return its raw ``results`` mapping."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    proc = subprocess.run(
        [sys.executable, str(RENDER_SCRIPT), "--root", str(root),
         "--env-file", str(REPO_ROOT / ".env")],
        cwd=REPO_ROOT,
        input=json.dumps({"requests": requests}, ensure_ascii=False).encode("utf-8"),
        capture_output=True,
        timeout=RENDER_TIMEOUT_S,
        env=env,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"render.py failed for {root} (exit {proc.returncode}):\n{stderr[-2000:]}")
    return json.loads(proc.stdout.decode("utf-8"))["results"]


def render_arm(root: Path, cases: list[Case]) -> dict[str, RenderResult]:
    results = run_renderer(root, [build_request(c) for c in cases])
    rendered: dict[str, RenderResult] = {}
    for case in cases:
        item = results.get(case.id)
        if item is None:
            rendered[case.id] = "MissingResult: renderer returned nothing for this case"
        elif "error" in item:
            rendered[case.id] = item["error"]
        else:
            rendered[case.id] = RenderedPrompt(
                system=item["system"], user=item["user"],
                dropped_kwargs=tuple(item.get("dropped_kwargs") or ()),
            )
    return rendered


def render_both(
    cases: list[Case], baseline_ref: Optional[str]
) -> tuple[str, dict[str, RenderResult], dict[str, RenderResult]]:
    """Render every case in the baseline tree and in the working tree."""
    sha = resolve_baseline_ref(baseline_ref)
    with baseline_worktree(sha) as tree:
        baseline = render_arm(tree, cases)
    candidate = render_arm(REPO_ROOT, cases)
    return sha, baseline, candidate


def is_unchanged(baseline: RenderResult, candidate: RenderResult) -> bool:
    return (
        isinstance(baseline, RenderedPrompt)
        and isinstance(candidate, RenderedPrompt)
        and baseline.system == candidate.system
        and baseline.user == candidate.user
    )
