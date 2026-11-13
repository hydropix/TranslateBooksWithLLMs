"""Renderer subprocess and baseline worktree checks."""

import subprocess

import pytest

from src.prompts.prompts import generate_translation_prompt
from tests.prompt_eval.arms import (
    REPO_ROOT,
    build_request,
    is_unchanged,
    render_arm,
    render_both,
    run_renderer,
)
from tests.prompt_eval.corpus import load_corpus
from tests.prompt_eval.model import RenderedPrompt


@pytest.fixture(scope="module")
def cases():
    return {c.id: c for c in load_corpus()}


def _worktrees() -> set[str]:
    out = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=REPO_ROOT,
                         capture_output=True, text=True, check=True).stdout
    return {line for line in out.splitlines() if line.startswith("worktree ")}


@pytest.mark.parametrize("case_id", ["T01_plain_paragraphs_en_fr", "T05_placeholders_en_fr"])
def test_render_matches_in_process_builder(cases, case_id):
    case = cases[case_id]
    rendered = render_arm(REPO_ROOT, [case])[case_id]
    assert isinstance(rendered, RenderedPrompt)
    assert rendered.system and rendered.user
    assert rendered.dropped_kwargs == ()

    kwargs = build_request(case)["kwargs"]
    expected = generate_translation_prompt(**kwargs, glossary_block="")
    assert (rendered.system, rendered.user) == (expected.system, expected.user)


def test_glossary_terms_reach_the_prompt(cases):
    case = cases["T08_glossary_en_fr"]
    request = build_request(case)
    assert "glossary_terms" not in request["kwargs"]["prompt_options"]
    rendered = render_arm(REPO_ROOT, [case])[case.id]
    assert "ravageur céleste" in rendered.system + rendered.user


def test_unknown_kwarg_is_reported(cases):
    request = build_request(cases["T01_plain_paragraphs_en_fr"])
    request["kwargs"]["nonexistent_option"] = 1
    result = run_renderer(REPO_ROOT, [request])["T01_plain_paragraphs_en_fr"]
    assert "nonexistent_option" in result["dropped_kwargs"]


def test_builder_error_is_reported_per_case(cases):
    good = build_request(cases["T01_plain_paragraphs_en_fr"])
    bad = dict(build_request(cases["T05_placeholders_en_fr"]), case_id="BROKEN")
    bad["kwargs"] = {"main_content": "x"}  # missing required positional args
    results = run_renderer(REPO_ROOT, [good, bad])
    assert "system" in results["T01_plain_paragraphs_en_fr"]
    assert results["BROKEN"]["error"].startswith("TypeError")


@pytest.mark.slow
def test_render_both_head_matches_untouched_cases_and_cleans_up(cases):
    before = _worktrees()
    selected = [cases["T01_plain_paragraphs_en_fr"], cases["T05_placeholders_en_fr"]]
    sha, baseline, candidate = render_both(selected, "HEAD")
    assert len(sha) == 40
    for case in selected:
        assert is_unchanged(baseline[case.id], candidate[case.id])
    assert _worktrees() == before
