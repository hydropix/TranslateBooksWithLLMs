"""Execute the target x repeat x case x arm matrix and persist every call."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from src.core.llm.factory import create_llm_provider
from tests.prompt_eval.arms import RenderResult, is_unchanged
from tests.prompt_eval.metrics import FAIL, score
from tests.prompt_eval.model import Case, RenderedPrompt, RunRecord, Target

ARMS = ("baseline", "candidate")

# Provider names accepted by src.core.llm.factory.create_llm_provider.
KNOWN_PROVIDERS = frozenset({
    "ollama", "openai", "gemini", "openrouter", "mistral", "deepseek", "poe", "nim", "litellm",
})


def parse_target(spec: str) -> Target:
    """``provider:model``, split on the first colon only."""
    provider, sep, model = spec.partition(":")
    provider = provider.strip().lower()
    if not sep or not model.strip():
        raise ValueError(f"Target must be PROVIDER:MODEL, got {spec!r}")
    if provider not in KNOWN_PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}; expected one of {sorted(KNOWN_PROVIDERS)}")
    return Target(provider=provider, model=model.strip())


@dataclass
class CaseSelection:
    to_run: list[Case] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    render_errors: dict[str, str] = field(default_factory=dict)   # case_id -> message


def select_cases(
    cases: list[Case],
    baseline: dict[str, RenderResult],
    candidate: dict[str, RenderResult],
    include_unchanged: bool,
) -> CaseSelection:
    selection = CaseSelection()
    for case in cases:
        b, c = baseline.get(case.id), candidate.get(case.id)
        if not isinstance(b, RenderedPrompt) or not isinstance(c, RenderedPrompt):
            messages = [f"{arm}: {r}" for arm, r in (("baseline", b), ("candidate", c))
                        if not isinstance(r, RenderedPrompt)]
            selection.render_errors[case.id] = "; ".join(messages)
            continue
        if is_unchanged(b, c):
            selection.unchanged.append(case.id)
            if not include_unchanged:
                continue
        selection.to_run.append(case)
    return selection


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


async def run_matrix(
    cases: list[Case],
    baseline: dict[str, RenderResult],
    candidate: dict[str, RenderResult],
    targets: list[Target],
    repeat: int,
    include_unchanged: bool,
    out_dir: Path,
    provider_factory: Callable = create_llm_provider,
) -> list[RunRecord]:
    selection = select_cases(cases, baseline, candidate, include_unchanged)
    prompts = {"baseline": baseline, "candidate": candidate}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[RunRecord] = []

    with (out_dir / "records.jsonl").open("w", encoding="utf-8") as sink:
        for target in targets:
            provider = provider_factory(target.provider, model=target.model)
            try:
                for rep in range(repeat):
                    for case in selection.to_run:
                        for arm in ARMS:
                            record = await _run_one(provider, target, rep, repeat, case, arm,
                                                    prompts[arm][case.id])
                            records.append(record)
                            sink.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                            sink.flush()
            finally:
                close = getattr(provider, "close", None)
                if close is not None:
                    try:
                        await close()
                    except Exception:
                        pass
    return records


async def _run_one(provider, target: Target, rep: int, repeat: int, case: Case, arm: str,
                   rendered: RenderedPrompt) -> RunRecord:
    started = time.perf_counter()
    raw: Optional[str] = None
    extracted: Optional[str] = None
    prompt_tokens = completion_tokens = None
    try:
        response = await provider.generate(rendered.user, system_prompt=rendered.system)
        if response is not None:
            raw = response.content
            extracted = provider.extract_translation(raw) if raw is not None else None
            prompt_tokens = getattr(response, "prompt_tokens", None)
            completion_tokens = getattr(response, "completion_tokens", None)
    except Exception as exc:
        # Type only: messages may echo endpoints or request details.
        _log(f"[{target}] request failed: {type(exc).__name__}")
        raw = None
    elapsed = time.perf_counter() - started

    metrics = score(case, raw, extracted)
    record = RunRecord(
        case_id=case.id, arm=arm, target=str(target), repeat=rep,
        raw=raw, extracted=extracted,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        elapsed_s=round(elapsed, 3), metrics=metrics,
    )
    if raw is None:
        status = "request failed"
    else:
        failed = [name for name, value in metrics.items() if value == FAIL]
        status = ", ".join(failed) if failed else "ok"
    _log(f"[{target}] r{rep + 1}/{repeat} {case.id} {arm}: {status} ({elapsed:.1f}s)")
    return record
