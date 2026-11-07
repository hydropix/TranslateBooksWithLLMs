"""Optional LLM judge, reusing the benchmark evaluator.

Scores are informative only: they never change the verdict. Refine records
are not judged in v1.
"""

from __future__ import annotations

import sys
from typing import Iterable

from tests.prompt_eval.model import Case, RunRecord


def judge_eligible(record: RunRecord, case: Case) -> bool:
    return case.builder == "translate" and bool(record.extracted and record.extracted.strip())


def _reference_text(case: Case):
    from benchmark.models import ReferenceText
    from src.core.common.source_residue import get_language_code
    return ReferenceText(
        id=case.id,
        title=case.id,
        author="",
        year=0,
        content=case.text,
        style=case.notes,
        source_language=get_language_code(case.source_language) or case.source_language,
    )


async def judge_records(records: Iterable[RunRecord], cases: Iterable[Case], provider: str) -> None:
    """Fill ``record.judge`` on eligible translate records. Failures leave it None."""
    from benchmark.config import BenchmarkConfig
    from benchmark.evaluator import TranslationEvaluator
    from src.core.common.source_residue import get_language_code

    by_id = {case.id: case for case in cases}
    evaluator = TranslationEvaluator(BenchmarkConfig.from_env(), provider=provider)
    try:
        for record in records:
            case = by_id.get(record.case_id)
            if case is None or not judge_eligible(record, case):
                continue
            try:
                scores, _tokens = await evaluator.evaluate(
                    _reference_text(case),
                    record.extracted,
                    get_language_code(case.target_language) or case.target_language,
                    case.target_language,
                )
                record.judge = scores.to_dict()
            except Exception as exc:
                record.judge = None
                print(f"judge failed on {record.case_id} {record.arm}: {type(exc).__name__}",
                      file=sys.stderr, flush=True)
    finally:
        close = getattr(evaluator, "close", None)
        if close is not None:
            try:
                await close()
            except Exception:
                pass


def count_judge_failures(records: Iterable[RunRecord], cases: Iterable[Case]) -> int:
    by_id = {case.id: case for case in cases}
    return sum(
        1 for r in records
        if r.case_id in by_id and judge_eligible(r, by_id[r.case_id]) and r.judge is None
    )
