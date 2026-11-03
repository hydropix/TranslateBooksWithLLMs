"""
LLM corrector for the term-fix pass (smart mode).

Snippets are sent in numbered batches with each match wrapped in markers.
The model returns each snippet with the correct term in the form the
sentence needs. Every answer goes through a guard before it is accepted. A
rejected snippet is retried once on its own, then kept as it was and flagged
`needs_review` in the report.
"""
import asyncio
import re
from typing import Callable, Dict, List, Optional, Sequence

from src.core.term_fix.diffing import changes_outside
from src.core.term_fix.matcher import find_spans
from src.core.term_fix.models import (
    STATUS_FIXED,
    STATUS_NEEDS_REVIEW,
    STATUS_SKIPPED,
    Correction,
    SnippetItem,
)
from src.core.term_fix.rules import TermRule

MARK_OPEN = "⟦"
MARK_CLOSE = "⟧"

#: Snippets per request. Small enough that a weak model keeps the ids straight.
DEFAULT_BATCH_SIZE = 12
#: Words on each side of a match the model may touch (articles, adjectives).
GUARD_WINDOW_WORDS = 4

#: Item wrapper. Not "<s>": "</s>" is the end-of-sequence token of many
#: models, and Ollama stops generating at the first one.
ITEM_TAG = "snippet"
# The closing tag is optional: a model that stops right after the last item
# (or forgets one closing tag) still yields every item it wrote.
_ITEM_RE = re.compile(
    rf'<{ITEM_TAG} id="(\d+)">(.*?)(?=</{ITEM_TAG}>|<{ITEM_TAG} id="|\Z)', re.DOTALL)
# Placeholders the EPUB/DOCX pipelines use; a snippet must keep them verbatim.
_PLACEHOLDER_RE = re.compile(r"\[\[\d+\]\]|\[id\d+\]")


def mark_snippet(item: SnippetItem) -> str:
    """The snippet text with every match wrapped in ⟦ ⟧."""
    out, pos = [], 0
    for start, end, _rule in sorted(item.spans):
        out.append(item.text[pos:start])
        out.append(f"{MARK_OPEN}{item.text[start:end]}{MARK_CLOSE}")
        pos = end
    out.append(item.text[pos:])
    return "".join(out)


def parse_response(content: str) -> Dict[int, str]:
    """{id: text} from a '<snippet id="N">...</snippet>' response.

    Duplicates keep the first. Markers a model forgot to remove are stripped:
    they only ever wrapped the term, so dropping them cannot change the text.
    Edge whitespace is stripped too (it is the layout between items); the
    corrector restores the snippet's own edges, see `_with_original_edges`.
    """
    result: Dict[int, str] = {}
    for m in _ITEM_RE.finditer(content or ""):
        text = m.group(2).replace(MARK_OPEN, "").replace(MARK_CLOSE, "").strip()
        result.setdefault(int(m.group(1)), text)
    return result


def _with_original_edges(original: str, answer: str) -> str:
    """`answer` with the leading/trailing whitespace of `original`."""
    core = original.strip()
    if not core:
        return answer
    start = original.index(core[0])
    end = original.rindex(core[-1]) + 1
    return original[:start] + answer + original[end:]


def _stem(word: str) -> str:
    """Leading part of a replacement word that survives inflection."""
    if len(word) <= 4:
        return word
    return word[:max(4, (len(word) * 3 + 4) // 5)]


def check_correction(
    item: SnippetItem,
    corrected: str,
    rules: Sequence[TermRule],
) -> Optional[str]:
    """Return why `corrected` must be rejected, or None when it is acceptable."""
    if MARK_OPEN in corrected or MARK_CLOSE in corrected:
        return "markers left in the answer"
    item_rules = sorted({r for _s, _e, r in item.spans})
    if find_spans(corrected, [rules[r] for r in item_rules]):
        return "a wrong rendering is still present"
    if sorted(_PLACEHOLDER_RE.findall(corrected)) != sorted(_PLACEHOLDER_RE.findall(item.text)):
        return "placeholders changed"
    lowered = corrected.lower()
    for r in item_rules:
        for word in rules[r].replacement.split():
            if _stem(word).lower() not in lowered:
                return f"the correct term '{rules[r].replacement}' is missing"
    if corrected.strip() and item.text.strip():
        ratio = len(corrected) / max(1, len(item.text))
        if not 0.4 <= ratio <= 2.5 and abs(len(corrected) - len(item.text)) > 30:
            return "length changed too much"
    if changes_outside(item.text, corrected, [(s, e) for s, e, _r in item.spans],
                       GUARD_WINDOW_WORDS):
        return "text changed away from the term"
    return None


class SmartCorrector:
    """LLM-backed corrector. `llm_client` is an LLMClient (make_request)."""

    def __init__(
        self,
        llm_client,
        model_name: str,
        target_language: str,
        rules: Sequence[TermRule],
        context_manager=None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        parallel: int = 1,
        log_callback: Optional[Callable] = None,
        stats_callback: Optional[Callable] = None,
        check_interruption_callback: Optional[Callable] = None,
    ):
        self.llm_client = llm_client
        self.model_name = model_name
        self.target_language = target_language
        self.rules = list(rules)
        self.context_manager = context_manager
        self.batch_size = max(1, batch_size)
        self.parallel = max(1, parallel)
        self.log_callback = log_callback
        self.stats_callback = stats_callback
        self.check_interruption_callback = check_interruption_callback

    def _log(self, key: str, message: str, data=None) -> None:
        if self.log_callback:
            if data is None:
                self.log_callback(key, message)
            else:
                self.log_callback(key, message, data=data)

    def _interrupted(self) -> bool:
        return bool(self.check_interruption_callback and self.check_interruption_callback())

    async def _ask(self, items: Sequence[SnippetItem]) -> Dict[int, str]:
        from src.prompts.prompts import generate_term_fix_prompt
        from src.core.epub.xhtml_translator import _request_refinement_with_adaptive_context

        used = sorted({r for item in items for _s, _e, r in item.spans})
        prompt = generate_term_fix_prompt(
            snippets=[mark_snippet(item) for item in items],
            rules=[self.rules[r] for r in used],
            target_language=self.target_language,
        )
        self._log("llm_request", "Sending term-fix request to LLM", data={
            'type': 'llm_request', 'system_prompt': prompt.system,
            'user_prompt': prompt.user, 'model': self.model_name,
        })
        response = await _request_refinement_with_adaptive_context(
            self.llm_client, self.context_manager, prompt, self.model_name, self.log_callback
        )
        content = response.content if response else ""
        self._log("llm_response", "LLM Response received", data={
            'type': 'llm_response', 'response': content, 'model': self.model_name,
        })
        return parse_response(content)

    def _judge(self, item: SnippetItem, answer: Optional[str]):
        """(answer with the snippet's own edge whitespace, rejection reason or None)."""
        if answer is None:
            return None, "no answer"
        answer = _with_original_edges(item.text, answer)
        return answer, check_correction(item, answer, self.rules)

    async def _correct_batch(self, items: Sequence[SnippetItem]) -> List[Correction]:
        from src.core.llm.exceptions import RateLimitError

        try:
            answers = await self._ask(items)
        except RateLimitError:
            raise
        except Exception as e:
            self._log("term_fix_batch_error", f"⚠️ Term-fix request failed: {e}")
            answers = {}

        results: List[Optional[Correction]] = []
        for idx, item in enumerate(items, start=1):
            answer, reason = self._judge(item, answers.get(idx))
            results.append(Correction(answer, STATUS_FIXED) if reason is None else None)

        # Retry each rejected snippet once, alone: a batch failure is often one
        # confused item, and a single-item prompt is much easier to follow.
        for pos, item in enumerate(items):
            if results[pos] is not None:
                continue
            reason = "no answer"
            try:
                answer, reason = self._judge(item, (await self._ask([item])).get(1))
            except RateLimitError:
                raise
            except Exception as e:
                answer, reason = None, f"request failed: {e}"
            if reason is None:
                results[pos] = Correction(answer, STATUS_FIXED)
            else:
                results[pos] = Correction(None, STATUS_NEEDS_REVIEW, reason)
        return results

    async def correct(self, items: Sequence[SnippetItem]) -> List[Correction]:
        batches = [list(items[i:i + self.batch_size])
                   for i in range(0, len(items), self.batch_size)]
        results: List[Optional[List[Correction]]] = [None] * len(batches)
        done = 0
        semaphore = asyncio.Semaphore(self.parallel)

        async def run(index: int) -> None:
            nonlocal done
            async with semaphore:
                if self._interrupted():
                    return
                results[index] = await self._correct_batch(batches[index])
                done += 1
                if self.stats_callback:
                    self.stats_callback({'total_chunks': len(batches),
                                         'completed_chunks': done, 'failed_chunks': 0})

        if self.stats_callback:
            self.stats_callback({'total_chunks': len(batches),
                                 'completed_chunks': 0, 'failed_chunks': 0})
        self._log("term_fix_smart_start",
                  f"Term fix: {len(items)} snippet(s) in {len(batches)} LLM request(s)")
        await asyncio.gather(*(run(i) for i in range(len(batches))))

        out: List[Correction] = []
        for batch, batch_results in zip(batches, results):
            if batch_results is None:
                out.extend(Correction(None, STATUS_SKIPPED, "interrupted") for _ in batch)
            else:
                out.extend(batch_results)
        return out
