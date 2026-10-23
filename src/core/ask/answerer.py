"""
Answer questions about a book using only its own text.

Pipeline per question:
  1. Rewrite the question (plus recent conversation) into a standalone search
     query in the book's language, so follow-ups and cross-language questions
     still retrieve well.
  2. Retrieve the top passages (BM25, fused with embeddings when enabled).
  3. Gate: if retrieval finds nothing relevant, answer "out of scope" without
     calling the LLM for an answer.
  4. Ask the LLM to answer strictly from the passages with [P<n>] citations, or
     to reply OUT_OF_SCOPE.
  5. Validate: an answer with no citation, or citing passages it was not
     given, is treated as out of scope rather than shown as grounded.
"""
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from src.core.llm.base import LLMProvider

from .retrieval import BookIndex, Hit

logger = logging.getLogger(__name__)

REWRITE_ATTEMPTS = 2

OUT_OF_SCOPE_TOKEN = "OUT_OF_SCOPE"

# Out-of-scope reasons, exposed to the UI so it can explain the refusal.
REASON_NO_MATCH = "no_match"            # retrieval found nothing relevant
REASON_MODEL_DECLINED = "model_declined"  # the LLM said the passages don't answer it
REASON_UNGROUNDED = "ungrounded"        # the LLM answered without valid citations
REASON_LLM_ERROR = "llm_error"          # the provider returned nothing

DEFAULT_TOP_K = 5
# Dense-similarity floor below which the best passage is considered unrelated.
# Only applies when embeddings are enabled; tune with ASK_MIN_SIMILARITY.
DEFAULT_MIN_SIMILARITY = 0.50

REWRITE_PROMPT = """Rewrite the user's latest question as ONE standalone search query, written in {book_language}.
- Translate the question into {book_language} if it is in another language or a mix of languages.
- Resolve references such as "it" or "that" using the conversation.
- Keep the question's own topic and technical terms. Do NOT add words or topics the user did not ask about.
Output only the query, nothing else.

Conversation so far:
{history}

Latest question: {question}
Search query:"""

ANSWER_SYSTEM_PROMPT = """You answer a reader's questions about one book, using ONLY the numbered book passages you are given.
Rules:
- Never use outside knowledge. If the passages do not contain the answer, reply with exactly: OUT_OF_SCOPE
- If the passages only partly answer the question, answer that part and say the book does not go into more detail.
- Cite the passages you used inline, like [P3] or [P3][P7]. Every answer must cite at least one passage.
- Be clear and concise (under about 200 words unless the question needs more).
- {language_rule}"""

ANSWER_PROMPT = """{history_block}Book passages:
{passages}

Question: {question}
({language_rule} Cite passages like [P3].)
Answer:"""

_CITATION_RE = re.compile(r"\[P(\d+)\]")


@dataclass
class Citation:
    passage_id: int
    section: str
    text: str


@dataclass
class AskResult:
    answer: str
    in_scope: bool
    reason: Optional[str] = None
    search_query: str = ""
    citations: List[Citation] = field(default_factory=list)
    top_similarity: Optional[float] = None
    top_lexical: float = 0.0

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "in_scope": self.in_scope,
            "reason": self.reason,
            "search_query": self.search_query,
            "citations": [c.__dict__ for c in self.citations],
            "top_similarity": self.top_similarity,
            "top_lexical": round(self.top_lexical, 3),
        }


def language_rule(answer_language: Optional[str]) -> str:
    if not answer_language or answer_language.lower() == "auto":
        return ("Answer in the same language and script as the question "
                "(for example, a question in Hinglish gets a Hinglish answer in Roman script).")
    return f"Answer in {answer_language}."


def format_history(history: Sequence[Tuple[str, str]], max_turns: int = 3, max_answer_chars: int = 400) -> str:
    return "\n".join(f"Q: {q}\nA: {a[:max_answer_chars]}" for q, a in list(history)[-max_turns:])


def strip_reasoning(text: str) -> str:
    """Drop <think> blocks emitted by reasoning models (including a truncated opening tag)."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE)
    if re.search(r"</think>", text, re.IGNORECASE):
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    return text.strip()


def validate_citations(answer: str, allowed_ids: Sequence[int]) -> Tuple[bool, List[int]]:
    """Return (is_grounded, cited_ids). Grounded means >=1 citation and all cite given passages."""
    cited = [int(x) for x in _CITATION_RE.findall(answer)]
    unique = list(dict.fromkeys(cited))
    return bool(unique) and set(unique) <= set(allowed_ids), unique


class BookQA:
    """Question answering over one indexed book."""

    def __init__(self, index: BookIndex, provider: LLMProvider, book_language: str = "English",
                 top_k: int = DEFAULT_TOP_K, min_similarity: float = DEFAULT_MIN_SIMILARITY):
        self.index = index
        self.provider = provider
        self.book_language = book_language or "English"
        self.top_k = top_k
        self.min_similarity = min_similarity

    async def _generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        response = await self.provider.generate(prompt, system_prompt=system_prompt)
        return strip_reasoning(response.content) if response and response.content else ""

    async def rewrite_query(self, question: str, history: Sequence[Tuple[str, str]]) -> str:
        prompt = REWRITE_PROMPT.format(book_language=self.book_language,
                                       history=format_history(history) or "(none)", question=question)
        # Some providers (notably free/reasoning models) occasionally return empty
        # content; searching with the raw question instead hurts cross-language
        # retrieval, so retry once before falling back.
        for _ in range(REWRITE_ATTEMPTS):
            rewritten = await self._generate(prompt)
            first_line = rewritten.strip().splitlines()[0].strip().strip('"') if rewritten.strip() else ""
            if first_line:
                return first_line
        logger.warning("Ask the Book: query rewrite returned no text; searching with the original question")
        return question

    def is_relevant(self, hits: List[Hit]) -> bool:
        """Retrieval gate: does anything in the book plausibly relate to the query?"""
        if not hits:
            return False
        if hits[0].similarity is not None:
            return max(h.similarity for h in hits) >= self.min_similarity
        return any(h.lexical > 0 for h in hits)

    async def ask(self, question: str, history: Sequence[Tuple[str, str]] = (),
                  answer_language: Optional[str] = "auto") -> AskResult:
        query = await self.rewrite_query(question, history)
        hits = self.index.search(query, self.top_k)
        top_sim = max((h.similarity for h in hits if h.similarity is not None), default=None)
        top_lex = max((h.lexical for h in hits), default=0.0)

        def out_of_scope(reason: str) -> AskResult:
            return AskResult("", False, reason, query, top_similarity=top_sim, top_lexical=top_lex)

        if not self.is_relevant(hits):
            return out_of_scope(REASON_NO_MATCH)

        passages = "\n\n".join(f"[P{h.passage.id}] ({h.passage.section or 'Untitled section'})\n{h.passage.text}"
                               for h in hits)
        history_text = format_history(history)
        answer = await self._generate(
            ANSWER_PROMPT.format(
                history_block=f"Earlier conversation:\n{history_text}\n\n" if history_text else "",
                passages=passages, question=question, language_rule=language_rule(answer_language)),
            system_prompt=ANSWER_SYSTEM_PROMPT.format(language_rule=language_rule(answer_language)),
        )
        if not answer:
            return out_of_scope(REASON_LLM_ERROR)
        if OUT_OF_SCOPE_TOKEN in answer:
            return out_of_scope(REASON_MODEL_DECLINED)

        grounded, cited = validate_citations(answer, [h.passage.id for h in hits])
        if not grounded:
            return out_of_scope(REASON_UNGROUNDED)

        by_id = {h.passage.id: h.passage for h in hits}
        citations = [Citation(pid, by_id[pid].section, by_id[pid].text) for pid in cited]
        return AskResult(answer, True, None, query, citations, top_sim, top_lex)
