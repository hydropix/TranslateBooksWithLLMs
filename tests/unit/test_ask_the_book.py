"""
Unit tests for Ask the Book (src/core/ask + /api/ask routes).

The LLM is replaced by a scripted provider, so these tests cover passage
splitting, retrieval, the out-of-scope gates and citation validation without
any network access.
"""
import sys
from pathlib import Path
from typing import List, Optional

import pytest
from flask import Flask

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.blueprints import ask_routes
from src.core.ask import BookIndex, BookQA, split_into_passages
from src.core.ask.answerer import (
    REASON_LLM_ERROR, REASON_MODEL_DECLINED, REASON_NO_MATCH, REASON_UNGROUNDED,
    language_rule, strip_reasoning, validate_citations,
)
from src.core.ask.retrieval import BM25, ollama_base_url, tokenize
from src.core.llm.base import LLMProvider, LLMResponse

BOOK = """# Chapter 1: Gardening

## 1.1 Tomatoes

Tomatoes need at least six hours of direct sunlight every day. Water them deeply
twice a week rather than a little every day, so the roots grow down.

## 1.2 Compost

Compost is made from kitchen scraps and dry leaves. Turn the pile every two weeks
so it gets enough air; a pile that smells bad is too wet.

# Chapter 2: Tools

A sturdy spade and a hand trowel cover most jobs in a small vegetable garden.

```
# not a heading inside a code fence
```
"""


class ScriptedProvider(LLMProvider):
    """Returns queued responses in order and records the prompts it saw."""

    def __init__(self, responses: List[Optional[str]]):
        super().__init__(model="fake")
        self.responses = list(responses)
        self.prompts: List[str] = []

    async def generate(self, prompt, timeout=0, system_prompt=None):
        self.prompts.append(prompt)
        content = self.responses.pop(0)
        return LLMResponse(content=content) if content is not None else None


def _index() -> BookIndex:
    return BookIndex.build(split_into_passages(BOOK))


def _passage_id(index: BookIndex, needle: str) -> int:
    return next(p.id for p in index.passages if needle in p.text)


# --- passages -------------------------------------------------------------

class TestPassages:
    def test_sections_are_tracked_and_code_fence_headings_ignored(self):
        passages = split_into_passages(BOOK, min_chars=0)
        sections = [p.section for p in passages]
        assert "Chapter 1: Gardening > 1.1 Tomatoes" in sections
        assert "Chapter 1: Gardening > 1.2 Compost" in sections
        assert not any("not a heading" in s for s in sections)

    def test_long_sections_are_split_and_ids_are_sequential(self):
        text = "# Big\n\n" + "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(30))
        passages = split_into_passages(text, max_chars=1000)
        assert len(passages) > 1
        assert all(len(p.text) <= 1000 for p in passages)
        assert [p.id for p in passages] == list(range(1, len(passages) + 1))

    def test_duplicate_passages_are_dropped(self):
        text = "# A\n\nThe same paragraph appears twice in this book.\n\n# B\n\nThe same paragraph appears twice in this book."
        assert len(split_into_passages(text, min_chars=0)) == 1

    def test_tiny_sections_fold_into_previous_passage_of_same_chapter(self):
        passages = split_into_passages(BOOK)  # default min_chars folds the short Compost section
        folded = next(p for p in passages if "Compost is made" in p.text)
        assert folded.section == "Chapter 1: Gardening > 1.1 Tomatoes"
        assert "1.2 Compost" in folded.text  # its heading is kept inline

    def test_plain_text_without_headings(self):
        passages = split_into_passages("First paragraph here.\n\nSecond paragraph here.")
        assert len(passages) == 1 and passages[0].section == ""


# --- retrieval ------------------------------------------------------------

class TestRetrieval:
    def test_tokenize_drops_stopwords_and_bigrams_cjk(self):
        assert tokenize("What is the KV cache?") == ["kv", "cache"]
        assert tokenize("推理引擎") == ["推理", "理引", "引擎"]

    def test_bm25_ranks_matching_document_first(self):
        bm25 = BM25(["tomatoes need sunlight", "compost needs air", "spade and trowel"])
        scores = bm25.scores("how much sunlight do tomatoes need")
        assert scores.index(max(scores)) == 0
        assert bm25.scores("quantum chromodynamics") == [0.0, 0.0, 0.0]

    def test_search_finds_relevant_passage(self):
        index = _index()
        hits = index.search("compost pile smells", k=2)
        assert "Compost is made" in hits[0].passage.text
        assert hits[0].similarity is None  # no embeddings configured

    @pytest.mark.parametrize("endpoint, expected", [
        ("http://localhost:11434/api/generate", "http://localhost:11434"),
        ("http://host:11434/api", "http://host:11434"),
        ("http://host:11434/", "http://host:11434"),
        (None, "http://localhost:11434"),
    ])
    def test_ollama_base_url(self, endpoint, expected):
        assert ollama_base_url(endpoint) == expected


# --- answering ------------------------------------------------------------

class TestBookQA:
    @pytest.mark.asyncio
    async def test_grounded_answer_with_citations(self):
        index = _index()
        pid = _passage_id(index, "six hours")
        provider = ScriptedProvider(["tomato sunlight", f"At least six hours of sun a day [P{pid}]."])
        result = await BookQA(index, provider).ask("Tamatar ko kitni dhoop chahiye?")
        assert result.in_scope
        assert result.search_query == "tomato sunlight"
        assert [c.passage_id for c in result.citations] == [pid]
        assert "Tamatar ko kitni dhoop chahiye?" in provider.prompts[1]

    @pytest.mark.asyncio
    async def test_no_lexical_match_is_out_of_scope_without_answer_call(self):
        provider = ScriptedProvider(["capital of australia"])
        result = await BookQA(_index(), provider).ask("What is the capital of Australia?")
        assert not result.in_scope and result.reason == REASON_NO_MATCH
        assert len(provider.prompts) == 1  # only the query rewrite ran

    @pytest.mark.asyncio
    async def test_model_declines(self):
        provider = ScriptedProvider(["tomatoes soup recipe", "OUT_OF_SCOPE"])
        result = await BookQA(_index(), provider).ask("Best tomato soup recipe?")
        assert not result.in_scope and result.reason == REASON_MODEL_DECLINED

    @pytest.mark.asyncio
    async def test_answer_without_citation_is_withheld(self):
        provider = ScriptedProvider(["tomato sunlight", "Tomatoes like lots of sun."])
        result = await BookQA(_index(), provider).ask("How much sun for tomatoes?")
        assert not result.in_scope and result.reason == REASON_UNGROUNDED
        assert result.answer == ""

    @pytest.mark.asyncio
    async def test_answer_citing_unknown_passage_is_withheld(self):
        provider = ScriptedProvider(["tomato sunlight", "Lots of sun [P999]."])
        result = await BookQA(_index(), provider).ask("How much sun for tomatoes?")
        assert result.reason == REASON_UNGROUNDED

    @pytest.mark.asyncio
    async def test_empty_llm_response(self):
        provider = ScriptedProvider(["tomato sunlight", None])
        result = await BookQA(_index(), provider).ask("How much sun for tomatoes?")
        assert result.reason == REASON_LLM_ERROR

    @pytest.mark.asyncio
    async def test_similarity_gate_with_embeddings(self):
        index = _index()
        index.embeddings = [[1.0, 0.0] for _ in index.passages]
        index.embedding_model = "fake-embed"
        import src.core.ask.retrieval as retrieval
        original = retrieval.embed_texts
        retrieval.embed_texts = lambda texts, model, endpoint, **kw: [[0.0, 1.0]]  # orthogonal query
        try:
            provider = ScriptedProvider(["tomato sunlight"])
            result = await BookQA(index, provider, min_similarity=0.5).ask("tomatoes?")
        finally:
            retrieval.embed_texts = original
        assert result.reason == REASON_NO_MATCH and result.top_similarity == 0.0

    @pytest.mark.asyncio
    async def test_empty_rewrite_is_retried(self):
        index = _index()
        pid = _passage_id(index, "six hours")
        provider = ScriptedProvider(["", "tomatoes sunlight", f"Six hours [P{pid}]."])
        result = await BookQA(index, provider).ask("Tamatar ko kitni dhoop chahiye?")
        assert result.search_query == "tomatoes sunlight" and result.in_scope

    @pytest.mark.asyncio
    async def test_rewrite_falls_back_to_question_after_retries(self):
        provider = ScriptedProvider(["", None, "OUT_OF_SCOPE"])
        result = await BookQA(_index(), provider).ask("tomatoes sunlight?")
        assert result.search_query == "tomatoes sunlight?"

    @pytest.mark.asyncio
    async def test_history_is_used_for_rewrite(self):
        index = _index()
        pid = _passage_id(index, "Compost is made")
        provider = ScriptedProvider(["compost air", f"Turn it every two weeks [P{pid}]."])
        await BookQA(index, provider).ask("how often?", [("What is compost?", "Scraps and leaves.")])
        assert "What is compost?" in provider.prompts[0]


class TestHelpers:
    def test_validate_citations(self):
        assert validate_citations("a [P1] b [P2][P1]", [1, 2, 3]) == (True, [1, 2])
        assert validate_citations("no cites", [1]) == (False, [])
        assert validate_citations("[P4]", [1, 2]) == (False, [4])

    def test_strip_reasoning(self):
        assert strip_reasoning("<think>hmm</think>Answer [P1]") == "Answer [P1]"
        assert strip_reasoning("truncated reasoning</think> Answer") == "Answer"

    def test_language_rule(self):
        assert "same language" in language_rule("auto")
        assert language_rule("Hinglish") == "Answer in Hinglish."


# --- routes ---------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "garden.txt").write_text(BOOK, encoding="utf-8")
    monkeypatch.setattr(ask_routes, "ASK_EMBEDDING_MODEL", "")
    app = Flask(__name__)
    app.register_blueprint(ask_routes.create_ask_blueprint(str(tmp_path)))
    with app.test_client() as c:
        c.uploads = uploads
        yield c


class TestRoutes:
    def test_index_reports_stats(self, client):
        resp = client.post("/api/ask/index", json={"file_path": str(client.uploads / "garden.txt")})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["passages"] == len(split_into_passages(BOOK)) and data["search"] == "bm25"

    def test_path_outside_uploads_is_rejected(self, client, tmp_path):
        outside = tmp_path / "secret.txt"
        outside.write_text("secret", encoding="utf-8")
        resp = client.post("/api/ask/index", json={"file_path": str(outside)})
        assert resp.status_code == 403

    def test_query_requires_question_and_model(self, client):
        path = str(client.uploads / "garden.txt")
        assert client.post("/api/ask/query", json={"file_path": path, "model": "m"}).status_code == 400
        assert client.post("/api/ask/query", json={"file_path": path, "question": "q"}).status_code == 400

    def test_query_returns_answer(self, client, monkeypatch):
        pid = next(p.id for p in split_into_passages(BOOK) if "Compost is made" in p.text)
        provider = ScriptedProvider(["compost air", f"Turn the pile every two weeks [P{pid}]."])
        monkeypatch.setattr(ask_routes, "_instantiate_provider", lambda data: provider)
        resp = client.post("/api/ask/query", json={
            "file_path": str(client.uploads / "garden.txt"), "question": "How often to turn compost?",
            "model": "fake", "history": [["What is compost?", "Scraps."], "malformed"],
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["in_scope"] and data["citations"][0]["passage_id"] == pid
