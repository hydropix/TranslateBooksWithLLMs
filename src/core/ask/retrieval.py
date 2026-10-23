"""
Passage retrieval for Ask the Book.

Lexical BM25 always runs (pure Python, no extra dependencies). When an Ollama
embedding model is configured, dense cosine similarity is added and both
rankings are merged with reciprocal rank fusion. Dense similarity is also the
stronger "is this question about the book at all?" signal.
"""
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import httpx

from .passages import Passage

# Characters from scripts written without spaces are indexed as overlapping bigrams.
_NO_SPACE_SCRIPTS = ("CJK", "HIRAGANA", "KATAKANA", "HANGUL", "THAI")

# Very common words carry no signal for "does the book talk about this?".
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i in is it its of on or "
    "the this that to was what when where which who why will with you your".split()
)


def _is_no_space_char(ch: str) -> bool:
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return False
    return name.startswith(_NO_SPACE_SCRIPTS)


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens; runs of CJK/Thai characters become character bigrams."""
    tokens: List[str] = []
    for word in re.findall(r"\w+", text.lower()):
        if any(_is_no_space_char(c) for c in word):
            chars = [c for c in word if not c.isdigit()]
            tokens.extend(a + b for a, b in zip(chars, chars[1:])) if len(chars) > 1 else tokens.extend(chars)
        elif word not in _STOPWORDS and (len(word) > 1 or word.isdigit()):
            tokens.append(word)
    return tokens


class BM25:
    """Okapi BM25 over a fixed list of documents."""

    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [Counter(tokenize(d)) for d in documents]
        self.doc_len = [sum(c.values()) for c in self.doc_tokens]
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        df: Counter = Counter()
        for counts in self.doc_tokens:
            df.update(counts.keys())
        n = len(documents)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> List[float]:
        terms = [t for t in tokenize(query) if t in self.idf]
        out = []
        for counts, length in zip(self.doc_tokens, self.doc_len):
            score = 0.0
            for t in terms:
                tf = counts.get(t, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
                    score += self.idf[t] * tf * (self.k1 + 1) / norm
            out.append(score)
        return out


def ollama_base_url(endpoint: Optional[str]) -> str:
    """Turn a configured Ollama endpoint (often .../api/generate) into its base URL."""
    url = (endpoint or "http://localhost:11434").rstrip("/")
    return re.sub(r"/api(/\w+)?$", "", url)


def embed_texts(texts: Sequence[str], model: str, endpoint: Optional[str],
                batch_size: int = 32, timeout: float = 300) -> List[List[float]]:
    """Embed texts with Ollama's /api/embed; returns L2-normalised vectors."""
    url = ollama_base_url(endpoint) + "/api/embed"
    vectors: List[List[float]] = []
    with httpx.Client(timeout=timeout) as client:
        for i in range(0, len(texts), batch_size):
            resp = client.post(url, json={"model": model, "input": list(texts[i:i + batch_size])})
            resp.raise_for_status()
            vectors.extend(resp.json()["embeddings"])
    return [_normalise(v) for v in vectors]


def _normalise(vec: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


@dataclass
class Hit:
    passage: Passage
    lexical: float            # raw BM25 score
    similarity: Optional[float]  # cosine similarity, when embeddings are enabled


class BookIndex:
    """Searchable index over one document's passages."""

    def __init__(self, passages: List[Passage], embeddings: Optional[List[List[float]]] = None,
                 embedding_model: Optional[str] = None, embedding_endpoint: Optional[str] = None):
        self.passages = passages
        self.bm25 = BM25([f"{p.section}\n{p.text}" for p in passages])
        self.embeddings = embeddings
        self.embedding_model = embedding_model
        self.embedding_endpoint = embedding_endpoint

    @classmethod
    def build(cls, passages: List[Passage], embedding_model: Optional[str] = None,
              embedding_endpoint: Optional[str] = None) -> "BookIndex":
        embeddings = None
        if embedding_model and passages:
            embeddings = embed_texts([f"{p.section}\n{p.text}" for p in passages],
                                     embedding_model, embedding_endpoint)
        return cls(passages, embeddings, embedding_model, embedding_endpoint)

    @property
    def has_embeddings(self) -> bool:
        return self.embeddings is not None

    def search(self, query: str, k: int = 5) -> List[Hit]:
        """Top-k passages by BM25, fused with dense similarity when available."""
        if not self.passages:
            return []
        lexical = self.bm25.scores(query)
        similarity: Optional[List[float]] = None
        if self.embeddings is not None:
            q = embed_texts([query], self.embedding_model, self.embedding_endpoint)[0]
            similarity = [sum(a * b for a, b in zip(q, v)) for v in self.embeddings]

        rankings = [sorted(range(len(lexical)), key=lambda i: -lexical[i])]
        if similarity is not None:
            rankings.append(sorted(range(len(similarity)), key=lambda i: -similarity[i]))
        fused: Dict[int, float] = {}
        for ranking in rankings:
            for rank, idx in enumerate(ranking[:50]):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (60 + rank)
        best = sorted(fused, key=lambda i: -fused[i])[:k]
        return [Hit(self.passages[i], lexical[i], similarity[i] if similarity else None) for i in best]
