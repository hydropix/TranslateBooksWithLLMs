# Ask the Book

Ask questions about a book and get answers taken **only from the book's own text**, with citations to the passages used. Questions the book does not cover are answered with "out of scope" instead of the model's general knowledge.

> Available in the Web UI (**Ask** tab) and the CLI (`python -m src.core.ask`). Works with TXT, EPUB, DOCX and PDF.

---

## Table of contents

- [Why use it?](#why-use-it)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Out-of-scope detection](#out-of-scope-detection)
- [Answer language](#answer-language)
- [Configuration](#configuration)
- [CLI](#cli)
- [REST API](#rest-api)
- [Limitations](#limitations)

---

## Why use it?

- Check what a book actually says about a topic before (or after) translating it.
- Read a technical book in a language you are more comfortable with: ask in your language, get the answer in your language, with links back to the original passages.
- Avoid confident answers that come from the model's training data rather than the book: anything the book does not support is refused.

---

## Quick start

1. *(Recommended)* Pull an embedding model for semantic search and enable it in `.env`:

   ```bash
   ollama pull bge-m3
   ```

   ```env
   ASK_EMBEDDING_MODEL=bge-m3
   ```

2. Start the web UI and pick a provider and model in **Settings**, as for a translation. Any provider works (Ollama, OpenRouter, OpenAI, Gemini, ...).
3. Open the **Ask** tab, drop a book, and wait for indexing (a few seconds for keyword search; about 20 s for a 600k-character book with `bge-m3` on a mid-range GPU).
4. Ask a question. Each answer shows citation chips such as `P42`; expand **Sources from the book** to read the passages it is based on.

---

## How it works

For every question:

1. **Rewrite**: the LLM turns the question (plus the recent conversation) into one standalone search query in the book's language. This makes follow-ups ("and how much memory does it use?") and cross-language questions retrieve the right passages.
2. **Retrieve**: the book is split into passages of about 1,800 characters that keep their chapter/section headings. Passages are ranked with BM25 and, when an embedding model is configured, by semantic similarity; both rankings are merged with reciprocal rank fusion.
3. **Gate**: if nothing in the book is related enough to the question, it is reported as out of scope without asking the LLM for an answer.
4. **Answer**: the LLM receives the top passages (5 by default) and must answer only from them, citing each passage it uses as `[P<n>]`, or reply `OUT_OF_SCOPE`.
5. **Validate**: an answer with no citation, or one citing a passage it was not given, is withheld and reported as out of scope.

The index is cached in memory per file, so only the first question about a book pays the indexing cost.

---

## Out-of-scope detection

Each refusal carries a reason, shown under the message in the UI:

| Reason | Meaning |
|--------|---------|
| `no_match` | No passage is related enough to the question (retrieval gate). |
| `model_declined` | Related passages were found, but the LLM said they do not answer the question. |
| `ungrounded` | The LLM answered without valid citations, so the answer was withheld. |
| `llm_error` | The provider returned no answer. |

**Use an embedding model.** With BM25 alone, the retrieval gate only rejects questions that share no words with the book, so most of the filtering falls to the LLM. With `bge-m3`, the similarity gate separated in-book from off-topic questions on a 600k-character English technical book in 31 of 32 cases, and the one off-topic question that passed was then declined by the LLM.

Larger models follow the "answer only from the passages" rule more reliably than small local ones; with a small model, rely on the embedding gate and check the cited sources.

---

## Answer language

The **Answer language** selector (CLI: `--answer-language`) controls the reply:

- **Same as my question** (default): the model answers in the language and script of the question. Small models sometimes drift back to the book's language; pick a language explicitly if that happens.
- A specific language, e.g. **Hinglish** (Hindi in Roman script mixed with English), **English**, **French**, ...

Questions can be asked in any language: the search query is always rewritten into the book's language first.

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ASK_EMBEDDING_MODEL` | *(empty)* | Ollama embedding model for semantic search (e.g. `bge-m3`, `nomic-embed-text`). Empty = BM25 keyword search only. Embeddings always run on the Ollama endpoint (`OLLAMA_API_ENDPOINT`), whatever provider answers the questions. |
| `ASK_MIN_SIMILARITY` | `0.50` | With embeddings: minimum cosine similarity for a question to count as in scope. Calibrated for `bge-m3`; other embedding models may need another value. Raise it to refuse more readily, lower it if in-book questions are refused. |
| `ASK_TOP_K` | `5` | Number of passages the LLM receives per question. |

The answering model is the one selected in **Settings** (Web UI) or with `--provider` / `-m` (CLI), using that provider's API key from `.env`.

---

## CLI

```bash
# Local model, keyword search
python -m src.core.ask book.epub -m qwen3:14b

# Semantic search + Hinglish answers with OpenRouter (OPENROUTER_API_KEY in .env)
python -m src.core.ask book.pdf --provider openrouter -m <model> \
    --embedding-model bge-m3 --answer-language Hinglish
```

| Option | Description | Default |
|--------|-------------|---------|
| `file` | Book to ask about (TXT, EPUB, DOCX, PDF) | required |
| `--provider`, `-m/--model`, `--api_endpoint` | Same as the translation CLI | from `.env` |
| `--embedding-model` | Ollama embedding model | `ASK_EMBEDDING_MODEL` |
| `--embedding-endpoint` | Ollama endpoint for embeddings | `--api_endpoint` when the provider is Ollama |
| `--answer-language` | Answer language, e.g. `Hinglish` | `auto` (same as the question) |
| `--min-similarity`, `--top-k` | Override `ASK_MIN_SIMILARITY` / `ASK_TOP_K` | from `.env` |
| `--debug` | Print the search query, retrieval scores and refusal reason | off |

Inside the prompt: `/sources` prints the passages behind the last answer, `/reset` clears the conversation, `/quit` exits.

---

## REST API

Both endpoints require the per-session API token like every other `/api/` route, and only accept files inside the uploads directory.

**`POST /api/ask/index`**: index an uploaded book (cached).

```json
{ "file_path": "<path returned by /api/upload>" }
```

Response: `{"passages": 328, "language": "English", "chars": 638969, "search": "hybrid"}` (`search` is `bm25` without embeddings).

**`POST /api/ask/query`**: answer one question.

```json
{
  "file_path": "<path returned by /api/upload>",
  "question": "What does chapter 3 say about batching?",
  "history": [{"question": "...", "answer": "..."}],
  "answer_language": "auto",
  "provider": "ollama",
  "model": "qwen3:14b",
  "api_key": "__USE_ENV__",
  "api_endpoint": "http://localhost:11434/api/generate"
}
```

Response: `{"answer", "in_scope", "reason", "search_query", "citations": [{"passage_id", "section", "text"}], "top_similarity", "top_lexical"}`.

---

## Limitations

- **SRT files** are not supported (subtitles are not a useful source for questions).
- **Embeddings require Ollama.** Cloud-only setups get BM25 search; OpenAI-compatible embedding endpoints are a possible follow-up.
- **Scanned PDFs** have no text layer and cannot be indexed.
- The index lives in memory: it is rebuilt after a server restart, and at most 4 books are cached at once.
- Answers are only as good as the retrieved passages. Questions whose answer is spread across many chapters ("summarize the whole book") get partial answers.
