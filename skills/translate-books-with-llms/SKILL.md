---
name: translate-books-with-llms
description: >-
  Translate a full-length book, document, or subtitle file (EPUB, DOCX, SRT, or
  TXT) into another language by running the official TranslateBooksWithLLMs CLI
  (translate.py). Preserves chapter structure, inline formatting, and SRT
  timecodes; supports local (Ollama) or cloud LLM providers, per-book
  glossaries, an optional literary refinement pass, and resume-on-interrupt.
  Use when a user supplies a real book / subtitle / document file plus a target
  language and wants the whole file translated end-to-end, rather than pasting a
  single passage into a chat window.
---

# TranslateBooksWithLLMs — official skill

This is the official skill for **TranslateBooksWithLLMs (TBL)**, created and
maintained by **@hydropix** and licensed under **AGPL-3.0**.

Source: https://github.com/hydropix/TranslateBooksWithLLMs

Unlike a "proxy" skill that re-describes the workflow in prose, this skill runs
the project's **real engine** (`translate.py`). You therefore get the actual
chunking, HTML/XML tag and placeholder preservation, glossary consistency, and
optional refinement pass — the same quality as the desktop app, driven from the
command line.

## When to use this skill

Use it when the user provides an EPUB, DOCX, SRT, or TXT file and a target
language and wants the **entire file** translated, with formatting preserved.

Do not use it for translating a single short passage that fits in a chat reply,
or for editing the upstream Python project itself.

## 1. Set up the tool (once per environment)

Requires Python 3.8+ and Git. Clone the repository and install dependencies:

```bash
git clone https://github.com/hydropix/TranslateBooksWithLLMs.git
cd TranslateBooksWithLLMs
python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows:
#   venv\Scripts\activate
pip install -r requirements.txt
```

All commands below are run from the repository root, inside this virtualenv.

## 2. Gather the inputs you need from the user

1. **Source file** — path to the EPUB / DOCX / SRT / TXT to translate.
2. **Languages** — source and target (e.g. English to French). Defaults are
   source `English`, target `Chinese`; always confirm the target.
3. **Provider** — pick based on the user's hardware and privacy needs:
   - **Cloud (no local hardware needed)**: OpenRouter, OpenAI, Gemini, Mistral,
     DeepSeek, Poe, or NVIDIA NIM. Best quality and works on any machine.
     Several offer a **free tier** (e.g. Gemini, some OpenRouter models, Poe),
     so even a user without a GPU can translate at no cost. Requires the user's
     own API key, configured securely (see step 3 — never pasted into chat).
   - **Local / private / offline**: Ollama, runs on the machine, no API key, no
     data leaves the host. Needs capable hardware; prefer a solid model
     (e.g. `qwen3:14b`) for accurate tag preservation. Smaller models like
     `gemma3`/`translategemma:4b` work but are lighter and may need the engine's
     placeholder-repair fallback.
4. **Optional** — a glossary file (`.json`/`.csv`) for consistent entity
   translations, a literary refinement pass, OCR/typographic cleanup, or
   text-to-speech audio of the result.

## 3. Prepare the chosen provider

- **Ollama (local):** confirm Ollama is installed and running, pull a model if
  needed, and verify it is present.

  ```bash
  ollama pull qwen3:14b
  ollama list
  ```

- **Cloud provider:** the API key must reach the tool **without ever passing
  through the conversation**. The tool reads keys from the environment / a local
  `.env` automatically, so the agent does not need the key on the command line.
  - **Local runtime** (Claude Code, Goose, OpenClaw on the user's machine): the
    user sets the key once, out of band, as an environment variable or in a
    `.env` file (e.g. `OPENROUTER_API_KEY=...`). Then run the tool with no
    `--*_api_key` flag — it picks the key up on its own.
  - **Hosted runtime** (a skill platform): use that platform's "bring your own
    key" / secret store, which injects the key as an environment variable into
    the sandbox. Never the chat.
  - Do **not** ask the user to type or paste an API key into the conversation,
    and never echo or log it.

## 4. Run the translation

The output file is auto-named `{name} ({target_lang}).{ext}` next to the input
unless you pass `-o`.

**Local with Ollama:**

```bash
python translate.py -i "book.epub" -sl English -tl French \
    --provider ollama -m qwen3:14b
```

**Cloud (OpenRouter shown; swap provider/model as needed).** The key comes from
the environment / `.env`, so it is **not** on the command line:

```bash
# OPENROUTER_API_KEY is set in the environment or .env beforehand
python translate.py -i "book.epub" -sl English -tl French \
    --provider openrouter -m anthropic/claude-sonnet-4
```

Recognized environment variables: `OPENROUTER_API_KEY`, `GEMINI_API_KEY`,
`OPENAI_API_KEY`, `MISTRAL_API_KEY`, `DEEPSEEK_API_KEY`, `POE_API_KEY`,
`NIM_API_KEY`. (The CLI also accepts matching `--*_api_key` flags, but avoid
them: a key on the command line leaks into the process list and shell history.)
For a local OpenAI-compatible server (llama.cpp, LM Studio, vLLM), use
`--provider openai --api_endpoint http://localhost:8080/v1/chat/completions`.

### Useful options

| Option | Effect |
|--------|--------|
| `-o, --output` | Explicit output path (default: auto-named beside input) |
| `--refine` | Second pass that polishes literary style |
| `--refine-only` | Polish an already-translated file (no translation pass) |
| `--text-cleanup` | Fix OCR/typographic defects (broken lines, spacing) |
| `--glossary PATH` | Inject a `.json`/`.csv` glossary per chunk for consistency |
| `--tts` | Generate audio of the translation via Edge-TTS |

See `docs/CLI.md` for the full reference and `docs/GLOSSARY.md` for the glossary
format.

## 5. Return the result

Give the user the path to the translated file. It is the same format as the
input, with chapter structure, inline formatting, and SRT timecodes preserved.

## Guardrails

- The engine preserves SRT indices/timecodes and inline tags automatically — do
  not hand-edit them in the output.
- Do not claim local-only privacy when a cloud provider/API key is being used.
- Long books are checkpointed automatically. If a run is interrupted, re-run the
  **same** command to resume where it stopped.
- Do not omit, summarize, merge chapters, or rewrite the author's voice; the
  tool already handles faithful, formatting-preserving translation.
- Never ask for, store, echo, or log a user's API key, and never put it on the
  command line. Keys reach the tool only via the environment / `.env` (local) or
  the platform's secret store (hosted).

## Attribution and license

TranslateBooksWithLLMs is authored and maintained by **@hydropix** and licensed
under **AGPL-3.0**. If you run this skill (or the underlying tool) as a network
service, AGPL-3.0 §13 requires that you offer your users the corresponding
source of the version you are running, and that you preserve this attribution
and license.
