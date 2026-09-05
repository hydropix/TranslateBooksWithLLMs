<div align="center">
  <h1>TBL - Translate Books with LLMs</h1>
</div>

<div align="center">

[![Download Windows](https://img.shields.io/badge/Download-Windows-blue?style=for-the-badge&logo=windows)](https://github.com/hydropix/TranslateBooksWithLLMs/releases/latest/download/TranslateBook-Windows.zip) [![Download macOS Intel](https://img.shields.io/badge/Download-macOS%20Intel-black?style=for-the-badge&logo=apple)](https://github.com/hydropix/TranslateBooksWithLLMs/releases/latest/download/TranslateBook-macOS-Intel.zip) [![Download macOS Apple Silicon](https://img.shields.io/badge/Download-macOS%20M1%2FM2%2FM3%2FM4-black?style=for-the-badge&logo=apple)](https://github.com/hydropix/TranslateBooksWithLLMs/releases/latest/download/TranslateBook-macOS-AppleSilicon.zip)

</div>

A desktop app that translates **books**, **subtitles**, and **documents** with AI - local or cloud. Drop a file, pick a language, get the result.

**Formats:** EPUB, SRT, DOCX, TXT

- **No size limit.** Process documents of any length - from a single page to thousand-page novels. The intelligent chunking system handles unlimited content while preserving context between segments.
- **Perfect preservation.** Your documents come out exactly as they went in: EPUB formatting, styles, and structure remain intact. SRT timecodes stay perfectly synchronized. Every tag, every timestamp, every formatting detail is preserved.
- **Resume anytime.** Interrupted translation? Pick up exactly where you left off. The checkpoint system saves progress automatically.
- **Reusable writing styles.** Extract a style preset from sample books (or write one by hand) and apply it to every chunk for a consistent register, rhythm, and imagery across the whole translation.
- **Auto glossary & style.** No glossary or preset ready? Pick Auto in the dropdown and the app derives both straight from the document being translated - one extra LLM call each before the job starts, nothing saved.

<img width="1150" height="579" alt="image" src="https://github.com/user-attachments/assets/d7dfa806-830b-4405-8300-f10208943d22" />


> **[Translation Quality Benchmarks](https://github.com/hydropix/TranslateBooksWithLLMs/wiki)** - Find the best model for your target language.

---

## Quick Start

1. **Download** the release for your platform (buttons above), or grab it from the [releases page](https://github.com/hydropix/TranslateBooksWithLLMs/releases/latest).
2. **Extract** the archive.
3. **Run** `TranslateBook.exe` (Windows) or `./TranslateBook` (macOS).
4. **Open** http://localhost:5000 in your browser.

That's it. On first launch, you choose a translation provider:

- **Free & local** - Install [Ollama](https://ollama.com/) and pull a model (e.g. `ollama pull qwen3:14b`). Nothing leaves your machine.
- **Cloud (often free tier available)** - Paste an API key from one of the providers below.

> **Note:** First run creates a `TranslateBook_Data` folder with your settings.
>
> **macOS:** On first launch, go to **System Settings > Privacy & Security** and click **Open Anyway**.

---

## Providers

<p align="center">
<a href="https://ollama.com/"><img src="src/web/static/img/providers/ollama.png" alt="Ollama" height="32"></a>&nbsp;&nbsp;
<a href="https://poe.com/"><img src="src/web/static/img/providers/poe.png" alt="Poe" height="32"></a>&nbsp;&nbsp;
<a href="https://openrouter.ai/"><img src="src/web/static/img/providers/openrouter.png" alt="OpenRouter" height="32"></a>&nbsp;&nbsp;
<a href="https://openai.com/"><img src="src/web/static/img/providers/openai.png" alt="OpenAI" height="32"></a>&nbsp;&nbsp;
<a href="https://mistral.ai/"><img src="src/web/static/img/providers/mistral.png" alt="Mistral" height="32"></a>&nbsp;&nbsp;
<a href="https://www.deepseek.com/"><img src="src/web/static/img/providers/deepseek.png" alt="DeepSeek" height="32"></a>&nbsp;&nbsp;
<a href="https://deepmind.google/technologies/gemini/"><img src="src/web/static/img/providers/gemini.png" alt="Gemini" height="32"></a>&nbsp;&nbsp;
<a href="https://build.nvidia.com/"><img src="src/web/static/img/providers/nvidia.png" alt="NVIDIA NIM" height="32"></a>
</p>

| Provider | Type | Get started |
|----------|------|-------------|
| **DeepSeek** | Cloud | [platform.deepseek.com](https://platform.deepseek.com/api_keys) |
| **Gemini** | Cloud | [Google AI Studio](https://aistudio.google.com/apikey) |
| **Mistral** | Cloud | [console.mistral.ai](https://console.mistral.ai/api-keys) |
| **NVIDIA NIM** | Cloud | [build.nvidia.com](https://build.nvidia.com/) |
| **Ollama** | Local | [ollama.com](https://ollama.com/) |
| **OpenAI** | Cloud | [platform.openai.com](https://platform.openai.com/api-keys) |
| **OpenAI-Compatible** | Local (llama.cpp, LM Studio, vLLM, LocalAI...) | Point to your server's endpoint |
| **OpenRouter** | Cloud (200+ models) | [openrouter.ai/keys](https://openrouter.ai/keys) |
| **Poe** | Cloud (multi-model) | [poe.com/api_key](https://poe.com/api_key) |

See [docs/PROVIDERS.md](docs/PROVIDERS.md) for detailed setup instructions.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Ollama won't connect | Check Ollama is running, test `curl http://localhost:11434/api/tags` |
| Model not found | Run `ollama list`, then `ollama pull model-name` |

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for more solutions.

---

## Advanced

<details>
<summary><b>Install from source</b></summary>

**Prerequisites:** [Python 3.8+](https://www.python.org/downloads/), [Ollama](https://ollama.com/), [Git](https://git-scm.com/)

```bash
git clone https://github.com/hydropix/TranslateBooksWithLLMs.git
cd TranslateBookWithLLM
ollama pull qwen3:14b    # Download a model

# Windows
start.bat

# Mac/Linux
chmod +x start.sh && ./start.sh
```

The web interface opens at **http://localhost:5000**.

</details>

<details>
<summary><b>Command line</b></summary>

```bash
# Basic (auto-generates "book (Chinese).epub")
python translate.py -i book.epub -sl English -tl Chinese

# With OpenRouter
python translate.py -i book.txt --provider openrouter \
    --openrouter_api_key YOUR_KEY -m anthropic/claude-sonnet-4 -tl French

# With OpenAI
python translate.py -i book.txt --provider openai \
    --openai_api_key YOUR_KEY -m gpt-4o -tl French

# With Gemini
python translate.py -i book.txt --provider gemini \
    --gemini_api_key YOUR_KEY -m gemini-2.0-flash -tl French

# With Mistral
python translate.py -i book.txt --provider mistral \
    --mistral_api_key YOUR_KEY -m mistral-large-latest -tl French

# With DeepSeek
python translate.py -i book.txt --provider deepseek \
    --deepseek_api_key YOUR_KEY -m deepseek-v4-pro -tl French

# With Poe
python translate.py -i book.txt --provider poe \
    --poe_api_key YOUR_KEY -m Claude-Sonnet-4 -tl French

# With NVIDIA NIM
python translate.py -i book.txt --provider nim \
    --nim_api_key YOUR_KEY -m meta/llama-3.1-8b-instruct -tl French

# With local OpenAI-compatible server (llama.cpp, LM Studio, vLLM, etc.)
python translate.py -i book.txt --provider openai \
    --api_endpoint http://localhost:8080/v1/chat/completions -m your-model -tl French
```

### Main options

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input` | Input file | Required |
| `-o, --output` | Output file | Auto: `{name} ({lang}).{ext}` |
| `-sl, --source_lang` | Source language | English |
| `-tl, --target_lang` | Target language | Chinese |
| `-m, --model` | Model name | qwen3:14b |
| `--provider` | ollama/openrouter/openai/gemini/mistral/deepseek/poe/nim | ollama |
| `--parallel` | Chunks translated concurrently (cloud only; Ollama stays at 1) | 1 |
| `--text-cleanup` | OCR/typographic cleanup | disabled |
| `--refine` | Second pass for literary polish | disabled |
| `--tts` | Generate audio (Edge-TTS) | disabled |

See [docs/CLI.md](docs/CLI.md) for all options (TTS voices, rates, formats, etc.).

</details>

<details>
<summary><b>Configuration (.env)</b></summary>

Copy `.env.example` to `.env` and edit:

```bash
# Provider
LLM_PROVIDER=ollama

# Ollama
API_ENDPOINT=http://localhost:11434/api/generate
DEFAULT_MODEL=qwen3:14b

# API Keys (if using cloud providers)
OPENROUTER_API_KEY=sk-or-v1-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
MISTRAL_API_KEY=...
DEEPSEEK_API_KEY=...
POE_API_KEY=...
NIM_API_KEY=...

# Performance
REQUEST_TIMEOUT=900
MAX_TOKENS_PER_CHUNK=450     # Token-based chunking (default: 450 tokens)
PARALLEL_TRANSLATIONS=1      # Concurrent chunks (cloud only; Ollama stays at 1)
```

**Faster on cloud providers?** Set `PARALLEL_TRANSLATIONS` (or `--parallel N`, or the "Parallel requests" field in the web UI) above 1 to translate several chunks at once. Local providers (Ollama) ignore it since a single instance serializes requests. Higher values are faster but can hit provider rate limits.

**Multiple API keys?** Any `*_API_KEY` variable accepts a comma-separated list (e.g. `GEMINI_API_KEY=key1,key2,key3`). The system rotates between keys automatically when one hits a rate limit - useful to chain free-tier accounts. See [docs/API_KEY_ROTATION.md](docs/API_KEY_ROTATION.md).

**Long jobs?** Get a push notification on your phone (via ntfy), Discord, Slack, gotify, or any HTTP endpoint when a translation finishes. Set `NOTIFY_WEBHOOK_URL` in `.env`. See [docs/NOTIFICATIONS.md](docs/NOTIFICATIONS.md).

</details>

<details>
<summary><b>Docker</b></summary>

```bash
docker build -t translatebook .
docker run -p 5000:5000 -v $(pwd)/translated_files:/app/translated_files translatebook
```

See [docs/DOCKER.md](docs/DOCKER.md) for more options.

</details>

### Using the web UI from several devices

One server is one shared workspace - there are no user accounts, so every browser pointed at it sees the same state. A running job's progress panel, paused/interrupted jobs, the completed translation history, and the files in `translated_files/` are all shared: another device picks them up live (within a second or two) or on page load. A small set of preferences - source/target language, TTS, text cleanup, bilingual mode, plain-text mode, and the selected custom instruction - is synced too: the server wins at page load, the last change wins after that, and a tab that is already open picks up changes made elsewhere on its next reload.

Two things stay local to the device: if you queue several files, other devices only see the one currently translating, not the rest of the queue (it's held in the browser that built it); and the UI language is per-browser. Provider, model, API keys, and endpoints are never synced - they always come from the server's `.env`.

---

## Documentation

| Guide | Description |
|-------|-------------|
| [docs/PROVIDERS.md](docs/PROVIDERS.md) | Detailed provider setup (Ollama, LM Studio, OpenRouter, OpenAI, Gemini) |
| [docs/API_KEY_ROTATION.md](docs/API_KEY_ROTATION.md) | Use multiple API keys per provider with automatic failover on rate-limit |
| [docs/NOTIFICATIONS.md](docs/NOTIFICATIONS.md) | Webhook notifications on completion (ntfy, gotify, Discord, Slack, custom) |
| [docs/GLOSSARY.md](docs/GLOSSARY.md) | Force consistent term translations across a book (Web UI + CLI, auto-extract via NER) |
| [docs/STYLE_EXTRACTION.md](docs/STYLE_EXTRACTION.md) | Extract a reusable writing-style preset from sample books and apply it to every chunk (Web UI) |
| [docs/CLI.md](docs/CLI.md) | Complete CLI reference |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Problem solutions |
| [docs/DOCKER.md](docs/DOCKER.md) | Docker deployment guide |

---

## Support

If TBL saves you time, you can support its development on Ko-fi:

<a href="https://ko-fi.com/Y1M021FYBG" target="_blank"><img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support me on Ko-fi" height="36"></a>

---

**License:** AGPL-3.0
