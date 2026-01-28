# <img src="https://github.com/hydropix/TranslateBooksWithLLMs/blob/main/src/web/static/TBL-Logo-alpha.png?raw=true" alt="Logo" height="40"> Translate Books with LLMs

Translate **books**, **subtitles**, and **documents** using AI - locally or in the cloud.

**No size limit.** Process documents of any length - from a single page to thousand-page novels. The intelligent chunking system handles unlimited content while preserving context between segments.

**Perfect preservation.** Your documents come out exactly as they went in: EPUB formatting, styles, and structure remain intact. SRT timecodes stay perfectly synchronized. Every tag, every timestamp, every formatting detail is preserved.

**Resume anytime.** Interrupted translation? Pick up exactly where you left off. The checkpoint system saves progress automatically.

Formats: **EPUB**, **SRT**, **DOCX**, **TXT**

Providers:

<p align="left">
<img src="src/web/static/img/providers/ollama.png" alt="Ollama" height="32">&nbsp;&nbsp;
<img src="src/web/static/img/providers/poe.png" alt="Poe" height="32">&nbsp;&nbsp;
<img src="src/web/static/img/providers/openrouter.png" alt="OpenRouter" height="32">&nbsp;&nbsp;
<img src="src/web/static/img/providers/openai.png" alt="OpenAI" height="32">&nbsp;&nbsp;
<img src="src/web/static/img/providers/mistral.png" alt="Mistral" height="32">&nbsp;&nbsp;
<img src="src/web/static/img/providers/deepseek.png" alt="DeepSeek" height="32">&nbsp;&nbsp;
<img src="src/web/static/img/providers/gemini.png" alt="Gemini" height="32">
</p>

- [**Ollama**](https://ollama.com/download) (local)
- [**Poe**](https://poe.com/api_key) ⭐ Recommended - Easy setup, multiple AI models
- [**OpenRouter**](https://openrouter.ai/keys) (200+ models)
- [**OpenAI**](https://platform.openai.com/api-keys) (compatible like LM Studio)
- [**Mistral**](https://console.mistral.ai/api-keys)
- [**DeepSeek**](https://platform.deepseek.com/api_keys)
- [**Gemini**](https://aistudio.google.com/apikey)

> **[Translation Quality Benchmarks](https://github.com/hydropix/TranslateBooksWithLLMs/wiki)** — Find the best model for your target language.

---

## Quick Start

### Download Executable (No Python Required!)

[![Download Windows](https://img.shields.io/badge/Download-Windows-blue?style=for-the-badge&logo=windows)](https://github.com/hydropix/TranslateBooksWithLLMs/releases/latest/download/TranslateBook-Windows.zip) [![Download macOS Intel](https://img.shields.io/badge/Download-macOS%20Intel-black?style=for-the-badge&logo=apple)](https://github.com/hydropix/TranslateBooksWithLLMs/releases/latest/download/TranslateBook-macOS-Intel.zip) [![Download macOS Apple Silicon](https://img.shields.io/badge/Download-macOS%20M1%2FM2%2FM3%2FM4-black?style=for-the-badge&logo=apple)](https://github.com/hydropix/TranslateBooksWithLLMs/releases/latest/download/TranslateBook-macOS-AppleSilicon.zip)

1. Download and extract the archive for your platform
2. Install [Ollama](https://ollama.com/) (for local AI models)
3. Run `TranslateBook.exe` (Windows) or `./TranslateBook` (macOS)
4. Open http://localhost:5000 in your browser

> **Note:** First run creates a `TranslateBook_Data` folder with configuration files.
>
> **macOS:** On first launch, go to System Settings > Privacy & Security and click "Open Anyway".

---

### For the Bearded Ones - Install from Source

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

The web interface opens at **http://localhost:5000**

---

## LLM Providers

| Provider | Type | Setup |
|----------|------|-------|
| **Ollama** | Local | [ollama.com](https://ollama.com/) |
| **Poe** ⭐ | Cloud (Recommended) | [poe.com/api_key](https://poe.com/api_key) |
| **OpenAI-Compatible** | Local | llama.cpp, LM Studio, vLLM, LocalAI... |
| **OpenRouter** | Cloud (200+ models) | [openrouter.ai/keys](https://openrouter.ai/keys) |
| **OpenAI** | Cloud | [platform.openai.com](https://platform.openai.com/api-keys) |
| **Mistral** | Cloud | [console.mistral.ai](https://console.mistral.ai/api-keys) |
| **DeepSeek** | Cloud | [platform.deepseek.com](https://platform.deepseek.com/api_keys) |
| **Gemini** | Cloud | [Google AI Studio](https://makersuite.google.com/app/apikey) |

> **OpenAI-Compatible servers:** Use `--provider openai` with your server's endpoint (e.g., llama.cpp: `http://localhost:8080/v1/chat/completions`, LM Studio: `http://localhost:1234/v1/chat/completions`)

See [docs/PROVIDERS.md](docs/PROVIDERS.md) for detailed setup instructions.

---

## Command Line

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

# With local OpenAI-compatible server (llama.cpp, LM Studio, vLLM, etc.)
python translate.py -i book.txt --provider openai \
    --api_endpoint http://localhost:8080/v1/chat/completions -m your-model -tl French
```

### Main Options

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input` | Input file | Required |
| `-o, --output` | Output file | Auto: `{name} ({lang}).{ext}` |
| `-sl, --source_lang` | Source language | English |
| `-tl, --target_lang` | Target language | Chinese |
| `-m, --model` | Model name | mistral-small:24b |
| `--provider` | ollama/openrouter/openai/gemini | ollama |
| `--text-cleanup` | OCR/typographic cleanup | disabled |
| `--refine` | Second pass for literary polish | disabled |
| `--tts` | Generate audio (Edge-TTS) | disabled |

See [docs/CLI.md](docs/CLI.md) for all options (TTS voices, rates, formats, etc.).

---

## Configuration (.env)

Copy `.env.example` to `.env` and edit:

```bash
# Provider
LLM_PROVIDER=ollama

# Ollama
API_ENDPOINT=http://localhost:11434/api/generate
DEFAULT_MODEL=mistral-small:24b

# API Keys (if using cloud providers)
OPENROUTER_API_KEY=sk-or-v1-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...

# Performance
REQUEST_TIMEOUT=900
MAX_TOKENS_PER_CHUNK=400  # Token-based chunking (default: 400 tokens)
```

---

## Docker

```bash
docker build -t translatebook .
docker run -p 5000:5000 -v $(pwd)/translated_files:/app/translated_files translatebook
```

See [DOCKER.md](DOCKER.md) for more options.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Ollama won't connect | Check Ollama is running, test `curl http://localhost:11434/api/tags` |
| Model not found | Run `ollama list`, then `ollama pull model-name` |

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for more solutions.

---

## Documentation

| Guide | Description |
|-------|-------------|
| [docs/PROVIDERS.md](docs/PROVIDERS.md) | Detailed provider setup (Ollama, LM Studio, OpenRouter, OpenAI, Gemini) |
| [docs/CLI.md](docs/CLI.md) | Complete CLI reference |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Problem solutions |
| [DOCKER.md](DOCKER.md) | Docker deployment guide |

---

## Links

- [Report Issues](https://github.com/hydropix/TranslateBooksWithLLMs/issues)
- [OpenRouter Models](https://openrouter.ai/models)

---

**License:** AGPL-3.0
