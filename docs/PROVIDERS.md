# LLM Providers Guide

TBL supports multiple LLM providers. This guide explains how to set up each one.

---

## Ollama (Local)

Runs models locally on your machine.

### Setup

1. Install from [ollama.com](https://ollama.com/)
2. Download a model: `ollama pull qwen3:14b`
3. Select "Ollama" in TBL

### Models by VRAM

| VRAM | Model | Size |
|------|-------|------|
| 6-10 GB | `qwen3:8b` | 5.2 GB |
| 10-16 GB | `qwen3:14b` | 9.3 GB |
| 16-24 GB | `qwen3:30b-instruct` | 19 GB |
| 48+ GB | `qwen3:235b` | 142 GB |

Browse models: [ollama.com/search](https://ollama.com/search)

### CLI Example

```bash
python translate.py -i book.txt -o book_fr.txt -m qwen3:14b
```

---

## OpenAI-Compatible Servers (Local)

TBL supports any server that implements the OpenAI API format. This includes:

- **llama.cpp** (`llama-server`) - Lightweight, direct model serving
- **LM Studio** - Desktop app with GUI
- **vLLM** - High-performance serving
- **LocalAI** - Drop-in OpenAI replacement
- **Text Generation Inference** - HuggingFace's serving solution

### Setup

1. Start your OpenAI-compatible server
2. In TBL:
   - Select "OpenAI-Compatible" provider
   - Set endpoint to your server URL (see table below)
   - Leave API key empty (local servers don't require it)

| Server | Default Endpoint |
|--------|------------------|
| llama.cpp (`llama-server`) | `http://localhost:8080/v1/chat/completions` |
| LM Studio | `http://localhost:1234/v1/chat/completions` |
| vLLM | `http://localhost:8000/v1/chat/completions` |
| LocalAI | `http://localhost:8080/v1/chat/completions` |

### CLI Examples

```bash
# llama.cpp (llama-server)
python translate.py -i book.txt -o book_fr.txt \
    --provider openai \
    --api_endpoint http://localhost:8080/v1/chat/completions \
    -m your-model-name

# LM Studio
python translate.py -i book.txt -o book_fr.txt \
    --provider openai \
    --api_endpoint http://localhost:1234/v1/chat/completions \
    -m your-model-name
```

---

## OpenRouter (Cloud)

Access to 200+ models from multiple providers through a single API.

### Setup

1. Get API key at [openrouter.ai/keys](https://openrouter.ai/keys)
2. In TBL: Select "OpenRouter", enter your key
3. Choose a model from the list

### CLI Example

```bash
python translate.py -i book.txt -o book_fr.txt \
    --provider openrouter \
    --openrouter_api_key sk-or-v1-your-key \
    -m anthropic/claude-sonnet-4
```

Browse models and pricing: [openrouter.ai/models](https://openrouter.ai/models)

---

## OpenAI Cloud

Official OpenAI API (GPT models). Uses the same "OpenAI-Compatible" provider in TBL.

### Models

- `gpt-4o` - Latest GPT-4
- `gpt-4o-mini` - Smaller, cheaper
- `gpt-4-turbo`
- `gpt-3.5-turbo`

### Setup

1. Get API key at [platform.openai.com](https://platform.openai.com/api-keys)
2. In TBL:
   - Select "OpenAI-Compatible" provider
   - Keep endpoint as `https://api.openai.com/v1/chat/completions`
   - Enter your API key

### CLI Example

```bash
python translate.py -i book.txt -o book_fr.txt \
    --provider openai \
    --openai_api_key sk-your-key \
    -m gpt-4o
```

Pricing: [openai.com/pricing](https://openai.com/pricing)

---

## Google Gemini (Cloud)

Google's Gemini models.

### Models

- `gemini-2.0-flash`
- `gemini-1.5-pro`
- `gemini-1.5-flash`

### Setup

1. Get API key at [Google AI Studio](https://makersuite.google.com/app/apikey)
2. In TBL: Select "Gemini", enter your key

### CLI Example

```bash
python translate.py -i book.txt -o book_fr.txt \
    --provider gemini \
    --gemini_api_key your-key \
    -m gemini-2.0-flash
```

---

## Mistral (Cloud)

European cloud provider with strong multilingual quality.

### Models

- `mistral-large-latest` — flagship
- `mistral-small-latest` — cheaper, fast
- `open-mistral-nemo`
- `codestral-latest`

### Setup

1. Get API key at [console.mistral.ai/api-keys](https://console.mistral.ai/api-keys)
2. In TBL: Select "Mistral", enter your key

### CLI Example

```bash
python translate.py -i book.txt -o book_fr.txt \
    --provider mistral \
    --mistral_api_key your-key \
    -m mistral-large-latest
```

Pricing: [mistral.ai/technology](https://mistral.ai/technology)

---

## DeepSeek (Cloud)

Chinese LLM provider with 64K context and OpenAI-compatible API. Supports thinking models.

### Models

- `deepseek-v4-pro` — high-quality model
- `deepseek-v4-flash` — faster economical model
- `deepseek-chat` — legacy alias scheduled for deprecation on 2026-07-24
- `deepseek-reasoner` — reasoning model with `<think>` blocks

### Setup

1. Get API key at [platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys)
2. In TBL: Select "DeepSeek", enter your key

### CLI Example

```bash
python translate.py -i book.txt -o book_fr.txt \
    --provider deepseek \
    --deepseek_api_key your-key \
    -m deepseek-v4-pro
```

Pricing: [api-docs.deepseek.com/quick_start/pricing](https://api-docs.deepseek.com/quick_start/pricing)

---

## Poe (Cloud)

Single key, many models — Claude, GPT, Gemini, Llama, Mistral, DeepSeek and more from one Poe account.

### Setup

1. Get API key at [poe.com/api_key](https://poe.com/api_key)
2. In TBL: Select "Poe", enter your key
3. Pick a model name from [poe.com](https://poe.com/) (case-sensitive, e.g. `Claude-Sonnet-4`)

### CLI Example

```bash
python translate.py -i book.txt -o book_fr.txt \
    --provider poe \
    --poe_api_key your-key \
    -m Claude-Sonnet-4
```

> Poe usage is metered in points — each model has its own cost. Check the model card on poe.com for the rate.

---

## NVIDIA NIM (Cloud)

Hosted models via NVIDIA's inference platform — OpenAI-compatible API, generous free tier.

### Setup

1. Get API key at [build.nvidia.com](https://build.nvidia.com/)
2. In TBL: Select "NVIDIA NIM", enter your key

### CLI Example

```bash
python translate.py -i book.txt -o book_fr.txt \
    --provider nim \
    --nim_api_key your-key \
    -m meta/llama-3.1-8b-instruct
```

Browse models: [build.nvidia.com](https://build.nvidia.com/)

---

## API Key Rotation

Every cloud provider above accepts a comma-separated list of keys (e.g. `key1,key2,key3`). The system automatically rotates keys on HTTP 429 — useful for chaining free-tier accounts. See [API_KEY_ROTATION.md](API_KEY_ROTATION.md) for details.

---

## Environment Variables

Store settings in `.env` file:

```bash
# Provider
LLM_PROVIDER=ollama

# API Keys (each accepts comma-separated values for automatic rotation)
OPENROUTER_API_KEY=sk-or-v1-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
MISTRAL_API_KEY=...
DEEPSEEK_API_KEY=...
POE_API_KEY=...
NIM_API_KEY=...

# Ollama settings
API_ENDPOINT=http://localhost:11434/api/generate
DEFAULT_MODEL=qwen3:14b
```
