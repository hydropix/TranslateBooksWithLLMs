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

### Thinking / Reasoning Control

Some Gemini models ("thinking" models — Gemini 2.5.x and 3.x) reason before
answering, and those reasoning tokens are billed at the (higher) *output*
token rate. For a mechanical task like translation, reasoning rarely
improves quality but can noticeably inflate the bill. Set one of these in
your `.env` file:

```bash
# Gemini 3.x models (gemini-3.6-flash, gemini-3.5-flash, gemini-3.5-flash-lite, ...)
# Valid values: minimal, low, medium, high
GEMINI_THINKING_LEVEL=minimal

# Gemini 2.5.x models (gemini-2.5-flash, ...) — explicit token budget
# 0 disables thinking, -1 enables dynamic thinking
GEMINI_THINKING_BUDGET=0
```

Leave both unset to keep the previous default (thinking off on 2.5 models,
model default on 3.x models). Not every level is supported by every model —
Pro-tier models, for example, reject `minimal`.

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

### Reasoning and web search

Poe bots ship with two defaults that cost tokens translation never uses, so TBL overrides both.

**Reasoning.** Most bots reason by default (`gemini-3.6-flash`: `thinking_level=medium`, `grok-4.5`
and `kimi-k3`: `reasoning_effort=high`, `glm-5.x`: `enable_thinking=true`). On a 102-token prompt,
`gemini-3.6-flash` spent 1167 reasoning tokens out of 1231 output tokens. TBL asks each bot for its
lowest reasoning setting; `POE_DISABLE_THINKING=false` keeps reasoning on.

**Web search.** `gemini-3.6-flash` and `grok-4.5` search the web by default, and every chunk pays for
it in prompt tokens: `grok-4.5` sent 1141 prompt tokens for an 84-token translation prompt, down to
283 with search off. A book is self-contained, so retrieval is cost at best and context pollution at
worst. `POE_DISABLE_WEB_SEARCH=false` allows it.

Poe has no universal switch for either: every bot advertises its own knobs in the `parameters` array
of `/v1/models` and rejects any knob it does not advertise with HTTP 400, so the settings are read
from that catalog rather than a hardcoded model list. A rejected knob is dropped and the request
retried, so a stale catalog entry can never fail a chunk.

One exception: `output_effort` (Claude 4.6/4.8) is left untouched, because it caps the whole answer
rather than just hidden reasoning. Those two bots keep reasoning.

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

## Endpoint Allowlist

The web API lets a request choose the endpoint the server calls, so the server checks that endpoint against an allowlist before using it. Accepted out of the box: the known provider hosts (`api.openai.com`, `generativelanguage.googleapis.com`, `openrouter.ai`, `api.mistral.ai`, `api.deepseek.com`, `api.poe.com`, `integrate.api.nvidia.com`), every `*_API_ENDPOINT` configured in your `.env`, and any loopback, LAN or `host.docker.internal` address so self-hosted Ollama, LM Studio, llama.cpp and vLLM keep working. Anything else returns HTTP 400.

To allow a self-hosted gateway on a public hostname, add it to `LLM_ENDPOINT_ALLOWLIST` in `.env` (comma-separated; subdomains of a listed host are covered):

```bash
LLM_ENDPOINT_ALLOWLIST=llm.internal.example.com,gateway.example.org
```

This variable is read at startup only and is deliberately not editable from the web UI. A second rule pairs with it: when a request supplies an endpoint that differs from the configured default, the server refuses to attach the API key stored in `.env` — that request must send its own key, or it is rejected. Together these guarantee a stored credential is never sent to a host the request chose.

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
