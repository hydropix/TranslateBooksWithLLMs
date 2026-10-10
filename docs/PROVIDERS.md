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

### Cloud models

`-cloud` models (e.g. `gemma3:27b-cloud`) are not run locally — the local Ollama daemon relays them to Ollama's hosted infrastructure using the account signed in with `ollama signin`. Once that account's usage limit is reached, the daemon returns HTTP 429. TBL pauses the job and saves a checkpoint, so you can resume once the quota resets or you add credits. The **"Don't auto-pause on rate limit (HTTP 429)"** setting is the opt-out: enable it and TBL waits and auto-resumes instead of pausing.

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
3. Choose a model from the list (your [presets](#presets) appear first)

### CLI Example

```bash
python translate.py -i book.txt -o book_fr.txt \
    --provider openrouter \
    --openrouter_api_key sk-or-v1-your-key \
    -m anthropic/claude-sonnet-4
```

Browse models and pricing: [openrouter.ai/models](https://openrouter.ai/models)

### Presets

An OpenRouter [preset](https://openrouter.ai/docs/guides/features/presets) is a named configuration
saved on [openrouter.ai/settings/presets](https://openrouter.ai/settings/presets): a system prompt,
a model with fallback models, provider routing (preferred or excluded providers, data policy,
quantization), sampling parameters and reasoning settings. You use one by sending
`@preset/<slug>` as the model.

Presets are worth using with TBL because they expose OpenRouter options that TBL has no UI or CLI
flag for. Provider routing and model fallbacks are configured only there. A preset is also edited
on OpenRouter's site, so you can tune it without touching TBL, and the same configuration works in
every tool that uses your OpenRouter account.

When TBL loads the OpenRouter model list, it also calls `GET /api/v1/presets` with the same API key
and lists your active presets first, as `@preset/<slug>` (the preset description is the tooltip).
Type `preset` or part of the slug in the model search to find them. On the CLI, pass the preset as
the model:

```bash
python translate.py -i book.txt -o book_fr.txt \
    --provider openrouter \
    -m @preset/my-translation-preset
```

Notes:
- TBL does not send its reasoning override for a preset, so the preset's own reasoning setting
  applies (see [Reasoning](#reasoning)).
- Presets have no fixed price, so the cost estimate shows pricing as unknown. You can enter a
  price by hand in the cost panel.
- If the preset list cannot be fetched, the regular model list still loads and a warning is logged.

### Reasoning

Many OpenRouter models reason by default (DeepSeek V4.x at effort `high`, Qwen 3.x), and reasoning
tokens are billed as output: DeepSeek V4.1 Flash was seen producing 4,700 to 31,000 output tokens
for translation chunks of about 3,000 input tokens. TBL reads each model's reasoning metadata from
`/api/v1/models` and sends OpenRouter's unified `reasoning` parameter (except for presets): `{"enabled": false}` when
reasoning can be turned off, or the lowest supported effort on models where it is mandatory (a
warning is logged for those). The per-request log line shows the reasoning token count whenever
the model reports one. `OPENROUTER_DISABLE_THINKING=false` keeps each model's default.

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

## Opencode (Cloud)

One [OpenCode](https://opencode.ai) API key reaches two catalogs:

| Catalog | Model id | Billed against |
|---|---|---|
| [OpenCode Go](https://opencode.ai/docs/go/) | `opencode-go/<model>` | Your Go subscription's usage limits |
| OpenCode Zen | `opencode/<model>` | Your workspace balance (pay-as-you-go) |

The ids are the ones `opencode models` prints, and the prefix decides where the
request goes. A bare `<model>` goes to Go, so a missing prefix never spends
balance. The model list shows both catalogs, grouped.

Whether a model actually works depends on your workspace, not on the key: Zen
models need a balance (or "use balance" enabled), and individual models can be
disabled in the OpenCode console. The model list cannot know this in advance,
so such refusals show up as clear errors when the translation starts.

### Session header

Every request carries an `x-opencode-session` header holding a stable id for the
conversation; without it Go answers `400 MissingSessionID`. TBL generates one id
per translation job (the provider instance lives for the whole job, so every
chunk and parallel worker shares it and the gateway's prompt cache stays warm)
and sends it on every call, including the model-list probe. It is random per
job, never derived from the machine or the install.

### Supported models

The gateway serves each model through one API family, and TBL picks it from
the model id, the same way opencode and hermes-agent do:

| Models | API | Endpoint |
|---|---|---|
| `gpt-*`, `grok-*`, `muse-spark*` | OpenAI Responses | `.../v1/responses` |
| `claude-*`, `qwen*`, `union-alpha*`, and on Go `minimax-*` | Anthropic Messages | `.../v1/messages` |
| `gemini-*` (Zen only) | Google generateContent | `.../v1/models/<model>:generateContent` |
| everything else (DeepSeek, GLM, Kimi, MiMo, LongCat, ...) | chat/completions | `.../v1/chat/completions` |

- New models that rotate into the catalogs go to chat/completions unless their
  id matches a prefix above. If the gateway refuses one
  (`ModelProtocolUnsupported`), the log says so.
- **Not supported yet:** `jev-*` on Zen (OpenCode's System One API). These
  models are hidden from the model list and rejected with a clear error if
  configured by hand.
- **Free Zen models** (`opencode/*-free`, `opencode/big-pickle`, ...) are listed,
  but OpenCode currently restricts them to its own app and refuses other
  clients with `FreeTierError`. TBL does not impersonate the OpenCode client.

When a Go usage limit is reached, the gateway answers 429 with a reset time that
can be hours or days away; the job auto-pauses with a checkpoint instead of
waiting, and can be resumed later.

### Setup

1. Get an API key from the [OpenCode console](https://opencode.ai).
2. In TBL: select "Opencode", enter your key.
3. Pick a model in the UI, or set `OPENCODE_MODEL` in `.env`.

### Configuration

```bash
OPENCODE_API_KEY=
OPENCODE_MODEL=opencode-go/deepseek-v4.1-flash
# OPENCODE_API_BASE=https://opencode.ai/zen
```

`OPENCODE_API_BASE` overrides the gateway root; the catalog path (`/go/v1` or
`/v1`) is appended to it. It is read at server startup, so changing it requires
a restart. Like most other cloud providers, the Opencode provider ignores the
per-request `llm_api_endpoint` field.

### CLI Example

```bash
python translate.py -i book.txt -o book_fr.txt \
    --provider opencode \
    --opencode_api_key YOUR_API_KEY_HERE \
    -m opencode-go/deepseek-v4.1-flash
```

---

## Endpoint Allowlist

The web API lets a request choose the endpoint the server calls, so the server checks that endpoint against an allowlist before using it. Accepted out of the box:

- the known provider hosts (`api.openai.com`, `generativelanguage.googleapis.com`, `openrouter.ai`, `api.mistral.ai`, `api.deepseek.com`, `api.poe.com`, `integrate.api.nvidia.com`);
- every `*_API_ENDPOINT` configured in your `.env`;
- anything on your own network, so self-hosted Ollama, LM Studio, llama.cpp and vLLM keep working: loopback and LAN addresses (including `100.64.0.0/10`, the range Tailscale uses), `localhost`, `host.docker.internal`, a single-label hostname such as `http://ollama:11434` (a Docker service or LXC name), and any host under `.local`, `.lan`, `.home`, `.home.arpa`, `.internal`, `.intranet`, `.corp`, `.private` or `.ts.net`;
- any other hostname that **resolves entirely to your local network**, so a LAN machine named under a domain you own (`ai-server.example.com` answering `192.168.1.50` from your internal DNS) works without any configuration. The lookup only happens for a host none of the rules above accepted, and the verdict is cached for a minute.

Anything else returns HTTP 400 with the rejected host and the fix in the response, and a `WARNING` line in the server log. A host that resolves to a public address, or does not resolve at all, is rejected.

The endpoint is only checked for the providers that actually read it (`ollama`, `openai`, `nim`). The web UI sends the field with every provider, so a stale value there never blocks a Gemini or OpenRouter job.

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
