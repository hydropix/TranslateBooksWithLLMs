# API Key Rotation

Translate longer documents on free-tier APIs by chaining multiple keys. When one key hits a rate limit, the system automatically switches to another — no manual intervention, no lost progress.

> Available for all cloud providers: Gemini, OpenRouter, OpenAI, Mistral, DeepSeek, Poe, NVIDIA NIM, Opencode.

---

## Why use it?

Most cloud LLM APIs enforce two kinds of limits:

- **Per-minute limits (RPM)** — e.g. Gemini free tier allows 15 requests/minute
- **Per-day limits (RPD)** — e.g. Gemini free tier allows 1500 requests/day

Translating a full book typically requires hundreds (sometimes thousands) of API calls. With a single free-tier key, you can hit the daily quota mid-translation and have to wait until the next day. By providing several keys (from different free accounts), you multiply your effective quota and the system rotates through them transparently.

**Common scenarios:**

| You want to | Solution |
| --- | --- |
| Translate a 600-page novel on Gemini free tier | Provide 2–3 Gemini keys |
| Test a translation without paying | Chain free OpenRouter models from different accounts |
| Keep paid translation going past a single key's RPM ceiling | Provide 2–3 paid keys to share load |
| Protect against a key being temporarily revoked | Add a backup key — failover is automatic |

---

## Quick start

You can supply multiple keys via **any** of the three configuration channels TBL supports. All accept the same comma-separated format.

### 1. Via `.env` file (recommended for repeated runs)

```bash
GEMINI_API_KEY=AIza...key1,AIza...key2,AIza...key3
```

Or with newlines for readability:

```bash
GEMINI_API_KEY=AIza...key1
                AIza...key2
                AIza...key3
```

> Same syntax works for `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `MISTRAL_API_KEY`, `DEEPSEEK_API_KEY`, `POE_API_KEY`, `NIM_API_KEY`, `OPENCODE_API_KEY`.

### 2. Via the Web UI

In the provider settings, enter your keys separated by commas in the API key field:

```text
key1, key2, key3
```

> Hover the input field to see the inline tip.

### 3. Via the CLI

```bash
python translate.py -i book.epub --provider gemini \
    --gemini_api_key "key1,key2,key3" \
    -m gemini-2.0-flash -tl French
```

---

## How it works

### Normal operation

Each translation request grabs the next key from a round-robin pool. With three keys `[k1, k2, k3]`, requests alternate `k1 → k2 → k3 → k1 → ...`. This spreads the load across all keys evenly so no single key hits its RPM limit too quickly.

### When a key gets rate-limited (HTTP 429)

```text
Request 42 with key #1 → HTTP 429 (rate-limited)
  ↓
Mark key #1 as throttled until expiry (Retry-After header value)
  ↓
[Pool: k1=throttled, k2=available, k3=available]
  ↓
Next request immediately uses key #2 — no waiting
```

The failed request is retried with a fresh key in the same iteration of the retry loop. **No sleep is added** as long as another key is available.

### When all keys are throttled

If every key in the pool is currently throttled, the system falls back to the original behavior: it sleeps until the earliest key becomes available, then resumes. If the maximum retry attempts are exhausted (configured via `MAX_TRANSLATION_ATTEMPTS`, default 3), it raises a `RateLimitError` and the translation auto-pauses (a checkpoint is saved — you can resume later with the same command).

### Throttle expiry

The system parses the API's response headers to determine how long to wait:

1. **`Retry-After` header** (in seconds) — used by most providers
2. **`X-RateLimit-Reset` header** (UTC milliseconds) — used by OpenRouter
3. **Exponential backoff** (4s, 8s, 16s, 32s, capped at 60s) — fallback when no header

Once the throttle window expires, the key returns to the available pool automatically.

---

## What you'll see in the logs

When rotation happens, the log shows:

```text
[gemini] key #1/3 rate-limited (retry-after 47s), rotating to next key
```

When all keys are throttled and the system needs to wait:

```text
[gemini] all 3 key(s) rate-limited (attempt 2/3), waiting 47s...
```

When the pool is fully exhausted (translation pauses for resume):

```text
RateLimitError: gemini rate limit exceeded after 3 attempts (all 3 key(s) exhausted)
```

---

## Recommended setups

### Free Gemini for full novels

Gemini's free tier (`gemini-2.0-flash`) is generous on quality but capped at 1500 requests/day per key. For a 400-page novel:

```bash
# 2 keys = ~3000 daily requests, enough for most novels
GEMINI_API_KEY=key_account_a,key_account_b
```

### Free OpenRouter models

OpenRouter free models share a `:free` suffix and have a combined 20 requests/minute limit per key. Multiple keys help when the ceiling is hit:

```bash
OPENROUTER_API_KEY=sk-or-v1-key1,sk-or-v1-key2
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free
```

### Paid throughput

For paid tiers, rotation helps when you want to push more requests-per-minute than a single key allows:

```bash
DEEPSEEK_API_KEY=key1,key2,key3   # Triples your effective RPM ceiling
```

---

## Limits & edge cases

### What rotation does NOT do

- **It does not parallelize requests.** Translation is still sequential — the keys are used one at a time, in turn. Parallel dispatch is a separate feature.
- **It does not pre-validate keys.** A revoked or invalid key (HTTP 401) stays in the pool and will keep generating 401 errors until you remove it. Only HTTP 429 is treated as transient.
- **It does not persist throttle state across restarts.** If you restart TBL mid-translation, all keys are considered fresh. Most rate-limit windows are short (1 minute for RPM), so this is rarely a problem in practice. Daily quotas (RPD) reset at midnight UTC anyway.

### When rotation is skipped

- **Single-key pools** behave exactly like the previous single-key code: sleep on 429, retry, raise on exhaustion. No rotation overhead.
- **Ollama** (local) has no rate limits, so it has no key pool. Rotation is irrelevant.
- **OpenAI-compatible local servers** (llama.cpp, vLLM, LM Studio) typically run without an API key; rotation is skipped automatically.

### Combining with checkpointing

If your translation runs out of keys and pauses, the checkpoint system saves your state. You can:

1. Wait for quotas to reset (e.g. next day at midnight UTC for RPD)
2. Add more keys to your `.env`
3. Resume with the same command — translation picks up exactly where it stopped

---

## FAQ

**Q: Will adding keys speed up my translation?**
A: Not directly. Translation is sequential, so requests run one at a time regardless of how many keys are in the pool. Rotation only helps you avoid pauses when limits are hit.

**Q: My provider has a single account that I share across machines. Should I split it?**
A: No. Multiple keys from the *same* account share the same quota — adding them won't help. Use keys from *different* accounts to multiply effective quotas.

**Q: Does the order of keys matter?**
A: Slightly. Rotation is round-robin starting from the first key. If your first key is faster or has higher quota, list it first to use it preferentially when others recover from throttle.

**Q: What if I have 1 paid key and 2 free keys?**
A: They'll all be used in rotation. If you'd rather use the paid one only as backup, you'd need a manual approach (currently not supported via UI; you could swap which key is "first" in the list).

**Q: Are keys logged anywhere?**
A: Only by index (e.g. "key #2/3"), never the actual value. The `.env` file stays local — rotate keys via standard provider dashboards if you suspect leakage.

**Q: Can I mix providers (e.g. some Gemini + some OpenAI)?**
A: No — each `*_API_KEY` variable corresponds to one provider. The pool only rotates within a provider. To switch providers, change `LLM_PROVIDER`.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Translation pauses despite providing 3 keys | All keys hit a per-day limit (RPD) | Wait for midnight UTC reset, or add keys from new accounts |
| "key #1/3 rate-limited" message even on first request | One key is invalid or already throttled from a previous run | Check the key's status in the provider dashboard; remove if revoked |
| Rotation doesn't seem to happen on 429 | Pool only has one key | Add at least one more key to enable rotation |
| Web UI accepts only one key | The UI text input *does* accept commas — separate keys with `,` | Hover the field for the inline tip |
