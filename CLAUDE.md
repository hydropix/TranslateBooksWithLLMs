# CLAUDE.md

Instructions for Claude Code when working in this repo.

## Language policy

Everything that lives on GitHub must be written in English: code, identifiers, code comments, docstrings, commit messages, PR titles and descriptions, issue text, documentation files (README, CLAUDE.md, etc.), and user-facing strings in committed files.

Replies to the user in chat may follow the language the user writes in (typically French), but any artifact that ends up committed to the repo must be English-only. If the user asks for a comment, doc, or commit message in another language, push back and write it in English.

## Security rules — API keys and secrets

**Absolutely forbidden** to hardcode an API key, token, password, URL containing a secret, or any other credential in a source file, a test, an ad-hoc script, a comment, a doc example, or a commit message. No exceptions, even temporary, even "just for debugging".

**Always** load credentials from `.env` (already gitignored) through the project's existing mechanism. If a variable is missing from `.env`, do not fabricate it or invent a value — ask the user, or fail cleanly with a clear message.

**For examples** in docs or tests that need to illustrate a key format, use an explicit, inert placeholder: `YOUR_API_KEY_HERE`, `sk-xxxxxxxx`, `<REDACTED>`. Never a real key, even expired or disabled — git history keeps it anyway.

**Before each commit**, if the diff includes a literal that looks like a key (long base64/hex string, prefix `sk-`/`Bearer`/`xoxp-`/`ghp_`/`AKIA`/etc., or a variable named `api_key`/`token`/`secret` with a non-empty value), stop and alert the user before proposing the commit.

**If a key is found in existing code** (old hardcoding, forgotten test file), flag it immediately to the user, recommend revoking it on the provider side, and propose moving it to `.env` plus removing the literal.

## Script organization rules

No ad-hoc test or debug scripts at the repo root. The patterns `check_*.py`, `test_*.py`, `debug_*.py`, `scratch_*.py`, `tmp_*.py` at the root are gitignored — this is intentional.

The real entrypoints at the root are `translate.py`, `translation_api.py`, `launcher.py`. Real tests live under `tests/` (pytest) or `tests/standalone/` (manual scripts). If an ad-hoc script becomes useful long-term, move it into `tests/standalone/` or `scripts/` before committing it.
