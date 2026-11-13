"""Prompt regression harness.

Renders the baseline prompts (from a git ref) and the candidate prompts (the
working tree) through the real prompt builders, sends both to real models over
a fixed corpus, scores every output with deterministic metrics and prints a
before/after Markdown report. See README.md in this folder.

Not collected by pytest (no module starts with ``test_``). Unit tests for the
deterministic parts live in ``tests/unit/prompt_eval/``.
"""
