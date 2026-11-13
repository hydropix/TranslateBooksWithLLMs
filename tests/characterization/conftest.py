"""Hermetic environment for the characterization suite.

Two kinds of leakage made these goldens depend on the machine they ran on:

* Filesystem. ``CheckpointManager`` creates the relative ``data/uploads``
  directory in its constructor and ``start_job`` copies the input there, both
  resolved against the current working directory. The tests built the manager
  (and started jobs) before chdir-ing into ``tmp_path``, so every run wrote
  into the repository's real ``data/uploads``. The autouse fixture below
  chdirs into ``tmp_path`` for the whole test instead.

* Configuration. ``src.config`` loads ``.env`` from the working directory at
  import time, and several knobs that shape the progress trace are read from
  it rather than from the arguments the recorder passes:

  - DOCX/EPUB size their HTML chunks from ``TranslationConfig``'s default,
    which is ``MAX_TOKENS_PER_CHUNK`` frozen at import, NOT from the
    ``max_tokens_per_chunk`` passed to ``translate_file``/``refine_file``.
  - SRT block size is ``SRT_LINES_PER_BLOCK``, bound at import by each
    consumer module.
  - The number of concurrent workers falls back to ``PARALLEL_TRANSLATIONS``.

  A developer ``.env`` overriding any of them silently changed chunk counts
  (and could reorder progress ticks). They are pinned here so the goldens
  record the same thing on every machine.
"""

import dataclasses
import importlib

import pytest

import src.config as config
import src.core.epub.container as epub_container

# The shipped default of MAX_TOKENS_PER_CHUNK. It is what DOCX/EPUB actually
# chunk with, since they ignore the recorder's max_tokens_per_chunk=60.
PINNED_CONTAINER_MAX_TOKENS = 450

# 50 fixture subtitles at 20 lines per block -> 3 translate blocks, the
# layout fixtures.build_srt and the README document.
PINNED_SRT_LINES_PER_BLOCK = 20

# Sequential translation: the shipped default, and the only deterministic
# progress-tick order.
PINNED_PARALLEL_TRANSLATIONS = 1

# Modules that bind SRT_LINES_PER_BLOCK by name at import time. A module that
# no longer exists or no longer binds the name is skipped.
_SRT_BLOCK_CONSUMERS = (
    "src.core.subtitle_translator",
    "src.core.refine.srt_refiner",
)


@dataclasses.dataclass
class _PinnedTranslationConfig(epub_container.TranslationConfig):
    max_tokens_per_chunk: int = PINNED_CONTAINER_MAX_TOKENS


def _pin_srt_block_size(monkeypatch) -> None:
    monkeypatch.setattr(config, "SRT_LINES_PER_BLOCK", PINNED_SRT_LINES_PER_BLOCK)
    for module_name in _SRT_BLOCK_CONSUMERS:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if hasattr(module, "SRT_LINES_PER_BLOCK"):
            monkeypatch.setattr(
                module, "SRT_LINES_PER_BLOCK", PINNED_SRT_LINES_PER_BLOCK
            )


@pytest.fixture(autouse=True)
def _hermetic_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "MAX_TOKENS_PER_CHUNK", PINNED_CONTAINER_MAX_TOKENS)
    monkeypatch.setattr(config, "PARALLEL_TRANSLATIONS", PINNED_PARALLEL_TRANSLATIONS)
    monkeypatch.setattr(epub_container, "TranslationConfig", _PinnedTranslationConfig)
    _pin_srt_block_size(monkeypatch)
