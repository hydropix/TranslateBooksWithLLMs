"""Acceptance harness for issue #261 - failed chunks must be recoverable.

Cluster context: #239 / #246 / #261 are three faces of "we announce a success
that is not one". The completion-classifier half (#239, #246) landed in
`src/api/completion_status.py` (commit 3d310be): a run with fallback chunks is
classified 'partial' and its checkpoint is kept. The retry half is #261, planned
in `plan/PLAN_Issue261_RetryFailedChunks.md` and asserted here.

This file started life as the reproduction script for the bug and is now the
acceptance harness for its fix. It runs a real 3-chapter EPUB translation
against the Ollama endpoint declared in `.env`, forcing chapter 2's only chunk
to exhaust every attempt (the LLM returns nothing for it), so that chunk goes
through the Phase 3 fallback and stays in the source language. Then it resumes
the job the way `POST /api/resume/<id>` does and checks what happens to that
chunk.

Two modes:

  --mode heal (default) - the case that unblocks the user.
      Pass 1 starves chapter 2 -> verdict 'partial', checkpoint kept, ticket
      recorded. Pass 2 resumes with the starvation LIFTED and must retry that
      one chunk, translate it, come back 'completed' with no ticket, no partial
      state and a cleaned-up checkpoint.

  --mode persist - the no-false-success case.
      Same pass 1, then a resume with the starvation STILL IN PLACE. The chunk
      must be retried exactly once, the job must stay 'partial', the ticket and
      the partial state must survive, and the output must be unchanged. No
      false success, and no retry loop either.

Pre-fix baseline (measured on `main` at 19faf6d, before the fix):
`--mode heal` failed with `retried=0` and chapter 2 still untranslated - the
resume pointer is a file index, so the file holding the bad chunk counted as
finished, and the stale per-file partial state declared it complete anyway.

What it prints is evidence, not opinion: chunk attempts and raw LLM calls per
pass, the job-level `epub_unfinished_units` map, the per-file partial states,
the checkpoint rows, the verdicts, and whether the sabotaged paragraph is still
in the source language in the output EPUB. Every gating assertion is reported as
a PASS/FAIL line; the exit code is 0 only if all of them pass.

The "pass 3" forced-re-entry probe of the original repro is kept as a
diagnostic only (never gating). It runs in `--mode persist` only: it rewinds the
file pointer to 0 while a ticket is outstanding, which used to translate nothing
at all. In `--mode heal` there is nothing left to force - the job completed and
its checkpoint was deliberately cleaned up - so the probe is skipped.

Run from repo root (needs a reachable Ollama endpoint in .env):
    python tests/standalone/repro_issue_261_failed_chunks_unrecoverable.py --mode heal
    python tests/standalone/repro_issue_261_failed_chunks_unrecoverable.py --mode persist
"""

import argparse
import asyncio
import os
import shutil
import sys
import uuid
import zipfile
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from src import config  # noqa: F401  (loads .env)
import src.core.epub.xhtml_translator as xhtml_translator_module
from src.api.completion_status import classify_completion
from src.core.adapters import translate_file
from src.persistence.checkpoint_manager import CheckpointManager

# The sabotaged chunk is identified by this sentinel: the wrapper below returns
# None for any chunk containing it, in both Phase 1 (with placeholders) and
# Phase 2 (placeholders stripped), which is what pushes it to Phase 3.
SENTINEL = "The keeper refused to name the seventh lamp"
MODEL = os.getenv("TBL_REPRO_MODEL", "gemma3:4b")
MAX_TOKENS_PER_CHUNK = 150

MODE_HEAL = "heal"
MODE_PERSIST = "persist"

# The file and chunk the whole harness is about.
SABOTAGED_HREF = "chapter2.xhtml"
SABOTAGED_CHUNK_INDEX = 0

PARAGRAPHS = [
    "The lighthouse keeper's journal had grown thick with salt and ink, and the "
    "third assistant watched the older man with the polite patience of someone "
    "who has been told he must learn a great deal in very little time.",
    "Every evening the lamp was wound, the wick trimmed, and the log signed in "
    "the same narrow hand that had signed it for thirty-one winters.",
    "On the mainland they said the light had never failed, which was true only "
    "if one agreed not to count the night of the storm.",
]


class Checks:
    """Collector for the gating assertions, printed as a PASS/FAIL report."""

    def __init__(self, mode):
        self.mode = mode
        self.rows = []

    def check(self, name, ok, evidence):
        self.rows.append((bool(ok), name, evidence))
        return bool(ok)

    def note(self, name, evidence):
        """Non-gating observation, printed with the report."""
        self.rows.append((None, name, evidence))

    def report(self):
        print("ASSERTIONS (--mode %s)" % self.mode)
        failures = 0
        for ok, name, evidence in self.rows:
            if ok is None:
                label = "NOTE"
            elif ok:
                label = "PASS"
            else:
                label = "FAIL"
                failures += 1
            print("  [%s] %s" % (label, name))
            print("         evidence: %s" % (evidence,))
        gating = sum(1 for ok, _, _ in self.rows if ok is not None)
        print()
        if failures:
            print("RESULT: FAIL - %d of %d assertions failed" % (failures, gating))
        else:
            print("RESULT: PASS - all %d assertions hold" % gating)
        return 1 if failures else 0


def _chapter_xhtml(title, paragraphs):
    body = "\n".join("    <p>%s</p>" % p for p in paragraphs)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE html>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en">\n'
        '<head>\n'
        '  <title>%s</title>\n'
        '</head>\n'
        '<body>\n'
        '  <h1>%s</h1>\n'
        '%s\n'
        '</body>\n'
        '</html>\n' % (title, title, body)
    )


def _build_epub(path):
    """3 chapters; chapter 2 carries the sentinel paragraph that will fail."""
    hrefs = ["chapter%d.xhtml" % (i + 1) for i in range(3)]
    manifest = "\n".join(
        '    <item id="ch%d" href="%s" media-type="application/xhtml+xml"/>'
        % (i + 1, h)
        for i, h in enumerate(hrefs)
    )
    spine = "\n".join('    <itemref idref="ch%d"/>' % (i + 1) for i in range(3))
    content_opf = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<package version="3.0" xmlns="http://www.idpf.org/2007/opf" unique-identifier="bookid">\n'
        '  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
        '    <dc:identifier id="bookid">urn:uuid:00000000-0000-0000-0000-000000000261</dc:identifier>\n'
        '    <dc:title>The Seventh Lamp</dc:title>\n'
        '    <dc:language>en</dc:language>\n'
        '    <meta property="dcterms:modified">2024-01-01T00:00:00Z</meta>\n'
        '  </metadata>\n'
        '  <manifest>\n'
        + manifest + '\n'
        '  </manifest>\n'
        '  <spine>\n'
        + spine + '\n'
        '  </spine>\n'
        '</package>\n'
    )
    container_xml = (
        '<?xml version="1.0"?>\n'
        '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">\n'
        '  <rootfiles>\n'
        '    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>\n'
        '  </rootfiles>\n'
        '</container>\n'
    )
    sabotaged = list(PARAGRAPHS)
    sabotaged[1] = (
        SENTINEL + ", and the assistant learned not to ask twice about the "
        "shuttered alcove at the top of the stair."
    )

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("mimetype", "application/epub+zip", zipfile.ZIP_STORED)
        z.writestr("META-INF/container.xml", container_xml)
        z.writestr("OEBPS/content.opf", content_opf)
        for i, href in enumerate(hrefs):
            paragraphs = sabotaged if i == 1 else PARAGRAPHS
            z.writestr("OEBPS/" + href,
                       _chapter_xhtml("Chapter %d" % (i + 1), paragraphs))


class ChunkSabotage:
    """Instruments one pass, and optionally starves the sentinel chunk.

    Two levels are wrapped, because they answer two different questions:

    * `generate_translation_request` is the raw LLM call, and the level where
      the starvation has to happen: returning None for any request carrying the
      SENTINEL starves the chunk in Phase 1 (placeholders present) *and* in
      Phase 2 (placeholders stripped), which is what pushes it into the Phase 3
      fallback. Counting here counts raw LLM calls, and one starved chunk costs
      several of them (MAX_TRANSLATION_ATTEMPTS Phase 1 attempts plus the
      single Phase 2 call). Useful evidence, useless as an assertion target.
    * `translate_chunk_with_fallback` is entered exactly once per chunk per
      pass, whatever the ladder does inside. Counting its sentinel-carrying
      invocations therefore counts *attempts on the sabotaged chunk*, which is
      the number every assertion in this harness is written against.

    So: assertions count chunk attempts, printed evidence also shows raw calls.
    Metadata translation goes through another module and is counted by neither.
    """

    def __init__(self, starve):
        self.starve = starve
        self._original_request = xhtml_translator_module.generate_translation_request
        self._original_chunk = xhtml_translator_module.translate_chunk_with_fallback
        self.llm_calls = 0
        self.sabotaged_llm_calls = 0
        self.chunk_attempts = 0
        self.sabotaged_chunk_attempts = 0

    def __enter__(self):
        original_request = self._original_request
        original_chunk = self._original_chunk

        async def wrapped_request(main_content, *args, **kwargs):
            self.llm_calls += 1
            if SENTINEL in main_content:
                self.sabotaged_llm_calls += 1
                if self.starve:
                    return None
            return await original_request(main_content, *args, **kwargs)

        async def wrapped_chunk(*args, **kwargs):
            chunk_text = kwargs.get('chunk_text')
            if chunk_text is None:
                chunk_text = args[0] if args else ""
            self.chunk_attempts += 1
            if SENTINEL in chunk_text:
                self.sabotaged_chunk_attempts += 1
            return await original_chunk(*args, **kwargs)

        xhtml_translator_module.generate_translation_request = wrapped_request
        xhtml_translator_module.translate_chunk_with_fallback = wrapped_chunk
        return self

    def __exit__(self, *exc):
        xhtml_translator_module.generate_translation_request = self._original_request
        xhtml_translator_module.translate_chunk_with_fallback = self._original_chunk
        return False


def _chapter_report(epub_path, index):
    """How many source paragraphs of a chapter survived verbatim in the output."""
    href = "chapter%d.xhtml" % (index + 1)
    text = _epub_chapter_text(epub_path, href)
    if not text:
        return href, "MISSING", 0, 3
    sources = list(PARAGRAPHS)
    if index == 1:
        sources[1] = SENTINEL
    left = sum(1 for p in sources if p[:60] in text)
    state = "untranslated" if left == len(sources) else (
        "translated" if left == 0 else "mixed")
    return href, state, left, len(sources)


def _epub_chapter_text(epub_path, href):
    if not Path(epub_path).exists():
        return ""
    with zipfile.ZipFile(epub_path) as z:
        for name in z.namelist():
            if name.endswith(href):
                return z.read(name).decode("utf-8", errors="replace")
    return ""


def _describe_checkpoint(cm, translation_id):
    data = cm.load_checkpoint(translation_id)
    if not data:
        return None
    job = data["job"]
    progress = job["progress"]
    return {
        "status": job["status"],
        "resume_from_index": data["resume_from_index"],
        "current_chunk_index": progress.get("current_chunk_index"),
        "total_chunks": progress.get("total_chunks"),
        "completed_chunks": progress.get("completed_chunks"),
        "failed_chunks_in_progress": progress.get("failed_chunks"),
        "epub_unfinished_units": progress.get("epub_unfinished_units"),
        "epub_accumulated_stats": progress.get("epub_accumulated_stats"),
        "chunk_rows": [(c["chunk_index"], c.get("status")) for c in data["chunks"]],
        "failed_chunk_indices": data["failed_chunk_indices"],
        "xhtml_partial_states": cm.list_xhtml_partial_states(translation_id),
    }


async def _run_pass(label, cm, translation_id, input_path, output_path,
                    resume_from_index, starve):
    stats_seen = {}

    def stats_callback(stats):
        stats_seen.update(stats)

    interesting = ("checkpoint", "restore", "fail", "fallback", "phase3",
                   "partial", "save", "retry")

    def log_callback(kind, message, **kwargs):
        if any(word in kind for word in interesting) or kind.endswith("translate_start"):
            print("    [%s] %s" % (kind, message))

    with ChunkSabotage(starve=starve) as sabotage:
        await translate_file(
            input_filepath=str(input_path),
            output_filepath=str(output_path),
            source_language="English",
            target_language="French",
            model_name=MODEL,
            llm_provider="ollama",
            checkpoint_manager=cm,
            translation_id=translation_id,
            log_callback=log_callback,
            stats_callback=stats_callback,
            resume_from_index=resume_from_index,
            llm_api_endpoint=os.getenv("OLLAMA_API_ENDPOINT"),
            context_window=4096,
            auto_adjust_context=True,
            max_tokens_per_chunk=MAX_TOKENS_PER_CHUNK,
            prompt_options={},
        )

    print("  %s: starvation=%s chunk attempts=%d (on the sabotaged chunk=%d)"
          % (label, "on" if starve else "off", sabotage.chunk_attempts,
             sabotage.sabotaged_chunk_attempts))
    print("  %s: raw chunk-level LLM calls=%d (on the sabotaged chunk=%d)"
          % (label, sabotage.llm_calls, sabotage.sabotaged_llm_calls))
    print("  %s: stats total=%s completed=%s failed=%s fallback_used=%s "
          "token_alignment_used=%s unfinished_chunks=%s unfinished_files=%s"
          % (label, stats_seen.get("total_chunks"),
             stats_seen.get("completed_chunks"), stats_seen.get("failed_chunks"),
             stats_seen.get("fallback_used"),
             stats_seen.get("token_alignment_used"),
             stats_seen.get("unfinished_chunks"),
             stats_seen.get("unfinished_files")))
    return stats_seen, sabotage


def _finalize(cm, translation_id, verdict):
    """Mirror the finalization branches of src/api/handlers.py."""
    if verdict.status == "partial":
        cm.mark_partial(translation_id)
    elif verdict.status == "error":
        cm.mark_error(translation_id)
    else:
        cm.delete_checkpoint(translation_id)


async def run(mode):
    endpoint = os.getenv("OLLAMA_API_ENDPOINT", "").strip()
    if not endpoint:
        print("FAIL: OLLAMA_API_ENDPOINT is not set in .env - cannot run the "
              "acceptance harness against a real endpoint")
        return 1

    checks = Checks(mode)
    work_dir = Path(os.getenv("TEMP", "/tmp")) / ("repro_261_" + uuid.uuid4().hex[:8])
    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = work_dir / "seventh_lamp.epub"
    output_path = work_dir / "seventh_lamp_fr.epub"
    _build_epub(input_path)

    translation_id = "repro261_" + uuid.uuid4().hex[:8]
    cm = CheckpointManager(db_path=str(work_dir / "jobs.db"),
                           server_session_id="repro")
    cm.start_job(
        translation_id=translation_id,
        file_type="epub",
        config={
            "file_path": str(input_path),
            "output_filename": output_path.name,
            "source_language": "English",
            "target_language": "French",
            "model": MODEL,
            "llm_provider": "ollama",
            "file_type": "epub",
        },
        input_file_path=str(input_path),
    )

    print("Mode:       %s" % mode)
    print("Endpoint:   %s" % endpoint)
    print("Model:      %s" % MODEL)
    print("Job:        %s" % translation_id)
    print("Input:      %s (3 chapters, chapter 2 sabotaged)" % input_path)
    print()

    try:
        # === PASS 1 - initial translation, chapter 2 starved ===
        print("PASS 1 - initial translation (chapter 2 starved)")
        stats1, sab1 = await _run_pass(
            "pass1", cm, translation_id, input_path, output_path,
            resume_from_index=0, starve=True)

        verdict1 = classify_completion(stats1, str(output_path))
        print("  pass1: classify_completion -> %s (failed=%d, fallback=%d, "
              "unfinished=%d)"
              % (verdict1.status, verdict1.failed_chunks, verdict1.fallback_chunks,
                 verdict1.unfinished_chunks))
        _finalize(cm, translation_id, verdict1)

        ch2_after_1 = _epub_chapter_text(output_path, SABOTAGED_HREF)
        sabotaged_left_after_1 = SENTINEL in ch2_after_1
        print("  pass1: sabotaged paragraph still in source language: %s"
              % sabotaged_left_after_1)

        cp1 = _describe_checkpoint(cm, translation_id)
        print("  pass1: checkpoint status=%s resume_from_index=%s current_chunk_index=%s"
              % (cp1["status"], cp1["resume_from_index"], cp1["current_chunk_index"]))
        print("  pass1: chunk rows=%s" % (cp1["chunk_rows"],))
        print("  pass1: failed_chunk_indices=%s" % (cp1["failed_chunk_indices"],))
        print("  pass1: epub_unfinished_units=%s" % (cp1["epub_unfinished_units"],))
        print("  pass1: xhtml partial states kept=%s" % (cp1["xhtml_partial_states"],))
        print("  pass1: epub_accumulated_stats fallback_used=%s"
              % ((cp1["epub_accumulated_stats"] or {}).get("fallback_used"),))
        print()

        units1 = cp1["epub_unfinished_units"] or {}
        states1 = cp1["xhtml_partial_states"] or []

        checks.check("pass 1 is classified 'partial'",
                     verdict1.status == "partial",
                     "verdict=%s failed=%d fallback=%d unfinished=%d"
                     % (verdict1.status, verdict1.failed_chunks,
                        verdict1.fallback_chunks, verdict1.unfinished_chunks))
        checks.check("pass 1 really starved the chunk (source text kept in the output)",
                     sabotaged_left_after_1,
                     "sentinel present in output %s: %s"
                     % (SABOTAGED_HREF, sabotaged_left_after_1))
        checks.check("pass 1 records the unfinished chunk under epub_unfinished_units",
                     units1.get(SABOTAGED_HREF) == [SABOTAGED_CHUNK_INDEX],
                     "epub_unfinished_units=%s" % (units1,))
        checks.check("pass 1 keeps the partial state of chapter 2 and only that one",
                     states1 == [SABOTAGED_HREF],
                     "list_xhtml_partial_states=%s" % (states1,))
        checks.note("pass 1 work performed",
                    "chunk attempts=%d (sabotaged=%d), raw LLM calls=%d "
                    "(sabotaged=%d), resume_from_index=%s"
                    % (sab1.chunk_attempts, sab1.sabotaged_chunk_attempts,
                       sab1.llm_calls, sab1.sabotaged_llm_calls,
                       cp1["resume_from_index"]))

        # === PASS 2 - resume, i.e. what the user clicks on the partial card ===
        starve_pass2 = (mode == MODE_PERSIST)
        print("PASS 2 - resume (starvation %s)"
              % ("still in place" if starve_pass2 else "lifted"))
        resume_index = cp1["resume_from_index"]
        preserved = cm.get_preserved_input_path(translation_id) or str(input_path)
        stats2, sab2 = await _run_pass(
            "pass2", cm, translation_id, preserved, output_path,
            resume_from_index=resume_index, starve=starve_pass2)

        verdict2 = classify_completion(stats2, str(output_path))
        print("  pass2: classify_completion -> %s (failed=%d, fallback=%d, "
              "unfinished=%d)"
              % (verdict2.status, verdict2.failed_chunks, verdict2.fallback_chunks,
                 verdict2.unfinished_chunks))
        ch2_after_2 = _epub_chapter_text(output_path, SABOTAGED_HREF)
        sabotaged_left_after_2 = SENTINEL in ch2_after_2

        cp2 = _describe_checkpoint(cm, translation_id)
        units2 = (cp2 or {}).get("epub_unfinished_units") or {}
        states2 = (cp2 or {}).get("xhtml_partial_states") or []
        print("  pass2: sabotaged paragraph still in source language: %s"
              % sabotaged_left_after_2)
        print("  pass2: epub_unfinished_units=%s" % (units2,))
        print("  pass2: xhtml partial states kept=%s" % (states2,))
        print("  pass2: resume_from_index=%s" % ((cp2 or {}).get("resume_from_index"),))
        print()

        # The one assertion both modes share: the sabotaged chunk was retried,
        # exactly once. Counted as chunk attempts, not raw LLM calls (see
        # ChunkSabotage): the Phase 1/2/3 ladder makes several calls per chunk.
        checks.check("resume retried the sabotaged chunk exactly once",
                     sab2.sabotaged_chunk_attempts == 1,
                     "sabotaged chunk attempts=%d (raw LLM calls on it=%d, "
                     "total chunk attempts this pass=%d)"
                     % (sab2.sabotaged_chunk_attempts, sab2.sabotaged_llm_calls,
                        sab2.chunk_attempts))
        checks.check("resume retried nothing else",
                     sab2.chunk_attempts == 1,
                     "total chunk attempts this pass=%d (expected 1: only the "
                     "unfinished chunk)" % sab2.chunk_attempts)

        if mode == MODE_HEAL:
            checks.check("chapter 2 is translated in the output EPUB",
                         not sabotaged_left_after_2,
                         "sentinel present in output %s: %s"
                         % (SABOTAGED_HREF, sabotaged_left_after_2))
            checks.check("resume is classified 'completed'",
                         verdict2.status == "completed",
                         "verdict=%s failed=%d fallback=%d unfinished=%d"
                         % (verdict2.status, verdict2.failed_chunks,
                            verdict2.fallback_chunks, verdict2.unfinished_chunks))
            checks.check("epub_unfinished_units is empty",
                         units2 == {},
                         "epub_unfinished_units=%s" % (units2,))
            checks.check("no XHTML partial state is left on disk",
                         states2 == [],
                         "list_xhtml_partial_states=%s" % (states2,))

            # Mirror handlers.py: a 'completed' verdict cleans the checkpoint up.
            _finalize(cm, translation_id, verdict2)
            after_cleanup = cm.load_checkpoint(translation_id)
            checks.check("the checkpoint is cleaned up after the healed run",
                         after_cleanup is None,
                         "load_checkpoint after delete_checkpoint -> %s"
                         % ("None" if after_cleanup is None
                            else "still present (status=%s)"
                                 % after_cleanup["job"]["status"]))
            print("PASS 3 - forced re-entry probe skipped: the job completed and "
                  "its checkpoint was cleaned up, there is nothing left to force")
            print()
        else:
            _finalize(cm, translation_id, verdict2)
            checks.check("the output is unchanged (chapter 2 still in the source language)",
                         sabotaged_left_after_2,
                         "sentinel present in output %s: %s"
                         % (SABOTAGED_HREF, sabotaged_left_after_2))
            checks.check("the failed retry stays classified 'partial'",
                         verdict2.status == "partial",
                         "verdict=%s failed=%d fallback=%d unfinished=%d"
                         % (verdict2.status, verdict2.failed_chunks,
                            verdict2.fallback_chunks, verdict2.unfinished_chunks))
            checks.check("the ticket is still in epub_unfinished_units",
                         units2.get(SABOTAGED_HREF) == [SABOTAGED_CHUNK_INDEX],
                         "epub_unfinished_units=%s" % (units2,))
            checks.check("the XHTML partial state of chapter 2 is still on disk",
                         states2 == [SABOTAGED_HREF],
                         "list_xhtml_partial_states=%s" % (states2,))
            state = cm.load_xhtml_partial_state(translation_id, SABOTAGED_HREF)
            checks.check("that partial state still validates and still holds the "
                         "source text of the failed chunk",
                         state is not None and state.validate() is True
                         and SENTINEL in (state.translated_chunks or [""])[
                             SABOTAGED_CHUNK_INDEX],
                         "state=%s chunk_statuses=%s"
                         % ("loaded" if state is not None else "None",
                            getattr(state, "chunk_statuses", None)))

            # Diagnostic only (never gating): rewind the file pointer to 0 with
            # the ticket still outstanding. Before the fix this translated
            # nothing at all, because the stale partial states declared every
            # file finished. It runs last because it overwrites the job's
            # progress, so every assertion above has already been evaluated.
            print("PASS 3 - forced full re-entry probe (resume_from_index=0), "
                  "diagnostic only")
            print("  xhtml partial states on disk before the probe: %s"
                  % (cm.list_xhtml_partial_states(translation_id),))
            output3 = output_path.with_name("pass3_" + output_path.name)
            stats3, sab3 = await _run_pass(
                "pass3", cm, translation_id, preserved, output3,
                resume_from_index=0, starve=True)
            verdict3 = classify_completion(stats3, str(output3))
            print("  pass3: classify_completion -> %s (unfinished=%d)"
                  % (verdict3.status, verdict3.unfinished_chunks))
            print("  pass3: sabotaged paragraph still in source language: %s"
                  % (SENTINEL in _epub_chapter_text(output3, SABOTAGED_HREF)))
            checks.note("pass 3 forced re-entry (diagnostic, not gating)",
                        "chunk attempts=%d (sabotaged=%d), verdict=%s"
                        % (sab3.chunk_attempts, sab3.sabotaged_chunk_attempts,
                           verdict3.status))
            print()

        print("OUTPUT EPUB, chapter by chapter (source paragraphs left verbatim)")
        for i in range(3):
            href, state_label, left, total = _chapter_report(output_path, i)
            print("  %-16s %-13s %d/%d source paragraphs still present"
                  % (href, state_label, left, total))
        print()

        print("Artifacts kept for inspection: %s" % work_dir)
        print()
        return checks.report()
    finally:
        # Always clean the job's upload directory, including on failure: the
        # partial states and preserved input live there.
        shutil.rmtree(Path("data/uploads") / translation_id, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Acceptance harness for issue #261 (retry of failed EPUB "
                    "chunks on resume), against the real Ollama endpoint "
                    "declared in .env.")
    parser.add_argument("--mode", choices=[MODE_HEAL, MODE_PERSIST],
                        default=MODE_HEAL,
                        help="heal: resume with the starvation lifted, the "
                             "chunk must be healed and the job completed. "
                             "persist: resume with the starvation still in "
                             "place, the job must stay partial and keep its "
                             "ticket. (default: heal)")
    args = parser.parse_args()
    return asyncio.run(run(args.mode))


if __name__ == "__main__":
    raise SystemExit(main())
