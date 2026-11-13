"""Render prompts from one source tree, as a standalone subprocess.

Usage:
    python render.py --root <tree> [--env-file <path>] < requests.json > results.json

Self-contained on purpose: module-level imports are stdlib only and nothing is
imported from ``tests.*``. The harness runs this same file (the working-tree
copy) once against the baseline worktree and once against the working tree,
so the only difference between the two arms is the tree under ``--root``.

Stdin:  {"requests": [{"case_id", "builder", "kwargs": {...},
                       "glossary_terms": {...} | null, "target_language"}]}
Stdout: {"results": {case_id: {"system", "user", "dropped_kwargs": [...]}
                              | {"error": "<type>: <message>"}}}

Exit code is 0 even when individual cases fail; non-zero only on malformed
input or when ``src.prompts.prompts`` cannot be imported.
"""

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

BUILDERS = {
    "translate": "generate_translation_prompt",
    "refine": "generate_refinement_prompt",
}


def _build_glossary_block(text, glossary_terms, target_language, dropped):
    """Build the glossary block the way the translator does, tolerating drift."""
    try:
        from src.core.translator import _build_chunk_glossary_block
    except Exception:
        dropped.append("glossary_block")
        return ""
    options = {"glossary_terms": glossary_terms}
    try:
        return _build_chunk_glossary_block(text, options, target_language=target_language)
    except TypeError:
        pass
    try:
        return _build_chunk_glossary_block(text, options)
    except Exception:
        dropped.append("glossary_block")
        return ""


def render_one(prompts_module, request):
    builder_name = BUILDERS[request["builder"]]
    builder = getattr(prompts_module, builder_name)
    kwargs = dict(request["kwargs"])
    dropped = []

    glossary_terms = request.get("glossary_terms")
    if glossary_terms:
        text_key = "main_content" if request["builder"] == "translate" else "draft_translation"
        kwargs["glossary_block"] = _build_glossary_block(
            kwargs.get(text_key, ""), glossary_terms, request.get("target_language", ""), dropped
        )
    else:
        kwargs.setdefault("glossary_block", "")

    accepted = inspect.signature(builder).parameters
    for name in list(kwargs):
        if name not in accepted:
            dropped.append(name)
            del kwargs[name]

    pair = builder(**kwargs)
    return {"system": pair.system, "user": pair.user, "dropped_kwargs": sorted(set(dropped))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, help="Source tree to import src from")
    parser.add_argument("--env-file", default=None, help=".env loaded before importing src")
    args = parser.parse_args(argv)

    # Anything the imported code prints must not corrupt the JSON on stdout.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    try:
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
        requests = payload["requests"]
        if not isinstance(requests, list):
            raise TypeError("'requests' must be a list")
    except Exception as exc:
        print(f"render.py: malformed input: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    root = str(Path(args.root).resolve())
    script_dir = Path(__file__).resolve().parent
    # Drop this script's folder so its sibling modules can never shadow imports.
    sys.path[:] = [root] + [p for p in sys.path if p and Path(p).resolve() != script_dir]

    if args.env_file and os.path.isfile(args.env_file):
        try:
            from dotenv import load_dotenv
            load_dotenv(args.env_file, override=False)
        except ImportError:
            pass

    try:
        import src.prompts.prompts as prompts_module
    except Exception as exc:
        print(f"render.py: cannot import src.prompts.prompts from {root}: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    imported_from = Path(prompts_module.__file__).resolve()
    if Path(root) not in imported_from.parents:
        print(f"render.py: src.prompts.prompts was imported from {imported_from}, not from {root}",
              file=sys.stderr)
        return 3

    results = {}
    for request in requests:
        case_id = request.get("case_id", "?")
        try:
            results[case_id] = render_one(prompts_module, request)
        except Exception as exc:
            results[case_id] = {"error": f"{type(exc).__name__}: {exc}"}

    real_stdout.buffer.write(json.dumps({"results": results}, ensure_ascii=False).encode("utf-8"))
    real_stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
