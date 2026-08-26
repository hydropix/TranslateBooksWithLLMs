"""
Custom instruction (style preset) routes.

Exposes CRUD for the extended-schema YAML presets stored under
`Custom_Instructions/`, plus two LLM-backed helpers:

- `POST /api/custom-instructions/assemble` turns a rule list into the two
  prose blocks injected into the translation/refinement prompts.
- `POST /api/custom-instructions/extract-style` samples uploaded documents
  and asks an LLM to characterize their writing style as a rule list.

`GET /api/custom-instructions` and `POST /api/custom-instructions/open-folder`
were moved here unchanged from `config_routes.py` — same URLs, same response
shapes — so `FormManager.loadCustomInstructions()` keeps working untouched.

All endpoints are mounted at the top level (not under a blueprint prefix) to
match the pre-existing URLs.
"""
import asyncio
import logging
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from flask import Blueprint, Response, jsonify, request

from src.api.api_keys import provider_env_var, resolve_api_key
from src.api.blueprints.config_routes import get_config_path
from src.api.services.endpoint_validator import EndpointValidator
from src.core.llm.exceptions import RateLimitError
from src.core.style.assembler import assemble_instructions
from src.core.style.extractor import extract_style
from src.core.style.lint import lint_instruction
from src.utils import document_sampler
from src.utils.custom_instructions import (
    PRESET_KEY_ORDER,
    delete_preset,
    filename_for_name,
    list_custom_instructions,
    read_preset,
    resolve_inside,
    slugify_preset_name,
    write_preset,
)

logger = logging.getLogger('custom_instruction_routes')

# --- extract-style upload/sampling limits -----------------------------------
_EXTRACT_UPLOAD_MAX_BYTES = 100 * 1024 * 1024
_EXTRACT_MAX_FILES = 5
_EXTRACT_MAX_CHARS_HARD_CAP = 12000
_EXTRACT_MIN_SAMPLE_SIZE = 1200
_EXTRACT_CONTEXT_WINDOW = 16384
_EXTRACT_DEFAULT_MAX_CHARS = 10000
_EXTRACT_DEFAULT_SAMPLE_COUNT = 6

# --- write-path caps table ---------------------------------------------------
_MAX_DESCRIPTION_CHARS = 300
_MAX_CONTEXT_CHARS = 600
_MAX_RULES = 40
_MAX_RULE_INSTRUCTION_CHARS = 500
_MAX_PROSE_CHARS = 20000
_MAX_SOURCE_FILES = 10
_MAX_SOURCE_FILE_CHARS = 260
_MAX_DUPLICATE_ATTEMPTS = 50


def _custom_instructions_dir() -> Path:
    """Resolve the presets directory at request time (not import time).

    Tests monkeypatch `get_config_path`, so this must never be cached at
    module load.
    """
    return Path(get_config_path()) / 'Custom_Instructions'


def create_custom_instruction_blueprint():
    """Create and configure the custom-instructions blueprint."""
    bp = Blueprint('custom_instructions', __name__)

    # -------------------------------------------------------------------
    # Moved unchanged from config_routes.py (I5, I6)
    # -------------------------------------------------------------------

    @bp.route('/api/custom-instructions', methods=['GET'])
    def get_custom_instructions():
        """List available custom instruction files from Custom_Instructions/ folder.

        Each entry carries `has_translation` / `has_refinement` so the UI can
        filter presets per phase. `.txt` files (legacy) apply to both phases;
        `.yaml`/`.yml` files report the phases actually present in the file.
        """
        try:
            custom_instructions_dir = _custom_instructions_dir()

            if not custom_instructions_dir.exists():
                return jsonify({"files": [], "count": 0, "status": "folder_not_found"})

            files = list_custom_instructions(custom_instructions_dir)
            return jsonify({"files": files, "count": len(files), "status": "ok"})

        except Exception as e:
            logger.error(f"Error listing custom instructions: {e}")
            return jsonify({"files": [], "count": 0, "status": "error", "error": str(e)})

    @bp.route('/api/custom-instructions/open-folder', methods=['POST'])
    def open_custom_instructions_folder():
        """Open the Custom_Instructions folder in the system file explorer"""
        try:
            custom_instructions_dir = _custom_instructions_dir()

            # Create folder if it doesn't exist
            if not custom_instructions_dir.exists():
                custom_instructions_dir.mkdir(parents=True, exist_ok=True)

            abs_path = str(custom_instructions_dir.resolve())
            system = platform.system()

            if system == 'Windows':
                os.startfile(abs_path)
            elif system == 'Darwin':  # macOS
                subprocess.run(['open', abs_path], check=True)
            else:  # Linux and others
                subprocess.run(['xdg-open', abs_path], check=True)

            return jsonify({"success": True, "path": abs_path})

        except Exception as e:
            logger.error(f"Error opening custom instructions folder: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    # -------------------------------------------------------------------
    # New CRUD endpoints (Phase 5)
    # -------------------------------------------------------------------

    @bp.route('/api/custom-instructions/<filename>', methods=['GET'])
    def get_custom_instruction(filename):
        directory = _custom_instructions_dir()
        try:
            preset = read_preset(filename, directory)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 404
        except yaml.YAMLError as e:
            return jsonify({"error": f"Malformed YAML in '{filename}': {e}"}), 422
        return jsonify(preset)

    @bp.route('/api/custom-instructions', methods=['POST'])
    def create_custom_instruction():
        body = request.get_json(silent=True) or {}

        name = body.get('name')
        if not isinstance(name, str) or not name.strip():
            return jsonify({"error": "'name' is required"}), 400

        try:
            filename = filename_for_name(name)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        payload, error = _validate_write_body(body, existing=None)
        if error:
            return jsonify(error[0]), error[1]

        directory = _custom_instructions_dir()
        overwrite = bool(body.get('overwrite'))
        return _write_preset_and_respond(filename, payload, directory, overwrite=overwrite, created=True)

    @bp.route('/api/custom-instructions/<filename>', methods=['PUT'])
    def update_custom_instruction(filename):
        directory = _custom_instructions_dir()
        try:
            existing = read_preset(filename, directory)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 404
        except yaml.YAMLError as e:
            return jsonify({"error": f"Malformed YAML in '{filename}': {e}"}), 422

        body = request.get_json(silent=True) or {}
        payload, error = _validate_write_body(body, existing=existing)
        if error:
            return jsonify(error[0]), error[1]

        return _write_preset_and_respond(filename, payload, directory, overwrite=True, created=False)

    @bp.route('/api/custom-instructions/<filename>', methods=['DELETE'])
    def delete_custom_instruction(filename):
        directory = _custom_instructions_dir()
        try:
            deleted = delete_preset(filename, directory)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        if not deleted:
            return jsonify({"error": f"Custom instructions file not found: {filename}"}), 404
        return jsonify({"deleted": True})

    @bp.route('/api/custom-instructions/<filename>/duplicate', methods=['POST'])
    def duplicate_custom_instruction(filename):
        directory = _custom_instructions_dir()
        try:
            source = read_preset(filename, directory)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 404
        except yaml.YAMLError as e:
            return jsonify({"error": f"Malformed YAML in '{filename}': {e}"}), 422

        body = request.get_json(silent=True) or {}
        explicit_name = body.get('name')

        if isinstance(explicit_name, str) and explicit_name.strip():
            try:
                slug = slugify_preset_name(explicit_name)
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            candidate_filename = f"{slug}.yaml"
            target = resolve_inside(directory, candidate_filename)
            if target is not None and target.exists():
                return jsonify({"error": f"Preset '{candidate_filename}' already exists."}), 409
        else:
            base_name = source['display_name']
            candidate_filename = None
            for attempt in range(_MAX_DUPLICATE_ATTEMPTS):
                suffix = '_copy' if attempt == 0 else f'_copy{attempt + 1}'
                try:
                    slug = slugify_preset_name(f"{base_name}{suffix}")
                except ValueError as e:
                    return jsonify({"error": str(e)}), 400
                fname = f"{slug}.yaml"
                target = resolve_inside(directory, fname)
                if target is not None and not target.exists():
                    candidate_filename = fname
                    break
            if candidate_filename is None:
                return jsonify({"error": "Could not find an available name for the duplicate."}), 409

        # Copy the source content verbatim. `manual: True` forces the shared
        # validator to keep the existing translation/refinement as-is instead
        # of re-assembling from rules, which would silently discard a manual
        # override the original preset might carry.
        dup_body = {
            'description': source.get('description', ''),
            'mode': source.get('mode'),
            'context': source.get('context', ''),
            'source_files': source.get('source_files', []),
            'rules': source.get('rules', []),
            'translation': source.get('translation'),
            'refinement': source.get('refinement'),
            'manual': True,
        }
        payload, error = _validate_write_body(dup_body, existing=None)
        if error:
            return jsonify(error[0]), error[1]

        return _write_preset_and_respond(candidate_filename, payload, directory, overwrite=False, created=True)

    @bp.route('/api/custom-instructions/<filename>/export', methods=['GET'])
    def export_custom_instruction(filename):
        directory = _custom_instructions_dir()
        try:
            preset = read_preset(filename, directory)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 404
        except yaml.YAMLError as e:
            return jsonify({"error": f"Malformed YAML in '{filename}': {e}"}), 422

        export_data = {
            key: preset[key]
            for key in PRESET_KEY_ORDER
            if preset.get(key) not in (None, [], '')
        }
        body_yaml = yaml.safe_dump(
            export_data, allow_unicode=True, sort_keys=False, default_flow_style=False, width=100000
        )
        export_filename = f"{preset['display_name']}.yaml"
        response = Response(body_yaml, mimetype='application/x-yaml')
        response.headers['Content-Disposition'] = f'attachment; filename="{export_filename}"'
        return response

    @bp.route('/api/custom-instructions/assemble', methods=['POST'])
    def assemble_custom_instruction():
        body = request.get_json(silent=True) or {}

        mode = body.get('mode')
        if mode not in ('source', 'model'):
            return jsonify({"error": "'mode' must be 'source' or 'model'"}), 400

        rules = body.get('rules')
        if not isinstance(rules, list):
            return jsonify({"error": "'rules' must be a list"}), 400

        context = body.get('context') or ''
        if not isinstance(context, str):
            return jsonify({"error": "'context' must be a string"}), 400
        if len(context) > _MAX_CONTEXT_CHARS:
            return jsonify({"error": f"'context' must be at most {_MAX_CONTEXT_CHARS} characters"}), 400

        assembled = assemble_instructions(mode, rules, context)
        flags = [
            lint_instruction(str(rule.get('instruction') or '')) if isinstance(rule, dict) else []
            for rule in rules
        ]

        return jsonify({
            "translation": assembled["translation"],
            "refinement": assembled["refinement"],
            "flags": flags,
        })

    @bp.route('/api/custom-instructions/extract-style', methods=['POST'])
    def extract_style_endpoint():
        """Sample uploaded documents and ask an LLM to characterize their style.

        Multipart only. See the Phase 5 plan for the sampling algorithm and
        the exact response shape. Nothing is written to disk here.
        """
        try:
            uploads = [f for f in request.files.getlist('files') if f and f.filename]
            if not uploads:
                return jsonify({"error": "At least one file upload is required"}), 400
            if len(uploads) > _EXTRACT_MAX_FILES:
                return jsonify({
                    "error": f"At most {_EXTRACT_MAX_FILES} files may be uploaded at once"
                }), 400

            data = request.form

            mode = data.get('mode') or 'source'
            if mode not in ('source', 'model'):
                return jsonify({"error": "'mode' must be 'source' or 'model'"}), 400

            source_lang = data.get('source_lang') or 'English'
            target_lang = data.get('target_lang') or 'English'

            try:
                max_chars = int(data.get('max_chars') or _EXTRACT_DEFAULT_MAX_CHARS)
            except (TypeError, ValueError):
                return jsonify({"error": "max_chars must be an integer"}), 400
            if max_chars <= 0:
                return jsonify({"error": "max_chars must be positive"}), 400
            if max_chars > _EXTRACT_MAX_CHARS_HARD_CAP:
                max_chars = _EXTRACT_MAX_CHARS_HARD_CAP

            try:
                sample_count = int(data.get('sample_count') or _EXTRACT_DEFAULT_SAMPLE_COUNT)
            except (TypeError, ValueError):
                return jsonify({"error": "sample_count must be an integer"}), 400
            if sample_count < 1 or sample_count > 20:
                return jsonify({"error": "sample_count must be between 1 and 20"}), 400

            file_payloads: List[Tuple[str, bytes]] = []
            for upload in uploads:
                ext = Path(upload.filename).suffix.lower()
                if ext not in document_sampler.SUPPORTED_EXTS:
                    return jsonify({
                        "error": (
                            f"Unsupported file type '{ext or '?'}'. "
                            "Expected one of: " + ", ".join(sorted(document_sampler.SUPPORTED_EXTS))
                        )
                    }), 400

                raw = upload.read(_EXTRACT_UPLOAD_MAX_BYTES + 1)
                if len(raw) > _EXTRACT_UPLOAD_MAX_BYTES:
                    return jsonify({
                        "error": (
                            f"File '{upload.filename}' is too large "
                            f"(max {_EXTRACT_UPLOAD_MAX_BYTES // (1024 * 1024)} MB)"
                        )
                    }), 413
                file_payloads.append((upload.filename, raw))

            n_files = len(file_payloads)
            per_file_budget = max_chars // n_files
            remainder = max_chars - per_file_budget * n_files
            per_file_samples = max(1, sample_count // n_files)

            pre_warnings: List[str] = []
            per_file_meta: List[Dict[str, Any]] = []
            joined_blocks: List[str] = []

            for index, (fname, raw) in enumerate(file_payloads):
                budget = per_file_budget + (remainder if index == 0 else 0)
                joined_i, effective_i, full_chars_i = document_sampler.extract_samples_from_upload(
                    raw, fname, budget, per_file_samples, min_sample_size=_EXTRACT_MIN_SAMPLE_SIZE
                )
                if not joined_i:
                    pre_warnings.append(f"Could not extract any text from '{fname}' — skipped.")
                    continue

                if effective_i < per_file_samples:
                    if effective_i == 1 and full_chars_i <= budget:
                        pre_warnings.append(
                            f"'{fname}' is short ({full_chars_i} chars) — sent in full instead of "
                            f"{per_file_samples} excerpts."
                        )
                    else:
                        pre_warnings.append(
                            f"'{fname}': reduced from {per_file_samples} to {effective_i} excerpts "
                            f"(each excerpt needs ≥ {_EXTRACT_MIN_SAMPLE_SIZE} chars)."
                        )

                header = f"\n\n===== EXCERPTS FROM: {fname} =====\n\n"
                joined_blocks.append(f"{header}{joined_i}")
                per_file_meta.append({
                    "filename": fname,
                    "sample_chars": len(joined_i),
                    "sample_count": effective_i,
                    "full_text_chars": full_chars_i,
                })

            if not joined_blocks:
                return jsonify({"error": "Could not extract any text from the uploaded files."}), 400

            combined_text = "".join(joined_blocks)
            sample_chars = len(combined_text)

            import src.config as _config
            from src.core.llm.factory import create_llm_provider

            provider_type = (data.get('provider') or _config.LLM_PROVIDER or 'ollama').lower()
            model = data.get('model') or _config.DEFAULT_MODEL
            api_endpoint = data.get('api_endpoint') or _config.provider_default_endpoint(provider_type)

            ok, endpoint_error = EndpointValidator.validate(api_endpoint)
            if not ok:
                return jsonify({"error": endpoint_error}), 400

            # The frontend sends the '__USE_ENV__' sentinel (or nothing) when the
            # key field is empty but a key is configured in .env; resolve_api_key
            # turns that back into the real, possibly multi-key, env value.
            # A caller-chosen endpoint gets no .env key: the stored credential
            # must never travel to a host the request picked.
            requested_endpoint = (api_endpoint or '').strip().rstrip('/')
            is_endpoint_override = _config.is_provider_endpoint_override(
                provider_type, requested_endpoint
            )
            env_var = provider_env_var(provider_type)
            api_key = resolve_api_key(
                data.get('api_key'),
                env_var,
                getattr(_config, env_var, '') if env_var else '',
                allow_env_fallback=not is_endpoint_override,
            ) or None

            try:
                provider = create_llm_provider(
                    provider_type=provider_type,
                    model=model,
                    api_endpoint=api_endpoint,
                    api_key=api_key,
                    context_window=_EXTRACT_CONTEXT_WINDOW,
                )
            except Exception as e:
                return jsonify({"error": f"Could not initialize provider '{provider_type}': {e}"}), 400

            async def _extract_with_cleanup():
                # Closing the provider on the same loop that opened it lets
                # httpx finalize its streaming async generators synchronously,
                # avoiding 'Task was destroyed' warnings on loop teardown.
                try:
                    return await extract_style(
                        text=combined_text,
                        mode=mode,
                        source_language=source_lang,
                        target_language=target_lang,
                        llm_provider=provider,
                        max_chars=max_chars,
                    )
                finally:
                    try:
                        await provider.close()
                    except Exception:
                        pass

            def _run_extract():
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(_extract_with_cleanup())
                    finally:
                        try:
                            loop.run_until_complete(loop.shutdown_asyncgens())
                        except Exception:
                            pass
                finally:
                    try:
                        loop.close()
                    finally:
                        asyncio.set_event_loop(None)

            # If a loop is already running on this thread, hop to a worker
            # thread so we don't recurse into it.
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None

            if running is None:
                style, extract_warnings = _run_extract()
            else:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    style, extract_warnings = pool.submit(_run_extract).result()

            combined_warnings = pre_warnings + list(extract_warnings or [])

            rules = style.get('rules', [])
            unflagged_rules = [rule for rule in rules if not rule.get('flags')]
            context = style.get('context', '')
            assembled = assemble_instructions(mode, unflagged_rules, context)

            return jsonify({
                "rules": rules,
                "summary": style.get('summary', ''),
                "suggested_name": style.get('suggested_name', 'extracted_style'),
                "context": context,
                "assembled": assembled,
                "mode": mode,
                "warnings": combined_warnings,
                "provider": provider_type,
                "model": model,
                "sample_chars": sample_chars,
                "per_file": per_file_meta,
            })

        except RateLimitError as e:
            logger.warning(
                f"Rate limited while extracting style: "
                f"provider={e.provider} retry_after={e.retry_after}"
            )
            payload = {
                "error": str(e),
                "provider": e.provider,
                "retry_after": e.retry_after,
            }
            response = jsonify(payload)
            response.status_code = 429
            if e.retry_after is not None:
                response.headers['Retry-After'] = str(e.retry_after)
            return response
        except Exception as e:
            logger.error(f"Error extracting style: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    return bp


def _validate_write_body(
    body: Dict[str, Any], existing: Optional[Dict[str, Any]]
) -> Tuple[Optional[Dict[str, Any]], Optional[Tuple[Dict[str, str], int]]]:
    """Validate a create/update/duplicate request body.

    Enforces the caps table and the rules -> prose assembly contract (plan
    Phase 5, "Shared validation for write paths"): when `rules` is provided
    and non-empty, `translation`/`refinement` are re-assembled from the
    rules and any client-supplied prose is ignored, unless the body carries
    `"manual": true`, in which case the client prose is stored verbatim and
    `rules` is still persisted as metadata.

    `existing` is the current `read_preset(...)` mapping when updating
    (PUT), or `None` when creating/duplicating. Only keys present in `body`
    are validated and included in the returned payload — write_preset's
    read-merge-write preserves whatever is omitted, which is what makes a
    PUT a true partial update.

    Returns `(payload, None)` on success, or `(None, (error_body, status))`
    on the first validation failure.
    """

    def fail(message: str, status: int = 400):
        return None, ({"error": message}, status)

    payload: Dict[str, Any] = {}

    if 'description' in body:
        description = body.get('description') or ''
        if not isinstance(description, str):
            return fail("'description' must be a string")
        if len(description) > _MAX_DESCRIPTION_CHARS:
            return fail(f"'description' must be at most {_MAX_DESCRIPTION_CHARS} characters")
        payload['description'] = description

    if 'context' in body:
        context = body.get('context') or ''
        if not isinstance(context, str):
            return fail("'context' must be a string")
        if len(context) > _MAX_CONTEXT_CHARS:
            return fail(f"'context' must be at most {_MAX_CONTEXT_CHARS} characters")
        payload['context'] = context

    mode = body.get('mode')
    if mode is not None and mode not in ('source', 'model'):
        return fail("'mode' must be 'source' or 'model'")
    if 'mode' in body:
        payload['mode'] = mode

    if 'source_files' in body:
        source_files = body.get('source_files') or []
        if not isinstance(source_files, list) or not all(isinstance(s, str) for s in source_files):
            return fail("'source_files' must be a list of strings")
        if len(source_files) > _MAX_SOURCE_FILES:
            return fail(f"'source_files' must contain at most {_MAX_SOURCE_FILES} entries")
        for entry in source_files:
            if len(entry) > _MAX_SOURCE_FILE_CHARS:
                return fail(f"each 'source_files' entry must be at most {_MAX_SOURCE_FILE_CHARS} characters")
        payload['source_files'] = source_files

    rules_provided = 'rules' in body
    normalized_rules: List[Dict[str, str]] = []
    if rules_provided:
        rules = body.get('rules') or []
        if not isinstance(rules, list):
            return fail("'rules' must be a list")
        if len(rules) > _MAX_RULES:
            return fail(f"'rules' must contain at most {_MAX_RULES} entries")
        for rule in rules:
            if not isinstance(rule, dict):
                return fail("each 'rules' entry must be an object")
            instruction = str(rule.get('instruction') or '')
            if len(instruction) > _MAX_RULE_INSTRUCTION_CHARS:
                return fail(f"each rule 'instruction' must be at most {_MAX_RULE_INSTRUCTION_CHARS} characters")
            normalized_rules.append({
                'dimension': str(rule.get('dimension') or ''),
                'instruction': instruction,
            })
        payload['rules'] = normalized_rules

    manual = bool(body.get('manual'))
    non_empty_rules = [rule for rule in normalized_rules if rule['instruction'].strip()]

    if rules_provided and non_empty_rules and not manual:
        effective_mode = mode if 'mode' in body else (existing or {}).get('mode')
        effective_mode = effective_mode or 'source'
        effective_context = payload['context'] if 'context' in payload else (existing or {}).get('context', '')
        assembled = assemble_instructions(effective_mode, non_empty_rules, effective_context or '')
        payload['translation'] = assembled['translation']
        payload['refinement'] = assembled['refinement']
    else:
        if 'translation' in body:
            translation = body.get('translation')
            if translation is not None and not isinstance(translation, str):
                return fail("'translation' must be a string")
            payload['translation'] = translation
        if 'refinement' in body:
            refinement = body.get('refinement')
            if refinement is not None and not isinstance(refinement, str):
                return fail("'refinement' must be a string")
            payload['refinement'] = refinement

    for phase in ('translation', 'refinement'):
        value = payload.get(phase)
        if isinstance(value, str) and len(value) > _MAX_PROSE_CHARS:
            return fail(f"'{phase}' must be at most {_MAX_PROSE_CHARS} characters")

    final_translation = payload['translation'] if 'translation' in payload else (existing or {}).get('translation')
    final_refinement = payload['refinement'] if 'refinement' in payload else (existing or {}).get('refinement')
    if not (final_translation or '').strip() and not (final_refinement or '').strip():
        return fail("A preset must define at least one of translation/refinement")

    return payload, None


def _write_preset_and_respond(
    filename: str, payload: Dict[str, Any], directory: Path, *, overwrite: bool, created: bool
):
    """Shared write-path used by create/update/duplicate.

    Calls `write_preset`, maps its exceptions to the documented status
    codes, and builds the `{filename, display_name}` success response.
    """
    directory.mkdir(parents=True, exist_ok=True)
    try:
        write_preset(filename, payload, directory, overwrite=overwrite)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileExistsError as e:
        return jsonify({"error": str(e)}), 409

    display_name = Path(filename).stem
    status = 201 if created else 200
    return jsonify({"filename": filename, "display_name": display_name}), status
