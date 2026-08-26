"""
Glossary management routes.

Provides REST endpoints to create/read/update/delete glossaries and their
terms, plus JSON/CSV import/export.

All endpoints are mounted under ``/api/glossaries``.
"""
import asyncio
import csv
import io
import logging
import re
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote

from flask import Blueprint, Response, jsonify, request

from src.core.glossary import Glossary, GlossaryStore, GlossaryTerm
from src.core.glossary import build_cast_block, build_glossary_block, filter_glossary
from src.core.glossary import suggest_terms as ner_suggest_terms
from src.core.glossary.models import GlossaryConfig
from src.core.llm.exceptions import RateLimitError
from src.api.api_keys import provider_env_var, resolve_api_key
from src.api.services.endpoint_validator import EndpointValidator
from src.utils import document_sampler
from src.utils.document_sampler import extract_samples_from_upload

_NER_UPLOAD_MAX_BYTES = 100 * 1024 * 1024
_NER_TEXT_EXTS = document_sampler.TEXT_EXTS
_NER_RICH_EXTS = document_sampler.RICH_EXTS
_NER_SUPPORTED_EXTS = document_sampler.SUPPORTED_EXTS

# Hard cap on text we pull from a single upload before sampling.
# 5M chars is more than the longest novels; protects memory on huge inputs.
_NER_FULL_TEXT_CAP = document_sampler.FULL_TEXT_CAP

# Minimum per-sample size: tinier excerpts are too fragmented for the LLM
# to spot recurring entities (≈80 words, the shortest useful passage).
_NER_MIN_SAMPLE_SIZE = 500

# Fixed context window used for the NER call (independent of OLLAMA_NUM_CTX).
# 8192 tokens is enough for the system prompt (~700) + ~6000 chars of
# CJK/dense text (worst case ~1 token/char) + ~1500 tokens for the response.
_NER_CONTEXT_WINDOW = 8192

# Hard cap on the source-text budget. Sized to fit safely in the context
# window above for every language (CJK is the worst case at 1 token/char).
_NER_MAX_CHARS_HARD_CAP = 6000

# Visible separator inserted between non-contiguous excerpts. The LLM treats
# it as a discontinuity hint without us needing to change the NER prompt.
_NER_EXCERPT_SEP = document_sampler.EXCERPT_SEPARATOR

logger = logging.getLogger('glossary_routes')


def create_glossary_blueprint(store: Optional[GlossaryStore] = None):
    """Create and configure the glossary blueprint.

    Args:
        store: Optional GlossaryStore. If not provided, a default one
               backed by ``data/glossaries.db`` is instantiated once for the
               lifetime of the blueprint.
    """
    bp = Blueprint('glossary', __name__)

    if store is None:
        store = GlossaryStore()

    # ------------------------------------------------------------------ helpers

    def _glossary_summary(glossary: Glossary) -> dict:
        """Return a glossary dict without its terms (for list views)."""
        data = glossary.to_dict()
        data.pop('terms', None)
        return data

    def _safe_filename(name: str) -> str:
        """Turn a glossary name into a download-safe file stem."""
        cleaned = re.sub(r'[^A-Za-z0-9._-]+', '_', name or 'glossary')
        cleaned = cleaned.strip('._') or 'glossary'
        return cleaned

    def _attachment_disposition(filename: str) -> str:
        """Build a WSGI-safe attachment header with UTF-8 filename support."""
        normalized = (filename or 'glossary').replace('\r', ' ').replace('\n', ' ').strip() or 'glossary'
        if '.' in normalized:
            stem, suffix = normalized.rsplit('.', 1)
            safe_suffix = re.sub(r'[^A-Za-z0-9]+', '', suffix)
            ascii_fallback = _safe_filename(stem)
            if safe_suffix:
                ascii_fallback = f'{ascii_fallback}.{safe_suffix}'
        else:
            ascii_fallback = _safe_filename(normalized)
        encoded = quote(normalized, safe='')
        return f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{encoded}'

    def _terms_from_payload(payload) -> List[GlossaryTerm]:
        """Build a list of GlossaryTerm from a JSON payload (list or dict)."""
        if isinstance(payload, dict):
            raw_terms = payload.get('terms') or []
        elif isinstance(payload, list):
            raw_terms = payload
        else:
            raw_terms = []

        terms: List[GlossaryTerm] = []
        for entry in raw_terms:
            if not isinstance(entry, dict):
                continue
            source = (entry.get('source') or entry.get('source_term') or '').strip()
            if not source:
                continue
            terms.append(GlossaryTerm.from_dict(entry))
        return terms

    def _terms_from_csv(text: str) -> List[GlossaryTerm]:
        """Parse a CSV (with header) into GlossaryTerm objects."""
        reader = csv.DictReader(io.StringIO(text))
        terms: List[GlossaryTerm] = []
        for row in reader:
            if not row:
                continue
            source = (row.get('source') or row.get('source_term') or '').strip()
            if not source:
                continue
            target = (row.get('target') or row.get('translated_term') or '').strip()
            category = (row.get('category') or '').strip() or None
            terms.append(GlossaryTerm(
                source_term=source,
                translated_term=target,
                category=category,
                gender=row.get('gender') or row.get('sex'),
            ))
        return terms

    # ------------------------------------------------------------ glossary CRUD

    @bp.route('/api/glossaries', methods=['GET'])
    def list_glossaries():
        """List all glossaries with their term counts (without the terms themselves)."""
        try:
            results = store.list_glossaries_with_counts()
            payload = []
            for glossary, term_count in results:
                summary = _glossary_summary(glossary)
                summary['term_count'] = term_count
                payload.append(summary)
            return jsonify({"glossaries": payload, "count": len(payload)})
        except Exception as e:
            logger.error(f"Error listing glossaries: {e}")
            return jsonify({"error": str(e)}), 500

    @bp.route('/api/glossaries', methods=['POST'])
    def create_glossary():
        """Create a new glossary, optionally seeding its terms."""
        try:
            data = request.get_json(silent=True) or {}
            name = (data.get('name') or '').strip()
            if not name:
                return jsonify({"error": "Field 'name' is required"}), 400

            source_lang = data.get('source_lang') or data.get('source_language') or ''
            target_lang = data.get('target_lang') or data.get('target_language') or ''

            try:
                glossary = store.create_glossary(
                    name=name,
                    source_language=source_lang,
                    target_language=target_lang,
                )
            except ValueError as e:
                return jsonify({"error": str(e)}), 409

            initial_terms = _terms_from_payload(data.get('terms') or [])
            if initial_terms:
                store.bulk_replace_terms(glossary.id, initial_terms)

            full = store.get_glossary(glossary.id)
            return jsonify(full.to_dict()), 201

        except Exception as e:
            logger.error(f"Error creating glossary: {e}")
            return jsonify({"error": str(e)}), 500

    @bp.route('/api/glossaries/<int:gid>', methods=['GET'])
    def get_glossary(gid: int):
        """Return a glossary with all its terms."""
        try:
            glossary = store.get_glossary(gid)
            if not glossary:
                return jsonify({"error": f"Glossary {gid} not found"}), 404
            return jsonify(glossary.to_dict())
        except Exception as e:
            logger.error(f"Error reading glossary {gid}: {e}")
            return jsonify({"error": str(e)}), 500

    @bp.route('/api/glossaries/<int:gid>', methods=['PUT'])
    def update_glossary(gid: int):
        """Patch fields on a glossary."""
        try:
            data = request.get_json(silent=True) or {}
            kwargs = {}
            if 'name' in data:
                name = (data.get('name') or '').strip()
                if not name:
                    return jsonify({"error": "Field 'name' cannot be empty"}), 400
                kwargs['name'] = name
            if 'source_lang' in data or 'source_language' in data:
                kwargs['source_language'] = data.get('source_lang') or data.get('source_language') or ''
            if 'target_lang' in data or 'target_language' in data:
                kwargs['target_language'] = data.get('target_lang') or data.get('target_language') or ''

            try:
                updated = store.update_glossary(gid, **kwargs)
            except ValueError as e:
                return jsonify({"error": str(e)}), 409

            if not updated:
                return jsonify({"error": f"Glossary {gid} not found"}), 404
            return jsonify(updated.to_dict())

        except Exception as e:
            logger.error(f"Error updating glossary {gid}: {e}")
            return jsonify({"error": str(e)}), 500

    @bp.route('/api/glossaries/<int:gid>', methods=['DELETE'])
    def delete_glossary(gid: int):
        """Delete a glossary (and its terms via cascade)."""
        try:
            deleted = store.delete_glossary(gid)
            if not deleted:
                return jsonify({"error": f"Glossary {gid} not found"}), 404
            return jsonify({"deleted": True})
        except Exception as e:
            logger.error(f"Error deleting glossary {gid}: {e}")
            return jsonify({"error": str(e)}), 500

    @bp.route('/api/glossaries/<int:gid>/duplicate', methods=['POST'])
    def duplicate_glossary(gid: int):
        """Clone a glossary and all its terms under a new unique name."""
        try:
            data = request.get_json(silent=True) or {}
            new_name = data.get('name')
            duplicated = store.duplicate_glossary(gid, new_name=new_name)
            if not duplicated:
                return jsonify({"error": f"Glossary {gid} not found"}), 404
            return jsonify(duplicated.to_dict()), 201
        except Exception as e:
            logger.error(f"Error duplicating glossary {gid}: {e}")
            return jsonify({"error": str(e)}), 500

    # ---------------------------------------------------------------- term CRUD

    @bp.route('/api/glossaries/<int:gid>/terms', methods=['POST'])
    def add_term(gid: int):
        """Add a single term to a glossary."""
        try:
            if not store.get_glossary(gid):
                return jsonify({"error": f"Glossary {gid} not found"}), 404

            data = request.get_json(silent=True) or {}
            source = (data.get('source') or data.get('source_term') or '').strip()
            target = (data.get('target') or data.get('translated_term') or '').strip()
            if not source:
                return jsonify({"error": "Field 'source' is required"}), 400

            term = GlossaryTerm(
                source_term=source,
                translated_term=target,
                category=(data.get('category') or '').strip() or None,
                gender=data.get('gender'),
            )

            try:
                created = store.add_term(gid, term)
            except ValueError as e:
                return jsonify({"error": str(e)}), 409

            return jsonify(created.to_dict()), 201

        except Exception as e:
            logger.error(f"Error adding term to glossary {gid}: {e}")
            return jsonify({"error": str(e)}), 500

    @bp.route('/api/glossaries/<int:gid>/terms/<int:tid>', methods=['PUT'])
    def update_term(gid: int, tid: int):
        """Patch fields on a term."""
        try:
            if not store.get_glossary(gid):
                return jsonify({"error": f"Glossary {gid} not found"}), 404

            data = request.get_json(silent=True) or {}
            kwargs = {}
            if 'source' in data or 'source_term' in data:
                kwargs['source_term'] = (data.get('source') or data.get('source_term') or '').strip()
            if 'target' in data or 'translated_term' in data:
                kwargs['translated_term'] = (data.get('target') or data.get('translated_term') or '').strip()
            if 'category' in data:
                kwargs['category'] = (data.get('category') or '').strip() or None
            if 'gender' in data:
                # The empty string clears the gender; store.update_term treats
                # only None as "leave unchanged", so '' must survive as ''.
                kwargs['gender'] = (data.get('gender') or '').strip()

            try:
                updated = store.update_term(tid, **kwargs)
            except ValueError as e:
                return jsonify({"error": str(e)}), 409

            if not updated:
                return jsonify({"error": f"Term {tid} not found"}), 404
            return jsonify(updated.to_dict())

        except Exception as e:
            logger.error(f"Error updating term {tid}: {e}")
            return jsonify({"error": str(e)}), 500

    @bp.route('/api/glossaries/<int:gid>/terms/bulk', methods=['POST'])
    def bulk_terms(gid: int):
        """Apply a bulk action to several terms at once.

        Body shapes:
          - ``{"action": "add", "terms": [{"source": "...", "target": "...", "category": "...", "gender": "..."}, ...]}``
          - ``{"action": "delete", "term_ids": [int, ...]}``
          - ``{"action": "set_category", "term_ids": [int, ...], "category": "..."}``
          - ``{"action": "set_gender", "term_ids": [int, ...], "gender": "male"|"female"|""}``
        """
        try:
            if not store.get_glossary(gid):
                return jsonify({"error": f"Glossary {gid} not found"}), 404

            data = request.get_json(silent=True) or {}
            action = (data.get('action') or '').strip().lower()

            if action == 'add':
                terms = _terms_from_payload(data.get('terms') or [])
                if not terms:
                    return jsonify({"error": "terms cannot be empty"}), 400
                added, conflicts, skipped_empty = store.bulk_add_terms(gid, terms)
                return jsonify({
                    "added": added,
                    "conflicts": conflicts,
                    "skipped_empty": skipped_empty,
                    "total_input": len(terms),
                })

            term_ids_raw = data.get('term_ids') or []
            try:
                term_ids = [int(x) for x in term_ids_raw if x is not None]
            except (TypeError, ValueError):
                return jsonify({"error": "term_ids must be a list of integers"}), 400
            if not term_ids:
                return jsonify({"error": "term_ids cannot be empty"}), 400

            if action == 'delete':
                deleted = store.bulk_delete_terms(gid, term_ids)
                return jsonify({"deleted": deleted})
            if action == 'set_category':
                category = data.get('category')
                if category is None:
                    return jsonify({"error": "category is required for set_category"}), 400
                updated = store.bulk_set_category(gid, term_ids, category)
                return jsonify({"updated": updated})
            if action == 'set_gender':
                gender = data.get('gender')
                if gender is None:
                    return jsonify({"error": "gender is required for set_gender"}), 400
                updated = store.bulk_set_gender(gid, term_ids, gender)
                return jsonify({"updated": updated})
            return jsonify({"error": f"Unknown action: {action}"}), 400
        except Exception as e:
            logger.error(f"Bulk action failed on glossary {gid}: {e}")
            return jsonify({"error": str(e)}), 500

    @bp.route('/api/glossaries/<int:gid>/terms/<int:tid>', methods=['DELETE'])
    def delete_term(gid: int, tid: int):
        """Delete a single term."""
        try:
            if not store.get_glossary(gid):
                return jsonify({"error": f"Glossary {gid} not found"}), 404

            deleted = store.delete_term(tid)
            if not deleted:
                return jsonify({"error": f"Term {tid} not found"}), 404
            return jsonify({"deleted": True})

        except Exception as e:
            logger.error(f"Error deleting term {tid}: {e}")
            return jsonify({"error": str(e)}), 500

    # ------------------------------------------------------------ import/export

    @bp.route('/api/glossaries/<int:gid>/import', methods=['POST'])
    def import_terms(gid: int):
        """Replace a glossary's terms from a JSON or CSV payload.

        Accepts either:
        - ``application/json`` with a body that is either a full Glossary
          dict (only the ``terms`` array is used) or a bare list of terms.
        - A multipart upload containing a ``.json`` or ``.csv`` file.
        """
        try:
            if not store.get_glossary(gid):
                return jsonify({"error": f"Glossary {gid} not found"}), 404

            terms: List[GlossaryTerm] = []
            content_type = (request.content_type or '').lower()

            if content_type.startswith('application/json'):
                payload = request.get_json(silent=True)
                if payload is None:
                    return jsonify({"error": "Invalid JSON body"}), 400
                terms = _terms_from_payload(payload)

            elif request.files:
                upload = next(iter(request.files.values()), None)
                if upload is None or not upload.filename:
                    return jsonify({"error": "No file uploaded"}), 400

                raw = upload.read()
                try:
                    text = raw.decode('utf-8-sig')
                except UnicodeDecodeError:
                    text = raw.decode('utf-8', errors='replace')

                filename = (upload.filename or '').lower()
                mimetype = (upload.mimetype or '').lower()

                if filename.endswith('.json') or 'json' in mimetype:
                    import json as _json
                    try:
                        payload = _json.loads(text) if text.strip() else {}
                    except ValueError as e:
                        return jsonify({"error": f"Invalid JSON file: {e}"}), 400
                    terms = _terms_from_payload(payload)
                elif filename.endswith('.csv') or 'csv' in mimetype or 'text' in mimetype:
                    terms = _terms_from_csv(text)
                else:
                    return jsonify({"error": "Unsupported file type (expected .json or .csv)"}), 400

            else:
                return jsonify({"error": "Expected application/json body or multipart file upload"}), 400

            result = store.bulk_replace_terms(gid, terms)
            return jsonify({
                "imported": result.inserted,
                "skipped_empty": result.skipped_empty,
                "skipped_duplicate": result.skipped_duplicate,
                "total_input": result.total_input,
            })

        except Exception as e:
            logger.error(f"Error importing terms into glossary {gid}: {e}")
            return jsonify({"error": str(e)}), 500

    @bp.route('/api/glossaries/<int:gid>/export', methods=['GET'])
    def export_glossary(gid: int):
        """Export a glossary as a JSON or CSV download."""
        try:
            glossary = store.get_glossary(gid)
            if not glossary:
                return jsonify({"error": f"Glossary {gid} not found"}), 404

            fmt = (request.args.get('format') or 'json').lower()

            if fmt == 'csv':
                buffer = io.StringIO()
                writer = csv.DictWriter(
                    buffer, fieldnames=['source', 'target', 'category', 'gender']
                )
                writer.writeheader()
                for term in glossary.terms:
                    writer.writerow({
                        'source': term.source_term,
                        'target': term.translated_term,
                        'category': term.category or '',
                        'gender': term.gender or '',
                    })
                return Response(
                    buffer.getvalue(),
                    mimetype='text/csv; charset=utf-8',
                    headers={
                        'Content-Disposition': _attachment_disposition(f'{glossary.name or "glossary"}.csv')
                    },
                )

            if fmt == 'json':
                import json as _json
                payload = {
                    'name': glossary.name,
                    'source_lang': glossary.source_language,
                    'target_lang': glossary.target_language,
                    'terms': [t.to_dict() for t in glossary.terms],
                }
                return Response(
                    _json.dumps(payload, ensure_ascii=False, indent=2),
                    mimetype='application/json; charset=utf-8',
                    headers={
                        'Content-Disposition': _attachment_disposition(f'{glossary.name or "glossary"}.json')
                    },
                )

            return jsonify({"error": f"Unsupported format: {fmt}"}), 400

        except Exception as e:
            logger.error(f"Error exporting glossary {gid}: {e}")
            return jsonify({"error": str(e)}), 500

    # ----------------------------------------------------------- preview block

    @bp.route('/api/glossaries/<int:gid>/preview-block', methods=['POST'])
    def preview_block(gid: int):
        """Render the glossary block exactly as it would be injected.

        Body: ``{"text": "<sample chunk>"}``. Returns the filtered terms
        (post word-boundary / CJK matching) and the formatted block string.

        The cast block is included when any entry carries a gender, because
        that is what actually reaches the model — it is not chunk-filtered, so
        it appears even for a sample text that matches no term at all.
        """
        try:
            glossary = store.get_glossary(gid)
            if not glossary:
                return jsonify({"error": f"Glossary {gid} not found"}), 404

            data = request.get_json(silent=True) or {}
            text = data.get('text') or ''

            terms_dict = glossary.terms_dict
            term_metadata = glossary.terms_metadata
            config = GlossaryConfig()

            filtered, capped = filter_glossary(text, terms_dict, config)
            glossary_block = build_glossary_block(
                filtered_terms=filtered,
                target_language=glossary.target_language or '',
                term_metadata=term_metadata,
            )
            cast_block, cast_capped = build_cast_block(
                terms_dict,
                term_metadata=term_metadata,
                max_entries=config.max_cast_entries,
            )
            block = "\n".join(b for b in (cast_block, glossary_block) if b)

            return jsonify({
                "block": block,
                "cast_block": cast_block,
                "glossary_block": glossary_block,
                "matched_count": len(filtered),
                "total_terms": len(terms_dict),
                "gendered_count": sum(1 for t in glossary.terms if t.gender),
                "capped": capped,
                "cast_capped": cast_capped,
            })
        except Exception as e:
            logger.error(f"Error previewing block for glossary {gid}: {e}")
            return jsonify({"error": str(e)}), 500

    # ------------------------------------------------------------ NER (Phase 2)

    @bp.route('/api/glossaries/<int:gid>/suggest-terms', methods=['POST'])
    def suggest_terms_endpoint(gid: int):
        """Run an LLM-driven NER pass on a sample of source text.

        Returns a list of candidate terms (with proposed target translations
        and a category). Nothing is persisted — the caller decides which
        candidates to add via ``POST /api/glossaries/<gid>/terms``.

        Accepts either:
          - ``application/json`` with ``text`` and the parameters below.
          - ``multipart/form-data`` with a ``file`` field (txt/srt/epub/docx)
            plus the parameters as form fields.

        Parameters:
          - ``source_lang`` (str, optional): defaults to glossary's source_lang.
          - ``target_lang`` (str, optional): defaults to glossary's target_lang.
          - ``provider`` (str, optional): defaults to LLM_PROVIDER from .env.
          - ``model`` (str, optional): defaults to provider's DEFAULT_MODEL.
          - ``api_key`` (str, optional): overrides the env-configured key.
          - ``api_endpoint`` (str, optional): overrides Ollama/OpenAI endpoint.
          - ``max_chars`` (int, optional, default 6000): total char budget sent to the LLM.
          - ``sample_count`` (int, optional, default 5): number of evenly-spaced
            excerpts drawn from the full text. ``1`` means "first ``max_chars``".
        """
        try:
            glossary = store.get_glossary(gid)
            if not glossary:
                return jsonify({"error": f"Glossary {gid} not found"}), 404

            content_type = (request.content_type or '').lower()
            is_multipart = content_type.startswith('multipart/form-data')

            if is_multipart:
                data = request.form
            else:
                data = request.get_json(silent=True) or {}

            try:
                max_chars = int(data.get('max_chars') or _NER_MAX_CHARS_HARD_CAP)
            except (TypeError, ValueError):
                return jsonify({"error": "max_chars must be an integer"}), 400
            if max_chars <= 0:
                return jsonify({"error": "max_chars must be positive"}), 400
            if max_chars > _NER_MAX_CHARS_HARD_CAP:
                max_chars = _NER_MAX_CHARS_HARD_CAP

            try:
                sample_count = int(data.get('sample_count') or 10)
            except (TypeError, ValueError):
                return jsonify({"error": "sample_count must be an integer"}), 400
            if sample_count < 1 or sample_count > 50:
                return jsonify({"error": "sample_count must be between 1 and 50"}), 400

            sample_source = None
            sample_filename: Optional[str] = None
            effective_sample_count = 1
            full_text_chars = 0
            pre_warnings: List[str] = []
            text = ''

            if is_multipart:
                upload = next(iter(request.files.values()), None)
                if upload is None or not upload.filename:
                    return jsonify({"error": "No file uploaded"}), 400

                ext = Path(upload.filename).suffix.lower()
                if ext not in _NER_SUPPORTED_EXTS:
                    return jsonify({
                        "error": (
                            f"Unsupported file type '{ext or '?'}'. "
                            "Expected one of: " + ", ".join(sorted(_NER_SUPPORTED_EXTS))
                        )
                    }), 400

                raw = upload.read(_NER_UPLOAD_MAX_BYTES + 1)
                if len(raw) > _NER_UPLOAD_MAX_BYTES:
                    return jsonify({
                        "error": f"File too large (max {_NER_UPLOAD_MAX_BYTES // (1024 * 1024)} MB)"
                    }), 413

                sampled, effective_sample_count, full_text_chars = extract_samples_from_upload(
                    raw, upload.filename, max_chars, sample_count
                )
                if not sampled:
                    return jsonify({
                        "error": "Could not extract any text from the uploaded file."
                    }), 400
                text = sampled.strip()
                sample_source = 'uploaded_file'
                sample_filename = upload.filename

                if sample_count > 1 and effective_sample_count < sample_count:
                    if effective_sample_count == 1 and full_text_chars <= max_chars:
                        pre_warnings.append(
                            f"Document is short ({full_text_chars} chars) — sent in full instead of {sample_count} excerpts."
                        )
                    else:
                        pre_warnings.append(
                            f"Reduced from {sample_count} to {effective_sample_count} excerpts "
                            f"(each excerpt needs ≥ {_NER_MIN_SAMPLE_SIZE} chars; raise 'Total chars' to fit more)."
                        )
            else:
                text = (data.get('text') or '').strip()
                if text:
                    sample_source = 'pasted_text'

            if not text:
                return jsonify({"error": "A file upload or 'text' field is required"}), 400

            source_lang = (data.get('source_lang') or data.get('source_language')
                           or glossary.source_language or 'English')
            target_lang = (data.get('target_lang') or data.get('target_language')
                           or glossary.target_language or 'English')

            existing_sources = {t.source_term for t in glossary.terms}

            from src.core.llm.factory import create_llm_provider
            import src.config as _config

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
                    context_window=_NER_CONTEXT_WINDOW,
                )
            except Exception as e:
                return jsonify({"error": f"Could not initialize provider '{provider_type}': {e}"}), 400

            async def _ner_with_cleanup():
                # Closing the provider on the same loop that opened it lets
                # httpx finalize its streaming async generators synchronously,
                # avoiding 'Task was destroyed' warnings on loop teardown.
                try:
                    return await ner_suggest_terms(
                        text=text,
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

            def _run_ner():
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(_ner_with_cleanup())
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
                candidates, warnings = _run_ner()
            else:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    candidates, warnings = pool.submit(_run_ner).result()

            for c in candidates:
                c['already_in_glossary'] = c['source'] in existing_sources

            new_count = sum(1 for c in candidates if not c['already_in_glossary'])

            if not candidates:
                logger.info(
                    "NER returned 0 candidates for glossary %s "
                    "(provider=%s, model=%s, sample_chars=%d, "
                    "effective_samples=%d, full_text_chars=%d, file=%s)",
                    gid, provider_type, model, len(text),
                    effective_sample_count, full_text_chars, sample_filename or '-',
                )

            combined_warnings = pre_warnings + list(warnings or [])

            return jsonify({
                "candidates": candidates,
                "count": len(candidates),
                "new_count": new_count,
                "warnings": combined_warnings,
                "provider": provider_type,
                "model": model,
                "sample_chars": len(text),
                "sample_source": sample_source,
                "sample_filename": sample_filename,
                "sample_count": effective_sample_count if sample_source == 'uploaded_file' else 1,
                "requested_sample_count": sample_count if sample_source == 'uploaded_file' else 1,
                "full_text_chars": full_text_chars,
            })

        except RateLimitError as e:
            logger.warning(
                f"Rate limited while suggesting terms for glossary {gid}: "
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
            logger.error(f"Error suggesting terms for glossary {gid}: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    return bp
