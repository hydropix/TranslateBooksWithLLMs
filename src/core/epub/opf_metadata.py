"""OPF package metadata written for a translated book.

Sets ``dc:language`` to the target language and, when attribution is on,
records the translator as a contributor and signs ``dc:description``.
"""

from lxml import etree

from src.config import ATTRIBUTION_ENABLED, GENERATOR_NAME, GENERATOR_SOURCE, NAMESPACES

from .lang_support import get_language_code


def _set_translator_role(
    opf_root: etree._Element,
    metadata: etree._Element,
    contributor_el: etree._Element,
) -> None:
    """Mark the contributor as translator in the form the package version allows.

    EPUB 2 expresses the role as an ``opf:role`` attribute on the element.
    EPUB 3 removed that attribute (EPUBCheck RSC-005) and expresses it as a
    ``meta`` refinement pointing at the contributor's id.
    """
    if opf_root.get('version', '').strip().startswith('2'):
        contributor_el.set('{http://www.idpf.org/2007/opf}role', 'trl')
        return

    used_ids = set(opf_root.xpath('//@id'))
    contributor_id = 'tbl-translator'
    suffix = 2
    while contributor_id in used_ids:
        contributor_id = f'tbl-translator-{suffix}'
        suffix += 1
    contributor_el.set('id', contributor_id)

    role_el = etree.SubElement(metadata, '{http://www.idpf.org/2007/opf}meta')
    role_el.set('refines', f'#{contributor_id}')
    role_el.set('property', 'role')
    role_el.set('scheme', 'marc:relators')
    role_el.text = 'trl'


def update_epub_metadata(
    opf_tree: etree._ElementTree,
    opf_path: str,
    target_language: str
) -> None:
    """Update EPUB metadata with target language and translation signature."""
    opf_root = opf_tree.getroot()
    metadata = opf_root.find('.//opf:metadata', namespaces=NAMESPACES)
    if metadata is not None:
        # Update language. dc:language must be an ISO 639-1 code; resolve it
        # with the same helper used for the XHTML lang attributes so OPF and
        # XHTML never contradict each other. When the target cannot be
        # resolved, leave the element unchanged rather than write a bogus code.
        lang_el = metadata.find('.//dc:language', namespaces=NAMESPACES)
        if lang_el is not None:
            lang_code = get_language_code(target_language)
            if lang_code:
                lang_el.text = lang_code

        # Add translation signature if enabled
        if ATTRIBUTION_ENABLED:
            # Add contributor (translator)
            contributor_el = etree.SubElement(
                metadata,
                '{http://purl.org/dc/elements/1.1/}contributor'
            )
            contributor_el.text = GENERATOR_NAME
            _set_translator_role(opf_root, metadata, contributor_el)

            # Add or update description with signature
            desc_el = metadata.find('.//dc:description', namespaces=NAMESPACES)
            signature_text = f"\n\nTranslated using {GENERATOR_NAME}\n{GENERATOR_SOURCE}"

            if desc_el is None:
                desc_el = etree.SubElement(
                    metadata,
                    '{http://purl.org/dc/elements/1.1/}description'
                )
                desc_el.text = signature_text.strip()
            else:
                if desc_el.text:
                    desc_el.text += signature_text
                else:
                    desc_el.text = signature_text.strip()

    opf_tree.write(opf_path, encoding='utf-8', xml_declaration=True, pretty_print=True)
