"""
End-to-end test for HTML entity protection in EPUB translation.

This test verifies that HTML entities are properly protected throughout
the entire EPUB translation pipeline when --preserve-technical is enabled.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.epub.tag_preservation import TagPreserver


def test_epub_entity_protection_e2e():
    """Test HTML entity protection in a realistic EPUB chunk scenario."""

    # Simulate EPUB HTML content with escaped code examples
    body_html = """<p>Here is an example of EPUB3 structure:</p>
<pre>&lt;figure aria-describedby="fig01-desc"&gt;
    &lt;img src="images/blob.jpeg" alt="the blob"/&gt;
    &lt;figcaption&gt;
        Figure 3.7 — The blob
        &lt;details id="fig01-desc"&gt;
            &lt;summary&gt;Description&lt;/summary&gt;
            &lt;p&gt;Photo description&lt;/p&gt;
        &lt;/details&gt;
    &lt;/figcaption&gt;
&lt;/figure&gt;</pre>
<p>Another option is to include a hyperlinked text label.</p>"""

    # Step 1: Tag preservation WITH technical protection (simulates --preserve-technical)
    print("\n" + "=" * 80)
    print("STEP 1: Tag Preservation (protect_technical=True)")
    print("=" * 80)

    preserver = TagPreserver(protect_technical=True)
    text_with_placeholders, global_tag_map = preserver.preserve_tags_and_technical_content(body_html)

    print(f"\nText with placeholders:\n{text_with_placeholders}\n")
    print(f"Global tag map has {len(global_tag_map)} entries")

    # Verify HTML entities are NOT in the placeholder text
    assert "&lt;" not in text_with_placeholders, "HTML entities should NOT be in processed text!"
    assert "&gt;" not in text_with_placeholders, "HTML entities should NOT be in processed text!"

    # Verify HTML entities ARE in the tag map
    entity_placeholders = [p for p, c in global_tag_map.items() if "&lt;" in c or "&gt;" in c]
    assert len(entity_placeholders) > 0, "HTML entities should be protected in tag map!"

    print(f"✓ HTML entities protected in {len(entity_placeholders)} placeholder(s)")

    # Step 2: Chunking (simulates HTML chunking)
    print("\n" + "=" * 80)
    print("STEP 2: Chunking")
    print("=" * 80)

    # For this test, we'll skip actual chunking and just simulate one chunk
    chunk_text = text_with_placeholders
    print(f"\nChunk text:\n{chunk_text}\n")

    # Verify entities still not visible
    assert "&lt;" not in chunk_text
    assert "&gt;" not in chunk_text
    print("✓ HTML entities still protected in chunk")

    # Step 3: Simulate translation (LLM would see this text)
    print("\n" + "=" * 80)
    print("STEP 3: Translation (What LLM Sees)")
    print("=" * 80)

    print(f"\nText sent to LLM:\n{chunk_text}\n")

    # Simulate LLM translation (just replace English with French for demo)
    translated = chunk_text.replace(
        "Here is an example of EPUB3 structure:",
        "Voici un exemple de structure EPUB3:"
    ).replace(
        "Another option is to include a hyperlinked text label.",
        "Une autre option consiste à inclure une étiquette de texte avec hyperlien."
    )

    print(f"Translated text:\n{translated}\n")

    # Verify placeholders preserved in translation
    for placeholder in global_tag_map.keys():
        assert placeholder in translated, f"Placeholder {placeholder} should be in translated text!"

    print("✓ All placeholders preserved in translation")

    # Step 4: Restoration
    print("\n" + "=" * 80)
    print("STEP 4: Restoration")
    print("=" * 80)

    restored_html = preserver.restore_tags(translated, global_tag_map)

    print(f"\nRestored HTML:\n{restored_html}\n")

    # Verify HTML entities are back
    assert "&lt;figure" in restored_html
    assert "&lt;img" in restored_html
    assert "&lt;details" in restored_html
    print("✓ HTML entities perfectly restored")

    # Verify HTML tags are back
    assert "<p>" in restored_html
    assert "<pre>" in restored_html
    print("✓ HTML tags restored")

    # Verify translation was applied
    assert "Voici un exemple" in restored_html
    assert "Une autre option" in restored_html
    print("✓ Translation applied")

    print("\n" + "=" * 80)
    print("SUCCESS: HTML entities protected throughout entire pipeline!")
    print("=" * 80)


def test_epub_entity_without_protection():
    """Test that WITHOUT --preserve-technical flag, entities are exposed (expected old behavior)."""

    body_html = """<p>Example:</p>
<pre>&lt;section&gt;&lt;h1&gt;Title&lt;/h1&gt;&lt;/section&gt;</pre>"""

    # WITHOUT technical protection (simulates missing --preserve-technical flag)
    preserver = TagPreserver(protect_technical=False)
    text_with_placeholders, global_tag_map = preserver.preserve_tags(body_html)

    # HTML entities WILL be visible (this is expected without the flag)
    assert "&lt;section" in text_with_placeholders
    assert "&gt;" in text_with_placeholders

    print("\n✓ Without --preserve-technical: HTML entities are exposed (expected)")


def test_comparison():
    """Side-by-side comparison of with/without protection."""

    text = "<p>&lt;code&gt;example&lt;/code&gt; is a tag.</p>"

    print("\n" + "=" * 80)
    print("COMPARISON: With vs Without Protection")
    print("=" * 80)

    # Without protection
    preserver_off = TagPreserver(protect_technical=False)
    processed_off, _ = preserver_off.preserve_tags(text)

    print(f"\nWITHOUT protection (protect_technical=False):")
    print(f"  Input:  {text}")
    print(f"  Output: {processed_off}")
    print(f"  Entities exposed: {'&lt;' in processed_off}")

    # With protection
    preserver_on = TagPreserver(protect_technical=True)
    processed_on, map_on = preserver_on.preserve_tags_and_technical_content(text)

    print(f"\nWITH protection (protect_technical=True):")
    print(f"  Input:  {text}")
    print(f"  Output: {processed_on}")
    print(f"  Entities exposed: {'&lt;' in processed_on}")
    print(f"  Entities in map: {any('&lt;' in c for c in map_on.values())}")


if __name__ == "__main__":
    test_epub_entity_protection_e2e()
    test_epub_entity_without_protection()
    test_comparison()
