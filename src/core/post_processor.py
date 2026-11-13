"""
Post-processing module for cleaning translated text.
Provides extensible cleaning operations for translation output.
"""
import re
from typing import List, Dict
from abc import ABC, abstractmethod

from src.config import PLACEHOLDER_PATTERN


class PostProcessingRule(ABC):
    """Abstract base class for post-processing rules"""
    
    @abstractmethod
    def apply(self, text: str) -> str:
        """Apply the cleaning rule to the text"""
        pass
    
    @abstractmethod
    def description(self) -> str:
        """Return a description of what this rule does"""
        pass


class RemoveResidualTagPlaceholdersRule(PostProcessingRule):
    """Remove residual tag placeholders that might remain after tag restoration"""

    def apply(self, text: str) -> str:
        # Remove [id0], [id1], etc. (unified format)
        text = re.sub(PLACEHOLDER_PATTERN, '', text)

        return text

    def description(self) -> str:
        return "Remove residual tag placeholders after restoration"




def is_french_typography(target_language: str) -> bool:
    """True when the target language puts a space before high punctuation."""
    lang = (target_language or "").strip().lower()
    return (
        lang == "fr" or lang.startswith(("fr-", "fr_"))
        or "french" in lang or "fran\u00e7ais" in lang or "francais" in lang
    )


class RemoveExtraWhitespaceRule(PostProcessingRule):
    """Clean up extra whitespace created by tag removal.

    Only plain ASCII spaces (U+0020) are touched: a non-breaking space
    (U+00A0) or narrow non-breaking space (U+202F) is always deliberate
    typography. Leading indentation is kept (code, verse, <pre> blocks), and
    for French targets the space before ! ? ; : is kept, since French
    typography requires it.
    """

    def __init__(self, target_language: str = ""):
        self.target_language = target_language

    def apply(self, text: str) -> str:
        return self.apply_for(text, self.target_language)

    @staticmethod
    def apply_for(text: str, target_language: str = "") -> str:
        # Collapse runs of spaces inside a line, never the indentation at its start
        text = re.sub(r'(?<=[^ \n]) {2,}', ' ', text)

        # Remove spaces before punctuation (not before high punctuation in French)
        punctuation = '.,' if is_french_typography(target_language) else '.,!?;:'
        text = re.sub(r'(?<=[^ \n]) +([' + re.escape(punctuation) + r'])', r'\1', text)

        # Clean up line breaks
        text = re.sub(r'\n(?:[ \t]*\n){2,}', '\n\n', text)

        # Drop outer blank lines and trailing spaces; keep the first line's
        # indentation when the text spans several lines.
        text = re.sub(r'\A(?:[ \t]*\n)+', '', text).rstrip(' \t\r\n')
        if '\n' not in text:
            text = text.lstrip(' \t')
        return text

    def description(self) -> str:
        return "Remove extra whitespace and clean punctuation spacing"


class HTMLEntityCleanupRule(PostProcessingRule):
    """Clean up HTML entities that may appear in translations"""
    
    def apply(self, text: str) -> str:
        # Replace multiple &nbsp; entities with actual non-breaking spaces
        text = re.sub(r'(&nbsp;)+', lambda m: '\u00A0' * (len(m.group()) // 6), text)
        
        # Replace other common HTML entities
        replacements = {
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&#39;': "'",
            '&apos;': "'",
            '&mdash;': '—',
            '&ndash;': '–',
            '&hellip;': '…'
        }
        
        for entity, replacement in replacements.items():
            text = text.replace(entity, replacement)
        
        return text
    
    def description(self) -> str:
        return "Clean up HTML entities"


class PostProcessor:
    """Main post-processor that applies all registered rules"""
    
    def __init__(self):
        self.rules: List[PostProcessingRule] = []
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Add default cleaning rules"""
        self.add_rule(HTMLEntityCleanupRule())
        self.add_rule(RemoveExtraWhitespaceRule())
    
    def add_rule(self, rule: PostProcessingRule):
        """Add a new post-processing rule"""
        self.rules.append(rule)
    
    def remove_rule(self, rule_type: type):
        """Remove a rule by its type"""
        self.rules = [r for r in self.rules if not isinstance(r, rule_type)]
    
    def process(self, text: str, verbose: bool = False, target_language: str = "") -> str:
        """
        Apply all post-processing rules to the text

        Args:
            text: The text to clean
            verbose: If True, print which rules are being applied
            target_language: The translation's target language; it decides
                the punctuation spacing kept (e.g. French high punctuation).

        Returns:
            The cleaned text
        """
        if not text:
            return text

        result = text
        for rule in self.rules:
            if verbose:
                print(f"Applying: {rule.description()}")
            if isinstance(rule, RemoveExtraWhitespaceRule):
                result = rule.apply_for(result, target_language or rule.target_language)
            else:
                result = rule.apply(result)

        return result
    
    def get_rules(self) -> List[Dict[str, str]]:
        """Get a list of all active rules and their descriptions"""
        return [
            {
                "name": rule.__class__.__name__,
                "description": rule.description()
            }
            for rule in self.rules
        ]


# Create a default instance
default_post_processor = PostProcessor()


def clean_translated_text(text: str, verbose: bool = False, target_language: str = "") -> str:
    """
    Convenience function to clean translated text with default rules

    Args:
        text: The translated text to clean
        verbose: If True, print which rules are being applied
        target_language: The translation's target language (see
            RemoveExtraWhitespaceRule); pass it whenever it is known.

    Returns:
        The cleaned text
    """
    return default_post_processor.process(text, verbose, target_language=target_language)


def clean_residual_tag_placeholders(text: str) -> str:
    """
    Clean up any residual TAG placeholders after tag restoration
    This should be used as a final step before saving EPUB files
    
    Args:
        text: The text to clean
        
    Returns:
        The cleaned text without residual placeholders
    """
    cleaner = RemoveResidualTagPlaceholdersRule()
    return cleaner.apply(text)


# Example of how to create a custom rule:
class CustomPunctuationRule(PostProcessingRule):
    """Example custom rule for specific punctuation cleaning"""
    
    def __init__(self, replacements: Dict[str, str]):
        self.replacements = replacements
    
    def apply(self, text: str) -> str:
        for old, new in self.replacements.items():
            text = text.replace(old, new)
        return text
    
    def description(self) -> str:
        return f"Custom punctuation replacements: {list(self.replacements.keys())}"


if __name__ == "__main__":
    # Test the post-processor
    test_text = """
    This is a test text.
    It also has    extra   spaces and &nbsp;&nbsp; HTML entities.
    """
    
    print("Original text:")
    print(test_text)
    print("\nCleaned text:")
    print(clean_translated_text(test_text, verbose=True))