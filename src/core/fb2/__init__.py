"""
FictionBook 2 (.fb2) translation module.

The file is translated in place: its text blocks go through the placeholder
pipeline shared with EPUB and DOCX, and everything else is written back
unchanged.
"""

from .translator import translate_fb2_file

__all__ = ['translate_fb2_file']
