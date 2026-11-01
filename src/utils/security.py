"""
Security utilities for file validation and protection
"""
import os
import re
import secrets
import mimetypes
import logging
import zipfile
from pathlib import Path
from typing import Set, Optional, Dict, Any, Iterable, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Custom exception for security-related errors"""
    pass


# === Archive path containment ===================================================
#
# Archive entry names and EPUB manifest hrefs are attacker-controlled: a crafted
# entry such as '../../evil.txt' would otherwise let an extraction escape its
# destination directory (zip slip).

_DRIVE_PREFIX_RE = re.compile(r'^[A-Za-z]:')


def is_safe_archive_member(entry_name: str) -> bool:
    """
    Return True if an archive entry name is safe to join to a destination.

    Pure and OS-independent: no filesystem access. A trailing '/' (directory
    entry) and a leading './' are allowed; '..' is only rejected when it is a
    whole path component.
    """
    if not entry_name or not entry_name.strip():
        return False

    # Control characters (including NUL) never belong in a legitimate entry name
    if any(ord(char) < 0x20 for char in entry_name):
        return False

    # The ZIP spec mandates '/' as the separator, so an entry containing '\' is
    # either malicious or already broken: treating it as a separator is the safe
    # reading on Windows.
    normalized = entry_name.replace('\\', '/')

    # POSIX absolute path or UNC path
    if normalized.startswith('/'):
        return False

    # Windows drive-relative or drive-absolute path
    if _DRIVE_PREFIX_RE.match(normalized):
        return False

    if any(component == '..' for component in normalized.split('/')):
        return False

    return True


def find_unsafe_archive_member(entry_names: Iterable[str]) -> Optional[str]:
    """Return the first unsafe entry name, or None if every name is safe."""
    for entry_name in entry_names:
        if not is_safe_archive_member(entry_name):
            return entry_name
    return None


def _is_within(base: Union[str, Path], target: Union[str, Path]) -> bool:
    """
    Return True only if `target` resolves to a location inside `base`.

    This mirrors PathValidator.is_within_directory (resolve both sides, then
    compare with Path.relative_to — never str.startswith, which would treat
    '/uploads-evil' as inside '/uploads'). It is reimplemented here on purpose:
    src/utils must not import from src/api.
    """
    try:
        base_resolved = Path(base).resolve()
        target_resolved = Path(target).resolve()
    except OSError:
        return False
    try:
        target_resolved.relative_to(base_resolved)
        return True
    except ValueError:
        return False


def safe_extract_zip(zip_ref: zipfile.ZipFile, destination: Union[str, Path]) -> None:
    """
    Extract a ZIP archive into `destination`, refusing any escaping entry.

    Every entry is validated before a single byte is written, so a rejected
    archive leaves the destination untouched.

    Raises:
        SecurityError: if any entry name would escape `destination`.
    """
    entry_names = zip_ref.namelist()

    bad = find_unsafe_archive_member(entry_names)
    if bad is not None:
        raise SecurityError(f"Unsafe archive entry path: {bad!r}")

    # Belt and braces: also verify the resolved join stays inside the destination
    for entry_name in entry_names:
        if not _is_within(destination, Path(destination) / entry_name):
            raise SecurityError(f"Unsafe archive entry path: {entry_name!r}")

    zip_ref.extractall(destination)


def resolve_within(base: Union[str, Path], relative: str) -> Path:
    """
    Resolve `relative` against `base`, refusing anything that escapes `base`.

    Raises:
        SecurityError: if `relative` is an unsafe archive member or resolves
            outside `base`.
    """
    if not is_safe_archive_member(relative):
        raise SecurityError(f"Unsafe relative path: {relative!r}")

    resolved = (Path(base) / relative).resolve()
    if not _is_within(base, resolved):
        raise SecurityError(f"Unsafe relative path: {relative!r}")

    return resolved


@dataclass
class FileValidationResult:
    """Result of file validation"""
    is_valid: bool
    file_path: Optional[Path] = None
    error_message: Optional[str] = None
    warnings: list = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class SecureFileHandler:
    """Secure file upload and validation handler"""

    # Allowed file extensions - includes common text extensions
    # The system will detect the actual content type for unknown extensions
    ALLOWED_EXTENSIONS: Set[str] = {
        # Primary supported formats (with dedicated processors)
        '.txt', '.epub', '.srt', '.pgn', '.docx', '.pdf',
        # Common text file extensions (will be processed as plain text)
        '.text', '.log', '.md', '.markdown', '.rst', '.asc',
        # Configuration/data files (text-based, can be translated)
        '.csv', '.json', '.xml', '.html', '.htm', '.yaml', '.yml',
        # Allow ANY extension for content-based detection
        # Users can upload .xyz files and they'll be analyzed
    }

    # Set to allow any extension (content will be validated)
    ALLOW_ANY_EXTENSION: bool = True

    # Blocked extensions (security risk - never allow these)
    BLOCKED_EXTENSIONS: Set[str] = {
        '.exe', '.bat', '.cmd', '.scr', '.com', '.pif', '.jar',
        '.vbs', '.vbe', '.js', '.jse', '.ws', '.wsf', '.wsc', '.wsh',
        '.ps1', '.psm1', '.psd1', '.msi', '.msp', '.mst',
        '.dll', '.sys', '.drv', '.ocx', '.cpl', '.inf',
        '.sh', '.bash', '.zsh', '.ksh', '.csh',
        '.app', '.dmg', '.pkg', '.deb', '.rpm',
        '.iso', '.img', '.vmdk', '.vhd', '.vhdx',
    }

    # Allowed MIME types
    ALLOWED_MIME_TYPES: Set[str] = {
        'text/plain',
        'application/epub+zip',
        'application/zip',  # Some EPUB files are detected as zip
        'application/x-subrip',  # SRT files
        'text/srt',  # Alternative MIME type for SRT
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',  # DOCX files
        'application/pdf',  # PDF files
        # Additional text MIME types
        'text/markdown',
        'text/x-markdown',
        'text/csv',
        'text/html',
        'text/xml',
        'application/json',
        'application/xml',
        'text/x-log',
        # Catch-all for generic text
        'application/octet-stream',  # Will be validated by content
    }
    
    # Maximum file size (100MB)
    MAX_FILE_SIZE: int = 100 * 1024 * 1024
    
    # Suspicious patterns to scan for in text files
    SUSPICIOUS_PATTERNS: Set[str] = {
        '<script',
        'javascript:',
        'data:',
        'vbscript:',
        'onload=',
        'onerror=',
        'eval(',
        'document.cookie',
        'window.location',
        '<?php',
        '<%',
        'exec(',
        'system(',
        'shell_exec(',
    }
    
    def __init__(self, upload_dir: Path):
        """
        Initialize secure file handler
        
        Args:
            upload_dir: Directory where uploaded files will be stored
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_and_save_file(self, file_data: bytes, original_filename: str) -> FileValidationResult:
        """
        Validate and securely save an uploaded file
        
        Args:
            file_data: Raw file data
            original_filename: Original filename from upload
            
        Returns:
            FileValidationResult with validation status and secure file path
        """
        try:
            # Step 1: Validate filename and extension
            validation_result = self._validate_filename(original_filename)
            if not validation_result.is_valid:
                return validation_result
            
            # Step 2: Check file size
            if len(file_data) > self.MAX_FILE_SIZE:
                return FileValidationResult(
                    is_valid=False,
                    error_message=f"File too large: {len(file_data)/1024/1024:.1f}MB. Maximum allowed: {self.MAX_FILE_SIZE/1024/1024:.0f}MB"
                )
            
            # Step 3: Create secure filename and path
            secure_filename = self._create_secure_filename(original_filename)
            secure_path = self._get_secure_path(secure_filename)
            
            # Step 4: Save file temporarily for validation
            temp_path = secure_path.with_suffix(secure_path.suffix + '.tmp')
            with open(temp_path, 'wb') as f:
                f.write(file_data)
            
            try:
                # Step 5: Validate file content
                content_validation = self._validate_file_content(temp_path, original_filename)
                if not content_validation.is_valid:
                    self._cleanup_temp_file(temp_path)
                    return content_validation

                # Step 6: Move temp file to final location
                temp_path.rename(secure_path)

                return FileValidationResult(
                    is_valid=True,
                    file_path=secure_path,
                    warnings=content_validation.warnings
                )

            except Exception as e:
                # Clean up temp file on error
                self._cleanup_temp_file(temp_path)
                raise e
                
        except Exception as e:
            return FileValidationResult(
                is_valid=False,
                error_message=f"Validation failed: {str(e)}"
            )
    
    def _validate_filename(self, filename: str) -> FileValidationResult:
        """Validate filename format and extension"""
        if not filename or not filename.strip():
            return FileValidationResult(is_valid=False, error_message="Filename cannot be empty")

        # Remove any path components (security)
        clean_filename = os.path.basename(filename.strip())

        if not clean_filename:
            return FileValidationResult(is_valid=False, error_message="Invalid filename")

        # Check file extension
        file_ext = Path(clean_filename).suffix.lower()

        # Handle case where path traversal removes extension
        if not file_ext and clean_filename:
            # Try to find extension in original filename
            original_ext = Path(filename).suffix.lower()
            if original_ext:
                file_ext = original_ext

        # Always block dangerous extensions (security)
        if file_ext in self.BLOCKED_EXTENSIONS:
            return FileValidationResult(
                is_valid=False,
                error_message=f"File type '{file_ext}' is not allowed for security reasons."
            )

        # If not allowing any extension, check against allowed list
        if not self.ALLOW_ANY_EXTENSION and file_ext not in self.ALLOWED_EXTENSIONS:
            return FileValidationResult(
                is_valid=False,
                error_message=f"File type '{file_ext}' not allowed. Allowed types: {', '.join(self.ALLOWED_EXTENSIONS)}"
            )

        # Check for suspicious characters
        if re.search(r'[<>:"|?*\x00-\x1f]', clean_filename):
            return FileValidationResult(is_valid=False, error_message="Filename contains invalid characters")

        # Check filename length
        if len(clean_filename) > 255:
            return FileValidationResult(is_valid=False, error_message="Filename too long")

        return FileValidationResult(is_valid=True)
    
    def _create_secure_filename(self, original_filename: str) -> str:
        """Create a secure filename preventing path traversal and conflicts"""
        # Get clean filename
        clean_name = os.path.basename(original_filename.strip())
        
        # Generate random prefix to prevent conflicts and add security
        random_prefix = secrets.token_hex(8)
        
        # Sanitize filename - keep only safe characters
        safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_"
        sanitized = ''.join(c if c in safe_chars else '_' for c in clean_name)
        
        # Ensure it's not too long
        if len(sanitized) > 100:
            name_part = sanitized[:80]
            ext_part = Path(sanitized).suffix[-20:] if Path(sanitized).suffix else ''
            sanitized = name_part + ext_part
        
        return f"{random_prefix}_{sanitized}"
    
    def _get_secure_path(self, filename: str) -> Path:
        """Get secure file path within upload directory"""
        file_path = self.upload_dir / filename
        
        # Resolve path and ensure it's within upload directory
        resolved_path = file_path.resolve()
        upload_dir_resolved = self.upload_dir.resolve()
        
        if not str(resolved_path).startswith(str(upload_dir_resolved)):
            raise SecurityError("Path traversal attempt detected")
        
        return resolved_path
    
    def _validate_file_content(self, file_path: Path, original_filename: str) -> FileValidationResult:
        """Validate file content based on type"""
        warnings = []

        # Determine expected file type from extension
        file_ext = Path(original_filename).suffix.lower()

        try:
            # For unknown extensions, detect content type first
            from src.utils.file_detector import detect_file_type_by_content

            # Check MIME type (but don't reject unknown types)
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type and mime_type not in self.ALLOWED_MIME_TYPES:
                # Don't reject immediately - content-based detection may work
                warnings.append(f"Unexpected MIME type: {mime_type}")

            # Map known extensions to validators
            extension_validators = {
                '.txt': self._validate_text_file,
                '.text': self._validate_text_file,
                '.log': self._validate_text_file,
                '.md': self._validate_text_file,
                '.markdown': self._validate_text_file,
                '.rst': self._validate_text_file,
                '.csv': self._validate_text_file,
                '.json': self._validate_text_file,
                '.xml': self._validate_text_file,
                '.html': self._validate_text_file,
                '.htm': self._validate_text_file,
                '.yaml': self._validate_text_file,
                '.yml': self._validate_text_file,
                '.epub': self._validate_epub_file,
                '.srt': self._validate_srt_file,
                '.pgn': self._validate_text_file,
                '.docx': self._validate_docx_file,
                '.pdf': self._validate_pdf_file,
            }

            # Check if we have a dedicated validator for this extension
            if file_ext in extension_validators:
                result = extension_validators[file_ext](file_path)
                if result.warnings:
                    result.warnings.extend(warnings)
                elif warnings:
                    result.warnings = warnings
                return result

            # Unknown extension: detect content type
            detected_type = detect_file_type_by_content(str(file_path))

            if detected_type == 'epub':
                return self._validate_epub_file(file_path)
            elif detected_type == 'docx':
                return self._validate_docx_file(file_path)
            elif detected_type == 'pdf':
                return self._validate_pdf_file(file_path)
            elif detected_type == 'srt':
                return self._validate_srt_file(file_path)
            elif detected_type == 'txt':
                # Validated as readable text
                result = self._validate_text_file(file_path)
                if result.is_valid:
                    result.warnings = result.warnings or []
                    result.warnings.append(f"File with extension '{file_ext}' detected as plain text and will be processed as such.")
                return result
            else:
                # Content type detection failed
                return FileValidationResult(
                    is_valid=False,
                    error_message=f"Cannot determine file type for extension '{file_ext}'. "
                                  f"The file does not appear to be a supported text or document format."
                )

        except Exception as e:
            return FileValidationResult(
                is_valid=False,
                error_message=f"Content validation failed: {str(e)}"
            )
    
    def _validate_text_file(self, file_path: Path) -> FileValidationResult:
        """Validate text file content"""
        warnings = []
        
        try:
            # Read first few KB to scan for suspicious content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                sample_content = f.read(8192)  # Read first 8KB
            
            # Check for suspicious patterns
            content_lower = sample_content.lower()
            found_patterns = []
            
            for pattern in self.SUSPICIOUS_PATTERNS:
                if pattern in content_lower:
                    found_patterns.append(pattern)
            
            if found_patterns:
                return FileValidationResult(
                    is_valid=False,
                    error_message=f"Suspicious content detected: {', '.join(found_patterns[:3])}"
                )
            
            # Check for excessive special characters (potential obfuscation)
            if len(sample_content) > 0:
                special_char_ratio = sum(1 for c in sample_content if not c.isalnum() and not c.isspace()) / len(sample_content)
                if special_char_ratio > 0.3:
                    warnings.append("High ratio of special characters detected")
            else:
                # Empty file case - reject empty files
                return FileValidationResult(
                    is_valid=False,
                    error_message="Empty file not allowed"
                )
            
            # Check encoding validity
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    f.read()
            except UnicodeDecodeError:
                warnings.append("File encoding may not be UTF-8")
            
            return FileValidationResult(is_valid=True, warnings=warnings)
            
        except Exception as e:
            return FileValidationResult(
                is_valid=False,
                error_message=f"Text file validation failed: {str(e)}"
            )
    
    def _validate_epub_file(self, file_path: Path) -> FileValidationResult:
        """Validate EPUB file structure"""
        warnings = []
        
        try:
            import zipfile
            
            # Check if it's a valid ZIP file
            if not zipfile.is_zipfile(file_path):
                return FileValidationResult(
                    is_valid=False,
                    error_message="EPUB file is not a valid ZIP archive"
                )
            
            # Basic EPUB structure validation
            with zipfile.ZipFile(file_path, 'r') as epub_zip:
                file_list = epub_zip.namelist()

                # Reject archives whose entries would escape the extraction dir
                bad = find_unsafe_archive_member(file_list)
                if bad is not None:
                    return FileValidationResult(
                        is_valid=False,
                        error_message=f"EPUB contains an unsafe entry path: {bad}"
                    )

                # Check for required EPUB files
                if 'mimetype' not in file_list:
                    warnings.append("Missing mimetype file")
                
                # Check for META-INF directory
                has_meta_inf = any(f.startswith('META-INF/') for f in file_list)
                if not has_meta_inf:
                    warnings.append("Missing META-INF directory")
                
                # Check for potential zip bombs (too many files)
                if len(file_list) > 10000:
                    return FileValidationResult(
                        is_valid=False,
                        error_message="EPUB contains too many files (potential zip bomb)"
                    )
                
                # Check for suspicious file extensions in EPUB
                suspicious_exts = {'.exe', '.bat', '.cmd', '.scr', '.com', '.pif', '.jar'}
                for file_name in file_list:
                    file_ext = Path(file_name).suffix.lower()
                    if file_ext in suspicious_exts:
                        return FileValidationResult(
                            is_valid=False,
                            error_message=f"EPUB contains suspicious file: {file_name}"
                        )
            
            return FileValidationResult(is_valid=True, warnings=warnings)
            
        except Exception as e:
            return FileValidationResult(
                is_valid=False,
                error_message=f"EPUB validation failed: {str(e)}"
            )
    
    def _validate_srt_file(self, file_path: Path) -> FileValidationResult:
        """Validate SRT subtitle file content"""
        warnings = []
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Check if file is empty
            if not content.strip():
                return FileValidationResult(
                    is_valid=False,
                    error_message="Empty SRT file not allowed"
                )
            
            # Check for basic SRT structure (number, timecode, text)
            # Look for at least one subtitle pattern
            import re
            srt_pattern = re.compile(
                r'\d+\s*\n'  # Subtitle number
                r'\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}',  # Timecode
                re.MULTILINE
            )
            
            if not srt_pattern.search(content):
                return FileValidationResult(
                    is_valid=False,
                    error_message="Invalid SRT format: no valid subtitle patterns found"
                )
            
            # Check for suspicious patterns (same as text files)
            content_lower = content.lower()
            found_patterns = []
            
            for pattern in self.SUSPICIOUS_PATTERNS:
                if pattern in content_lower:
                    found_patterns.append(pattern)
            
            if found_patterns:
                return FileValidationResult(
                    is_valid=False,
                    error_message=f"Suspicious content detected in SRT: {', '.join(found_patterns[:3])}"
                )
            
            # Check encoding validity
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    f.read()
            except UnicodeDecodeError:
                warnings.append("SRT file encoding may not be UTF-8")
            
            # Count subtitles
            subtitle_count = len(re.findall(r'^\d+\s*$', content, re.MULTILINE))
            if subtitle_count == 0:
                warnings.append("No subtitle entries detected")
            elif subtitle_count > 10000:
                return FileValidationResult(
                    is_valid=False,
                    error_message="SRT file contains too many subtitles (>10000)"
                )
            
            return FileValidationResult(is_valid=True, warnings=warnings)

        except Exception as e:
            return FileValidationResult(
                is_valid=False,
                error_message=f"SRT file validation failed: {str(e)}"
            )

    def _validate_docx_file(self, file_path: Path) -> FileValidationResult:
        """Validate DOCX file structure"""
        warnings = []

        try:
            import zipfile

            # Check if it's a valid ZIP file (DOCX is a ZIP archive)
            if not zipfile.is_zipfile(file_path):
                return FileValidationResult(
                    is_valid=False,
                    error_message="DOCX file is not a valid ZIP archive"
                )

            # Basic DOCX structure validation
            with zipfile.ZipFile(file_path, 'r') as docx_zip:
                file_list = docx_zip.namelist()

                # Reject archives whose entries would escape the extraction dir
                bad = find_unsafe_archive_member(file_list)
                if bad is not None:
                    return FileValidationResult(
                        is_valid=False,
                        error_message=f"DOCX contains an unsafe entry path: {bad}"
                    )

                # Check for required DOCX files
                has_content_types = '[Content_Types].xml' in file_list
                has_rels = any(f.startswith('_rels/') for f in file_list)
                has_word = any(f.startswith('word/') for f in file_list)

                if not has_content_types:
                    warnings.append("Missing [Content_Types].xml file")
                if not has_rels:
                    warnings.append("Missing _rels directory")
                if not has_word:
                    return FileValidationResult(
                        is_valid=False,
                        error_message="Invalid DOCX: missing word/ directory"
                    )

                # Check for potential zip bombs (too many files)
                if len(file_list) > 10000:
                    return FileValidationResult(
                        is_valid=False,
                        error_message="DOCX contains too many files (potential zip bomb)"
                    )

                # Check for suspicious file extensions in DOCX
                suspicious_exts = {'.exe', '.bat', '.cmd', '.scr', '.com', '.pif', '.jar'}
                for file_name in file_list:
                    file_ext = Path(file_name).suffix.lower()
                    if file_ext in suspicious_exts:
                        return FileValidationResult(
                            is_valid=False,
                            error_message=f"DOCX contains suspicious file: {file_name}"
                        )

            return FileValidationResult(is_valid=True, warnings=warnings)

        except Exception as e:
            return FileValidationResult(
                is_valid=False,
                error_message=f"DOCX validation failed: {str(e)}"
            )

    def _validate_pdf_file(self, file_path: Path) -> FileValidationResult:
        """
        Validate PDF file structure.

        The file is untrusted and parsed by a native library (PyMuPDF), so every
        PyMuPDF call is contained here: no exception escapes, and the document
        is always closed before returning (the caller renames or deletes the
        file afterwards, which fails on Windows while a handle is still open).
        """
        warnings = []
        doc = None

        try:
            # Check the PDF header (the spec tolerates junk bytes before it)
            with open(file_path, 'rb') as f:
                header = f.read(1024)
            if b'%PDF-' not in header:
                return FileValidationResult(
                    is_valid=False,
                    error_message="File is not a valid PDF (missing %PDF header)"
                )

            try:
                import pymupdf
            except ImportError:
                return FileValidationResult(
                    is_valid=False,
                    error_message="PDF support is not installed (missing pymupdf)"
                )

            # Force the PDF parser: the file on disk carries a temporary suffix,
            # and no other MuPDF document handler must ever see it.
            try:
                doc = pymupdf.open(str(file_path), filetype="pdf")
            except Exception as e:
                return FileValidationResult(
                    is_valid=False,
                    error_message=f"PDF validation failed: {str(e)}"
                )

            if doc.needs_pass:
                return FileValidationResult(
                    is_valid=False,
                    error_message="Password-protected PDFs are not supported. Remove the password and try again."
                )

            page_count = doc.page_count
            if page_count == 0:
                return FileValidationResult(
                    is_valid=False,
                    error_message="The PDF has no pages."
                )
            if page_count > 5000:
                return FileValidationResult(
                    is_valid=False,
                    error_message="PDF has too many pages (max 5000)."
                )

            # Probe the text layer on the first pages: a scanned document is
            # accepted, but the user is warned that translation will fail.
            pages_probed = min(10, page_count)
            text_chars = sum(
                len(doc[page_index].get_text().strip())
                for page_index in range(pages_probed)
            )
            if text_chars < 20 * pages_probed:
                warnings.append(
                    "This PDF seems to contain no text layer (scanned document?). "
                    "Translation will fail: OCR is not supported."
                )

            return FileValidationResult(is_valid=True, warnings=warnings)

        except Exception as e:
            return FileValidationResult(
                is_valid=False,
                error_message=f"PDF validation failed: {str(e)}"
            )
        finally:
            if doc is not None:
                try:
                    doc.close()
                except Exception as e:
                    logger.warning(f"Failed to close PDF during validation: {str(e)}")

    def _cleanup_temp_file(self, temp_path: Path) -> None:
        """
        Safely cleanup temporary file with error handling

        Args:
            temp_path: Path to the temporary file to remove
        """
        if not temp_path.exists():
            return

        try:
            temp_path.unlink()
            logger.debug(f"Successfully cleaned up temporary file: {temp_path.name}")
        except PermissionError:
            logger.warning(f"Permission denied when trying to delete temporary file: {temp_path.name}")
        except OSError as e:
            logger.warning(f"OS error when deleting temporary file {temp_path.name}: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error when cleaning up temporary file {temp_path.name}: {str(e)}")

    def cleanup_old_files(self, max_age_hours: int = 24):
        """Clean up old uploaded files"""
        import time
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for file_path in self.upload_dir.iterdir():
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                        print(f"Cleaned up old file: {file_path.name}")
                    except Exception as e:
                        print(f"Failed to cleanup {file_path.name}: {e}")


class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self):
        self._requests = {}  # IP -> list of timestamps
        self._max_requests = 10  # requests per window
        self._window_seconds = 60  # 1 minute window
    
    def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed for this IP"""
        import time
        
        current_time = time.time()
        window_start = current_time - self._window_seconds
        
        # Clean old requests
        if client_ip in self._requests:
            self._requests[client_ip] = [
                timestamp for timestamp in self._requests[client_ip]
                if timestamp > window_start
            ]
        else:
            self._requests[client_ip] = []
        
        # Check if under limit
        if len(self._requests[client_ip]) >= self._max_requests:
            return False
        
        # Add current request
        self._requests[client_ip].append(current_time)
        return True
    
    def get_remaining_requests(self, client_ip: str) -> int:
        """Get remaining requests for this IP"""
        if client_ip not in self._requests:
            return self._max_requests
        return max(0, self._max_requests - len(self._requests[client_ip]))


# Global instances
rate_limiter = RateLimiter()


def get_client_ip(request) -> str:
    """Get client IP address from Flask request"""
    # Check for X-Forwarded-For header (proxy/load balancer)
    if 'X-Forwarded-For' in request.headers:
        return request.headers['X-Forwarded-For'].split(',')[0].strip()
    
    # Check for X-Real-IP header (nginx)
    if 'X-Real-IP' in request.headers:
        return request.headers['X-Real-IP']
    
    # Fallback to remote address
    return request.remote_addr or '127.0.0.1'