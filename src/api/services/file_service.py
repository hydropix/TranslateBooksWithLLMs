"""
File service for centralized file operations
"""
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict


class FileService:
    """Handles file operations for the translation API"""

    def __init__(self, output_dir: str):
        """
        Initialize file service

        Args:
            output_dir: Base directory for file operations
        """
        self.output_dir = Path(output_dir)
        self.uploads_dir = self.output_dir / 'uploads'

    def find_file(self, filename: str) -> Optional[Path]:
        """
        Find file in main directory or uploads subdirectory

        Args:
            filename: Name of the file to find

        Returns:
            Path object if found, None otherwise
        """
        # Try main directory first
        main_path = self.output_dir / filename
        if main_path.exists() and main_path.is_file():
            return main_path

        # Try uploads subdirectory
        upload_path = self.uploads_dir / filename
        if upload_path.exists() and upload_path.is_file():
            return upload_path

        return None

    def delete_file(self, filename: str) -> bool:
        """
        Delete file from main directory or uploads subdirectory

        Args:
            filename: Name of the file to delete

        Returns:
            True if file was deleted, False if not found
        """
        file_path = self.find_file(filename)
        if file_path:
            file_path.unlink()
            return True
        return False

    def list_all_files(self) -> List[Dict]:
        """
        List all files in output directory only
        Excludes:
        - Temporary files (.tmp files and tmpXXXXXX pattern files)
        - Files in uploads/ subdirectory (source files used for translation)

        Returns:
            List of file information dictionaries
        """
        files_info = []

        # Get files from main directory only (translated output files)
        # Do NOT include files from uploads/ subdirectory
        for file_path in self.output_dir.iterdir():
            if file_path.is_file() and self._is_visible_file(file_path):
                files_info.append(self._get_file_info(file_path, is_upload=False))

        # Sort by modified time (newest first)
        files_info.sort(key=lambda x: x['modified_time'], reverse=True)

        return files_info

    def _is_visible_file(self, file_path: Path) -> bool:
        """
        Check if a file should be visible in File Management
        Excludes temporary files created during upload validation and text processing

        Args:
            file_path: Path to the file

        Returns:
            True if file should be visible, False otherwise
        """
        filename = file_path.name

        # Exclude .tmp files (from upload validation)
        if filename.endswith('.tmp'):
            return False

        # Exclude the refine pass's working copy of a translated output; it
        # only lives while a refine-after pass is paused (see JobRunner).
        if filename.endswith('.pre-refine'):
            return False

        # Exclude tmpXXXXXX pattern files (from text input processing)
        # These are created by tempfile.NamedTemporaryFile()
        if filename.startswith('tmp') and len(filename) > 3:
            # Check if it follows the pattern tmpXXXXXX.txt or similar
            # tmpXXXXXX creates filenames like tmp1a2b3c4d.txt
            name_without_ext = filename.split('.')[0]
            if name_without_ext.startswith('tmp') and len(name_without_ext) > 3:
                # It's likely a temp file pattern
                return False

        return True

    def _get_file_info(self, file_path: Path, is_upload: bool = False) -> Dict:
        """
        Get file information dictionary

        Args:
            file_path: Path to the file
            is_upload: Whether file is in uploads directory

        Returns:
            Dictionary with file metadata
        """
        stat = file_path.stat()
        info = {
            "filename": file_path.name,
            "file_path": str(file_path),
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified_time": stat.st_mtime,
            "modified_date": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "file_type": file_path.suffix.lower()[1:] if file_path.suffix else "unknown"
        }
        if is_upload:
            info["is_upload"] = True
        return info

    def delete_uploaded_file(self, file_path_str: str) -> tuple[bool, str]:
        """
        Delete an uploaded file with security validation

        Args:
            file_path_str: String path to the file

        Returns:
            Tuple of (success, error_message)
        """
        from .path_validator import PathValidator

        try:
            file_path = Path(file_path_str)

            # Security check - ensure file is within uploads directory
            if not PathValidator.is_within_directory(file_path, self.uploads_dir):
                return False, "Security: File not in uploads directory"

            # Delete the file if it exists
            if file_path.exists() and file_path.is_file():
                file_path.unlink()
                return True, ""
            else:
                return False, "File not found"

        except Exception as e:
            return False, str(e)

    def get_total_size(self, files_info: List[Dict]) -> tuple[int, float]:
        """
        Calculate total size of files

        Args:
            files_info: List of file information dictionaries

        Returns:
            Tuple of (total_bytes, total_mb)
        """
        total_bytes = sum(f['size_bytes'] for f in files_info)
        total_mb = round(total_bytes / (1024 * 1024), 2)
        return total_bytes, total_mb

    def open_file(self, filename: str) -> tuple[bool, str, Optional[str]]:
        """
        Open a file with the system's default application

        Args:
            filename: Name of the file to open

        Returns:
            Tuple of (success, message, absolute_path)
        """
        import subprocess
        import platform

        from .path_validator import PathValidator

        # os.startfile / open / xdg-open run whatever handler the OS maps to
        # the extension, so only translated documents and TTS audio qualify:
        # never a script or executable that landed in the output directory.
        if not PathValidator.is_openable(filename):
            return False, "Opening this file type is not allowed", None

        file_path = self.find_file(filename)
        if not file_path:
            return False, "File not found", None

        abs_path = str(file_path.resolve())
        if not PathValidator.is_openable(abs_path):
            return False, "Opening this file type is not allowed", None
        system = platform.system()

        try:
            if system == 'Windows':
                os.startfile(abs_path)
            elif system == 'Darwin':  # macOS
                subprocess.run(['open', abs_path], check=True)
            else:  # Linux and others
                subprocess.run(['xdg-open', abs_path], check=True)

            return True, f"File opened: {filename}", abs_path

        except Exception as e:
            return False, f"Failed to open file: {str(e)}", abs_path

    def open_output_folder(self) -> tuple[bool, str, Optional[str]]:
        """
        Open the output (translations) folder in the system's file explorer.

        Returns:
            Tuple of (success, message, absolute_path)
        """
        import subprocess
        import platform

        folder = self.output_dir.resolve()
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return False, f"Failed to prepare output folder: {str(e)}", str(folder)

        abs_path = str(folder)
        system = platform.system()

        try:
            if system == 'Windows':
                subprocess.Popen(['explorer', abs_path], close_fds=True)
            elif system == 'Darwin':
                subprocess.run(['open', abs_path], check=True)
            else:
                subprocess.run(['xdg-open', abs_path], check=True)

            return True, f"Opened folder: {abs_path}", abs_path

        except Exception as e:
            return False, f"Failed to open folder: {str(e)}", abs_path

    def reveal_file(self, filename: str) -> tuple[bool, str, Optional[str]]:
        """
        Reveal a file in the system's file explorer (selecting the file when possible)

        Args:
            filename: Name of the file to reveal

        Returns:
            Tuple of (success, message, absolute_path)
        """
        import subprocess
        import platform

        file_path = self.find_file(filename)
        if not file_path:
            return False, "File not found", None

        abs_path = str(file_path.resolve())
        parent_dir = str(file_path.parent.resolve())
        system = platform.system()

        try:
            if system == 'Windows':
                # /select, and the path must be passed as two separate arguments,
                # otherwise subprocess quotes them together and Explorer falls back
                # to opening the user's default folder.
                # Explorer returns exit code 1 even on success, so we don't check it.
                subprocess.Popen(['explorer', '/select,', abs_path], close_fds=True)
            elif system == 'Darwin':  # macOS
                subprocess.run(['open', '-R', abs_path], check=True)
            else:  # Linux and others — most file managers don't support file selection
                subprocess.run(['xdg-open', parent_dir], check=True)

            return True, f"Revealed file: {filename}", abs_path

        except Exception as e:
            return False, f"Failed to reveal file: {str(e)}", abs_path
