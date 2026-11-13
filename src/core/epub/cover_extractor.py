"""
EPUB Cover Extraction Module

This module extracts and processes cover images from EPUB files.
Generates thumbnail images (48x64px) for display in the web interface.
"""
from pathlib import Path
from typing import Optional
import zipfile
from lxml import etree
from PIL import Image
import io

from src.config import NAMESPACES


class EPUBCoverExtractor:
    """Extract and process EPUB cover images."""

    # Security settings
    MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
    ALLOWED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
    THUMBNAIL_SIZE = (48, 64)  # Width x Height
    JPEG_QUALITY = 85

    @classmethod
    def extract_cover(cls, epub_path: str, output_dir: Path) -> Optional[str]:
        """
        Extract cover image from EPUB and save as thumbnail.

        Args:
            epub_path: Path to the EPUB file
            output_dir: Directory to save the thumbnail

        Returns:
            Filename of the generated thumbnail, or None if no cover found

        Strategy:
            1. Try OPF metadata (standard method)
            2. Try common naming conventions
            3. Try first image in manifest (fallback)
        """
        try:
            with zipfile.ZipFile(epub_path, 'r') as epub_zip:
                # Find and parse content.opf
                opf_path = cls._find_opf_file(epub_zip)
                if not opf_path:
                    return None

                opf_content = epub_zip.read(opf_path)
                opf_tree = etree.fromstring(opf_content)

                # Get the base directory of content.opf for relative paths
                opf_dir = str(Path(opf_path).parent)

                # Try multiple strategies to find cover
                cover_path = (
                    cls._find_cover_from_metadata(opf_tree, epub_zip, opf_dir)
                    or cls._find_cover_by_naming(epub_zip)
                    or cls._find_first_image(opf_tree, epub_zip, opf_dir)
                )

                if not cover_path:
                    return None

                # Extract and process the image
                image_data = epub_zip.read(cover_path)

                # Security: validate image size
                if len(image_data) > cls.MAX_IMAGE_SIZE:
                    return None

                # Security: validate file extension
                ext = Path(cover_path).suffix.lower()
                if ext not in cls.ALLOWED_FORMATS:
                    return None

                # Process and save thumbnail
                return cls._create_thumbnail(
                    image_data,
                    epub_path,
                    output_dir
                )

        except Exception:
            # Graceful degradation: return None on any error
            return None

    @classmethod
    def _find_opf_file(cls, epub_zip: zipfile.ZipFile) -> Optional[str]:
        """Find the content.opf file in the EPUB."""
        # Common locations
        common_paths = [
            'content.opf',
            'OEBPS/content.opf',
            'OPS/content.opf',
        ]

        for path in common_paths:
            if path in epub_zip.namelist():
                return path

        # Search for any .opf file
        for name in epub_zip.namelist():
            if name.endswith('.opf'):
                return name

        return None

    @classmethod
    def _find_cover_from_metadata(
        cls,
        opf_tree: etree._Element,
        epub_zip: zipfile.ZipFile,
        opf_dir: str
    ) -> Optional[str]:
        """
        Find cover using OPF metadata (standard method).

        Looks for: <meta name="cover" content="cover-image-id"/>
        Then finds the manifest item with that ID.
        """
        try:
            # Find cover metadata
            meta_elements = opf_tree.xpath(
                '//opf:metadata/opf:meta[@name="cover"]',
                namespaces=NAMESPACES
            )

            if not meta_elements:
                return None

            cover_id = meta_elements[0].get('content')
            if not cover_id:
                return None

            # Find manifest item with this ID
            manifest_items = opf_tree.xpath(
                f'//opf:manifest/opf:item[@id="{cover_id}"]',
                namespaces=NAMESPACES
            )

            if not manifest_items:
                return None

            href = manifest_items[0].get('href')
            if not href:
                return None

            # Construct full path
            cover_path = str(Path(opf_dir) / href)

            # Normalize path separators for Windows
            cover_path = cover_path.replace('\\', '/')

            # Verify file exists in EPUB
            if cover_path in epub_zip.namelist():
                return cover_path

        except Exception:
            pass

        return None

    @classmethod
    def _find_cover_by_naming(cls, epub_zip: zipfile.ZipFile) -> Optional[str]:
        """
        Find cover using common naming conventions.

        Searches for files like:
        - cover.jpg, cover.png
        - images/cover.*, OEBPS/Images/cover.*
        """
        common_names = [
            'cover.jpg', 'cover.jpeg', 'cover.png', 'cover.gif',
            'Cover.jpg', 'Cover.jpeg', 'Cover.png', 'Cover.gif',
        ]

        common_dirs = ['', 'images/', 'Images/', 'OEBPS/', 'OEBPS/images/', 'OEBPS/Images/', 'OPS/', 'OPS/images/', 'OPS/Images/']

        for directory in common_dirs:
            for name in common_names:
                path = directory + name
                if path in epub_zip.namelist():
                    return path

        return None

    @classmethod
    def _find_first_image(
        cls,
        opf_tree: etree._Element,
        epub_zip: zipfile.ZipFile,
        opf_dir: str
    ) -> Optional[str]:
        """
        Fallback: Find first image in manifest.

        Returns the first item with media-type="image/*"
        """
        try:
            manifest_items = opf_tree.xpath(
                '//opf:manifest/opf:item[starts-with(@media-type, "image/")]',
                namespaces=NAMESPACES
            )

            for item in manifest_items:
                href = item.get('href')
                if not href:
                    continue

                # Construct full path
                image_path = str(Path(opf_dir) / href)

                # Normalize path separators
                image_path = image_path.replace('\\', '/')

                # Verify file exists
                if image_path in epub_zip.namelist():
                    return image_path

        except Exception:
            pass

        return None

    @classmethod
    def _create_thumbnail(
        cls,
        image_data: bytes,
        epub_path: str,
        output_dir: Path
    ) -> Optional[str]:
        """
        Create thumbnail from image data.

        Args:
            image_data: Raw image bytes
            epub_path: Original EPUB file path (to extract prefix)
            output_dir: Directory to save thumbnail

        Returns:
            Thumbnail filename or None on error
        """
        try:
            # Open and validate image using Pillow
            image = Image.open(io.BytesIO(image_data))

            # Convert to RGB if necessary (handles RGBA, P, etc.)
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')

            # Calculate thumbnail size maintaining aspect ratio
            image.thumbnail(cls.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)

            # Create new image with exact dimensions (add padding if needed)
            thumb = Image.new('RGB', cls.THUMBNAIL_SIZE, (255, 255, 255))

            # Center the thumbnail
            offset_x = (cls.THUMBNAIL_SIZE[0] - image.width) // 2
            offset_y = (cls.THUMBNAIL_SIZE[1] - image.height) // 2
            thumb.paste(image, (offset_x, offset_y))

            # Generate filename from EPUB path
            epub_filename = Path(epub_path).stem  # e.g., "abc123_book"
            thumbnail_filename = f"{epub_filename}_cover.jpg"

            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save thumbnail
            thumbnail_path = output_dir / thumbnail_filename
            thumb.save(
                thumbnail_path,
                'JPEG',
                quality=cls.JPEG_QUALITY,
                optimize=True
            )

            return thumbnail_filename

        except Exception:
            # Graceful degradation
            return None
