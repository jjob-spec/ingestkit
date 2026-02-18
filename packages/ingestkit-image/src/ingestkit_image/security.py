"""Pre-flight security scanner for image files.

Validates file extension, magic bytes, file size, and image dimensions
before any processing begins.  Returns ``ImageMetadata`` and a list of
errors/warnings for the router to act on.
"""

from __future__ import annotations

import hashlib
import logging
import os

from PIL import Image

from ingestkit_image.config import ImageProcessorConfig
from ingestkit_image.errors import ImageErrorCode, ImageIngestError
from ingestkit_image.models import ImageMetadata, ImageType

logger = logging.getLogger("ingestkit_image")

# ---------------------------------------------------------------------------
# Magic byte signatures for supported image formats
# ---------------------------------------------------------------------------

_MAGIC_BYTES: dict[str, list[bytes]] = {
    "jpeg": [b"\xff\xd8\xff"],
    "png": [b"\x89PNG\r\n\x1a\n"],
    "tiff": [b"II\x2a\x00", b"MM\x00\x2a"],
    "webp": [b"RIFF"],  # Also requires "WEBP" at offset 8
    "bmp": [b"BM"],
    "gif": [b"GIF87a", b"GIF89a"],
}

_EXTENSION_MAP: dict[str, str] = {
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".png": "png",
    ".tiff": "tiff",
    ".tif": "tiff",
    ".webp": "webp",
    ".bmp": "bmp",
    ".gif": "gif",
}


class ImageSecurityScanner:
    """Run pre-flight security checks on an image file.

    Returns image metadata and a list of errors/warnings. Fatal errors
    (``E_*`` codes) mean the file should not be processed further.
    """

    def __init__(self, config: ImageProcessorConfig) -> None:
        self.config = config

    def scan(
        self, file_path: str
    ) -> tuple[ImageMetadata | None, list[ImageIngestError]]:
        """Run all pre-flight checks.

        Returns:
            A tuple of (ImageMetadata or None, list of errors/warnings).
            Fatal errors have codes starting with ``E_``.
        """
        errors: list[ImageIngestError] = []

        # --- 1. Extension whitelist ---
        ext = os.path.splitext(file_path)[1].lower()
        image_format = _EXTENSION_MAP.get(ext)

        if image_format is None or image_format not in self.config.supported_formats:
            errors.append(
                ImageIngestError(
                    code=ImageErrorCode.E_IMAGE_UNSUPPORTED_FORMAT.value,
                    message=f"Unsupported image extension '{ext}': {file_path}",
                    stage="security",
                    file_path=file_path,
                )
            )
            return None, errors

        # --- 2. File size ---
        try:
            file_size = os.path.getsize(file_path)
        except OSError as exc:
            errors.append(
                ImageIngestError(
                    code=ImageErrorCode.E_IMAGE_CORRUPT.value,
                    message=f"Cannot stat file: {exc}",
                    stage="security",
                    file_path=file_path,
                )
            )
            return None, errors

        # --- 3. Empty file check ---
        if file_size == 0:
            errors.append(
                ImageIngestError(
                    code=ImageErrorCode.E_IMAGE_EMPTY.value,
                    message=f"File is empty (0 bytes): {file_path}",
                    stage="security",
                    file_path=file_path,
                )
            )
            return None, errors

        max_bytes = self.config.max_file_size_mb * 1024 * 1024
        if file_size > max_bytes:
            errors.append(
                ImageIngestError(
                    code=ImageErrorCode.E_IMAGE_TOO_LARGE.value,
                    message=(
                        f"File size {file_size} bytes exceeds limit "
                        f"of {max_bytes} bytes"
                    ),
                    stage="security",
                    file_path=file_path,
                )
            )
            return None, errors

        # --- 4. Magic bytes ---
        if not self._check_magic_bytes(file_path, image_format):
            errors.append(
                ImageIngestError(
                    code=ImageErrorCode.E_IMAGE_CORRUPT.value,
                    message=(
                        f"Magic bytes do not match expected format "
                        f"'{image_format}': {file_path}"
                    ),
                    stage="security",
                    file_path=file_path,
                )
            )
            return None, errors

        # --- 5. Dimension check via Pillow ---
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                color_mode = img.mode
                has_exif = hasattr(img, "_getexif") and img._getexif() is not None
        except Exception as exc:
            errors.append(
                ImageIngestError(
                    code=ImageErrorCode.E_IMAGE_CORRUPT.value,
                    message=f"Pillow cannot open image: {exc}",
                    stage="security",
                    file_path=file_path,
                )
            )
            return None, errors

        if width > self.config.max_image_width or height > self.config.max_image_height:
            errors.append(
                ImageIngestError(
                    code=ImageErrorCode.E_IMAGE_DIMENSIONS_EXCEEDED.value,
                    message=(
                        f"Image dimensions {width}x{height} exceed limits "
                        f"{self.config.max_image_width}x{self.config.max_image_height}"
                    ),
                    stage="security",
                    file_path=file_path,
                )
            )
            return None, errors

        # --- 6. Compute content hash ---
        content_hash = hashlib.sha256(
            open(file_path, "rb").read()
        ).hexdigest()

        # --- 7. Build metadata ---
        image_type = ImageType(image_format)
        metadata = ImageMetadata(
            file_path=file_path,
            file_size_bytes=file_size,
            image_type=image_type,
            width=width,
            height=height,
            content_hash=content_hash,
            has_exif=has_exif,
            color_mode=color_mode,
        )

        logger.debug(
            "ingestkit_image | security_scan | file=%s | format=%s | "
            "size=%d | dimensions=%dx%d",
            os.path.basename(file_path),
            image_format,
            file_size,
            width,
            height,
        )

        return metadata, errors

    @staticmethod
    def _check_magic_bytes(file_path: str, image_format: str) -> bool:
        """Check if the file starts with expected magic bytes for the format."""
        signatures = _MAGIC_BYTES.get(image_format, [])
        if not signatures:
            return True  # No known signature to check

        try:
            with open(file_path, "rb") as f:
                header = f.read(12)  # Read enough for WEBP check
        except OSError:
            return False

        for sig in signatures:
            if header[: len(sig)] == sig:
                # Additional WEBP validation: "WEBP" at offset 8
                if image_format == "webp":
                    return header[8:12] == b"WEBP"
                return True

        return False
