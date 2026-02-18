"""Pre-flight security scanner for .doc files.

Validates extension, file size, emptiness, OLE2 magic bytes, and
mammoth availability before any extraction begins.
"""

from __future__ import annotations

import logging
import os

from ingestkit_doc.config import DocProcessorConfig
from ingestkit_doc.errors import ErrorCode, IngestError

logger = logging.getLogger("ingestkit_doc")

_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_LARGE_FILE_THRESHOLD_MB = 10


class DocSecurityScanner:
    """Run pre-flight security checks on a .doc file.

    Returns a list of errors/warnings.  Fatal errors (``E_*`` codes)
    mean the file should not be processed further.
    """

    def __init__(self, config: DocProcessorConfig) -> None:
        self.config = config

    def scan(self, file_path: str) -> list[IngestError]:
        """Run all pre-flight checks.

        Returns
        -------
        list[IngestError]
            A list of errors/warnings.  Fatal errors have codes starting
            with ``E_``.
        """
        errors: list[IngestError] = []

        # --- 1. Extension whitelist ---
        ext = os.path.splitext(file_path)[1].lower()
        if ext != ".doc":
            errors.append(
                IngestError(
                    code=ErrorCode.E_SECURITY_BAD_EXTENSION,
                    message=f"File does not have .doc extension: {file_path}",
                    stage="security",
                )
            )
            return errors

        # --- 2. File existence and readability ---
        if not os.path.isfile(file_path):
            errors.append(
                IngestError(
                    code=ErrorCode.E_PARSE_CORRUPT,
                    message=f"File not found or not readable: {file_path}",
                    stage="security",
                )
            )
            return errors

        # --- 3. Empty file ---
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            errors.append(
                IngestError(
                    code=ErrorCode.E_PARSE_EMPTY,
                    message=f"File is empty (0 bytes): {file_path}",
                    stage="security",
                )
            )
            return errors

        # --- 4. File size limit ---
        max_bytes = self.config.max_file_size_mb * 1024 * 1024
        if file_size > max_bytes:
            errors.append(
                IngestError(
                    code=ErrorCode.E_SECURITY_TOO_LARGE,
                    message=(
                        f"File size {file_size} bytes exceeds limit of "
                        f"{max_bytes} bytes ({self.config.max_file_size_mb} MB)"
                    ),
                    stage="security",
                )
            )
            return errors

        # --- 5. Large file warning ---
        large_threshold = _LARGE_FILE_THRESHOLD_MB * 1024 * 1024
        if file_size > large_threshold:
            errors.append(
                IngestError(
                    code=ErrorCode.W_LARGE_FILE,
                    message=(
                        f"File is {file_size / (1024 * 1024):.1f} MB "
                        f"(> {_LARGE_FILE_THRESHOLD_MB} MB)"
                    ),
                    stage="security",
                    recoverable=True,
                )
            )

        # --- 6. OLE2 magic bytes ---
        try:
            with open(file_path, "rb") as f:
                header = f.read(len(_OLE2_MAGIC))
            if header != _OLE2_MAGIC:
                errors.append(
                    IngestError(
                        code=ErrorCode.E_SECURITY_BAD_MAGIC,
                        message="File does not have valid OLE2 magic bytes",
                        stage="security",
                    )
                )
                return errors
        except OSError as exc:
            errors.append(
                IngestError(
                    code=ErrorCode.E_PARSE_CORRUPT,
                    message=f"Cannot read file header: {exc}",
                    stage="security",
                )
            )
            return errors

        # --- 7. mammoth availability ---
        try:
            import mammoth  # type: ignore[import-untyped]  # noqa: F401
        except ImportError:
            errors.append(
                IngestError(
                    code=ErrorCode.E_DOC_MAMMOTH_UNAVAILABLE,
                    message=(
                        "mammoth is required to process .doc files. "
                        "Install it with: pip install mammoth"
                    ),
                    stage="security",
                )
            )
            return errors

        return errors
