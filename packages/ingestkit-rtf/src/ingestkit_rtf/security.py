"""Pre-flight security scanner for RTF files.

Validates extension, file size, emptiness, RTF magic bytes, and
striprtf availability before any extraction begins.
"""

from __future__ import annotations

import logging
import os

from ingestkit_rtf.config import RTFProcessorConfig
from ingestkit_rtf.errors import ErrorCode, IngestError

logger = logging.getLogger("ingestkit_rtf")

_RTF_MAGIC = b"{\\rtf"
_LARGE_FILE_THRESHOLD_MB = 10


class RTFSecurityScanner:
    """Run pre-flight security checks on an RTF file.

    Returns a list of errors/warnings.  Fatal errors (``E_*`` codes)
    mean the file should not be processed further.
    """

    def __init__(self, config: RTFProcessorConfig) -> None:
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
        if ext != ".rtf":
            errors.append(
                IngestError(
                    code=ErrorCode.E_SECURITY_BAD_EXTENSION,
                    message=f"File does not have .rtf extension: {file_path}",
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

        # --- 6. RTF magic bytes ---
        try:
            with open(file_path, "rb") as f:
                header = f.read(len(_RTF_MAGIC))
            if header != _RTF_MAGIC:
                errors.append(
                    IngestError(
                        code=ErrorCode.E_SECURITY_BAD_MAGIC,
                        message="File does not have valid RTF magic bytes ({\\rtf)",
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

        # --- 7. striprtf availability ---
        try:
            from striprtf.striprtf import rtf_to_text  # noqa: F401
        except ImportError:
            errors.append(
                IngestError(
                    code=ErrorCode.E_RTF_STRIPRTF_UNAVAILABLE,
                    message=(
                        "striprtf is required to process RTF files. "
                        "Install it with: pip install striprtf"
                    ),
                    stage="security",
                )
            )
            return errors

        return errors
