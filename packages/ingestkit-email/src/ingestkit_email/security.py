"""Pre-flight security scanner for email files.

Validates extension, file size, empty files, MSG magic bytes, and
extract-msg availability before any conversion begins.
"""

from __future__ import annotations

import logging
import os

from ingestkit_email.config import EmailProcessorConfig
from ingestkit_email.errors import ErrorCode, IngestError

logger = logging.getLogger("ingestkit_email")

_ALLOWED_EXTENSIONS = {".eml", ".msg"}
_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


class EmailSecurityScanner:
    """Run pre-flight security checks on an email file.

    Returns a list of errors/warnings.  Fatal errors (``E_*`` codes)
    mean the file should not be processed further.
    """

    def __init__(self, config: EmailProcessorConfig) -> None:
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
        ext = os.path.splitext(file_path)[1].lower()

        # 1. Extension whitelist
        if ext not in _ALLOWED_EXTENSIONS:
            errors.append(
                IngestError(
                    code=ErrorCode.E_EMAIL_UNSUPPORTED_FORMAT,
                    message=f"Unsupported file extension '{ext}'. Allowed: {_ALLOWED_EXTENSIONS}",
                    stage="security",
                )
            )
            return errors

        # 2. File size
        try:
            file_size = os.path.getsize(file_path)
        except OSError as exc:
            errors.append(
                IngestError(
                    code=ErrorCode.E_EMAIL_FILE_CORRUPT,
                    message=f"Cannot read file: {exc}",
                    stage="security",
                )
            )
            return errors

        max_bytes = self.config.max_file_size_mb * 1024 * 1024
        if file_size > max_bytes:
            errors.append(
                IngestError(
                    code=ErrorCode.E_EMAIL_TOO_LARGE,
                    message=f"File size {file_size} bytes exceeds limit of {max_bytes} bytes",
                    stage="security",
                )
            )
            return errors

        # 3. Empty file
        if file_size == 0:
            errors.append(
                IngestError(
                    code=ErrorCode.E_EMAIL_FILE_CORRUPT,
                    message="File is empty (0 bytes)",
                    stage="security",
                )
            )
            return errors

        # 4. MSG magic bytes
        if ext == ".msg":
            try:
                with open(file_path, "rb") as f:
                    header = f.read(len(_OLE2_MAGIC))
                if header != _OLE2_MAGIC:
                    errors.append(
                        IngestError(
                            code=ErrorCode.E_EMAIL_FILE_CORRUPT,
                            message="MSG file does not have valid OLE2 magic bytes",
                            stage="security",
                        )
                    )
                    return errors
            except OSError as exc:
                errors.append(
                    IngestError(
                        code=ErrorCode.E_EMAIL_FILE_CORRUPT,
                        message=f"Cannot read file header: {exc}",
                        stage="security",
                    )
                )
                return errors

            # 5. MSG dependency check
            try:
                import extract_msg  # type: ignore[import-untyped]  # noqa: F401
            except ImportError:
                errors.append(
                    IngestError(
                        code=ErrorCode.E_EMAIL_MSG_UNAVAILABLE,
                        message=(
                            "extract-msg is required to process .msg files. "
                            "Install it with: pip install extract-msg"
                        ),
                        stage="security",
                    )
                )
                return errors

        return errors
