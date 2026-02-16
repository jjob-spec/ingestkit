"""Security controls and input validation for ingestkit-forms.

Provides file validation (size, extension, magic bytes), regex timeout
protection against ReDoS, and pattern compilation validation.
See spec section 13.
"""

from __future__ import annotations

import logging
import os
import pathlib
import re
import threading
from typing import TYPE_CHECKING

from ingestkit_forms.errors import FormErrorCode, FormIngestError

if TYPE_CHECKING:
    from ingestkit_forms.config import FormProcessorConfig

logger = logging.getLogger("ingestkit_forms")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAGIC_BYTES: dict[str, tuple[bytes, int] | None] = {
    ".pdf": (b"%PDF-", 5),
    ".xlsx": (b"PK\x03\x04", 4),  # ZIP/OOXML container
    ".jpg": (b"\xff\xd8\xff", 3),
    ".jpeg": (b"\xff\xd8\xff", 3),
    ".png": (b"\x89PNG\r\n\x1a\n", 8),
    ".tiff": None,  # special: two endianness variants
    ".tif": None,
}
_TIFF_MAGIC_LE = b"II\x2a\x00"  # little-endian
_TIFF_MAGIC_BE = b"MM\x00\x2a"  # big-endian

_ALLOWED_EXTENSIONS = frozenset(_MAGIC_BYTES.keys())
_REGEX_TIMEOUT_SECONDS = 1.0


# ---------------------------------------------------------------------------
# FormSecurityScanner
# ---------------------------------------------------------------------------


class FormSecurityScanner:
    """File-level security scanner for form documents.

    Validates file extension, size, and magic bytes before processing.
    All checks are fail-fast: the first fatal error stops further checks.
    """

    def __init__(self, config: FormProcessorConfig) -> None:
        self._config = config

    def scan(self, file_path: str) -> list[FormIngestError]:
        """Run all security checks on a file.

        Returns a list of errors (empty if all checks pass).
        Fatal errors have ``E_`` prefix codes.
        """
        errors: list[FormIngestError] = []

        # 1. Extension whitelist
        suffix = pathlib.Path(file_path).suffix.lower()
        if suffix not in _ALLOWED_EXTENSIONS:
            errors.append(
                FormIngestError(
                    code=FormErrorCode.E_FORM_UNSUPPORTED_FORMAT,
                    message=(
                        f"File extension '{suffix}' is not allowed. "
                        f"Allowed: {sorted(_ALLOWED_EXTENSIONS)}"
                    ),
                    stage="security",
                    recoverable=False,
                )
            )
            return errors

        # 2. File size check
        try:
            file_size = os.path.getsize(file_path)
        except OSError as exc:
            errors.append(
                FormIngestError(
                    code=FormErrorCode.E_FORM_FILE_CORRUPT,
                    message=f"Cannot read file: {exc}",
                    stage="security",
                    recoverable=False,
                )
            )
            return errors

        max_bytes = self._config.max_file_size_mb * 1024 * 1024
        if file_size > max_bytes:
            errors.append(
                FormIngestError(
                    code=FormErrorCode.E_FORM_FILE_TOO_LARGE,
                    message=(
                        f"File size {file_size} bytes exceeds limit of "
                        f"{self._config.max_file_size_mb} MB"
                    ),
                    stage="security",
                    recoverable=False,
                )
            )
            return errors

        # 3. Empty file check (before magic byte read)
        if file_size == 0:
            errors.append(
                FormIngestError(
                    code=FormErrorCode.E_FORM_FILE_CORRUPT,
                    message="File is empty (0 bytes)",
                    stage="security",
                    recoverable=False,
                )
            )
            return errors

        # 4. Magic byte verification
        magic_spec = _MAGIC_BYTES.get(suffix)
        try:
            with open(file_path, "rb") as fh:
                header = fh.read(8)  # max needed is 8 (PNG)
        except OSError as exc:
            errors.append(
                FormIngestError(
                    code=FormErrorCode.E_FORM_FILE_CORRUPT,
                    message=f"Cannot read file header: {exc}",
                    stage="security",
                    recoverable=False,
                )
            )
            return errors

        if magic_spec is not None:
            expected_bytes, expected_len = magic_spec
            if header[:expected_len] != expected_bytes:
                errors.append(
                    FormIngestError(
                        code=FormErrorCode.E_FORM_FILE_CORRUPT,
                        message=(
                            f"Magic byte mismatch for '{suffix}' file. "
                            f"Expected {expected_bytes!r}, got {header[:expected_len]!r}"
                        ),
                        stage="security",
                        recoverable=False,
                    )
                )
                return errors
        else:
            # TIFF: check both endianness variants
            if header[:4] != _TIFF_MAGIC_LE and header[:4] != _TIFF_MAGIC_BE:
                errors.append(
                    FormIngestError(
                        code=FormErrorCode.E_FORM_FILE_CORRUPT,
                        message=(
                            f"Magic byte mismatch for '{suffix}' file. "
                            f"Expected TIFF header, got {header[:4]!r}"
                        ),
                        stage="security",
                        recoverable=False,
                    )
                )
                return errors

        return errors


# ---------------------------------------------------------------------------
# Regex utilities
# ---------------------------------------------------------------------------


def regex_match_with_timeout(
    pattern: str,
    value: str,
    timeout: float = _REGEX_TIMEOUT_SECONDS,
    *,
    match_mode: str = "fullmatch",
) -> bool | None:
    """Match a regex pattern with timeout protection against ReDoS.

    Args:
        pattern: Regex pattern string.
        value: String to match against.
        timeout: Maximum seconds to wait for the match.
        match_mode: ``"fullmatch"`` (default) or ``"match"``.

    Returns:
        True if matches, False if no match, None if timeout or invalid pattern.
    """
    result: list[bool | None] = []

    def _do_match() -> None:
        try:
            if match_mode == "fullmatch":
                result.append(bool(re.fullmatch(pattern, value)))
            else:
                result.append(bool(re.match(pattern, value)))
        except re.error:
            result.append(None)

    thread = threading.Thread(target=_do_match, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if not result:
        # Timeout occurred
        return None
    return result[0]


def validate_regex_pattern(pattern: str) -> str | None:
    """Validate that a regex pattern compiles successfully.

    Returns:
        None if valid, or an error description string if invalid.
    """
    try:
        re.compile(pattern)
        return None
    except re.error as exc:
        return str(exc)


_SAFE_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,127}$")


def validate_table_name(name: str) -> str | None:
    """Validate that a string is a safe SQL identifier.

    Enforces: starts with letter or underscore, followed by letters/digits/underscores,
    max 128 characters total.

    Returns:
        None if valid, or an error description string if invalid.
    """
    if not name:
        return "Identifier cannot be empty"
    if len(name) > 128:
        return f"Identifier exceeds maximum length of 128 characters (got {len(name)})"
    if not _SAFE_IDENTIFIER_RE.match(name):
        return f"Identifier '{name}' does not match safe pattern ^[a-zA-Z_][a-zA-Z0-9_]{{0,127}}$"
    return None
