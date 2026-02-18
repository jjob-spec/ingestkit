"""Pre-flight security scanner for JSON files.

Rejects dangerous or oversized JSON files before any extraction begins.
Checks file extension, size, emptiness, JSON validity, and nesting depth.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from ingestkit_json.config import JSONProcessorConfig
from ingestkit_json.errors import ErrorCode, IngestError

logger = logging.getLogger("ingestkit_json")

_LARGE_FILE_THRESHOLD_MB = 10


class JSONSecurityScanner:
    """Run pre-flight security checks on a JSON file.

    Returns a list of errors/warnings.  Fatal errors (``E_*`` codes) mean
    the file should not be processed further.
    """

    def __init__(self, config: JSONProcessorConfig) -> None:
        self.config = config

    def scan(self, file_path: str) -> list[IngestError]:
        """Run all pre-flight checks.

        Returns:
            List of errors/warnings.  Fatal errors have codes starting
            with ``E_``.
        """
        errors: list[IngestError] = []

        # --- 1. Extension check ---
        if not file_path.lower().endswith(".json"):
            errors.append(
                IngestError(
                    code=ErrorCode.E_SECURITY_BAD_EXTENSION,
                    message=f"File does not have .json extension: {file_path}",
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

        # --- 6. JSON validity + nesting depth ---
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            errors.append(
                IngestError(
                    code=ErrorCode.E_SECURITY_INVALID_JSON,
                    message=f"Invalid JSON: {exc}",
                    stage="security",
                )
            )
            return errors
        except OSError as exc:
            errors.append(
                IngestError(
                    code=ErrorCode.E_PARSE_CORRUPT,
                    message=f"Cannot read file: {exc}",
                    stage="security",
                )
            )
            return errors

        depth = _measure_depth(data)
        if depth > self.config.max_nesting_depth:
            errors.append(
                IngestError(
                    code=ErrorCode.E_SECURITY_NESTING_BOMB,
                    message=(
                        f"Nesting depth {depth} exceeds limit of "
                        f"{self.config.max_nesting_depth}"
                    ),
                    stage="security",
                )
            )
            return errors

        return errors


def _measure_depth(data: Any, _current: int = 0) -> int:
    """Recursively measure the maximum nesting depth of a JSON structure."""
    if isinstance(data, dict):
        if not data:
            return _current + 1
        return max(_measure_depth(v, _current + 1) for v in data.values())
    elif isinstance(data, list):
        if not data:
            return _current + 1
        return max(_measure_depth(item, _current + 1) for item in data)
    return _current
