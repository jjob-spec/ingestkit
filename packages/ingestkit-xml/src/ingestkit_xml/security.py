"""Pre-flight security scanner for XML files.

Rejects dangerous or oversized XML files before any extraction begins.
Checks file extension, size, emptiness, entity declarations (billion laughs /
XXE prevention), XML validity, and nesting depth.
"""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET

from ingestkit_xml.config import XMLProcessorConfig
from ingestkit_xml.errors import ErrorCode, IngestError

logger = logging.getLogger("ingestkit_xml")

_LARGE_FILE_THRESHOLD_MB = 10


class XMLSecurityScanner:
    """Run pre-flight security checks on an XML file.

    Returns a list of errors/warnings.  Fatal errors (``E_*`` codes) mean
    the file should not be processed further.
    """

    def __init__(self, config: XMLProcessorConfig) -> None:
        self.config = config

    def scan(self, file_path: str) -> list[IngestError]:
        """Run all pre-flight checks.

        Returns:
            List of errors/warnings.  Fatal errors have codes starting
            with ``E_``.
        """
        errors: list[IngestError] = []

        # --- 1. Extension check ---
        if not file_path.lower().endswith(".xml"):
            errors.append(
                IngestError(
                    code=ErrorCode.E_SECURITY_BAD_EXTENSION,
                    message=f"File does not have .xml extension: {file_path}",
                    stage="security",
                )
            )
            return errors

        # --- 2. File existence ---
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

        # --- 6. Entity declaration scan (billion laughs / XXE prevention) ---
        try:
            with open(file_path, "rb") as fh:
                raw = fh.read()
        except OSError as exc:
            errors.append(
                IngestError(
                    code=ErrorCode.E_PARSE_CORRUPT,
                    message=f"Cannot read file: {exc}",
                    stage="security",
                )
            )
            return errors

        raw_upper = raw.upper()
        if b"<!ENTITY" in raw_upper:
            errors.append(
                IngestError(
                    code=ErrorCode.E_SECURITY_ENTITY_DECLARATION,
                    message="File contains <!ENTITY declaration (potential billion laughs / XXE attack)",
                    stage="security",
                )
            )
            return errors

        if b"<!DOCTYPE" in raw_upper:
            # Check for internal subset (indicated by '[' after DOCTYPE)
            doctype_pos = raw_upper.find(b"<!DOCTYPE")
            # Look for '[' within the DOCTYPE declaration
            bracket_pos = raw.find(b"[", doctype_pos)
            close_pos = raw.find(b">", doctype_pos)
            if bracket_pos != -1 and (close_pos == -1 or bracket_pos < close_pos):
                errors.append(
                    IngestError(
                        code=ErrorCode.E_SECURITY_ENTITY_DECLARATION,
                        message="File contains <!DOCTYPE with internal subset (potential entity expansion attack)",
                        stage="security",
                    )
                )
                return errors

        # --- 7. XML validity ---
        try:
            tree = ET.parse(file_path)  # noqa: S314
        except ET.ParseError as exc:
            errors.append(
                IngestError(
                    code=ErrorCode.E_SECURITY_INVALID_XML,
                    message=f"Invalid XML: {exc}",
                    stage="security",
                )
            )
            return errors

        # --- 8. Depth check ---
        root = tree.getroot()
        depth = _measure_depth(root)
        if depth > self.config.max_depth:
            errors.append(
                IngestError(
                    code=ErrorCode.E_SECURITY_DEPTH_BOMB,
                    message=(
                        f"XML nesting depth {depth} exceeds limit of "
                        f"{self.config.max_depth}"
                    ),
                    stage="security",
                )
            )
            return errors

        # --- 9. Element count check (warning, not fatal) ---
        count = _count_elements(root)
        if count > self.config.max_elements:
            errors.append(
                IngestError(
                    code=ErrorCode.W_TRUNCATED,
                    message=(
                        f"Element count {count} exceeds limit of "
                        f"{self.config.max_elements}; output will be truncated"
                    ),
                    stage="security",
                    recoverable=True,
                )
            )

        return errors


def _measure_depth(element: ET.Element, _current: int = 1) -> int:
    """Recursively measure the maximum nesting depth of an XML tree."""
    children = list(element)
    if not children:
        return _current
    return max(_measure_depth(child, _current + 1) for child in children)


def _count_elements(element: ET.Element) -> int:
    """Count all elements in the tree (including root)."""
    count = 1
    for child in element:
        count += _count_elements(child)
    return count
