"""Error codes and structured error model for the ingestkit-xls package.

``ErrorCode`` contains all error/warning codes relevant to .xls ingestion.
``IngestError`` extends ``BaseIngestError`` with a ``sheet_name`` field for
location context.
"""

from __future__ import annotations

from enum import Enum

from ingestkit_core.errors import BaseIngestError


class ErrorCode(str, Enum):
    """Error codes for .xls ingestion.

    Values equal their names so they are stable strings suitable for
    metrics and alerting.  ``E_`` prefix = fatal, ``W_`` prefix = warning.
    """

    # Security
    E_SECURITY_BAD_EXTENSION = "E_SECURITY_BAD_EXTENSION"
    E_SECURITY_TOO_LARGE = "E_SECURITY_TOO_LARGE"
    E_SECURITY_BAD_MAGIC = "E_SECURITY_BAD_MAGIC"

    # Parse
    E_PARSE_CORRUPT = "E_PARSE_CORRUPT"
    E_PARSE_EMPTY = "E_PARSE_EMPTY"
    E_XLS_XLRD_UNAVAILABLE = "E_XLS_XLRD_UNAVAILABLE"
    E_XLS_EXTRACT_FAILED = "E_XLS_EXTRACT_FAILED"
    E_XLS_EMPTY_TEXT = "E_XLS_EMPTY_TEXT"
    E_XLS_PASSWORD_PROTECTED = "E_XLS_PASSWORD_PROTECTED"

    # Backend
    E_BACKEND_VECTOR_TIMEOUT = "E_BACKEND_VECTOR_TIMEOUT"
    E_BACKEND_VECTOR_CONNECT = "E_BACKEND_VECTOR_CONNECT"
    E_BACKEND_EMBED_TIMEOUT = "E_BACKEND_EMBED_TIMEOUT"
    E_BACKEND_EMBED_CONNECT = "E_BACKEND_EMBED_CONNECT"

    # Warnings (non-fatal)
    W_XLS_EMPTY_SHEET_SKIPPED = "W_XLS_EMPTY_SHEET_SKIPPED"
    W_XLS_DATE_CONVERSION_FAILED = "W_XLS_DATE_CONVERSION_FAILED"
    W_LARGE_FILE = "W_LARGE_FILE"


class IngestError(BaseIngestError):
    """Structured error with .xls-specific location context.

    Extends the core ``BaseIngestError`` with a ``sheet_name`` field
    indicating which sheet of the workbook caused the issue.
    """

    sheet_name: str | None = None
