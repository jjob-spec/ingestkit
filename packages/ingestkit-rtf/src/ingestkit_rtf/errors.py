"""Error codes and structured error model for the ingestkit-rtf package.

``ErrorCode`` contains all error/warning codes relevant to RTF ingestion.
``IngestError`` extends ``BaseIngestError`` with an ``rtf_section`` field for
location context.
"""

from __future__ import annotations

from enum import Enum

from ingestkit_core.errors import BaseIngestError


class ErrorCode(str, Enum):
    """Error codes for RTF ingestion.

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
    E_RTF_STRIPRTF_UNAVAILABLE = "E_RTF_STRIPRTF_UNAVAILABLE"
    E_RTF_EXTRACT_FAILED = "E_RTF_EXTRACT_FAILED"
    E_RTF_EMPTY_TEXT = "E_RTF_EMPTY_TEXT"

    # Backend
    E_BACKEND_VECTOR_TIMEOUT = "E_BACKEND_VECTOR_TIMEOUT"
    E_BACKEND_VECTOR_CONNECT = "E_BACKEND_VECTOR_CONNECT"
    E_BACKEND_EMBED_TIMEOUT = "E_BACKEND_EMBED_TIMEOUT"
    E_BACKEND_EMBED_CONNECT = "E_BACKEND_EMBED_CONNECT"

    # Warnings (non-fatal)
    W_LARGE_FILE = "W_LARGE_FILE"


class IngestError(BaseIngestError):
    """Structured error with RTF-specific location context.

    Extends the core ``BaseIngestError`` with an ``rtf_section`` field
    indicating which part of the document caused the issue.
    """

    rtf_section: str | None = None
