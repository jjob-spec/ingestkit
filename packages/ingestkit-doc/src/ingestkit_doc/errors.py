"""Error codes and structured error model for the ingestkit-doc package.

``ErrorCode`` contains all error/warning codes relevant to .doc ingestion.
``IngestError`` extends ``BaseIngestError`` with a ``doc_section`` field for
location context.
"""

from __future__ import annotations

from enum import Enum

from ingestkit_core.errors import BaseIngestError


class ErrorCode(str, Enum):
    """Error codes for .doc ingestion.

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
    E_DOC_MAMMOTH_UNAVAILABLE = "E_DOC_MAMMOTH_UNAVAILABLE"
    E_DOC_EXTRACT_FAILED = "E_DOC_EXTRACT_FAILED"
    E_DOC_EMPTY_TEXT = "E_DOC_EMPTY_TEXT"

    # Backend
    E_BACKEND_VECTOR_TIMEOUT = "E_BACKEND_VECTOR_TIMEOUT"
    E_BACKEND_VECTOR_CONNECT = "E_BACKEND_VECTOR_CONNECT"
    E_BACKEND_EMBED_TIMEOUT = "E_BACKEND_EMBED_TIMEOUT"
    E_BACKEND_EMBED_CONNECT = "E_BACKEND_EMBED_CONNECT"

    # Warnings (non-fatal)
    W_DOC_LEGACY_UNSUPPORTED = "W_DOC_LEGACY_UNSUPPORTED"
    W_DOC_MAMMOTH_MESSAGES = "W_DOC_MAMMOTH_MESSAGES"
    W_LARGE_FILE = "W_LARGE_FILE"


class IngestError(BaseIngestError):
    """Structured error with .doc-specific location context.

    Extends the core ``BaseIngestError`` with a ``doc_section`` field
    indicating which part of the document caused the issue.
    """

    doc_section: str | None = None
