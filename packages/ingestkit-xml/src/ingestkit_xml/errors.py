"""Error codes and structured error model for the ingestkit-xml package.

``ErrorCode`` contains all error/warning codes relevant to XML ingestion.
``IngestError`` extends ``BaseIngestError`` with an ``xpath`` field for
location context.
"""

from __future__ import annotations

from enum import Enum

from ingestkit_core.errors import BaseIngestError


class ErrorCode(str, Enum):
    """Error codes for XML ingestion.

    Values equal their names so they are stable strings suitable for
    metrics and alerting.  ``E_`` prefix = fatal, ``W_`` prefix = warning.
    """

    # Security
    E_SECURITY_TOO_LARGE = "E_SECURITY_TOO_LARGE"
    E_SECURITY_BAD_EXTENSION = "E_SECURITY_BAD_EXTENSION"
    E_SECURITY_ENTITY_DECLARATION = "E_SECURITY_ENTITY_DECLARATION"
    E_SECURITY_INVALID_XML = "E_SECURITY_INVALID_XML"
    E_SECURITY_DEPTH_BOMB = "E_SECURITY_DEPTH_BOMB"

    # Parse
    E_PARSE_EMPTY = "E_PARSE_EMPTY"
    E_PARSE_CORRUPT = "E_PARSE_CORRUPT"

    # Backend
    E_BACKEND_VECTOR_TIMEOUT = "E_BACKEND_VECTOR_TIMEOUT"
    E_BACKEND_VECTOR_CONNECT = "E_BACKEND_VECTOR_CONNECT"
    E_BACKEND_EMBED_TIMEOUT = "E_BACKEND_EMBED_TIMEOUT"
    E_BACKEND_EMBED_CONNECT = "E_BACKEND_EMBED_CONNECT"

    # Warnings (non-fatal)
    W_LARGE_FILE = "W_LARGE_FILE"
    W_TRUNCATED = "W_TRUNCATED"
    W_MALFORMED_FALLBACK = "W_MALFORMED_FALLBACK"


class IngestError(BaseIngestError):
    """Structured error with XML-specific location context.

    Extends the core ``BaseIngestError`` with an ``xpath`` field
    indicating which element path caused the issue.
    """

    xpath: str | None = None
