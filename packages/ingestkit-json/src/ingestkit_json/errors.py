"""Error codes and structured error model for the ingestkit-json package.

``ErrorCode`` contains all error/warning codes relevant to JSON ingestion.
``IngestError`` extends ``BaseIngestError`` with a ``json_path`` field for
location context.
"""

from __future__ import annotations

from enum import Enum

from ingestkit_core.errors import BaseIngestError


class ErrorCode(str, Enum):
    """Error codes for JSON ingestion.

    Values equal their names so they are stable strings suitable for
    metrics and alerting.  ``E_`` prefix = fatal, ``W_`` prefix = warning.
    """

    # Security
    E_SECURITY_TOO_LARGE = "E_SECURITY_TOO_LARGE"
    E_SECURITY_INVALID_JSON = "E_SECURITY_INVALID_JSON"
    E_SECURITY_NESTING_BOMB = "E_SECURITY_NESTING_BOMB"
    E_SECURITY_BAD_EXTENSION = "E_SECURITY_BAD_EXTENSION"

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


class IngestError(BaseIngestError):
    """Structured error with JSON-specific location context.

    Extends the core ``BaseIngestError`` with a ``json_path`` field
    indicating which key path caused the issue (e.g. ``"items[0].name"``).
    """

    json_path: str | None = None
