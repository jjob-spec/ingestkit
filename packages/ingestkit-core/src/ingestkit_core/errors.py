"""Shared error codes and base error model for the ingestkit framework.

``CoreErrorCode`` contains the 15 error/warning codes common to all ingestkit
packages.  ``BaseIngestError`` is a Pydantic model that each package extends
with its own location field (e.g. ``sheet_name`` for Excel, ``page_number``
for PDF).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class CoreErrorCode(str, Enum):
    """Error codes shared across all ingestkit packages.

    Each package maintains its own *complete* ``ErrorCode`` enum that includes
    both the shared codes here and package-specific codes.  Values equal their
    names so they are stable strings suitable for metrics and alerting.
    """

    # Parse errors
    E_PARSE_CORRUPT = "E_PARSE_CORRUPT"
    E_PARSE_PASSWORD = "E_PARSE_PASSWORD"
    E_PARSE_EMPTY = "E_PARSE_EMPTY"

    # Classification / LLM errors
    E_CLASSIFY_INCONCLUSIVE = "E_CLASSIFY_INCONCLUSIVE"
    E_LLM_TIMEOUT = "E_LLM_TIMEOUT"
    E_LLM_MALFORMED_JSON = "E_LLM_MALFORMED_JSON"
    E_LLM_SCHEMA_INVALID = "E_LLM_SCHEMA_INVALID"
    E_LLM_CONFIDENCE_OOB = "E_LLM_CONFIDENCE_OOB"

    # Backend errors
    E_BACKEND_VECTOR_TIMEOUT = "E_BACKEND_VECTOR_TIMEOUT"
    E_BACKEND_VECTOR_CONNECT = "E_BACKEND_VECTOR_CONNECT"
    E_BACKEND_DB_TIMEOUT = "E_BACKEND_DB_TIMEOUT"
    E_BACKEND_DB_CONNECT = "E_BACKEND_DB_CONNECT"
    E_BACKEND_EMBED_TIMEOUT = "E_BACKEND_EMBED_TIMEOUT"
    E_BACKEND_EMBED_CONNECT = "E_BACKEND_EMBED_CONNECT"

    # Warnings (non-fatal)
    W_LLM_RETRY = "W_LLM_RETRY"


class BaseIngestError(BaseModel):
    """Base structured error with code, message, and context.

    Each package extends this model with a location field specific to its
    document type (e.g. ``sheet_name`` for Excel, ``page_number`` for PDF).
    The ``code`` field is typed as ``str`` so it accepts any package-specific
    ``ErrorCode`` enum member.
    """

    code: str
    message: str
    stage: str | None = None
    recoverable: bool = False
