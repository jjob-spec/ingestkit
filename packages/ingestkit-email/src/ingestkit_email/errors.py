"""Error codes and structured error model for the ingestkit-email package.

``ErrorCode`` contains all email-specific error/warning codes plus the shared
backend codes from the core taxonomy.  ``IngestError`` extends
``BaseIngestError`` with the narrowed ``code`` type.
"""

from __future__ import annotations

from enum import Enum

from ingestkit_core.errors import BaseIngestError


class ErrorCode(str, Enum):
    """Error codes for ingestkit-email.

    Fatal codes use an ``E_`` prefix; warnings use ``W_``.
    Values equal their names for stable metric/alerting strings.
    """

    # Email-specific fatal errors
    E_EMAIL_UNSUPPORTED_FORMAT = "E_EMAIL_UNSUPPORTED_FORMAT"
    E_EMAIL_TOO_LARGE = "E_EMAIL_TOO_LARGE"
    E_EMAIL_FILE_CORRUPT = "E_EMAIL_FILE_CORRUPT"
    E_EMAIL_PARSE_FAILED = "E_EMAIL_PARSE_FAILED"
    E_EMAIL_EMPTY_BODY = "E_EMAIL_EMPTY_BODY"
    E_EMAIL_MSG_UNAVAILABLE = "E_EMAIL_MSG_UNAVAILABLE"

    # Backend errors (reused from core taxonomy)
    E_BACKEND_VECTOR_TIMEOUT = "E_BACKEND_VECTOR_TIMEOUT"
    E_BACKEND_VECTOR_CONNECT = "E_BACKEND_VECTOR_CONNECT"
    E_BACKEND_EMBED_TIMEOUT = "E_BACKEND_EMBED_TIMEOUT"
    E_BACKEND_EMBED_CONNECT = "E_BACKEND_EMBED_CONNECT"

    # Warnings (non-fatal)
    W_EMAIL_ATTACHMENT_SKIPPED = "W_EMAIL_ATTACHMENT_SKIPPED"
    W_EMAIL_HTML_ONLY = "W_EMAIL_HTML_ONLY"
    W_EMAIL_NO_SUBJECT = "W_EMAIL_NO_SUBJECT"
    W_EMAIL_NO_DATE = "W_EMAIL_NO_DATE"


class IngestError(BaseIngestError):
    """Structured error for the email pipeline.

    Narrows the ``code`` field to ``ErrorCode`` for type safety while
    remaining serialisation-compatible with the base class.
    """

    code: ErrorCode
