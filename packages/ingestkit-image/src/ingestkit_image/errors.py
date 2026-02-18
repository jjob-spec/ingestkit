"""Error codes and structured error model for the ingestkit-image package.

``ImageErrorCode`` contains all error/warning codes specific to image
processing.  ``ImageIngestError`` extends ``BaseIngestError`` with a
``file_path`` location field.
"""

from __future__ import annotations

from enum import Enum

from ingestkit_core.errors import BaseIngestError


class ImageErrorCode(str, Enum):
    """Error codes for image ingestion.

    Each value equals its name so codes are stable strings suitable for
    metrics and alerting.  ``E_`` prefix indicates fatal errors;
    ``W_`` prefix indicates non-fatal warnings.
    """

    # Security / validation
    E_IMAGE_UNSUPPORTED_FORMAT = "E_IMAGE_UNSUPPORTED_FORMAT"
    E_IMAGE_TOO_LARGE = "E_IMAGE_TOO_LARGE"
    E_IMAGE_CORRUPT = "E_IMAGE_CORRUPT"
    E_IMAGE_DIMENSIONS_EXCEEDED = "E_IMAGE_DIMENSIONS_EXCEEDED"
    E_IMAGE_EMPTY = "E_IMAGE_EMPTY"

    # VLM errors
    E_IMAGE_VLM_UNAVAILABLE = "E_IMAGE_VLM_UNAVAILABLE"
    E_IMAGE_VLM_TIMEOUT = "E_IMAGE_VLM_TIMEOUT"
    E_IMAGE_VLM_EMPTY_RESPONSE = "E_IMAGE_VLM_EMPTY_RESPONSE"

    # Backend errors
    E_BACKEND_VECTOR_TIMEOUT = "E_BACKEND_VECTOR_TIMEOUT"
    E_BACKEND_VECTOR_CONNECT = "E_BACKEND_VECTOR_CONNECT"
    E_BACKEND_EMBED_TIMEOUT = "E_BACKEND_EMBED_TIMEOUT"
    E_BACKEND_EMBED_CONNECT = "E_BACKEND_EMBED_CONNECT"

    # OCR errors
    E_IMAGE_OCR_UNAVAILABLE = "E_IMAGE_OCR_UNAVAILABLE"
    E_IMAGE_OCR_TIMEOUT = "E_IMAGE_OCR_TIMEOUT"
    E_IMAGE_OCR_EMPTY_TEXT = "E_IMAGE_OCR_EMPTY_TEXT"
    E_IMAGE_OCR_FAILED = "E_IMAGE_OCR_FAILED"
    E_IMAGE_NO_BACKEND = "E_IMAGE_NO_BACKEND"

    # Warnings (non-fatal)
    W_IMAGE_VLM_LOW_DETAIL = "W_IMAGE_VLM_LOW_DETAIL"
    W_IMAGE_VLM_RETRY = "W_IMAGE_VLM_RETRY"
    W_IMAGE_OCR_LOW_CONFIDENCE = "W_IMAGE_OCR_LOW_CONFIDENCE"
    W_IMAGE_OCR_RETRY = "W_IMAGE_OCR_RETRY"
    W_IMAGE_OCR_RESIZED = "W_IMAGE_OCR_RESIZED"


class ImageIngestError(BaseIngestError):
    """Structured error for image ingestion with file path context.

    Extends ``BaseIngestError`` with a ``file_path`` location field
    analogous to ``page_number`` for PDF or ``sheet_name`` for Excel.
    """

    file_path: str | None = None
