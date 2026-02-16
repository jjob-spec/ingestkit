"""Normalized error codes and structured error model for the ingestkit-forms pipeline.

Extends the base error taxonomy from ingestkit-core with form-specific
error codes for template matching, extraction, and output stages.
See spec section 12 for the authoritative list.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field

from ingestkit_core.errors import BaseIngestError


class FormErrorCode(str, Enum):
    """Normalized error codes for the ingestkit-forms pipeline.

    Extends the base ErrorCode taxonomy with form-specific codes.
    All codes use the same ``E_`` (error) and ``W_`` (warning) prefix convention.
    Every member's name equals its string value for stable metrics and alerting.
    """

    # Template errors (4)
    E_FORM_TEMPLATE_NOT_FOUND = "E_FORM_TEMPLATE_NOT_FOUND"
    E_FORM_TEMPLATE_INVALID = "E_FORM_TEMPLATE_INVALID"
    E_FORM_TEMPLATE_VERSION_CONFLICT = "E_FORM_TEMPLATE_VERSION_CONFLICT"
    E_FORM_TEMPLATE_STORE_UNAVAILABLE = "E_FORM_TEMPLATE_STORE_UNAVAILABLE"

    # Matching errors (2)
    E_FORM_NO_MATCH = "E_FORM_NO_MATCH"
    E_FORM_FINGERPRINT_FAILED = "E_FORM_FINGERPRINT_FAILED"

    # Extraction errors (6)
    E_FORM_EXTRACTION_FAILED = "E_FORM_EXTRACTION_FAILED"
    E_FORM_EXTRACTION_LOW_CONFIDENCE = "E_FORM_EXTRACTION_LOW_CONFIDENCE"
    E_FORM_EXTRACTION_TIMEOUT = "E_FORM_EXTRACTION_TIMEOUT"
    E_FORM_UNSUPPORTED_FORMAT = "E_FORM_UNSUPPORTED_FORMAT"
    E_FORM_OCR_FAILED = "E_FORM_OCR_FAILED"
    E_FORM_NATIVE_FIELDS_UNAVAILABLE = "E_FORM_NATIVE_FIELDS_UNAVAILABLE"

    # Output errors (3)
    E_FORM_DB_SCHEMA_EVOLUTION_FAILED = "E_FORM_DB_SCHEMA_EVOLUTION_FAILED"
    E_FORM_DB_WRITE_FAILED = "E_FORM_DB_WRITE_FAILED"
    E_FORM_CHUNK_WRITE_FAILED = "E_FORM_CHUNK_WRITE_FAILED"

    # Dual-write errors (1)
    E_FORM_DUAL_WRITE_PARTIAL = "E_FORM_DUAL_WRITE_PARTIAL"

    # Manual override errors (1)
    E_FORM_FORMAT_MISMATCH = "E_FORM_FORMAT_MISMATCH"

    # VLM errors (2)
    E_FORM_VLM_UNAVAILABLE = "E_FORM_VLM_UNAVAILABLE"
    E_FORM_VLM_TIMEOUT = "E_FORM_VLM_TIMEOUT"

    # Security errors (2)
    E_FORM_FILE_TOO_LARGE = "E_FORM_FILE_TOO_LARGE"
    E_FORM_FILE_CORRUPT = "E_FORM_FILE_CORRUPT"

    # Backend errors (reuse base codes) (6)
    E_BACKEND_VECTOR_TIMEOUT = "E_BACKEND_VECTOR_TIMEOUT"
    E_BACKEND_VECTOR_CONNECT = "E_BACKEND_VECTOR_CONNECT"
    E_BACKEND_DB_TIMEOUT = "E_BACKEND_DB_TIMEOUT"
    E_BACKEND_DB_CONNECT = "E_BACKEND_DB_CONNECT"
    E_BACKEND_EMBED_TIMEOUT = "E_BACKEND_EMBED_TIMEOUT"
    E_BACKEND_EMBED_CONNECT = "E_BACKEND_EMBED_CONNECT"

    # Warnings (non-fatal) (14)
    W_FORM_FIELD_LOW_CONFIDENCE = "W_FORM_FIELD_LOW_CONFIDENCE"
    W_FORM_FIELD_VALIDATION_FAILED = "W_FORM_FIELD_VALIDATION_FAILED"
    W_FORM_FIELD_MISSING_REQUIRED = "W_FORM_FIELD_MISSING_REQUIRED"
    W_FORM_FIELD_TYPE_COERCION = "W_FORM_FIELD_TYPE_COERCION"
    W_FORM_FIELDS_FLATTENED = "W_FORM_FIELDS_FLATTENED"
    W_FORM_MATCH_BELOW_THRESHOLD = "W_FORM_MATCH_BELOW_THRESHOLD"
    W_FORM_MULTI_MATCH = "W_FORM_MULTI_MATCH"
    W_FORM_OCR_DEGRADED = "W_FORM_OCR_DEGRADED"
    W_FORM_MERGED_CELL_RESOLVED = "W_FORM_MERGED_CELL_RESOLVED"
    W_FORM_SCHEMA_EVOLVED = "W_FORM_SCHEMA_EVOLVED"
    W_FORM_ROLLBACK_FAILED = "W_FORM_ROLLBACK_FAILED"
    W_FORM_PARTIAL_WRITE = "W_FORM_PARTIAL_WRITE"
    W_FORM_VLM_FALLBACK_USED = "W_FORM_VLM_FALLBACK_USED"
    W_FORM_VLM_BUDGET_EXHAUSTED = "W_FORM_VLM_BUDGET_EXHAUSTED"
    W_FORM_NATIVE_FIELDS_UNAVAILABLE = "W_FORM_NATIVE_FIELDS_UNAVAILABLE"


class FormIngestError(BaseIngestError):
    """Structured error with code, message, and form-specific context.

    Extends ``BaseIngestError`` from ingestkit-core with form-specific
    location and diagnostic fields. Follows the same pattern as
    ``IngestError`` in ingestkit-excel.

    Note: This is a Pydantic model (data structure), not a Python Exception.
    To raise errors, use ``FormIngestException`` which wraps this model
    and can be used with ``raise``/``except``.
    """

    code: FormErrorCode  # type: ignore[assignment]  # narrows base str to FormErrorCode
    template_id: str | None = None
    template_version: int | None = None
    field_name: str | None = None
    page_number: int | None = None
    # Diagnostic context (P1)
    candidate_matches: list[dict] | None = Field(
        default=None,
        description="Top template match candidates with confidences (for match-related errors).",
    )
    backend_operation_id: str | None = Field(
        default=None,
        description="Backend-specific operation ID for tracing write/rollback failures.",
    )
    fallback_reason: str | None = Field(
        default=None,
        description="Why fallback was triggered (for degraded-path errors).",
    )


class FormIngestException(Exception):
    """Raisable exception wrapping a FormIngestError data model.

    Use this when you need to raise/catch form errors in control flow.
    Carries the structured ``FormIngestError`` as the ``.error`` attribute
    for inspection and serialization.

    Convenience properties delegate to the underlying error model for
    common fields (``code``, ``message``, ``stage``, ``recoverable``).
    """

    def __init__(self, **kwargs: object) -> None:
        self.error = FormIngestError(**kwargs)  # type: ignore[arg-type]
        super().__init__(self.error.message)

    @property
    def code(self) -> FormErrorCode:
        return self.error.code

    @property
    def message(self) -> str:
        return self.error.message

    @property
    def stage(self) -> str | None:
        return self.error.stage

    @property
    def recoverable(self) -> bool:
        return self.error.recoverable
