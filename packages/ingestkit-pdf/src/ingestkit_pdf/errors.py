"""Normalized error codes and structured error model for the ingestkit-pdf pipeline."""

from __future__ import annotations

from enum import Enum

from ingestkit_core.errors import BaseIngestError


class ErrorCode(str, Enum):
    """Normalized error codes for the ingestkit-pdf pipeline.

    All errors and warnings use a stable string code suitable for metrics,
    alerting, and programmatic handling. Codes prefixed with ``E_`` are errors;
    codes prefixed with ``W_`` are non-fatal warnings.
    """

    # Pre-flight / Security errors
    E_SECURITY_INVALID_PDF = "E_SECURITY_INVALID_PDF"
    E_SECURITY_DECOMPRESSION_BOMB = "E_SECURITY_DECOMPRESSION_BOMB"
    E_SECURITY_JAVASCRIPT = "E_SECURITY_JAVASCRIPT"
    E_SECURITY_TOO_LARGE = "E_SECURITY_TOO_LARGE"
    E_SECURITY_TOO_MANY_PAGES = "E_SECURITY_TOO_MANY_PAGES"

    # Parse / Extraction errors
    E_PARSE_CORRUPT = "E_PARSE_CORRUPT"
    E_PARSE_PASSWORD = "E_PARSE_PASSWORD"
    E_PARSE_EMPTY = "E_PARSE_EMPTY"
    E_PARSE_GARBLED = "E_PARSE_GARBLED"
    E_PARSE_REPAIR_FAILED = "E_PARSE_REPAIR_FAILED"

    # OCR errors
    E_OCR_ENGINE_UNAVAILABLE = "E_OCR_ENGINE_UNAVAILABLE"
    E_OCR_TIMEOUT = "E_OCR_TIMEOUT"
    E_OCR_FAILED = "E_OCR_FAILED"

    # Classification errors
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

    # Processing errors
    E_PROCESS_TABLE_EXTRACT = "E_PROCESS_TABLE_EXTRACT"
    E_PROCESS_CHUNK = "E_PROCESS_CHUNK"
    E_PROCESS_HEADER_FOOTER = "E_PROCESS_HEADER_FOOTER"

    # Warnings (non-fatal)
    W_PAGE_SKIPPED_BLANK = "W_PAGE_SKIPPED_BLANK"
    W_PAGE_SKIPPED_TOC = "W_PAGE_SKIPPED_TOC"
    W_PAGE_SKIPPED_VECTOR_ONLY = "W_PAGE_SKIPPED_VECTOR_ONLY"
    W_PAGE_LOW_OCR_CONFIDENCE = "W_PAGE_LOW_OCR_CONFIDENCE"
    W_QUALITY_LOW_NATIVE = "W_QUALITY_LOW_NATIVE"
    W_OCR_FALLBACK = "W_OCR_FALLBACK"
    W_OCR_ENGINE_FALLBACK = "W_OCR_ENGINE_FALLBACK"
    W_TABLE_CONTINUATION = "W_TABLE_CONTINUATION"
    W_ENCRYPTED_OWNER_ONLY = "W_ENCRYPTED_OWNER_ONLY"
    W_DOCUMENT_SIGNED = "W_DOCUMENT_SIGNED"
    W_EMBEDDED_FILES = "W_EMBEDDED_FILES"
    W_LLM_RETRY = "W_LLM_RETRY"
    W_LLM_UNAVAILABLE = "W_LLM_UNAVAILABLE"
    W_CLASSIFICATION_DEGRADED = "W_CLASSIFICATION_DEGRADED"
    W_SECURITY_OVERRIDE = "W_SECURITY_OVERRIDE"


class IngestError(BaseIngestError):
    """Structured error with code, message, and context.

    Unlike the Excel variant (which uses ``sheet_name``), the PDF variant
    uses ``page_number`` to identify the location of the error within the
    document.
    """

    code: ErrorCode  # type: ignore[assignment]  # narrows base str to ErrorCode
    page_number: int | None = None
