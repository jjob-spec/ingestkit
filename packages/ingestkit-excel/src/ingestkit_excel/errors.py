"""Normalized error codes and structured error model for the ingestkit-excel pipeline."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class ErrorCode(str, Enum):
    """Normalized error codes for the ingestkit-excel pipeline.

    All errors and warnings use a stable string code suitable for metrics,
    alerting, and programmatic handling. Codes prefixed with ``E_`` are errors;
    codes prefixed with ``W_`` are non-fatal warnings.
    """

    # Parse errors
    E_PARSE_CORRUPT = "E_PARSE_CORRUPT"
    E_PARSE_OPENPYXL_FAIL = "E_PARSE_OPENPYXL_FAIL"
    E_PARSE_PANDAS_FAIL = "E_PARSE_PANDAS_FAIL"
    E_PARSE_PASSWORD = "E_PARSE_PASSWORD"
    E_PARSE_EMPTY = "E_PARSE_EMPTY"
    E_PARSE_TOO_LARGE = "E_PARSE_TOO_LARGE"

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
    E_PROCESS_REGION_DETECT = "E_PROCESS_REGION_DETECT"
    E_PROCESS_SERIALIZE = "E_PROCESS_SERIALIZE"
    E_PROCESS_SCHEMA_GEN = "E_PROCESS_SCHEMA_GEN"

    # Warnings (non-fatal)
    W_SHEET_SKIPPED_CHART = "W_SHEET_SKIPPED_CHART"
    W_SHEET_SKIPPED_HIDDEN = "W_SHEET_SKIPPED_HIDDEN"
    W_SHEET_SKIPPED_PASSWORD = "W_SHEET_SKIPPED_PASSWORD"
    W_PARSER_FALLBACK = "W_PARSER_FALLBACK"
    W_LLM_RETRY = "W_LLM_RETRY"
    W_ROWS_TRUNCATED = "W_ROWS_TRUNCATED"


class IngestError(BaseModel):
    """Structured error with code, message, and context.

    Provides a normalized error representation for pipeline failures and
    warnings. Each error carries an ``ErrorCode``, a human-readable message,
    and optional context about which sheet and processing stage produced it.
    """

    code: ErrorCode
    message: str
    sheet_name: str | None = None
    stage: str | None = None
    recoverable: bool = False
