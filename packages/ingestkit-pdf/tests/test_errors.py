"""Tests for ingestkit_pdf.errors — error codes and IngestError model."""

from __future__ import annotations

import pytest

from ingestkit_pdf.errors import ErrorCode, IngestError


# ---------------------------------------------------------------------------
# ErrorCode Enum Tests
# ---------------------------------------------------------------------------


class TestErrorCodeValues:
    """Verify all error codes have correct string values matching their names."""

    @pytest.mark.parametrize(
        "member",
        list(ErrorCode),
        ids=[m.name for m in ErrorCode],
    )
    def test_value_equals_name(self, member: ErrorCode):
        assert member.value == member.name

    def test_security_error_count(self):
        security = [c for c in ErrorCode if c.value.startswith("E_SECURITY_")]
        assert len(security) == 5

    def test_parse_error_count(self):
        parse = [c for c in ErrorCode if c.value.startswith("E_PARSE_")]
        assert len(parse) == 5

    def test_ocr_error_count(self):
        ocr = [c for c in ErrorCode if c.value.startswith("E_OCR_")]
        assert len(ocr) == 3

    def test_classify_and_llm_error_count(self):
        classify = [
            c
            for c in ErrorCode
            if c.value.startswith("E_CLASSIFY_") or c.value.startswith("E_LLM_")
        ]
        assert len(classify) == 5

    def test_backend_error_count(self):
        backend = [c for c in ErrorCode if c.value.startswith("E_BACKEND_")]
        assert len(backend) == 6

    def test_process_error_count(self):
        process = [c for c in ErrorCode if c.value.startswith("E_PROCESS_")]
        assert len(process) == 3

    def test_warning_count(self):
        warnings = [c for c in ErrorCode if c.value.startswith("W_")]
        assert len(warnings) == 15

    def test_total_member_count(self):
        assert len(ErrorCode) == 42

    def test_all_errors_prefixed(self):
        for code in ErrorCode:
            assert code.value.startswith("E_") or code.value.startswith("W_"), (
                f"{code.name} missing E_/W_ prefix"
            )


class TestErrorCodeLookup:
    def test_lookup_by_value(self):
        assert ErrorCode("E_SECURITY_INVALID_PDF") == ErrorCode.E_SECURITY_INVALID_PDF

    def test_lookup_by_name(self):
        assert ErrorCode["E_PARSE_CORRUPT"] == ErrorCode.E_PARSE_CORRUPT

    def test_string_comparison(self):
        assert ErrorCode.W_OCR_FALLBACK == "W_OCR_FALLBACK"


# ---------------------------------------------------------------------------
# IngestError Model Tests
# ---------------------------------------------------------------------------


class TestIngestError:
    def test_minimal(self):
        e = IngestError(
            code=ErrorCode.E_PARSE_CORRUPT,
            message="PyMuPDF cannot open file",
        )
        assert e.code == ErrorCode.E_PARSE_CORRUPT
        assert e.page_number is None
        assert e.stage is None
        assert e.recoverable is False

    def test_with_page_number(self):
        e = IngestError(
            code=ErrorCode.E_OCR_TIMEOUT,
            message="OCR exceeded timeout on page 5",
            page_number=5,
            stage="ocr",
        )
        assert e.page_number == 5
        assert e.stage == "ocr"

    def test_recoverable_warning(self):
        e = IngestError(
            code=ErrorCode.W_PAGE_SKIPPED_BLANK,
            message="Page 3 is blank, skipping",
            page_number=3,
            stage="parse",
            recoverable=True,
        )
        assert e.recoverable is True

    def test_all_valid_stages(self):
        stages = ["security", "parse", "ocr", "classify", "process", "embed"]
        for stage in stages:
            e = IngestError(
                code=ErrorCode.E_PARSE_EMPTY,
                message="test",
                stage=stage,
            )
            assert e.stage == stage

    def test_serialization_round_trip(self):
        e = IngestError(
            code=ErrorCode.W_LLM_UNAVAILABLE,
            message="LLM backend unreachable",
            stage="classify",
            recoverable=True,
        )
        data = e.model_dump()
        e2 = IngestError.model_validate(data)
        assert e2.code == e.code
        assert e2.message == e.message
        assert e2.recoverable is True

    def test_security_override_warning(self):
        e = IngestError(
            code=ErrorCode.W_SECURITY_OVERRIDE,
            message="reject_javascript overridden: TICKET-4521",
            stage="security",
            recoverable=True,
        )
        assert e.code == ErrorCode.W_SECURITY_OVERRIDE

    def test_classification_degraded_warning(self):
        e = IngestError(
            code=ErrorCode.W_CLASSIFICATION_DEGRADED,
            message="LLM unavailable, using Tier 1 only",
            stage="classify",
            recoverable=True,
        )
        assert e.code == ErrorCode.W_CLASSIFICATION_DEGRADED
