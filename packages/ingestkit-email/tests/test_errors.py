"""Tests for ingestkit_email.errors."""

import pytest

from ingestkit_email.errors import ErrorCode, IngestError


class TestErrorCodes:
    def test_error_codes_prefixed(self):
        """All error codes start with E_ (fatal) or W_ (warning)."""
        for code in ErrorCode:
            assert code.value.startswith("E_") or code.value.startswith("W_"), (
                f"ErrorCode {code.name} does not start with E_ or W_"
            )

    def test_error_code_values_match_names(self):
        """Value == name for all members."""
        for code in ErrorCode:
            assert code.value == code.name

    def test_ingest_error_creation(self):
        """IngestError populates all fields correctly."""
        err = IngestError(
            code=ErrorCode.E_EMAIL_PARSE_FAILED,
            message="Parse failed",
            stage="convert",
            recoverable=False,
        )
        assert err.code == ErrorCode.E_EMAIL_PARSE_FAILED
        assert err.message == "Parse failed"
        assert err.stage == "convert"
        assert err.recoverable is False
