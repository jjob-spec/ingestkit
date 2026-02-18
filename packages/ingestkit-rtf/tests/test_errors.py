"""Tests for ingestkit_rtf.errors."""

from __future__ import annotations

import pytest

from ingestkit_rtf.errors import ErrorCode, IngestError


class TestErrorCode:
    """ErrorCode enum values must equal their names."""

    def test_all_values_equal_names(self):
        for member in ErrorCode:
            assert member.value == member.name

    def test_fatal_codes_start_with_e(self):
        for member in ErrorCode:
            if member.name.startswith("E_"):
                assert member.value.startswith("E_")

    def test_warning_codes_start_with_w(self):
        for member in ErrorCode:
            if member.name.startswith("W_"):
                assert member.value.startswith("W_")


class TestIngestError:
    """IngestError serialisation and fields."""

    def test_round_trip(self):
        err = IngestError(
            code=ErrorCode.E_RTF_EXTRACT_FAILED,
            message="test error",
            stage="extract",
            rtf_section="Body",
        )
        data = err.model_dump()
        restored = IngestError(**data)
        assert restored.code == ErrorCode.E_RTF_EXTRACT_FAILED.value
        assert restored.message == "test error"
        assert restored.rtf_section == "Body"

    def test_rtf_section_default_none(self):
        err = IngestError(
            code=ErrorCode.E_PARSE_CORRUPT,
            message="corrupt",
        )
        assert err.rtf_section is None

    def test_recoverable_default_false(self):
        err = IngestError(
            code=ErrorCode.E_PARSE_CORRUPT,
            message="corrupt",
        )
        assert err.recoverable is False
