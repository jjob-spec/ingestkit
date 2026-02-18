"""Tests for ingestkit_xls.errors."""

from __future__ import annotations

import pytest

from ingestkit_xls.errors import ErrorCode, IngestError


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
            code=ErrorCode.E_XLS_EXTRACT_FAILED,
            message="test error",
            stage="extract",
            sheet_name="Sheet1",
        )
        data = err.model_dump()
        restored = IngestError(**data)
        assert restored.code == ErrorCode.E_XLS_EXTRACT_FAILED.value
        assert restored.message == "test error"
        assert restored.sheet_name == "Sheet1"

    def test_sheet_name_default_none(self):
        err = IngestError(
            code=ErrorCode.E_PARSE_CORRUPT,
            message="corrupt",
        )
        assert err.sheet_name is None

    def test_recoverable_default_false(self):
        err = IngestError(
            code=ErrorCode.E_PARSE_CORRUPT,
            message="corrupt",
        )
        assert err.recoverable is False

    def test_inherits_from_base_ingest_error(self):
        from ingestkit_core.errors import BaseIngestError

        err = IngestError(
            code=ErrorCode.E_PARSE_CORRUPT,
            message="corrupt",
        )
        assert isinstance(err, BaseIngestError)
