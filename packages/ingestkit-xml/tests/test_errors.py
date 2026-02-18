"""Unit tests for ingestkit_xml.errors -- error codes and model."""

from __future__ import annotations

import pytest

from ingestkit_xml.errors import ErrorCode, IngestError


class TestErrorCodeValues:
    """All ErrorCode values should equal their names."""

    def test_all_values_equal_names(self):
        for code in ErrorCode:
            assert code.value == code.name

    def test_fatal_codes_start_with_E(self):
        fatal = [c for c in ErrorCode if c.value.startswith("E_")]
        assert len(fatal) > 0
        for code in fatal:
            assert code.name.startswith("E_")

    def test_warning_codes_start_with_W(self):
        warnings = [c for c in ErrorCode if c.value.startswith("W_")]
        assert len(warnings) > 0
        for code in warnings:
            assert code.name.startswith("W_")


class TestIngestError:
    """Tests for IngestError model."""

    def test_has_xpath_field(self):
        err = IngestError(
            code="E_PARSE_CORRUPT",
            message="test",
            stage="parse",
            xpath="/root/item[0]",
        )
        assert err.xpath == "/root/item[0]"

    def test_xpath_defaults_to_none(self):
        err = IngestError(
            code="E_PARSE_CORRUPT",
            message="test",
            stage="parse",
        )
        assert err.xpath is None
