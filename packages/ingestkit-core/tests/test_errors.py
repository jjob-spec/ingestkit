"""Tests for ingestkit_core.errors -- shared error codes and base error model."""

from __future__ import annotations

import pytest

from ingestkit_core.errors import BaseIngestError, CoreErrorCode


class TestSharedErrorCodes:
    """Verify all shared error codes exist and have correct values."""

    # Backend errors (shared across all packages)
    @pytest.mark.parametrize("code_name", [
        "E_BACKEND_VECTOR_TIMEOUT",
        "E_BACKEND_VECTOR_CONNECT",
        "E_BACKEND_DB_TIMEOUT",
        "E_BACKEND_DB_CONNECT",
        "E_BACKEND_EMBED_TIMEOUT",
        "E_BACKEND_EMBED_CONNECT",
    ])
    def test_backend_error_exists(self, code_name: str):
        code = CoreErrorCode[code_name]
        assert code.value == code_name

    # Classification/LLM errors (shared)
    @pytest.mark.parametrize("code_name", [
        "E_CLASSIFY_INCONCLUSIVE",
        "E_LLM_TIMEOUT",
        "E_LLM_MALFORMED_JSON",
        "E_LLM_SCHEMA_INVALID",
        "E_LLM_CONFIDENCE_OOB",
    ])
    def test_classify_error_exists(self, code_name: str):
        code = CoreErrorCode[code_name]
        assert code.value == code_name

    # Parse errors (shared subset)
    @pytest.mark.parametrize("code_name", [
        "E_PARSE_CORRUPT",
        "E_PARSE_PASSWORD",
        "E_PARSE_EMPTY",
    ])
    def test_parse_error_exists(self, code_name: str):
        code = CoreErrorCode[code_name]
        assert code.value == code_name

    # Shared warning
    def test_w_llm_retry_exists(self):
        assert CoreErrorCode.W_LLM_RETRY.value == "W_LLM_RETRY"


class TestErrorCodeProperties:
    """Structural properties of the CoreErrorCode enum."""

    def test_all_str_enum(self):
        for member in CoreErrorCode:
            assert isinstance(member, str)
            assert isinstance(member.value, str)

    def test_all_have_prefix(self):
        for member in CoreErrorCode:
            assert member.value.startswith("E_") or member.value.startswith("W_"), (
                f"{member.name} missing E_/W_ prefix"
            )

    def test_value_equals_name(self):
        for member in CoreErrorCode:
            assert member.value == member.name

    def test_member_count(self):
        assert len(CoreErrorCode) == 15


class TestBaseIngestError:
    """Tests for the base IngestError model (shared fields only)."""

    def test_minimal_construction(self):
        e = BaseIngestError(
            code=CoreErrorCode.E_PARSE_CORRUPT,
            message="File is corrupt",
        )
        assert e.code == CoreErrorCode.E_PARSE_CORRUPT
        assert e.message == "File is corrupt"
        assert e.stage is None
        assert e.recoverable is False

    def test_full_construction(self):
        e = BaseIngestError(
            code=CoreErrorCode.W_LLM_RETRY,
            message="LLM retry #1",
            stage="classify",
            recoverable=True,
        )
        assert e.stage == "classify"
        assert e.recoverable is True

    def test_serialization_round_trip(self):
        e = BaseIngestError(
            code=CoreErrorCode.E_BACKEND_VECTOR_TIMEOUT,
            message="Vector store timed out",
            stage="embed",
            recoverable=False,
        )
        data = e.model_dump()
        e2 = BaseIngestError.model_validate(data)
        assert e2.code == e.code
        assert e2.message == e.message
        assert e2.stage == e.stage
        assert e2.recoverable == e.recoverable

    def test_empty_message_allowed(self):
        e = BaseIngestError(code=CoreErrorCode.E_PARSE_EMPTY, message="")
        assert e.message == ""

    def test_unicode_message(self):
        e = BaseIngestError(
            code=CoreErrorCode.E_PARSE_CORRUPT,
            message="Datei ist besch\u00e4digt",
        )
        assert "besch\u00e4digt" in e.message
