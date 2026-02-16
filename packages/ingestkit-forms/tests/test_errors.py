"""Unit tests for ingestkit_forms.errors -- FormErrorCode enum and FormIngestError model."""

from __future__ import annotations

import pytest

from ingestkit_core.errors import BaseIngestError, CoreErrorCode
from ingestkit_forms.errors import FormErrorCode, FormIngestError


# ---------------------------------------------------------------------------
# FormErrorCode enum tests
# ---------------------------------------------------------------------------


class TestFormErrorCodeEnum:
    """Tests for the FormErrorCode(str, Enum) taxonomy."""

    def test_total_member_count(self) -> None:
        """Enum has exactly 42 members (21 E_FORM_* + 6 E_BACKEND_* + 15 W_FORM_*)."""
        assert len(FormErrorCode) == 42

    def test_all_names_equal_values(self) -> None:
        """Every enum member's name equals its string value (ENUM_VALUE guard)."""
        for member in FormErrorCode:
            assert member.name == member.value, (
                f"ENUM_VALUE mismatch: {member.name!r} != {member.value!r}"
            )

    def test_all_members_are_strings(self) -> None:
        """FormErrorCode inherits from str, so every member is a str."""
        for member in FormErrorCode:
            assert isinstance(member, str)

    def test_error_codes_prefixed_with_e(self) -> None:
        """All E_ codes start with 'E_'."""
        error_codes = [m for m in FormErrorCode if m.value.startswith("E_")]
        assert len(error_codes) == 27  # 21 form-specific + 6 backend

    def test_warning_codes_prefixed_with_w(self) -> None:
        """All W_ codes start with 'W_'."""
        warning_codes = [m for m in FormErrorCode if m.value.startswith("W_")]
        assert len(warning_codes) == 15

    def test_form_specific_error_codes_count(self) -> None:
        """21 form-specific E_ codes (excluding E_BACKEND_*)."""
        form_errors = [
            m for m in FormErrorCode
            if m.value.startswith("E_FORM_")
        ]
        assert len(form_errors) == 21

    def test_backend_codes_count(self) -> None:
        """6 E_BACKEND_* codes reused from core."""
        backend_codes = [
            m for m in FormErrorCode
            if m.value.startswith("E_BACKEND_")
        ]
        assert len(backend_codes) == 6

    def test_backend_codes_match_core(self) -> None:
        """E_BACKEND_* codes in FormErrorCode match values in CoreErrorCode."""
        core_backend = {
            m.value for m in CoreErrorCode if m.value.startswith("E_BACKEND_")
        }
        form_backend = {
            m.value for m in FormErrorCode if m.value.startswith("E_BACKEND_")
        }
        assert form_backend == core_backend

    @pytest.mark.parametrize(
        "code",
        [
            "E_FORM_TEMPLATE_NOT_FOUND",
            "E_FORM_TEMPLATE_INVALID",
            "E_FORM_TEMPLATE_VERSION_CONFLICT",
            "E_FORM_TEMPLATE_STORE_UNAVAILABLE",
            "E_FORM_NO_MATCH",
            "E_FORM_FINGERPRINT_FAILED",
            "E_FORM_EXTRACTION_FAILED",
            "E_FORM_EXTRACTION_LOW_CONFIDENCE",
            "E_FORM_EXTRACTION_TIMEOUT",
            "E_FORM_UNSUPPORTED_FORMAT",
            "E_FORM_OCR_FAILED",
            "E_FORM_NATIVE_FIELDS_UNAVAILABLE",
            "E_FORM_DB_SCHEMA_EVOLUTION_FAILED",
            "E_FORM_DB_WRITE_FAILED",
            "E_FORM_CHUNK_WRITE_FAILED",
            "E_FORM_DUAL_WRITE_PARTIAL",
            "E_FORM_FORMAT_MISMATCH",
            "E_FORM_VLM_UNAVAILABLE",
            "E_FORM_VLM_TIMEOUT",
            "E_FORM_FILE_TOO_LARGE",
            "E_FORM_FILE_CORRUPT",
        ],
    )
    def test_form_error_code_exists(self, code: str) -> None:
        """Each spec-defined E_FORM_* code exists in the enum."""
        assert FormErrorCode(code).value == code

    @pytest.mark.parametrize(
        "code",
        [
            "W_FORM_FIELD_LOW_CONFIDENCE",
            "W_FORM_FIELD_VALIDATION_FAILED",
            "W_FORM_FIELD_MISSING_REQUIRED",
            "W_FORM_FIELD_TYPE_COERCION",
            "W_FORM_FIELDS_FLATTENED",
            "W_FORM_MATCH_BELOW_THRESHOLD",
            "W_FORM_MULTI_MATCH",
            "W_FORM_OCR_DEGRADED",
            "W_FORM_MERGED_CELL_RESOLVED",
            "W_FORM_SCHEMA_EVOLVED",
            "W_FORM_ROLLBACK_FAILED",
            "W_FORM_PARTIAL_WRITE",
            "W_FORM_VLM_FALLBACK_USED",
            "W_FORM_VLM_BUDGET_EXHAUSTED",
        ],
    )
    def test_form_warning_code_exists(self, code: str) -> None:
        """Each spec-defined W_FORM_* code exists in the enum."""
        assert FormErrorCode(code).value == code


# ---------------------------------------------------------------------------
# FormIngestError model tests
# ---------------------------------------------------------------------------


class TestFormIngestError:
    """Tests for the FormIngestError Pydantic model."""

    def test_is_subclass_of_base_ingest_error(self) -> None:
        """FormIngestError extends BaseIngestError."""
        assert issubclass(FormIngestError, BaseIngestError)

    def test_minimal_construction(self) -> None:
        """Can construct with just code and message."""
        err = FormIngestError(
            code=FormErrorCode.E_FORM_NO_MATCH,
            message="No template matched the input document.",
        )
        assert err.code == FormErrorCode.E_FORM_NO_MATCH
        assert err.message == "No template matched the input document."

    def test_default_recoverable_is_false(self) -> None:
        """The recoverable field defaults to False."""
        err = FormIngestError(
            code=FormErrorCode.E_FORM_OCR_FAILED,
            message="OCR engine unavailable.",
        )
        assert err.recoverable is False

    def test_default_optional_fields_are_none(self) -> None:
        """All optional fields default to None."""
        err = FormIngestError(
            code=FormErrorCode.E_FORM_EXTRACTION_FAILED,
            message="Extraction failed.",
        )
        assert err.template_id is None
        assert err.template_version is None
        assert err.field_name is None
        assert err.page_number is None
        assert err.stage is None
        assert err.candidate_matches is None
        assert err.backend_operation_id is None
        assert err.fallback_reason is None

    def test_full_construction(self) -> None:
        """Can construct with all fields populated."""
        err = FormIngestError(
            code=FormErrorCode.E_FORM_NO_MATCH,
            message="No template matched.",
            template_id="tpl-abc-123",
            template_version=3,
            field_name="employee_name",
            page_number=2,
            stage="matching",
            recoverable=True,
            candidate_matches=[
                {"template_id": "tpl-xyz", "confidence": 0.45},
            ],
            backend_operation_id="op-789",
            fallback_reason="VLM budget exhausted",
        )
        assert err.template_id == "tpl-abc-123"
        assert err.template_version == 3
        assert err.field_name == "employee_name"
        assert err.page_number == 2
        assert err.stage == "matching"
        assert err.recoverable is True
        assert len(err.candidate_matches) == 1
        assert err.backend_operation_id == "op-789"
        assert err.fallback_reason == "VLM budget exhausted"

    def test_serialization_round_trip(self) -> None:
        """Pydantic model_dump / model_validate round-trip preserves data."""
        original = FormIngestError(
            code=FormErrorCode.E_FORM_DUAL_WRITE_PARTIAL,
            message="Partial write -- rollback initiated.",
            template_id="tpl-dual",
            template_version=1,
            stage="output",
            recoverable=False,
            backend_operation_id="op-dual-001",
        )
        data = original.model_dump()
        restored = FormIngestError.model_validate(data)
        assert restored == original
        assert restored.code == FormErrorCode.E_FORM_DUAL_WRITE_PARTIAL
        assert restored.backend_operation_id == "op-dual-001"

    def test_json_round_trip(self) -> None:
        """JSON serialization and deserialization preserves data."""
        original = FormIngestError(
            code=FormErrorCode.W_FORM_VLM_FALLBACK_USED,
            message="VLM fallback used for field extraction.",
            field_name="date_signed",
            fallback_reason="Primary OCR confidence below threshold.",
        )
        json_str = original.model_dump_json()
        restored = FormIngestError.model_validate_json(json_str)
        assert restored == original

    def test_code_field_accepts_warning_codes(self) -> None:
        """Warning codes are valid FormErrorCode values for the code field."""
        err = FormIngestError(
            code=FormErrorCode.W_FORM_SCHEMA_EVOLVED,
            message="Schema was evolved to accommodate new fields.",
            recoverable=True,
        )
        assert err.code == FormErrorCode.W_FORM_SCHEMA_EVOLVED

    def test_code_field_accepts_backend_codes(self) -> None:
        """Backend codes are valid FormErrorCode values for the code field."""
        err = FormIngestError(
            code=FormErrorCode.E_BACKEND_DB_TIMEOUT,
            message="Database operation timed out.",
            backend_operation_id="op-timeout-123",
        )
        assert err.code == FormErrorCode.E_BACKEND_DB_TIMEOUT
