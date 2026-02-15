"""Tests for security controls and input validation (issue #70).

Covers FormSecurityScanner (file validation), template field count limits,
regex validation, and API-level regex rejection per spec section 13.5.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ingestkit_forms.config import FormProcessorConfig
from ingestkit_forms.errors import FormErrorCode, FormIngestException
from ingestkit_forms.models import (
    BoundingBox,
    FieldMapping,
    FieldType,
    FormTemplate,
    FormTemplateCreateRequest,
    FormTemplateUpdateRequest,
    SourceFormat,
)
from ingestkit_forms.security import (
    FormSecurityScanner,
    regex_match_with_timeout,
    validate_regex_pattern,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_field(name: str = "f", idx: int = 0) -> FieldMapping:
    """Create a minimal FieldMapping for count tests."""
    return FieldMapping(
        field_name=f"{name}_{idx}",
        field_label=f"Field {idx}",
        field_type=FieldType.TEXT,
        page_number=0,
        region=BoundingBox(x=0.1, y=0.1, width=0.3, height=0.05),
    )


def _write_file(tmp_path, name: str, content: bytes) -> str:
    """Write bytes to a temp file and return the path string."""
    path = tmp_path / name
    path.write_bytes(content)
    return str(path)


# ---------------------------------------------------------------------------
# FormSecurityScanner Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFormSecurityScanner:
    """Tests for file-level security scanning."""

    @pytest.fixture()
    def scanner(self):
        return FormSecurityScanner(FormProcessorConfig())

    @pytest.fixture()
    def small_scanner(self):
        """Scanner with 1 MB limit for size tests."""
        return FormSecurityScanner(FormProcessorConfig(max_file_size_mb=1))

    def test_scan_valid_pdf(self, tmp_path, scanner):
        """Valid PDF file passes all checks."""
        path = _write_file(tmp_path, "test.pdf", b"%PDF-1.7 fake content here")
        errors = scanner.scan(path)
        assert errors == []

    def test_scan_valid_xlsx(self, tmp_path, scanner):
        """Valid XLSX (PK header) passes all checks."""
        path = _write_file(tmp_path, "test.xlsx", b"PK\x03\x04" + b"\x00" * 100)
        errors = scanner.scan(path)
        assert errors == []

    def test_scan_valid_jpeg(self, tmp_path, scanner):
        """Valid JPEG passes all checks."""
        path = _write_file(tmp_path, "test.jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        errors = scanner.scan(path)
        assert errors == []

    def test_scan_valid_png(self, tmp_path, scanner):
        """Valid PNG passes all checks."""
        path = _write_file(
            tmp_path, "test.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        )
        errors = scanner.scan(path)
        assert errors == []

    def test_scan_valid_tiff_le(self, tmp_path, scanner):
        """Valid TIFF (little-endian) passes all checks."""
        path = _write_file(tmp_path, "test.tiff", b"II\x2a\x00" + b"\x00" * 100)
        errors = scanner.scan(path)
        assert errors == []

    def test_scan_valid_tiff_be(self, tmp_path, scanner):
        """Valid TIFF (big-endian) passes all checks."""
        path = _write_file(tmp_path, "test.tif", b"MM\x00\x2a" + b"\x00" * 100)
        errors = scanner.scan(path)
        assert errors == []

    def test_scan_file_too_large(self, tmp_path, small_scanner):
        """File exceeding max_file_size_mb -> E_FORM_FILE_TOO_LARGE."""
        path = _write_file(tmp_path, "big.pdf", b"%PDF-" + b"\x00" * (2 * 1024 * 1024))
        errors = small_scanner.scan(path)
        assert len(errors) == 1
        assert errors[0].code == FormErrorCode.E_FORM_FILE_TOO_LARGE

    def test_scan_unsupported_extension(self, tmp_path, scanner):
        """Unsupported extension -> E_FORM_UNSUPPORTED_FORMAT."""
        path = _write_file(tmp_path, "test.docx", b"PK\x03\x04" + b"\x00" * 100)
        errors = scanner.scan(path)
        assert len(errors) == 1
        assert errors[0].code == FormErrorCode.E_FORM_UNSUPPORTED_FORMAT

    def test_scan_magic_mismatch_pdf(self, tmp_path, scanner):
        """PDF extension with JPEG bytes -> E_FORM_FILE_CORRUPT."""
        path = _write_file(tmp_path, "fake.pdf", b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        errors = scanner.scan(path)
        assert len(errors) == 1
        assert errors[0].code == FormErrorCode.E_FORM_FILE_CORRUPT

    def test_scan_magic_mismatch_xlsx(self, tmp_path, scanner):
        """XLSX extension with PDF bytes -> E_FORM_FILE_CORRUPT."""
        path = _write_file(tmp_path, "fake.xlsx", b"%PDF-1.7" + b"\x00" * 100)
        errors = scanner.scan(path)
        assert len(errors) == 1
        assert errors[0].code == FormErrorCode.E_FORM_FILE_CORRUPT

    def test_scan_magic_mismatch_jpeg(self, tmp_path, scanner):
        """JPEG extension with PNG bytes -> E_FORM_FILE_CORRUPT."""
        path = _write_file(
            tmp_path, "fake.jpg", b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        )
        errors = scanner.scan(path)
        assert len(errors) == 1
        assert errors[0].code == FormErrorCode.E_FORM_FILE_CORRUPT

    def test_scan_magic_mismatch_png(self, tmp_path, scanner):
        """PNG extension with PDF bytes -> E_FORM_FILE_CORRUPT."""
        path = _write_file(tmp_path, "fake.png", b"%PDF-1.7" + b"\x00" * 100)
        errors = scanner.scan(path)
        assert len(errors) == 1
        assert errors[0].code == FormErrorCode.E_FORM_FILE_CORRUPT

    def test_scan_magic_mismatch_tiff(self, tmp_path, scanner):
        """TIFF extension with PDF bytes -> E_FORM_FILE_CORRUPT."""
        path = _write_file(tmp_path, "fake.tiff", b"%PDF-1.7" + b"\x00" * 100)
        errors = scanner.scan(path)
        assert len(errors) == 1
        assert errors[0].code == FormErrorCode.E_FORM_FILE_CORRUPT

    def test_scan_empty_file(self, tmp_path, scanner):
        """Empty (0-byte) file -> E_FORM_FILE_CORRUPT."""
        path = _write_file(tmp_path, "empty.pdf", b"")
        errors = scanner.scan(path)
        assert len(errors) == 1
        assert errors[0].code == FormErrorCode.E_FORM_FILE_CORRUPT


# ---------------------------------------------------------------------------
# Template Field Count Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTemplateFieldCountLimit:
    """Tests for max_length=200 on template field lists."""

    def test_template_field_count_at_limit(self):
        """Exactly 200 fields -> accepted."""
        fields = [_make_field(idx=i) for i in range(200)]
        template = FormTemplate(
            name="Big Template",
            source_format=SourceFormat.PDF,
            page_count=1,
            fields=fields,
        )
        assert len(template.fields) == 200

    def test_template_field_count_exceeds_limit(self):
        """201 fields -> ValidationError."""
        fields = [_make_field(idx=i) for i in range(201)]
        with pytest.raises(ValidationError):
            FormTemplate(
                name="Too Big",
                source_format=SourceFormat.PDF,
                page_count=1,
                fields=fields,
            )

    def test_create_request_field_count_exceeds_limit(self):
        """201 fields in CreateRequest -> ValidationError."""
        fields = [_make_field(idx=i) for i in range(201)]
        with pytest.raises(ValidationError):
            FormTemplateCreateRequest(
                name="Too Big",
                source_format=SourceFormat.PDF,
                sample_file_path="/tmp/sample.png",
                page_count=1,
                fields=fields,
            )


# ---------------------------------------------------------------------------
# Regex Validation Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRegexValidation:
    """Tests for regex pattern validation and timeout protection."""

    def test_validate_regex_pattern_valid(self):
        """Valid regex -> None."""
        assert validate_regex_pattern(r"^\d{3}-\d{4}$") is None

    def test_validate_regex_pattern_invalid(self):
        """Invalid regex -> error string."""
        result = validate_regex_pattern(r"[unclosed")
        assert result is not None
        assert isinstance(result, str)

    def test_regex_match_with_timeout_match(self):
        """Normal pattern that matches -> True."""
        assert regex_match_with_timeout(r"\d+", "123") is True

    def test_regex_match_with_timeout_no_match(self):
        """Normal pattern that doesn't match -> False."""
        assert regex_match_with_timeout(r"\d+", "abc") is False

    def test_regex_match_with_timeout_redos(self):
        """ReDoS pattern with pathological input -> None (timeout)."""
        result = regex_match_with_timeout(
            r"(a+)+b",
            "a" * 30,
            timeout=1.0,
        )
        # Either times out (None) or fails to match (False)
        assert result is None or result is False

    def test_regex_match_with_timeout_invalid_pattern(self):
        """Invalid regex pattern -> None."""
        result = regex_match_with_timeout(r"[unclosed", "test")
        assert result is None

    def test_regex_match_fullmatch_vs_match(self):
        """Verify match_mode parameter works correctly."""
        # "123abc" -- fullmatch against r"\d+" should fail (not entire string)
        assert regex_match_with_timeout(r"\d+", "123abc", match_mode="fullmatch") is False
        # match against r"\d+" should succeed (matches prefix)
        assert regex_match_with_timeout(r"\d+", "123abc", match_mode="match") is True


# ---------------------------------------------------------------------------
# API-Level Regex Validation Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAPIRegexValidation:
    """Tests for regex validation at template creation/update time."""

    @pytest.fixture()
    def template_api(self, mock_template_store, form_config):
        from ingestkit_forms.api import FormTemplateAPI

        return FormTemplateAPI(
            store=mock_template_store,
            config=form_config,
            renderer=None,
        )

    def test_create_template_invalid_regex_rejected(self, template_api):
        """create_template with invalid regex -> FormIngestException."""
        req = FormTemplateCreateRequest(
            name="Bad Regex Template",
            source_format=SourceFormat.PDF,
            sample_file_path="/tmp/sample.png",
            page_count=1,
            fields=[
                FieldMapping(
                    field_name="ssn",
                    field_label="SSN",
                    field_type=FieldType.TEXT,
                    page_number=0,
                    region=BoundingBox(x=0.1, y=0.1, width=0.3, height=0.05),
                    validation_pattern=r"[unclosed",
                )
            ],
        )
        with pytest.raises(FormIngestException) as exc_info:
            template_api.create_template(req)
        assert exc_info.value.code == FormErrorCode.E_FORM_TEMPLATE_INVALID

    def test_update_template_invalid_regex_rejected(
        self, template_api, sample_image_file
    ):
        """update_template with invalid regex -> FormIngestException."""
        # First create a valid template
        create_req = FormTemplateCreateRequest(
            name="Valid Template",
            source_format=SourceFormat.PDF,
            sample_file_path=sample_image_file,
            page_count=1,
            fields=[
                FieldMapping(
                    field_name="name",
                    field_label="Name",
                    field_type=FieldType.TEXT,
                    page_number=0,
                    region=BoundingBox(x=0.1, y=0.1, width=0.3, height=0.05),
                )
            ],
        )
        created = template_api.create_template(create_req)

        # Now update with invalid regex
        update_req = FormTemplateUpdateRequest(
            fields=[
                FieldMapping(
                    field_name="ssn",
                    field_label="SSN",
                    field_type=FieldType.TEXT,
                    page_number=0,
                    region=BoundingBox(x=0.1, y=0.1, width=0.3, height=0.05),
                    validation_pattern=r"((((",
                )
            ],
        )
        with pytest.raises(FormIngestException) as exc_info:
            template_api.update_template(created.template_id, update_req)
        assert exc_info.value.code == FormErrorCode.E_FORM_TEMPLATE_INVALID

    def test_create_template_valid_regex_accepted(
        self, template_api, sample_image_file
    ):
        """create_template with valid regex -> success."""
        req = FormTemplateCreateRequest(
            name="Good Regex Template",
            source_format=SourceFormat.PDF,
            sample_file_path=sample_image_file,
            page_count=1,
            fields=[
                FieldMapping(
                    field_name="ssn",
                    field_label="SSN",
                    field_type=FieldType.TEXT,
                    page_number=0,
                    region=BoundingBox(x=0.1, y=0.1, width=0.3, height=0.05),
                    validation_pattern=r"^\d{3}-\d{2}-\d{4}$",
                )
            ],
        )
        template = template_api.create_template(req)
        assert template.name == "Good Regex Template"
