"""Tests for ingestkit-image error codes and error model."""

from __future__ import annotations

import pytest

from ingestkit_core.errors import BaseIngestError

from ingestkit_image.errors import ImageErrorCode, ImageIngestError


@pytest.mark.unit
class TestImageErrorCode:
    """Test ImageErrorCode enum."""

    def test_all_e_codes_are_fatal(self):
        """All E_ codes should represent fatal errors."""
        for code in ImageErrorCode:
            if code.name.startswith("E_"):
                assert code.value.startswith("E_"), (
                    f"{code.name} should have value starting with E_"
                )

    def test_all_w_codes_are_warnings(self):
        """All W_ codes should represent warnings."""
        for code in ImageErrorCode:
            if code.name.startswith("W_"):
                assert code.value.startswith("W_"), (
                    f"{code.name} should have value starting with W_"
                )

    def test_name_equals_value(self):
        """Each code's name should equal its value."""
        for code in ImageErrorCode:
            assert code.name == code.value

    def test_string_enum(self):
        """ImageErrorCode should be a string enum."""
        assert isinstance(ImageErrorCode.E_IMAGE_CORRUPT, str)
        assert ImageErrorCode.E_IMAGE_CORRUPT == "E_IMAGE_CORRUPT"

    def test_security_codes_exist(self):
        assert ImageErrorCode.E_IMAGE_UNSUPPORTED_FORMAT
        assert ImageErrorCode.E_IMAGE_TOO_LARGE
        assert ImageErrorCode.E_IMAGE_CORRUPT
        assert ImageErrorCode.E_IMAGE_DIMENSIONS_EXCEEDED
        assert ImageErrorCode.E_IMAGE_EMPTY

    def test_vlm_codes_exist(self):
        assert ImageErrorCode.E_IMAGE_VLM_UNAVAILABLE
        assert ImageErrorCode.E_IMAGE_VLM_TIMEOUT
        assert ImageErrorCode.E_IMAGE_VLM_EMPTY_RESPONSE

    def test_backend_codes_exist(self):
        assert ImageErrorCode.E_BACKEND_VECTOR_TIMEOUT
        assert ImageErrorCode.E_BACKEND_VECTOR_CONNECT
        assert ImageErrorCode.E_BACKEND_EMBED_TIMEOUT
        assert ImageErrorCode.E_BACKEND_EMBED_CONNECT

    def test_warning_codes_exist(self):
        assert ImageErrorCode.W_IMAGE_VLM_LOW_DETAIL
        assert ImageErrorCode.W_IMAGE_VLM_RETRY


@pytest.mark.unit
class TestImageIngestError:
    """Test ImageIngestError model."""

    def test_extends_base_ingest_error(self):
        assert issubclass(ImageIngestError, BaseIngestError)

    def test_construction(self):
        err = ImageIngestError(
            code=ImageErrorCode.E_IMAGE_CORRUPT.value,
            message="File is corrupt",
            stage="security",
            file_path="/tmp/bad.png",
        )
        assert err.code == "E_IMAGE_CORRUPT"
        assert err.message == "File is corrupt"
        assert err.stage == "security"
        assert err.file_path == "/tmp/bad.png"
        assert err.recoverable is False

    def test_warning_is_recoverable(self):
        err = ImageIngestError(
            code=ImageErrorCode.W_IMAGE_VLM_LOW_DETAIL.value,
            message="Caption too short",
            stage="caption",
            recoverable=True,
        )
        assert err.recoverable is True
        assert err.file_path is None

    def test_default_file_path_is_none(self):
        err = ImageIngestError(
            code="E_IMAGE_CORRUPT",
            message="test",
        )
        assert err.file_path is None
