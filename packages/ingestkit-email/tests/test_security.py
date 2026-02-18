"""Tests for ingestkit_email.security."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from ingestkit_email.config import EmailProcessorConfig
from ingestkit_email.errors import ErrorCode
from ingestkit_email.security import EmailSecurityScanner, _OLE2_MAGIC


class TestEmailSecurityScanner:
    def setup_method(self):
        self.config = EmailProcessorConfig()
        self.scanner = EmailSecurityScanner(self.config)

    def test_valid_eml_passes(self, sample_eml_file):
        """Valid EML produces no errors."""
        errors = self.scanner.scan(sample_eml_file)
        assert len(errors) == 0

    def test_unsupported_extension(self, tmp_path):
        """Non-email extension produces E_EMAIL_UNSUPPORTED_FORMAT."""
        p = tmp_path / "test.txt"
        p.write_text("hello")
        errors = self.scanner.scan(str(p))
        assert len(errors) == 1
        assert errors[0].code == ErrorCode.E_EMAIL_UNSUPPORTED_FORMAT

    def test_file_too_large(self, tmp_path):
        """File over limit produces E_EMAIL_TOO_LARGE."""
        config = EmailProcessorConfig(max_file_size_mb=0)  # 0 MB = reject everything
        scanner = EmailSecurityScanner(config)
        p = tmp_path / "big.eml"
        p.write_bytes(b"x" * 100)
        errors = scanner.scan(str(p))
        assert any(e.code == ErrorCode.E_EMAIL_TOO_LARGE for e in errors)

    def test_empty_file(self, tmp_path):
        """0 bytes produces E_EMAIL_FILE_CORRUPT."""
        p = tmp_path / "empty.eml"
        p.write_bytes(b"")
        errors = self.scanner.scan(str(p))
        assert len(errors) == 1
        assert errors[0].code == ErrorCode.E_EMAIL_FILE_CORRUPT

    def test_msg_magic_valid(self, tmp_path):
        """MSG with valid OLE2 header passes magic check."""
        p = tmp_path / "test.msg"
        p.write_bytes(_OLE2_MAGIC + b"\x00" * 100)
        errors = self.scanner.scan(str(p))
        # Should not have magic-related errors (may have MSG unavailable if
        # extract-msg not installed, but no corrupt error)
        corrupt_errors = [e for e in errors if e.code == ErrorCode.E_EMAIL_FILE_CORRUPT]
        assert len(corrupt_errors) == 0

    def test_msg_magic_invalid(self, tmp_path):
        """MSG with wrong magic bytes produces E_EMAIL_FILE_CORRUPT."""
        p = tmp_path / "bad.msg"
        p.write_bytes(b"NOT_OLE2" + b"\x00" * 100)
        errors = self.scanner.scan(str(p))
        assert any(e.code == ErrorCode.E_EMAIL_FILE_CORRUPT for e in errors)

    def test_eml_no_magic_check(self, tmp_path):
        """EML skips magic byte check (text-based format)."""
        p = tmp_path / "test.eml"
        p.write_bytes(b"From: test\nSubject: test\n\nBody")
        errors = self.scanner.scan(str(p))
        # No corrupt errors -- EML has no magic requirement
        corrupt_errors = [e for e in errors if e.code == ErrorCode.E_EMAIL_FILE_CORRUPT]
        assert len(corrupt_errors) == 0

    def test_msg_without_library(self, tmp_path):
        """Missing extract-msg produces E_EMAIL_MSG_UNAVAILABLE."""
        p = tmp_path / "test.msg"
        p.write_bytes(_OLE2_MAGIC + b"\x00" * 100)
        with patch.dict(sys.modules, {"extract_msg": None}):
            errors = self.scanner.scan(str(p))
        assert any(e.code == ErrorCode.E_EMAIL_MSG_UNAVAILABLE for e in errors)
