"""Tests for ingestkit_rtf.security."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from ingestkit_rtf.config import RTFProcessorConfig
from ingestkit_rtf.errors import ErrorCode
from ingestkit_rtf.security import RTFSecurityScanner

_RTF_MAGIC = b"{\\rtf"


@pytest.fixture
def scanner(default_config: RTFProcessorConfig) -> RTFSecurityScanner:
    return RTFSecurityScanner(default_config)


class TestExtensionCheck:
    """Extension whitelist."""

    def test_rtf_accepted(self, scanner, tmp_rtf_file):
        path = tmp_rtf_file()
        errors = scanner.scan(path)
        fatal = [e for e in errors if e.code.startswith("E_")]
        assert not fatal

    def test_doc_rejected(self, scanner, tmp_path):
        path = tmp_path / "test.doc"
        path.write_bytes(b"dummy")
        errors = scanner.scan(str(path))
        assert len(errors) == 1
        assert errors[0].code == ErrorCode.E_SECURITY_BAD_EXTENSION

    def test_txt_rejected(self, scanner, tmp_path):
        path = tmp_path / "test.txt"
        path.write_bytes(b"dummy")
        errors = scanner.scan(str(path))
        assert errors[0].code == ErrorCode.E_SECURITY_BAD_EXTENSION

    def test_pdf_rejected(self, scanner, tmp_path):
        path = tmp_path / "test.pdf"
        path.write_bytes(b"dummy")
        errors = scanner.scan(str(path))
        assert errors[0].code == ErrorCode.E_SECURITY_BAD_EXTENSION

    def test_uppercase_rtf_accepted(self, scanner, tmp_path):
        path = tmp_path / "test.RTF"
        path.write_bytes(_RTF_MAGIC + b"1\\ansi Hello.}")
        errors = scanner.scan(str(path))
        fatal = [e for e in errors if e.code.startswith("E_")]
        assert not fatal


class TestFileExistence:
    """File existence and readability."""

    def test_missing_file(self, scanner):
        errors = scanner.scan("/nonexistent/file.rtf")
        assert len(errors) == 1
        assert errors[0].code == ErrorCode.E_PARSE_CORRUPT


class TestEmptyFile:
    """Empty file detection."""

    def test_empty_file(self, scanner, tmp_path):
        path = tmp_path / "empty.rtf"
        path.write_bytes(b"")
        errors = scanner.scan(str(path))
        assert len(errors) == 1
        assert errors[0].code == ErrorCode.E_PARSE_EMPTY


class TestFileSize:
    """File size limits."""

    def test_oversized_file(self, tmp_path):
        config = RTFProcessorConfig(max_file_size_mb=1)
        scanner = RTFSecurityScanner(config)
        path = tmp_path / "big.rtf"
        # Write just over 1 MB (with RTF header)
        path.write_bytes(_RTF_MAGIC + b"1\\ansi " + b"\x00" * (1024 * 1024 + 1) + b"}")
        errors = scanner.scan(str(path))
        fatal = [e for e in errors if e.code.startswith("E_")]
        assert any(e.code == ErrorCode.E_SECURITY_TOO_LARGE for e in fatal)

    def test_large_file_warning(self, scanner, tmp_path):
        path = tmp_path / "large.rtf"
        # Write 11 MB (RTF header + padding)
        path.write_bytes(_RTF_MAGIC + b"1\\ansi " + b"\x00" * (11 * 1024 * 1024) + b"}")
        errors = scanner.scan(str(path))
        warnings = [e for e in errors if e.code.startswith("W_")]
        assert any(e.code == ErrorCode.W_LARGE_FILE for e in warnings)


class TestMagicBytes:
    """RTF magic byte validation."""

    def test_valid_magic_passes(self, scanner, tmp_rtf_file):
        path = tmp_rtf_file()
        errors = scanner.scan(path)
        fatal = [e for e in errors if e.code.startswith("E_")]
        assert not fatal

    def test_invalid_magic_rejected(self, scanner, tmp_path):
        path = tmp_path / "bad.rtf"
        path.write_bytes(b"PK\x03\x04" + b"\x00" * 508)  # ZIP magic, not RTF
        errors = scanner.scan(str(path))
        assert any(e.code == ErrorCode.E_SECURITY_BAD_MAGIC for e in errors)


class TestStriprtfAvailability:
    """striprtf import guard."""

    def test_striprtf_unavailable(self, scanner, tmp_rtf_file):
        path = tmp_rtf_file()
        # Temporarily make striprtf unimportable
        with patch.dict(sys.modules, {"striprtf": None, "striprtf.striprtf": None}):
            import builtins
            real_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "striprtf.striprtf" or name == "striprtf":
                    raise ImportError("No module named 'striprtf'")
                return real_import(name, *args, **kwargs)

            with patch.object(builtins, "__import__", side_effect=mock_import):
                errors = scanner.scan(path)
                assert any(
                    e.code == ErrorCode.E_RTF_STRIPRTF_UNAVAILABLE for e in errors
                )
