"""Tests for ingestkit_doc.security."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from ingestkit_doc.config import DocProcessorConfig
from ingestkit_doc.errors import ErrorCode
from ingestkit_doc.security import DocSecurityScanner

_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


@pytest.fixture
def scanner(default_config: DocProcessorConfig) -> DocSecurityScanner:
    return DocSecurityScanner(default_config)


class TestExtensionCheck:
    """Extension whitelist."""

    def test_doc_accepted(self, scanner, tmp_doc_file):
        path = tmp_doc_file()
        errors = scanner.scan(path)
        fatal = [e for e in errors if e.code.startswith("E_")]
        assert not fatal

    def test_docx_rejected(self, scanner, tmp_path):
        path = tmp_path / "test.docx"
        path.write_bytes(b"dummy")
        errors = scanner.scan(str(path))
        assert len(errors) == 1
        assert errors[0].code == ErrorCode.E_SECURITY_BAD_EXTENSION

    def test_txt_rejected(self, scanner, tmp_path):
        path = tmp_path / "test.txt"
        path.write_bytes(b"dummy")
        errors = scanner.scan(str(path))
        assert errors[0].code == ErrorCode.E_SECURITY_BAD_EXTENSION

    def test_msg_rejected(self, scanner, tmp_path):
        path = tmp_path / "test.msg"
        path.write_bytes(b"dummy")
        errors = scanner.scan(str(path))
        assert errors[0].code == ErrorCode.E_SECURITY_BAD_EXTENSION

    def test_uppercase_doc_accepted(self, scanner, tmp_path):
        path = tmp_path / "test.DOC"
        path.write_bytes(_OLE2_MAGIC + b"\x00" * 504)
        errors = scanner.scan(str(path))
        fatal = [e for e in errors if e.code.startswith("E_")]
        assert not fatal


class TestFileExistence:
    """File existence and readability."""

    def test_missing_file(self, scanner):
        errors = scanner.scan("/nonexistent/file.doc")
        assert len(errors) == 1
        assert errors[0].code == ErrorCode.E_PARSE_CORRUPT


class TestEmptyFile:
    """Empty file detection."""

    def test_empty_file(self, scanner, tmp_path):
        path = tmp_path / "empty.doc"
        path.write_bytes(b"")
        errors = scanner.scan(str(path))
        assert len(errors) == 1
        assert errors[0].code == ErrorCode.E_PARSE_EMPTY


class TestFileSize:
    """File size limits."""

    def test_oversized_file(self, tmp_path):
        config = DocProcessorConfig(max_file_size_mb=1)
        scanner = DocSecurityScanner(config)
        path = tmp_path / "big.doc"
        # Write just over 1 MB (with OLE2 header)
        path.write_bytes(_OLE2_MAGIC + b"\x00" * (1024 * 1024 + 1))
        errors = scanner.scan(str(path))
        fatal = [e for e in errors if e.code.startswith("E_")]
        assert any(e.code == ErrorCode.E_SECURITY_TOO_LARGE for e in fatal)

    def test_large_file_warning(self, scanner, tmp_path):
        path = tmp_path / "large.doc"
        # Write 11 MB (OLE2 header + padding)
        path.write_bytes(_OLE2_MAGIC + b"\x00" * (11 * 1024 * 1024))
        errors = scanner.scan(str(path))
        warnings = [e for e in errors if e.code.startswith("W_")]
        assert any(e.code == ErrorCode.W_LARGE_FILE for e in warnings)


class TestMagicBytes:
    """OLE2 magic byte validation."""

    def test_valid_magic_passes(self, scanner, tmp_doc_file):
        path = tmp_doc_file()
        errors = scanner.scan(path)
        fatal = [e for e in errors if e.code.startswith("E_")]
        assert not fatal

    def test_invalid_magic_rejected(self, scanner, tmp_path):
        path = tmp_path / "bad.doc"
        path.write_bytes(b"PK\x03\x04" + b"\x00" * 508)  # ZIP magic, not OLE2
        errors = scanner.scan(str(path))
        assert any(e.code == ErrorCode.E_SECURITY_BAD_MAGIC for e in errors)


class TestMammothAvailability:
    """mammoth import guard."""

    def test_mammoth_unavailable(self, scanner, tmp_doc_file):
        path = tmp_doc_file()
        # Temporarily remove mammoth from importable modules
        with patch.dict(sys.modules, {"mammoth": None}):
            # Also need to make the import fail
            import builtins
            real_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "mammoth":
                    raise ImportError("No module named 'mammoth'")
                return real_import(name, *args, **kwargs)

            with patch.object(builtins, "__import__", side_effect=mock_import):
                errors = scanner.scan(path)
                assert any(
                    e.code == ErrorCode.E_DOC_MAMMOTH_UNAVAILABLE for e in errors
                )
