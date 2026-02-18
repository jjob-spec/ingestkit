"""Unit tests for ingestkit_json.security -- pre-flight scanner."""

from __future__ import annotations

import json

import pytest

from ingestkit_json.config import JSONProcessorConfig
from ingestkit_json.errors import ErrorCode
from ingestkit_json.security import JSONSecurityScanner


@pytest.fixture
def scanner(default_config) -> JSONSecurityScanner:
    return JSONSecurityScanner(default_config)


class TestExtensionCheck:
    """Tests for file extension validation."""

    def test_wrong_extension(self, scanner, tmp_path):
        fp = tmp_path / "data.txt"
        fp.write_text('{"a": 1}')
        errors = scanner.scan(str(fp))
        assert len(errors) == 1
        assert errors[0].code == ErrorCode.E_SECURITY_BAD_EXTENSION

    def test_json_extension_accepted(self, scanner, tmp_json_file):
        fp = tmp_json_file({"a": 1})
        errors = scanner.scan(fp)
        fatal = [e for e in errors if e.code.startswith("E_")]
        assert len(fatal) == 0


class TestEmptyFile:
    """Tests for empty file detection."""

    def test_empty_file(self, scanner, tmp_path):
        fp = tmp_path / "empty.json"
        fp.write_text("")
        errors = scanner.scan(str(fp))
        assert any(e.code == ErrorCode.E_PARSE_EMPTY for e in errors)


class TestFileSizeLimit:
    """Tests for file size limit."""

    def test_too_large(self, tmp_path):
        config = JSONProcessorConfig(max_file_size_mb=0)  # 0 MB limit
        scanner = JSONSecurityScanner(config)
        fp = tmp_path / "big.json"
        fp.write_text('{"a": 1}')
        errors = scanner.scan(str(fp))
        assert any(e.code == ErrorCode.E_SECURITY_TOO_LARGE for e in errors)

    def test_large_file_warning(self, tmp_path):
        """File > 10MB but under limit should produce a warning."""
        config = JSONProcessorConfig(max_file_size_mb=200)
        scanner = JSONSecurityScanner(config)
        fp = tmp_path / "medium.json"
        # Write ~11 MB of JSON
        data = {"key": "x" * (11 * 1024 * 1024)}
        with open(fp, "w") as fh:
            json.dump(data, fh)
        errors = scanner.scan(str(fp))
        warning_codes = [e.code for e in errors if e.recoverable]
        assert ErrorCode.W_LARGE_FILE in warning_codes


class TestInvalidJSON:
    """Tests for invalid JSON detection."""

    def test_invalid_json(self, scanner, tmp_path):
        fp = tmp_path / "bad.json"
        fp.write_text("{not valid json}")
        errors = scanner.scan(str(fp))
        assert any(e.code == ErrorCode.E_SECURITY_INVALID_JSON for e in errors)


class TestNestingBomb:
    """Tests for nesting depth bomb protection."""

    def test_deeply_nested(self, tmp_path):
        config = JSONProcessorConfig(max_nesting_depth=5)
        scanner = JSONSecurityScanner(config)
        # Build deeply nested JSON
        data = {"a": {"b": {"c": {"d": {"e": {"f": {"g": "deep"}}}}}}}
        fp = tmp_path / "deep.json"
        with open(fp, "w") as fh:
            json.dump(data, fh)
        errors = scanner.scan(str(fp))
        assert any(e.code == ErrorCode.E_SECURITY_NESTING_BOMB for e in errors)


class TestFileNotFound:
    """Tests for non-existent file handling."""

    def test_nonexistent_file(self, scanner, tmp_path):
        fp = str(tmp_path / "nonexistent.json")
        errors = scanner.scan(fp)
        assert any(e.code == ErrorCode.E_PARSE_CORRUPT for e in errors)


class TestValidFile:
    """Tests for valid JSON files passing all checks."""

    def test_valid_json_passes(self, scanner, tmp_json_file):
        fp = tmp_json_file({"name": "test", "value": 42})
        errors = scanner.scan(fp)
        fatal = [e for e in errors if e.code.startswith("E_")]
        assert len(fatal) == 0
