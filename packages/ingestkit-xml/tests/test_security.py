"""Unit tests for ingestkit_xml.security -- pre-flight scanner."""

from __future__ import annotations

import pytest

from ingestkit_xml.config import XMLProcessorConfig
from ingestkit_xml.errors import ErrorCode
from ingestkit_xml.security import XMLSecurityScanner


@pytest.fixture
def scanner(default_config) -> XMLSecurityScanner:
    return XMLSecurityScanner(default_config)


class TestExtensionCheck:
    """Tests for file extension validation."""

    def test_wrong_extension(self, scanner, tmp_path):
        fp = tmp_path / "data.txt"
        fp.write_text("<root/>")
        errors = scanner.scan(str(fp))
        assert len(errors) == 1
        assert errors[0].code == ErrorCode.E_SECURITY_BAD_EXTENSION

    def test_xml_extension_accepted(self, scanner, tmp_xml_file):
        fp = tmp_xml_file("<root><item>test</item></root>")
        errors = scanner.scan(fp)
        fatal = [e for e in errors if e.code.startswith("E_")]
        assert len(fatal) == 0


class TestFileNotFound:
    """Tests for non-existent file handling."""

    def test_nonexistent_file(self, scanner, tmp_path):
        fp = str(tmp_path / "nonexistent.xml")
        errors = scanner.scan(fp)
        assert any(e.code == ErrorCode.E_PARSE_CORRUPT for e in errors)


class TestEmptyFile:
    """Tests for empty file detection."""

    def test_empty_file(self, scanner, tmp_path):
        fp = tmp_path / "empty.xml"
        fp.write_text("")
        errors = scanner.scan(str(fp))
        assert any(e.code == ErrorCode.E_PARSE_EMPTY for e in errors)


class TestFileSizeLimit:
    """Tests for file size limit."""

    def test_too_large(self, tmp_path):
        config = XMLProcessorConfig(max_file_size_mb=0)
        scanner = XMLSecurityScanner(config)
        fp = tmp_path / "big.xml"
        fp.write_text("<root>data</root>")
        errors = scanner.scan(str(fp))
        assert any(e.code == ErrorCode.E_SECURITY_TOO_LARGE for e in errors)

    def test_large_file_warning(self, tmp_path):
        """File > 10MB but under limit should produce a warning."""
        config = XMLProcessorConfig(max_file_size_mb=200)
        scanner = XMLSecurityScanner(config)
        fp = tmp_path / "medium.xml"
        # Write ~11 MB of XML
        fp.write_text("<root>" + "x" * (11 * 1024 * 1024) + "</root>")
        errors = scanner.scan(str(fp))
        warning_codes = [e.code for e in errors if e.recoverable]
        assert ErrorCode.W_LARGE_FILE in warning_codes


class TestEntityDeclaration:
    """Tests for entity declaration scan (billion laughs / XXE prevention)."""

    def test_entity_declaration_rejected(self, tmp_path):
        fp = tmp_path / "entity.xml"
        fp.write_text(
            '<?xml version="1.0"?>\n'
            '<!DOCTYPE foo [\n'
            '  <!ENTITY xxe "malicious">\n'
            ']>\n'
            '<root>&xxe;</root>'
        )
        scanner = XMLSecurityScanner(XMLProcessorConfig())
        errors = scanner.scan(str(fp))
        assert any(e.code == ErrorCode.E_SECURITY_ENTITY_DECLARATION for e in errors)

    def test_doctype_with_internal_subset_rejected(self, tmp_path):
        fp = tmp_path / "doctype.xml"
        fp.write_text(
            '<?xml version="1.0"?>\n'
            '<!DOCTYPE root [\n'
            '  <!ELEMENT root (#PCDATA)>\n'
            ']>\n'
            '<root>data</root>'
        )
        scanner = XMLSecurityScanner(XMLProcessorConfig())
        errors = scanner.scan(str(fp))
        assert any(e.code == ErrorCode.E_SECURITY_ENTITY_DECLARATION for e in errors)

    def test_billion_laughs_payload_rejected(self, tmp_path):
        fp = tmp_path / "laughs.xml"
        fp.write_text(
            '<?xml version="1.0"?>\n'
            '<!DOCTYPE lolz [\n'
            '  <!ENTITY lol "lol">\n'
            '  <!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">\n'
            ']>\n'
            '<root>&lol2;</root>'
        )
        scanner = XMLSecurityScanner(XMLProcessorConfig())
        errors = scanner.scan(str(fp))
        assert any(e.code == ErrorCode.E_SECURITY_ENTITY_DECLARATION for e in errors)


class TestInvalidXML:
    """Tests for invalid XML detection."""

    def test_invalid_xml(self, scanner, tmp_path):
        fp = tmp_path / "bad.xml"
        fp.write_text("<root><unclosed>")
        errors = scanner.scan(str(fp))
        assert any(e.code == ErrorCode.E_SECURITY_INVALID_XML for e in errors)


class TestDepthBomb:
    """Tests for nesting depth bomb protection."""

    def test_deeply_nested(self, tmp_path):
        config = XMLProcessorConfig(max_depth=3)
        scanner = XMLSecurityScanner(config)
        # Build deeply nested XML: depth 5
        xml = "<a><b><c><d><e>deep</e></d></c></b></a>"
        fp = tmp_path / "deep.xml"
        fp.write_text(xml)
        errors = scanner.scan(str(fp))
        assert any(e.code == ErrorCode.E_SECURITY_DEPTH_BOMB for e in errors)


class TestValidFile:
    """Tests for valid XML files passing all checks."""

    def test_valid_xml_passes(self, scanner, tmp_xml_file):
        fp = tmp_xml_file("<root><item>Test</item></root>")
        errors = scanner.scan(fp)
        fatal = [e for e in errors if e.code.startswith("E_")]
        assert len(fatal) == 0
