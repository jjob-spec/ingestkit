"""Tests for ingestkit_email.converters.eml."""

from __future__ import annotations

from pathlib import Path

import pytest

from ingestkit_email.config import EmailProcessorConfig
from ingestkit_email.converters.eml import EMLConverter


class TestEMLConverter:
    def setup_method(self):
        self.converter = EMLConverter()
        self.config = EmailProcessorConfig()

    def test_convert_plain_text(self, sample_eml_file):
        """Plain EML produces correct EmailContent."""
        result = self.converter.convert(sample_eml_file, self.config)
        assert "test email body" in result.body_text
        assert result.body_source == "plain"

    def test_convert_html_only(self, sample_eml_html_file):
        """HTML-only EML converts body and sets body_source='html_converted'."""
        result = self.converter.convert(sample_eml_html_file, self.config)
        assert "HTML" in result.body_text or "body" in result.body_text.lower()
        assert result.body_source == "html_converted"

    def test_multipart_prefers_plain(self, sample_eml_multipart_file):
        """With both parts, plain text is preferred when configured."""
        result = self.converter.convert(sample_eml_multipart_file, self.config)
        assert result.body_source == "plain"
        assert "test email body" in result.body_text

    def test_multipart_html_fallback(self, sample_eml_multipart_file):
        """When prefer_plain_text=False, HTML is preferred if available."""
        config = EmailProcessorConfig(prefer_plain_text=False)
        result = self.converter.convert(sample_eml_multipart_file, config)
        # With multipart containing both, and prefer_plain_text=False,
        # it should use html_converted since html parts exist
        assert result.body_source == "html_converted"

    def test_headers_extracted(self, sample_eml_file):
        """From/To/Date/Subject/Message-ID populated."""
        result = self.converter.convert(sample_eml_file, self.config)
        assert result.from_address == "sender@example.com"
        assert result.to_address == "recipient@example.com"
        assert result.subject == "Test Subject"
        assert result.message_id == "<test-123@example.com>"
        assert result.date is not None

    def test_attachments_listed(self, sample_eml_multipart_file):
        """Binary attachment names are collected."""
        result = self.converter.convert(sample_eml_multipart_file, self.config)
        assert "image.png" in result.attachment_names

    def test_empty_email(self, tmp_path):
        """Email with no body produces empty body_text."""
        from tests.conftest import _build_eml_bytes

        eml_bytes = _build_eml_bytes(plain="", html=None)
        p = tmp_path / "empty.eml"
        p.write_bytes(eml_bytes)

        result = self.converter.convert(str(p), self.config)
        assert result.body_text == ""

    def test_malformed_eml(self, tmp_path):
        """Garbage input is handled gracefully (no crash)."""
        p = tmp_path / "garbage.eml"
        p.write_bytes(b"This is not a valid email at all\x00\xff\xfe")

        result = self.converter.convert(str(p), self.config)
        # Should not raise -- just return whatever it can parse
        assert isinstance(result.body_text, str)
