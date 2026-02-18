"""Tests for ingestkit_email.converters.msg."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from ingestkit_email.config import EmailProcessorConfig
from ingestkit_email.converters.msg import MSGConverter


class TestMSGConverter:
    def setup_method(self):
        self.converter = MSGConverter()
        self.config = EmailProcessorConfig()

    def test_convert_basic(self, tmp_path):
        """Mock extract_msg.Message produces correct EmailContent."""
        mock_msg = MagicMock()
        mock_msg.sender = "alice@example.com"
        mock_msg.to = "bob@example.com"
        mock_msg.cc = None
        mock_msg.date = "2026-02-17"
        mock_msg.subject = "Test MSG"
        mock_msg.messageId = "<msg-001>"
        mock_msg.body = "Hello from Outlook."
        mock_msg.htmlBody = None
        mock_msg.attachments = []
        mock_msg.close = MagicMock()

        mock_module = MagicMock()
        mock_module.Message.return_value = mock_msg

        with patch.dict(sys.modules, {"extract_msg": mock_module}):
            result = self.converter.convert("/fake/test.msg", self.config)

        assert result.from_address == "alice@example.com"
        assert result.to_address == "bob@example.com"
        assert result.subject == "Test MSG"
        assert result.body_text == "Hello from Outlook."
        assert result.body_source == "plain"
        mock_msg.close.assert_called_once()

    def test_headers_extracted(self, tmp_path):
        """Header attributes mapped correctly."""
        mock_msg = MagicMock()
        mock_msg.sender = "x@y.com"
        mock_msg.to = "a@b.com"
        mock_msg.cc = "cc@cc.com"
        mock_msg.date = "2026-01-01"
        mock_msg.subject = "Headers Test"
        mock_msg.messageId = "<hdr-001>"
        mock_msg.body = "body"
        mock_msg.htmlBody = None
        mock_msg.attachments = []
        mock_msg.close = MagicMock()

        mock_module = MagicMock()
        mock_module.Message.return_value = mock_msg

        with patch.dict(sys.modules, {"extract_msg": mock_module}):
            result = self.converter.convert("/fake/test.msg", self.config)

        assert result.from_address == "x@y.com"
        assert result.cc_address == "cc@cc.com"
        assert result.message_id == "<hdr-001>"
        assert "From" in result.raw_headers
        assert "Subject" in result.raw_headers

    def test_import_error_message(self):
        """Missing extract-msg produces helpful ImportError."""
        # Temporarily hide extract_msg
        with patch.dict(sys.modules, {"extract_msg": None}):
            with pytest.raises(ImportError, match="extract-msg"):
                self.converter.convert("/fake/test.msg", self.config)
