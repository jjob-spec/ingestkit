"""Tests for ingestkit_email.config."""

from __future__ import annotations

import json

import pytest

from ingestkit_email.config import EmailProcessorConfig


class TestEmailProcessorConfig:
    def test_defaults(self):
        """All defaults match spec."""
        cfg = EmailProcessorConfig()
        assert cfg.parser_version == "ingestkit_email:1.0.0"
        assert cfg.tenant_id is None
        assert cfg.prefer_plain_text is True
        assert cfg.include_headers is True
        assert cfg.header_format == "key_value"
        assert cfg.skip_attachments is True
        assert cfg.max_file_size_mb == 50
        assert cfg.embedding_model == "nomic-embed-text"
        assert cfg.embedding_dimension == 768
        assert cfg.embedding_batch_size == 64
        assert cfg.default_collection == "helpdesk"
        assert cfg.backend_timeout_seconds == 30.0
        assert cfg.backend_max_retries == 2
        assert cfg.log_sample_data is False
        assert cfg.redact_patterns == []

    def test_from_file_json(self, tmp_path):
        """JSON override loads correctly."""
        data = {"tenant_id": "acme", "max_file_size_mb": 100}
        p = tmp_path / "config.json"
        p.write_text(json.dumps(data))

        cfg = EmailProcessorConfig.from_file(str(p))
        assert cfg.tenant_id == "acme"
        assert cfg.max_file_size_mb == 100
        # Defaults preserved
        assert cfg.embedding_model == "nomic-embed-text"

    def test_from_file_yaml(self, tmp_path):
        """YAML override loads correctly."""
        yaml_content = "tenant_id: acme\nmax_file_size_mb: 100\n"
        p = tmp_path / "config.yaml"
        p.write_text(yaml_content)

        cfg = EmailProcessorConfig.from_file(str(p))
        assert cfg.tenant_id == "acme"
        assert cfg.max_file_size_mb == 100

    def test_from_file_not_found(self):
        """Raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            EmailProcessorConfig.from_file("/nonexistent/config.json")

    def test_parser_version_format(self):
        """Parser version starts with 'ingestkit_email:'."""
        cfg = EmailProcessorConfig()
        assert cfg.parser_version.startswith("ingestkit_email:")
