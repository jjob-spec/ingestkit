"""Tests for ingestkit_rtf.config."""

from __future__ import annotations

import json

import pytest

from ingestkit_rtf.config import RTFProcessorConfig


class TestDefaults:
    """Default config values."""

    def test_parser_version(self):
        config = RTFProcessorConfig()
        assert config.parser_version == "ingestkit_rtf:1.0.0"

    def test_tenant_id_none(self):
        config = RTFProcessorConfig()
        assert config.tenant_id is None

    def test_chunk_size(self):
        config = RTFProcessorConfig()
        assert config.chunk_size_tokens == 512

    def test_embedding_model(self):
        config = RTFProcessorConfig()
        assert config.embedding_model == "nomic-embed-text"


class TestFromFile:
    """Loading config from files."""

    def test_from_json(self, tmp_path):
        config_data = {"tenant_id": "test-tenant", "chunk_size_tokens": 256}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = RTFProcessorConfig.from_file(str(config_file))
        assert config.tenant_id == "test-tenant"
        assert config.chunk_size_tokens == 256
        # Defaults preserved
        assert config.embedding_model == "nomic-embed-text"

    def test_from_yaml(self, tmp_path):
        yaml = pytest.importorskip("yaml")
        config_data = {"tenant_id": "yaml-tenant", "max_file_size_mb": 50}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        config = RTFProcessorConfig.from_file(str(config_file))
        assert config.tenant_id == "yaml-tenant"
        assert config.max_file_size_mb == 50

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            RTFProcessorConfig.from_file("/nonexistent/config.json")

    def test_bad_extension_raises(self, tmp_path):
        config_file = tmp_path / "config.txt"
        config_file.write_text("{}")

        with pytest.raises(ValueError, match="Unsupported config file extension"):
            RTFProcessorConfig.from_file(str(config_file))
