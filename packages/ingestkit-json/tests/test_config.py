"""Unit tests for ingestkit_json.config."""

from __future__ import annotations

import json

import pytest

from ingestkit_json.config import JSONProcessorConfig


class TestDefaults:
    """Tests for default configuration values."""

    def test_defaults(self):
        config = JSONProcessorConfig()
        assert config.parser_version == "ingestkit_json:1.0.0"
        assert config.tenant_id is None
        assert config.max_file_size_mb == 100
        assert config.max_nesting_depth == 100
        assert config.max_keys == 500_000
        assert config.chunk_size_tokens == 512
        assert config.chunk_overlap_tokens == 50
        assert config.embedding_model == "nomic-embed-text"
        assert config.embedding_dimension == 768
        assert config.embedding_batch_size == 64
        assert config.default_collection == "helpdesk"
        assert config.backend_timeout_seconds == 30.0
        assert config.backend_max_retries == 2
        assert config.backend_backoff_base == 1.0
        assert config.log_sample_data is False
        assert config.redact_patterns == []
        assert config.array_index_notation is True
        assert config.include_null_values is False
        assert config.path_separator == "."


class TestCustomOverrides:
    """Tests for custom config overrides."""

    def test_overrides(self):
        config = JSONProcessorConfig(
            tenant_id="t1",
            max_file_size_mb=50,
            chunk_size_tokens=256,
        )
        assert config.tenant_id == "t1"
        assert config.max_file_size_mb == 50
        assert config.chunk_size_tokens == 256


class TestFromFile:
    """Tests for from_file() classmethod."""

    def test_from_json_file(self, tmp_path):
        config_data = {"tenant_id": "test-tenant", "max_file_size_mb": 50}
        fp = tmp_path / "config.json"
        with open(fp, "w") as fh:
            json.dump(config_data, fh)

        config = JSONProcessorConfig.from_file(str(fp))
        assert config.tenant_id == "test-tenant"
        assert config.max_file_size_mb == 50
        # Defaults preserved
        assert config.chunk_size_tokens == 512

    def test_from_yaml_file(self, tmp_path):
        yaml = pytest.importorskip("yaml")
        config_data = {"tenant_id": "yaml-tenant", "max_nesting_depth": 50}
        fp = tmp_path / "config.yaml"
        with open(fp, "w") as fh:
            yaml.dump(config_data, fh)

        config = JSONProcessorConfig.from_file(str(fp))
        assert config.tenant_id == "yaml-tenant"
        assert config.max_nesting_depth == 50

    def test_from_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            JSONProcessorConfig.from_file(str(tmp_path / "missing.json"))

    def test_from_file_unsupported_extension(self, tmp_path):
        fp = tmp_path / "config.toml"
        fp.write_text("")
        with pytest.raises(ValueError, match="Unsupported"):
            JSONProcessorConfig.from_file(str(fp))
