"""Unit tests for ingestkit_xml.config -- configuration model."""

from __future__ import annotations

import json

import pytest

from ingestkit_xml.config import XMLProcessorConfig


class TestDefaults:
    """Test that default values are correct."""

    def test_parser_version(self):
        config = XMLProcessorConfig()
        assert config.parser_version == "ingestkit_xml:1.0.0"

    def test_tenant_id_none(self):
        config = XMLProcessorConfig()
        assert config.tenant_id is None

    def test_max_file_size_mb(self):
        config = XMLProcessorConfig()
        assert config.max_file_size_mb == 100

    def test_max_depth(self):
        config = XMLProcessorConfig()
        assert config.max_depth == 100

    def test_max_elements(self):
        config = XMLProcessorConfig()
        assert config.max_elements == 500_000

    def test_chunk_size_tokens(self):
        config = XMLProcessorConfig()
        assert config.chunk_size_tokens == 512

    def test_strip_namespaces_default(self):
        config = XMLProcessorConfig()
        assert config.strip_namespaces is True

    def test_include_attributes_default(self):
        config = XMLProcessorConfig()
        assert config.include_attributes is True

    def test_skip_attribute_prefixes_default(self):
        config = XMLProcessorConfig()
        assert config.skip_attribute_prefixes == ["xmlns", "xsi"]

    def test_include_tail_text_default(self):
        config = XMLProcessorConfig()
        assert config.include_tail_text is True

    def test_indent_text_default(self):
        config = XMLProcessorConfig()
        assert config.indent_text is False


class TestFromFile:
    """Test loading configuration from files."""

    def test_from_json_file(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"max_depth": 50, "tenant_id": "t-1"}))
        config = XMLProcessorConfig.from_file(str(config_file))
        assert config.max_depth == 50
        assert config.tenant_id == "t-1"

    def test_from_yaml_file(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("max_depth: 25\ntenant_id: t-2\n")
        config = XMLProcessorConfig.from_file(str(config_file))
        assert config.max_depth == 25
        assert config.tenant_id == "t-2"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            XMLProcessorConfig.from_file("/nonexistent/config.json")

    def test_unsupported_extension_raises(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text("max_depth = 50")
        with pytest.raises(ValueError, match="Unsupported config file extension"):
            XMLProcessorConfig.from_file(str(config_file))
