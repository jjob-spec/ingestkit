"""Tests for ImageProcessorConfig."""

from __future__ import annotations

import json
import os

import pytest
import yaml

from ingestkit_image.config import ImageProcessorConfig


@pytest.mark.unit
class TestImageProcessorConfigDefaults:
    """Test default configuration values."""

    def test_default_parser_version(self):
        config = ImageProcessorConfig()
        assert config.parser_version == "ingestkit_image:1.0.0"

    def test_default_tenant_id_is_none(self):
        config = ImageProcessorConfig()
        assert config.tenant_id is None

    def test_default_max_file_size(self):
        config = ImageProcessorConfig()
        assert config.max_file_size_mb == 50

    def test_default_vision_model(self):
        config = ImageProcessorConfig()
        assert config.vision_model == "llama3.2-vision:11b"

    def test_default_supported_formats(self):
        config = ImageProcessorConfig()
        assert "jpeg" in config.supported_formats
        assert "png" in config.supported_formats
        assert "tiff" in config.supported_formats
        assert "webp" in config.supported_formats
        assert "bmp" in config.supported_formats
        assert "gif" in config.supported_formats

    def test_default_embedding_model(self):
        config = ImageProcessorConfig()
        assert config.embedding_model == "nomic-embed-text"
        assert config.embedding_dimension == 768

    def test_default_collection(self):
        config = ImageProcessorConfig()
        assert config.default_collection == "helpdesk"

    def test_default_log_flags_false(self):
        config = ImageProcessorConfig()
        assert config.log_sample_data is False
        assert config.log_captions is False


@pytest.mark.unit
class TestImageProcessorConfigCustom:
    """Test custom configuration values."""

    def test_custom_tenant_id(self):
        config = ImageProcessorConfig(tenant_id="acme-corp")
        assert config.tenant_id == "acme-corp"

    def test_custom_vision_model(self):
        config = ImageProcessorConfig(vision_model="custom-vlm:7b")
        assert config.vision_model == "custom-vlm:7b"

    def test_custom_max_file_size(self):
        config = ImageProcessorConfig(max_file_size_mb=100)
        assert config.max_file_size_mb == 100


@pytest.mark.unit
class TestImageProcessorConfigFromFile:
    """Test from_file() classmethod."""

    def test_from_yaml_file(self, tmp_path):
        yaml_data = {"vision_model": "test-model:1b", "max_file_size_mb": 25}
        yaml_path = str(tmp_path / "config.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f)

        config = ImageProcessorConfig.from_file(yaml_path)
        assert config.vision_model == "test-model:1b"
        assert config.max_file_size_mb == 25
        # defaults preserved
        assert config.embedding_model == "nomic-embed-text"

    def test_from_json_file(self, tmp_path):
        json_data = {"tenant_id": "test-tenant", "vlm_temperature": 0.5}
        json_path = str(tmp_path / "config.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f)

        config = ImageProcessorConfig.from_file(json_path)
        assert config.tenant_id == "test-tenant"
        assert config.vlm_temperature == 0.5

    def test_from_yml_extension(self, tmp_path):
        yaml_data = {"max_image_width": 5000}
        yml_path = str(tmp_path / "config.yml")
        with open(yml_path, "w") as f:
            yaml.dump(yaml_data, f)

        config = ImageProcessorConfig.from_file(yml_path)
        assert config.max_image_width == 5000

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            ImageProcessorConfig.from_file("/nonexistent/config.yaml")

    def test_unsupported_extension_raises(self, tmp_path):
        txt_path = str(tmp_path / "config.txt")
        with open(txt_path, "w") as f:
            f.write("hello")

        with pytest.raises(ValueError, match="Unsupported config file extension"):
            ImageProcessorConfig.from_file(txt_path)

    def test_empty_yaml_returns_defaults(self, tmp_path):
        yaml_path = str(tmp_path / "empty.yaml")
        with open(yaml_path, "w") as f:
            f.write("")

        config = ImageProcessorConfig.from_file(yaml_path)
        assert config.vision_model == "llama3.2-vision:11b"
