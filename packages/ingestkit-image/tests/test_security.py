"""Tests for ImageSecurityScanner."""

from __future__ import annotations

import os
import struct

import pytest
from PIL import Image

from ingestkit_image.config import ImageProcessorConfig
from ingestkit_image.errors import ImageErrorCode
from ingestkit_image.security import ImageSecurityScanner


@pytest.mark.unit
class TestImageSecurityScanner:
    """Test ImageSecurityScanner validation checks."""

    def test_valid_png_passes(self, sample_image_path, image_config):
        scanner = ImageSecurityScanner(image_config)
        metadata, errors = scanner.scan(sample_image_path)

        assert metadata is not None
        assert len(errors) == 0
        assert metadata.width == 100
        assert metadata.height == 100
        assert metadata.image_type.value == "png"

    def test_valid_jpeg_passes(self, sample_jpeg_path, image_config):
        scanner = ImageSecurityScanner(image_config)
        metadata, errors = scanner.scan(sample_jpeg_path)

        assert metadata is not None
        assert len(errors) == 0
        assert metadata.image_type.value == "jpeg"

    def test_unsupported_extension_rejected(self, tmp_path, image_config):
        # Create a file with unsupported extension
        bad_path = str(tmp_path / "test.svg")
        with open(bad_path, "w") as f:
            f.write("<svg></svg>")

        scanner = ImageSecurityScanner(image_config)
        metadata, errors = scanner.scan(bad_path)

        assert metadata is None
        assert len(errors) == 1
        assert errors[0].code == ImageErrorCode.E_IMAGE_UNSUPPORTED_FORMAT.value

    def test_file_too_large_rejected(self, sample_image_path):
        config = ImageProcessorConfig(max_file_size_mb=0)  # 0 MB limit
        scanner = ImageSecurityScanner(config)
        metadata, errors = scanner.scan(sample_image_path)

        assert metadata is None
        assert any(e.code == ImageErrorCode.E_IMAGE_TOO_LARGE.value for e in errors)

    def test_empty_file_rejected(self, tmp_path, image_config):
        empty_path = str(tmp_path / "empty.png")
        with open(empty_path, "wb") as f:
            pass  # 0 bytes

        scanner = ImageSecurityScanner(image_config)
        metadata, errors = scanner.scan(empty_path)

        assert metadata is None
        assert any(e.code == ImageErrorCode.E_IMAGE_EMPTY.value for e in errors)

    def test_magic_bytes_mismatch_rejected(self, tmp_path, image_config):
        # Create a file with .png extension but JPEG content
        bad_path = str(tmp_path / "fake.png")
        with open(bad_path, "wb") as f:
            f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # JPEG magic bytes

        scanner = ImageSecurityScanner(image_config)
        metadata, errors = scanner.scan(bad_path)

        assert metadata is None
        assert any(e.code == ImageErrorCode.E_IMAGE_CORRUPT.value for e in errors)

    def test_oversized_dimensions_rejected(self, tmp_path):
        # Create a valid image
        img = Image.new("RGB", (100, 100), color=(0, 0, 0))
        path = str(tmp_path / "small.png")
        img.save(path, format="PNG")

        # Use config with very small dimension limits
        config = ImageProcessorConfig(max_image_width=50, max_image_height=50)
        scanner = ImageSecurityScanner(config)
        metadata, errors = scanner.scan(path)

        assert metadata is None
        assert any(
            e.code == ImageErrorCode.E_IMAGE_DIMENSIONS_EXCEEDED.value for e in errors
        )

    def test_content_hash_computed(self, sample_image_path, image_config):
        scanner = ImageSecurityScanner(image_config)
        metadata, errors = scanner.scan(sample_image_path)

        assert metadata is not None
        assert len(metadata.content_hash) == 64  # SHA-256 hex digest

    def test_color_mode_detected(self, sample_image_path, image_config):
        scanner = ImageSecurityScanner(image_config)
        metadata, errors = scanner.scan(sample_image_path)

        assert metadata is not None
        assert metadata.color_mode == "RGB"

    def test_bmp_file_passes(self, tmp_path, image_config):
        img = Image.new("RGB", (50, 50), color=(0, 255, 0))
        path = str(tmp_path / "test.bmp")
        img.save(path, format="BMP")

        scanner = ImageSecurityScanner(image_config)
        metadata, errors = scanner.scan(path)

        assert metadata is not None
        assert len(errors) == 0
        assert metadata.image_type.value == "bmp"

    def test_gif_file_passes(self, tmp_path, image_config):
        img = Image.new("RGB", (50, 50), color=(128, 128, 128))
        path = str(tmp_path / "test.gif")
        img.save(path, format="GIF")

        scanner = ImageSecurityScanner(image_config)
        metadata, errors = scanner.scan(path)

        assert metadata is not None
        assert len(errors) == 0
        assert metadata.image_type.value == "gif"

    def test_tiff_file_passes(self, tmp_path, image_config):
        img = Image.new("RGB", (50, 50), color=(0, 0, 128))
        path = str(tmp_path / "test.tiff")
        img.save(path, format="TIFF")

        scanner = ImageSecurityScanner(image_config)
        metadata, errors = scanner.scan(path)

        assert metadata is not None
        assert len(errors) == 0
        assert metadata.image_type.value == "tiff"
