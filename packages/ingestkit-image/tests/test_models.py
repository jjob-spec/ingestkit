"""Tests for ingestkit-image models."""

from __future__ import annotations

import pytest

from ingestkit_core.models import BaseChunkMetadata, WrittenArtifacts

from ingestkit_image.errors import ImageIngestError
from ingestkit_image.models import (
    CaptionResult,
    ImageChunkMetadata,
    ImageMetadata,
    ImageProcessingResult,
    ImageType,
)


@pytest.mark.unit
class TestImageType:
    """Test ImageType enum."""

    def test_all_formats_present(self):
        assert ImageType.JPEG.value == "jpeg"
        assert ImageType.PNG.value == "png"
        assert ImageType.TIFF.value == "tiff"
        assert ImageType.WEBP.value == "webp"
        assert ImageType.BMP.value == "bmp"
        assert ImageType.GIF.value == "gif"

    def test_string_enum(self):
        assert isinstance(ImageType.JPEG, str)
        assert ImageType.JPEG == "jpeg"


@pytest.mark.unit
class TestImageMetadata:
    """Test ImageMetadata model."""

    def test_construction(self):
        meta = ImageMetadata(
            file_path="/tmp/test.png",
            file_size_bytes=1024,
            image_type=ImageType.PNG,
            width=800,
            height=600,
            content_hash="abc123",
        )
        assert meta.file_path == "/tmp/test.png"
        assert meta.file_size_bytes == 1024
        assert meta.image_type == ImageType.PNG
        assert meta.width == 800
        assert meta.height == 600
        assert meta.has_exif is False
        assert meta.color_mode is None

    def test_with_optional_fields(self):
        meta = ImageMetadata(
            file_path="/tmp/test.jpg",
            file_size_bytes=2048,
            image_type=ImageType.JPEG,
            width=1920,
            height=1080,
            content_hash="def456",
            has_exif=True,
            color_mode="RGB",
        )
        assert meta.has_exif is True
        assert meta.color_mode == "RGB"


@pytest.mark.unit
class TestImageChunkMetadata:
    """Test ImageChunkMetadata model."""

    def test_extends_base_chunk_metadata(self):
        assert issubclass(ImageChunkMetadata, BaseChunkMetadata)

    def test_default_source_type(self):
        meta = ImageChunkMetadata(
            source_uri="/tmp/test.png",
            source_format="image",
            ingestion_method="vlm_caption",
            parser_version="ingestkit_image:1.0.0",
            chunk_index=0,
            chunk_hash="abc",
            ingest_key="key123",
        )
        assert meta.source_type == "image_caption"
        assert meta.source_format == "image"

    def test_image_specific_fields(self):
        meta = ImageChunkMetadata(
            source_uri="/tmp/test.png",
            source_format="image",
            ingestion_method="vlm_caption",
            parser_version="ingestkit_image:1.0.0",
            chunk_index=0,
            chunk_hash="abc",
            ingest_key="key123",
            image_type="png",
            image_width=800,
            image_height=600,
            vlm_model="llama3.2-vision:11b",
            caption_prompt="Describe this image.",
        )
        assert meta.image_type == "png"
        assert meta.image_width == 800
        assert meta.vlm_model == "llama3.2-vision:11b"


@pytest.mark.unit
class TestCaptionResult:
    """Test CaptionResult model."""

    def test_construction(self):
        result = CaptionResult(
            caption="A red building with white windows.",
            model_used="llama3.2-vision:11b",
            caption_duration_seconds=2.5,
        )
        assert result.caption == "A red building with white windows."
        assert result.model_used == "llama3.2-vision:11b"
        assert result.caption_duration_seconds == 2.5


@pytest.mark.unit
class TestImageProcessingResult:
    """Test ImageProcessingResult model."""

    def test_construction_minimal(self):
        result = ImageProcessingResult(
            file_path="/tmp/test.png",
            ingest_key="key123",
            ingest_run_id="run-1",
            chunks_created=0,
            written=WrittenArtifacts(),
            errors=["E_IMAGE_VLM_UNAVAILABLE"],
            warnings=[],
            processing_time_seconds=0.1,
        )
        assert result.file_path == "/tmp/test.png"
        assert result.chunks_created == 0
        assert result.tenant_id is None
        assert result.image_metadata is None
        assert result.caption_result is None

    def test_tenant_id_propagation(self):
        result = ImageProcessingResult(
            file_path="/tmp/test.png",
            ingest_key="key123",
            ingest_run_id="run-1",
            tenant_id="acme",
            chunks_created=1,
            written=WrittenArtifacts(),
            errors=[],
            warnings=[],
            processing_time_seconds=0.5,
        )
        assert result.tenant_id == "acme"
