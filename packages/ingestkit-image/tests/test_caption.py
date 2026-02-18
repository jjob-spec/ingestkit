"""Tests for ImageCaptionConverter."""

from __future__ import annotations

import pytest

from ingestkit_image.caption import CaptionError, ImageCaptionConverter
from ingestkit_image.config import ImageProcessorConfig
from ingestkit_image.errors import ImageErrorCode, ImageIngestError
from ingestkit_image.models import ImageChunkMetadata, ImageType


# ---------------------------------------------------------------------------
# Local mock (needed for direct construction with custom parameters)
# ---------------------------------------------------------------------------


class _MockVLM:
    """Inline mock VLM for tests that need custom constructor args."""

    def __init__(
        self,
        caption_text: str = "A photo of a building with a red roof.",
        available: bool = True,
        raise_on_caption: Exception | None = None,
    ) -> None:
        self._caption_text = caption_text
        self._available = available
        self._raise_on_caption = raise_on_caption
        self.caption_calls: list[dict] = []

    def caption(self, image_bytes, prompt, model, temperature=0.3, timeout=None) -> str:
        self.caption_calls.append({
            "image_bytes_len": len(image_bytes),
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            "timeout": timeout,
        })
        if self._raise_on_caption is not None:
            raise self._raise_on_caption
        return self._caption_text

    def model_name(self) -> str:
        return "mock-vlm:test"

    def is_available(self) -> bool:
        return self._available


@pytest.mark.unit
class TestImageCaptionConverterCaption:
    """Test ImageCaptionConverter.caption() method."""

    def test_successful_caption(
        self, sample_image_path, sample_image_metadata, image_config
    ):
        vlm = _MockVLM(caption_text="A red square on a white background.")
        converter = ImageCaptionConverter(vlm, image_config)

        result, warnings = converter.caption(sample_image_path, sample_image_metadata)

        assert result.caption == "A red square on a white background."
        assert result.model_used == image_config.vision_model
        assert result.caption_duration_seconds >= 0
        assert len(warnings) == 0
        assert len(vlm.caption_calls) == 1

    def test_vlm_unavailable_raises(
        self, sample_image_path, sample_image_metadata, image_config
    ):
        vlm = _MockVLM(available=False)
        converter = ImageCaptionConverter(vlm, image_config)

        with pytest.raises(CaptionError) as exc_info:
            converter.caption(sample_image_path, sample_image_metadata)

        assert exc_info.value.error.code == ImageErrorCode.E_IMAGE_VLM_UNAVAILABLE.value

    def test_vlm_timeout_raises(
        self, sample_image_path, sample_image_metadata, image_config
    ):
        vlm = _MockVLM(raise_on_caption=TimeoutError("timed out"))
        converter = ImageCaptionConverter(vlm, image_config)

        with pytest.raises(CaptionError) as exc_info:
            converter.caption(sample_image_path, sample_image_metadata)

        assert exc_info.value.error.code == ImageErrorCode.E_IMAGE_VLM_TIMEOUT.value

    def test_vlm_connection_error_raises(
        self, sample_image_path, sample_image_metadata, image_config
    ):
        vlm = _MockVLM(raise_on_caption=ConnectionError("refused"))
        converter = ImageCaptionConverter(vlm, image_config)

        with pytest.raises(CaptionError) as exc_info:
            converter.caption(sample_image_path, sample_image_metadata)

        assert exc_info.value.error.code == ImageErrorCode.E_IMAGE_VLM_UNAVAILABLE.value

    def test_vlm_empty_response_raises(
        self, sample_image_path, sample_image_metadata, image_config
    ):
        vlm = _MockVLM(caption_text="")
        converter = ImageCaptionConverter(vlm, image_config)

        with pytest.raises(CaptionError) as exc_info:
            converter.caption(sample_image_path, sample_image_metadata)

        assert exc_info.value.error.code == ImageErrorCode.E_IMAGE_VLM_EMPTY_RESPONSE.value

    def test_short_caption_warning(
        self, sample_image_path, sample_image_metadata
    ):
        config = ImageProcessorConfig(min_caption_length=50)
        vlm = _MockVLM(caption_text="Short text.")
        converter = ImageCaptionConverter(vlm, config)

        result, warnings = converter.caption(sample_image_path, sample_image_metadata)

        assert result.caption == "Short text."
        assert len(warnings) == 1
        assert warnings[0].code == ImageErrorCode.W_IMAGE_VLM_LOW_DETAIL.value

    def test_retry_on_timeout(self, sample_image_path, sample_image_metadata):
        """With max_retries=1, should try twice then fail."""
        config = ImageProcessorConfig(vlm_max_retries=1)
        vlm = _MockVLM(raise_on_caption=TimeoutError("timed out"))
        converter = ImageCaptionConverter(vlm, config)

        with pytest.raises(CaptionError) as exc_info:
            converter.caption(sample_image_path, sample_image_metadata)

        assert exc_info.value.error.code == ImageErrorCode.E_IMAGE_VLM_TIMEOUT.value
        # Should have been called 2 times (initial + 1 retry)
        assert len(vlm.caption_calls) == 2


@pytest.mark.unit
class TestImageCaptionConverterBuildChunk:
    """Test ImageCaptionConverter.build_chunk() method."""

    def test_build_chunk_correct_structure(
        self, sample_image_metadata, image_config, mock_vlm_backend
    ):
        converter = ImageCaptionConverter(mock_vlm_backend, image_config)

        chunk = converter.build_chunk(
            caption="A building with a red roof.",
            image_metadata=sample_image_metadata,
            ingest_key="test-key-123",
            ingest_run_id="run-456",
            vector=[0.1] * 768,
            chunk_index=0,
        )

        assert chunk.text == "A building with a red roof."
        assert len(chunk.vector) == 768
        assert chunk.id  # UUID string

    def test_build_chunk_metadata_source_type(
        self, sample_image_metadata, image_config, mock_vlm_backend
    ):
        converter = ImageCaptionConverter(mock_vlm_backend, image_config)

        chunk = converter.build_chunk(
            caption="Test caption",
            image_metadata=sample_image_metadata,
            ingest_key="key",
            ingest_run_id="run",
            vector=[0.1] * 768,
        )

        assert isinstance(chunk.metadata, ImageChunkMetadata)
        assert chunk.metadata.source_type == "image_caption"
        assert chunk.metadata.source_format == "image"
        assert chunk.metadata.ingestion_method == "vlm_caption"

    def test_build_chunk_image_metadata_propagated(
        self, sample_image_metadata, image_config, mock_vlm_backend
    ):
        converter = ImageCaptionConverter(mock_vlm_backend, image_config)

        chunk = converter.build_chunk(
            caption="Test",
            image_metadata=sample_image_metadata,
            ingest_key="key",
            ingest_run_id="run",
            vector=[0.1] * 768,
        )

        meta = chunk.metadata
        assert isinstance(meta, ImageChunkMetadata)
        assert meta.image_type == "png"
        assert meta.image_width == 100
        assert meta.image_height == 100
        assert meta.vlm_model == image_config.vision_model

    def test_build_chunk_tenant_id_propagated(
        self, sample_image_metadata, mock_vlm_backend
    ):
        config = ImageProcessorConfig(tenant_id="acme-corp")
        converter = ImageCaptionConverter(mock_vlm_backend, config)

        chunk = converter.build_chunk(
            caption="Test",
            image_metadata=sample_image_metadata,
            ingest_key="key",
            ingest_run_id="run",
            vector=[0.1] * 768,
        )

        assert chunk.metadata.tenant_id == "acme-corp"

    def test_build_chunk_ingest_key_propagated(
        self, sample_image_metadata, image_config, mock_vlm_backend
    ):
        converter = ImageCaptionConverter(mock_vlm_backend, image_config)

        chunk = converter.build_chunk(
            caption="Test",
            image_metadata=sample_image_metadata,
            ingest_key="my-ingest-key",
            ingest_run_id="my-run-id",
            vector=[0.1] * 768,
        )

        assert chunk.metadata.ingest_key == "my-ingest-key"
        assert chunk.metadata.ingest_run_id == "my-run-id"
