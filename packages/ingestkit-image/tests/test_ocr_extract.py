"""Tests for ImageOCRExtractor."""

from __future__ import annotations

import os

import pytest
from PIL import Image

from ingestkit_image.config import ImageProcessorConfig
from ingestkit_image.errors import ImageErrorCode
from ingestkit_image.models import ImageMetadata, ImageType, OCRTextResult
from ingestkit_image.ocr_extract import ImageOCRExtractor, OCRExtractError
from ingestkit_image.protocols import OCRResult


# ---------------------------------------------------------------------------
# Local mock OCR backend for tests needing custom behaviour
# ---------------------------------------------------------------------------


class _MockOCR:
    """Inline mock OCR backend for tests that need custom constructor args."""

    def __init__(
        self,
        ocr_text: str = "Sample OCR text from image.",
        confidence: float = 0.92,
        engine: str = "tesseract",
        language: str = "eng",
        raise_on_ocr: Exception | None = None,
    ) -> None:
        self._ocr_text = ocr_text
        self._confidence = confidence
        self._engine = engine
        self._language = language
        self._raise_on_ocr = raise_on_ocr
        self.ocr_calls: list[dict] = []

    def ocr_image(self, image_bytes, language="en", config=None, timeout=None) -> OCRResult:
        self.ocr_calls.append({
            "image_bytes_len": len(image_bytes),
            "language": language,
            "config": config,
            "timeout": timeout,
        })
        if self._raise_on_ocr is not None:
            raise self._raise_on_ocr
        return OCRResult(
            text=self._ocr_text,
            confidence=self._confidence,
            engine=self._engine,
            language=self._language,
        )

    def engine_name(self) -> str:
        return self._engine


def _make_image_metadata(path: str, width: int = 100, height: int = 100) -> ImageMetadata:
    """Create ImageMetadata for testing."""
    import hashlib

    content_hash = hashlib.sha256(open(path, "rb").read()).hexdigest()
    return ImageMetadata(
        file_path=path,
        file_size_bytes=os.path.getsize(path),
        image_type=ImageType.PNG,
        width=width,
        height=height,
        content_hash=content_hash,
        has_exif=False,
        color_mode="RGB",
    )


@pytest.mark.unit
class TestImageOCRExtractorHappyPath:
    """Test OCR extraction happy path."""

    def test_extract_returns_ocr_text_result(self, sample_image_path, sample_image_metadata):
        ocr = _MockOCR(ocr_text="Hello World from OCR")
        config = ImageProcessorConfig(enable_ocr=True)
        extractor = ImageOCRExtractor(ocr, config)

        result, warnings = extractor.extract(sample_image_path, sample_image_metadata)

        assert isinstance(result, OCRTextResult)
        assert result.text == "Hello World from OCR"
        assert result.confidence == 0.92
        assert result.engine == "tesseract"
        assert result.language == "eng"
        assert result.ocr_duration_seconds > 0
        assert result.was_resized is False
        assert len(warnings) == 0

    def test_extract_passes_config_to_backend(self, sample_image_path, sample_image_metadata):
        ocr = _MockOCR()
        config = ImageProcessorConfig(
            enable_ocr=True,
            ocr_language="fra",
            ocr_config="--psm 6",
            ocr_timeout_seconds=45.0,
        )
        extractor = ImageOCRExtractor(ocr, config)

        extractor.extract(sample_image_path, sample_image_metadata)

        assert len(ocr.ocr_calls) == 1
        call = ocr.ocr_calls[0]
        assert call["language"] == "fra"
        assert call["config"] == "--psm 6"
        assert call["timeout"] == 45.0


@pytest.mark.unit
class TestImageOCRExtractorEmptyText:
    """Test OCR extraction with empty text."""

    def test_empty_text_raises_ocr_extract_error(self, sample_image_path, sample_image_metadata):
        ocr = _MockOCR(ocr_text="")
        config = ImageProcessorConfig(enable_ocr=True)
        extractor = ImageOCRExtractor(ocr, config)

        with pytest.raises(OCRExtractError) as exc_info:
            extractor.extract(sample_image_path, sample_image_metadata)

        assert exc_info.value.error.code == ImageErrorCode.E_IMAGE_OCR_EMPTY_TEXT.value

    def test_whitespace_only_raises_ocr_extract_error(self, sample_image_path, sample_image_metadata):
        ocr = _MockOCR(ocr_text="   \n\t  ")
        config = ImageProcessorConfig(enable_ocr=True)
        extractor = ImageOCRExtractor(ocr, config)

        with pytest.raises(OCRExtractError) as exc_info:
            extractor.extract(sample_image_path, sample_image_metadata)

        assert exc_info.value.error.code == ImageErrorCode.E_IMAGE_OCR_EMPTY_TEXT.value


@pytest.mark.unit
class TestImageOCRExtractorBackendErrors:
    """Test OCR extraction with backend errors."""

    def test_connection_error_raises_unavailable(self, sample_image_path, sample_image_metadata):
        ocr = _MockOCR(raise_on_ocr=ConnectionError("OCR service down"))
        config = ImageProcessorConfig(enable_ocr=True)
        extractor = ImageOCRExtractor(ocr, config)

        with pytest.raises(OCRExtractError) as exc_info:
            extractor.extract(sample_image_path, sample_image_metadata)

        assert exc_info.value.error.code == ImageErrorCode.E_IMAGE_OCR_UNAVAILABLE.value

    def test_timeout_with_retry_then_success(self, sample_image_path, sample_image_metadata):
        """First call times out, second succeeds."""
        call_count = 0

        class TimingOutThenSuccessOCR:
            def ocr_image(self, image_bytes, language="en", config=None, timeout=None):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise TimeoutError("timed out")
                return OCRResult(
                    text="OCR text after retry",
                    confidence=0.85,
                    engine="tesseract",
                    language="eng",
                )

            def engine_name(self):
                return "tesseract"

        config = ImageProcessorConfig(enable_ocr=True, ocr_max_retries=1)
        extractor = ImageOCRExtractor(TimingOutThenSuccessOCR(), config)

        result, warnings = extractor.extract(sample_image_path, sample_image_metadata)

        assert result.text == "OCR text after retry"
        assert call_count == 2
        retry_warnings = [w for w in warnings if w.code == ImageErrorCode.W_IMAGE_OCR_RETRY.value]
        assert len(retry_warnings) == 1

    def test_timeout_exhausted_raises_error(self, sample_image_path, sample_image_metadata):
        """All retries time out."""
        ocr = _MockOCR(raise_on_ocr=TimeoutError("timed out"))
        config = ImageProcessorConfig(enable_ocr=True, ocr_max_retries=1)
        extractor = ImageOCRExtractor(ocr, config)

        with pytest.raises(OCRExtractError) as exc_info:
            extractor.extract(sample_image_path, sample_image_metadata)

        assert exc_info.value.error.code == ImageErrorCode.E_IMAGE_OCR_TIMEOUT.value


@pytest.mark.unit
class TestImageOCRExtractorPreprocessing:
    """Test image preprocessing (resize) for OCR."""

    def test_large_image_is_resized(self, tmp_path):
        """Image >20MP should be resized, with W_IMAGE_OCR_RESIZED warning."""
        # Create a large image: 5000x5000 = 25MP (> 20MP threshold)
        img = Image.new("RGB", (5000, 5000), color=(128, 128, 128))
        path = str(tmp_path / "large_image.png")
        img.save(path, format="PNG")

        metadata = _make_image_metadata(path, width=5000, height=5000)

        ocr = _MockOCR()
        config = ImageProcessorConfig(
            enable_ocr=True,
            ocr_megapixel_threshold=20.0,
            ocr_max_dimension=4096,
            max_image_width=10000,
            max_image_height=10000,
        )
        extractor = ImageOCRExtractor(ocr, config)

        result, warnings = extractor.extract(path, metadata)

        assert result.was_resized is True
        assert result.ocr_dimensions is not None
        assert max(result.ocr_dimensions) <= 4096
        resize_warnings = [w for w in warnings if w.code == ImageErrorCode.W_IMAGE_OCR_RESIZED.value]
        assert len(resize_warnings) == 1

    def test_small_image_not_resized(self, sample_image_path, sample_image_metadata):
        """Image <20MP should not be resized."""
        ocr = _MockOCR()
        config = ImageProcessorConfig(enable_ocr=True)
        extractor = ImageOCRExtractor(ocr, config)

        result, warnings = extractor.extract(sample_image_path, sample_image_metadata)

        assert result.was_resized is False
        assert result.ocr_dimensions is None
        resize_warnings = [w for w in warnings if w.code == ImageErrorCode.W_IMAGE_OCR_RESIZED.value]
        assert len(resize_warnings) == 0


@pytest.mark.unit
class TestImageOCRExtractorChunkBuilding:
    """Test chunk building from OCR text."""

    def test_build_chunk_produces_correct_metadata(self, sample_image_path, sample_image_metadata):
        ocr = _MockOCR()
        config = ImageProcessorConfig(enable_ocr=True, tenant_id="test-tenant")
        extractor = ImageOCRExtractor(ocr, config)

        ocr_result = OCRTextResult(
            text="Extracted text from image",
            confidence=0.90,
            engine="tesseract",
            language="eng",
            ocr_duration_seconds=1.5,
        )

        chunk = extractor.build_chunk(
            ocr_text="Extracted text from image",
            ocr_result=ocr_result,
            image_metadata=sample_image_metadata,
            ingest_key="abc123",
            ingest_run_id="run-456",
            vector=[0.1] * 768,
            chunk_index=0,
        )

        assert chunk.text == "Extracted text from image"
        assert chunk.metadata.source_type == "image_ocr_text"
        assert chunk.metadata.ingestion_method == "ocr_extract"
        assert chunk.metadata.source_format == "image"
        assert chunk.metadata.ocr_engine == "tesseract"
        assert chunk.metadata.ocr_confidence == 0.90
        assert chunk.metadata.ocr_language == "eng"
        assert chunk.metadata.tenant_id == "test-tenant"
        assert chunk.metadata.ingest_key == "abc123"
        assert chunk.vector == [0.1] * 768


@pytest.mark.unit
class TestImageOCRExtractorLowConfidence:
    """Test OCR low confidence warnings."""

    def test_low_confidence_produces_warning(self, sample_image_path, sample_image_metadata):
        ocr = _MockOCR(ocr_text="Some text", confidence=0.3)
        config = ImageProcessorConfig(enable_ocr=True)
        extractor = ImageOCRExtractor(ocr, config)

        result, warnings = extractor.extract(sample_image_path, sample_image_metadata)

        assert result.text == "Some text"
        confidence_warnings = [
            w for w in warnings if w.code == ImageErrorCode.W_IMAGE_OCR_LOW_CONFIDENCE.value
        ]
        assert len(confidence_warnings) >= 1

    def test_short_text_produces_warning(self, sample_image_path, sample_image_metadata):
        ocr = _MockOCR(ocr_text="Hi")
        config = ImageProcessorConfig(enable_ocr=True, ocr_min_text_length=5)
        extractor = ImageOCRExtractor(ocr, config)

        result, warnings = extractor.extract(sample_image_path, sample_image_metadata)

        assert result.text == "Hi"
        short_warnings = [
            w for w in warnings if w.code == ImageErrorCode.W_IMAGE_OCR_LOW_CONFIDENCE.value
        ]
        assert len(short_warnings) >= 1
