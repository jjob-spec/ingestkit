"""Tests for ImageRouter."""

from __future__ import annotations

import pytest

from ingestkit_image.config import ImageProcessorConfig
from ingestkit_image.errors import ImageErrorCode
from ingestkit_image.protocols import OCRResult
from ingestkit_image.router import ImageRouter


# ---------------------------------------------------------------------------
# Local mock classes (needed for direct construction with custom parameters)
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


class _MockVectorStore:
    """Inline mock VectorStoreBackend."""

    def __init__(self) -> None:
        self.upserted: list = []
        self.collections_ensured: list[str] = []

    def upsert_chunks(self, collection, chunks):
        self.upserted.extend(chunks)
        return len(chunks)

    def ensure_collection(self, collection, vector_size):
        self.collections_ensured.append(collection)

    def create_payload_index(self, collection, field, field_type):
        pass

    def delete_by_ids(self, collection, ids):
        return 0


class _MockEmbedder:
    """Inline mock EmbeddingBackend."""

    def __init__(self, dim: int = 768) -> None:
        self._dim = dim

    def embed(self, texts, timeout=None):
        return [[0.1] * self._dim for _ in texts]

    def dimension(self):
        return self._dim


@pytest.mark.unit
class TestImageRouterCanHandle:
    """Test ImageRouter.can_handle() method."""

    def test_can_handle_png(self):
        router = ImageRouter(_MockVLM(), _MockVectorStore(), _MockEmbedder())
        assert router.can_handle("/tmp/photo.png") is True

    def test_can_handle_jpg(self):
        router = ImageRouter(_MockVLM(), _MockVectorStore(), _MockEmbedder())
        assert router.can_handle("/tmp/photo.jpg") is True

    def test_can_handle_jpeg(self):
        router = ImageRouter(_MockVLM(), _MockVectorStore(), _MockEmbedder())
        assert router.can_handle("/tmp/photo.jpeg") is True

    def test_can_handle_tiff(self):
        router = ImageRouter(_MockVLM(), _MockVectorStore(), _MockEmbedder())
        assert router.can_handle("/tmp/scan.tiff") is True

    def test_can_handle_webp(self):
        router = ImageRouter(_MockVLM(), _MockVectorStore(), _MockEmbedder())
        assert router.can_handle("/tmp/img.webp") is True

    def test_can_handle_bmp(self):
        router = ImageRouter(_MockVLM(), _MockVectorStore(), _MockEmbedder())
        assert router.can_handle("/tmp/img.bmp") is True

    def test_can_handle_gif(self):
        router = ImageRouter(_MockVLM(), _MockVectorStore(), _MockEmbedder())
        assert router.can_handle("/tmp/anim.gif") is True

    def test_cannot_handle_pdf(self):
        router = ImageRouter(_MockVLM(), _MockVectorStore(), _MockEmbedder())
        assert router.can_handle("/tmp/doc.pdf") is False

    def test_cannot_handle_xlsx(self):
        router = ImageRouter(_MockVLM(), _MockVectorStore(), _MockEmbedder())
        assert router.can_handle("/tmp/data.xlsx") is False

    def test_cannot_handle_svg(self):
        router = ImageRouter(_MockVLM(), _MockVectorStore(), _MockEmbedder())
        assert router.can_handle("/tmp/vector.svg") is False


@pytest.mark.unit
class TestImageRouterProcess:
    """Test ImageRouter.process() method."""

    def test_happy_path(self, sample_image_path):
        vs = _MockVectorStore()
        vlm = _MockVLM(caption_text="A red square on a white background.")
        router = ImageRouter(vlm, vs, _MockEmbedder())

        result = router.process(sample_image_path)

        assert result.chunks_created == 1
        assert result.image_metadata is not None
        assert result.caption_result is not None
        assert result.caption_result.caption == "A red square on a white background."
        assert result.embed_result is not None
        assert result.ingest_key != ""
        assert len(result.errors) == 0
        assert result.written.vector_point_ids
        assert result.written.vector_collection == "helpdesk"
        assert result.processing_time_seconds > 0

    def test_happy_path_vector_store_upserted(self, sample_image_path):
        vs = _MockVectorStore()
        vlm = _MockVLM()
        router = ImageRouter(vlm, vs, _MockEmbedder())

        result = router.process(sample_image_path)

        assert len(vs.upserted) == 1
        chunk = vs.upserted[0]
        assert chunk.metadata.source_type == "image_caption"

    def test_vlm_unavailable_graceful_fallback(self, sample_image_path):
        vlm = _MockVLM(available=False)
        router = ImageRouter(vlm, _MockVectorStore(), _MockEmbedder())

        result = router.process(sample_image_path)

        assert result.chunks_created == 0
        assert ImageErrorCode.E_IMAGE_VLM_UNAVAILABLE.value in result.errors
        assert result.caption_result is None
        assert result.written == result.written.__class__()

    def test_vlm_timeout_graceful_fallback(self, sample_image_path):
        vlm = _MockVLM(raise_on_caption=TimeoutError("timed out"))
        config = ImageProcessorConfig(vlm_max_retries=0)
        router = ImageRouter(vlm, _MockVectorStore(), _MockEmbedder(), config=config)

        result = router.process(sample_image_path)

        assert result.chunks_created == 0
        assert ImageErrorCode.E_IMAGE_VLM_TIMEOUT.value in result.errors

    def test_security_failure_early_return(self, tmp_path):
        bad_path = str(tmp_path / "test.svg")
        with open(bad_path, "w") as f:
            f.write("<svg></svg>")

        router = ImageRouter(_MockVLM(), _MockVectorStore(), _MockEmbedder())
        result = router.process(bad_path)

        assert result.chunks_created == 0
        assert ImageErrorCode.E_IMAGE_UNSUPPORTED_FORMAT.value in result.errors
        assert result.ingest_key == ""
        assert result.image_metadata is None

    def test_tenant_id_propagated(self, sample_image_path):
        vs = _MockVectorStore()
        vlm = _MockVLM()
        config = ImageProcessorConfig(tenant_id="acme-corp")
        router = ImageRouter(vlm, vs, _MockEmbedder(), config=config)

        result = router.process(sample_image_path)

        assert result.tenant_id == "acme-corp"
        chunk = vs.upserted[0]
        assert chunk.metadata.tenant_id == "acme-corp"

    def test_caption_below_min_length_warning(self, sample_image_path):
        vlm = _MockVLM(caption_text="Short.")
        config = ImageProcessorConfig(min_caption_length=50)
        router = ImageRouter(vlm, _MockVectorStore(), _MockEmbedder(), config=config)

        result = router.process(sample_image_path)

        assert result.chunks_created == 1
        assert ImageErrorCode.W_IMAGE_VLM_LOW_DETAIL.value in result.warnings

    def test_embed_failure_returns_zero_chunks(self, sample_image_path):
        vlm = _MockVLM()

        class FailingEmbedder:
            def embed(self, texts, timeout=None):
                raise ConnectionError("embed service down")
            def dimension(self):
                return 768

        router = ImageRouter(vlm, _MockVectorStore(), FailingEmbedder())
        result = router.process(sample_image_path)

        assert result.chunks_created == 0
        assert ImageErrorCode.E_BACKEND_EMBED_CONNECT.value in result.errors

    def test_vector_store_failure_returns_zero_chunks(self, sample_image_path):
        vlm = _MockVLM()

        class FailingVectorStore:
            def upsert_chunks(self, collection, chunks):
                raise ConnectionError("vector store down")
            def ensure_collection(self, collection, vector_size):
                raise ConnectionError("vector store down")
            def create_payload_index(self, collection, field, field_type):
                pass
            def delete_by_ids(self, collection, ids):
                return 0

        router = ImageRouter(vlm, FailingVectorStore(), _MockEmbedder())
        result = router.process(sample_image_path)

        assert result.chunks_created == 0
        assert ImageErrorCode.E_BACKEND_VECTOR_CONNECT.value in result.errors

    def test_default_config_used_when_none(self):
        router = ImageRouter(_MockVLM(), _MockVectorStore(), _MockEmbedder(), config=None)
        assert router._config.vision_model == "llama3.2-vision:11b"


# ---------------------------------------------------------------------------
# OCR-related mock
# ---------------------------------------------------------------------------


class _MockOCR:
    """Inline mock OCR backend for router tests."""

    def __init__(
        self,
        ocr_text: str = "OCR extracted text from image.",
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


# ---------------------------------------------------------------------------
# Dual-mode router tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestImageRouterOCROnly:
    """Test ImageRouter in OCR-only mode (no VLM)."""

    def test_ocr_only_happy_path(self, sample_image_path):
        vs = _MockVectorStore()
        ocr = _MockOCR(ocr_text="Hello from OCR extraction.")
        config = ImageProcessorConfig(enable_ocr=True)
        router = ImageRouter(
            vlm=None,
            vector_store=vs,
            embedder=_MockEmbedder(),
            ocr=ocr,
            config=config,
        )

        result = router.process(sample_image_path)

        assert result.chunks_created == 1
        assert result.ocr_result is not None
        assert result.ocr_result.text == "Hello from OCR extraction."
        assert result.caption_result is None
        assert len(result.errors) == 0
        assert len(vs.upserted) == 1
        assert vs.upserted[0].metadata.source_type == "image_ocr_text"

    def test_ocr_only_requires_enable_ocr_config(self, sample_image_path):
        """If enable_ocr is False, OCR won't run even with backend provided."""
        vs = _MockVectorStore()
        ocr = _MockOCR()
        config = ImageProcessorConfig(enable_ocr=False)
        router = ImageRouter(
            vlm=None,
            vector_store=vs,
            embedder=_MockEmbedder(),
            ocr=ocr,
            config=config,
        )

        result = router.process(sample_image_path)

        # No texts to embed -> 0 chunks
        assert result.chunks_created == 0
        assert len(ocr.ocr_calls) == 0


@pytest.mark.unit
class TestImageRouterVLMAndOCR:
    """Test ImageRouter in vlm_and_ocr mode (both backends provided)."""

    def test_dual_mode_produces_two_chunks(self, sample_image_path):
        vs = _MockVectorStore()
        vlm = _MockVLM(caption_text="VLM description of the image.")
        ocr = _MockOCR(ocr_text="OCR text from the image.")
        config = ImageProcessorConfig(enable_ocr=True)
        router = ImageRouter(
            vlm=vlm,
            vector_store=vs,
            embedder=_MockEmbedder(),
            ocr=ocr,
            config=config,
        )

        result = router.process(sample_image_path)

        assert result.chunks_created == 2
        assert result.caption_result is not None
        assert result.ocr_result is not None
        assert result.caption_result.caption == "VLM description of the image."
        assert result.ocr_result.text == "OCR text from the image."
        assert len(vs.upserted) == 2
        source_types = {c.metadata.source_type for c in vs.upserted}
        assert source_types == {"image_caption", "image_ocr_text"}

    def test_ocr_failure_vlm_success_produces_one_chunk(self, sample_image_path):
        """OCR fails but VLM succeeds -> 1 chunk + OCR error."""
        vs = _MockVectorStore()
        vlm = _MockVLM(caption_text="VLM caption works.")
        ocr = _MockOCR(raise_on_ocr=ConnectionError("OCR down"))
        config = ImageProcessorConfig(enable_ocr=True)
        router = ImageRouter(
            vlm=vlm,
            vector_store=vs,
            embedder=_MockEmbedder(),
            ocr=ocr,
            config=config,
        )

        result = router.process(sample_image_path)

        assert result.chunks_created == 1
        assert result.caption_result is not None
        assert result.ocr_result is None
        assert ImageErrorCode.E_IMAGE_OCR_UNAVAILABLE.value in result.errors
        assert len(vs.upserted) == 1
        assert vs.upserted[0].metadata.source_type == "image_caption"

    def test_vlm_failure_ocr_success_produces_one_chunk(self, sample_image_path):
        """VLM fails but OCR succeeds -> 1 chunk + VLM error."""
        vs = _MockVectorStore()
        vlm = _MockVLM(available=False)
        ocr = _MockOCR(ocr_text="OCR text works.")
        config = ImageProcessorConfig(enable_ocr=True)
        router = ImageRouter(
            vlm=vlm,
            vector_store=vs,
            embedder=_MockEmbedder(),
            ocr=ocr,
            config=config,
        )

        result = router.process(sample_image_path)

        assert result.chunks_created == 1
        assert result.caption_result is None
        assert result.ocr_result is not None
        assert ImageErrorCode.E_IMAGE_VLM_UNAVAILABLE.value in result.errors
        assert len(vs.upserted) == 1
        assert vs.upserted[0].metadata.source_type == "image_ocr_text"

    def test_both_fail_produces_zero_chunks(self, sample_image_path):
        """Both VLM and OCR fail -> 0 chunks + both errors."""
        vs = _MockVectorStore()
        vlm = _MockVLM(available=False)
        ocr = _MockOCR(raise_on_ocr=ConnectionError("OCR down"))
        config = ImageProcessorConfig(enable_ocr=True)
        router = ImageRouter(
            vlm=vlm,
            vector_store=vs,
            embedder=_MockEmbedder(),
            ocr=ocr,
            config=config,
        )

        result = router.process(sample_image_path)

        assert result.chunks_created == 0
        assert ImageErrorCode.E_IMAGE_VLM_UNAVAILABLE.value in result.errors
        assert ImageErrorCode.E_IMAGE_OCR_UNAVAILABLE.value in result.errors


@pytest.mark.unit
class TestImageRouterConstructorValidation:
    """Test ImageRouter constructor validation."""

    def test_neither_backend_raises_value_error(self):
        with pytest.raises(ValueError, match="At least one of"):
            ImageRouter(
                vlm=None,
                vector_store=_MockVectorStore(),
                embedder=_MockEmbedder(),
                ocr=None,
            )

    def test_vlm_only_construction_succeeds(self):
        router = ImageRouter(_MockVLM(), _MockVectorStore(), _MockEmbedder())
        assert router._vlm is not None
        assert router._ocr is None

    def test_ocr_only_construction_succeeds(self):
        router = ImageRouter(
            vlm=None,
            vector_store=_MockVectorStore(),
            embedder=_MockEmbedder(),
            ocr=_MockOCR(),
        )
        assert router._vlm is None
        assert router._ocr is not None
