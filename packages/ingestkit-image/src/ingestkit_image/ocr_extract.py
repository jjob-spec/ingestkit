"""OCR text extraction from images via an OCR backend.

``ImageOCRExtractor`` sends image bytes to an OCR backend and wraps
the response as a ``ChunkPayload`` with ``source_type=image_ocr_text``
metadata suitable for vector store indexing.  Parallel to ``caption.py``
which handles VLM captioning.
"""

from __future__ import annotations

import hashlib
import io
import logging
import time
import uuid

from PIL import Image

from ingestkit_core.models import ChunkPayload

from ingestkit_image.config import ImageProcessorConfig
from ingestkit_image.errors import ImageErrorCode, ImageIngestError
from ingestkit_image.models import (
    ImageChunkMetadata,
    ImageMetadata,
    OCRTextResult,
)
from ingestkit_image.protocols import ImageOCRBackend

logger = logging.getLogger("ingestkit_image")


class OCRExtractError(Exception):
    """Raised when OCR extraction fails with a structured error detail."""

    def __init__(self, error: ImageIngestError) -> None:
        self.error = error
        super().__init__(error.message)


class ImageOCRExtractor:
    """Extracts text from an image via OCR.

    Reads image bytes, optionally resizes large images, calls the OCR
    backend with retry logic, validates the result, and builds a
    ``ChunkPayload`` for vector store indexing.
    """

    def __init__(
        self,
        ocr: ImageOCRBackend,
        config: ImageProcessorConfig,
    ) -> None:
        self._ocr = ocr
        self._config = config

    def extract(
        self, image_path: str, image_metadata: ImageMetadata
    ) -> tuple[OCRTextResult, list[ImageIngestError]]:
        """Run OCR on the full image.

        Returns a tuple of (OCRTextResult, warnings).

        Raises
        ------
        OCRExtractError
            For fatal errors (OCR unavailable, timeout, empty text).
        """
        warnings: list[ImageIngestError] = []

        # 1. Preprocess: resize if needed
        image_bytes, was_resized, ocr_dimensions = self._preprocess_image(
            image_path, image_metadata
        )

        if was_resized:
            warnings.append(
                ImageIngestError(
                    code=ImageErrorCode.W_IMAGE_OCR_RESIZED.value,
                    message=(
                        f"Image resized from {image_metadata.width}x{image_metadata.height} "
                        f"to {ocr_dimensions[0]}x{ocr_dimensions[1]} for OCR"
                    ),
                    stage="ocr",
                    file_path=image_path,
                    recoverable=True,
                )
            )

        # 2. Call OCR backend with retry
        ocr_text = ""
        ocr_confidence = 0.0
        ocr_engine = ""
        ocr_language = ""
        start = time.monotonic()
        last_exc: Exception | None = None

        for attempt in range(1 + self._config.ocr_max_retries):
            try:
                result = self._ocr.ocr_image(
                    image_bytes=image_bytes,
                    language=self._config.ocr_language,
                    config=self._config.ocr_config,
                    timeout=self._config.ocr_timeout_seconds,
                )
                ocr_text = result.text
                ocr_confidence = result.confidence
                ocr_engine = result.engine
                ocr_language = result.language
                last_exc = None
                break
            except TimeoutError as exc:
                last_exc = exc
                if attempt < self._config.ocr_max_retries:
                    warnings.append(
                        ImageIngestError(
                            code=ImageErrorCode.W_IMAGE_OCR_RETRY.value,
                            message=f"OCR timeout on attempt {attempt + 1}, retrying",
                            stage="ocr",
                            file_path=image_path,
                            recoverable=True,
                        )
                    )
            except ConnectionError as exc:
                last_exc = exc
                break  # No retry for connection errors

        duration = time.monotonic() - start

        # Handle fatal errors after retries
        if last_exc is not None:
            if isinstance(last_exc, TimeoutError):
                raise OCRExtractError(
                    ImageIngestError(
                        code=ImageErrorCode.E_IMAGE_OCR_TIMEOUT.value,
                        message=f"OCR timed out after {self._config.ocr_max_retries + 1} attempts",
                        stage="ocr",
                        file_path=image_path,
                    )
                )
            raise OCRExtractError(
                ImageIngestError(
                    code=ImageErrorCode.E_IMAGE_OCR_UNAVAILABLE.value,
                    message=f"OCR backend connection error: {last_exc}",
                    stage="ocr",
                    file_path=image_path,
                )
            )

        # 3. Validate result
        if not ocr_text or not ocr_text.strip():
            raise OCRExtractError(
                ImageIngestError(
                    code=ImageErrorCode.E_IMAGE_OCR_EMPTY_TEXT.value,
                    message="OCR returned empty text",
                    stage="ocr",
                    file_path=image_path,
                )
            )

        ocr_text = ocr_text.strip()

        if len(ocr_text) < self._config.ocr_min_text_length:
            warnings.append(
                ImageIngestError(
                    code=ImageErrorCode.W_IMAGE_OCR_LOW_CONFIDENCE.value,
                    message=(
                        f"OCR text length {len(ocr_text)} below minimum "
                        f"{self._config.ocr_min_text_length}"
                    ),
                    stage="ocr",
                    file_path=image_path,
                    recoverable=True,
                )
            )

        if ocr_confidence < 0.5:
            warnings.append(
                ImageIngestError(
                    code=ImageErrorCode.W_IMAGE_OCR_LOW_CONFIDENCE.value,
                    message=f"OCR confidence {ocr_confidence:.2f} is below 0.50 threshold",
                    stage="ocr",
                    file_path=image_path,
                    recoverable=True,
                )
            )

        logger.debug(
            "ingestkit_image | ocr | file=%s | engine=%s | "
            "text_length=%d | confidence=%.2f | time=%.1fs",
            image_path,
            ocr_engine,
            len(ocr_text),
            ocr_confidence,
            duration,
        )

        return OCRTextResult(
            text=ocr_text,
            confidence=ocr_confidence,
            engine=ocr_engine,
            language=ocr_language,
            ocr_duration_seconds=duration,
            was_resized=was_resized,
            original_dimensions=(image_metadata.width, image_metadata.height),
            ocr_dimensions=ocr_dimensions,
        ), warnings

    def _preprocess_image(
        self, image_path: str, image_metadata: ImageMetadata
    ) -> tuple[bytes, bool, tuple[int, int] | None]:
        """Resize if needed, convert to PNG bytes.

        Returns (image_bytes, was_resized, ocr_dimensions).
        ``ocr_dimensions`` is the (width, height) after resize, or
        ``None`` if no resize was needed.
        """
        megapixels = (image_metadata.width * image_metadata.height) / 1_000_000

        if megapixels > self._config.ocr_megapixel_threshold:
            # Resize proportionally so the longer side fits within max_dimension
            with Image.open(image_path) as img:
                max_dim = self._config.ocr_max_dimension
                ratio = min(
                    max_dim / img.width,
                    max_dim / img.height,
                )
                new_width = int(img.width * ratio)
                new_height = int(img.height * ratio)
                resized = img.resize(
                    (new_width, new_height), Image.LANCZOS
                )
                buf = io.BytesIO()
                resized.save(buf, format="PNG")
                return buf.getvalue(), True, (new_width, new_height)

        # No resize needed -- read raw bytes
        with open(image_path, "rb") as f:
            return f.read(), False, None

    def build_chunk(
        self,
        ocr_text: str,
        ocr_result: OCRTextResult,
        image_metadata: ImageMetadata,
        ingest_key: str,
        ingest_run_id: str,
        vector: list[float],
        chunk_index: int = 0,
    ) -> ChunkPayload:
        """Build a ChunkPayload from OCR text.

        Sets ``source_type=image_ocr_text`` in metadata for downstream
        consumers to identify OCR-derived chunks.
        """
        chunk_id = str(uuid.uuid4())
        chunk_hash = hashlib.sha256(ocr_text.encode()).hexdigest()

        metadata = ImageChunkMetadata(
            source_uri=image_metadata.file_path,
            source_format="image",
            source_type="image_ocr_text",
            ingestion_method="ocr_extract",
            parser_version=self._config.parser_version,
            chunk_index=chunk_index,
            chunk_hash=chunk_hash,
            ingest_key=ingest_key,
            ingest_run_id=ingest_run_id,
            tenant_id=self._config.tenant_id,
            image_type=image_metadata.image_type.value,
            image_width=image_metadata.width,
            image_height=image_metadata.height,
            ocr_engine=ocr_result.engine,
            ocr_confidence=ocr_result.confidence,
            ocr_language=ocr_result.language,
        )

        return ChunkPayload(
            id=chunk_id,
            text=ocr_text,
            vector=vector,
            metadata=metadata,
        )
