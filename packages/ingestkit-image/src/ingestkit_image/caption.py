"""Core image captioning logic via Vision-Language Model.

``ImageCaptionConverter`` sends image bytes to a VLM backend and wraps
the response as a ``ChunkPayload`` with ``source_type=image_caption``
metadata suitable for vector store indexing.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid

from ingestkit_core.models import ChunkPayload

from ingestkit_image.config import ImageProcessorConfig
from ingestkit_image.errors import ImageErrorCode, ImageIngestError
from ingestkit_image.models import (
    CaptionResult,
    ImageChunkMetadata,
    ImageMetadata,
)
from ingestkit_image.protocols import ImageVLMBackend

logger = logging.getLogger("ingestkit_image")


class CaptionError(Exception):
    """Raised when captioning fails with a structured error detail."""

    def __init__(self, error: ImageIngestError) -> None:
        self.error = error
        super().__init__(error.message)


class ImageCaptionConverter:
    """Converts an image into a text caption via VLM.

    Reads image bytes, calls the VLM backend, validates the response,
    and builds a ``ChunkPayload`` for vector store indexing.
    """

    def __init__(
        self,
        vlm: ImageVLMBackend,
        config: ImageProcessorConfig,
    ) -> None:
        self._vlm = vlm
        self._config = config

    def caption(
        self, image_path: str, image_metadata: ImageMetadata
    ) -> tuple[CaptionResult, list[ImageIngestError]]:
        """Send image to VLM and return caption text.

        Reads image bytes, calls VLM backend, validates response.
        Returns a tuple of (CaptionResult, warnings).

        Raises
        ------
        CaptionError
            For fatal errors (VLM unavailable, timeout, empty response).
        """
        warnings: list[ImageIngestError] = []

        # 1. Read image bytes
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # 2. Check VLM availability
        if not self._vlm.is_available():
            raise CaptionError(
                ImageIngestError(
                    code=ImageErrorCode.E_IMAGE_VLM_UNAVAILABLE.value,
                    message="VLM backend is not available",
                    stage="caption",
                    file_path=image_path,
                )
            )

        # 3. Call VLM with retry
        caption_text = ""
        start = time.monotonic()
        last_exc: Exception | None = None

        for attempt in range(1 + self._config.vlm_max_retries):
            try:
                caption_text = self._vlm.caption(
                    image_bytes=image_bytes,
                    prompt=self._config.caption_prompt,
                    model=self._config.vision_model,
                    temperature=self._config.vlm_temperature,
                    timeout=float(self._config.vlm_timeout_seconds),
                )
                last_exc = None
                break
            except TimeoutError as exc:
                last_exc = exc
                if attempt < self._config.vlm_max_retries:
                    warnings.append(
                        ImageIngestError(
                            code=ImageErrorCode.W_IMAGE_VLM_RETRY.value,
                            message=f"VLM timeout on attempt {attempt + 1}, retrying",
                            stage="caption",
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
                raise CaptionError(
                    ImageIngestError(
                        code=ImageErrorCode.E_IMAGE_VLM_TIMEOUT.value,
                        message=f"VLM timed out after {self._config.vlm_max_retries + 1} attempts",
                        stage="caption",
                        file_path=image_path,
                    )
                )
            raise CaptionError(
                ImageIngestError(
                    code=ImageErrorCode.E_IMAGE_VLM_UNAVAILABLE.value,
                    message=f"VLM connection error: {last_exc}",
                    stage="caption",
                    file_path=image_path,
                )
            )

        # 4. Validate response
        if not caption_text or not caption_text.strip():
            raise CaptionError(
                ImageIngestError(
                    code=ImageErrorCode.E_IMAGE_VLM_EMPTY_RESPONSE.value,
                    message="VLM returned empty caption",
                    stage="caption",
                    file_path=image_path,
                )
            )

        caption_text = caption_text.strip()

        if len(caption_text) < self._config.min_caption_length:
            warnings.append(
                ImageIngestError(
                    code=ImageErrorCode.W_IMAGE_VLM_LOW_DETAIL.value,
                    message=(
                        f"Caption length {len(caption_text)} below minimum "
                        f"{self._config.min_caption_length}"
                    ),
                    stage="caption",
                    file_path=image_path,
                    recoverable=True,
                )
            )

        if self._config.log_captions:
            logger.debug(
                "ingestkit_image | caption | file=%s | model=%s | "
                "length=%d | time=%.1fs",
                image_path,
                self._config.vision_model,
                len(caption_text),
                duration,
            )

        return CaptionResult(
            caption=caption_text,
            model_used=self._config.vision_model,
            caption_duration_seconds=duration,
        ), warnings

    def build_chunk(
        self,
        caption: str,
        image_metadata: ImageMetadata,
        ingest_key: str,
        ingest_run_id: str,
        vector: list[float],
        chunk_index: int = 0,
    ) -> ChunkPayload:
        """Build a ChunkPayload from a caption string.

        Sets ``source_type=image_caption`` in metadata for downstream
        consumers to identify image-derived chunks.
        """
        chunk_id = str(uuid.uuid4())
        chunk_hash = hashlib.sha256(caption.encode()).hexdigest()

        metadata = ImageChunkMetadata(
            source_uri=image_metadata.file_path,
            source_format="image",
            source_type="image_caption",
            ingestion_method="vlm_caption",
            parser_version=self._config.parser_version,
            chunk_index=chunk_index,
            chunk_hash=chunk_hash,
            ingest_key=ingest_key,
            ingest_run_id=ingest_run_id,
            tenant_id=self._config.tenant_id,
            image_type=image_metadata.image_type.value,
            image_width=image_metadata.width,
            image_height=image_metadata.height,
            vlm_model=self._config.vision_model,
            caption_prompt=self._config.caption_prompt,
        )

        return ChunkPayload(
            id=chunk_id,
            text=caption,
            vector=vector,
            metadata=metadata,
        )
