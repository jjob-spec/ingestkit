"""ImageRouter -- orchestrator and public API for the ingestkit-image pipeline.

Routes image files through one or both processing paths:

- **VLM path**: Security scan -> VLM caption -> embed -> upsert
- **OCR path**: Security scan -> OCR extract -> embed -> upsert
- **Dual path**: Both VLM and OCR, producing two chunks per image

Processing mode is determined by which backends are provided at
construction time:

- ``vlm`` only -> vlm_only mode
- ``ocr`` only -> ocr_only mode
- Both ``vlm`` and ``ocr`` -> vlm_and_ocr mode
- Neither -> ``ValueError`` at construction
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid

from ingestkit_core.idempotency import compute_ingest_key
from ingestkit_core.models import ChunkPayload, EmbedStageResult, WrittenArtifacts

from ingestkit_image.caption import CaptionError, ImageCaptionConverter
from ingestkit_image.config import ImageProcessorConfig
from ingestkit_image.errors import ImageErrorCode, ImageIngestError
from ingestkit_image.models import (
    CaptionResult,
    ImageMetadata,
    ImageProcessingResult,
    OCRTextResult,
)
from ingestkit_image.ocr_extract import ImageOCRExtractor, OCRExtractError
from ingestkit_image.protocols import (
    EmbeddingBackend,
    ImageOCRBackend,
    ImageVLMBackend,
    VectorStoreBackend,
)
from ingestkit_image.security import ImageSecurityScanner, _EXTENSION_MAP

logger = logging.getLogger("ingestkit_image")


class ImageRouter:
    """Orchestrator for the image processing pipeline.

    Supports three modes depending on which backends are provided:
    vlm_only, ocr_only, or vlm_and_ocr.
    """

    def __init__(
        self,
        vlm: ImageVLMBackend | None = None,
        vector_store: VectorStoreBackend | None = None,
        embedder: EmbeddingBackend | None = None,
        ocr: ImageOCRBackend | None = None,
        config: ImageProcessorConfig | None = None,
    ) -> None:
        if vlm is None and ocr is None:
            raise ValueError("At least one of 'vlm' or 'ocr' must be provided")

        self._config = config or ImageProcessorConfig()
        self._vlm = vlm
        self._ocr = ocr
        self._vector_store = vector_store
        self._embedder = embedder
        self._security_scanner = ImageSecurityScanner(self._config)
        self._caption_converter: ImageCaptionConverter | None = None
        self._ocr_extractor: ImageOCRExtractor | None = None

        if self._vlm is not None:
            self._caption_converter = ImageCaptionConverter(self._vlm, self._config)
        if self._ocr is not None:
            self._ocr_extractor = ImageOCRExtractor(self._ocr, self._config)

    def can_handle(self, file_path: str) -> bool:
        """Return True if file extension is a supported image format."""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in _EXTENSION_MAP

    def process(
        self,
        file_path: str,
        source_uri: str | None = None,
    ) -> ImageProcessingResult:
        """Process a single image file. Synchronous.

        Parameters
        ----------
        file_path:
            Filesystem path to the image file.
        source_uri:
            Optional override for the source URI stored in the ingest key.

        Returns
        -------
        ImageProcessingResult
            The fully-assembled result.
        """
        overall_start = time.monotonic()
        config = self._config
        filename = os.path.basename(file_path)
        ingest_run_id = str(uuid.uuid4())
        all_errors: list[str] = []
        all_warnings: list[str] = []
        all_error_details: list[ImageIngestError] = []

        # ==============================================================
        # Step 1: Security Scan
        # ==============================================================
        image_metadata, security_errors = self._security_scanner.scan(file_path)

        fatal_errors = [
            e for e in security_errors if e.code.startswith("E_")
        ]
        security_warnings = [
            e for e in security_errors if not e.code.startswith("E_")
        ]

        if fatal_errors:
            elapsed = time.monotonic() - overall_start
            logger.error(
                "ingestkit_image | file=%s | code=%s | detail=%s",
                filename,
                fatal_errors[0].code,
                fatal_errors[0].message,
            )
            return ImageProcessingResult(
                file_path=file_path,
                ingest_key="",
                ingest_run_id=ingest_run_id,
                tenant_id=config.tenant_id,
                image_metadata=None,
                caption_result=None,
                ocr_result=None,
                embed_result=None,
                chunks_created=0,
                written=WrittenArtifacts(),
                errors=[e.code for e in fatal_errors],
                warnings=[e.code for e in security_warnings],
                error_details=security_errors,
                processing_time_seconds=elapsed,
            )

        all_warnings.extend(e.code for e in security_warnings)
        all_error_details.extend(security_warnings)

        # ==============================================================
        # Step 2: Compute Ingest Key
        # ==============================================================
        ingest_key_obj = compute_ingest_key(
            file_path=file_path,
            parser_version=config.parser_version,
            tenant_id=config.tenant_id,
            source_uri=source_uri,
        )
        ingest_key = ingest_key_obj.key

        assert image_metadata is not None  # guaranteed by security scan passing

        # ==============================================================
        # Step 3: Run VLM and/or OCR pipelines
        # ==============================================================
        caption_result: CaptionResult | None = None
        ocr_result: OCRTextResult | None = None
        texts_to_embed: list[str] = []
        text_labels: list[str] = []  # Track which text is which

        # --- VLM pipeline ---
        if self._caption_converter is not None:
            caption_result, vlm_ok = self._run_vlm_pipeline(
                file_path, image_metadata, all_errors, all_warnings,
                all_error_details,
            )
            if caption_result is not None:
                texts_to_embed.append(caption_result.caption)
                text_labels.append("vlm")

        # --- OCR pipeline ---
        if self._ocr_extractor is not None and config.enable_ocr:
            ocr_result, ocr_ok = self._run_ocr_pipeline(
                file_path, image_metadata, all_errors, all_warnings,
                all_error_details,
            )
            if ocr_result is not None:
                texts_to_embed.append(ocr_result.text)
                text_labels.append("ocr")

        # If no texts to embed, return early with errors
        if not texts_to_embed:
            elapsed = time.monotonic() - overall_start
            return ImageProcessingResult(
                file_path=file_path,
                ingest_key=ingest_key,
                ingest_run_id=ingest_run_id,
                tenant_id=config.tenant_id,
                image_metadata=image_metadata,
                caption_result=caption_result,
                ocr_result=ocr_result,
                embed_result=None,
                chunks_created=0,
                written=WrittenArtifacts(),
                errors=all_errors,
                warnings=all_warnings,
                error_details=all_error_details,
                processing_time_seconds=elapsed,
            )

        # ==============================================================
        # Step 4: Embed all texts in a single batch
        # ==============================================================
        embed_start = time.monotonic()
        try:
            vectors = self._embedder.embed(texts_to_embed)
        except (ConnectionError, TimeoutError) as exc:
            elapsed = time.monotonic() - overall_start
            error_code = (
                ImageErrorCode.E_BACKEND_EMBED_TIMEOUT.value
                if isinstance(exc, TimeoutError)
                else ImageErrorCode.E_BACKEND_EMBED_CONNECT.value
            )
            all_errors.append(error_code)
            all_error_details.append(
                ImageIngestError(
                    code=error_code,
                    message=f"Embedding backend error: {exc}",
                    stage="embed",
                    file_path=file_path,
                )
            )
            return ImageProcessingResult(
                file_path=file_path,
                ingest_key=ingest_key,
                ingest_run_id=ingest_run_id,
                tenant_id=config.tenant_id,
                image_metadata=image_metadata,
                caption_result=caption_result,
                ocr_result=ocr_result,
                embed_result=None,
                chunks_created=0,
                written=WrittenArtifacts(),
                errors=all_errors,
                warnings=all_warnings,
                error_details=all_error_details,
                processing_time_seconds=elapsed,
            )

        embed_duration = time.monotonic() - embed_start
        embed_result = EmbedStageResult(
            texts_embedded=len(vectors),
            embedding_dimension=len(vectors[0]),
            embed_duration_seconds=embed_duration,
        )

        # ==============================================================
        # Step 5: Build Chunks
        # ==============================================================
        chunks: list[ChunkPayload] = []
        chunk_index = 0

        for i, label in enumerate(text_labels):
            if label == "vlm" and caption_result is not None:
                chunk = self._caption_converter.build_chunk(
                    caption=caption_result.caption,
                    image_metadata=image_metadata,
                    ingest_key=ingest_key,
                    ingest_run_id=ingest_run_id,
                    vector=vectors[i],
                    chunk_index=chunk_index,
                )
                chunks.append(chunk)
                chunk_index += 1
            elif label == "ocr" and ocr_result is not None:
                chunk = self._ocr_extractor.build_chunk(
                    ocr_text=ocr_result.text,
                    ocr_result=ocr_result,
                    image_metadata=image_metadata,
                    ingest_key=ingest_key,
                    ingest_run_id=ingest_run_id,
                    vector=vectors[i],
                    chunk_index=chunk_index,
                )
                chunks.append(chunk)
                chunk_index += 1

        # ==============================================================
        # Step 6: Upsert to Vector Store
        # ==============================================================
        try:
            self._vector_store.ensure_collection(
                config.default_collection,
                config.embedding_dimension,
            )
            upserted = self._vector_store.upsert_chunks(
                config.default_collection,
                chunks,
            )
        except (ConnectionError, TimeoutError) as exc:
            elapsed = time.monotonic() - overall_start
            error_code = (
                ImageErrorCode.E_BACKEND_VECTOR_TIMEOUT.value
                if isinstance(exc, TimeoutError)
                else ImageErrorCode.E_BACKEND_VECTOR_CONNECT.value
            )
            all_errors.append(error_code)
            all_error_details.append(
                ImageIngestError(
                    code=error_code,
                    message=f"Vector store error: {exc}",
                    stage="upsert",
                    file_path=file_path,
                )
            )
            return ImageProcessingResult(
                file_path=file_path,
                ingest_key=ingest_key,
                ingest_run_id=ingest_run_id,
                tenant_id=config.tenant_id,
                image_metadata=image_metadata,
                caption_result=caption_result,
                ocr_result=ocr_result,
                embed_result=embed_result,
                chunks_created=0,
                written=WrittenArtifacts(),
                errors=all_errors,
                warnings=all_warnings,
                error_details=all_error_details,
                processing_time_seconds=elapsed,
            )

        written = WrittenArtifacts(
            vector_point_ids=[c.id for c in chunks],
            vector_collection=config.default_collection,
        )

        # ==============================================================
        # Final Result
        # ==============================================================
        elapsed = time.monotonic() - overall_start

        logger.info(
            "ingestkit_image | file=%s | ingest_key=%s | format=%s | "
            "dimensions=%dx%d | chunks=%d | time=%.1fs",
            filename,
            ingest_key[:8],
            image_metadata.image_type.value,
            image_metadata.width,
            image_metadata.height,
            len(chunks),
            elapsed,
        )

        return ImageProcessingResult(
            file_path=file_path,
            ingest_key=ingest_key,
            ingest_run_id=ingest_run_id,
            tenant_id=config.tenant_id,
            image_metadata=image_metadata,
            caption_result=caption_result,
            ocr_result=ocr_result,
            embed_result=embed_result,
            chunks_created=len(chunks),
            written=written,
            errors=all_errors,
            warnings=all_warnings,
            error_details=all_error_details,
            processing_time_seconds=elapsed,
        )

    def _run_vlm_pipeline(
        self,
        file_path: str,
        image_metadata: ImageMetadata,
        all_errors: list[str],
        all_warnings: list[str],
        all_error_details: list[ImageIngestError],
    ) -> tuple[CaptionResult | None, bool]:
        """Run the VLM captioning pipeline.

        Returns (CaptionResult or None, success_flag).
        On failure, appends to error lists and returns (None, False).
        """
        try:
            caption_result, caption_warnings = self._caption_converter.caption(
                image_path=file_path,
                image_metadata=image_metadata,
            )
        except CaptionError as exc:
            err = exc.error
            all_errors.append(err.code)
            all_error_details.append(err)
            logger.warning(
                "ingestkit_image | file=%s | code=%s | detail=%s",
                os.path.basename(file_path),
                err.code,
                err.message,
            )
            return None, False
        except (ConnectionError, TimeoutError) as exc:
            error_code = (
                ImageErrorCode.E_IMAGE_VLM_TIMEOUT.value
                if isinstance(exc, TimeoutError)
                else ImageErrorCode.E_IMAGE_VLM_UNAVAILABLE.value
            )
            all_errors.append(error_code)
            all_error_details.append(
                ImageIngestError(
                    code=error_code,
                    message=f"VLM backend error: {exc}",
                    stage="caption",
                    file_path=file_path,
                )
            )
            return None, False

        # Collect caption warnings
        for w in caption_warnings:
            all_warnings.append(w.code)
            all_error_details.append(w)

        return caption_result, True

    def _run_ocr_pipeline(
        self,
        file_path: str,
        image_metadata: ImageMetadata,
        all_errors: list[str],
        all_warnings: list[str],
        all_error_details: list[ImageIngestError],
    ) -> tuple[OCRTextResult | None, bool]:
        """Run the OCR extraction pipeline.

        Returns (OCRTextResult or None, success_flag).
        On failure, appends to error lists and returns (None, False).
        """
        try:
            ocr_result, ocr_warnings = self._ocr_extractor.extract(
                image_path=file_path,
                image_metadata=image_metadata,
            )
        except OCRExtractError as exc:
            err = exc.error
            all_errors.append(err.code)
            all_error_details.append(err)
            logger.warning(
                "ingestkit_image | file=%s | code=%s | detail=%s",
                os.path.basename(file_path),
                err.code,
                err.message,
            )
            return None, False
        except (ConnectionError, TimeoutError) as exc:
            error_code = (
                ImageErrorCode.E_IMAGE_OCR_TIMEOUT.value
                if isinstance(exc, TimeoutError)
                else ImageErrorCode.E_IMAGE_OCR_UNAVAILABLE.value
            )
            all_errors.append(error_code)
            all_error_details.append(
                ImageIngestError(
                    code=error_code,
                    message=f"OCR backend error: {exc}",
                    stage="ocr",
                    file_path=file_path,
                )
            )
            return None, False

        # Collect OCR warnings
        for w in ocr_warnings:
            all_warnings.append(w.code)
            all_error_details.append(w)

        return ocr_result, True

    async def aprocess(
        self,
        file_path: str,
        source_uri: str | None = None,
    ) -> ImageProcessingResult:
        """Async wrapper via asyncio.to_thread().

        Offloads the synchronous ``process()`` call to a thread so
        callers using async frameworks can ``await`` without blocking
        the event loop.
        """
        return await asyncio.to_thread(self.process, file_path, source_uri)
