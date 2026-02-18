"""ImageRouter -- orchestrator and public API for the ingestkit-image pipeline.

Routes image files through the captioning pipeline:

1. Security scan via :class:`ImageSecurityScanner`.
2. Compute deterministic :class:`IngestKey` for deduplication.
3. VLM caption via :class:`ImageCaptionConverter`.
4. Embed caption text via :class:`EmbeddingBackend`.
5. Build :class:`ChunkPayload` and upsert to vector store.
6. Return a fully-assembled :class:`ImageProcessingResult`.

Graceful fallback: if the VLM is unavailable, returns a result with
``E_IMAGE_VLM_UNAVAILABLE`` error and zero chunks.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid

from ingestkit_core.idempotency import compute_ingest_key
from ingestkit_core.models import EmbedStageResult, WrittenArtifacts

from ingestkit_image.caption import CaptionError, ImageCaptionConverter
from ingestkit_image.config import ImageProcessorConfig
from ingestkit_image.errors import ImageErrorCode, ImageIngestError
from ingestkit_image.models import ImageProcessingResult
from ingestkit_image.protocols import (
    EmbeddingBackend,
    ImageVLMBackend,
    VectorStoreBackend,
)
from ingestkit_image.security import ImageSecurityScanner, _EXTENSION_MAP

logger = logging.getLogger("ingestkit_image")


class ImageRouter:
    """Orchestrator for the image captioning pipeline.

    Pipeline: security scan -> ingest key -> VLM caption -> embed -> vector store
    """

    def __init__(
        self,
        vlm: ImageVLMBackend,
        vector_store: VectorStoreBackend,
        embedder: EmbeddingBackend,
        config: ImageProcessorConfig | None = None,
    ) -> None:
        self._config = config or ImageProcessorConfig()
        self._vlm = vlm
        self._vector_store = vector_store
        self._embedder = embedder
        self._security_scanner = ImageSecurityScanner(self._config)
        self._caption_converter = ImageCaptionConverter(self._vlm, self._config)

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

        # ==============================================================
        # Step 3: VLM Caption (graceful fallback)
        # ==============================================================
        assert image_metadata is not None  # guaranteed by security scan passing

        try:
            caption_result, caption_warnings = self._caption_converter.caption(
                image_path=file_path,
                image_metadata=image_metadata,
            )
        except CaptionError as exc:
            # Graceful fallback: VLM unavailable or timeout
            elapsed = time.monotonic() - overall_start
            err = exc.error
            all_errors.append(err.code)
            all_error_details.append(err)
            logger.warning(
                "ingestkit_image | file=%s | code=%s | detail=%s",
                filename,
                err.code,
                err.message,
            )
            return ImageProcessingResult(
                file_path=file_path,
                ingest_key=ingest_key,
                ingest_run_id=ingest_run_id,
                tenant_id=config.tenant_id,
                image_metadata=image_metadata,
                caption_result=None,
                embed_result=None,
                chunks_created=0,
                written=WrittenArtifacts(),
                errors=all_errors,
                warnings=all_warnings,
                error_details=all_error_details,
                processing_time_seconds=elapsed,
            )
        except (ConnectionError, TimeoutError) as exc:
            # Catch raw connection/timeout errors from backend
            elapsed = time.monotonic() - overall_start
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
            return ImageProcessingResult(
                file_path=file_path,
                ingest_key=ingest_key,
                ingest_run_id=ingest_run_id,
                tenant_id=config.tenant_id,
                image_metadata=image_metadata,
                caption_result=None,
                embed_result=None,
                chunks_created=0,
                written=WrittenArtifacts(),
                errors=all_errors,
                warnings=all_warnings,
                error_details=all_error_details,
                processing_time_seconds=elapsed,
            )

        # Collect caption warnings
        for w in caption_warnings:
            all_warnings.append(w.code)
            all_error_details.append(w)

        # ==============================================================
        # Step 4: Embed Caption
        # ==============================================================
        embed_start = time.monotonic()
        try:
            vectors = self._embedder.embed([caption_result.caption])
            vector = vectors[0]
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
            texts_embedded=1,
            embedding_dimension=len(vector),
            embed_duration_seconds=embed_duration,
        )

        # ==============================================================
        # Step 5: Build Chunk
        # ==============================================================
        chunk = self._caption_converter.build_chunk(
            caption=caption_result.caption,
            image_metadata=image_metadata,
            ingest_key=ingest_key,
            ingest_run_id=ingest_run_id,
            vector=vector,
        )

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
                [chunk],
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
                embed_result=embed_result,
                chunks_created=0,
                written=WrittenArtifacts(),
                errors=all_errors,
                warnings=all_warnings,
                error_details=all_error_details,
                processing_time_seconds=elapsed,
            )

        written = WrittenArtifacts(
            vector_point_ids=[chunk.id],
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
            1,
            elapsed,
        )

        return ImageProcessingResult(
            file_path=file_path,
            ingest_key=ingest_key,
            ingest_run_id=ingest_run_id,
            tenant_id=config.tenant_id,
            image_metadata=image_metadata,
            caption_result=caption_result,
            embed_result=embed_result,
            chunks_created=1,
            written=written,
            errors=all_errors,
            warnings=all_warnings,
            error_details=all_error_details,
            processing_time_seconds=elapsed,
        )

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
