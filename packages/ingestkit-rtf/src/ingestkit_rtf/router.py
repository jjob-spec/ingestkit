"""RTFRouter -- orchestrator and public API for the ingestkit-rtf pipeline.

Routes RTF files through the full ingestion pipeline:

1. Security scan via :class:`RTFSecurityScanner`.
2. Compute deterministic :class:`IngestKey` for deduplication.
3. Extract text via :func:`extract_text`.
4. Validate non-empty text.
5. Chunk via :func:`chunk_text`.
6. Embed chunks via :class:`EmbeddingBackend`.
7. Build :class:`ChunkPayload` list with :class:`RTFChunkMetadata`.
8. Upsert via :class:`VectorStoreBackend`.
9. Assemble and return :class:`ProcessingResult`.

The router enforces **fail-closed** semantics: any fatal error returns a
result with error codes and zero chunks.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
import uuid

from ingestkit_core.idempotency import compute_ingest_key
from ingestkit_core.models import ChunkPayload, EmbedStageResult, WrittenArtifacts
from ingestkit_core.protocols import EmbeddingBackend, VectorStoreBackend

from ingestkit_rtf.config import RTFProcessorConfig
from ingestkit_rtf.converter import chunk_text, extract_text
from ingestkit_rtf.errors import ErrorCode, IngestError
from ingestkit_rtf.models import RTFChunkMetadata, ProcessingResult
from ingestkit_rtf.security import RTFSecurityScanner

logger = logging.getLogger("ingestkit_rtf")


class RTFRouter:
    """Top-level orchestrator for the ingestkit-rtf pipeline.

    Builds all internal components from the injected backends and config,
    then exposes :meth:`can_handle`, :meth:`process`, and :meth:`aprocess`
    as the public API.

    Parameters
    ----------
    vector_store:
        Backend for vector storage (e.g. Qdrant).
    embedder:
        Backend for text embedding.
    config:
        Pipeline configuration.  Uses defaults when *None*.
    """

    def __init__(
        self,
        vector_store: VectorStoreBackend,
        embedder: EmbeddingBackend,
        config: RTFProcessorConfig | None = None,
    ) -> None:
        self._config = config or RTFProcessorConfig()
        self._vector_store = vector_store
        self._embedder = embedder
        self._security_scanner = RTFSecurityScanner(self._config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def can_handle(self, file_path: str) -> bool:
        """Return True if *file_path* ends with ``.rtf`` (case-insensitive)."""
        return file_path.lower().endswith(".rtf")

    def process(
        self,
        file_path: str,
        source_uri: str | None = None,
    ) -> ProcessingResult:
        """Extract, chunk, embed, and upsert a single RTF file.

        Parameters
        ----------
        file_path:
            Filesystem path to the RTF file.
        source_uri:
            Optional override for the source URI stored in the ingest key.

        Returns
        -------
        ProcessingResult
            The fully-assembled result.
        """
        overall_start = time.monotonic()
        config = self._config
        filename = os.path.basename(file_path)
        ingest_run_id = str(uuid.uuid4())

        # ==============================================================
        # Step 1: Security Scan
        # ==============================================================
        security_errors = self._security_scanner.scan(file_path)
        fatal_errors = [e for e in security_errors if e.code.startswith("E_")]
        warnings = [e for e in security_errors if not e.code.startswith("E_")]

        if fatal_errors:
            elapsed = time.monotonic() - overall_start
            logger.error(
                "ingestkit_rtf | file=%s | code=%s | detail=%s",
                filename,
                fatal_errors[0].code,
                fatal_errors[0].message,
            )
            return ProcessingResult(
                file_path=file_path,
                ingest_key="",
                ingest_run_id=ingest_run_id,
                tenant_id=config.tenant_id,
                errors=[e.code for e in fatal_errors],
                warnings=[e.code for e in warnings],
                error_details=security_errors,
                processing_time_seconds=elapsed,
            )

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
        # Step 3: Extract Text
        # ==============================================================
        result_warnings: list[str] = [e.code for e in warnings]
        try:
            extract_result = extract_text(file_path)
        except Exception as exc:
            elapsed = time.monotonic() - overall_start
            err = IngestError(
                code=ErrorCode.E_RTF_EXTRACT_FAILED,
                message=f"Failed to extract text from RTF file: {exc}",
                stage="extract",
            )
            logger.error(
                "ingestkit_rtf | file=%s | code=%s | detail=%s",
                filename,
                ErrorCode.E_RTF_EXTRACT_FAILED,
                str(exc),
            )
            return ProcessingResult(
                file_path=file_path,
                ingest_key=ingest_key,
                ingest_run_id=ingest_run_id,
                tenant_id=config.tenant_id,
                errors=[ErrorCode.E_RTF_EXTRACT_FAILED.value],
                warnings=result_warnings,
                error_details=[err],
                processing_time_seconds=elapsed,
            )

        # ==============================================================
        # Step 4: Validate Non-Empty Text
        # ==============================================================
        if not extract_result.text.strip():
            elapsed = time.monotonic() - overall_start
            err = IngestError(
                code=ErrorCode.E_RTF_EMPTY_TEXT,
                message="Document parsed but produced zero text",
                stage="extract",
            )
            return ProcessingResult(
                file_path=file_path,
                ingest_key=ingest_key,
                ingest_run_id=ingest_run_id,
                tenant_id=config.tenant_id,
                errors=[ErrorCode.E_RTF_EMPTY_TEXT.value],
                warnings=result_warnings,
                error_details=[err],
                processing_time_seconds=elapsed,
            )

        # ==============================================================
        # Step 5: Chunk Text
        # ==============================================================
        chunks_text = chunk_text(extract_result.text, config)

        if not chunks_text:
            elapsed = time.monotonic() - overall_start
            return ProcessingResult(
                file_path=file_path,
                ingest_key=ingest_key,
                ingest_run_id=ingest_run_id,
                tenant_id=config.tenant_id,
                word_count=extract_result.word_count,
                warnings=result_warnings,
                error_details=list(warnings),
                processing_time_seconds=elapsed,
            )

        # ==============================================================
        # Step 6: Embed
        # ==============================================================
        embed_start = time.monotonic()
        try:
            all_vectors: list[list[float]] = []
            for i in range(0, len(chunks_text), config.embedding_batch_size):
                batch = chunks_text[i : i + config.embedding_batch_size]
                vectors = self._embedder.embed(
                    batch, timeout=config.backend_timeout_seconds
                )
                all_vectors.extend(vectors)
            embed_duration = time.monotonic() - embed_start
        except (ConnectionError, TimeoutError) as exc:
            elapsed = time.monotonic() - overall_start
            err = IngestError(
                code=ErrorCode.E_BACKEND_EMBED_TIMEOUT,
                message=f"Embedding backend error: {exc}",
                stage="embed",
            )
            return ProcessingResult(
                file_path=file_path,
                ingest_key=ingest_key,
                ingest_run_id=ingest_run_id,
                tenant_id=config.tenant_id,
                word_count=extract_result.word_count,
                errors=[ErrorCode.E_BACKEND_EMBED_TIMEOUT.value],
                warnings=result_warnings,
                error_details=[err],
                processing_time_seconds=elapsed,
            )
        except Exception as exc:
            elapsed = time.monotonic() - overall_start
            err = IngestError(
                code=ErrorCode.E_BACKEND_EMBED_CONNECT,
                message=f"Embedding backend error: {exc}",
                stage="embed",
            )
            return ProcessingResult(
                file_path=file_path,
                ingest_key=ingest_key,
                ingest_run_id=ingest_run_id,
                tenant_id=config.tenant_id,
                word_count=extract_result.word_count,
                errors=[ErrorCode.E_BACKEND_EMBED_CONNECT.value],
                warnings=result_warnings,
                error_details=[err],
                processing_time_seconds=elapsed,
            )

        embed_result = EmbedStageResult(
            texts_embedded=len(chunks_text),
            embedding_dimension=len(all_vectors[0]) if all_vectors else 0,
            embed_duration_seconds=embed_duration,
        )

        # ==============================================================
        # Step 7: Build ChunkPayloads
        # ==============================================================
        source_uri_final = source_uri or file_path
        payloads: list[ChunkPayload] = []
        point_ids: list[str] = []

        for idx, (text, vector) in enumerate(zip(chunks_text, all_vectors)):
            chunk_hash = hashlib.sha256(text.encode()).hexdigest()
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{ingest_key}:{idx}"))

            metadata = RTFChunkMetadata(
                source_uri=source_uri_final,
                source_format="rtf",
                ingestion_method="rtf_striprtf",
                parser_version=config.parser_version,
                chunk_index=idx,
                chunk_hash=chunk_hash,
                ingest_key=ingest_key,
                ingest_run_id=ingest_run_id,
                tenant_id=config.tenant_id,
                word_count=extract_result.word_count,
            )
            payload = ChunkPayload(
                id=chunk_id,
                text=text,
                vector=vector,
                metadata=metadata,
            )
            payloads.append(payload)
            point_ids.append(chunk_id)

        # ==============================================================
        # Step 8: Upsert to Vector Store
        # ==============================================================
        collection = config.default_collection
        try:
            self._vector_store.ensure_collection(
                collection, config.embedding_dimension
            )
            upserted = self._vector_store.upsert_chunks(collection, payloads)
        except (ConnectionError, TimeoutError) as exc:
            elapsed = time.monotonic() - overall_start
            err = IngestError(
                code=ErrorCode.E_BACKEND_VECTOR_TIMEOUT,
                message=f"Vector store error: {exc}",
                stage="upsert",
            )
            return ProcessingResult(
                file_path=file_path,
                ingest_key=ingest_key,
                ingest_run_id=ingest_run_id,
                tenant_id=config.tenant_id,
                embed_result=embed_result,
                word_count=extract_result.word_count,
                errors=[ErrorCode.E_BACKEND_VECTOR_TIMEOUT.value],
                warnings=result_warnings,
                error_details=[err],
                processing_time_seconds=elapsed,
            )
        except Exception as exc:
            elapsed = time.monotonic() - overall_start
            err = IngestError(
                code=ErrorCode.E_BACKEND_VECTOR_CONNECT,
                message=f"Vector store error: {exc}",
                stage="upsert",
            )
            return ProcessingResult(
                file_path=file_path,
                ingest_key=ingest_key,
                ingest_run_id=ingest_run_id,
                tenant_id=config.tenant_id,
                embed_result=embed_result,
                word_count=extract_result.word_count,
                errors=[ErrorCode.E_BACKEND_VECTOR_CONNECT.value],
                warnings=result_warnings,
                error_details=[err],
                processing_time_seconds=elapsed,
            )

        written = WrittenArtifacts(
            vector_point_ids=point_ids,
            vector_collection=collection,
        )

        # ==============================================================
        # Step 9: Assemble Result
        # ==============================================================
        elapsed = time.monotonic() - overall_start

        logger.info(
            "ingestkit_rtf | file=%s | ingest_key=%s | words=%d | "
            "chunks=%d | time=%.1fs",
            filename,
            ingest_key[:8],
            extract_result.word_count,
            len(payloads),
            elapsed,
        )

        return ProcessingResult(
            file_path=file_path,
            ingest_key=ingest_key,
            ingest_run_id=ingest_run_id,
            tenant_id=config.tenant_id,
            embed_result=embed_result,
            chunks_created=len(payloads),
            written=written,
            word_count=extract_result.word_count,
            warnings=result_warnings,
            error_details=list(warnings),
            processing_time_seconds=elapsed,
        )

    async def aprocess(
        self,
        file_path: str,
        source_uri: str | None = None,
    ) -> ProcessingResult:
        """Async wrapper around :meth:`process`.

        Offloads the synchronous ``process()`` call to a thread via
        ``asyncio.to_thread()``.
        """
        return await asyncio.to_thread(self.process, file_path, source_uri)
