"""XMLRouter -- orchestrator and public API for the ingestkit-xml pipeline.

Routes XML files through the full ingestion pipeline:

1. Security scan via :class:`XMLSecurityScanner`.
2. Compute deterministic :class:`IngestKey` for deduplication.
3. Parse XML (with plain-text fallback on parse failure).
4. Extract via :func:`extract_xml`.
5. Chunk via :func:`chunk_text`.
6. Embed chunks via :class:`EmbeddingBackend`.
7. Build :class:`ChunkPayload` list with :class:`XMLChunkMetadata`.
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
import xml.etree.ElementTree as ET

from ingestkit_core.idempotency import compute_ingest_key
from ingestkit_core.models import ChunkPayload, EmbedStageResult, WrittenArtifacts
from ingestkit_core.protocols import EmbeddingBackend, VectorStoreBackend

from ingestkit_xml.config import XMLProcessorConfig
from ingestkit_xml.converter import chunk_text, extract_xml
from ingestkit_xml.errors import ErrorCode, IngestError
from ingestkit_xml.models import ExtractResult, ProcessingResult, XMLChunkMetadata
from ingestkit_xml.security import XMLSecurityScanner

logger = logging.getLogger("ingestkit_xml")


class XMLRouter:
    """Top-level orchestrator for the ingestkit-xml pipeline.

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
        config: XMLProcessorConfig | None = None,
    ) -> None:
        self._config = config or XMLProcessorConfig()
        self._vector_store = vector_store
        self._embedder = embedder
        self._security_scanner = XMLSecurityScanner(self._config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def can_handle(self, file_path: str) -> bool:
        """Return True if *file_path* ends with ``.xml`` (case-insensitive)."""
        return file_path.lower().endswith(".xml")

    def process(
        self,
        file_path: str,
        source_uri: str | None = None,
    ) -> ProcessingResult:
        """Extract, chunk, embed, and upsert a single XML file.

        Parameters
        ----------
        file_path:
            Filesystem path to the XML file.
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
                "ingestkit_xml | file=%s | code=%s | detail=%s",
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
        # Step 3: Parse XML (with plain-text fallback)
        # ==============================================================
        result_warnings: list[str] = [e.code for e in warnings]
        fallback_used = False

        try:
            tree = ET.parse(file_path)  # noqa: S314
            root = tree.getroot()
        except ET.ParseError:
            # Fallback: read as plain text
            fallback_used = True
            result_warnings.append(ErrorCode.W_MALFORMED_FALLBACK.value)
            try:
                with open(file_path, "r", encoding="utf-8") as fh:
                    raw_text = fh.read()
                lines = [line for line in raw_text.splitlines() if line.strip()]
            except Exception as exc:
                elapsed = time.monotonic() - overall_start
                err = IngestError(
                    code=ErrorCode.E_PARSE_CORRUPT,
                    message=f"Failed to read XML file as text: {exc}",
                    stage="parse",
                )
                return ProcessingResult(
                    file_path=file_path,
                    ingest_key=ingest_key,
                    ingest_run_id=ingest_run_id,
                    tenant_id=config.tenant_id,
                    errors=[ErrorCode.E_PARSE_CORRUPT.value],
                    warnings=result_warnings,
                    error_details=[err],
                    processing_time_seconds=elapsed,
                )

            extract_result = ExtractResult(
                lines=lines,
                total_elements=0,
                max_depth=0,
                namespaces=[],
                root_tag="",
                truncated=False,
                fallback_used=True,
            )
            root = None  # type: ignore[assignment]

        if not fallback_used:
            # ==============================================================
            # Step 4: Extract
            # ==============================================================
            extract_result = extract_xml(root, config)

            if extract_result.truncated:
                result_warnings.append(ErrorCode.W_TRUNCATED.value)

        # ==============================================================
        # Step 5: Chunk
        # ==============================================================
        chunks_text = chunk_text(extract_result.lines, config)

        if not chunks_text:
            elapsed = time.monotonic() - overall_start
            return ProcessingResult(
                file_path=file_path,
                ingest_key=ingest_key,
                ingest_run_id=ingest_run_id,
                tenant_id=config.tenant_id,
                total_elements=extract_result.total_elements,
                max_depth=extract_result.max_depth,
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
                total_elements=extract_result.total_elements,
                max_depth=extract_result.max_depth,
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
                total_elements=extract_result.total_elements,
                max_depth=extract_result.max_depth,
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

            metadata = XMLChunkMetadata(
                source_uri=source_uri_final,
                source_format="xml",
                ingestion_method="xml_extract",
                parser_version=config.parser_version,
                chunk_index=idx,
                chunk_hash=chunk_hash,
                ingest_key=ingest_key,
                ingest_run_id=ingest_run_id,
                tenant_id=config.tenant_id,
                root_tag=extract_result.root_tag,
                total_elements=extract_result.total_elements,
                max_depth=extract_result.max_depth,
                namespace_count=len(extract_result.namespaces),
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
            self._vector_store.upsert_chunks(collection, payloads)
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
                total_elements=extract_result.total_elements,
                max_depth=extract_result.max_depth,
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
                total_elements=extract_result.total_elements,
                max_depth=extract_result.max_depth,
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
            "ingestkit_xml | file=%s | ingest_key=%s | elements=%d | depth=%d | "
            "chunks=%d | time=%.1fs",
            filename,
            ingest_key[:8],
            extract_result.total_elements,
            extract_result.max_depth,
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
            total_elements=extract_result.total_elements,
            max_depth=extract_result.max_depth,
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
