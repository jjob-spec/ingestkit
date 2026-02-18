"""EmailRouter -- orchestrator and public API for the ingestkit-email pipeline.

Routes email files through: security scan, ingest key computation,
conversion (EML or MSG), embedding, and vector store upsert.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
import uuid

from ingestkit_core.idempotency import compute_ingest_key
from ingestkit_core.models import (
    ChunkPayload,
    EmbedStageResult,
    WrittenArtifacts,
)
from ingestkit_core.protocols import EmbeddingBackend, VectorStoreBackend

from ingestkit_email.config import EmailProcessorConfig
from ingestkit_email.converters.base import EmailContent
from ingestkit_email.converters.eml import EMLConverter
from ingestkit_email.converters.msg import MSGConverter
from ingestkit_email.errors import ErrorCode, IngestError
from ingestkit_email.models import (
    EmailChunkMetadata,
    EmailContentType,
    EmailMetadata,
    EmailType,
    ProcessingResult,
)
from ingestkit_email.security import EmailSecurityScanner

logger = logging.getLogger("ingestkit_email")


class EmailRouter:
    """Top-level orchestrator for the ingestkit-email pipeline.

    Parameters
    ----------
    vector_store:
        Backend for vector storage (e.g. Qdrant).
    embedder:
        Backend for text embedding.
    config:
        Pipeline configuration. Uses defaults when *None*.
    """

    def __init__(
        self,
        vector_store: VectorStoreBackend,
        embedder: EmbeddingBackend,
        config: EmailProcessorConfig | None = None,
    ) -> None:
        self._config = config or EmailProcessorConfig()
        self._vector_store = vector_store
        self._embedder = embedder
        self._security_scanner = EmailSecurityScanner(self._config)
        self._eml_converter = EMLConverter()
        self._msg_converter = MSGConverter()

    def can_handle(self, file_path: str) -> bool:
        """Return True if *file_path* is a supported email format."""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in (".eml", ".msg")

    def process(
        self,
        file_path: str,
        source_uri: str | None = None,
    ) -> ProcessingResult:
        """Process a single email file through the full pipeline.

        Parameters
        ----------
        file_path:
            Filesystem path to the email file.
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
        fatal_errors = [e for e in security_errors if e.code.value.startswith("E_")]

        if fatal_errors:
            elapsed = time.monotonic() - overall_start
            logger.error(
                "ingestkit_email | file=%s | code=%s | detail=%s",
                filename,
                fatal_errors[0].code.value,
                fatal_errors[0].message,
            )
            return ProcessingResult(
                file_path=file_path,
                ingest_key="",
                ingest_run_id=ingest_run_id,
                tenant_id=config.tenant_id,
                errors=[e.code.value for e in fatal_errors],
                error_details=fatal_errors,
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
        # Step 3: Select and Run Converter
        # ==============================================================
        ext = os.path.splitext(file_path)[1].lower()
        email_type = EmailType.EML if ext == ".eml" else EmailType.MSG

        try:
            if email_type == EmailType.EML:
                content = self._eml_converter.convert(file_path, config)
            else:
                content = self._msg_converter.convert(file_path, config)
        except Exception as exc:
            elapsed = time.monotonic() - overall_start
            err = IngestError(
                code=ErrorCode.E_EMAIL_PARSE_FAILED,
                message=f"Failed to parse email: {exc}",
                stage="convert",
            )
            logger.error(
                "ingestkit_email | file=%s | code=%s | detail=%s",
                filename,
                ErrorCode.E_EMAIL_PARSE_FAILED.value,
                str(exc),
            )
            return ProcessingResult(
                file_path=file_path,
                ingest_key=ingest_key,
                ingest_run_id=ingest_run_id,
                tenant_id=config.tenant_id,
                errors=[ErrorCode.E_EMAIL_PARSE_FAILED.value],
                error_details=[err],
                processing_time_seconds=elapsed,
            )

        # ==============================================================
        # Step 4: Validate Body
        # ==============================================================
        if not content.body_text.strip():
            elapsed = time.monotonic() - overall_start
            err = IngestError(
                code=ErrorCode.E_EMAIL_EMPTY_BODY,
                message="Email has no extractable body text",
                stage="convert",
            )
            return ProcessingResult(
                file_path=file_path,
                ingest_key=ingest_key,
                ingest_run_id=ingest_run_id,
                tenant_id=config.tenant_id,
                errors=[ErrorCode.E_EMAIL_EMPTY_BODY.value],
                error_details=[err],
                processing_time_seconds=elapsed,
            )

        # ==============================================================
        # Step 5: Collect Warnings
        # ==============================================================
        warnings: list[str] = []
        warning_details: list[IngestError] = []

        if content.attachment_names:
            for att_name in content.attachment_names:
                warnings.append(ErrorCode.W_EMAIL_ATTACHMENT_SKIPPED.value)
                warning_details.append(
                    IngestError(
                        code=ErrorCode.W_EMAIL_ATTACHMENT_SKIPPED,
                        message=f"Attachment skipped: {att_name}",
                        stage="convert",
                        recoverable=True,
                    )
                )

        if content.body_source == "html_converted":
            warnings.append(ErrorCode.W_EMAIL_HTML_ONLY.value)
            warning_details.append(
                IngestError(
                    code=ErrorCode.W_EMAIL_HTML_ONLY,
                    message="Email body was converted from HTML (no plain text part)",
                    stage="convert",
                    recoverable=True,
                )
            )

        if not content.subject:
            warnings.append(ErrorCode.W_EMAIL_NO_SUBJECT.value)
            warning_details.append(
                IngestError(
                    code=ErrorCode.W_EMAIL_NO_SUBJECT,
                    message="Email has no Subject header",
                    stage="convert",
                    recoverable=True,
                )
            )

        if not content.date:
            warnings.append(ErrorCode.W_EMAIL_NO_DATE.value)
            warning_details.append(
                IngestError(
                    code=ErrorCode.W_EMAIL_NO_DATE,
                    message="Email has no Date header",
                    stage="convert",
                    recoverable=True,
                )
            )

        # ==============================================================
        # Step 6: Build Email Metadata
        # ==============================================================
        content_type = (
            EmailContentType.HTML_CONVERTED
            if content.body_source == "html_converted"
            else EmailContentType.PLAIN_TEXT
        )

        email_metadata = EmailMetadata(
            from_address=content.from_address,
            to_address=content.to_address,
            cc_address=content.cc_address,
            date=content.date,
            subject=content.subject,
            message_id=content.message_id,
            content_type=content_type,
            attachment_count=len(content.attachment_names),
            has_html_body=content.body_source == "html_converted",
            has_plain_body=content.body_source == "plain",
        )

        # ==============================================================
        # Step 7: Format Chunk Text
        # ==============================================================
        chunk_text = self._format_chunk_text(content, config)

        # ==============================================================
        # Step 8: Compute Chunk Hash and Build Payload
        # ==============================================================
        chunk_hash = hashlib.sha256(chunk_text.encode()).hexdigest()
        chunk_id = f"{ingest_key[:16]}_{chunk_hash[:16]}"

        # ==============================================================
        # Step 9: Embed
        # ==============================================================
        embed_start = time.monotonic()
        try:
            vectors = self._embedder.embed([chunk_text])
            vector = vectors[0]
            embed_duration = time.monotonic() - embed_start
        except Exception as exc:
            elapsed = time.monotonic() - overall_start
            err = IngestError(
                code=ErrorCode.E_BACKEND_EMBED_TIMEOUT,
                message=f"Embedding failed: {exc}",
                stage="embed",
            )
            return ProcessingResult(
                file_path=file_path,
                ingest_key=ingest_key,
                ingest_run_id=ingest_run_id,
                tenant_id=config.tenant_id,
                email_metadata=email_metadata,
                errors=[ErrorCode.E_BACKEND_EMBED_TIMEOUT.value],
                warnings=warnings,
                error_details=[err] + warning_details,
                processing_time_seconds=elapsed,
            )

        embed_result = EmbedStageResult(
            texts_embedded=1,
            embedding_dimension=len(vector),
            embed_duration_seconds=embed_duration,
        )

        # ==============================================================
        # Step 10: Build Chunk Metadata and Payload
        # ==============================================================
        chunk_metadata = EmailChunkMetadata(
            source_uri=source_uri or file_path,
            source_format="email",
            ingestion_method="email_conversion",
            parser_version=config.parser_version,
            chunk_index=0,
            chunk_hash=chunk_hash,
            ingest_key=ingest_key,
            ingest_run_id=ingest_run_id,
            tenant_id=config.tenant_id,
            email_type=email_type,
            from_address=content.from_address,
            to_address=content.to_address,
            subject=content.subject,
            date=content.date,
            message_id=content.message_id,
            content_type=content_type,
        )

        chunk_payload = ChunkPayload(
            id=chunk_id,
            text=chunk_text,
            vector=vector,
            metadata=chunk_metadata,
        )

        # ==============================================================
        # Step 11: Upsert to Vector Store
        # ==============================================================
        try:
            self._vector_store.ensure_collection(
                config.default_collection, config.embedding_dimension
            )
            upserted = self._vector_store.upsert_chunks(
                config.default_collection, [chunk_payload]
            )
        except Exception as exc:
            elapsed = time.monotonic() - overall_start
            err = IngestError(
                code=ErrorCode.E_BACKEND_VECTOR_CONNECT,
                message=f"Vector store upsert failed: {exc}",
                stage="upsert",
            )
            return ProcessingResult(
                file_path=file_path,
                ingest_key=ingest_key,
                ingest_run_id=ingest_run_id,
                tenant_id=config.tenant_id,
                email_metadata=email_metadata,
                embed_result=embed_result,
                errors=[ErrorCode.E_BACKEND_VECTOR_CONNECT.value],
                warnings=warnings,
                error_details=[err] + warning_details,
                processing_time_seconds=elapsed,
            )

        written = WrittenArtifacts(
            vector_point_ids=[chunk_id],
            vector_collection=config.default_collection,
        )

        # ==============================================================
        # Step 12: Assemble Result
        # ==============================================================
        elapsed = time.monotonic() - overall_start

        logger.info(
            "ingestkit_email | file=%s | ingest_key=%s | type=%s | "
            "content_type=%s | chunks=%d | time=%.1fs",
            filename,
            ingest_key[:8],
            email_type.value,
            content_type.value,
            1,
            elapsed,
        )

        return ProcessingResult(
            file_path=file_path,
            ingest_key=ingest_key,
            ingest_run_id=ingest_run_id,
            tenant_id=config.tenant_id,
            email_metadata=email_metadata,
            embed_result=embed_result,
            chunks_created=1,
            written=written,
            errors=[],
            warnings=warnings,
            error_details=warning_details,
            processing_time_seconds=elapsed,
        )

    @staticmethod
    def _format_chunk_text(
        content: EmailContent, config: EmailProcessorConfig
    ) -> str:
        """Format the chunk text from email content.

        When ``config.include_headers`` is True, prepends a header block
        to the body text.
        """
        parts: list[str] = []

        if config.include_headers:
            header_lines: list[str] = []
            if content.from_address:
                header_lines.append(f"From: {content.from_address}")
            if content.to_address:
                header_lines.append(f"To: {content.to_address}")
            if content.cc_address:
                header_lines.append(f"Cc: {content.cc_address}")
            if content.date:
                header_lines.append(f"Date: {content.date}")
            if content.subject:
                header_lines.append(f"Subject: {content.subject}")
            if header_lines:
                parts.append("\n".join(header_lines))
                parts.append("")  # blank line separator

        parts.append(content.body_text)
        return "\n".join(parts)


def create_default_router(**overrides) -> EmailRouter:
    """Create an EmailRouter with default backends.

    Convenience factory for local development and testing.

    Returns
    -------
    EmailRouter
        A fully-configured router ready for ``process()`` calls.

    Raises
    ------
    ImportError
        If optional backend dependencies are not installed.
    """
    from ingestkit_excel.backends import (
        OllamaEmbedding,
        QdrantVectorStore,
    )

    router_keys = {"vector_store", "embedder", "config"}
    router_kwargs = {k: v for k, v in overrides.items() if k in router_keys}
    config_kwargs = {k: v for k, v in overrides.items() if k not in router_keys}

    config = router_kwargs.pop("config", None)
    if config is None and config_kwargs:
        config = EmailProcessorConfig(**config_kwargs)
    elif config is None:
        config = EmailProcessorConfig()

    vector_store = router_kwargs.pop("vector_store", None)
    if vector_store is None:
        vector_store = QdrantVectorStore()

    embedder = router_kwargs.pop("embedder", None)
    if embedder is None:
        embedder = OllamaEmbedding(model=config.embedding_model)

    return EmailRouter(
        vector_store=vector_store,
        embedder=embedder,
        config=config,
    )
