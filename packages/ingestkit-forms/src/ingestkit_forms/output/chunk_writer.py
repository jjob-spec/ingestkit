"""RAG chunk serializer and embedder for form data.

Converts extracted form fields into FormChunkPayload objects with
embeddings for vector store ingestion.

See spec sections 8.2-8.4 for authoritative definitions.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone

from ingestkit_forms.config import FormProcessorConfig
from ingestkit_forms.errors import FormErrorCode, FormIngestException
from ingestkit_forms.models import (
    EmbedStageResult,
    ExtractedField,
    FieldType,
    FormChunkMetadata,
    FormChunkPayload,
    FormExtractionResult,
    FormTemplate,
)
from ingestkit_forms.protocols import EmbeddingBackend, VectorStoreBackend

logger = logging.getLogger("ingestkit_forms")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def serialize_field_value(field: ExtractedField) -> str:
    """Serialize a single field value to text.

    None -> ``[not extracted]``; CHECKBOX/SIGNATURE booleans -> Yes/No;
    everything else -> ``str(value)``.
    """
    if field.value is None:
        return "[not extracted]"
    if field.field_type in (FieldType.CHECKBOX, FieldType.SIGNATURE):
        return "Yes" if field.value else "No"
    return str(field.value)


def serialize_form_to_text(
    extraction: FormExtractionResult, fields: list[ExtractedField]
) -> str:
    """Serialize a form (or chunk of fields) into the spec-defined text format.

    Format per spec 8.2::

        Form: {template_name} (v{version})
        Date Extracted: {YYYY-MM-DD}

        {field_label}: {value}
        ...
    """
    lines: list[str] = [
        f"Form: {extraction.template_name} (v{extraction.template_version})",
        f"Date Extracted: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        "",
    ]
    for field in fields:
        lines.append(f"{field.field_label}: {serialize_field_value(field)}")
    return "\n".join(lines)


def split_fields_into_chunks(
    fields: list[ExtractedField],
    chunk_max_fields: int,
    template: FormTemplate,
) -> list[list[ExtractedField]]:
    """Split fields into chunks respecting page boundaries.

    If the total number of fields fits within ``chunk_max_fields``,
    a single chunk is returned. Otherwise, fields are grouped by page
    number (looked up from the template's ``FieldMapping`` objects by
    ``field_id``). If a single page exceeds ``chunk_max_fields``, it
    is kept as one chunk (no splitting within a page).

    The ``template`` parameter is required because ``ExtractedField``
    does not carry ``page_number`` -- that information lives on the
    ``FieldMapping`` in the template. This avoids modifying the shared
    ``ExtractedField`` model that other agents are concurrently working
    with (PLAN-CHECK ISSUE-1 fix).
    """
    if len(fields) <= chunk_max_fields:
        return [fields]

    # Build field_id -> page_number lookup from template
    page_lookup: dict[str, int] = {
        fm.field_id: fm.page_number for fm in template.fields
    }

    # Group fields by page, preserving order
    page_groups: OrderedDict[int, list[ExtractedField]] = OrderedDict()
    for field in fields:
        page_num = page_lookup.get(field.field_id, 0)
        if page_num not in page_groups:
            page_groups[page_num] = []
        page_groups[page_num].append(field)

    return list(page_groups.values())


def build_chunk_metadata(
    extraction: FormExtractionResult,
    chunk_fields: list[ExtractedField],
    chunk_index: int,
    chunk_hash: str,
    ingest_key: str,
    ingest_run_id: str,
    config: FormProcessorConfig,
    template: FormTemplate,
) -> FormChunkMetadata:
    """Build metadata for a single form chunk.

    Populates all base fields from ``BaseChunkMetadata`` plus the 11
    form-specific fields defined in ``FormChunkMetadata``.
    """
    # Build field_id -> page_number lookup from template
    page_lookup: dict[str, int] = {
        fm.field_id: fm.page_number for fm in template.fields
    }

    field_names = [f.field_name for f in chunk_fields]
    per_field_conf = {f.field_name: f.confidence for f in chunk_fields}

    # Extract page numbers for this chunk's fields
    page_numbers = sorted(
        {page_lookup.get(f.field_id, 0) for f in chunk_fields}
    )

    # Scan for a date field value
    form_date: str | None = None
    for f in chunk_fields:
        if f.field_type == FieldType.DATE and f.value is not None:
            form_date = str(f.value)
            break

    return FormChunkMetadata(
        source_uri=extraction.source_uri,
        source_format=extraction.source_format,
        ingestion_method="form_extraction",
        parser_version=config.parser_version,
        chunk_index=chunk_index,
        chunk_hash=chunk_hash,
        ingest_key=ingest_key,
        ingest_run_id=ingest_run_id,
        tenant_id=config.tenant_id,
        template_id=extraction.template_id,
        template_name=extraction.template_name,
        template_version=extraction.template_version,
        form_id=extraction.form_id,
        field_names=field_names,
        extraction_method=extraction.extraction_method,
        overall_confidence=extraction.overall_confidence,
        per_field_confidence=per_field_conf,
        form_date=form_date,
        page_numbers=page_numbers,
        match_method=extraction.match_method,
    )


# ---------------------------------------------------------------------------
# FormChunkWriter
# ---------------------------------------------------------------------------


class FormChunkWriter:
    """Serializes form fields into RAG chunks, embeds, and upserts to vector store.

    Handles field splitting, text serialization, embedding with retry,
    and vector store upsert.
    """

    def __init__(
        self,
        vector_store: VectorStoreBackend,
        embedder: EmbeddingBackend,
        config: FormProcessorConfig,
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._config = config

    def write_chunks(
        self,
        extraction: FormExtractionResult,
        template: FormTemplate,
        source_uri: str,
        ingest_key: str,
        ingest_run_id: str,
    ) -> tuple[list[str], EmbedStageResult]:
        """Serialize, embed, and upsert form chunks.

        Returns ``(vector_point_ids, embed_result)``.
        """
        config = self._config

        # Split fields into chunks
        chunk_groups = split_fields_into_chunks(
            extraction.fields, config.chunk_max_fields, template
        )

        # Ensure vector collection exists
        self._vector_store.ensure_collection(
            config.default_collection, self._embedder.dimension()
        )

        # Build chunk payloads (without vectors yet)
        payloads: list[FormChunkPayload] = []
        texts: list[str] = []

        for i, chunk_fields in enumerate(chunk_groups):
            text = serialize_form_to_text(extraction, chunk_fields)
            chunk_hash = hashlib.sha256(text.encode()).hexdigest()
            chunk_id = str(
                uuid.uuid5(uuid.NAMESPACE_URL, f"{ingest_key}:{chunk_hash}")
            )
            metadata = build_chunk_metadata(
                extraction=extraction,
                chunk_fields=chunk_fields,
                chunk_index=i,
                chunk_hash=chunk_hash,
                ingest_key=ingest_key,
                ingest_run_id=ingest_run_id,
                config=config,
                template=template,
            )
            payloads.append(
                FormChunkPayload(
                    id=chunk_id, text=text, vector=[], metadata=metadata
                )
            )
            texts.append(text)

        # Embed with retry
        vectors = self._embed_with_retry(texts)

        # Assign vectors
        for payload, vector in zip(payloads, vectors):
            payload.vector = vector

        # Upsert with retry
        self._upsert_with_retry(config.default_collection, payloads)

        embed_result = EmbedStageResult(
            texts_embedded=len(texts),
            embedding_dimension=self._embedder.dimension(),
            embed_duration_seconds=0.0,  # Not measured at this level
        )

        point_ids = [p.id for p in payloads]
        return point_ids, embed_result

    def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Embed texts with retry logic."""
        max_attempts = self._config.backend_max_retries + 1
        last_exc: Exception | None = None

        for attempt in range(max_attempts):
            try:
                return self._embedder.embed(
                    texts, timeout=self._config.backend_timeout_seconds
                )
            except Exception as exc:
                last_exc = exc
                if attempt < max_attempts - 1:
                    sleep_time = self._config.backend_backoff_base * (2 ** attempt)
                    logger.warning(
                        "forms.write.embed_retry",
                        extra={
                            "attempt": attempt + 1,
                            "max_attempts": max_attempts,
                            "retry_delay_s": sleep_time,
                            "error": str(exc),
                        },
                    )
                    time.sleep(sleep_time)

        raise FormIngestException(
            code=FormErrorCode.E_FORM_CHUNK_WRITE_FAILED,
            message=f"Embedding failed after {max_attempts} attempts: {last_exc}",
            stage="output",
            recoverable=True,
        )

    def _upsert_with_retry(
        self, collection: str, payloads: list[FormChunkPayload]
    ) -> None:
        """Upsert chunks with retry logic."""
        max_attempts = self._config.backend_max_retries + 1
        last_exc: Exception | None = None

        for attempt in range(max_attempts):
            try:
                # VectorStoreBackend.upsert_chunks expects list[ChunkPayload]
                # FormChunkPayload has the same structural interface
                self._vector_store.upsert_chunks(collection, payloads)  # type: ignore[arg-type]
                return
            except Exception as exc:
                last_exc = exc
                if attempt < max_attempts - 1:
                    sleep_time = self._config.backend_backoff_base * (2 ** attempt)
                    logger.warning(
                        "forms.write.upsert_retry",
                        extra={
                            "attempt": attempt + 1,
                            "max_attempts": max_attempts,
                            "retry_delay_s": sleep_time,
                            "error": str(exc),
                        },
                    )
                    time.sleep(sleep_time)

        raise FormIngestException(
            code=FormErrorCode.E_FORM_CHUNK_WRITE_FAILED,
            message=f"Vector upsert failed after {max_attempts} attempts: {last_exc}",
            stage="output",
            recoverable=True,
        )
