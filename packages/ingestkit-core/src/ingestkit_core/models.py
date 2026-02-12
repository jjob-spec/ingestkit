"""Shared Pydantic models and enumerations for the ingestkit framework.

Contains the six core types extracted from sibling packages:
``ClassificationTier``, ``IngestKey``, ``EmbedStageResult``,
``BaseChunkMetadata``, ``ChunkPayload``, and ``WrittenArtifacts``.
"""

from __future__ import annotations

import hashlib
from enum import Enum

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ClassificationTier(str, Enum):
    """Which detection tier produced the classification result.

    Tier 1 is rule-based, Tier 2 uses a lightweight LLM, and Tier 3
    escalates to a reasoning model for ambiguous files.
    """

    RULE_BASED = "rule_based"
    LLM_BASIC = "llm_basic"
    LLM_REASONING = "llm_reasoning"


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class IngestKey(BaseModel):
    """Deterministic key for deduplication.

    Combines content hash, source URI, parser version, and optional tenant ID
    into a single SHA-256 digest that callers can use to detect duplicate
    ingestion runs.
    """

    content_hash: str
    source_uri: str
    parser_version: str
    tenant_id: str | None = None

    @property
    def key(self) -> str:
        """Deterministic string key for dedup lookups."""
        parts = [self.content_hash, self.source_uri, self.parser_version]
        if self.tenant_id:
            parts.append(self.tenant_id)
        return hashlib.sha256("|".join(parts).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Stage Artifacts
# ---------------------------------------------------------------------------


class EmbedStageResult(BaseModel):
    """Typed output of the embedding stage."""

    texts_embedded: int
    embedding_dimension: int
    embed_duration_seconds: float


# ---------------------------------------------------------------------------
# Chunk Models
# ---------------------------------------------------------------------------


class BaseChunkMetadata(BaseModel):
    """Common metadata fields shared by all chunk types.

    Each package extends this with format-specific fields (e.g. Excel adds
    ``sheet_name``, PDF adds ``page_numbers``).  The ``source_format`` field
    has no default -- subclasses must provide one (e.g. ``"xlsx"``, ``"pdf"``).
    """

    source_uri: str
    source_format: str
    ingestion_method: str
    parser_version: str
    chunk_index: int
    chunk_hash: str
    ingest_key: str
    ingest_run_id: str | None = None
    tenant_id: str | None = None
    table_name: str | None = None
    row_count: int | None = None
    columns: list[str] | None = None
    section_title: str | None = None


class ChunkPayload(BaseModel):
    """A single chunk ready for vector store upsert."""

    id: str
    text: str
    vector: list[float]
    metadata: BaseChunkMetadata


# ---------------------------------------------------------------------------
# Result Models
# ---------------------------------------------------------------------------


class WrittenArtifacts(BaseModel):
    """IDs of everything written to backends, enabling caller-side rollback."""

    vector_point_ids: list[str] = []
    vector_collection: str | None = None
    db_table_names: list[str] = []
