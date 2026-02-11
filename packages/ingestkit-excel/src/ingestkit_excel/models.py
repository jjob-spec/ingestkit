"""Pydantic data models, enumerations, and stage artifacts for ingestkit-excel.

This module defines the complete data model layer referenced throughout the
pipeline: five classification/processing enums, the deterministic ``IngestKey``,
three typed stage-artifact models, and eight core domain models.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any

from pydantic import BaseModel

from ingestkit_excel.errors import IngestError


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class FileType(str, Enum):
    """Structural classification of an Excel file.

    Drives routing to the appropriate processing path:
    Type A (tabular_data), Type B (formatted_document), or Type C (hybrid).
    """

    TABULAR_DATA = "tabular_data"
    FORMATTED_DOCUMENT = "formatted_document"
    HYBRID = "hybrid"


class ClassificationTier(str, Enum):
    """Which detection tier produced the classification result.

    Tier 1 is rule-based, Tier 2 uses a lightweight LLM, and Tier 3
    escalates to a reasoning model for ambiguous files.
    """

    RULE_BASED = "rule_based"
    LLM_BASIC = "llm_basic"
    LLM_REASONING = "llm_reasoning"


class IngestionMethod(str, Enum):
    """Processing path used after classification.

    Maps directly to the three processor implementations:
    Path A (sql_agent), Path B (text_serialization), Path C (hybrid_split).
    """

    SQL_AGENT = "sql_agent"
    TEXT_SERIALIZATION = "text_serialization"
    HYBRID_SPLIT = "hybrid_split"


class RegionType(str, Enum):
    """Type of a detected region within a worksheet (used by Path C splitter)."""

    DATA_TABLE = "data_table"
    TEXT_BLOCK = "text_block"
    HEADER_BLOCK = "header_block"
    FOOTER_BLOCK = "footer_block"
    MATRIX_BLOCK = "matrix_block"
    CHART_ONLY = "chart_only"
    EMPTY = "empty"


class ParserUsed(str, Enum):
    """Which parser successfully processed a sheet in the fallback chain."""

    OPENPYXL = "openpyxl"
    PANDAS_FALLBACK = "pandas_fallback"
    RAW_TEXT_FALLBACK = "raw_text_fallback"


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


class ParseStageResult(BaseModel):
    """Typed output of the parsing stage."""

    parser_used: ParserUsed
    fallback_reason_code: str | None = None
    sheets_parsed: int
    sheets_skipped: int
    skipped_reasons: dict[str, str]
    parse_duration_seconds: float


class ClassificationStageResult(BaseModel):
    """Typed output of the classification stage."""

    tier_used: ClassificationTier
    file_type: FileType
    confidence: float
    signals: dict[str, Any] | None = None
    reasoning: str
    per_sheet_types: dict[str, FileType] | None = None
    classification_duration_seconds: float


class EmbedStageResult(BaseModel):
    """Typed output of the embedding stage."""

    texts_embedded: int
    embedding_dimension: int
    embed_duration_seconds: float


# ---------------------------------------------------------------------------
# Core Models
# ---------------------------------------------------------------------------


class SheetProfile(BaseModel):
    """Structural profile of a single worksheet."""

    name: str
    row_count: int
    col_count: int
    merged_cell_count: int
    merged_cell_ratio: float
    header_row_detected: bool
    header_values: list[str]
    column_type_consistency: float
    numeric_ratio: float
    text_ratio: float
    empty_ratio: float
    sample_rows: list[list[str]]
    has_formulas: bool
    is_hidden: bool
    parser_used: ParserUsed


class FileProfile(BaseModel):
    """Aggregate structural profile of an Excel file."""

    file_path: str
    file_size_bytes: int
    sheet_count: int
    sheet_names: list[str]
    sheets: list[SheetProfile]
    has_password_protected_sheets: bool
    has_chart_only_sheets: bool
    total_merged_cells: int
    total_rows: int
    content_hash: str


class ClassificationResult(BaseModel):
    """Result of the tiered classification."""

    file_type: FileType
    confidence: float
    tier_used: ClassificationTier
    reasoning: str
    per_sheet_types: dict[str, FileType] | None = None
    signals: dict[str, Any] | None = None


class ChunkMetadata(BaseModel):
    """Standardized metadata attached to every chunk.

    Provides a canonical schema across all processing paths so that downstream
    consumers can uniformly query chunk provenance.
    """

    source_uri: str
    source_format: str = "xlsx"
    sheet_name: str
    region_id: str | None = None
    ingestion_method: str
    parser_used: str
    parser_version: str
    chunk_index: int
    chunk_hash: str
    ingest_key: str
    ingest_run_id: str
    tenant_id: str | None = None
    table_name: str | None = None
    db_uri: str | None = None
    row_count: int | None = None
    columns: list[str] | None = None
    section_title: str | None = None
    original_structure: str | None = None


class ChunkPayload(BaseModel):
    """A single chunk ready for vector store upsert."""

    id: str
    text: str
    vector: list[float]
    metadata: ChunkMetadata


class SheetRegion(BaseModel):
    """A detected region within a worksheet (used by Path C splitter)."""

    sheet_name: str
    region_id: str
    start_row: int
    end_row: int
    start_col: int
    end_col: int
    region_type: RegionType
    detection_confidence: float
    classified_as: FileType | None = None


class WrittenArtifacts(BaseModel):
    """IDs of everything written to backends, enabling caller-side rollback."""

    vector_point_ids: list[str] = []
    vector_collection: str | None = None
    db_table_names: list[str] = []


class ProcessingResult(BaseModel):
    """Final result returned after processing a file."""

    file_path: str
    ingest_key: str
    ingest_run_id: str
    tenant_id: str | None = None

    parse_result: ParseStageResult
    classification_result: ClassificationStageResult
    embed_result: EmbedStageResult | None = None

    classification: ClassificationResult
    ingestion_method: IngestionMethod

    chunks_created: int
    tables_created: int
    tables: list[str]
    written: WrittenArtifacts

    errors: list[str]
    warnings: list[str]
    error_details: list[IngestError] = []

    processing_time_seconds: float
