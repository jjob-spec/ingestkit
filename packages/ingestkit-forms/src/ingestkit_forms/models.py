"""Pydantic v2 data models for ingestkit-forms.

Defines all data structures for the form ingestor plugin: template definitions,
matching results, extraction results, processing results, chunk metadata, and
request/response models. See spec sections 5, 6, 8, 9, 10.

Shared types (``IngestKey``, ``EmbedStageResult``, ``WrittenArtifacts``,
``BaseChunkMetadata``) are re-exported from ``ingestkit_core`` for convenience.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_serializer, model_validator

from ingestkit_core.models import (
    BaseChunkMetadata,
    EmbedStageResult,
    IngestKey,
    WrittenArtifacts,
)
from ingestkit_forms.errors import FormIngestError


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SourceFormat(str, Enum):
    """File format the template was designed for."""

    PDF = "pdf"
    XLSX = "xlsx"
    IMAGE = "image"


class FieldType(str, Enum):
    """Data type of a form field."""

    TEXT = "text"
    NUMBER = "number"
    DATE = "date"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    SIGNATURE = "signature"
    DROPDOWN = "dropdown"


class DualWriteMode(str, Enum):
    """Controls dual-write failure semantics."""

    BEST_EFFORT = "best_effort"
    STRICT_ATOMIC = "strict_atomic"


class TemplateStatus(str, Enum):
    """Lifecycle status of a form template."""

    DRAFT = "draft"
    APPROVED = "approved"
    ARCHIVED = "archived"


# ---------------------------------------------------------------------------
# Template Models (spec section 5.1)
# ---------------------------------------------------------------------------


class BoundingBox(BaseModel):
    """Normalized bounding box for a field region.

    All coordinates are normalized to 0.0-1.0 relative to page dimensions.
    This ensures templates work across different scan resolutions, DPIs,
    and page sizes.
    """

    x: float = Field(ge=0.0, le=1.0, description="Left edge, normalized 0.0-1.0")
    y: float = Field(ge=0.0, le=1.0, description="Top edge, normalized 0.0-1.0")
    width: float = Field(gt=0.0, le=1.0, description="Width, normalized 0.0-1.0")
    height: float = Field(gt=0.0, le=1.0, description="Height, normalized 0.0-1.0")


class CellAddress(BaseModel):
    """Excel cell address or range for cell-mapping extraction.

    Used only when source_format is XLSX. Specifies the exact cell(s) to
    read for a given field.
    """

    cell: str = Field(description="Cell address, e.g. 'B2' or range 'D5:D7'")
    sheet_name: str | None = Field(
        default=None,
        description="Sheet name. None means the active/first sheet.",
    )


class FieldMapping(BaseModel):
    """Mapping of a single form field within a template.

    Defines where the field is located, what type of data it contains,
    and how to validate the extracted value.
    """

    field_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this field within the template.",
    )
    field_name: str = Field(
        description="Machine-readable name, e.g. 'employee_name', 'leave_type'.",
    )
    field_label: str = Field(
        description="Human-readable display label, e.g. 'Employee Name'.",
    )
    field_type: FieldType
    page_number: int = Field(
        ge=0,
        description="0-indexed page number where this field appears.",
    )
    region: BoundingBox | None = Field(
        default=None,
        description="Bounding box for PDF/image extraction. None for Excel.",
    )
    cell_address: CellAddress | None = Field(
        default=None,
        description="Cell address for Excel extraction. None for PDF/image.",
    )
    required: bool = False
    validation_pattern: str | None = Field(
        default=None,
        description="Regex pattern for value validation.",
    )
    default_value: str | None = None
    sensitive: bool = Field(
        default=False,
        description="Whether this field contains sensitive/PII data.",
    )
    options: list[str] | None = Field(
        default=None,
        description="Valid options for CHECKBOX, RADIO, or DROPDOWN fields.",
    )
    extraction_hint: str | None = Field(
        default=None,
        description=(
            "Optional hint to improve OCR accuracy, e.g. 'numeric_only', "
            "'uppercase', 'date_format:MM/DD/YYYY'."
        ),
    )

    @model_validator(mode="after")
    def _validate_address_type(self) -> FieldMapping:
        """Ensure exactly one addressing scheme is set.

        PDF/image templates require ``region``; Excel templates require
        ``cell_address``. A field must have exactly one of the two.
        """
        if self.region is None and self.cell_address is None:
            raise ValueError(
                f"Field '{self.field_name}' must have either 'region' "
                f"(for PDF/image) or 'cell_address' (for Excel)."
            )
        if self.region is not None and self.cell_address is not None:
            raise ValueError(
                f"Field '{self.field_name}' cannot have both 'region' and "
                f"'cell_address'. Use one addressing scheme."
            )
        return self


class FormTemplate(BaseModel):
    """Reusable form template definition.

    Represents the complete structural definition of a form type.
    Created by an admin via the UI; applied to incoming documents
    during extraction.
    """

    template_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique template identifier (UUID).",
    )
    name: str = Field(
        description="Human-readable template name, e.g. 'W-4 2026'.",
    )
    description: str = Field(
        default="",
        description="Optional description of the form type.",
    )
    version: int = Field(
        default=1,
        ge=1,
        description="Template version. Incremented when the form layout changes.",
    )
    source_format: SourceFormat
    page_count: int = Field(
        ge=1,
        description="Number of pages in the form.",
    )
    fields: list[FieldMapping] = Field(
        min_length=1,
        max_length=200,
        description="At least one field mapping is required. Max 200 fields.",
    )
    layout_fingerprint: bytes | None = Field(
        default=None,
        description="Structural fingerprint for auto-matching.",
    )
    thumbnail: bytes | None = Field(
        default=None,
        description="Optional thumbnail image for display in admin UI.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    created_by: str = Field(
        default="system",
        description="User or system identifier that created this template.",
    )
    tenant_id: str | None = None
    status: TemplateStatus = Field(
        default=TemplateStatus.DRAFT,
        description="Lifecycle status: draft, approved, or archived.",
    )
    approved_by: str | None = Field(
        default=None,
        description="User identifier who approved this template.",
    )
    approved_at: datetime | None = Field(
        default=None,
        description="Timestamp when the template was approved.",
    )

    @field_serializer("layout_fingerprint", "thumbnail")
    @classmethod
    def _serialize_bytes(cls, v: bytes | None) -> str | None:
        """Serialize bytes fields to hex strings for JSON output."""
        return v.hex() if v else None


# ---------------------------------------------------------------------------
# Matching Models (spec section 6.2, 6.3)
# ---------------------------------------------------------------------------


class TemplateMatch(BaseModel):
    """Result of matching a document against a single template."""

    template_id: str
    template_name: str
    template_version: int
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall match confidence.",
    )
    per_page_confidence: list[float] = Field(
        description="Per-page similarity scores.",
    )
    matched_features: list[str] = Field(
        description=(
            "Which features contributed to the match: "
            "'layout_grid', 'text_anchors', 'field_positions'."
        ),
    )


class FormIngestRequest(BaseModel):
    """Request to ingest a form document.

    If template_id is provided, auto-detection is skipped and the
    specified template is applied directly.
    """

    file_path: str
    template_id: str | None = None
    template_version: int | None = Field(
        default=None,
        description="Specific version. None means latest.",
    )
    tenant_id: str | None = None
    source_uri: str | None = None
    metadata: dict[str, str] | None = None


# ---------------------------------------------------------------------------
# Extraction Result Models (spec section 10.1)
# ---------------------------------------------------------------------------


class ExtractedField(BaseModel):
    """A single extracted field value with confidence and provenance."""

    field_id: str
    field_name: str
    field_label: str
    field_type: FieldType
    value: str | bool | float | None = Field(
        description="Extracted value. None if extraction failed or confidence below threshold.",
    )
    raw_value: str | None = Field(
        default=None,
        description="Raw string before type coercion. Useful for debugging.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Extraction confidence for this field.",
    )
    extraction_method: str = Field(
        description=(
            "'native_fields', 'ocr_overlay', 'cell_mapping', "
            "'native_fields_with_ocr_fallback', or 'vlm_fallback'."
        ),
    )
    bounding_box: BoundingBox | None = Field(
        default=None,
        description="Actual bounding box where the value was found.",
    )
    validation_passed: bool | None = Field(
        default=None,
        description="True if validation_pattern matched, False if failed, None if no pattern.",
    )
    warnings: list[str] = []
    redacted: bool = Field(
        default=False,
        description="True if this field's value was redacted during output.",
    )


class FormExtractionResult(BaseModel):
    """Complete extraction result for a single form instance.

    This is the intermediate result before dual-write. It contains all
    extracted fields and metadata needed to produce both DB rows and chunks.
    """

    form_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID for this form instance.",
    )
    template_id: str
    template_name: str
    template_version: int
    source_uri: str
    source_format: str
    fields: list[ExtractedField]
    overall_confidence: float
    extraction_method: str = Field(
        description="Primary extraction method used: 'native_fields', 'ocr_overlay', 'cell_mapping'.",
    )
    match_method: str = Field(
        description="'auto_detect' or 'manual_override'.",
    )
    match_confidence: float | None = Field(
        default=None,
        description="Template match confidence. None if manual override.",
    )
    pages_processed: int
    extraction_duration_seconds: float
    warnings: list[str] = []


# ---------------------------------------------------------------------------
# Chunk Models (spec section 8.3, 8.4)
# ---------------------------------------------------------------------------


class FormChunkMetadata(BaseChunkMetadata):
    """Metadata attached to every form chunk for vector store upsert.

    Extends ``BaseChunkMetadata`` from core with form-specific fields.
    Inherits standard fields: ``source_uri``, ``source_format``,
    ``ingestion_method``, ``parser_version``, ``chunk_index``, ``chunk_hash``,
    ``ingest_key``, ``ingest_run_id``, ``tenant_id``, etc.

    Note: ``ingest_run_id`` is inherited as ``str | None = None`` from core.
    Callers should always provide a value for form chunks.
    """

    ingestion_method: str = "form_extraction"

    # Form-specific fields
    template_id: str
    template_name: str
    template_version: int
    form_id: str = Field(
        description="Unique ID for this form instance (matches DB row _form_id).",
    )
    field_names: list[str] = Field(
        description="Names of fields included in this chunk.",
    )
    extraction_method: str  # "native_fields", "ocr_overlay", "cell_mapping"
    overall_confidence: float
    per_field_confidence: dict[str, float] = Field(
        description="Map of field_name -> confidence for fields in this chunk.",
    )
    form_date: str | None = Field(
        default=None,
        description="Date extracted from the form content, if a date field exists.",
    )
    page_numbers: list[int] = Field(
        description="Page numbers covered by this chunk.",
    )
    match_method: str = Field(
        description="How the template was matched: 'auto_detect' or 'manual_override'.",
    )


class FormChunkPayload(BaseModel):
    """A single form chunk ready for vector store upsert."""

    id: str
    text: str
    vector: list[float]
    metadata: FormChunkMetadata


# ---------------------------------------------------------------------------
# Result Models (spec section 8.5, 10.2)
# ---------------------------------------------------------------------------


class FormWrittenArtifacts(WrittenArtifacts):
    """Written artifact IDs extended with form-specific DB row tracking.

    Extends core ``WrittenArtifacts`` with ``db_row_ids`` for form-level
    row identification, enabling targeted rollback of DB writes.
    """

    db_row_ids: list[str] = []


class FormProcessingResult(BaseModel):
    """Final result returned after processing a form document.

    Follows the same pattern as ProcessingResult in ingestkit-excel.
    """

    file_path: str
    ingest_key: str
    ingest_run_id: str
    tenant_id: str | None = None

    extraction_result: FormExtractionResult
    embed_result: EmbedStageResult | None = None

    chunks_created: int
    tables_created: int
    tables: list[str]
    written: FormWrittenArtifacts

    errors: list[str]
    warnings: list[str]
    error_details: list[FormIngestError] = []

    processing_time_seconds: float


class RollbackResult(BaseModel):
    """Result of a rollback operation."""

    vector_points_deleted: int = 0
    db_rows_deleted: int = 0
    errors: list[str] = []
    fully_rolled_back: bool = True


# ---------------------------------------------------------------------------
# Request/Response Models (spec section 9.2)
# ---------------------------------------------------------------------------


class FormTemplateCreateRequest(BaseModel):
    """Request to create a new form template."""

    name: str
    description: str = ""
    source_format: SourceFormat
    sample_file_path: str = Field(
        description="Path to a sample form document used to compute fingerprint.",
    )
    page_count: int = Field(ge=1)
    fields: list[FieldMapping] = Field(min_length=1, max_length=200)
    tenant_id: str | None = None
    created_by: str = "system"
    initial_status: str = Field(
        default="draft",
        description="Initial template status. Must be a valid TemplateStatus value.",
    )


class FormTemplateUpdateRequest(BaseModel):
    """Request to update a form template (creates a new version)."""

    name: str | None = None
    description: str | None = None
    sample_file_path: str | None = Field(
        default=None,
        description="New sample document for recomputing fingerprint.",
    )
    page_count: int | None = None
    fields: list[FieldMapping] | None = Field(default=None, max_length=200)


class ExtractionPreview(BaseModel):
    """Preview of extraction results without persistence."""

    template_id: str
    template_name: str
    template_version: int
    fields: list[ExtractedField]
    overall_confidence: float
    extraction_method: str
    warnings: list[str]


# ---------------------------------------------------------------------------
# Re-exports from ingestkit-core
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "SourceFormat",
    "FieldType",
    "DualWriteMode",
    "TemplateStatus",
    # Template
    "BoundingBox",
    "CellAddress",
    "FieldMapping",
    "FormTemplate",
    # Matching
    "TemplateMatch",
    "FormIngestRequest",
    # Extraction
    "ExtractedField",
    "FormExtractionResult",
    # Chunks
    "FormChunkMetadata",
    "FormChunkPayload",
    # Results
    "FormWrittenArtifacts",
    "FormProcessingResult",
    "RollbackResult",
    # Request/Response
    "FormTemplateCreateRequest",
    "FormTemplateUpdateRequest",
    "ExtractionPreview",
    # Core re-exports
    "IngestKey",
    "EmbedStageResult",
    "WrittenArtifacts",
    "BaseChunkMetadata",
]
