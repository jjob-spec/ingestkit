"""Pydantic data models, enumerations, and stage artifacts for ingestkit-pdf.

This module defines the complete data model layer referenced throughout the
pipeline: eight classification/processing enums, the deterministic ``IngestKey``,
four typed stage-artifact models, per-page and document-level models, and the
final ``ProcessingResult``.

Shared types (``IngestKey``, ``ClassificationTier``, ``EmbedStageResult``,
``ChunkPayload``, ``WrittenArtifacts``, ``BaseChunkMetadata``) are re-exported
from ``ingestkit_core`` so that existing import paths continue to work.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel

from ingestkit_core.models import (
    BaseChunkMetadata,
    ChunkPayload,
    ClassificationTier,
    EmbedStageResult,
    IngestKey,
    WrittenArtifacts,
)
from ingestkit_pdf.errors import IngestError


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class PDFType(str, Enum):
    """Structural classification of a PDF file.

    Drives routing to the appropriate processing path:
    Type A (text_native), Type B (scanned), or Type C (complex).
    """

    TEXT_NATIVE = "text_native"
    SCANNED = "scanned"
    COMPLEX = "complex"


class PageType(str, Enum):
    """Classification of a single PDF page."""

    TEXT = "text"
    SCANNED = "scanned"
    TABLE_HEAVY = "table_heavy"
    FORM = "form"
    MIXED = "mixed"
    BLANK = "blank"
    VECTOR_ONLY = "vector_only"
    TOC = "toc"


class IngestionMethod(str, Enum):
    """Processing path used after classification."""

    TEXT_EXTRACTION = "text_extraction"
    OCR_PIPELINE = "ocr_pipeline"
    COMPLEX_PROCESSING = "complex_processing"


class OCREngine(str, Enum):
    """Available OCR engines."""

    TESSERACT = "tesseract"
    PADDLEOCR = "paddleocr"


class ExtractionQualityGrade(str, Enum):
    """Quality grade derived from the composite extraction score."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ContentType(str, Enum):
    """Content type within a page or chunk."""

    NARRATIVE = "narrative"
    TABLE = "table"
    LIST = "list"
    HEADING = "heading"
    FORM_FIELD = "form_field"
    IMAGE_DESCRIPTION = "image_description"
    FOOTER = "footer"
    HEADER = "header"


# ---------------------------------------------------------------------------
# Per-Page Models
# ---------------------------------------------------------------------------


class ExtractionQuality(BaseModel):
    """Quality assessment of text extraction on a page or document."""

    printable_ratio: float
    avg_words_per_page: float
    pages_with_text: int
    total_pages: int
    extraction_method: str

    @property
    def score(self) -> float:
        """Composite quality score 0.0-1.0."""
        coverage = self.pages_with_text / max(self.total_pages, 1)
        text_quality = min(self.printable_ratio, 1.0)
        density = min(self.avg_words_per_page / 100, 1.0)
        return coverage * 0.4 + text_quality * 0.4 + density * 0.2

    @property
    def grade(self) -> ExtractionQualityGrade:
        s = self.score
        if s >= 0.9:
            return ExtractionQualityGrade.HIGH
        if s >= 0.6:
            return ExtractionQualityGrade.MEDIUM
        return ExtractionQualityGrade.LOW


class PageProfile(BaseModel):
    """Structural profile of a single PDF page."""

    page_number: int
    text_length: int
    word_count: int
    image_count: int
    image_coverage_ratio: float
    table_count: int
    font_count: int
    font_names: list[str]
    has_form_fields: bool
    is_multi_column: bool
    page_type: PageType
    extraction_quality: ExtractionQuality


# ---------------------------------------------------------------------------
# Document-Level Models
# ---------------------------------------------------------------------------


class DocumentMetadata(BaseModel):
    """Metadata extracted from PDF document properties."""

    title: str | None = None
    author: str | None = None
    subject: str | None = None
    keywords: str | None = None
    creator: str | None = None
    producer: str | None = None
    creation_date: str | None = None
    modification_date: str | None = None
    pdf_version: str | None = None
    page_count: int = 0
    file_size_bytes: int = 0
    is_encrypted: bool = False
    needs_password: bool = False
    is_signed: bool = False
    has_form_fields: bool = False
    is_linearized: bool = False


class DocumentProfile(BaseModel):
    """Aggregate structural profile of a PDF file."""

    file_path: str
    file_size_bytes: int
    page_count: int
    content_hash: str
    metadata: DocumentMetadata
    pages: list[PageProfile]
    page_type_distribution: dict[str, int]
    detected_languages: list[str]
    has_toc: bool
    toc_entries: list[tuple[int, str, int]] | None = None
    overall_quality: ExtractionQuality
    security_warnings: list[str]


class ClassificationResult(BaseModel):
    """Result of the tiered classification."""

    pdf_type: PDFType
    confidence: float
    tier_used: ClassificationTier
    reasoning: str
    per_page_types: dict[int, PageType]
    signals: dict[str, Any] | None = None
    degraded: bool = False


# ---------------------------------------------------------------------------
# Processing Models
# ---------------------------------------------------------------------------


class OCRResult(BaseModel):
    """Per-page OCR extraction result."""

    page_number: int
    text: str
    confidence: float
    engine_used: OCREngine
    dpi: int
    preprocessing_steps: list[str]
    language_detected: str | None = None


class TableResult(BaseModel):
    """Extracted table from a PDF page."""

    page_number: int
    table_index: int
    row_count: int
    col_count: int
    headers: list[str] | None = None
    is_continuation: bool = False
    continuation_group_id: str | None = None


class PDFChunkMetadata(BaseChunkMetadata):
    """Standardized metadata attached to every chunk.

    Extends ``BaseChunkMetadata`` from core with PDF-specific fields.
    """

    source_format: str = "pdf"
    page_numbers: list[int]
    ingest_run_id: str  # type: ignore[assignment]  # PDF requires this (base has Optional)
    heading_path: list[str] | None = None
    content_type: str | None = None
    doc_title: str | None = None
    doc_author: str | None = None
    doc_date: str | None = None
    ocr_engine: str | None = None
    ocr_confidence: float | None = None
    ocr_dpi: int | None = None
    ocr_preprocessing: list[str] | None = None
    table_index: int | None = None
    language: str | None = None


# ---------------------------------------------------------------------------
# Stage Artifacts
# ---------------------------------------------------------------------------


class ParseStageResult(BaseModel):
    """Typed output of the PDF extraction stage."""

    pages_extracted: int
    pages_skipped: int
    skipped_reasons: dict[int, str]
    extraction_method: str
    overall_quality: ExtractionQuality
    parse_duration_seconds: float


class ClassificationStageResult(BaseModel):
    """Typed output of the classification stage."""

    tier_used: ClassificationTier
    pdf_type: PDFType
    confidence: float
    signals: dict[str, Any] | None = None
    reasoning: str
    per_page_types: dict[int, PageType]
    classification_duration_seconds: float
    degraded: bool = False


class OCRStageResult(BaseModel):
    """Typed output of the OCR stage (Path B and C)."""

    pages_ocrd: int
    engine_used: OCREngine
    avg_confidence: float
    low_confidence_pages: list[int]
    ocr_duration_seconds: float
    engine_fallback_used: bool = False


# ---------------------------------------------------------------------------
# Result Models
# ---------------------------------------------------------------------------


class ProcessingResult(BaseModel):
    """Final result returned after processing a file."""

    file_path: str
    ingest_key: str
    ingest_run_id: str
    tenant_id: str | None = None

    parse_result: ParseStageResult
    classification_result: ClassificationStageResult
    ocr_result: OCRStageResult | None = None
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
