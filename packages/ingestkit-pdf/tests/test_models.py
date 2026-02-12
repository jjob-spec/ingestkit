"""Tests for ingestkit_pdf.models — enumerations, computed properties, and model construction."""

from __future__ import annotations

import hashlib

import pytest

from ingestkit_pdf.models import (
    ClassificationResult,
    ClassificationStageResult,
    ClassificationTier,
    ContentType,
    DocumentMetadata,
    EmbedStageResult,
    ExtractionQuality,
    ExtractionQualityGrade,
    IngestKey,
    IngestionMethod,
    OCREngine,
    OCRResult,
    OCRStageResult,
    PDFChunkMetadata,
    PDFType,
    PageProfile,
    PageType,
    ParseStageResult,
    ProcessingResult,
    TableResult,
    WrittenArtifacts,
)


# ---------------------------------------------------------------------------
# Enum Tests
# ---------------------------------------------------------------------------


class TestPDFType:
    def test_values(self):
        assert PDFType.TEXT_NATIVE == "text_native"
        assert PDFType.SCANNED == "scanned"
        assert PDFType.COMPLEX == "complex"

    def test_member_count(self):
        assert len(PDFType) == 3


class TestPageType:
    def test_values(self):
        assert PageType.TEXT == "text"
        assert PageType.SCANNED == "scanned"
        assert PageType.TABLE_HEAVY == "table_heavy"
        assert PageType.FORM == "form"
        assert PageType.MIXED == "mixed"
        assert PageType.BLANK == "blank"
        assert PageType.VECTOR_ONLY == "vector_only"
        assert PageType.TOC == "toc"

    def test_member_count(self):
        assert len(PageType) == 8


class TestClassificationTier:
    def test_values(self):
        assert ClassificationTier.RULE_BASED == "rule_based"
        assert ClassificationTier.LLM_BASIC == "llm_basic"
        assert ClassificationTier.LLM_REASONING == "llm_reasoning"

    def test_member_count(self):
        assert len(ClassificationTier) == 3


class TestIngestionMethod:
    def test_values(self):
        assert IngestionMethod.TEXT_EXTRACTION == "text_extraction"
        assert IngestionMethod.OCR_PIPELINE == "ocr_pipeline"
        assert IngestionMethod.COMPLEX_PROCESSING == "complex_processing"

    def test_member_count(self):
        assert len(IngestionMethod) == 3


class TestOCREngine:
    def test_values(self):
        assert OCREngine.TESSERACT == "tesseract"
        assert OCREngine.PADDLEOCR == "paddleocr"

    def test_member_count(self):
        assert len(OCREngine) == 2


class TestExtractionQualityGrade:
    def test_values(self):
        assert ExtractionQualityGrade.HIGH == "high"
        assert ExtractionQualityGrade.MEDIUM == "medium"
        assert ExtractionQualityGrade.LOW == "low"

    def test_member_count(self):
        assert len(ExtractionQualityGrade) == 3


class TestContentType:
    def test_values(self):
        assert ContentType.NARRATIVE == "narrative"
        assert ContentType.TABLE == "table"
        assert ContentType.LIST == "list"
        assert ContentType.HEADING == "heading"
        assert ContentType.FORM_FIELD == "form_field"
        assert ContentType.IMAGE_DESCRIPTION == "image_description"
        assert ContentType.FOOTER == "footer"
        assert ContentType.HEADER == "header"

    def test_member_count(self):
        assert len(ContentType) == 8


# ---------------------------------------------------------------------------
# ExtractionQuality Tests
# ---------------------------------------------------------------------------


def _make_quality(
    printable_ratio: float = 0.95,
    avg_words_per_page: float = 200.0,
    pages_with_text: int = 10,
    total_pages: int = 10,
    extraction_method: str = "native",
) -> ExtractionQuality:
    return ExtractionQuality(
        printable_ratio=printable_ratio,
        avg_words_per_page=avg_words_per_page,
        pages_with_text=pages_with_text,
        total_pages=total_pages,
        extraction_method=extraction_method,
    )


class TestExtractionQuality:
    def test_perfect_score(self):
        q = _make_quality(
            printable_ratio=1.0,
            avg_words_per_page=200.0,
            pages_with_text=10,
            total_pages=10,
        )
        # coverage=1.0, text_quality=1.0, density=min(200/100,1)=1.0
        # score = 1.0*0.4 + 1.0*0.4 + 1.0*0.2 = 1.0
        assert q.score == pytest.approx(1.0)
        assert q.grade == ExtractionQualityGrade.HIGH

    def test_high_grade_boundary(self):
        # score exactly 0.9 → HIGH
        q = _make_quality(
            printable_ratio=0.85,
            avg_words_per_page=100.0,
            pages_with_text=10,
            total_pages=10,
        )
        # coverage=1.0, text_quality=0.85, density=1.0
        # score = 0.4 + 0.34 + 0.2 = 0.94
        assert q.score >= 0.9
        assert q.grade == ExtractionQualityGrade.HIGH

    def test_medium_grade(self):
        q = _make_quality(
            printable_ratio=0.7,
            avg_words_per_page=50.0,
            pages_with_text=8,
            total_pages=10,
        )
        # coverage=0.8, text_quality=0.7, density=0.5
        # score = 0.32 + 0.28 + 0.10 = 0.70
        assert 0.6 <= q.score < 0.9
        assert q.grade == ExtractionQualityGrade.MEDIUM

    def test_low_grade(self):
        q = _make_quality(
            printable_ratio=0.3,
            avg_words_per_page=10.0,
            pages_with_text=2,
            total_pages=10,
        )
        # coverage=0.2, text_quality=0.3, density=0.1
        # score = 0.08 + 0.12 + 0.02 = 0.22
        assert q.score < 0.6
        assert q.grade == ExtractionQualityGrade.LOW

    def test_zero_pages(self):
        q = _make_quality(total_pages=0, pages_with_text=0)
        # coverage = 0/max(0,1) = 0
        assert q.score >= 0.0

    def test_printable_ratio_clamped(self):
        q = _make_quality(printable_ratio=1.5)
        # text_quality = min(1.5, 1.0) = 1.0
        assert q.score <= 1.0

    def test_density_clamped(self):
        q = _make_quality(avg_words_per_page=500.0)
        # density = min(500/100, 1.0) = 1.0
        assert q.score <= 1.0

    def test_score_formula(self):
        q = _make_quality(
            printable_ratio=0.5,
            avg_words_per_page=60.0,
            pages_with_text=5,
            total_pages=10,
        )
        coverage = 5 / 10  # 0.5
        text_quality = 0.5
        density = 60 / 100  # 0.6
        expected = coverage * 0.4 + text_quality * 0.4 + density * 0.2
        assert q.score == pytest.approx(expected)

    def test_medium_high_boundary(self):
        # score just below 0.9 → MEDIUM
        q = _make_quality(
            printable_ratio=0.8,
            avg_words_per_page=100.0,
            pages_with_text=9,
            total_pages=10,
        )
        # coverage=0.9, text_quality=0.8, density=1.0
        # score = 0.36 + 0.32 + 0.2 = 0.88
        assert q.score < 0.9
        assert q.grade == ExtractionQualityGrade.MEDIUM

    def test_low_medium_boundary(self):
        # score just below 0.6 → LOW
        q = _make_quality(
            printable_ratio=0.5,
            avg_words_per_page=30.0,
            pages_with_text=5,
            total_pages=10,
        )
        # coverage=0.5, text_quality=0.5, density=0.3
        # score = 0.2 + 0.2 + 0.06 = 0.46
        assert q.score < 0.6
        assert q.grade == ExtractionQualityGrade.LOW


# ---------------------------------------------------------------------------
# IngestKey Tests
# ---------------------------------------------------------------------------


class TestIngestKey:
    def test_deterministic_key(self):
        k = IngestKey(
            content_hash="abc123",
            source_uri="file:///tmp/test.pdf",
            parser_version="1.0.0",
        )
        expected = hashlib.sha256(
            "abc123|file:///tmp/test.pdf|1.0.0".encode()
        ).hexdigest()
        assert k.key == expected

    def test_deterministic_key_with_tenant(self):
        k = IngestKey(
            content_hash="abc123",
            source_uri="file:///tmp/test.pdf",
            parser_version="1.0.0",
            tenant_id="tenant-42",
        )
        expected = hashlib.sha256(
            "abc123|file:///tmp/test.pdf|1.0.0|tenant-42".encode()
        ).hexdigest()
        assert k.key == expected

    def test_same_inputs_same_key(self):
        kwargs = dict(
            content_hash="abc",
            source_uri="uri",
            parser_version="v1",
        )
        assert IngestKey(**kwargs).key == IngestKey(**kwargs).key

    def test_different_inputs_different_key(self):
        k1 = IngestKey(content_hash="a", source_uri="u", parser_version="v1")
        k2 = IngestKey(content_hash="b", source_uri="u", parser_version="v1")
        assert k1.key != k2.key

    def test_tenant_changes_key(self):
        base = dict(content_hash="a", source_uri="u", parser_version="v1")
        k1 = IngestKey(**base)
        k2 = IngestKey(**base, tenant_id="t1")
        assert k1.key != k2.key


# ---------------------------------------------------------------------------
# Document Model Tests
# ---------------------------------------------------------------------------


class TestDocumentMetadata:
    def test_defaults(self):
        m = DocumentMetadata()
        assert m.title is None
        assert m.page_count == 0
        assert m.is_encrypted is False
        assert m.is_linearized is False

    def test_full_construction(self):
        m = DocumentMetadata(
            title="Test PDF",
            author="Author",
            page_count=10,
            file_size_bytes=1024,
            is_encrypted=True,
            needs_password=True,
        )
        assert m.title == "Test PDF"
        assert m.is_encrypted is True


class TestClassificationResult:
    def test_construction(self):
        r = ClassificationResult(
            pdf_type=PDFType.TEXT_NATIVE,
            confidence=0.95,
            tier_used=ClassificationTier.RULE_BASED,
            reasoning="High text content",
            per_page_types={0: PageType.TEXT, 1: PageType.TEXT},
        )
        assert r.pdf_type == PDFType.TEXT_NATIVE
        assert r.degraded is False

    def test_degraded_flag(self):
        r = ClassificationResult(
            pdf_type=PDFType.TEXT_NATIVE,
            confidence=0.7,
            tier_used=ClassificationTier.RULE_BASED,
            reasoning="LLM unavailable, used Tier 1",
            per_page_types={0: PageType.TEXT},
            degraded=True,
        )
        assert r.degraded is True


# ---------------------------------------------------------------------------
# Processing Model Tests
# ---------------------------------------------------------------------------


class TestOCRResult:
    def test_construction(self):
        r = OCRResult(
            page_number=0,
            text="Hello world",
            confidence=0.92,
            engine_used=OCREngine.TESSERACT,
            dpi=300,
            preprocessing_steps=["deskew", "binarize"],
        )
        assert r.engine_used == OCREngine.TESSERACT
        assert r.language_detected is None


class TestTableResult:
    def test_construction(self):
        t = TableResult(
            page_number=3,
            table_index=0,
            row_count=10,
            col_count=4,
            headers=["A", "B", "C", "D"],
        )
        assert t.is_continuation is False
        assert t.continuation_group_id is None

    def test_continuation(self):
        t = TableResult(
            page_number=4,
            table_index=0,
            row_count=5,
            col_count=4,
            is_continuation=True,
            continuation_group_id="grp-1",
        )
        assert t.is_continuation is True


class TestPDFChunkMetadata:
    def test_minimal(self):
        m = PDFChunkMetadata(
            source_uri="file:///test.pdf",
            page_numbers=[0, 1],
            ingestion_method="text_extraction",
            parser_version="1.0.0",
            chunk_index=0,
            chunk_hash="abc",
            ingest_key="key",
            ingest_run_id="run-1",
        )
        assert m.source_format == "pdf"
        assert m.tenant_id is None
        assert m.ocr_engine is None

    def test_full(self):
        m = PDFChunkMetadata(
            source_uri="file:///test.pdf",
            page_numbers=[5],
            ingestion_method="ocr_pipeline",
            parser_version="1.0.0",
            chunk_index=3,
            chunk_hash="def",
            ingest_key="key2",
            ingest_run_id="run-2",
            tenant_id="t1",
            heading_path=["Chapter 1", "Section 1.1"],
            content_type="narrative",
            ocr_engine="tesseract",
            ocr_confidence=0.88,
            ocr_dpi=300,
            language="en",
        )
        assert m.heading_path == ["Chapter 1", "Section 1.1"]
        assert m.ocr_confidence == 0.88


# ---------------------------------------------------------------------------
# Stage Artifact Tests
# ---------------------------------------------------------------------------


class TestStageArtifacts:
    def test_parse_stage_result(self):
        r = ParseStageResult(
            pages_extracted=10,
            pages_skipped=2,
            skipped_reasons={3: "W_PAGE_SKIPPED_BLANK", 7: "W_PAGE_SKIPPED_TOC"},
            extraction_method="pymupdf",
            overall_quality=_make_quality(),
            parse_duration_seconds=1.5,
        )
        assert r.pages_extracted == 10

    def test_classification_stage_result(self):
        r = ClassificationStageResult(
            tier_used=ClassificationTier.RULE_BASED,
            pdf_type=PDFType.TEXT_NATIVE,
            confidence=0.95,
            reasoning="High text ratio",
            per_page_types={0: PageType.TEXT},
            classification_duration_seconds=0.1,
        )
        assert r.degraded is False

    def test_ocr_stage_result(self):
        r = OCRStageResult(
            pages_ocrd=5,
            engine_used=OCREngine.TESSERACT,
            avg_confidence=0.87,
            low_confidence_pages=[2, 4],
            ocr_duration_seconds=12.3,
        )
        assert r.engine_fallback_used is False

    def test_ocr_stage_with_fallback(self):
        r = OCRStageResult(
            pages_ocrd=5,
            engine_used=OCREngine.TESSERACT,
            avg_confidence=0.82,
            low_confidence_pages=[],
            ocr_duration_seconds=8.0,
            engine_fallback_used=True,
        )
        assert r.engine_fallback_used is True

    def test_embed_stage_result(self):
        r = EmbedStageResult(
            texts_embedded=50,
            embedding_dimension=384,
            embed_duration_seconds=2.1,
        )
        assert r.embedding_dimension == 384


# ---------------------------------------------------------------------------
# WrittenArtifacts & ProcessingResult Tests
# ---------------------------------------------------------------------------


class TestWrittenArtifacts:
    def test_defaults(self):
        w = WrittenArtifacts()
        assert w.vector_point_ids == []
        assert w.vector_collection is None
        assert w.db_table_names == []


class TestProcessingResult:
    def test_full_construction(self):
        quality = _make_quality()
        r = ProcessingResult(
            file_path="/tmp/test.pdf",
            ingest_key="key123",
            ingest_run_id="run-1",
            parse_result=ParseStageResult(
                pages_extracted=10,
                pages_skipped=0,
                skipped_reasons={},
                extraction_method="pymupdf",
                overall_quality=quality,
                parse_duration_seconds=1.0,
            ),
            classification_result=ClassificationStageResult(
                tier_used=ClassificationTier.RULE_BASED,
                pdf_type=PDFType.TEXT_NATIVE,
                confidence=0.95,
                reasoning="High text",
                per_page_types={0: PageType.TEXT},
                classification_duration_seconds=0.1,
            ),
            classification=ClassificationResult(
                pdf_type=PDFType.TEXT_NATIVE,
                confidence=0.95,
                tier_used=ClassificationTier.RULE_BASED,
                reasoning="High text",
                per_page_types={0: PageType.TEXT},
            ),
            ingestion_method=IngestionMethod.TEXT_EXTRACTION,
            chunks_created=50,
            tables_created=0,
            tables=[],
            written=WrittenArtifacts(
                vector_point_ids=["p1", "p2"],
                vector_collection="docs",
            ),
            errors=[],
            warnings=[],
            processing_time_seconds=5.0,
        )
        assert r.chunks_created == 50
        assert r.ocr_result is None
        assert r.embed_result is None
        assert r.error_details == []


# ---------------------------------------------------------------------------
# Serialization Round-Trip Tests
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_ingest_key_round_trip(self):
        k = IngestKey(
            content_hash="abc",
            source_uri="file:///test.pdf",
            parser_version="1.0.0",
            tenant_id="t1",
        )
        data = k.model_dump()
        k2 = IngestKey.model_validate(data)
        assert k2.key == k.key

    def test_extraction_quality_round_trip(self):
        q = _make_quality()
        data = q.model_dump()
        q2 = ExtractionQuality.model_validate(data)
        assert q2.score == pytest.approx(q.score)

    def test_classification_result_round_trip(self):
        r = ClassificationResult(
            pdf_type=PDFType.SCANNED,
            confidence=0.85,
            tier_used=ClassificationTier.LLM_BASIC,
            reasoning="Image-heavy pages",
            per_page_types={0: PageType.SCANNED, 1: PageType.SCANNED},
            degraded=False,
        )
        data = r.model_dump()
        r2 = ClassificationResult.model_validate(data)
        assert r2.pdf_type == PDFType.SCANNED
        assert r2.per_page_types == {0: PageType.SCANNED, 1: PageType.SCANNED}

    def test_page_profile_round_trip(self):
        p = PageProfile(
            page_number=0,
            text_length=500,
            word_count=100,
            image_count=0,
            image_coverage_ratio=0.0,
            table_count=0,
            font_count=2,
            font_names=["Arial", "Times"],
            has_form_fields=False,
            is_multi_column=False,
            page_type=PageType.TEXT,
            extraction_quality=_make_quality(),
        )
        data = p.model_dump()
        p2 = PageProfile.model_validate(data)
        assert p2.page_type == PageType.TEXT
        assert p2.extraction_quality.grade == p.extraction_quality.grade
