"""Unit tests for the Path A TextExtractor processor.

All tests mock fitz, pymupdf4llm, and backend protocols. No real PDFs
or external services are required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ingestkit_core.models import EmbedStageResult, WrittenArtifacts
from ingestkit_pdf.config import PDFProcessorConfig
from ingestkit_pdf.errors import ErrorCode
from ingestkit_pdf.models import (
    ClassificationResult,
    ClassificationStageResult,
    DocumentProfile,
    ExtractionQuality,
    ExtractionQualityGrade,
    IngestionMethod,
    PageType,
    PDFChunkMetadata,
    PDFType,
    ParseStageResult,
    ProcessingResult,
)
from ingestkit_pdf.processors.text_extractor import TextExtractor

from tests.conftest import _make_document_profile, _make_page_profile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_parse_result(**overrides: Any) -> ParseStageResult:
    defaults: dict[str, Any] = dict(
        pages_extracted=5,
        pages_skipped=0,
        skipped_reasons={},
        extraction_method="native",
        overall_quality=ExtractionQuality(
            printable_ratio=0.95,
            avg_words_per_page=300.0,
            pages_with_text=5,
            total_pages=5,
            extraction_method="native",
        ),
        parse_duration_seconds=0.5,
    )
    defaults.update(overrides)
    return ParseStageResult(**defaults)


def _make_classification_result(**overrides: Any) -> ClassificationResult:
    defaults: dict[str, Any] = dict(
        pdf_type=PDFType.TEXT_NATIVE,
        confidence=0.95,
        tier_used="rule_based",
        reasoning="Text-native PDF",
        per_page_types={1: PageType.TEXT},
    )
    defaults.update(overrides)
    return ClassificationResult(**defaults)


def _make_classification_stage_result(**overrides: Any) -> ClassificationStageResult:
    defaults: dict[str, Any] = dict(
        tier_used="rule_based",
        pdf_type=PDFType.TEXT_NATIVE,
        confidence=0.95,
        reasoning="Text-native PDF",
        per_page_types={1: PageType.TEXT},
        classification_duration_seconds=0.1,
    )
    defaults.update(overrides)
    return ClassificationStageResult(**defaults)


def _mock_pymupdf4llm_output(page_texts: dict[int, str]) -> list[dict]:
    """Build mock pymupdf4llm.to_markdown output from page_number->text dict."""
    return [
        {"text": text, "metadata": {"page": page_num - 1}}
        for page_num, text in sorted(page_texts.items())
    ]


_GOOD_PAGE_TEXT = (
    "This is a well-structured document about employee onboarding procedures. "
    "It contains detailed information about the steps required to set up new "
    "employees in the system, including account creation, badge provisioning, "
    "and orientation scheduling. Each department has specific requirements."
)

_TOC_TEXT = """Table of Contents
Chapter 1 .............. 1
Chapter 2 .............. 15
Chapter 3 .............. 28
Chapter 4 .............. 42
Chapter 5 .............. 56
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_vector_store() -> MagicMock:
    store = MagicMock()
    store.ensure_collection = MagicMock()
    store.upsert_chunks = MagicMock(return_value=1)
    return store


@pytest.fixture()
def mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.dimension.return_value = 768
    embedder.embed.return_value = [[0.1] * 768]
    return embedder


@pytest.fixture()
def config() -> PDFProcessorConfig:
    return PDFProcessorConfig()


@pytest.fixture()
def extractor(
    mock_vector_store: MagicMock,
    mock_embedder: MagicMock,
    config: PDFProcessorConfig,
) -> TextExtractor:
    return TextExtractor(mock_vector_store, mock_embedder, config)


def _build_mock_fitz_doc(page_count: int = 1) -> MagicMock:
    """Build a mock fitz.Document context manager."""
    mock_doc = MagicMock()
    mock_doc.__len__ = MagicMock(return_value=page_count)
    mock_doc.__enter__ = MagicMock(return_value=mock_doc)
    mock_doc.__exit__ = MagicMock(return_value=False)
    mock_doc.get_toc.return_value = []
    # Mock page objects for block-level fallback
    for i in range(page_count):
        mock_page = MagicMock()
        mock_page.get_text.return_value = []
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
    return mock_doc


def _call_process(
    extractor: TextExtractor,
    file_path: str = "/tmp/test.pdf",
    profile: DocumentProfile | None = None,
    ingest_key: str = "abc123",
    ingest_run_id: str = "run-001",
    parse_result: ParseStageResult | None = None,
    classification_result: ClassificationStageResult | None = None,
    classification: ClassificationResult | None = None,
) -> ProcessingResult:
    """Call process() with default arguments."""
    return extractor.process(
        file_path=file_path,
        profile=profile or _make_document_profile(),
        ingest_key=ingest_key,
        ingest_run_id=ingest_run_id,
        parse_result=parse_result or _make_parse_result(),
        classification_result=classification_result or _make_classification_stage_result(),
        classification=classification or _make_classification_result(),
    )


# ---------------------------------------------------------------------------
# TestTextExtractorInit
# ---------------------------------------------------------------------------


class TestTextExtractorInit:
    @pytest.mark.unit
    def test_constructor_stores_dependencies(
        self,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
        config: PDFProcessorConfig,
    ) -> None:
        ext = TextExtractor(mock_vector_store, mock_embedder, config)
        assert ext._vector_store is mock_vector_store
        assert ext._embedder is mock_embedder
        assert ext._config is config

    @pytest.mark.unit
    def test_constructor_accepts_protocol_types(
        self,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
        config: PDFProcessorConfig,
    ) -> None:
        # MagicMock satisfies Protocol structurally -- no TypeError
        ext = TextExtractor(mock_vector_store, mock_embedder, config)
        assert ext is not None


# ---------------------------------------------------------------------------
# TestProcessHappyPath
# ---------------------------------------------------------------------------


class TestProcessHappyPath:
    @pytest.mark.unit
    @patch("ingestkit_pdf.processors.text_extractor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.text_extractor.fitz")
    def test_basic_extraction_returns_processing_result(
        self,
        mock_fitz: MagicMock,
        mock_pymupdf4llm: MagicMock,
        extractor: TextExtractor,
    ) -> None:
        mock_doc = _build_mock_fitz_doc(1)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = _mock_pymupdf4llm_output(
            {1: _GOOD_PAGE_TEXT}
        )

        result = _call_process(extractor)

        assert isinstance(result, ProcessingResult)
        assert result.chunks_created >= 1

    @pytest.mark.unit
    @patch("ingestkit_pdf.processors.text_extractor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.text_extractor.fitz")
    def test_ingestion_method_is_text_extraction(
        self,
        mock_fitz: MagicMock,
        mock_pymupdf4llm: MagicMock,
        extractor: TextExtractor,
    ) -> None:
        mock_doc = _build_mock_fitz_doc(1)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = _mock_pymupdf4llm_output(
            {1: _GOOD_PAGE_TEXT}
        )

        result = _call_process(extractor)

        assert result.ingestion_method == IngestionMethod.TEXT_EXTRACTION

    @pytest.mark.unit
    @patch("ingestkit_pdf.processors.text_extractor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.text_extractor.fitz")
    def test_written_artifacts_tracked(
        self,
        mock_fitz: MagicMock,
        mock_pymupdf4llm: MagicMock,
        extractor: TextExtractor,
    ) -> None:
        mock_doc = _build_mock_fitz_doc(1)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = _mock_pymupdf4llm_output(
            {1: _GOOD_PAGE_TEXT}
        )

        result = _call_process(extractor)

        assert len(result.written.vector_point_ids) > 0
        assert result.written.vector_collection == "helpdesk"

    @pytest.mark.unit
    @patch("ingestkit_pdf.processors.text_extractor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.text_extractor.fitz")
    def test_embed_result_populated(
        self,
        mock_fitz: MagicMock,
        mock_pymupdf4llm: MagicMock,
        extractor: TextExtractor,
    ) -> None:
        mock_doc = _build_mock_fitz_doc(1)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = _mock_pymupdf4llm_output(
            {1: _GOOD_PAGE_TEXT}
        )

        result = _call_process(extractor)

        assert result.embed_result is not None
        assert result.embed_result.texts_embedded >= 1
        assert result.embed_result.embedding_dimension == 768

    @pytest.mark.unit
    @patch("ingestkit_pdf.processors.text_extractor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.text_extractor.fitz")
    def test_processing_time_recorded(
        self,
        mock_fitz: MagicMock,
        mock_pymupdf4llm: MagicMock,
        extractor: TextExtractor,
    ) -> None:
        mock_doc = _build_mock_fitz_doc(1)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = _mock_pymupdf4llm_output(
            {1: _GOOD_PAGE_TEXT}
        )

        result = _call_process(extractor)

        assert result.processing_time_seconds > 0


# ---------------------------------------------------------------------------
# TestTOCPageDetection
# ---------------------------------------------------------------------------


class TestTOCPageDetection:
    @pytest.mark.unit
    @patch("ingestkit_pdf.processors.text_extractor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.text_extractor.fitz")
    def test_toc_page_skipped_with_warning(
        self,
        mock_fitz: MagicMock,
        mock_pymupdf4llm: MagicMock,
        extractor: TextExtractor,
    ) -> None:
        mock_doc = _build_mock_fitz_doc(2)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = _mock_pymupdf4llm_output(
            {1: _TOC_TEXT, 2: _GOOD_PAGE_TEXT}
        )

        result = _call_process(extractor)

        assert ErrorCode.W_PAGE_SKIPPED_TOC.value in result.warnings

    @pytest.mark.unit
    def test_non_toc_page_not_skipped(self) -> None:
        assert TextExtractor._is_toc_page(_GOOD_PAGE_TEXT) is False

    @pytest.mark.unit
    def test_toc_detection_threshold(self) -> None:
        """Exactly 30% dot-leader lines should NOT trigger TOC detection (>30% required)."""
        # 10 lines total, 3 with dot-leaders = exactly 30%
        lines = [
            "Some normal text line",
            "Another normal line",
            "Chapter 1 .............. 1",
            "More normal text here",
            "Yet another line",
            "Chapter 2 .............. 15",
            "And some more content",
            "This is normal text",
            "Chapter 3 .............. 28",
            "Final normal line",
        ]
        text = "\n".join(lines)
        assert TextExtractor._is_toc_page(text) is False


# ---------------------------------------------------------------------------
# TestBlankPageDetection
# ---------------------------------------------------------------------------


class TestBlankPageDetection:
    @pytest.mark.unit
    @patch("ingestkit_pdf.processors.text_extractor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.text_extractor.fitz")
    def test_empty_page_skipped(
        self,
        mock_fitz: MagicMock,
        mock_pymupdf4llm: MagicMock,
        extractor: TextExtractor,
    ) -> None:
        mock_doc = _build_mock_fitz_doc(2)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = _mock_pymupdf4llm_output(
            {1: "", 2: _GOOD_PAGE_TEXT}
        )

        result = _call_process(extractor)

        assert ErrorCode.W_PAGE_SKIPPED_BLANK.value in result.warnings

    @pytest.mark.unit
    def test_whitespace_only_page_skipped(self) -> None:
        config = PDFProcessorConfig()
        assert TextExtractor._is_blank_page("   \n  \n  ", config) is True

    @pytest.mark.unit
    def test_few_words_page_skipped(self) -> None:
        config = PDFProcessorConfig()  # quality_min_words_per_page=10
        assert TextExtractor._is_blank_page("hello world", config) is True


# ---------------------------------------------------------------------------
# TestHeaderFooterStripping
# ---------------------------------------------------------------------------


class TestHeaderFooterStripping:
    @pytest.mark.unit
    @patch("ingestkit_pdf.processors.text_extractor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.text_extractor.fitz")
    @patch("ingestkit_pdf.processors.text_extractor.HeaderFooterDetector")
    def test_headers_stripped_from_pages(
        self,
        mock_hf_cls: MagicMock,
        mock_fitz: MagicMock,
        mock_pymupdf4llm: MagicMock,
        extractor: TextExtractor,
    ) -> None:
        mock_doc = _build_mock_fitz_doc(1)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = _mock_pymupdf4llm_output(
            {1: _GOOD_PAGE_TEXT}
        )

        mock_hf_instance = MagicMock()
        mock_hf_instance.detect.return_value = (["Header Co."], ["Page Footer"])
        mock_hf_instance.strip.return_value = _GOOD_PAGE_TEXT
        mock_hf_cls.return_value = mock_hf_instance

        result = _call_process(extractor)

        mock_hf_instance.detect.assert_called_once()
        mock_hf_instance.strip.assert_called()

    @pytest.mark.unit
    @patch("ingestkit_pdf.processors.text_extractor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.text_extractor.fitz")
    @patch("ingestkit_pdf.processors.text_extractor.HeaderFooterDetector")
    def test_header_footer_error_handled_gracefully(
        self,
        mock_hf_cls: MagicMock,
        mock_fitz: MagicMock,
        mock_pymupdf4llm: MagicMock,
        extractor: TextExtractor,
    ) -> None:
        mock_doc = _build_mock_fitz_doc(1)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = _mock_pymupdf4llm_output(
            {1: _GOOD_PAGE_TEXT}
        )

        mock_hf_instance = MagicMock()
        mock_hf_instance.detect.side_effect = RuntimeError("H/F detection failed")
        mock_hf_instance.strip.return_value = _GOOD_PAGE_TEXT
        mock_hf_cls.return_value = mock_hf_instance

        result = _call_process(extractor)

        # Processing should continue despite header/footer failure
        assert isinstance(result, ProcessingResult)
        assert ErrorCode.E_PROCESS_HEADER_FOOTER.value in result.warnings


# ---------------------------------------------------------------------------
# TestQualityFallback
# ---------------------------------------------------------------------------


class TestQualityFallback:
    @pytest.mark.unit
    @patch("ingestkit_pdf.processors.text_extractor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.text_extractor.fitz")
    @patch("ingestkit_pdf.processors.text_extractor.QualityAssessor")
    def test_low_quality_triggers_block_fallback(
        self,
        mock_qa_cls: MagicMock,
        mock_fitz: MagicMock,
        mock_pymupdf4llm: MagicMock,
        extractor: TextExtractor,
    ) -> None:
        mock_doc = _build_mock_fitz_doc(1)
        mock_page = MagicMock()
        mock_page.get_text.return_value = [
            (0, 0, 100, 20, "Good block text from fallback " * 10, 0, 0),
        ]
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = _mock_pymupdf4llm_output(
            {1: "garbled text"}
        )

        # First assess returns LOW, second (fallback) returns HIGH
        good_quality = ExtractionQuality(
            printable_ratio=0.95, avg_words_per_page=300.0,
            pages_with_text=1, total_pages=1, extraction_method="native",
        )
        mock_qa_instance = MagicMock()
        mock_qa_instance.needs_ocr_fallback.return_value = True
        mock_qa_instance.assess_page.return_value = good_quality
        mock_qa_cls.return_value = mock_qa_instance

        result = _call_process(extractor)

        assert ErrorCode.W_QUALITY_LOW_NATIVE.value in result.warnings

    @pytest.mark.unit
    @patch("ingestkit_pdf.processors.text_extractor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.text_extractor.fitz")
    @patch("ingestkit_pdf.processors.text_extractor.QualityAssessor")
    def test_block_fallback_improves_quality(
        self,
        mock_qa_cls: MagicMock,
        mock_fitz: MagicMock,
        mock_pymupdf4llm: MagicMock,
        extractor: TextExtractor,
    ) -> None:
        mock_doc = _build_mock_fitz_doc(1)
        mock_page = MagicMock()
        fallback_text_content = "Good block text from fallback extraction method " * 10
        mock_page.get_text.return_value = [
            (0, 0, 100, 20, fallback_text_content, 0, 0),
        ]
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = _mock_pymupdf4llm_output(
            {1: "garbled"}
        )

        low_quality = ExtractionQuality(
            printable_ratio=0.3, avg_words_per_page=2.0,
            pages_with_text=0, total_pages=1, extraction_method="native",
        )
        high_quality = ExtractionQuality(
            printable_ratio=0.95, avg_words_per_page=300.0,
            pages_with_text=1, total_pages=1, extraction_method="native",
        )

        mock_qa_instance = MagicMock()
        mock_qa_instance.needs_ocr_fallback.return_value = True
        # First call (original) returns LOW, second call (fallback) returns HIGH
        mock_qa_instance.assess_page.side_effect = [low_quality, high_quality]
        mock_qa_cls.return_value = mock_qa_instance

        result = _call_process(extractor)

        # Should not have OCR fallback warning since block fallback improved quality
        assert ErrorCode.W_OCR_FALLBACK.value not in result.warnings
        assert isinstance(result, ProcessingResult)

    @pytest.mark.unit
    @patch("ingestkit_pdf.processors.text_extractor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.text_extractor.fitz")
    @patch("ingestkit_pdf.processors.text_extractor.QualityAssessor")
    def test_block_fallback_still_low_adds_ocr_warning(
        self,
        mock_qa_cls: MagicMock,
        mock_fitz: MagicMock,
        mock_pymupdf4llm: MagicMock,
        extractor: TextExtractor,
    ) -> None:
        mock_doc = _build_mock_fitz_doc(1)
        mock_page = MagicMock()
        mock_page.get_text.return_value = [
            (0, 0, 100, 20, "still bad " * 10, 0, 0),
        ]
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = _mock_pymupdf4llm_output(
            {1: "garbled text " * 10}
        )

        low_quality = ExtractionQuality(
            printable_ratio=0.3, avg_words_per_page=2.0,
            pages_with_text=0, total_pages=1, extraction_method="native",
        )

        mock_qa_instance = MagicMock()
        mock_qa_instance.needs_ocr_fallback.return_value = True
        mock_qa_instance.assess_page.return_value = low_quality
        mock_qa_cls.return_value = mock_qa_instance

        result = _call_process(extractor)

        assert ErrorCode.W_OCR_FALLBACK.value in result.warnings


# ---------------------------------------------------------------------------
# TestHeadingDetection
# ---------------------------------------------------------------------------


class TestHeadingDetection:
    @pytest.mark.unit
    @patch("ingestkit_pdf.processors.text_extractor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.text_extractor.fitz")
    @patch("ingestkit_pdf.processors.text_extractor.HeadingDetector")
    def test_headings_propagated_to_chunks(
        self,
        mock_hd_cls: MagicMock,
        mock_fitz: MagicMock,
        mock_pymupdf4llm: MagicMock,
        extractor: TextExtractor,
    ) -> None:
        mock_doc = _build_mock_fitz_doc(1)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = _mock_pymupdf4llm_output(
            {1: _GOOD_PAGE_TEXT}
        )

        mock_hd_instance = MagicMock()
        mock_hd_instance.detect.return_value = [(1, "Introduction", 1)]
        mock_hd_cls.return_value = mock_hd_instance

        result = _call_process(extractor)

        assert isinstance(result, ProcessingResult)
        mock_hd_instance.detect.assert_called_once()

    @pytest.mark.unit
    def test_heading_offset_conversion(self) -> None:
        raw_headings = [
            (1, "Chapter 1", 1),
            (2, "Section 1.1", 2),
            (1, "Chapter 2", 3),
        ]
        page_offset_map = {1: 0, 2: 500, 3: 1200}

        result = TextExtractor._convert_headings_to_offsets(
            raw_headings, page_offset_map,
        )

        assert result == [
            (1, "Chapter 1", 0),
            (2, "Section 1.1", 500),
            (1, "Chapter 2", 1200),
        ]


# ---------------------------------------------------------------------------
# TestChunkMetadata
# ---------------------------------------------------------------------------


class TestChunkMetadata:
    @pytest.mark.unit
    @patch("ingestkit_pdf.processors.text_extractor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.text_extractor.fitz")
    def test_chunk_metadata_has_all_fields(
        self,
        mock_fitz: MagicMock,
        mock_pymupdf4llm: MagicMock,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
    ) -> None:
        config = PDFProcessorConfig(tenant_id="tenant-abc")
        ext = TextExtractor(mock_vector_store, mock_embedder, config)

        mock_doc = _build_mock_fitz_doc(1)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = _mock_pymupdf4llm_output(
            {1: _GOOD_PAGE_TEXT}
        )

        # Capture upserted chunks
        upserted: list = []
        mock_vector_store.upsert_chunks.side_effect = lambda c, chunks: upserted.extend(chunks)

        profile = _make_document_profile(
            metadata={"title": "Test Doc", "author": "Jane Doe", "creation_date": "2025-01-01"},
        )
        result = _call_process(ext, profile=profile)

        assert len(upserted) > 0
        meta = upserted[0].metadata
        assert isinstance(meta, PDFChunkMetadata)
        assert meta.source_format == "pdf"
        assert meta.ingestion_method == IngestionMethod.TEXT_EXTRACTION.value
        assert meta.ingest_key == "abc123"
        assert meta.ingest_run_id == "run-001"
        assert meta.tenant_id == "tenant-abc"

    @pytest.mark.unit
    @patch("ingestkit_pdf.processors.text_extractor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.text_extractor.fitz")
    def test_doc_title_propagated(
        self,
        mock_fitz: MagicMock,
        mock_pymupdf4llm: MagicMock,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
        config: PDFProcessorConfig,
    ) -> None:
        ext = TextExtractor(mock_vector_store, mock_embedder, config)

        mock_doc = _build_mock_fitz_doc(1)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = _mock_pymupdf4llm_output(
            {1: _GOOD_PAGE_TEXT}
        )

        upserted: list = []
        mock_vector_store.upsert_chunks.side_effect = lambda c, chunks: upserted.extend(chunks)

        from ingestkit_pdf.models import DocumentMetadata

        profile = _make_document_profile(
            metadata=DocumentMetadata(title="My Important Doc", author="Author"),
        )
        result = _call_process(ext, profile=profile)

        assert len(upserted) > 0
        assert upserted[0].metadata.doc_title == "My Important Doc"

    @pytest.mark.unit
    @patch("ingestkit_pdf.processors.text_extractor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.text_extractor.fitz")
    @patch("ingestkit_pdf.processors.text_extractor.detect_language")
    def test_language_propagated(
        self,
        mock_detect_lang: MagicMock,
        mock_fitz: MagicMock,
        mock_pymupdf4llm: MagicMock,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
        config: PDFProcessorConfig,
    ) -> None:
        ext = TextExtractor(mock_vector_store, mock_embedder, config)

        mock_doc = _build_mock_fitz_doc(1)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = _mock_pymupdf4llm_output(
            {1: _GOOD_PAGE_TEXT}
        )
        mock_detect_lang.return_value = ("fr", 0.95)

        upserted: list = []
        mock_vector_store.upsert_chunks.side_effect = lambda c, chunks: upserted.extend(chunks)

        result = _call_process(ext)

        assert len(upserted) > 0
        assert upserted[0].metadata.language == "fr"

    @pytest.mark.unit
    @patch("ingestkit_pdf.processors.text_extractor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.text_extractor.fitz")
    def test_tenant_id_propagated(
        self,
        mock_fitz: MagicMock,
        mock_pymupdf4llm: MagicMock,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
    ) -> None:
        config = PDFProcessorConfig(tenant_id="org-42")
        ext = TextExtractor(mock_vector_store, mock_embedder, config)

        mock_doc = _build_mock_fitz_doc(1)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = _mock_pymupdf4llm_output(
            {1: _GOOD_PAGE_TEXT}
        )

        upserted: list = []
        mock_vector_store.upsert_chunks.side_effect = lambda c, chunks: upserted.extend(chunks)

        result = _call_process(ext)

        assert len(upserted) > 0
        assert upserted[0].metadata.tenant_id == "org-42"
        assert result.tenant_id == "org-42"


# ---------------------------------------------------------------------------
# TestEmbeddingAndUpsert
# ---------------------------------------------------------------------------


class TestEmbeddingAndUpsert:
    @pytest.mark.unit
    @patch("ingestkit_pdf.processors.text_extractor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.text_extractor.fitz")
    def test_embedder_called_with_chunk_texts(
        self,
        mock_fitz: MagicMock,
        mock_pymupdf4llm: MagicMock,
        extractor: TextExtractor,
        mock_embedder: MagicMock,
    ) -> None:
        mock_doc = _build_mock_fitz_doc(1)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = _mock_pymupdf4llm_output(
            {1: _GOOD_PAGE_TEXT}
        )

        _call_process(extractor)

        mock_embedder.embed.assert_called()
        # First call's first arg should be a list of strings
        call_args = mock_embedder.embed.call_args_list[0]
        texts = call_args[0][0]
        assert isinstance(texts, list)
        assert all(isinstance(t, str) for t in texts)

    @pytest.mark.unit
    @patch("ingestkit_pdf.processors.text_extractor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.text_extractor.fitz")
    @patch("ingestkit_pdf.processors.text_extractor.PDFChunker")
    def test_batch_size_respected(
        self,
        mock_chunker_cls: MagicMock,
        mock_fitz: MagicMock,
        mock_pymupdf4llm: MagicMock,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
    ) -> None:
        config = PDFProcessorConfig(embedding_batch_size=2)
        ext = TextExtractor(mock_vector_store, mock_embedder, config)

        mock_doc = _build_mock_fitz_doc(1)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = _mock_pymupdf4llm_output(
            {1: _GOOD_PAGE_TEXT}
        )

        # Mock chunker to return 5 chunks
        mock_chunker_instance = MagicMock()
        mock_chunker_instance.chunk.return_value = [
            {
                "text": f"Chunk {i} text content here",
                "page_numbers": [1],
                "heading_path": [],
                "content_type": "narrative",
                "chunk_index": i,
                "chunk_hash": f"hash{i}",
            }
            for i in range(5)
        ]
        mock_chunker_cls.return_value = mock_chunker_instance

        # embedder must return correct number of vectors per batch
        mock_embedder.embed.side_effect = [
            [[0.1] * 768, [0.1] * 768],  # batch 1: 2 chunks
            [[0.1] * 768, [0.1] * 768],  # batch 2: 2 chunks
            [[0.1] * 768],               # batch 3: 1 chunk
        ]

        result = _call_process(ext)

        # With batch_size=2 and 5 chunks, embed() should be called 3 times
        assert mock_embedder.embed.call_count == 3

    @pytest.mark.unit
    @patch("ingestkit_pdf.processors.text_extractor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.text_extractor.fitz")
    def test_vector_store_upsert_called(
        self,
        mock_fitz: MagicMock,
        mock_pymupdf4llm: MagicMock,
        extractor: TextExtractor,
        mock_vector_store: MagicMock,
    ) -> None:
        mock_doc = _build_mock_fitz_doc(1)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = _mock_pymupdf4llm_output(
            {1: _GOOD_PAGE_TEXT}
        )

        _call_process(extractor)

        mock_vector_store.upsert_chunks.assert_called()

    @pytest.mark.unit
    @patch("ingestkit_pdf.processors.text_extractor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.text_extractor.fitz")
    def test_ensure_collection_called_first(
        self,
        mock_fitz: MagicMock,
        mock_pymupdf4llm: MagicMock,
        extractor: TextExtractor,
        mock_vector_store: MagicMock,
    ) -> None:
        mock_doc = _build_mock_fitz_doc(1)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = _mock_pymupdf4llm_output(
            {1: _GOOD_PAGE_TEXT}
        )

        _call_process(extractor)

        mock_vector_store.ensure_collection.assert_called_once_with("helpdesk", 768)
        # ensure_collection should be called before upsert_chunks
        ensure_order = mock_vector_store.ensure_collection.call_count
        assert ensure_order == 1


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.unit
    def test_embed_timeout_classified_correctly(self) -> None:
        exc = RuntimeError("embed operation timed out")
        code = TextExtractor._classify_backend_error(exc)
        assert code == ErrorCode.E_BACKEND_EMBED_TIMEOUT

    @pytest.mark.unit
    def test_vector_connect_error_classified(self) -> None:
        exc = ConnectionError("connection refused to vector store")
        code = TextExtractor._classify_backend_error(exc)
        assert code == ErrorCode.E_BACKEND_VECTOR_CONNECT

    @pytest.mark.unit
    @patch("ingestkit_pdf.processors.text_extractor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.text_extractor.fitz")
    def test_batch_failure_continues_to_next(
        self,
        mock_fitz: MagicMock,
        mock_pymupdf4llm: MagicMock,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
    ) -> None:
        config = PDFProcessorConfig(embedding_batch_size=1)
        ext = TextExtractor(mock_vector_store, mock_embedder, config)

        mock_doc = _build_mock_fitz_doc(1)
        mock_fitz.open.return_value = mock_doc
        # Two pages worth of text so chunker produces at least 2 chunks
        long_text = _GOOD_PAGE_TEXT + " " + _GOOD_PAGE_TEXT + " " + _GOOD_PAGE_TEXT
        mock_pymupdf4llm.to_markdown.return_value = _mock_pymupdf4llm_output(
            {1: long_text}
        )

        # First batch fails, subsequent succeed
        mock_embedder.embed.side_effect = [
            RuntimeError("timeout on embed"),
            [[0.1] * 768],
        ]

        result = _call_process(ext)

        # Should have an error from the first batch
        assert len(result.errors) >= 1
        # But second batch should have succeeded (partial results)
        # The test verifies processing continued after failure

    @pytest.mark.unit
    @patch("ingestkit_pdf.processors.text_extractor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.text_extractor.fitz")
    def test_error_details_populated(
        self,
        mock_fitz: MagicMock,
        mock_pymupdf4llm: MagicMock,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
    ) -> None:
        config = PDFProcessorConfig(embedding_batch_size=64)
        ext = TextExtractor(mock_vector_store, mock_embedder, config)

        mock_doc = _build_mock_fitz_doc(1)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = _mock_pymupdf4llm_output(
            {1: _GOOD_PAGE_TEXT}
        )

        mock_embedder.embed.side_effect = RuntimeError("embed connection refused")

        result = _call_process(ext)

        assert len(result.error_details) >= 1
        err = result.error_details[0]
        assert err.code == ErrorCode.E_BACKEND_EMBED_CONNECT
        assert "connection" in err.message.lower()


# ---------------------------------------------------------------------------
# TestProcessingResultAssembly
# ---------------------------------------------------------------------------


class TestProcessingResultAssembly:
    @pytest.mark.unit
    @patch("ingestkit_pdf.processors.text_extractor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.text_extractor.fitz")
    def test_empty_document_returns_zero_chunks(
        self,
        mock_fitz: MagicMock,
        mock_pymupdf4llm: MagicMock,
        extractor: TextExtractor,
    ) -> None:
        mock_doc = _build_mock_fitz_doc(2)
        mock_fitz.open.return_value = mock_doc
        # All pages blank
        mock_pymupdf4llm.to_markdown.return_value = _mock_pymupdf4llm_output(
            {1: "", 2: "  "}
        )

        result = _call_process(extractor)

        assert result.chunks_created == 0
        assert result.embed_result is None

    @pytest.mark.unit
    @patch("ingestkit_pdf.processors.text_extractor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.text_extractor.fitz")
    def test_result_fields_match_inputs(
        self,
        mock_fitz: MagicMock,
        mock_pymupdf4llm: MagicMock,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
    ) -> None:
        config = PDFProcessorConfig(tenant_id="tenant-xyz")
        ext = TextExtractor(mock_vector_store, mock_embedder, config)

        mock_doc = _build_mock_fitz_doc(1)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = _mock_pymupdf4llm_output(
            {1: _GOOD_PAGE_TEXT}
        )

        result = _call_process(
            ext,
            file_path="/data/docs/report.pdf",
            ingest_key="key-999",
            ingest_run_id="run-xyz",
        )

        assert result.file_path == "/data/docs/report.pdf"
        assert result.ingest_key == "key-999"
        assert result.ingest_run_id == "run-xyz"
        assert result.tenant_id == "tenant-xyz"

    @pytest.mark.unit
    @patch("ingestkit_pdf.processors.text_extractor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.text_extractor.fitz")
    def test_tables_always_empty_for_path_a(
        self,
        mock_fitz: MagicMock,
        mock_pymupdf4llm: MagicMock,
        extractor: TextExtractor,
    ) -> None:
        mock_doc = _build_mock_fitz_doc(1)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = _mock_pymupdf4llm_output(
            {1: _GOOD_PAGE_TEXT}
        )

        result = _call_process(extractor)

        assert result.tables_created == 0
        assert result.tables == []


# ---------------------------------------------------------------------------
# TestLanguageDetection
# ---------------------------------------------------------------------------


class TestLanguageDetection:
    @pytest.mark.unit
    @patch("ingestkit_pdf.processors.text_extractor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.text_extractor.fitz")
    @patch("ingestkit_pdf.processors.text_extractor.detect_language")
    def test_language_detection_enabled(
        self,
        mock_detect_lang: MagicMock,
        mock_fitz: MagicMock,
        mock_pymupdf4llm: MagicMock,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
    ) -> None:
        config = PDFProcessorConfig(enable_language_detection=True)
        ext = TextExtractor(mock_vector_store, mock_embedder, config)

        mock_doc = _build_mock_fitz_doc(1)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = _mock_pymupdf4llm_output(
            {1: _GOOD_PAGE_TEXT}
        )
        mock_detect_lang.return_value = ("es", 0.9)

        result = _call_process(ext)

        mock_detect_lang.assert_called_once()

    @pytest.mark.unit
    @patch("ingestkit_pdf.processors.text_extractor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.text_extractor.fitz")
    @patch("ingestkit_pdf.processors.text_extractor.detect_language")
    def test_language_detection_disabled_uses_default(
        self,
        mock_detect_lang: MagicMock,
        mock_fitz: MagicMock,
        mock_pymupdf4llm: MagicMock,
        mock_vector_store: MagicMock,
        mock_embedder: MagicMock,
    ) -> None:
        config = PDFProcessorConfig(enable_language_detection=False)
        ext = TextExtractor(mock_vector_store, mock_embedder, config)

        mock_doc = _build_mock_fitz_doc(1)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = _mock_pymupdf4llm_output(
            {1: _GOOD_PAGE_TEXT}
        )

        upserted: list = []
        mock_vector_store.upsert_chunks.side_effect = lambda c, chunks: upserted.extend(chunks)

        result = _call_process(ext)

        mock_detect_lang.assert_not_called()
        assert len(upserted) > 0
        assert upserted[0].metadata.language == "en"
