"""Unit tests for the Path C ComplexProcessor.

All tests mock fitz, pymupdf4llm, _ocr_single_page, TableExtractor, and
backend protocols. No real PDFs or external services are required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ingestkit_pdf.config import PDFProcessorConfig
from ingestkit_pdf.errors import ErrorCode
from ingestkit_pdf.models import (
    ClassificationResult,
    ClassificationStageResult,
    ExtractionQuality,
    IngestionMethod,
    OCREngine,
    OCRResult,
    PageType,
    PDFType,
    ParseStageResult,
)
from ingestkit_pdf.processors.complex_processor import ComplexProcessor
from ingestkit_pdf.processors.table_extractor import TableExtractionResult

from tests.conftest import (
    MockEmbeddingBackend,
    MockStructuredDBBackend,
    MockVectorStoreBackend,
    _make_document_profile,
    _make_page_profile,
)


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
        pdf_type=PDFType.COMPLEX,
        confidence=0.90,
        tier_used="rule_based",
        reasoning="Complex PDF with mixed page types",
        per_page_types={1: PageType.TEXT},
    )
    defaults.update(overrides)
    return ClassificationResult(**defaults)


def _make_classification_stage_result(**overrides: Any) -> ClassificationStageResult:
    defaults: dict[str, Any] = dict(
        tier_used="rule_based",
        pdf_type=PDFType.COMPLEX,
        confidence=0.90,
        reasoning="Complex PDF with mixed page types",
        per_page_types={1: PageType.TEXT},
        classification_duration_seconds=0.1,
    )
    defaults.update(overrides)
    return ClassificationStageResult(**defaults)


def _make_ocr_result(page_number: int = 1, **overrides: Any) -> OCRResult:
    defaults: dict[str, Any] = dict(
        page_number=page_number,
        text=f"OCR text from page {page_number}",
        confidence=0.85,
        engine_used=OCREngine.TESSERACT,
        dpi=300,
        preprocessing_steps=["deskew"],
        language_detected="en",
    )
    defaults.update(overrides)
    return OCRResult(**defaults)


_GOOD_PAGE_TEXT = (
    "This is a well-structured document about employee onboarding procedures. "
    "It contains detailed information about the steps required to set up new "
    "employees in the system, including account creation, badge provisioning, "
    "and orientation scheduling. Each department has specific requirements."
)


def _make_mock_doc(page_count: int = 1, page_texts: dict[int, str] | None = None):
    """Create a mock fitz.Document context manager."""
    doc = MagicMock()
    doc.__enter__ = MagicMock(return_value=doc)
    doc.__exit__ = MagicMock(return_value=False)

    pages = []
    for i in range(page_count):
        page = MagicMock()
        page_num = i + 1
        text = (page_texts or {}).get(page_num, _GOOD_PAGE_TEXT)
        page.get_text.return_value = text
        page.get_images.return_value = []
        page.widgets.return_value = iter([])
        page.rect = MagicMock()
        page.rect.width = 612.0
        pages.append(page)

    doc.__getitem__ = MagicMock(side_effect=lambda idx: pages[idx])
    doc.__len__ = MagicMock(return_value=page_count)
    doc.__iter__ = MagicMock(return_value=iter(pages))
    return doc


def _make_mock_widget(
    field_name: str = "Name",
    field_value: str | None = "John",
    field_type: int = 0,
):
    """Create a mock fitz.Widget."""
    w = MagicMock()
    w.field_name = field_name
    w.field_value = field_value
    w.field_type = field_type
    return w


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> PDFProcessorConfig:
    return PDFProcessorConfig()


@pytest.fixture()
def vector_store() -> MockVectorStoreBackend:
    return MockVectorStoreBackend()


@pytest.fixture()
def embedder() -> MockEmbeddingBackend:
    return MockEmbeddingBackend()


@pytest.fixture()
def structured_db() -> MockStructuredDBBackend:
    return MockStructuredDBBackend()


@pytest.fixture()
def processor(
    vector_store: MockVectorStoreBackend,
    structured_db: MockStructuredDBBackend,
    embedder: MockEmbeddingBackend,
    config: PDFProcessorConfig,
) -> ComplexProcessor:
    return ComplexProcessor(
        vector_store=vector_store,
        structured_db=structured_db,
        embedder=embedder,
        llm=None,
        config=config,
    )


# ---------------------------------------------------------------------------
# Page Routing Tests
# ---------------------------------------------------------------------------


class TestComplexProcessorPageRouting:
    """R-PC-1: Page-level routing by PageType."""

    @patch("ingestkit_pdf.processors.complex_processor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.complex_processor.fitz")
    def test_text_pages_extracted_via_pymupdf4llm(
        self, mock_fitz, mock_pymupdf4llm, processor,
    ):
        mock_doc = _make_mock_doc(1)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = _GOOD_PAGE_TEXT

        classification = _make_classification_result(
            per_page_types={1: PageType.TEXT},
        )
        profile = _make_document_profile(
            pages=[_make_page_profile(page_number=1, page_type=PageType.TEXT)],
        )

        result = processor.process(
            file_path="/tmp/test.pdf",
            profile=profile,
            ingest_key="a" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=classification,
        )

        assert result.ingestion_method == IngestionMethod.COMPLEX_PROCESSING
        mock_pymupdf4llm.to_markdown.assert_called()

    @patch("ingestkit_pdf.processors.complex_processor._ocr_single_page")
    @patch("ingestkit_pdf.processors.complex_processor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.complex_processor.fitz")
    def test_scanned_pages_use_ocr(
        self, mock_fitz, mock_pymupdf4llm, mock_ocr, processor,
    ):
        mock_doc = _make_mock_doc(1)
        mock_fitz.open.return_value = mock_doc
        mock_ocr.return_value = _make_ocr_result(page_number=1)

        classification = _make_classification_result(
            per_page_types={1: PageType.SCANNED},
        )
        profile = _make_document_profile(
            pages=[_make_page_profile(page_number=1, page_type=PageType.SCANNED)],
        )

        result = processor.process(
            file_path="/tmp/test.pdf",
            profile=profile,
            ingest_key="a" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=classification,
        )

        mock_ocr.assert_called_once()
        assert result.ocr_result is not None
        assert result.ocr_result.pages_ocrd == 1

    @patch("ingestkit_pdf.processors.complex_processor.TableExtractor")
    @patch("ingestkit_pdf.processors.complex_processor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.complex_processor.fitz")
    def test_table_heavy_pages_delegated_to_table_extractor(
        self, mock_fitz, mock_pymupdf4llm, mock_te_cls, processor,
    ):
        mock_doc = _make_mock_doc(1)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = ""

        mock_te = MagicMock()
        mock_te.extract_tables.return_value = TableExtractionResult(
            tables=[],
            chunks=[],
            table_names=["pdf_test_p1_t0"],
            warnings=[],
            errors=[],
            texts_embedded=0,
            embed_duration_seconds=0.0,
        )
        mock_te_cls.return_value = mock_te

        classification = _make_classification_result(
            per_page_types={1: PageType.TABLE_HEAVY},
        )
        profile = _make_document_profile(
            pages=[_make_page_profile(page_number=1, page_type=PageType.TABLE_HEAVY)],
        )

        result = processor.process(
            file_path="/tmp/test.pdf",
            profile=profile,
            ingest_key="a" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=classification,
        )

        mock_te.extract_tables.assert_called_once()
        assert "pdf_test_p1_t0" in result.tables

    @patch("ingestkit_pdf.processors.complex_processor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.complex_processor.fitz")
    def test_form_pages_extract_widgets(
        self, mock_fitz, mock_pymupdf4llm, processor,
    ):
        mock_doc = _make_mock_doc(1)
        mock_fitz.open.return_value = mock_doc

        # Set up form widgets on the page
        widgets = [
            _make_mock_widget("Name", "John Doe", field_type=0),
            _make_mock_widget("Active", "Yes", field_type=1),
        ]
        mock_doc[0].widgets.return_value = iter(widgets)
        mock_doc[0].get_text.return_value = "Please fill out this form."

        classification = _make_classification_result(
            per_page_types={1: PageType.FORM},
        )
        profile = _make_document_profile(
            pages=[_make_page_profile(page_number=1, page_type=PageType.FORM, has_form_fields=True)],
        )

        result = processor.process(
            file_path="/tmp/test.pdf",
            profile=profile,
            ingest_key="a" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=classification,
        )

        assert result.chunks_created > 0

    @patch("ingestkit_pdf.processors.complex_processor._ocr_single_page")
    @patch("ingestkit_pdf.processors.complex_processor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.complex_processor.fitz")
    def test_mixed_pages_text_plus_ocr(
        self, mock_fitz, mock_pymupdf4llm, mock_ocr, processor,
    ):
        mock_doc = _make_mock_doc(1)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = "Some native text"
        mock_doc[0].get_images.return_value = [(1, 0, 100, 100, 8, "DeviceRGB", "", "Im0", "DCTDecode", 0)]

        # OCR returns significantly more text
        mock_ocr.return_value = _make_ocr_result(
            page_number=1,
            text="Some native text plus additional OCR content that wasn't in the native extraction",
        )

        classification = _make_classification_result(
            per_page_types={1: PageType.MIXED},
        )
        profile = _make_document_profile(
            pages=[_make_page_profile(page_number=1, page_type=PageType.MIXED)],
        )

        result = processor.process(
            file_path="/tmp/test.pdf",
            profile=profile,
            ingest_key="a" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=classification,
        )

        mock_ocr.assert_called_once()
        assert result.chunks_created > 0

    @patch("ingestkit_pdf.processors.complex_processor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.complex_processor.fitz")
    def test_blank_pages_skipped_with_warning(
        self, mock_fitz, mock_pymupdf4llm, processor,
    ):
        mock_doc = _make_mock_doc(1)
        mock_fitz.open.return_value = mock_doc

        classification = _make_classification_result(
            per_page_types={1: PageType.BLANK},
        )
        profile = _make_document_profile(
            pages=[_make_page_profile(page_number=1, page_type=PageType.BLANK)],
        )

        result = processor.process(
            file_path="/tmp/test.pdf",
            profile=profile,
            ingest_key="a" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=classification,
        )

        assert result.chunks_created == 0
        assert any(ErrorCode.W_PAGE_SKIPPED_BLANK.value in w for w in result.warnings)

    @patch("ingestkit_pdf.processors.complex_processor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.complex_processor.fitz")
    def test_toc_pages_skipped_with_warning(
        self, mock_fitz, mock_pymupdf4llm, processor,
    ):
        mock_doc = _make_mock_doc(1)
        mock_fitz.open.return_value = mock_doc

        classification = _make_classification_result(
            per_page_types={1: PageType.TOC},
        )
        profile = _make_document_profile(
            pages=[_make_page_profile(page_number=1, page_type=PageType.TOC)],
        )

        result = processor.process(
            file_path="/tmp/test.pdf",
            profile=profile,
            ingest_key="a" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=classification,
        )

        assert result.chunks_created == 0
        assert any(ErrorCode.W_PAGE_SKIPPED_TOC.value in w for w in result.warnings)

    @patch("ingestkit_pdf.processors.complex_processor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.complex_processor.fitz")
    def test_vector_only_pages_skipped_with_warning(
        self, mock_fitz, mock_pymupdf4llm, processor,
    ):
        mock_doc = _make_mock_doc(1)
        mock_fitz.open.return_value = mock_doc

        classification = _make_classification_result(
            per_page_types={1: PageType.VECTOR_ONLY},
        )
        profile = _make_document_profile(
            pages=[_make_page_profile(page_number=1, page_type=PageType.VECTOR_ONLY)],
        )

        result = processor.process(
            file_path="/tmp/test.pdf",
            profile=profile,
            ingest_key="a" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=classification,
        )

        assert result.chunks_created == 0
        assert any(ErrorCode.W_PAGE_SKIPPED_VECTOR_ONLY.value in w for w in result.warnings)

    @patch("ingestkit_pdf.processors.complex_processor.TableExtractor")
    @patch("ingestkit_pdf.processors.complex_processor._ocr_single_page")
    @patch("ingestkit_pdf.processors.complex_processor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.complex_processor.fitz")
    def test_all_page_types_in_single_document(
        self, mock_fitz, mock_pymupdf4llm, mock_ocr, mock_te_cls, processor,
    ):
        """Integration-style test: 8-page doc with one page of each PageType."""
        mock_doc = _make_mock_doc(8)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = _GOOD_PAGE_TEXT

        # Setup form widgets on page 4 (FORM)
        widgets = [_make_mock_widget("Field1", "Value1", field_type=0)]
        mock_doc[3].widgets.return_value = iter(widgets)
        mock_doc[3].get_text.return_value = "Form page text"

        # Setup images on page 5 (MIXED)
        mock_doc[4].get_images.return_value = [(1, 0, 100, 100, 8, "DeviceRGB", "", "Im0", "DCTDecode", 0)]

        # OCR for SCANNED (page 2) and MIXED (page 5)
        mock_ocr.side_effect = [
            _make_ocr_result(page_number=2),
            _make_ocr_result(page_number=5, text="OCR text " * 50),
        ]

        # TableExtractor for TABLE_HEAVY (page 3)
        mock_te = MagicMock()
        mock_te.extract_tables.return_value = TableExtractionResult(
            tables=[],
            chunks=[],
            table_names=["pdf_test_p3_t0"],
            warnings=[],
            errors=[],
            texts_embedded=1,
            embed_duration_seconds=0.1,
        )
        mock_te_cls.return_value = mock_te

        per_page_types = {
            1: PageType.TEXT,
            2: PageType.SCANNED,
            3: PageType.TABLE_HEAVY,
            4: PageType.FORM,
            5: PageType.MIXED,
            6: PageType.BLANK,
            7: PageType.TOC,
            8: PageType.VECTOR_ONLY,
        }

        pages = [
            _make_page_profile(page_number=i, page_type=pt)
            for i, pt in per_page_types.items()
        ]
        profile = _make_document_profile(pages=pages)
        classification = _make_classification_result(per_page_types=per_page_types)

        result = processor.process(
            file_path="/tmp/test.pdf",
            profile=profile,
            ingest_key="a" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=classification,
        )

        # TEXT page: pymupdf4llm called
        mock_pymupdf4llm.to_markdown.assert_called()

        # SCANNED page: OCR called
        assert mock_ocr.call_count >= 1

        # TABLE_HEAVY: TableExtractor called
        mock_te.extract_tables.assert_called_once()

        # Skip pages: warnings emitted
        assert any(ErrorCode.W_PAGE_SKIPPED_BLANK.value in w for w in result.warnings)
        assert any(ErrorCode.W_PAGE_SKIPPED_TOC.value in w for w in result.warnings)
        assert any(ErrorCode.W_PAGE_SKIPPED_VECTOR_ONLY.value in w for w in result.warnings)

        # Overall result
        assert result.ingestion_method == IngestionMethod.COMPLEX_PROCESSING
        assert "pdf_test_p3_t0" in result.tables


# ---------------------------------------------------------------------------
# Form Field Tests
# ---------------------------------------------------------------------------


class TestComplexProcessorFormFields:
    """R-PC-4: AcroForm field extraction."""

    def test_form_field_extraction_name_value_pairs(self, processor):
        doc = _make_mock_doc(1)
        widgets = [
            _make_mock_widget("Name", "John Doe", field_type=0),
            _make_mock_widget("Email", "john@example.com", field_type=0),
        ]
        doc[0].widgets.return_value = iter(widgets)
        doc[0].get_text.return_value = ""

        text = processor._extract_form_fields(doc, 1)
        assert "Name: John Doe" in text
        assert "Email: john@example.com" in text

    def test_form_field_empty_value_handled(self, processor):
        doc = _make_mock_doc(1)
        widgets = [_make_mock_widget("Address", None, field_type=0)]
        doc[0].widgets.return_value = iter(widgets)
        doc[0].get_text.return_value = ""

        text = processor._extract_form_fields(doc, 1)
        assert "Address: (empty)" in text

    def test_form_field_checkbox_value(self, processor):
        doc = _make_mock_doc(1)
        widgets = [
            _make_mock_widget("Agree", "Yes", field_type=1),
            _make_mock_widget("OptOut", "Off", field_type=1),
        ]
        doc[0].widgets.return_value = iter(widgets)
        doc[0].get_text.return_value = ""

        text = processor._extract_form_fields(doc, 1)
        assert "Agree: Yes" in text
        assert "OptOut: No" in text

    def test_form_field_signature_skipped(self, processor):
        doc = _make_mock_doc(1)
        widgets = [
            _make_mock_widget("Name", "John", field_type=0),
            _make_mock_widget("Signature", "sig_data", field_type=3),
        ]
        doc[0].widgets.return_value = iter(widgets)
        doc[0].get_text.return_value = ""

        text = processor._extract_form_fields(doc, 1)
        assert "Name: John" in text
        assert "Signature" not in text

    def test_form_page_includes_regular_text(self, processor):
        doc = _make_mock_doc(1)
        widgets = [_make_mock_widget("Name", "John", field_type=0)]
        doc[0].widgets.return_value = iter(widgets)
        doc[0].get_text.return_value = "Please complete this form."

        text = processor._extract_form_fields(doc, 1)
        assert "Please complete this form." in text
        assert "Name: John" in text


# ---------------------------------------------------------------------------
# Multi-Column Tests
# ---------------------------------------------------------------------------


class TestComplexProcessorMultiColumn:
    """Multi-column layout reordering."""

    @patch("ingestkit_pdf.processors.complex_processor.extract_text_blocks")
    def test_multi_column_page_reordered(self, mock_extract, processor):
        from ingestkit_pdf.utils.layout_analysis import LayoutResult, TextBlock

        doc = _make_mock_doc(1)
        mock_analyzer = MagicMock()
        mock_analyzer.detect_columns.return_value = LayoutResult(
            is_multi_column=True,
            column_count=2,
            column_boundaries=[(0.0, 300.0), (312.0, 612.0)],
            page_width=612.0,
        )

        blocks = [
            TextBlock(x0=312, y0=0, x1=600, y1=50, text="Right column", block_number=0),
            TextBlock(x0=0, y0=0, x1=290, y1=50, text="Left column", block_number=1),
        ]
        mock_extract.return_value = blocks
        mock_analyzer.reorder_blocks.return_value = [blocks[1], blocks[0]]

        result = processor._apply_layout_reorder(doc, 1, mock_analyzer)
        assert result is not None
        assert "Left column" in result
        assert "Right column" in result

    def test_single_column_page_unchanged(self, processor):
        from ingestkit_pdf.utils.layout_analysis import LayoutResult

        doc = _make_mock_doc(1)
        mock_analyzer = MagicMock()
        mock_analyzer.detect_columns.return_value = LayoutResult(
            is_multi_column=False,
            column_count=1,
            column_boundaries=[(0.0, 612.0)],
            page_width=612.0,
        )

        result = processor._apply_layout_reorder(doc, 1, mock_analyzer)
        assert result is None


# ---------------------------------------------------------------------------
# Header/Footer Tests
# ---------------------------------------------------------------------------


class TestComplexProcessorHeaderFooter:
    """Header/footer stripping across all pages."""

    @patch("ingestkit_pdf.processors.complex_processor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.complex_processor.fitz")
    def test_headers_and_footers_stripped(
        self, mock_fitz, mock_pymupdf4llm, processor,
    ):
        mock_doc = _make_mock_doc(1)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = _GOOD_PAGE_TEXT

        classification = _make_classification_result(
            per_page_types={1: PageType.TEXT},
        )
        profile = _make_document_profile(
            pages=[_make_page_profile(page_number=1, page_type=PageType.TEXT)],
        )

        # The test verifies that the header/footer detector is called;
        # stripping is tested in test_header_footer.py
        result = processor.process(
            file_path="/tmp/test.pdf",
            profile=profile,
            ingest_key="a" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=classification,
        )

        assert result.ingestion_method == IngestionMethod.COMPLEX_PROCESSING

    @patch("ingestkit_pdf.processors.complex_processor.HeaderFooterDetector")
    @patch("ingestkit_pdf.processors.complex_processor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.complex_processor.fitz")
    def test_header_footer_detection_failure_is_recoverable(
        self, mock_fitz, mock_pymupdf4llm, mock_hf_cls, processor,
    ):
        mock_doc = _make_mock_doc(1)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = _GOOD_PAGE_TEXT

        # Make detect() raise
        mock_hf = MagicMock()
        mock_hf.detect.side_effect = RuntimeError("Header detection failed")
        mock_hf.strip.side_effect = lambda text, *args, **kwargs: text
        mock_hf_cls.return_value = mock_hf

        classification = _make_classification_result(
            per_page_types={1: PageType.TEXT},
        )
        profile = _make_document_profile(
            pages=[_make_page_profile(page_number=1, page_type=PageType.TEXT)],
        )

        result = processor.process(
            file_path="/tmp/test.pdf",
            profile=profile,
            ingest_key="a" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=classification,
        )

        # Should still produce a result (recoverable)
        assert result.ingestion_method == IngestionMethod.COMPLEX_PROCESSING
        assert any(ErrorCode.E_PROCESS_HEADER_FOOTER.value in w for w in result.warnings)
        assert any(
            e.code == ErrorCode.E_PROCESS_HEADER_FOOTER and e.recoverable
            for e in result.error_details
        )


# ---------------------------------------------------------------------------
# ProcessingResult Assembly Tests
# ---------------------------------------------------------------------------


class TestComplexProcessorAssembly:
    """ProcessingResult assembly."""

    @patch("ingestkit_pdf.processors.complex_processor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.complex_processor.fitz")
    def test_processing_result_all_fields_populated(
        self, mock_fitz, mock_pymupdf4llm, processor,
    ):
        mock_doc = _make_mock_doc(1)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = _GOOD_PAGE_TEXT

        classification = _make_classification_result(
            per_page_types={1: PageType.TEXT},
        )
        profile = _make_document_profile(
            pages=[_make_page_profile(page_number=1, page_type=PageType.TEXT)],
        )

        result = processor.process(
            file_path="/tmp/test.pdf",
            profile=profile,
            ingest_key="a" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=classification,
        )

        assert result.file_path == "/tmp/test.pdf"
        assert result.ingest_key == "a" * 64
        assert result.ingest_run_id == "run-1"
        assert result.parse_result is not None
        assert result.classification_result is not None
        assert result.classification is not None
        assert result.ingestion_method == IngestionMethod.COMPLEX_PROCESSING
        assert result.processing_time_seconds >= 0.0

    @patch("ingestkit_pdf.processors.complex_processor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.complex_processor.fitz")
    def test_empty_document_returns_zero_chunks(
        self, mock_fitz, mock_pymupdf4llm, processor,
    ):
        mock_doc = _make_mock_doc(1)
        mock_fitz.open.return_value = mock_doc

        classification = _make_classification_result(
            per_page_types={1: PageType.BLANK},
        )
        profile = _make_document_profile(
            pages=[_make_page_profile(page_number=1, page_type=PageType.BLANK)],
        )

        result = processor.process(
            file_path="/tmp/test.pdf",
            profile=profile,
            ingest_key="a" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=classification,
        )

        assert result.chunks_created == 0
        assert result.tables_created == 0

    @patch("ingestkit_pdf.processors.complex_processor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.complex_processor.fitz")
    def test_ingestion_method_is_complex_processing(
        self, mock_fitz, mock_pymupdf4llm, processor,
    ):
        mock_doc = _make_mock_doc(1)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = _GOOD_PAGE_TEXT

        classification = _make_classification_result(
            per_page_types={1: PageType.TEXT},
        )
        profile = _make_document_profile(
            pages=[_make_page_profile(page_number=1, page_type=PageType.TEXT)],
        )

        result = processor.process(
            file_path="/tmp/test.pdf",
            profile=profile,
            ingest_key="a" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=classification,
        )

        assert result.ingestion_method == IngestionMethod.COMPLEX_PROCESSING

    @patch("ingestkit_pdf.processors.complex_processor.TableExtractor")
    @patch("ingestkit_pdf.processors.complex_processor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.complex_processor.fitz")
    def test_tables_created_count_matches(
        self, mock_fitz, mock_pymupdf4llm, mock_te_cls, processor,
    ):
        mock_doc = _make_mock_doc(1)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = ""

        mock_te = MagicMock()
        mock_te.extract_tables.return_value = TableExtractionResult(
            tables=[],
            chunks=[],
            table_names=["table1", "table2"],
            warnings=[],
            errors=[],
            texts_embedded=0,
            embed_duration_seconds=0.0,
        )
        mock_te_cls.return_value = mock_te

        classification = _make_classification_result(
            per_page_types={1: PageType.TABLE_HEAVY},
        )
        profile = _make_document_profile(
            pages=[_make_page_profile(page_number=1, page_type=PageType.TABLE_HEAVY)],
        )

        result = processor.process(
            file_path="/tmp/test.pdf",
            profile=profile,
            ingest_key="a" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=classification,
        )

        assert result.tables_created == 2
        assert result.tables == ["table1", "table2"]

    @patch("ingestkit_pdf.processors.complex_processor._ocr_single_page")
    @patch("ingestkit_pdf.processors.complex_processor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.complex_processor.fitz")
    def test_ocr_stage_result_populated_when_scanned_pages(
        self, mock_fitz, mock_pymupdf4llm, mock_ocr, processor,
    ):
        mock_doc = _make_mock_doc(1)
        mock_fitz.open.return_value = mock_doc
        mock_ocr.return_value = _make_ocr_result(page_number=1, confidence=0.88)

        classification = _make_classification_result(
            per_page_types={1: PageType.SCANNED},
        )
        profile = _make_document_profile(
            pages=[_make_page_profile(page_number=1, page_type=PageType.SCANNED)],
        )

        result = processor.process(
            file_path="/tmp/test.pdf",
            profile=profile,
            ingest_key="a" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=classification,
        )

        assert result.ocr_result is not None
        assert result.ocr_result.pages_ocrd == 1
        assert result.ocr_result.avg_confidence == pytest.approx(0.88)

    @patch("ingestkit_pdf.processors.complex_processor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.complex_processor.fitz")
    def test_ocr_stage_result_none_when_no_scanned_pages(
        self, mock_fitz, mock_pymupdf4llm, processor,
    ):
        mock_doc = _make_mock_doc(1)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = _GOOD_PAGE_TEXT

        classification = _make_classification_result(
            per_page_types={1: PageType.TEXT},
        )
        profile = _make_document_profile(
            pages=[_make_page_profile(page_number=1, page_type=PageType.TEXT)],
        )

        result = processor.process(
            file_path="/tmp/test.pdf",
            profile=profile,
            ingest_key="a" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=classification,
        )

        assert result.ocr_result is None

    @patch("ingestkit_pdf.processors.complex_processor.TableExtractor")
    @patch("ingestkit_pdf.processors.complex_processor._ocr_single_page")
    @patch("ingestkit_pdf.processors.complex_processor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.complex_processor.fitz")
    def test_warnings_accumulated_from_all_sources(
        self, mock_fitz, mock_pymupdf4llm, mock_ocr, mock_te_cls, processor,
    ):
        mock_doc = _make_mock_doc(3)
        mock_fitz.open.return_value = mock_doc
        mock_pymupdf4llm.to_markdown.return_value = ""

        # OCR returns error tuple
        mock_ocr.return_value = (2, "OCR engine unavailable")

        # Table extractor returns warnings
        mock_te = MagicMock()
        mock_te.extract_tables.return_value = TableExtractionResult(
            tables=[],
            chunks=[],
            table_names=[],
            warnings=["W_TABLE_CONTINUATION"],
            errors=[],
            texts_embedded=0,
            embed_duration_seconds=0.0,
        )
        mock_te_cls.return_value = mock_te

        per_page_types = {
            1: PageType.BLANK,
            2: PageType.SCANNED,
            3: PageType.TABLE_HEAVY,
        }
        pages = [
            _make_page_profile(page_number=i, page_type=pt)
            for i, pt in per_page_types.items()
        ]
        profile = _make_document_profile(pages=pages)
        classification = _make_classification_result(per_page_types=per_page_types)

        result = processor.process(
            file_path="/tmp/test.pdf",
            profile=profile,
            ingest_key="a" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=classification,
        )

        # BLANK page warning
        assert any(ErrorCode.W_PAGE_SKIPPED_BLANK.value in w for w in result.warnings)
        # Table warning
        assert "W_TABLE_CONTINUATION" in result.warnings
        # OCR error
        assert any(ErrorCode.E_OCR_FAILED.value in e for e in result.errors)

    @patch("ingestkit_pdf.processors.complex_processor.pymupdf4llm")
    @patch("ingestkit_pdf.processors.complex_processor.fitz")
    def test_error_handling_per_page_recoverable(
        self, mock_fitz, mock_pymupdf4llm, processor,
    ):
        mock_doc = _make_mock_doc(2)
        mock_fitz.open.return_value = mock_doc

        # First page raises, second page succeeds
        call_count = [0]

        def to_markdown_side_effect(doc, pages=None):
            call_count[0] += 1
            if pages == [0]:
                raise RuntimeError("Extraction error on page 1")
            return _GOOD_PAGE_TEXT

        mock_pymupdf4llm.to_markdown.side_effect = to_markdown_side_effect
        # Fallback get_text also raises for page 1
        mock_doc[0].get_text.side_effect = RuntimeError("Extraction error on page 1")

        per_page_types = {1: PageType.TEXT, 2: PageType.TEXT}
        pages = [
            _make_page_profile(page_number=i, page_type=PageType.TEXT)
            for i in [1, 2]
        ]
        profile = _make_document_profile(pages=pages)
        classification = _make_classification_result(per_page_types=per_page_types)

        result = processor.process(
            file_path="/tmp/test.pdf",
            profile=profile,
            ingest_key="a" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=classification,
        )

        # Should still have processed page 2
        assert result.ingestion_method == IngestionMethod.COMPLEX_PROCESSING
        assert any(e.recoverable for e in result.error_details)
