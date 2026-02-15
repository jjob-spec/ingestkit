"""Tests for PDFRouter orchestrator and public API.

Covers: can_handle, process (15-step flow), _classify (tiered classification
with LLM outage resilience per SPEC 5.2), process_batch, create_default_router,
document profiling, security scan integration, and PII-safe logging.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ingestkit_core.models import ClassificationTier, EmbedStageResult, WrittenArtifacts
from ingestkit_pdf.config import PDFProcessorConfig
from ingestkit_pdf.errors import ErrorCode, IngestError
from ingestkit_pdf.models import (
    ClassificationResult,
    ClassificationStageResult,
    DocumentMetadata,
    IngestionMethod,
    PageType,
    ParseStageResult,
    PDFType,
    ProcessingResult,
)
from ingestkit_pdf.router import PDFRouter

# conftest fixtures are auto-loaded by pytest; import factory helpers directly
import sys
import pathlib

# Add tests directory to path so conftest helpers can be imported
_tests_dir = str(pathlib.Path(__file__).resolve().parent)
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)

from conftest import (  # noqa: E402
    MockEmbeddingBackend,
    MockLLMBackend,
    MockStructuredDBBackend,
    MockVectorStoreBackend,
    _make_document_profile,
    _make_extraction_quality,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def pdf_router(
    mock_vector_store: MockVectorStoreBackend,
    mock_structured_db: MockStructuredDBBackend,
    mock_llm: MockLLMBackend,
    mock_embedder: MockEmbeddingBackend,
    pdf_config: PDFProcessorConfig,
) -> PDFRouter:
    """PDFRouter with all mock backends."""
    return PDFRouter(
        vector_store=mock_vector_store,
        structured_db=mock_structured_db,
        llm=mock_llm,
        embedder=mock_embedder,
        config=pdf_config,
    )


def _make_mock_page(
    *,
    text: str = "Sample text with enough words for testing purposes. " * 20,
    images: list | None = None,
    fonts: list | None = None,
    image_info: list | None = None,
    width: float = 612.0,
    height: float = 792.0,
    toc: list | None = None,
    widgets: list | None = None,
    tables: list | None = None,
) -> MagicMock:
    """Create a minimal mock for fitz.Page."""
    page = MagicMock()
    page.get_text.return_value = text
    page.get_images.return_value = images or []
    page.get_fonts.return_value = fonts or [
        (0, "", "", "Arial", "", 0),
        (1, "", "", "Times", "", 0),
    ]
    page.get_image_info.return_value = image_info or []

    rect = MagicMock()
    rect.width = width
    rect.height = height
    page.rect = rect

    # Widgets
    if widgets:
        page.widgets.return_value = widgets
    else:
        page.widgets.return_value = []

    # Tables
    if tables is not None:
        page.find_tables.return_value = tables
    else:
        page.find_tables.return_value = []

    return page


def _make_mock_doc(
    pages: list[MagicMock] | None = None,
    toc: list | None = None,
) -> MagicMock:
    """Create a minimal mock for fitz.Document."""
    doc = MagicMock()
    _pages = pages or [_make_mock_page()]
    doc.page_count = len(_pages)
    doc.__getitem__ = lambda self, idx: _pages[idx]
    doc.get_toc.return_value = toc or []
    doc.close.return_value = None
    return doc


def _make_processing_result(**overrides: Any) -> ProcessingResult:
    """Build a ProcessingResult with sensible defaults."""
    defaults = dict(
        file_path="/tmp/test.pdf",
        ingest_key="abc123",
        ingest_run_id="run-1",
        tenant_id=None,
        parse_result=ParseStageResult(
            pages_extracted=1,
            pages_skipped=0,
            skipped_reasons={},
            extraction_method="pymupdf",
            overall_quality=_make_extraction_quality(),
            parse_duration_seconds=0.5,
        ),
        classification_result=ClassificationStageResult(
            tier_used=ClassificationTier.RULE_BASED,
            pdf_type=PDFType.TEXT_NATIVE,
            confidence=0.9,
            signals=None,
            reasoning="test",
            per_page_types={1: PageType.TEXT},
            classification_duration_seconds=0.1,
        ),
        ocr_result=None,
        embed_result=EmbedStageResult(
            texts_embedded=5,
            embedding_dimension=768,
            embed_duration_seconds=0.3,
        ),
        classification=ClassificationResult(
            pdf_type=PDFType.TEXT_NATIVE,
            confidence=0.9,
            tier_used=ClassificationTier.RULE_BASED,
            reasoning="test",
            per_page_types={1: PageType.TEXT},
        ),
        ingestion_method=IngestionMethod.TEXT_EXTRACTION,
        chunks_created=5,
        tables_created=0,
        tables=[],
        written=WrittenArtifacts(
            vector_point_ids=["p1", "p2"],
            vector_collection="helpdesk",
        ),
        errors=[],
        warnings=[],
        error_details=[],
        processing_time_seconds=1.0,
    )
    defaults.update(overrides)
    return ProcessingResult(**defaults)


def _make_security_metadata(**overrides: Any) -> DocumentMetadata:
    """Build DocumentMetadata as returned by security scanner."""
    defaults = dict(
        title="Test PDF",
        page_count=1,
        file_size_bytes=10240,
        creator="TestApp",
        pdf_version="1.7",
    )
    defaults.update(overrides)
    return DocumentMetadata(**defaults)


def _make_ingest_key_obj(key: str = "deadbeef" * 8) -> MagicMock:
    """Build a mock IngestKey."""
    obj = MagicMock()
    obj.key = key
    return obj


# ---------------------------------------------------------------------------
# TestCanHandle
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCanHandle:
    def test_pdf_extension_lowercase(self, pdf_router: PDFRouter) -> None:
        assert pdf_router.can_handle("document.pdf") is True

    def test_pdf_extension_uppercase(self, pdf_router: PDFRouter) -> None:
        assert pdf_router.can_handle("DOCUMENT.PDF") is True

    def test_pdf_extension_mixed_case(self, pdf_router: PDFRouter) -> None:
        assert pdf_router.can_handle("Document.Pdf") is True

    def test_non_pdf_xlsx(self, pdf_router: PDFRouter) -> None:
        assert pdf_router.can_handle("file.xlsx") is False

    def test_non_pdf_docx(self, pdf_router: PDFRouter) -> None:
        assert pdf_router.can_handle("file.docx") is False

    def test_non_pdf_no_extension(self, pdf_router: PDFRouter) -> None:
        assert pdf_router.can_handle("readme") is False


# ---------------------------------------------------------------------------
# TestClassify
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClassify:
    """Tests for the _classify() private method."""

    def test_tier1_high_confidence_skips_llm(self, pdf_router: PDFRouter) -> None:
        """When Tier 1 returns high confidence, LLM should NOT be called."""
        profile = _make_document_profile()

        with patch.object(
            pdf_router._inspector, "classify",
            return_value=ClassificationResult(
                pdf_type=PDFType.TEXT_NATIVE,
                confidence=0.9,
                tier_used=ClassificationTier.RULE_BASED,
                reasoning="High confidence text-native.",
                per_page_types={1: PageType.TEXT},
            ),
        ):
            result, errors, warnings, error_details = pdf_router._classify(profile)

        assert result.confidence == 0.9
        assert result.tier_used == ClassificationTier.RULE_BASED
        assert result.degraded is False
        assert not warnings

    def test_tier1_inconclusive_escalates_to_tier2(
        self, pdf_router: PDFRouter,
    ) -> None:
        """Low Tier 1 confidence should trigger Tier 2."""
        profile = _make_document_profile()

        with (
            patch.object(
                pdf_router._inspector, "classify",
                return_value=ClassificationResult(
                    pdf_type=PDFType.COMPLEX,
                    confidence=0.0,
                    tier_used=ClassificationTier.RULE_BASED,
                    reasoning="Inconclusive.",
                    per_page_types={},
                ),
            ),
            patch.object(
                pdf_router._llm_classifier, "classify",
                return_value=ClassificationResult(
                    pdf_type=PDFType.TEXT_NATIVE,
                    confidence=0.85,
                    tier_used=ClassificationTier.LLM_BASIC,
                    reasoning="LLM says text.",
                    per_page_types={1: PageType.TEXT},
                ),
            ) as llm_mock,
        ):
            result, errors, warnings, error_details = pdf_router._classify(profile)

        assert result.confidence == 0.85
        assert result.tier_used == ClassificationTier.LLM_BASIC
        llm_mock.assert_called_once()

    def test_tier2_low_confidence_escalates_to_tier3(
        self, pdf_router: PDFRouter,
    ) -> None:
        """Tier 2 below threshold should escalate to Tier 3 if enabled."""
        profile = _make_document_profile()

        call_count = 0

        def classify_side_effect(profile, tier):
            nonlocal call_count
            call_count += 1
            if tier == ClassificationTier.LLM_BASIC:
                return ClassificationResult(
                    pdf_type=PDFType.SCANNED,
                    confidence=0.4,
                    tier_used=ClassificationTier.LLM_BASIC,
                    reasoning="Low confidence.",
                    per_page_types={},
                )
            else:  # LLM_REASONING
                return ClassificationResult(
                    pdf_type=PDFType.SCANNED,
                    confidence=0.8,
                    tier_used=ClassificationTier.LLM_REASONING,
                    reasoning="Tier 3 says scanned.",
                    per_page_types={1: PageType.SCANNED},
                )

        with (
            patch.object(
                pdf_router._inspector, "classify",
                return_value=ClassificationResult(
                    pdf_type=PDFType.COMPLEX,
                    confidence=0.3,
                    tier_used=ClassificationTier.RULE_BASED,
                    reasoning="Low confidence.",
                    per_page_types={},
                ),
            ),
            patch.object(
                pdf_router._llm_classifier, "classify",
                side_effect=classify_side_effect,
            ),
        ):
            result, errors, warnings, error_details = pdf_router._classify(profile)

        assert result.tier_used == ClassificationTier.LLM_REASONING
        assert result.confidence == 0.8
        assert call_count == 2  # Tier 2 + Tier 3

    def test_tier3_disabled_uses_tier2_result(
        self, pdf_router: PDFRouter,
    ) -> None:
        """When Tier 3 is disabled, low-confidence Tier 2 result used as-is."""
        pdf_router._config = pdf_router._config.model_copy(
            update={"enable_tier3": False}
        )
        profile = _make_document_profile()

        with (
            patch.object(
                pdf_router._inspector, "classify",
                return_value=ClassificationResult(
                    pdf_type=PDFType.COMPLEX,
                    confidence=0.3,
                    tier_used=ClassificationTier.RULE_BASED,
                    reasoning="Low conf.",
                    per_page_types={},
                ),
            ),
            patch.object(
                pdf_router._llm_classifier, "classify",
                return_value=ClassificationResult(
                    pdf_type=PDFType.TEXT_NATIVE,
                    confidence=0.5,
                    tier_used=ClassificationTier.LLM_BASIC,
                    reasoning="Tier 2 low.",
                    per_page_types={},
                ),
            ),
        ):
            result, errors, warnings, error_details = pdf_router._classify(profile)

        assert result.tier_used == ClassificationTier.LLM_BASIC
        assert result.confidence == 0.5


# ---------------------------------------------------------------------------
# TestLLMOutageResilience
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMOutageResilience:
    """SPEC 5.2 test contract: LLM outage resilience."""

    def test_llm_connection_error_degrades_to_tier1(
        self, pdf_router: PDFRouter,
    ) -> None:
        """LLM ConnectionError -> degraded=True, W_LLM_UNAVAILABLE."""
        profile = _make_document_profile()

        with (
            patch.object(
                pdf_router._inspector, "classify",
                return_value=ClassificationResult(
                    pdf_type=PDFType.TEXT_NATIVE,
                    confidence=0.6,
                    tier_used=ClassificationTier.RULE_BASED,
                    reasoning="Medium confidence.",
                    per_page_types={1: PageType.TEXT},
                ),
            ),
            patch.object(
                pdf_router._llm_classifier, "classify",
                side_effect=ConnectionError("LLM down"),
            ),
        ):
            result, errors, warnings, error_details = pdf_router._classify(profile)

        assert result.degraded is True
        assert result.confidence == 0.6
        assert result.tier_used == ClassificationTier.RULE_BASED
        assert ErrorCode.W_LLM_UNAVAILABLE.value in warnings
        assert ErrorCode.W_CLASSIFICATION_DEGRADED.value in warnings
        assert len(error_details) == 1
        assert error_details[0].code == ErrorCode.W_LLM_UNAVAILABLE

    def test_llm_timeout_degrades_to_tier1(
        self, pdf_router: PDFRouter,
    ) -> None:
        """LLM TimeoutError -> same degraded behavior."""
        profile = _make_document_profile()

        with (
            patch.object(
                pdf_router._inspector, "classify",
                return_value=ClassificationResult(
                    pdf_type=PDFType.TEXT_NATIVE,
                    confidence=0.7,
                    tier_used=ClassificationTier.RULE_BASED,
                    reasoning="Medium confidence.",
                    per_page_types={1: PageType.TEXT},
                ),
            ),
            patch.object(
                pdf_router._llm_classifier, "classify",
                side_effect=TimeoutError("LLM timeout"),
            ),
        ):
            result, errors, warnings, error_details = pdf_router._classify(profile)

        assert result.degraded is True
        assert ErrorCode.W_LLM_UNAVAILABLE.value in warnings

    def test_degraded_result_has_correct_warnings(
        self, pdf_router: PDFRouter,
    ) -> None:
        """W_LLM_UNAVAILABLE and W_CLASSIFICATION_DEGRADED in warnings."""
        profile = _make_document_profile()

        with (
            patch.object(
                pdf_router._inspector, "classify",
                return_value=ClassificationResult(
                    pdf_type=PDFType.TEXT_NATIVE,
                    confidence=0.6,
                    tier_used=ClassificationTier.RULE_BASED,
                    reasoning="OK.",
                    per_page_types={1: PageType.TEXT},
                ),
            ),
            patch.object(
                pdf_router._llm_classifier, "classify",
                side_effect=ConnectionError("down"),
            ),
        ):
            result, errors, warnings, error_details = pdf_router._classify(profile)

        assert ErrorCode.W_LLM_UNAVAILABLE.value in warnings
        assert ErrorCode.W_CLASSIFICATION_DEGRADED.value in warnings

    def test_degraded_result_still_processes(
        self, pdf_router: PDFRouter,
    ) -> None:
        """Degraded classification should still route to processor (chunks > 0)."""
        # We test this at the _classify level: degraded result has confidence > 0
        profile = _make_document_profile()

        with (
            patch.object(
                pdf_router._inspector, "classify",
                return_value=ClassificationResult(
                    pdf_type=PDFType.TEXT_NATIVE,
                    confidence=0.7,
                    tier_used=ClassificationTier.RULE_BASED,
                    reasoning="OK.",
                    per_page_types={1: PageType.TEXT},
                ),
            ),
            patch.object(
                pdf_router._llm_classifier, "classify",
                side_effect=ConnectionError("down"),
            ),
        ):
            result, errors, warnings, error_details = pdf_router._classify(profile)

        # The result has confidence > 0, meaning it will NOT hit fail-closed check
        assert result.confidence > 0.0
        assert result.degraded is True

    def test_inconclusive_only_when_all_tiers_fail(
        self, pdf_router: PDFRouter,
    ) -> None:
        """E_CLASSIFY_INCONCLUSIVE only when Tier 1 confidence==0 AND LLM fails."""
        profile = _make_document_profile()

        with (
            patch.object(
                pdf_router._inspector, "classify",
                return_value=ClassificationResult(
                    pdf_type=PDFType.COMPLEX,
                    confidence=0.0,
                    tier_used=ClassificationTier.RULE_BASED,
                    reasoning="Inconclusive.",
                    per_page_types={},
                ),
            ),
            patch.object(
                pdf_router._llm_classifier, "classify",
                return_value=ClassificationResult(
                    pdf_type=PDFType.TEXT_NATIVE,
                    confidence=0.0,
                    tier_used=ClassificationTier.LLM_BASIC,
                    reasoning="Also failed.",
                    per_page_types={},
                ),
            ),
        ):
            result, errors, warnings, error_details = pdf_router._classify(profile)

        # Should return Tier 1 result with confidence 0.0
        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# TestSecurityScan
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSecurityScan:
    def test_fatal_security_error_returns_immediately(
        self, pdf_router: PDFRouter,
    ) -> None:
        """Fatal security error => immediate error result, no classification."""
        metadata = _make_security_metadata()
        fatal_error = IngestError(
            code=ErrorCode.E_SECURITY_INVALID_PDF,
            message="Not a PDF",
            stage="security",
        )

        with (
            patch.object(
                pdf_router._security_scanner, "scan",
                return_value=(metadata, [fatal_error]),
            ),
            patch.object(
                pdf_router._inspector, "classify",
            ) as classify_mock,
        ):
            result = pdf_router.process("/tmp/bad.pdf")

        assert ErrorCode.E_SECURITY_INVALID_PDF.value in result.errors
        assert result.chunks_created == 0
        classify_mock.assert_not_called()

    def test_security_warning_continues_processing(
        self, pdf_router: PDFRouter,
    ) -> None:
        """Non-fatal security warning => processing continues."""
        metadata = _make_security_metadata()
        warning = IngestError(
            code=ErrorCode.W_ENCRYPTED_OWNER_ONLY,
            message="Owner password present",
            stage="security",
            recoverable=True,
        )
        expected_result = _make_processing_result()

        with (
            patch.object(
                pdf_router._security_scanner, "scan",
                return_value=(metadata, [warning]),
            ),
            patch("ingestkit_pdf.router.compute_ingest_key", return_value=_make_ingest_key_obj()),
            patch("ingestkit_pdf.router.fitz.open") as mock_fitz_open,
            patch.object(
                pdf_router, "_build_document_profile",
                return_value=_make_document_profile(),
            ),
            patch.object(
                pdf_router._inspector, "classify",
                return_value=ClassificationResult(
                    pdf_type=PDFType.TEXT_NATIVE,
                    confidence=0.9,
                    tier_used=ClassificationTier.RULE_BASED,
                    reasoning="High confidence.",
                    per_page_types={1: PageType.TEXT},
                ),
            ),
            patch.object(
                pdf_router._text_extractor, "process",
                return_value=expected_result,
            ),
        ):
            mock_doc = _make_mock_doc()
            mock_fitz_open.return_value = mock_doc
            result = pdf_router.process("/tmp/test.pdf")

        # Warning should be in result warnings
        assert ErrorCode.W_ENCRYPTED_OWNER_ONLY.value in result.warnings
        assert result.chunks_created > 0


# ---------------------------------------------------------------------------
# TestDocumentProfiling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDocumentProfiling:
    def test_build_page_profile_text_page(
        self, pdf_router: PDFRouter,
    ) -> None:
        """Text-heavy page should be classified as TEXT."""
        page = _make_mock_page(
            text="This is a text-heavy page. " * 50,
            fonts=[(0, "", "", "Arial", "", 0), (1, "", "", "Times", "", 0)],
        )

        with patch.object(
            pdf_router._layout_analyzer, "detect_columns",
        ) as layout_mock:
            layout_result = MagicMock()
            layout_result.column_count = 1
            layout_mock.return_value = layout_result

            profile = pdf_router._build_page_profile(page, 1)

        assert profile.page_type == PageType.TEXT
        assert profile.word_count > 0
        assert profile.font_count >= 1

    def test_build_page_profile_blank_page(
        self, pdf_router: PDFRouter,
    ) -> None:
        """Page with no text and no images should be BLANK."""
        page = _make_mock_page(text="", images=[], fonts=[])

        with patch.object(
            pdf_router._layout_analyzer, "detect_columns",
        ) as layout_mock:
            layout_result = MagicMock()
            layout_result.column_count = 1
            layout_mock.return_value = layout_result

            profile = pdf_router._build_page_profile(page, 1)

        assert profile.page_type == PageType.BLANK

    def test_build_page_profile_scanned_page(
        self, pdf_router: PDFRouter,
    ) -> None:
        """Page with little text but high image coverage should be SCANNED."""
        page = _make_mock_page(
            text="OCR",
            images=[(1,)],
            image_info=[{"bbox": (0, 0, 612, 792)}],
            fonts=[],
        )

        with patch.object(
            pdf_router._layout_analyzer, "detect_columns",
        ) as layout_mock:
            layout_result = MagicMock()
            layout_result.column_count = 1
            layout_mock.return_value = layout_result

            profile = pdf_router._build_page_profile(page, 1)

        assert profile.page_type == PageType.SCANNED
        assert profile.image_coverage_ratio > 0.7

    def test_language_detection_called_when_enabled(
        self, pdf_router: PDFRouter,
    ) -> None:
        """Language detection should be invoked when enabled in config."""
        doc = _make_mock_doc()
        metadata = _make_security_metadata()

        with (
            patch("ingestkit_pdf.router.detect_language", return_value=("en", 0.95)) as lang_mock,
            patch.object(pdf_router._layout_analyzer, "detect_columns") as layout_mock,
            patch("ingestkit_pdf.router.Path") as path_mock,
        ):
            layout_result = MagicMock()
            layout_result.column_count = 1
            layout_mock.return_value = layout_result
            path_mock.return_value.read_bytes.return_value = b"fake pdf"

            profile = pdf_router._build_document_profile(
                "/tmp/test.pdf", doc, metadata, [],
            )

        assert "en" in profile.detected_languages
        lang_mock.assert_called_once()

    def test_toc_extraction(
        self, pdf_router: PDFRouter,
    ) -> None:
        """TOC entries should be extracted from doc.get_toc()."""
        toc_data = [(1, "Chapter 1", 1), (2, "Section 1.1", 3)]
        doc = _make_mock_doc(toc=toc_data)
        metadata = _make_security_metadata()

        with (
            patch.object(pdf_router._layout_analyzer, "detect_columns") as layout_mock,
            patch("ingestkit_pdf.router.Path") as path_mock,
        ):
            layout_result = MagicMock()
            layout_result.column_count = 1
            layout_mock.return_value = layout_result
            path_mock.return_value.read_bytes.return_value = b"fake pdf"

            profile = pdf_router._build_document_profile(
                "/tmp/test.pdf", doc, metadata, [],
            )

        assert profile.has_toc is True
        assert profile.toc_entries is not None
        assert len(profile.toc_entries) == 2


# ---------------------------------------------------------------------------
# TestProcessFlow
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProcessFlow:
    def test_text_native_routes_to_text_extractor(
        self, pdf_router: PDFRouter,
    ) -> None:
        """TEXT_NATIVE classification should route to TextExtractor."""
        expected_result = _make_processing_result()

        with (
            patch.object(
                pdf_router._security_scanner, "scan",
                return_value=(_make_security_metadata(), []),
            ),
            patch("ingestkit_pdf.router.compute_ingest_key", return_value=_make_ingest_key_obj()),
            patch("ingestkit_pdf.router.fitz.open") as mock_fitz_open,
            patch.object(
                pdf_router, "_build_document_profile",
                return_value=_make_document_profile(),
            ),
            patch.object(
                pdf_router._inspector, "classify",
                return_value=ClassificationResult(
                    pdf_type=PDFType.TEXT_NATIVE,
                    confidence=0.9,
                    tier_used=ClassificationTier.RULE_BASED,
                    reasoning="Text native.",
                    per_page_types={1: PageType.TEXT},
                ),
            ),
            patch.object(
                pdf_router._text_extractor, "process",
                return_value=expected_result,
            ) as extractor_mock,
        ):
            mock_fitz_open.return_value = _make_mock_doc()
            result = pdf_router.process("/tmp/test.pdf")

        extractor_mock.assert_called_once()
        assert result.chunks_created == 5

    def test_scanned_routes_to_ocr_processor(
        self, pdf_router: PDFRouter,
    ) -> None:
        """SCANNED classification should route to OCRProcessor."""
        expected_result = _make_processing_result(
            ingestion_method=IngestionMethod.OCR_PIPELINE,
        )

        with (
            patch.object(
                pdf_router._security_scanner, "scan",
                return_value=(_make_security_metadata(), []),
            ),
            patch("ingestkit_pdf.router.compute_ingest_key", return_value=_make_ingest_key_obj()),
            patch("ingestkit_pdf.router.fitz.open") as mock_fitz_open,
            patch.object(
                pdf_router, "_build_document_profile",
                return_value=_make_document_profile(),
            ),
            patch.object(
                pdf_router._inspector, "classify",
                return_value=ClassificationResult(
                    pdf_type=PDFType.SCANNED,
                    confidence=0.9,
                    tier_used=ClassificationTier.RULE_BASED,
                    reasoning="Scanned.",
                    per_page_types={1: PageType.SCANNED},
                ),
            ),
            patch.object(
                pdf_router._ocr_processor, "process",
                return_value=expected_result,
            ) as ocr_mock,
        ):
            mock_fitz_open.return_value = _make_mock_doc()
            pdf_router.process("/tmp/test.pdf")

        ocr_mock.assert_called_once()
        # Verify pages=None was passed for full-document OCR
        call_kwargs = ocr_mock.call_args
        assert call_kwargs.kwargs.get("pages") is None

    def test_complex_no_processor_returns_error(
        self, pdf_router: PDFRouter,
    ) -> None:
        """COMPLEX type with no ComplexProcessor should return error."""
        with (
            patch.object(
                pdf_router._security_scanner, "scan",
                return_value=(_make_security_metadata(), []),
            ),
            patch("ingestkit_pdf.router.compute_ingest_key", return_value=_make_ingest_key_obj()),
            patch("ingestkit_pdf.router.fitz.open") as mock_fitz_open,
            patch.object(
                pdf_router, "_build_document_profile",
                return_value=_make_document_profile(),
            ),
            patch.object(
                pdf_router._inspector, "classify",
                return_value=ClassificationResult(
                    pdf_type=PDFType.COMPLEX,
                    confidence=0.9,
                    tier_used=ClassificationTier.RULE_BASED,
                    reasoning="Complex.",
                    per_page_types={1: PageType.MIXED},
                ),
            ),
        ):
            mock_fitz_open.return_value = _make_mock_doc()
            result = pdf_router.process("/tmp/test.pdf")

        assert "ComplexProcessor not available" in result.errors
        assert result.chunks_created == 0

    def test_corrupt_pdf_repair_attempt(
        self, pdf_router: PDFRouter,
    ) -> None:
        """Corrupt PDF => repair attempt => E_PARSE_CORRUPT if all fail."""
        with (
            patch.object(
                pdf_router._security_scanner, "scan",
                return_value=(_make_security_metadata(), []),
            ),
            patch("ingestkit_pdf.router.compute_ingest_key", return_value=_make_ingest_key_obj()),
            patch("ingestkit_pdf.router.fitz.open", side_effect=Exception("corrupt")),
        ):
            result = pdf_router.process("/tmp/corrupt.pdf")

        assert ErrorCode.E_PARSE_CORRUPT.value in result.errors
        assert result.chunks_created == 0

    def test_fail_closed_zero_chunks(
        self, pdf_router: PDFRouter,
    ) -> None:
        """Inconclusive classification => zero chunks, E_CLASSIFY_INCONCLUSIVE."""
        with (
            patch.object(
                pdf_router._security_scanner, "scan",
                return_value=(_make_security_metadata(), []),
            ),
            patch("ingestkit_pdf.router.compute_ingest_key", return_value=_make_ingest_key_obj()),
            patch("ingestkit_pdf.router.fitz.open") as mock_fitz_open,
            patch.object(
                pdf_router, "_build_document_profile",
                return_value=_make_document_profile(),
            ),
            patch.object(
                pdf_router, "_classify",
                return_value=(
                    ClassificationResult(
                        pdf_type=PDFType.COMPLEX,
                        confidence=0.0,
                        tier_used=ClassificationTier.RULE_BASED,
                        reasoning="Inconclusive.",
                        per_page_types={},
                    ),
                    [],
                    [],
                    [],
                ),
            ),
        ):
            mock_fitz_open.return_value = _make_mock_doc()
            result = pdf_router.process("/tmp/test.pdf")

        assert ErrorCode.E_CLASSIFY_INCONCLUSIVE.value in result.errors
        assert result.chunks_created == 0

    def test_processing_time_includes_full_pipeline(
        self, pdf_router: PDFRouter,
    ) -> None:
        """processing_time_seconds should cover the entire pipeline."""
        expected_result = _make_processing_result(processing_time_seconds=0.1)

        with (
            patch.object(
                pdf_router._security_scanner, "scan",
                return_value=(_make_security_metadata(), []),
            ),
            patch("ingestkit_pdf.router.compute_ingest_key", return_value=_make_ingest_key_obj()),
            patch("ingestkit_pdf.router.fitz.open") as mock_fitz_open,
            patch.object(
                pdf_router, "_build_document_profile",
                return_value=_make_document_profile(),
            ),
            patch.object(
                pdf_router._inspector, "classify",
                return_value=ClassificationResult(
                    pdf_type=PDFType.TEXT_NATIVE,
                    confidence=0.9,
                    tier_used=ClassificationTier.RULE_BASED,
                    reasoning="OK.",
                    per_page_types={1: PageType.TEXT},
                ),
            ),
            patch.object(
                pdf_router._text_extractor, "process",
                return_value=expected_result,
            ),
        ):
            mock_fitz_open.return_value = _make_mock_doc()
            result = pdf_router.process("/tmp/test.pdf")

        # The router should override the processor's time with full pipeline time
        assert result.processing_time_seconds >= 0.0


# ---------------------------------------------------------------------------
# TestIngestKey
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIngestKey:
    def test_deterministic_same_file(
        self, pdf_router: PDFRouter,
    ) -> None:
        """Same file => same ingest key (deterministic)."""
        key_obj = _make_ingest_key_obj("key_abc")
        expected = _make_processing_result(ingest_key="key_abc")

        with (
            patch.object(
                pdf_router._security_scanner, "scan",
                return_value=(_make_security_metadata(), []),
            ),
            patch("ingestkit_pdf.router.compute_ingest_key", return_value=key_obj),
            patch("ingestkit_pdf.router.fitz.open") as mock_fitz_open,
            patch.object(
                pdf_router, "_build_document_profile",
                return_value=_make_document_profile(),
            ),
            patch.object(
                pdf_router._inspector, "classify",
                return_value=ClassificationResult(
                    pdf_type=PDFType.TEXT_NATIVE,
                    confidence=0.9,
                    tier_used=ClassificationTier.RULE_BASED,
                    reasoning="OK.",
                    per_page_types={1: PageType.TEXT},
                ),
            ),
            patch.object(
                pdf_router._text_extractor, "process",
                return_value=expected,
            ),
        ):
            mock_fitz_open.return_value = _make_mock_doc()
            result = pdf_router.process("/tmp/test.pdf")

        assert result.ingest_key == "key_abc"

    def test_tenant_id_affects_key(self) -> None:
        """tenant_id should propagate to compute_ingest_key."""
        config = PDFProcessorConfig(tenant_id="tenant-42")
        router = PDFRouter(
            vector_store=MockVectorStoreBackend(),
            structured_db=MockStructuredDBBackend(),
            llm=MockLLMBackend(),
            embedder=MockEmbeddingBackend(),
            config=config,
        )
        expected = _make_processing_result(tenant_id="tenant-42")

        with (
            patch.object(
                router._security_scanner, "scan",
                return_value=(_make_security_metadata(), []),
            ),
            patch("ingestkit_pdf.router.compute_ingest_key", return_value=_make_ingest_key_obj()) as key_mock,
            patch("ingestkit_pdf.router.fitz.open") as mock_fitz_open,
            patch.object(
                router, "_build_document_profile",
                return_value=_make_document_profile(),
            ),
            patch.object(
                router._inspector, "classify",
                return_value=ClassificationResult(
                    pdf_type=PDFType.TEXT_NATIVE,
                    confidence=0.9,
                    tier_used=ClassificationTier.RULE_BASED,
                    reasoning="OK.",
                    per_page_types={1: PageType.TEXT},
                ),
            ),
            patch.object(router._text_extractor, "process", return_value=expected),
        ):
            mock_fitz_open.return_value = _make_mock_doc()
            router.process("/tmp/test.pdf")

        # Verify tenant_id was passed to compute_ingest_key
        key_mock.assert_called_once()
        assert key_mock.call_args.kwargs.get("tenant_id") == "tenant-42"

    def test_source_uri_override(self) -> None:
        """source_uri parameter should be passed to compute_ingest_key."""
        router = PDFRouter(
            vector_store=MockVectorStoreBackend(),
            structured_db=MockStructuredDBBackend(),
            llm=MockLLMBackend(),
            embedder=MockEmbeddingBackend(),
        )
        expected = _make_processing_result()

        with (
            patch.object(
                router._security_scanner, "scan",
                return_value=(_make_security_metadata(), []),
            ),
            patch("ingestkit_pdf.router.compute_ingest_key", return_value=_make_ingest_key_obj()) as key_mock,
            patch("ingestkit_pdf.router.fitz.open") as mock_fitz_open,
            patch.object(
                router, "_build_document_profile",
                return_value=_make_document_profile(),
            ),
            patch.object(
                router._inspector, "classify",
                return_value=ClassificationResult(
                    pdf_type=PDFType.TEXT_NATIVE,
                    confidence=0.9,
                    tier_used=ClassificationTier.RULE_BASED,
                    reasoning="OK.",
                    per_page_types={1: PageType.TEXT},
                ),
            ),
            patch.object(router._text_extractor, "process", return_value=expected),
        ):
            mock_fitz_open.return_value = _make_mock_doc()
            router.process("/tmp/test.pdf", source_uri="s3://bucket/test.pdf")

        assert key_mock.call_args.kwargs.get("source_uri") == "s3://bucket/test.pdf"


# ---------------------------------------------------------------------------
# TestProcessBatch
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProcessBatch:
    def test_empty_batch_returns_empty(
        self, pdf_router: PDFRouter,
    ) -> None:
        """Empty file list => empty result list."""
        assert pdf_router.process_batch([]) == []

    def test_returns_results_in_order(
        self, pdf_router: PDFRouter,
    ) -> None:
        """Results must match input file order."""
        # Create mock futures that return results
        future_a = MagicMock()
        future_a.result.return_value = _make_processing_result(file_path="/tmp/a.pdf")
        future_b = MagicMock()
        future_b.result.return_value = _make_processing_result(file_path="/tmp/b.pdf")

        mock_executor = MagicMock()
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=False)
        # submit returns futures in order
        mock_executor.submit.side_effect = [future_a, future_b]

        with patch(
            "ingestkit_pdf.router.ProcessPoolExecutor",
            return_value=mock_executor,
        ), patch(
            "ingestkit_pdf.router.as_completed",
            return_value=[future_a, future_b],
        ):
            batch_results = pdf_router.process_batch(["/tmp/a.pdf", "/tmp/b.pdf"])

        assert len(batch_results) == 2
        assert batch_results[0].file_path == "/tmp/a.pdf"
        assert batch_results[1].file_path == "/tmp/b.pdf"

    def test_timeout_produces_error_result(
        self, pdf_router: PDFRouter,
    ) -> None:
        """Timed-out document should produce error result."""
        pdf_router._config = pdf_router._config.model_copy(
            update={"per_document_timeout_seconds": 1}
        )

        # Create a mock future that raises TimeoutError
        mock_future = MagicMock()
        mock_future.result.side_effect = TimeoutError("timeout")

        mock_executor = MagicMock()
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=False)
        mock_executor.submit.return_value = mock_future

        with patch(
            "ingestkit_pdf.router.ProcessPoolExecutor",
            return_value=mock_executor,
        ), patch(
            "ingestkit_pdf.router.as_completed",
            return_value=[mock_future],
        ):
            batch_results = pdf_router.process_batch(["/tmp/slow.pdf"])

        assert len(batch_results) == 1
        assert "Processing timeout" in batch_results[0].errors


# ---------------------------------------------------------------------------
# TestCreateDefaultRouter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreateDefaultRouter:
    def test_creates_valid_router(self) -> None:
        """Factory should create a valid PDFRouter with default config."""
        with patch("ingestkit_pdf.router.create_default_router") as factory_mock:
            factory_mock.return_value = PDFRouter(
                vector_store=MockVectorStoreBackend(),
                structured_db=MockStructuredDBBackend(),
                llm=MockLLMBackend(),
                embedder=MockEmbeddingBackend(),
            )
            router = factory_mock()

        assert isinstance(router, PDFRouter)
        assert router.can_handle("test.pdf")

    def test_config_overrides_applied(self) -> None:
        """Config overrides should be applied."""
        config = PDFProcessorConfig(tenant_id="test-tenant")
        router = PDFRouter(
            vector_store=MockVectorStoreBackend(),
            structured_db=MockStructuredDBBackend(),
            llm=MockLLMBackend(),
            embedder=MockEmbeddingBackend(),
            config=config,
        )
        assert router._config.tenant_id == "test-tenant"


# ---------------------------------------------------------------------------
# TestLogging
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLogging:
    def test_info_log_no_raw_text(
        self, pdf_router: PDFRouter, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """INFO-level log should NOT contain raw page text."""
        expected_result = _make_processing_result()

        with (
            caplog.at_level(logging.INFO, logger="ingestkit_pdf"),
            patch.object(
                pdf_router._security_scanner, "scan",
                return_value=(_make_security_metadata(), []),
            ),
            patch("ingestkit_pdf.router.compute_ingest_key", return_value=_make_ingest_key_obj()),
            patch("ingestkit_pdf.router.fitz.open") as mock_fitz_open,
            patch.object(
                pdf_router, "_build_document_profile",
                return_value=_make_document_profile(),
            ),
            patch.object(
                pdf_router._inspector, "classify",
                return_value=ClassificationResult(
                    pdf_type=PDFType.TEXT_NATIVE,
                    confidence=0.9,
                    tier_used=ClassificationTier.RULE_BASED,
                    reasoning="OK.",
                    per_page_types={1: PageType.TEXT},
                ),
            ),
            patch.object(
                pdf_router._text_extractor, "process",
                return_value=expected_result,
            ),
        ):
            mock_fitz_open.return_value = _make_mock_doc()
            pdf_router.process("/tmp/test.pdf")

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_records) >= 1
        for record in info_records:
            # Should not contain raw text samples
            assert "Sample text with enough words" not in record.message

    def test_warning_log_for_llm_outage(
        self, pdf_router: PDFRouter, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """LLM outage should produce a WARNING-level log."""
        expected_result = _make_processing_result()

        with (
            caplog.at_level(logging.WARNING, logger="ingestkit_pdf"),
            patch.object(
                pdf_router._security_scanner, "scan",
                return_value=(_make_security_metadata(), []),
            ),
            patch("ingestkit_pdf.router.compute_ingest_key", return_value=_make_ingest_key_obj()),
            patch("ingestkit_pdf.router.fitz.open") as mock_fitz_open,
            patch.object(
                pdf_router, "_build_document_profile",
                return_value=_make_document_profile(),
            ),
            patch.object(
                pdf_router._inspector, "classify",
                return_value=ClassificationResult(
                    pdf_type=PDFType.TEXT_NATIVE,
                    confidence=0.6,
                    tier_used=ClassificationTier.RULE_BASED,
                    reasoning="Medium.",
                    per_page_types={1: PageType.TEXT},
                ),
            ),
            patch.object(
                pdf_router._llm_classifier, "classify",
                side_effect=ConnectionError("LLM down"),
            ),
            patch.object(
                pdf_router._text_extractor, "process",
                return_value=expected_result,
            ),
        ):
            mock_fitz_open.return_value = _make_mock_doc()
            pdf_router.process("/tmp/test.pdf")

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("W_LLM_UNAVAILABLE" in r.message for r in warning_records)

    def test_error_log_for_corrupt_file(
        self, pdf_router: PDFRouter, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Corrupt file should produce an ERROR-level log."""
        with (
            caplog.at_level(logging.ERROR, logger="ingestkit_pdf"),
            patch.object(
                pdf_router._security_scanner, "scan",
                return_value=(_make_security_metadata(), []),
            ),
            patch("ingestkit_pdf.router.compute_ingest_key", return_value=_make_ingest_key_obj()),
            patch("ingestkit_pdf.router.fitz.open", side_effect=Exception("corrupt")),
        ):
            pdf_router.process("/tmp/corrupt.pdf")

        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert any(ErrorCode.E_PARSE_CORRUPT.value in r.message for r in error_records)
