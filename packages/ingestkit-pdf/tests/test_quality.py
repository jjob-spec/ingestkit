"""Tests for ingestkit_pdf.quality — extraction quality scoring."""

from __future__ import annotations

import pytest

from ingestkit_pdf.config import PDFProcessorConfig
from ingestkit_pdf.models import ExtractionQuality, ExtractionQualityGrade
from ingestkit_pdf.quality import QualityAssessor


@pytest.fixture
def assessor():
    return QualityAssessor(PDFProcessorConfig())


# ---------------------------------------------------------------------------
# assess_page Tests
# ---------------------------------------------------------------------------


class TestAssessPage:
    def test_clean_text_high_quality(self, assessor):
        text = "This is a clean digital text page with many words. " * 20
        q = assessor.assess_page(text, page_number=0)
        assert q.printable_ratio > 0.9
        assert q.avg_words_per_page > 10
        assert q.pages_with_text == 1
        assert q.total_pages == 1
        assert q.grade in (ExtractionQualityGrade.HIGH, ExtractionQualityGrade.MEDIUM)

    def test_empty_text(self, assessor):
        q = assessor.assess_page("", page_number=0)
        assert q.printable_ratio == 0.0
        assert q.avg_words_per_page == 0.0
        assert q.pages_with_text == 0
        assert q.grade == ExtractionQualityGrade.LOW

    def test_garbled_text(self, assessor):
        # Simulate garbled CIDFont output with many non-printable chars
        garbled = "\x00\x01\x02\x03\x04" * 100 + "some words here"
        q = assessor.assess_page(garbled, page_number=0)
        assert q.printable_ratio < 0.85

    def test_sparse_text(self, assessor):
        # Very few words (below quality_min_words_per_page=10)
        q = assessor.assess_page("hello world", page_number=0)
        assert q.pages_with_text == 0  # below min words threshold
        assert q.avg_words_per_page == 2.0

    def test_whitespace_only(self, assessor):
        q = assessor.assess_page("   \n\t  \n  ", page_number=0)
        assert q.pages_with_text == 0
        assert q.avg_words_per_page == 0.0

    def test_extraction_method_native(self, assessor):
        q = assessor.assess_page("Test text", page_number=0)
        assert q.extraction_method == "native"

    def test_single_word_above_threshold(self):
        config = PDFProcessorConfig(quality_min_words_per_page=1)
        assessor = QualityAssessor(config)
        q = assessor.assess_page("hello", page_number=0)
        assert q.pages_with_text == 1

    def test_printable_ratio_all_printable(self, assessor):
        text = "All printable ASCII characters"
        q = assessor.assess_page(text, page_number=0)
        assert q.printable_ratio == 1.0


# ---------------------------------------------------------------------------
# assess_document Tests
# ---------------------------------------------------------------------------


def _make_quality(
    printable_ratio: float = 0.95,
    avg_words_per_page: float = 200.0,
    pages_with_text: int = 1,
    total_pages: int = 1,
    extraction_method: str = "native",
) -> ExtractionQuality:
    return ExtractionQuality(
        printable_ratio=printable_ratio,
        avg_words_per_page=avg_words_per_page,
        pages_with_text=pages_with_text,
        total_pages=total_pages,
        extraction_method=extraction_method,
    )


class TestAssessDocument:
    def test_all_high_quality(self, assessor):
        pages = [_make_quality() for _ in range(10)]
        doc_q = assessor.assess_document(pages)
        assert doc_q.total_pages == 10
        assert doc_q.pages_with_text == 10
        assert doc_q.grade == ExtractionQualityGrade.HIGH

    def test_mixed_quality(self, assessor):
        pages = [
            _make_quality(printable_ratio=0.95, avg_words_per_page=200, pages_with_text=1),
            _make_quality(printable_ratio=0.3, avg_words_per_page=5, pages_with_text=0),
            _make_quality(printable_ratio=0.95, avg_words_per_page=200, pages_with_text=1),
        ]
        doc_q = assessor.assess_document(pages)
        assert doc_q.total_pages == 3
        assert doc_q.pages_with_text == 2

    def test_empty_pages_list(self, assessor):
        doc_q = assessor.assess_document([])
        assert doc_q.total_pages == 0
        assert doc_q.grade == ExtractionQualityGrade.LOW

    def test_single_page(self, assessor):
        doc_q = assessor.assess_document([_make_quality()])
        assert doc_q.total_pages == 1
        assert doc_q.pages_with_text == 1

    def test_extraction_method_unified(self, assessor):
        pages = [_make_quality(extraction_method="native") for _ in range(3)]
        doc_q = assessor.assess_document(pages)
        assert doc_q.extraction_method == "native"

    def test_extraction_method_mixed_with_ocr_fallback(self, assessor):
        pages = [
            _make_quality(extraction_method="native"),
            _make_quality(extraction_method="ocr_fallback"),
        ]
        doc_q = assessor.assess_document(pages)
        assert doc_q.extraction_method == "ocr_fallback"

    def test_all_low_quality(self, assessor):
        pages = [
            _make_quality(printable_ratio=0.2, avg_words_per_page=3, pages_with_text=0)
            for _ in range(5)
        ]
        doc_q = assessor.assess_document(pages)
        assert doc_q.grade == ExtractionQualityGrade.LOW


# ---------------------------------------------------------------------------
# needs_ocr_fallback Tests
# ---------------------------------------------------------------------------


class TestNeedsOCRFallback:
    def test_low_quality_with_auto_fallback(self, assessor):
        q = _make_quality(printable_ratio=0.2, avg_words_per_page=3, pages_with_text=0)
        assert q.grade == ExtractionQualityGrade.LOW
        assert assessor.needs_ocr_fallback(q) is True

    def test_high_quality_no_fallback(self, assessor):
        q = _make_quality()
        assert q.grade == ExtractionQualityGrade.HIGH
        assert assessor.needs_ocr_fallback(q) is False

    def test_medium_quality_no_fallback(self, assessor):
        q = _make_quality(printable_ratio=0.7, avg_words_per_page=80, pages_with_text=1)
        assert q.grade == ExtractionQualityGrade.MEDIUM
        assert assessor.needs_ocr_fallback(q) is False

    def test_low_quality_fallback_disabled(self):
        config = PDFProcessorConfig(auto_ocr_fallback=False)
        assessor = QualityAssessor(config)
        q = _make_quality(printable_ratio=0.2, avg_words_per_page=3, pages_with_text=0)
        assert assessor.needs_ocr_fallback(q) is False

    def test_empty_text_triggers_fallback(self, assessor):
        q = _make_quality(printable_ratio=0.0, avg_words_per_page=0, pages_with_text=0)
        assert assessor.needs_ocr_fallback(q) is True


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_unicode_text(self, assessor):
        """Unicode text should still compute printable ratio correctly."""
        text = "Caf\u00e9 r\u00e9sum\u00e9 na\u00efve" * 10
        q = assessor.assess_page(text, page_number=0)
        # Non-ASCII chars are not in string.printable
        assert 0.0 < q.printable_ratio < 1.0

    def test_very_long_text(self, assessor):
        text = "word " * 10000
        q = assessor.assess_page(text, page_number=0)
        assert q.avg_words_per_page == 10000
        assert q.pages_with_text == 1

    def test_custom_min_words_threshold(self):
        config = PDFProcessorConfig(quality_min_words_per_page=50)
        assessor = QualityAssessor(config)
        # 20 words — below custom threshold
        text = "word " * 20
        q = assessor.assess_page(text.strip(), page_number=0)
        assert q.pages_with_text == 0

    def test_custom_min_printable_ratio(self):
        """Config threshold is used in the config, but assess_page computes raw ratio."""
        config = PDFProcessorConfig(quality_min_printable_ratio=0.5)
        assessor = QualityAssessor(config)
        text = "clean text " * 10
        q = assessor.assess_page(text, page_number=0)
        assert q.printable_ratio > 0.9
