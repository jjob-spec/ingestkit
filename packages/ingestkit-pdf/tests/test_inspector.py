"""Tests for the PDFInspector Tier 1 rule-based classifier.

Covers per-page signal evaluation, per-page classification (text-native,
scanned, complex, borderline), document-level aggregation (agreement,
disagreement), special page handling (blank, TOC, vector-only),
inconclusive escalation, edge cases, result field validation,
custom configs, and boundary values.
"""

from __future__ import annotations

import pytest

from ingestkit_pdf.config import PDFProcessorConfig
from ingestkit_pdf.inspector import PDFInspector
from ingestkit_pdf.models import (
    ClassificationResult,
    ClassificationTier,
    DocumentMetadata,
    DocumentProfile,
    ExtractionQuality,
    PageProfile,
    PageType,
    PDFType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_extraction_quality(**overrides: object) -> ExtractionQuality:
    defaults = dict(
        printable_ratio=0.95,
        avg_words_per_page=80.0,
        pages_with_text=1,
        total_pages=1,
        extraction_method="pdfplumber",
    )
    defaults.update(overrides)
    return ExtractionQuality(**defaults)


def _make_page_profile(**overrides: object) -> PageProfile:
    """Build a PageProfile with sensible text-native defaults."""
    defaults = dict(
        page_number=1,
        text_length=500,
        word_count=80,
        image_count=0,
        image_coverage_ratio=0.05,
        table_count=0,
        font_count=3,
        font_names=["Arial", "Arial-Bold", "TimesNewRoman"],
        has_form_fields=False,
        is_multi_column=False,
        page_type=PageType.TEXT,
        extraction_quality=_default_extraction_quality(),
    )
    defaults.update(overrides)
    return PageProfile(**defaults)


def _make_document_profile(
    pages: list[PageProfile], **overrides: object
) -> DocumentProfile:
    """Build a DocumentProfile from a list of PageProfile objects."""
    from collections import Counter

    type_dist = dict(Counter(p.page_type.value for p in pages))
    defaults = dict(
        file_path="/tmp/test.pdf",
        file_size_bytes=50000,
        page_count=len(pages),
        content_hash="a" * 64,
        metadata=DocumentMetadata(),
        pages=pages,
        page_type_distribution=type_dist,
        detected_languages=["en"],
        has_toc=False,
        overall_quality=_default_extraction_quality(
            pages_with_text=len(pages),
            total_pages=len(pages),
        ),
        security_warnings=[],
    )
    defaults.update(overrides)
    return DocumentProfile(**defaults)


def _text_native_page(page_number: int = 1, **overrides: object) -> PageProfile:
    """Page with strong text-native signals."""
    return _make_page_profile(
        page_number=page_number,
        text_length=500,
        image_coverage_ratio=0.05,
        font_count=3,
        table_count=0,
        is_multi_column=False,
        has_form_fields=False,
        page_type=PageType.TEXT,
        **overrides,
    )


def _scanned_page(page_number: int = 1, **overrides: object) -> PageProfile:
    """Page with strong scanned signals."""
    return _make_page_profile(
        page_number=page_number,
        text_length=10,
        image_coverage_ratio=0.95,
        font_count=0,
        table_count=0,
        is_multi_column=False,
        has_form_fields=False,
        page_type=PageType.SCANNED,
        **overrides,
    )


def _complex_page(page_number: int = 1, **overrides: object) -> PageProfile:
    """Page with complex signals (tables + multi-column)."""
    return _make_page_profile(
        page_number=page_number,
        text_length=300,
        image_coverage_ratio=0.1,
        font_count=4,
        table_count=3,
        is_multi_column=True,
        has_form_fields=False,
        page_type=PageType.TABLE_HEAVY,
        **overrides,
    )


def _blank_page(page_number: int = 1) -> PageProfile:
    """Blank page."""
    return _make_page_profile(
        page_number=page_number,
        text_length=0,
        word_count=0,
        image_count=0,
        image_coverage_ratio=0.0,
        font_count=0,
        table_count=0,
        is_multi_column=False,
        has_form_fields=False,
        page_type=PageType.BLANK,
    )


# ---------------------------------------------------------------------------
# Signal Evaluation
# ---------------------------------------------------------------------------


class TestPerPageSignalEvaluation:
    """Tests that _evaluate_signals returns correct values for each signal."""

    def test_text_sufficient_true(self, pdf_inspector: PDFInspector) -> None:
        page = _make_page_profile(text_length=500)
        signals = pdf_inspector._evaluate_signals(page)
        assert signals["text_sufficient"] is True

    def test_text_sufficient_false(self, pdf_inspector: PDFInspector) -> None:
        page = _make_page_profile(text_length=100)
        signals = pdf_inspector._evaluate_signals(page)
        assert signals["text_sufficient"] is False

    def test_text_sparse_true(self, pdf_inspector: PDFInspector) -> None:
        page = _make_page_profile(text_length=30)
        signals = pdf_inspector._evaluate_signals(page)
        assert signals["text_sparse"] is True

    def test_text_sparse_false(self, pdf_inspector: PDFInspector) -> None:
        page = _make_page_profile(text_length=100)
        signals = pdf_inspector._evaluate_signals(page)
        assert signals["text_sparse"] is False

    def test_image_coverage_low_true(self, pdf_inspector: PDFInspector) -> None:
        page = _make_page_profile(image_coverage_ratio=0.1)
        signals = pdf_inspector._evaluate_signals(page)
        assert signals["image_coverage_low"] is True

    def test_image_coverage_low_false(self, pdf_inspector: PDFInspector) -> None:
        page = _make_page_profile(image_coverage_ratio=0.5)
        signals = pdf_inspector._evaluate_signals(page)
        assert signals["image_coverage_low"] is False

    def test_image_coverage_high_true(self, pdf_inspector: PDFInspector) -> None:
        page = _make_page_profile(image_coverage_ratio=0.8)
        signals = pdf_inspector._evaluate_signals(page)
        assert signals["image_coverage_high"] is True

    def test_image_coverage_high_false(self, pdf_inspector: PDFInspector) -> None:
        page = _make_page_profile(image_coverage_ratio=0.5)
        signals = pdf_inspector._evaluate_signals(page)
        assert signals["image_coverage_high"] is False

    def test_fonts_present_true(self, pdf_inspector: PDFInspector) -> None:
        page = _make_page_profile(font_count=3)
        signals = pdf_inspector._evaluate_signals(page)
        assert signals["fonts_present"] is True

    def test_fonts_present_false(self, pdf_inspector: PDFInspector) -> None:
        page = _make_page_profile(font_count=0)
        signals = pdf_inspector._evaluate_signals(page)
        assert signals["fonts_present"] is False

    def test_fonts_absent_true(self, pdf_inspector: PDFInspector) -> None:
        page = _make_page_profile(font_count=0)
        signals = pdf_inspector._evaluate_signals(page)
        assert signals["fonts_absent"] is True

    def test_fonts_absent_false(self, pdf_inspector: PDFInspector) -> None:
        page = _make_page_profile(font_count=1)
        signals = pdf_inspector._evaluate_signals(page)
        assert signals["fonts_absent"] is False

    def test_has_tables_true(self, pdf_inspector: PDFInspector) -> None:
        page = _make_page_profile(table_count=2)
        signals = pdf_inspector._evaluate_signals(page)
        assert signals["has_tables"] is True

    def test_has_tables_false(self, pdf_inspector: PDFInspector) -> None:
        page = _make_page_profile(table_count=0)
        signals = pdf_inspector._evaluate_signals(page)
        assert signals["has_tables"] is False

    def test_is_multi_column_true(self, pdf_inspector: PDFInspector) -> None:
        page = _make_page_profile(is_multi_column=True)
        signals = pdf_inspector._evaluate_signals(page)
        assert signals["is_multi_column"] is True

    def test_is_multi_column_false(self, pdf_inspector: PDFInspector) -> None:
        page = _make_page_profile(is_multi_column=False)
        signals = pdf_inspector._evaluate_signals(page)
        assert signals["is_multi_column"] is False

    def test_has_form_fields_true(self, pdf_inspector: PDFInspector) -> None:
        page = _make_page_profile(has_form_fields=True)
        signals = pdf_inspector._evaluate_signals(page)
        assert signals["has_form_fields"] is True

    def test_has_form_fields_false(self, pdf_inspector: PDFInspector) -> None:
        page = _make_page_profile(has_form_fields=False)
        signals = pdf_inspector._evaluate_signals(page)
        assert signals["has_form_fields"] is False

    def test_raw_values_included(self, pdf_inspector: PDFInspector) -> None:
        page = _make_page_profile(
            text_length=350, image_coverage_ratio=0.25, font_count=5, table_count=2
        )
        signals = pdf_inspector._evaluate_signals(page)
        assert signals["text_length"] == 350
        assert signals["image_coverage_ratio"] == 0.25
        assert signals["font_count"] == 5
        assert signals["table_count"] == 2


# ---------------------------------------------------------------------------
# Per-Page Classification
# ---------------------------------------------------------------------------


class TestPerPageClassification:
    """Tests _classify_page for each archetype."""

    def test_text_native_all_3_signals_high_confidence(
        self, pdf_inspector: PDFInspector
    ) -> None:
        page = _text_native_page()
        page_type, pdf_type, confidence, _ = pdf_inspector._classify_page(page)
        assert page_type == PageType.TEXT
        assert pdf_type == PDFType.TEXT_NATIVE
        assert confidence == 0.9

    def test_text_native_2_of_3_signals_medium_confidence(
        self, pdf_inspector: PDFInspector
    ) -> None:
        # text=500 (sufficient), img=0.05 (low), fonts=0 (not present) -> 2 of 3
        page = _make_page_profile(
            text_length=500, image_coverage_ratio=0.05, font_count=0, table_count=0
        )
        page_type, pdf_type, confidence, _ = pdf_inspector._classify_page(page)
        assert page_type == PageType.TEXT
        assert pdf_type == PDFType.TEXT_NATIVE
        assert confidence == 0.7

    def test_scanned_all_3_signals_high_confidence(
        self, pdf_inspector: PDFInspector
    ) -> None:
        page = _scanned_page()
        page_type, pdf_type, confidence, _ = pdf_inspector._classify_page(page)
        assert page_type == PageType.SCANNED
        assert pdf_type == PDFType.SCANNED
        assert confidence == 0.9

    def test_scanned_2_of_3_signals_medium_confidence(
        self, pdf_inspector: PDFInspector
    ) -> None:
        # text=10 (sparse), img=0.95 (high), fonts=1 (present, not absent) -> 2 of 3
        page = _make_page_profile(
            text_length=10,
            image_coverage_ratio=0.95,
            font_count=1,
            table_count=0,
            is_multi_column=False,
            has_form_fields=False,
            page_type=PageType.SCANNED,
        )
        page_type, pdf_type, confidence, _ = pdf_inspector._classify_page(page)
        assert page_type == PageType.SCANNED
        assert pdf_type == PDFType.SCANNED
        assert confidence == 0.7

    def test_complex_tables_only(self, pdf_inspector: PDFInspector) -> None:
        page = _make_page_profile(
            text_length=500, table_count=3, is_multi_column=False, has_form_fields=False
        )
        page_type, pdf_type, confidence, _ = pdf_inspector._classify_page(page)
        assert page_type == PageType.TABLE_HEAVY
        assert pdf_type == PDFType.COMPLEX
        assert confidence == 0.7

    def test_complex_tables_and_multi_column(
        self, pdf_inspector: PDFInspector
    ) -> None:
        page = _make_page_profile(
            text_length=500, table_count=3, is_multi_column=True, has_form_fields=False
        )
        page_type, pdf_type, confidence, _ = pdf_inspector._classify_page(page)
        assert page_type == PageType.TABLE_HEAVY
        assert pdf_type == PDFType.COMPLEX
        assert confidence == 0.9

    def test_complex_form_fields(self, pdf_inspector: PDFInspector) -> None:
        page = _make_page_profile(
            text_length=500, table_count=0, is_multi_column=False, has_form_fields=True
        )
        page_type, pdf_type, confidence, _ = pdf_inspector._classify_page(page)
        assert page_type == PageType.FORM
        assert pdf_type == PDFType.COMPLEX
        assert confidence == 0.7

    def test_complex_all_three_indicators(
        self, pdf_inspector: PDFInspector
    ) -> None:
        page = _make_page_profile(
            table_count=2, is_multi_column=True, has_form_fields=True
        )
        page_type, pdf_type, confidence, _ = pdf_inspector._classify_page(page)
        assert page_type == PageType.FORM
        assert pdf_type == PDFType.COMPLEX
        assert confidence == 0.9

    def test_complex_takes_priority_over_text_native(
        self, pdf_inspector: PDFInspector
    ) -> None:
        # All text-native signals + tables -> complex wins.
        page = _make_page_profile(
            text_length=500,
            image_coverage_ratio=0.05,
            font_count=3,
            table_count=2,
            is_multi_column=False,
            has_form_fields=False,
        )
        page_type, pdf_type, confidence, _ = pdf_inspector._classify_page(page)
        assert page_type == PageType.TABLE_HEAVY
        assert pdf_type == PDFType.COMPLEX

    def test_borderline_returns_none_pdf_type(
        self, pdf_inspector: PDFInspector
    ) -> None:
        # text=100 (not sufficient, not sparse), img=0.4, fonts=1, no tables
        # text_count=1 (low_image=False, has_text=False, has_fonts=True)
        # scan_count=0
        page = _make_page_profile(
            text_length=100,
            image_coverage_ratio=0.4,
            font_count=1,
            table_count=0,
            is_multi_column=False,
            has_form_fields=False,
        )
        page_type, pdf_type, confidence, _ = pdf_inspector._classify_page(page)
        assert page_type == PageType.MIXED
        assert pdf_type is None
        assert confidence == 0.0


# ---------------------------------------------------------------------------
# Special Pages
# ---------------------------------------------------------------------------


class TestSpecialPages:
    """Tests that blank, TOC, and vector-only pages are handled correctly."""

    def test_blank_page_skipped_in_classification(
        self, pdf_inspector: PDFInspector
    ) -> None:
        profile = _make_document_profile([_blank_page(1)])
        result = pdf_inspector.classify(profile)
        assert result.confidence == 0.0
        assert "blank" in result.reasoning.lower() or "skippable" in result.reasoning.lower()

    def test_blank_page_among_text_pages(
        self, pdf_inspector: PDFInspector
    ) -> None:
        pages = [_text_native_page(1), _text_native_page(2), _blank_page(3)]
        profile = _make_document_profile(pages)
        result = pdf_inspector.classify(profile)
        assert result.pdf_type == PDFType.TEXT_NATIVE

    def test_toc_page_skipped(self, pdf_inspector: PDFInspector) -> None:
        toc = _make_page_profile(page_number=1, page_type=PageType.TOC)
        pages = [toc, _text_native_page(2), _text_native_page(3)]
        profile = _make_document_profile(pages)
        result = pdf_inspector.classify(profile)
        assert result.pdf_type == PDFType.TEXT_NATIVE

    def test_vector_only_page_skipped(self, pdf_inspector: PDFInspector) -> None:
        vec = _make_page_profile(page_number=1, page_type=PageType.VECTOR_ONLY)
        pages = [vec, _text_native_page(2), _text_native_page(3)]
        profile = _make_document_profile(pages)
        result = pdf_inspector.classify(profile)
        assert result.pdf_type == PDFType.TEXT_NATIVE

    def test_all_pages_skippable(self, pdf_inspector: PDFInspector) -> None:
        pages = [_blank_page(1), _blank_page(2), _blank_page(3)]
        profile = _make_document_profile(pages)
        result = pdf_inspector.classify(profile)
        assert result.confidence == 0.0

    def test_per_page_types_include_special_pages(
        self, pdf_inspector: PDFInspector
    ) -> None:
        pages = [_blank_page(1), _text_native_page(2)]
        profile = _make_document_profile(pages)
        result = pdf_inspector.classify(profile)
        assert 1 in result.per_page_types
        assert result.per_page_types[1] == PageType.BLANK
        assert 2 in result.per_page_types
        assert result.per_page_types[2] == PageType.TEXT


# ---------------------------------------------------------------------------
# Document-Level Agreement
# ---------------------------------------------------------------------------


class TestDocumentLevelAgreement:
    """Tests when all non-skippable pages agree."""

    def test_all_pages_text_native(self, pdf_inspector: PDFInspector) -> None:
        pages = [_text_native_page(i) for i in range(1, 4)]
        profile = _make_document_profile(pages)
        result = pdf_inspector.classify(profile)
        assert result.pdf_type == PDFType.TEXT_NATIVE

    def test_all_pages_scanned(self, pdf_inspector: PDFInspector) -> None:
        pages = [_scanned_page(i) for i in range(1, 4)]
        profile = _make_document_profile(pages)
        result = pdf_inspector.classify(profile)
        assert result.pdf_type == PDFType.SCANNED

    def test_all_pages_complex(self, pdf_inspector: PDFInspector) -> None:
        pages = [_complex_page(i) for i in range(1, 4)]
        profile = _make_document_profile(pages)
        result = pdf_inspector.classify(profile)
        assert result.pdf_type == PDFType.COMPLEX

    def test_single_page_text_native(self, pdf_inspector: PDFInspector) -> None:
        profile = _make_document_profile([_text_native_page(1)])
        result = pdf_inspector.classify(profile)
        assert result.pdf_type == PDFType.TEXT_NATIVE

    def test_confidence_is_minimum_across_pages(
        self, pdf_inspector: PDFInspector
    ) -> None:
        # One high confidence (3/3 signals), one medium (2/3 signals).
        high = _text_native_page(1)  # 0.9
        medium = _make_page_profile(
            page_number=2,
            text_length=500,
            image_coverage_ratio=0.05,
            font_count=0,  # missing one signal -> 0.7
            table_count=0,
            is_multi_column=False,
            has_form_fields=False,
        )
        profile = _make_document_profile([high, medium])
        result = pdf_inspector.classify(profile)
        assert result.pdf_type == PDFType.TEXT_NATIVE
        assert result.confidence == 0.7


# ---------------------------------------------------------------------------
# Document-Level Disagreement
# ---------------------------------------------------------------------------


class TestDocumentLevelDisagreement:
    """Tests when non-skippable pages disagree (should become COMPLEX)."""

    def test_text_and_scanned_pages_produce_complex(
        self, pdf_inspector: PDFInspector
    ) -> None:
        pages = [_text_native_page(1), _text_native_page(2), _scanned_page(3)]
        profile = _make_document_profile(pages)
        result = pdf_inspector.classify(profile)
        assert result.pdf_type == PDFType.COMPLEX

    def test_text_and_complex_pages_produce_complex(
        self, pdf_inspector: PDFInspector
    ) -> None:
        pages = [_text_native_page(1), _text_native_page(2), _complex_page(3)]
        profile = _make_document_profile(pages)
        result = pdf_inspector.classify(profile)
        assert result.pdf_type == PDFType.COMPLEX

    def test_scanned_and_complex_pages_produce_complex(
        self, pdf_inspector: PDFInspector
    ) -> None:
        pages = [_scanned_page(1), _scanned_page(2), _complex_page(3)]
        profile = _make_document_profile(pages)
        result = pdf_inspector.classify(profile)
        assert result.pdf_type == PDFType.COMPLEX

    def test_disagreement_confidence_is_high(
        self, pdf_inspector: PDFInspector
    ) -> None:
        pages = [_text_native_page(1), _scanned_page(2)]
        profile = _make_document_profile(pages)
        result = pdf_inspector.classify(profile)
        assert result.confidence == 0.9

    def test_disagreement_reasoning_describes_mix(
        self, pdf_inspector: PDFInspector
    ) -> None:
        pages = [_text_native_page(1), _scanned_page(2)]
        profile = _make_document_profile(pages)
        result = pdf_inspector.classify(profile)
        assert "disagree" in result.reasoning.lower()

    def test_per_page_types_populated_on_disagreement(
        self, pdf_inspector: PDFInspector
    ) -> None:
        pages = [_text_native_page(1), _scanned_page(2)]
        profile = _make_document_profile(pages)
        result = pdf_inspector.classify(profile)
        assert 1 in result.per_page_types
        assert 2 in result.per_page_types


# ---------------------------------------------------------------------------
# Inconclusive Escalation
# ---------------------------------------------------------------------------


class TestInconclusiveEscalation:
    """Tests that borderline pages trigger inconclusive results for Tier 2."""

    def test_borderline_page_produces_zero_confidence(
        self, pdf_inspector: PDFInspector
    ) -> None:
        # Borderline page: ambiguous signals
        page = _make_page_profile(
            text_length=100,
            image_coverage_ratio=0.4,
            font_count=1,
            table_count=0,
            is_multi_column=False,
            has_form_fields=False,
        )
        profile = _make_document_profile([page])
        result = pdf_inspector.classify(profile)
        assert result.confidence == 0.0

    def test_borderline_among_clear_pages(
        self, pdf_inspector: PDFInspector
    ) -> None:
        borderline = _make_page_profile(
            page_number=3,
            text_length=100,
            image_coverage_ratio=0.4,
            font_count=1,
            table_count=0,
            is_multi_column=False,
            has_form_fields=False,
        )
        pages = [_text_native_page(1), _text_native_page(2), borderline]
        profile = _make_document_profile(pages)
        result = pdf_inspector.classify(profile)
        assert result.confidence == 0.0

    def test_inconclusive_reasoning_mentions_page(
        self, pdf_inspector: PDFInspector
    ) -> None:
        borderline = _make_page_profile(
            page_number=5,
            text_length=100,
            image_coverage_ratio=0.4,
            font_count=1,
            table_count=0,
            is_multi_column=False,
            has_form_fields=False,
        )
        profile = _make_document_profile([borderline])
        result = pdf_inspector.classify(profile)
        assert "page 5" in result.reasoning.lower()


# ---------------------------------------------------------------------------
# Empty Profile
# ---------------------------------------------------------------------------


class TestEmptyProfile:
    """Tests edge cases with no pages."""

    def test_no_pages_returns_inconclusive(
        self, pdf_inspector: PDFInspector
    ) -> None:
        profile = _make_document_profile([], page_count=0)
        result = pdf_inspector.classify(profile)
        assert result.confidence == 0.0

    def test_empty_pages_list_returns_inconclusive(
        self, pdf_inspector: PDFInspector
    ) -> None:
        # page_count says 5 but pages list is empty.
        profile = _make_document_profile([], page_count=5)
        result = pdf_inspector.classify(profile)
        assert result.confidence == 0.0

    def test_inconclusive_reasoning_mentions_no_pages(
        self, pdf_inspector: PDFInspector
    ) -> None:
        profile = _make_document_profile([], page_count=0)
        result = pdf_inspector.classify(profile)
        assert "no pages" in result.reasoning.lower()


# ---------------------------------------------------------------------------
# Classification Result Fields
# ---------------------------------------------------------------------------


class TestClassificationResultFields:
    """Validates all output fields are correctly populated."""

    def test_tier_used_is_always_rule_based(
        self, pdf_inspector: PDFInspector
    ) -> None:
        profile = _make_document_profile([_text_native_page(1)])
        result = pdf_inspector.classify(profile)
        assert result.tier_used == ClassificationTier.RULE_BASED

    def test_signals_dict_populated(self, pdf_inspector: PDFInspector) -> None:
        profile = _make_document_profile([_text_native_page(1)])
        result = pdf_inspector.classify(profile)
        assert result.signals is not None
        assert "per_page" in result.signals
        assert 1 in result.signals["per_page"]

    def test_per_page_types_always_populated(
        self, pdf_inspector: PDFInspector
    ) -> None:
        pages = [_text_native_page(1), _text_native_page(2)]
        profile = _make_document_profile(pages)
        result = pdf_inspector.classify(profile)
        assert 1 in result.per_page_types
        assert 2 in result.per_page_types

    def test_reasoning_is_non_empty_string(
        self, pdf_inspector: PDFInspector
    ) -> None:
        profile = _make_document_profile([_text_native_page(1)])
        result = pdf_inspector.classify(profile)
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0

    def test_pdf_type_uses_enum_values(
        self, pdf_inspector: PDFInspector
    ) -> None:
        profile = _make_document_profile([_text_native_page(1)])
        result = pdf_inspector.classify(profile)
        assert result.pdf_type.value in ("text_native", "scanned", "complex")

    def test_degraded_is_always_false(
        self, pdf_inspector: PDFInspector
    ) -> None:
        profile = _make_document_profile([_text_native_page(1)])
        result = pdf_inspector.classify(profile)
        assert result.degraded is False


# ---------------------------------------------------------------------------
# Custom Config
# ---------------------------------------------------------------------------


class TestCustomConfig:
    """Tests that custom thresholds are respected."""

    def test_custom_min_chars_per_page(self) -> None:
        config = PDFProcessorConfig(min_chars_per_page=500)
        inspector = PDFInspector(config)
        # Page with 300 chars, below custom threshold.
        page = _make_page_profile(
            text_length=300, image_coverage_ratio=0.05, font_count=3, table_count=0
        )
        _, pdf_type, confidence, _ = inspector._classify_page(page)
        # Only 2 of 3 text signals match (image low + fonts present, but not text sufficient)
        assert pdf_type == PDFType.TEXT_NATIVE
        assert confidence == 0.7  # medium, not high

    def test_custom_max_image_coverage_for_text(self) -> None:
        config = PDFProcessorConfig(max_image_coverage_for_text=0.1)
        inspector = PDFInspector(config)
        # Page with 0.2 image coverage, above custom threshold.
        page = _make_page_profile(
            text_length=500, image_coverage_ratio=0.2, font_count=3, table_count=0
        )
        _, pdf_type, confidence, _ = inspector._classify_page(page)
        # Only 2 of 3 text signals match (text sufficient + fonts present, but image not low)
        assert pdf_type == PDFType.TEXT_NATIVE
        assert confidence == 0.7

    def test_custom_min_font_count(self) -> None:
        config = PDFProcessorConfig(min_font_count_for_digital=3)
        inspector = PDFInspector(config)
        # Page with 2 fonts, below custom threshold.
        page = _make_page_profile(
            text_length=500, image_coverage_ratio=0.05, font_count=2, table_count=0
        )
        _, pdf_type, confidence, _ = inspector._classify_page(page)
        # Only 2 of 3 text signals match (text sufficient + image low, but not enough fonts)
        assert pdf_type == PDFType.TEXT_NATIVE
        assert confidence == 0.7

    def test_custom_min_table_count_for_complex(self) -> None:
        config = PDFProcessorConfig(min_table_count_for_complex=3)
        inspector = PDFInspector(config)
        # Page with 2 tables, below custom threshold.
        page = _make_page_profile(
            text_length=500,
            image_coverage_ratio=0.05,
            font_count=3,
            table_count=2,
            is_multi_column=False,
            has_form_fields=False,
        )
        _, pdf_type, _, _ = inspector._classify_page(page)
        # Tables below threshold, so not complex -- should be text-native.
        assert pdf_type == PDFType.TEXT_NATIVE


# ---------------------------------------------------------------------------
# Boundary Values
# ---------------------------------------------------------------------------


class TestBoundaryValues:
    """Tests exact threshold boundaries."""

    def test_text_length_exactly_at_min_chars(
        self, pdf_inspector: PDFInspector
    ) -> None:
        page = _make_page_profile(text_length=200)
        signals = pdf_inspector._evaluate_signals(page)
        assert signals["text_sufficient"] is True

    def test_text_length_one_below_min_chars(
        self, pdf_inspector: PDFInspector
    ) -> None:
        page = _make_page_profile(text_length=199)
        signals = pdf_inspector._evaluate_signals(page)
        assert signals["text_sufficient"] is False

    def test_text_length_exactly_at_scanned_max(
        self, pdf_inspector: PDFInspector
    ) -> None:
        # < 50 is the threshold, so 50 is NOT sparse.
        page = _make_page_profile(text_length=50)
        signals = pdf_inspector._evaluate_signals(page)
        assert signals["text_sparse"] is False

    def test_text_length_one_below_scanned_max(
        self, pdf_inspector: PDFInspector
    ) -> None:
        page = _make_page_profile(text_length=49)
        signals = pdf_inspector._evaluate_signals(page)
        assert signals["text_sparse"] is True

    def test_image_coverage_exactly_at_text_max(
        self, pdf_inspector: PDFInspector
    ) -> None:
        # < 0.3 is the threshold, so 0.3 is NOT low.
        page = _make_page_profile(image_coverage_ratio=0.3)
        signals = pdf_inspector._evaluate_signals(page)
        assert signals["image_coverage_low"] is False

    def test_image_coverage_exactly_at_scanned_min(
        self, pdf_inspector: PDFInspector
    ) -> None:
        # > 0.7 is the threshold, so 0.7 is NOT high.
        page = _make_page_profile(image_coverage_ratio=0.7)
        signals = pdf_inspector._evaluate_signals(page)
        assert signals["image_coverage_high"] is False

    def test_font_count_exactly_at_threshold(
        self, pdf_inspector: PDFInspector
    ) -> None:
        page = _make_page_profile(font_count=1)
        signals = pdf_inspector._evaluate_signals(page)
        assert signals["fonts_present"] is True

    def test_table_count_exactly_at_threshold(
        self, pdf_inspector: PDFInspector
    ) -> None:
        page = _make_page_profile(table_count=1)
        signals = pdf_inspector._evaluate_signals(page)
        assert signals["has_tables"] is True


# ---------------------------------------------------------------------------
# Confidence Calculation
# ---------------------------------------------------------------------------


class TestConfidenceCalculation:
    """Tests confidence model validation."""

    def test_high_confidence_text_native(
        self, pdf_inspector: PDFInspector
    ) -> None:
        page = _text_native_page()
        _, _, confidence, _ = pdf_inspector._classify_page(page)
        assert confidence == 0.9

    def test_medium_confidence_text_native(
        self, pdf_inspector: PDFInspector
    ) -> None:
        page = _make_page_profile(
            text_length=500, image_coverage_ratio=0.05, font_count=0, table_count=0
        )
        _, _, confidence, _ = pdf_inspector._classify_page(page)
        assert confidence == 0.7

    def test_high_confidence_scanned(
        self, pdf_inspector: PDFInspector
    ) -> None:
        page = _scanned_page()
        _, _, confidence, _ = pdf_inspector._classify_page(page)
        assert confidence == 0.9

    def test_medium_confidence_scanned(
        self, pdf_inspector: PDFInspector
    ) -> None:
        # 2 of 3 scanned signals
        page = _make_page_profile(
            text_length=10,
            image_coverage_ratio=0.95,
            font_count=1,
            table_count=0,
            is_multi_column=False,
            has_form_fields=False,
        )
        _, _, confidence, _ = pdf_inspector._classify_page(page)
        assert confidence == 0.7

    def test_high_confidence_complex(
        self, pdf_inspector: PDFInspector
    ) -> None:
        # 2+ complex indicators
        page = _make_page_profile(table_count=2, is_multi_column=True)
        _, _, confidence, _ = pdf_inspector._classify_page(page)
        assert confidence == 0.9

    def test_medium_confidence_complex(
        self, pdf_inspector: PDFInspector
    ) -> None:
        # 1 complex indicator
        page = _make_page_profile(
            table_count=1, is_multi_column=False, has_form_fields=False
        )
        _, _, confidence, _ = pdf_inspector._classify_page(page)
        assert confidence == 0.7

    def test_document_confidence_is_page_minimum(
        self, pdf_inspector: PDFInspector
    ) -> None:
        high = _text_native_page(1)  # 0.9
        medium = _make_page_profile(
            page_number=2,
            text_length=500,
            image_coverage_ratio=0.05,
            font_count=0,
            table_count=0,
            is_multi_column=False,
            has_form_fields=False,
        )
        profile = _make_document_profile([high, medium])
        result = pdf_inspector.classify(profile)
        assert result.confidence == 0.7

    def test_borderline_confidence_is_zero(
        self, pdf_inspector: PDFInspector
    ) -> None:
        page = _make_page_profile(
            text_length=100,
            image_coverage_ratio=0.4,
            font_count=1,
            table_count=0,
            is_multi_column=False,
            has_form_fields=False,
        )
        _, _, confidence, _ = pdf_inspector._classify_page(page)
        assert confidence == 0.0
