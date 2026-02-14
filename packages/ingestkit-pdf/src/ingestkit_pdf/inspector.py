"""Tier 1 rule-based structural inspector for PDF files.

Evaluates seven signals per page and applies threshold-based decision
logic to classify files as text-native (Type A), scanned (Type B),
or complex (Type C) -- without any LLM call.

Signal evaluation and document-level aggregation logic are defined in
SPEC.md section 9.  All configurable thresholds live in
:class:`~ingestkit_pdf.config.PDFProcessorConfig`.
"""

from __future__ import annotations

import logging
from typing import Any

from ingestkit_pdf.config import PDFProcessorConfig
from ingestkit_pdf.models import (
    ClassificationResult,
    ClassificationTier,
    DocumentProfile,
    PageProfile,
    PageType,
    PDFType,
)

logger = logging.getLogger("ingestkit_pdf")

# SPEC §9.2: scanned page thresholds (not configurable)
_MAX_CHARS_FOR_SCANNED: int = 50
_MIN_IMAGE_COVERAGE_FOR_SCANNED: float = 0.7

# Page types that do not contribute to document-level type agreement.
_SKIPPABLE_PAGE_TYPES: frozenset[PageType] = frozenset(
    {
        PageType.BLANK,
        PageType.TOC,
        PageType.VECTOR_ONLY,
    }
)

# Warning codes for skipped page types.
_SKIP_WARNING_CODES: dict[PageType, str] = {
    PageType.BLANK: "W_PAGE_SKIPPED_BLANK",
    PageType.TOC: "W_PAGE_SKIPPED_TOC",
    PageType.VECTOR_ONLY: "W_PAGE_SKIPPED_VECTOR_ONLY",
}


# ---------------------------------------------------------------------------
# Inspector
# ---------------------------------------------------------------------------


class PDFInspector:
    """Tier 1 rule-based structural inspector for PDF files.

    Evaluates 7 signals per page and uses threshold-based decision
    logic to classify files without any LLM call.
    """

    def __init__(self, config: PDFProcessorConfig) -> None:
        self._config = config

    # -- public API ----------------------------------------------------------

    def classify(self, profile: DocumentProfile) -> ClassificationResult:
        """Classify a PDF based on structural signals from its profile.

        Args:
            profile: The :class:`DocumentProfile` produced by the parser.

        Returns:
            A :class:`ClassificationResult` with PDF type, confidence,
            tier information, and signal breakdown.
        """
        # Edge case: no pages at all.
        if profile.page_count == 0 or len(profile.pages) == 0:
            logger.info(
                "No pages found in %s -- returning inconclusive.",
                profile.file_path,
            )
            return ClassificationResult(
                pdf_type=PDFType.COMPLEX,
                confidence=0.0,
                tier_used=ClassificationTier.RULE_BASED,
                reasoning="Inconclusive: document contains no pages to classify.",
                per_page_types={},
                signals={"per_page": {}},
            )

        # Classify each page independently.
        page_results: list[
            tuple[int, PageType, PDFType | None, float, dict[str, Any]]
        ] = []
        for page in profile.pages:
            page_type, pdf_type, confidence, signals = self._classify_page(page)
            page_results.append(
                (page.page_number, page_type, pdf_type, confidence, signals)
            )

        # Build aggregated signals dict.
        agg_signals: dict[str, Any] = {
            "per_page": {
                page_num: {
                    "page_type": page_type.value,
                    "pdf_type": pdf_type.value if pdf_type else None,
                    "confidence": conf,
                    "signals": sigs,
                }
                for page_num, page_type, pdf_type, conf, sigs in page_results
            }
        }

        # Build per_page_types (always populated).
        per_page_types: dict[int, PageType] = {
            page_num: page_type
            for page_num, page_type, _, _, _ in page_results
        }

        # Filter out skippable pages for document-level aggregation.
        non_skippable = [
            (page_num, page_type, pdf_type, conf, sigs)
            for page_num, page_type, pdf_type, conf, sigs in page_results
            if page_type not in _SKIPPABLE_PAGE_TYPES
        ]

        # If all pages are skippable, return inconclusive.
        if not non_skippable:
            logger.info(
                "All pages in %s are skippable -- returning inconclusive.",
                profile.file_path,
            )
            return ClassificationResult(
                pdf_type=PDFType.COMPLEX,
                confidence=0.0,
                tier_used=ClassificationTier.RULE_BASED,
                reasoning=(
                    "Inconclusive: all pages are blank, TOC, or vector-only."
                ),
                per_page_types=per_page_types,
                signals=agg_signals,
            )

        # Check for borderline pages (pdf_type is None).
        borderline = [
            page_num
            for page_num, _, pdf_type, _, _ in non_skippable
            if pdf_type is None
        ]
        if borderline:
            first = borderline[0]
            reasoning = (
                f"Inconclusive: page {first} could not be classified "
                "with sufficient confidence."
            )
            logger.info(
                "Inconclusive classification for %s -- %s",
                profile.file_path,
                reasoning,
            )
            return ClassificationResult(
                pdf_type=PDFType.COMPLEX,
                confidence=0.0,
                tier_used=ClassificationTier.RULE_BASED,
                reasoning=reasoning,
                per_page_types=per_page_types,
                signals=agg_signals,
            )

        # Check document-level agreement (Signal 7).
        distinct_types: set[PDFType] = {
            pdf_type  # type: ignore[misc]
            for _, _, pdf_type, _, _ in non_skippable
        }

        if len(distinct_types) == 1:
            # All non-skippable pages agree.
            agreed_type: PDFType = next(iter(distinct_types))
            min_confidence = min(conf for _, _, _, conf, _ in non_skippable)
            n = len(non_skippable)
            reasoning = (
                f"All {n} classifiable page(s) classified as {agreed_type.value} "
                f"with {min_confidence} confidence by Tier 1 rule-based inspector."
            )
            logger.info(
                "Classified %s as %s (confidence=%s, tier=rule_based).",
                profile.file_path,
                agreed_type.value,
                min_confidence,
            )
            return ClassificationResult(
                pdf_type=agreed_type,
                confidence=min_confidence,
                tier_used=ClassificationTier.RULE_BASED,
                reasoning=reasoning,
                per_page_types=per_page_types,
                signals=agg_signals,
            )

        # Pages disagree -- classify as complex.
        type_counts: dict[str, int] = {}
        for _, _, pdf_type, _, _ in non_skippable:
            val = pdf_type.value  # type: ignore[union-attr]
            type_counts[val] = type_counts.get(val, 0) + 1

        reasoning = (
            f"Pages disagree on type: {type_counts}. "
            "Classified as complex."
        )
        logger.info(
            "Classified %s as complex due to page disagreement (tier=rule_based).",
            profile.file_path,
        )
        return ClassificationResult(
            pdf_type=PDFType.COMPLEX,
            confidence=0.9,
            tier_used=ClassificationTier.RULE_BASED,
            reasoning=reasoning,
            per_page_types=per_page_types,
            signals=agg_signals,
        )

    # -- internal helpers ----------------------------------------------------

    def _classify_page(
        self, page: PageProfile
    ) -> tuple[PageType, PDFType | None, float, dict[str, Any]]:
        """Classify a single page.

        Returns:
            A tuple of ``(page_type, pdf_type_or_None, confidence, signals)``.
            ``pdf_type`` is ``None`` for skippable or borderline pages.
        """
        signals = self._evaluate_signals(page)

        # Check for special/skippable page types first.
        if page.page_type in _SKIPPABLE_PAGE_TYPES:
            warn_code = _SKIP_WARNING_CODES.get(page.page_type, "")
            logger.debug(
                "Page %d is %s (%s) -- skipping for classification.",
                page.page_number,
                page.page_type.value,
                warn_code,
            )
            return page.page_type, None, 1.0, signals

        # Check for complex indicators (signals 4, 5, 6).
        has_tables = page.table_count >= self._config.min_table_count_for_complex
        is_multi_column = page.is_multi_column
        has_form_fields = page.has_form_fields
        complex_count = sum([has_tables, is_multi_column, has_form_fields])

        if complex_count >= 1:
            # Determine specific PageType.
            if has_form_fields:
                page_type = PageType.FORM
            elif has_tables:
                page_type = PageType.TABLE_HEAVY
            else:
                page_type = PageType.MIXED

            confidence = 0.9 if complex_count >= 2 else 0.7

            logger.debug(
                "Page %d classified as %s (complex_count=%d, confidence=%s).",
                page.page_number,
                page_type.value,
                complex_count,
                confidence,
            )
            return page_type, PDFType.COMPLEX, confidence, signals

        # Check for text-native indicators (signals 1, 2, 3).
        has_text = page.text_length >= self._config.min_chars_per_page
        low_image = page.image_coverage_ratio < self._config.max_image_coverage_for_text
        has_fonts = page.font_count >= self._config.min_font_count_for_digital
        text_count = sum([has_text, low_image, has_fonts])

        # Check for scanned indicators (signals 1, 2, 3 inverted).
        few_chars = page.text_length < _MAX_CHARS_FOR_SCANNED
        high_image = page.image_coverage_ratio > _MIN_IMAGE_COVERAGE_FOR_SCANNED
        no_fonts = page.font_count == 0
        scan_count = sum([few_chars, high_image, no_fonts])

        if text_count == 3:
            return PageType.TEXT, PDFType.TEXT_NATIVE, 0.9, signals
        if text_count == 2:
            return PageType.TEXT, PDFType.TEXT_NATIVE, 0.7, signals

        if scan_count == 3:
            return PageType.SCANNED, PDFType.SCANNED, 0.9, signals
        if scan_count == 2:
            return PageType.SCANNED, PDFType.SCANNED, 0.7, signals

        # Borderline -- triggers Tier 2 escalation.
        logger.debug(
            "Page %d is borderline (text_count=%d, scan_count=%d).",
            page.page_number,
            text_count,
            scan_count,
        )
        return PageType.MIXED, None, 0.0, signals

    def _evaluate_signals(self, page: PageProfile) -> dict[str, Any]:
        """Evaluate the 6 per-page signals.

        Signal 7 (page consistency) is a document-level signal handled
        in :meth:`classify`.
        """
        cfg = self._config
        return {
            "text_length": page.text_length,
            "text_sufficient": page.text_length >= cfg.min_chars_per_page,
            "text_sparse": page.text_length < _MAX_CHARS_FOR_SCANNED,
            "image_coverage_ratio": page.image_coverage_ratio,
            "image_coverage_low": (
                page.image_coverage_ratio < cfg.max_image_coverage_for_text
            ),
            "image_coverage_high": (
                page.image_coverage_ratio > _MIN_IMAGE_COVERAGE_FOR_SCANNED
            ),
            "font_count": page.font_count,
            "fonts_present": page.font_count >= cfg.min_font_count_for_digital,
            "fonts_absent": page.font_count == 0,
            "table_count": page.table_count,
            "has_tables": page.table_count >= cfg.min_table_count_for_complex,
            "is_multi_column": page.is_multi_column,
            "has_form_fields": page.has_form_fields,
        }
