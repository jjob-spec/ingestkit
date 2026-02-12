"""Extraction quality scoring for PDF text extraction.

Assesses text extraction quality per-page and per-document using the
composite formula from SPEC §8.3. Low-quality extraction triggers
automatic OCR fallback when enabled.
"""

from __future__ import annotations

import string

from ingestkit_pdf.config import PDFProcessorConfig
from ingestkit_pdf.models import ExtractionQuality, ExtractionQualityGrade

_PRINTABLE = set(string.printable)


class QualityAssessor:
    """Assess extraction quality and gate OCR fallback decisions."""

    def __init__(self, config: PDFProcessorConfig) -> None:
        self.config = config

    def assess_page(self, page_text: str, page_number: int) -> ExtractionQuality:
        """Assess extraction quality for a single page of text.

        Args:
            page_text: Raw text extracted from the page.
            page_number: 0-indexed page number (for context only).

        Returns:
            An ExtractionQuality with metrics for this single page.
        """
        total_chars = len(page_text)
        if total_chars == 0:
            return ExtractionQuality(
                printable_ratio=0.0,
                avg_words_per_page=0.0,
                pages_with_text=0,
                total_pages=1,
                extraction_method="native",
            )

        printable_count = sum(1 for c in page_text if c in _PRINTABLE)
        printable_ratio = printable_count / total_chars

        words = page_text.split()
        word_count = len(words)

        has_text = word_count >= self.config.quality_min_words_per_page
        pages_with_text = 1 if has_text else 0

        return ExtractionQuality(
            printable_ratio=printable_ratio,
            avg_words_per_page=float(word_count),
            pages_with_text=pages_with_text,
            total_pages=1,
            extraction_method="native",
        )

    def assess_document(
        self, page_qualities: list[ExtractionQuality]
    ) -> ExtractionQuality:
        """Aggregate per-page quality assessments into a document-level score.

        Args:
            page_qualities: Quality assessments for each page.

        Returns:
            A document-level ExtractionQuality.
        """
        total_pages = len(page_qualities)
        if total_pages == 0:
            return ExtractionQuality(
                printable_ratio=0.0,
                avg_words_per_page=0.0,
                pages_with_text=0,
                total_pages=0,
                extraction_method="native",
            )

        total_printable_ratio = sum(q.printable_ratio for q in page_qualities)
        avg_printable_ratio = total_printable_ratio / total_pages

        total_words = sum(q.avg_words_per_page for q in page_qualities)
        pages_with_text = sum(q.pages_with_text for q in page_qualities)
        avg_words = total_words / max(pages_with_text, 1)

        methods = {q.extraction_method for q in page_qualities}
        if len(methods) == 1:
            method = methods.pop()
        elif "ocr_fallback" in methods:
            method = "ocr_fallback"
        else:
            method = "native"

        return ExtractionQuality(
            printable_ratio=avg_printable_ratio,
            avg_words_per_page=avg_words,
            pages_with_text=pages_with_text,
            total_pages=total_pages,
            extraction_method=method,
        )

    def needs_ocr_fallback(self, quality: ExtractionQuality) -> bool:
        """Determine if OCR fallback should be triggered for the given quality.

        Returns True if auto_ocr_fallback is enabled and the quality grade
        is LOW.
        """
        if not self.config.auto_ocr_fallback:
            return False
        return quality.grade == ExtractionQualityGrade.LOW
