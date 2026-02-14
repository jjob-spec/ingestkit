"""Tests for ingestkit_pdf.utils.header_footer — HeaderFooterDetector."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ingestkit_pdf.config import PDFProcessorConfig
from ingestkit_pdf.utils.header_footer import HeaderFooterDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PAGE_HEIGHT = 792.0  # Standard US-Letter height in points


def _make_mock_page(
    header_text: str = "",
    body_text: str = "Body content here.",
    footer_text: str = "",
    page_height: float = PAGE_HEIGHT,
    *,
    extra_blocks: list[tuple] | None = None,
) -> MagicMock:
    """Build a mock ``fitz.Page`` with blocks in header/body/footer zones.

    Block tuple format: ``(x0, y0, x1, y1, text, block_no, type)``
    """
    blocks: list[tuple] = []
    block_no = 0

    if header_text:
        # y0=10 puts it well within the top 10% zone (79.2 for 792pt page)
        blocks.append((72, 10, 500, 50, header_text, block_no, 0))
        block_no += 1

    if body_text:
        blocks.append((72, 200, 500, 600, body_text, block_no, 0))
        block_no += 1

    if footer_text:
        # y1=780 puts it within the bottom 10% zone (>712.8 for 792pt page)
        blocks.append((72, 750, 500, 780, footer_text, block_no, 0))
        block_no += 1

    if extra_blocks:
        blocks.extend(extra_blocks)

    page = MagicMock()
    page.rect.height = page_height
    page.get_text.return_value = blocks
    return page


def _make_mock_doc(pages: list[MagicMock]) -> MagicMock:
    """Build a mock ``fitz.Document`` wrapping the given pages."""
    doc = MagicMock()
    doc.__len__ = MagicMock(return_value=len(pages))
    doc.__getitem__ = MagicMock(side_effect=lambda i: pages[i])
    return doc


# ---------------------------------------------------------------------------
# TestSelectSampleIndices
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSelectSampleIndices:
    def _make_detector(self, sample_pages: int = 5) -> HeaderFooterDetector:
        config = PDFProcessorConfig(header_footer_sample_pages=sample_pages)
        return HeaderFooterDetector(config)

    def test_fewer_pages_than_sample(self):
        det = self._make_detector(sample_pages=5)
        assert det._select_sample_indices(3) == [0, 1, 2]

    def test_exact_pages(self):
        det = self._make_detector(sample_pages=5)
        assert det._select_sample_indices(5) == [0, 1, 2, 3, 4]

    def test_more_pages_evenly_distributed(self):
        det = self._make_detector(sample_pages=5)
        result = det._select_sample_indices(20)
        assert result == [0, 5, 10, 14, 19]

    def test_single_page(self):
        det = self._make_detector(sample_pages=5)
        assert det._select_sample_indices(1) == [0]


# ---------------------------------------------------------------------------
# TestExtractZoneText
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractZoneText:
    def _make_detector(self) -> HeaderFooterDetector:
        return HeaderFooterDetector(PDFProcessorConfig())

    def test_header_zone_extraction(self):
        page = _make_mock_page(header_text="Company Inc.", body_text="Body")
        det = self._make_detector()
        result = det._extract_zone_text(page, "header")
        assert result == ["Company Inc."]

    def test_footer_zone_extraction(self):
        page = _make_mock_page(footer_text="Page 1 of 10", body_text="Body")
        det = self._make_detector()
        result = det._extract_zone_text(page, "footer")
        assert result == ["Page 1 of 10"]

    def test_image_blocks_ignored(self):
        """Blocks with type=1 (image) should be excluded."""
        # Image block in header zone: type=1
        image_block = (72, 10, 500, 50, "image-data", 0, 1)
        page = _make_mock_page(
            body_text="Body", extra_blocks=[image_block]
        )
        det = self._make_detector()
        result = det._extract_zone_text(page, "header")
        assert result == []

    def test_empty_text_excluded(self):
        """Blocks with empty/whitespace text should be excluded."""
        empty_block = (72, 10, 500, 50, "   ", 0, 0)
        page = _make_mock_page(body_text="Body", extra_blocks=[empty_block])
        det = self._make_detector()
        result = det._extract_zone_text(page, "header")
        assert result == []


# ---------------------------------------------------------------------------
# TestDetect
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDetect:
    def _make_detector(self, **kwargs) -> HeaderFooterDetector:
        return HeaderFooterDetector(PDFProcessorConfig(**kwargs))

    def test_single_page_returns_empty(self):
        doc = _make_mock_doc([_make_mock_page(header_text="Header")])
        det = self._make_detector()
        assert det.detect(doc) == ([], [])

    def test_consistent_header_detected(self):
        pages = [
            _make_mock_page(header_text="Company Name") for _ in range(5)
        ]
        doc = _make_mock_doc(pages)
        det = self._make_detector()
        headers, footers = det.detect(doc)
        assert "Company Name" in headers
        assert footers == []

    def test_consistent_footer_detected(self):
        pages = [
            _make_mock_page(footer_text="Page 1 of 10") for _ in range(5)
        ]
        doc = _make_mock_doc(pages)
        det = self._make_detector()
        headers, footers = det.detect(doc)
        assert headers == []
        # All pages have identical text so similarity=1.0 -> detected
        assert len(footers) == 1

    def test_header_missing_one_page_still_detected(self):
        """5 pages, 4 with same header -> detected (>= sampled_pages - 1)."""
        pages = [
            _make_mock_page(header_text="Acme Corp") for _ in range(4)
        ]
        pages.append(_make_mock_page(header_text=""))  # no header on page 5
        doc = _make_mock_doc(pages)
        det = self._make_detector()
        headers, _footers = det.detect(doc)
        assert "Acme Corp" in headers

    def test_header_missing_two_pages_not_detected(self):
        """5 pages, only 3 have same header -> NOT detected."""
        pages = [
            _make_mock_page(header_text="Acme Corp") for _ in range(3)
        ]
        pages.extend(
            [_make_mock_page(header_text="") for _ in range(2)]
        )
        doc = _make_mock_doc(pages)
        det = self._make_detector()
        headers, _footers = det.detect(doc)
        assert headers == []

    def test_unique_text_not_detected(self):
        unique_texts = [
            "Quarterly Financial Summary",
            "Employee Onboarding Guide",
            "Technical Architecture Review",
            "Customer Satisfaction Report",
            "Vendor Risk Assessment Matrix",
        ]
        pages = [
            _make_mock_page(header_text=unique_texts[i])
            for i in range(5)
        ]
        doc = _make_mock_doc(pages)
        det = self._make_detector()
        headers, footers = det.detect(doc)
        assert headers == []
        assert footers == []

    def test_both_header_and_footer_detected(self):
        pages = [
            _make_mock_page(
                header_text="Company Name", footer_text="Confidential"
            )
            for _ in range(5)
        ]
        doc = _make_mock_doc(pages)
        det = self._make_detector()
        headers, footers = det.detect(doc)
        assert len(headers) == 1
        assert len(footers) == 1

    def test_empty_document(self):
        doc = _make_mock_doc([])
        det = self._make_detector()
        assert det.detect(doc) == ([], [])

    def test_two_page_document(self):
        """Minimum viable case: 2 pages with same header."""
        pages = [
            _make_mock_page(header_text="Report Title") for _ in range(2)
        ]
        doc = _make_mock_doc(pages)
        det = self._make_detector()
        headers, _footers = det.detect(doc)
        assert "Report Title" in headers

    def test_custom_config_thresholds(self):
        """Higher threshold should reject slightly different text."""
        pages = [
            _make_mock_page(header_text="Page 1 of 10"),
            _make_mock_page(header_text="Page 2 of 10"),
            _make_mock_page(header_text="Page 3 of 10"),
            _make_mock_page(header_text="Page 4 of 10"),
            _make_mock_page(header_text="Page 5 of 10"),
        ]
        doc = _make_mock_doc(pages)

        # Default threshold 0.7 should detect these (they're similar)
        det_low = self._make_detector(header_footer_similarity_threshold=0.7)
        headers_low, _ = det_low.detect(doc)
        assert len(headers_low) >= 1

        # Very high threshold should reject them
        det_high = self._make_detector(header_footer_similarity_threshold=0.99)
        headers_high, _ = det_high.detect(doc)
        assert headers_high == []


# ---------------------------------------------------------------------------
# TestStrip
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStrip:
    def _make_detector(self, **kwargs) -> HeaderFooterDetector:
        return HeaderFooterDetector(PDFProcessorConfig(**kwargs))

    def test_strip_header_from_text(self):
        det = self._make_detector()
        text = "Company Name\n\nThis is the body.\nMore body text."
        result = det.strip(text, page_number=0, headers=["Company Name"], footers=[])
        assert "Company Name" not in result
        assert "This is the body." in result

    def test_strip_footer_from_text(self):
        det = self._make_detector()
        text = "Body content.\n\nPage 3 of 10"
        result = det.strip(text, page_number=2, headers=[], footers=["Page 3 of 10"])
        assert "Page 3 of 10" not in result
        assert "Body content." in result

    def test_strip_both_header_and_footer(self):
        det = self._make_detector()
        text = "Company Name\n\nBody text here.\n\nConfidential"
        result = det.strip(
            text, page_number=0,
            headers=["Company Name"],
            footers=["Confidential"],
        )
        assert "Company Name" not in result
        assert "Confidential" not in result
        assert "Body text here." in result

    def test_strip_preserves_body(self):
        det = self._make_detector()
        text = "Header\n\nImportant paragraph.\nAnother line.\n\nFooter"
        result = det.strip(
            text, page_number=0,
            headers=["Header"],
            footers=["Footer"],
        )
        assert "Important paragraph." in result
        assert "Another line." in result

    def test_strip_similar_but_not_exact_match(self):
        """Slightly different footer text still stripped if similarity >= threshold."""
        det = self._make_detector()
        text = "Body content.\n\nPage 1 of 10"
        # Pattern is from a different page number, but should still match
        result = det.strip(
            text, page_number=0,
            headers=[],
            footers=["Page 3 of 10"],
        )
        assert "Page 1 of 10" not in result
        assert "Body content." in result

    def test_strip_no_patterns_returns_unchanged(self):
        det = self._make_detector()
        text = "Some text\nMore text"
        result = det.strip(text, page_number=0, headers=[], footers=[])
        assert result == text

    def test_strip_no_match_returns_unchanged(self):
        det = self._make_detector()
        text = "Completely different content.\nNothing matches."
        result = det.strip(
            text, page_number=0,
            headers=["ZZZZZ Nonexistent Header ZZZZZ"],
            footers=["ZZZZZ Nonexistent Footer ZZZZZ"],
        )
        assert "Completely different content." in result
        assert "Nothing matches." in result


# ---------------------------------------------------------------------------
# TestIntegration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIntegration:
    def test_detect_then_strip_pipeline(self):
        """Full flow: detect on mock doc, then strip sample text."""
        pages = [
            _make_mock_page(
                header_text="ACME Corp - Internal",
                body_text="Page body content.",
                footer_text="Confidential Document",
            )
            for _ in range(5)
        ]
        doc = _make_mock_doc(pages)

        det = HeaderFooterDetector(PDFProcessorConfig())
        headers, footers = det.detect(doc)

        assert len(headers) >= 1
        assert len(footers) >= 1

        raw_text = "ACME Corp - Internal\n\nThis is the real content.\n\nConfidential Document"
        cleaned = det.strip(raw_text, page_number=0, headers=headers, footers=footers)

        assert "ACME Corp - Internal" not in cleaned
        assert "Confidential Document" not in cleaned
        assert "This is the real content." in cleaned
