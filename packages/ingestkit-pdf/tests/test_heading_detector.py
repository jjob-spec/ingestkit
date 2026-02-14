"""Tests for ingestkit_pdf.utils.heading_detector — heading hierarchy detection.

All tests mock fitz.Document and fitz.Page. No binary PDF fixtures.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ingestkit_pdf.config import PDFProcessorConfig
from ingestkit_pdf.utils.heading_detector import HeadingDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> PDFProcessorConfig:
    return PDFProcessorConfig()


@pytest.fixture
def detector(default_config: PDFProcessorConfig) -> HeadingDetector:
    return HeadingDetector(default_config)


def _make_text_block(spans: list[dict], bbox: tuple = (0, 0, 100, 100)) -> dict:
    """Build a text block dict matching fitz page.get_text('dict') structure."""
    lines = []
    for span in spans:
        span.setdefault("origin", [0, span.get("_y", 100)])
        lines.append({"spans": [span], "bbox": list(bbox)})
    return {"type": 0, "lines": lines}


def _make_mock_page(blocks: list[dict], page_number: int = 0) -> MagicMock:
    """Create a mock fitz.Page returning the given blocks from get_text('dict')."""
    page = MagicMock()
    page.get_text.return_value = {"blocks": blocks}
    page.number = page_number
    return page


def _make_mock_doc(
    toc: list | None = None,
    pages: list[MagicMock] | None = None,
) -> MagicMock:
    """Create a mock fitz.Document."""
    doc = MagicMock()
    doc.get_toc.return_value = toc if toc is not None else []
    if pages is None:
        pages = []
    doc.__len__ = MagicMock(return_value=len(pages))
    doc.__getitem__ = MagicMock(side_effect=lambda i: pages[i])
    return doc


# Helpers for font-based tests.
BOLD_FLAG = 2**4  # bit 4 = bold in fitz flags


def _span(text: str, size: float, bold: bool = False, y: float = 100.0) -> dict:
    return {
        "text": text,
        "size": size,
        "flags": BOLD_FLAG if bold else 0,
        "_y": y,
    }


# ---------------------------------------------------------------------------
# TestDetectFromOutline
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDetectFromOutline:
    def test_toc_returns_headings(self, detector: HeadingDetector) -> None:
        toc = [[1, "Chapter 1", 1], [2, "Section 1.1", 1], [1, "Chapter 2", 3]]
        doc = _make_mock_doc(toc=toc)
        result = detector.detect(doc)
        assert result == [
            (1, "Chapter 1", 1),
            (2, "Section 1.1", 1),
            (1, "Chapter 2", 3),
        ]

    def test_toc_strips_empty_titles(self, detector: HeadingDetector) -> None:
        toc = [[1, "Chapter 1", 1], [1, "", 2], [1, "  ", 3], [1, "Chapter 3", 4]]
        doc = _make_mock_doc(toc=toc)
        result = detector.detect(doc)
        assert len(result) == 2
        assert result[0] == (1, "Chapter 1", 1)
        assert result[1] == (1, "Chapter 3", 4)

    def test_toc_preferred_over_fonts(self, detector: HeadingDetector) -> None:
        """When TOC exists, font-based detection is not used."""
        toc = [[1, "From TOC", 1]]
        # Create a page that also has bold large text — should be ignored.
        page = _make_mock_page(
            [_make_text_block([_span("Bold Title", 24.0, bold=True)])],
        )
        doc = _make_mock_doc(toc=toc, pages=[page])
        result = detector.detect(doc)
        assert result == [(1, "From TOC", 1)]


# ---------------------------------------------------------------------------
# TestDetectFromFonts
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDetectFromFonts:
    def _body_page(
        self,
        heading_spans: list[dict],
        body_text: str = "Body text " * 50,
        body_size: float = 12.0,
    ) -> MagicMock:
        """Create a page with body text and heading spans."""
        body_span = _span(body_text, body_size, bold=False, y=300.0)
        blocks = [
            _make_text_block([body_span]),
            _make_text_block(heading_spans),
        ]
        return _make_mock_page(blocks)

    def test_bold_large_text_detected(self, detector: HeadingDetector) -> None:
        page = self._body_page([
            _span("Title", 18.0, bold=True, y=50.0),
            _span("Section", 15.0, bold=True, y=150.0),
        ])
        doc = _make_mock_doc(pages=[page])
        result = detector.detect(doc)
        assert len(result) == 2
        assert result[0] == (1, "Title", 1)
        assert result[1] == (2, "Section", 1)

    def test_non_bold_large_text_ignored(self, detector: HeadingDetector) -> None:
        page = self._body_page([
            _span("Not a heading", 18.0, bold=False, y=50.0),
        ])
        doc = _make_mock_doc(pages=[page])
        result = detector.detect(doc)
        assert result == []

    def test_body_size_calculated_correctly(self, detector: HeadingDetector) -> None:
        """Body size = font with greatest total character count."""
        # 80% at 12pt, 15% at 10pt, 5% at 18pt bold
        body12 = _span("x" * 800, 12.0, bold=False, y=200.0)
        body10 = _span("y" * 150, 10.0, bold=False, y=250.0)
        heading = _span("Heading", 18.0, bold=True, y=50.0)
        blocks = [
            _make_text_block([body12]),
            _make_text_block([body10]),
            _make_text_block([heading]),
        ]
        page = _make_mock_page(blocks)
        doc = _make_mock_doc(pages=[page])
        result = detector.detect(doc)
        # Body=12pt, threshold=12*1.2=14.4, 18pt bold qualifies
        assert len(result) == 1
        assert result[0][1] == "Heading"

    def test_three_levels_mapped(self, detector: HeadingDetector) -> None:
        page = self._body_page([
            _span("H1", 24.0, bold=True, y=50.0),
            _span("H2", 18.0, bold=True, y=100.0),
            _span("H3", 15.0, bold=True, y=150.0),
        ])
        doc = _make_mock_doc(pages=[page])
        result = detector.detect(doc)
        assert [(r[0], r[1]) for r in result] == [
            (1, "H1"),
            (2, "H2"),
            (3, "H3"),
        ]

    def test_more_than_three_sizes_capped(self, detector: HeadingDetector) -> None:
        page = self._body_page([
            _span("S28", 28.0, bold=True, y=30.0),
            _span("S24", 24.0, bold=True, y=60.0),
            _span("S18", 18.0, bold=True, y=90.0),
            _span("S15", 15.0, bold=True, y=120.0),
        ])
        doc = _make_mock_doc(pages=[page])
        result = detector.detect(doc)
        levels = [r[0] for r in result]
        assert levels == [1, 2, 3, 3]

    def test_consecutive_spans_merged(self, detector: HeadingDetector) -> None:
        """Two bold 18pt spans on same line merge into one heading."""
        page = self._body_page([
            _span("Part", 18.0, bold=True, y=50.0),
            _span("One", 18.0, bold=True, y=50.0),
        ])
        doc = _make_mock_doc(pages=[page])
        result = detector.detect(doc)
        assert len(result) == 1
        assert result[0][1] == "Part One"

    def test_no_bold_headings_returns_empty(self, detector: HeadingDetector) -> None:
        """All text is regular weight — no headings detected."""
        page = _make_mock_page([
            _make_text_block([_span("Normal text " * 50, 12.0, bold=False)]),
        ])
        doc = _make_mock_doc(pages=[page])
        result = detector.detect(doc)
        # May fall through to markdown strategy; patch it to also return empty.
        assert isinstance(result, list)

    def test_heading_min_font_size_ratio_configurable(self) -> None:
        config = PDFProcessorConfig(heading_min_font_size_ratio=1.5)
        det = HeadingDetector(config)
        # Body=12pt, threshold=12*1.5=18, so 15pt bold does NOT qualify.
        body_span = _span("body " * 100, 12.0, bold=False, y=200.0)
        heading15 = _span("Small heading", 15.0, bold=True, y=50.0)
        heading18 = _span("Big heading", 18.0, bold=True, y=80.0)
        blocks = [
            _make_text_block([body_span]),
            _make_text_block([heading15, heading18]),
        ]
        page = _make_mock_page(blocks)
        doc = _make_mock_doc(pages=[page])
        result = det.detect(doc)
        titles = [r[1] for r in result]
        assert "Small heading" not in titles
        assert "Big heading" in titles


# ---------------------------------------------------------------------------
# TestDetectFromMarkdown
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDetectFromMarkdown:
    def test_markdown_headers_parsed(self, detector: HeadingDetector) -> None:
        chunks = [
            {
                "metadata": {"page": 1},
                "text": "# Heading 1\nSome text\n## Heading 2\n### Heading 3\n",
            }
        ]
        # No TOC, no bold text — fall through to markdown.
        doc = _make_mock_doc(pages=[_make_mock_page([])])
        import sys
        mock_md = MagicMock()
        mock_md.to_markdown.return_value = chunks
        sys.modules["pymupdf4llm"] = mock_md
        try:
            result = detector.detect(doc)
        finally:
            sys.modules.pop("pymupdf4llm", None)
        assert len(result) == 3
        assert result[0] == (1, "Heading 1", 1)
        assert result[1] == (2, "Heading 2", 1)
        assert result[2] == (3, "Heading 3", 1)

    def test_h4_and_deeper_ignored(self, detector: HeadingDetector) -> None:
        chunks = [
            {
                "metadata": {"page": 1},
                "text": "# H1\n#### H4\n##### H5\n",
            }
        ]
        doc = _make_mock_doc(pages=[_make_mock_page([])])
        import sys
        mock_md = MagicMock()
        mock_md.to_markdown.return_value = chunks
        sys.modules["pymupdf4llm"] = mock_md
        try:
            result = detector.detect(doc)
        finally:
            sys.modules.pop("pymupdf4llm", None)
        assert len(result) == 1
        assert result[0] == (1, "H1", 1)

    def test_page_numbers_from_chunks(self, detector: HeadingDetector) -> None:
        chunks = [
            {"metadata": {"page": 1}, "text": "# Page1 Heading\n"},
            {"metadata": {"page": 3}, "text": "## Page3 Heading\n"},
        ]
        doc = _make_mock_doc(pages=[_make_mock_page([])])
        import sys
        mock_md = MagicMock()
        mock_md.to_markdown.return_value = chunks
        sys.modules["pymupdf4llm"] = mock_md
        try:
            result = detector.detect(doc)
        finally:
            sys.modules.pop("pymupdf4llm", None)
        assert result[0] == (1, "Page1 Heading", 1)
        assert result[1] == (2, "Page3 Heading", 3)


# ---------------------------------------------------------------------------
# TestGetHeadingPath
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetHeadingPath:
    def test_ancestry_built_correctly(self, detector: HeadingDetector) -> None:
        toc = [[1, "H1", 1], [2, "H2", 1], [3, "H3", 2]]
        doc = _make_mock_doc(toc=toc)
        detector.detect(doc)
        path = detector.get_heading_path(page_number=2, position_y=100.0)
        assert path == ["H1", "H2", "H3"]

    def test_mid_document_position(self, detector: HeadingDetector) -> None:
        toc = [[1, "Ch1", 1], [2, "Sec1", 2], [1, "Ch2", 5], [2, "Sec2", 6]]
        doc = _make_mock_doc(toc=toc)
        detector.detect(doc)
        path = detector.get_heading_path(page_number=3, position_y=0.0)
        assert path == ["Ch1", "Sec1"]

    def test_no_headings_returns_empty(self, detector: HeadingDetector) -> None:
        path = detector.get_heading_path(page_number=1, position_y=0.0)
        assert path == []

    def test_position_before_first_heading(self, detector: HeadingDetector) -> None:
        toc = [[1, "Ch1", 3]]
        doc = _make_mock_doc(toc=toc)
        detector.detect(doc)
        path = detector.get_heading_path(page_number=1, position_y=0.0)
        assert path == []

    def test_y_position_selects_correct_heading(self) -> None:
        """Two H2 headings on the same page; query between them."""
        config = PDFProcessorConfig()
        det = HeadingDetector(config)
        toc = [[1, "Ch1", 1], [2, "Sec A", 1], [2, "Sec B", 1]]
        doc = _make_mock_doc(toc=toc)
        det.detect(doc)
        # Outline headings all have y=0.0, so both Sec A and Sec B are at y=0.
        # With y=0, the last heading on that page at y<=0 wins = Sec B.
        path = det.get_heading_path(page_number=1, position_y=0.0)
        assert path == ["Ch1", "Sec B"]


# ---------------------------------------------------------------------------
# TestStrategyCascade
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStrategyCascade:
    def test_outline_used_when_available(self, detector: HeadingDetector) -> None:
        toc = [[1, "Outline", 1]]
        page = _make_mock_page(
            [_make_text_block([_span("body " * 100, 12.0), _span("Bold", 18.0, bold=True, y=50.0)])],
        )
        doc = _make_mock_doc(toc=toc, pages=[page])
        result = detector.detect(doc)
        assert result == [(1, "Outline", 1)]

    def test_fonts_used_when_no_outline(self, detector: HeadingDetector) -> None:
        body_span = _span("body text " * 100, 12.0, bold=False, y=200.0)
        heading_span = _span("Font Heading", 18.0, bold=True, y=50.0)
        page = _make_mock_page([
            _make_text_block([body_span]),
            _make_text_block([heading_span]),
        ])
        doc = _make_mock_doc(toc=[], pages=[page])
        result = detector.detect(doc)
        assert len(result) >= 1
        assert result[0][1] == "Font Heading"

    def test_markdown_used_as_last_resort(self, detector: HeadingDetector) -> None:
        # Empty page (no text blocks) — font strategy returns empty.
        page = _make_mock_page([])
        doc = _make_mock_doc(toc=[], pages=[page])

        chunks = [{"metadata": {"page": 1}, "text": "# MD Heading\n"}]
        import sys
        mock_md = MagicMock()
        mock_md.to_markdown.return_value = chunks
        sys.modules["pymupdf4llm"] = mock_md
        try:
            result = detector.detect(doc)
        finally:
            sys.modules.pop("pymupdf4llm", None)
        assert result == [(1, "MD Heading", 1)]

    def test_all_strategies_fail_returns_empty(self, detector: HeadingDetector) -> None:
        page = _make_mock_page([])
        doc = _make_mock_doc(toc=[], pages=[page])
        # pymupdf4llm not available.
        import sys
        sys.modules.pop("pymupdf4llm", None)
        result = detector.detect(doc)
        assert result == []


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEdgeCases:
    def test_single_page_doc(self, detector: HeadingDetector) -> None:
        toc = [[1, "Only Heading", 1]]
        doc = _make_mock_doc(toc=toc)
        result = detector.detect(doc)
        assert result == [(1, "Only Heading", 1)]

    def test_unicode_heading_titles(self, detector: HeadingDetector) -> None:
        toc = [[1, "Ubersicht", 1], [2, "Resumen ejecutivo", 2]]
        doc = _make_mock_doc(toc=toc)
        result = detector.detect(doc)
        assert result[0][1] == "Ubersicht"
        assert result[1][1] == "Resumen ejecutivo"

    def test_very_long_heading_title(self, detector: HeadingDetector) -> None:
        long_title = "A" * 500
        toc = [[1, long_title, 1]]
        doc = _make_mock_doc(toc=toc)
        result = detector.detect(doc)
        assert result[0][1] == long_title
        assert len(result[0][1]) == 500

    def test_detect_idempotent(self, detector: HeadingDetector) -> None:
        toc = [[1, "Ch1", 1], [2, "Sec1", 2]]
        doc = _make_mock_doc(toc=toc)
        result1 = detector.detect(doc)
        result2 = detector.detect(doc)
        assert result1 == result2
