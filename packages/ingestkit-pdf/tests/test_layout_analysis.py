"""Tests for ingestkit_pdf.utils.layout_analysis — LayoutAnalyzer."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ingestkit_pdf.config import PDFProcessorConfig
from ingestkit_pdf.utils.layout_analysis import (
    LayoutAnalyzer,
    LayoutResult,
    TextBlock,
    extract_text_blocks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PAGE_WIDTH = 612.0  # Standard US-Letter width in points
PAGE_HEIGHT = 792.0


def _make_block(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    text: str,
    block_no: int = 0,
    block_type: int = 0,
) -> tuple:
    """Create a fitz-style block tuple."""
    return (x0, y0, x1, y1, text, block_no, block_type)


def _make_mock_page(
    blocks: list[tuple],
    width: float = PAGE_WIDTH,
    height: float = PAGE_HEIGHT,
) -> MagicMock:
    """Create a mock fitz.Page returning the given blocks."""
    page = MagicMock()
    page.get_text.return_value = blocks
    page.rect = MagicMock()
    page.rect.width = width
    page.rect.height = height
    return page


def _make_analyzer() -> LayoutAnalyzer:
    """Create a LayoutAnalyzer with default config."""
    return LayoutAnalyzer(config=PDFProcessorConfig())


# ---------------------------------------------------------------------------
# T1-T3: Single-column detection
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSingleColumn:
    """T1-T3: Single-column detection scenarios."""

    def test_t1_single_column_aligned_left(self) -> None:
        """T1: Five blocks aligned at the same x0 -> single column."""
        blocks = [
            _make_block(72, y * 100, 300, y * 100 + 80, f"Para {y}", y)
            for y in range(5)
        ]
        page = _make_mock_page(blocks)
        analyzer = _make_analyzer()

        result = analyzer.detect_columns(page)

        assert result.is_multi_column is False
        assert result.column_count == 1

    def test_t2_no_text_blocks_image_only(self) -> None:
        """T2: Page with only image blocks -> single column."""
        blocks = [
            _make_block(72, 100, 300, 200, "img", 0, block_type=1),
            _make_block(72, 300, 300, 400, "img", 1, block_type=1),
        ]
        page = _make_mock_page(blocks)
        analyzer = _make_analyzer()

        result = analyzer.detect_columns(page)

        assert result.is_multi_column is False
        assert result.column_count == 1

    def test_t3_fewer_than_three_text_blocks(self) -> None:
        """T3: Only two text blocks -> single column (threshold not met)."""
        blocks = [
            _make_block(72, 100, 300, 200, "Block 1", 0),
            _make_block(350, 100, 550, 200, "Block 2", 1),
        ]
        page = _make_mock_page(blocks)
        analyzer = _make_analyzer()

        result = analyzer.detect_columns(page)

        assert result.is_multi_column is False
        assert result.column_count == 1


# ---------------------------------------------------------------------------
# T4-T5: Two-column detection
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTwoColumn:
    """T4-T5: Two-column detection scenarios."""

    def test_t4_classic_two_column(self) -> None:
        """T4: 3 blocks left (x0=72) + 3 blocks right (x0=320) -> 2 columns."""
        blocks = [
            _make_block(72, 100, 280, 180, "Left 1", 0),
            _make_block(72, 200, 280, 280, "Left 2", 1),
            _make_block(72, 300, 280, 380, "Left 3", 2),
            _make_block(320, 100, 540, 180, "Right 1", 3),
            _make_block(320, 200, 540, 280, "Right 2", 4),
            _make_block(320, 300, 540, 380, "Right 3", 5),
        ]
        page = _make_mock_page(blocks)
        analyzer = _make_analyzer()

        result = analyzer.detect_columns(page)

        assert result.is_multi_column is True
        assert result.column_count == 2

    def test_t5_two_columns_unequal_block_count(self) -> None:
        """T5: 4 blocks left + 2 blocks right -> 2 columns."""
        blocks = [
            _make_block(72, 100, 280, 180, "L1", 0),
            _make_block(72, 200, 280, 280, "L2", 1),
            _make_block(72, 300, 280, 380, "L3", 2),
            _make_block(72, 400, 280, 480, "L4", 3),
            _make_block(320, 100, 540, 180, "R1", 4),
            _make_block(320, 200, 540, 280, "R2", 5),
        ]
        page = _make_mock_page(blocks)
        analyzer = _make_analyzer()

        result = analyzer.detect_columns(page)

        assert result.is_multi_column is True
        assert result.column_count == 2


# ---------------------------------------------------------------------------
# T6: Three-column detection
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestThreeColumn:
    """T6: Three-column detection."""

    def test_t6_three_column_layout(self) -> None:
        """T6: 2 blocks each at x0=50, 220, 400 -> 3 columns."""
        blocks = [
            _make_block(50, 100, 180, 200, "Col1-A", 0),
            _make_block(50, 250, 180, 350, "Col1-B", 1),
            _make_block(220, 100, 360, 200, "Col2-A", 2),
            _make_block(220, 250, 360, 350, "Col2-B", 3),
            _make_block(400, 100, 560, 200, "Col3-A", 4),
            _make_block(400, 250, 560, 350, "Col3-B", 5),
        ]
        page = _make_mock_page(blocks)
        analyzer = _make_analyzer()

        result = analyzer.detect_columns(page)

        assert result.is_multi_column is True
        assert result.column_count == 3


# ---------------------------------------------------------------------------
# T7-T8: Mixed layout (full-width header + columns)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMixedLayout:
    """T7-T8: Mixed full-width and columnar layouts."""

    def test_t7_full_width_title_plus_two_columns(self) -> None:
        """T7: One full-width header block + 4 columnar blocks -> 2 columns."""
        # Full-width block spans 90% of page (550 / 612 = ~90%)
        blocks = [
            _make_block(30, 50, 580, 90, "Page Title", 0),
            _make_block(72, 150, 280, 250, "Left 1", 1),
            _make_block(72, 270, 280, 370, "Left 2", 2),
            _make_block(320, 150, 540, 250, "Right 1", 3),
            _make_block(320, 270, 540, 370, "Right 2", 4),
        ]
        page = _make_mock_page(blocks)
        analyzer = _make_analyzer()

        result = analyzer.detect_columns(page)

        assert result.is_multi_column is True
        assert result.column_count == 2

    def test_t8_all_blocks_full_width(self) -> None:
        """T8: All blocks span >75% of page -> single column."""
        blocks = [
            _make_block(30, 100, 580, 200, "Wide block 1", 0),
            _make_block(30, 250, 580, 350, "Wide block 2", 1),
            _make_block(30, 400, 580, 500, "Wide block 3", 2),
        ]
        page = _make_mock_page(blocks)
        analyzer = _make_analyzer()

        result = analyzer.detect_columns(page)

        assert result.is_multi_column is False
        assert result.column_count == 1


# ---------------------------------------------------------------------------
# T9-T11: Reading order (reorder_blocks)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReorderBlocks:
    """T9-T11: Reading order correctness."""

    def test_t9_two_column_reading_order(self) -> None:
        """T9: Two-column blocks reordered: left col top-to-bottom, then right."""
        layout = LayoutResult(
            is_multi_column=True,
            column_count=2,
            column_boundaries=[(72.0, 280.0), (320.0, 540.0)],
            page_width=PAGE_WIDTH,
        )
        blocks = [
            TextBlock(x0=320, y0=100, x1=540, y1=180, text="Right 1", block_number=0),
            TextBlock(x0=72, y0=200, x1=280, y1=280, text="Left 2", block_number=1),
            TextBlock(x0=72, y0=100, x1=280, y1=180, text="Left 1", block_number=2),
            TextBlock(x0=320, y0=200, x1=540, y1=280, text="Right 2", block_number=3),
        ]
        analyzer = _make_analyzer()

        reordered = analyzer.reorder_blocks(blocks, layout)
        texts = [b.text for b in reordered]

        assert texts == ["Left 1", "Left 2", "Right 1", "Right 2"]

    def test_t10_single_column_sorted_by_y(self) -> None:
        """T10: Single-column blocks sorted by y0 (top to bottom)."""
        layout = LayoutResult(
            is_multi_column=False,
            column_count=1,
            column_boundaries=[(0.0, PAGE_WIDTH)],
            page_width=PAGE_WIDTH,
        )
        blocks = [
            TextBlock(x0=72, y0=300, x1=500, y1=380, text="Third", block_number=0),
            TextBlock(x0=72, y0=100, x1=500, y1=180, text="First", block_number=1),
            TextBlock(x0=72, y0=200, x1=500, y1=280, text="Second", block_number=2),
        ]
        analyzer = _make_analyzer()

        reordered = analyzer.reorder_blocks(blocks, layout)
        texts = [b.text for b in reordered]

        assert texts == ["First", "Second", "Third"]

    def test_t11_mixed_header_plus_two_columns(self) -> None:
        """T11: Full-width header first, then left column, then right column."""
        layout = LayoutResult(
            is_multi_column=True,
            column_count=2,
            column_boundaries=[(72.0, 280.0), (320.0, 540.0)],
            page_width=PAGE_WIDTH,
        )
        # Full-width header: 580 - 30 = 550 > 612 * 0.75 = 459
        blocks = [
            TextBlock(x0=320, y0=150, x1=540, y1=230, text="Right 1", block_number=0),
            TextBlock(x0=30, y0=50, x1=580, y1=90, text="Header", block_number=1),
            TextBlock(x0=72, y0=150, x1=280, y1=230, text="Left 1", block_number=2),
        ]
        analyzer = _make_analyzer()

        reordered = analyzer.reorder_blocks(blocks, layout)
        texts = [b.text for b in reordered]

        assert texts == ["Header", "Left 1", "Right 1"]


# ---------------------------------------------------------------------------
# T12-T16: Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEdgeCases:
    """T12-T16: Edge case handling."""

    def test_t12_image_blocks_filtered_out(self) -> None:
        """T12: Image blocks (type=1) are excluded from extraction."""
        blocks = [
            _make_block(72, 100, 300, 200, "Text block", 0, block_type=0),
            _make_block(72, 250, 300, 400, "image", 1, block_type=1),
            _make_block(72, 450, 300, 550, "Another text", 2, block_type=0),
        ]
        page = _make_mock_page(blocks)

        result = extract_text_blocks(page)

        assert len(result) == 2
        assert all(b.text.strip() != "image" for b in result)

    def test_t13_empty_text_blocks_filtered(self) -> None:
        """T13: Text blocks with whitespace-only content are excluded."""
        blocks = [
            _make_block(72, 100, 300, 200, "Real content", 0),
            _make_block(72, 250, 300, 350, "   \n  ", 1),
            _make_block(72, 400, 300, 500, "", 2),
            _make_block(72, 550, 300, 650, "More content", 3),
        ]
        page = _make_mock_page(blocks)

        result = extract_text_blocks(page)

        assert len(result) == 2
        assert result[0].text.strip() == "Real content"
        assert result[1].text.strip() == "More content"

    def test_t14_narrow_page_gap_too_small(self) -> None:
        """T14: Very narrow page where gap between x0 values is < 10% width."""
        # Page width 200; 10% = 20. Gap between x0=50 and x0=65 is only 15.
        blocks = [
            _make_block(50, 100, 100, 200, "A", 0),
            _make_block(50, 250, 100, 350, "B", 1),
            _make_block(65, 100, 120, 200, "C", 2),
            _make_block(65, 250, 120, 350, "D", 3),
        ]
        page = _make_mock_page(blocks, width=200.0)
        analyzer = _make_analyzer()

        result = analyzer.detect_columns(page)

        assert result.is_multi_column is False
        assert result.column_count == 1

    def test_t15_overlapping_x_ranges_assigned_by_midpoint(self) -> None:
        """T15: Blocks with overlapping x ranges are assigned by midpoint."""
        layout = LayoutResult(
            is_multi_column=True,
            column_count=2,
            column_boundaries=[(72.0, 280.0), (320.0, 540.0)],
            page_width=PAGE_WIDTH,
        )
        # This block has x0=250, x1=330 -> midpoint 290 -> closer to col 1 mid (176)
        # Actually midpoint 290 vs col1 mid 176 (dist 114) vs col2 mid 430 (dist 140)
        # So assigned to column 1.
        overlap_block = TextBlock(
            x0=250, y0=100, x1=330, y1=180, text="Overlap", block_number=0
        )
        left_block = TextBlock(
            x0=72, y0=100, x1=280, y1=180, text="Left", block_number=1
        )
        right_block = TextBlock(
            x0=320, y0=100, x1=540, y1=180, text="Right", block_number=2
        )
        analyzer = _make_analyzer()

        reordered = analyzer.reorder_blocks(
            [right_block, overlap_block, left_block], layout
        )
        texts = [b.text for b in reordered]

        # Left and Overlap both in col 1 (sorted by y0, same y0 -> stable),
        # then Right in col 2.
        assert texts[0] in ("Left", "Overlap")
        assert texts[1] in ("Left", "Overlap")
        assert texts[2] == "Right"

    def test_t16_exception_in_get_text_returns_default(self) -> None:
        """T16: If page.get_text raises, return single-column default."""
        page = MagicMock()
        page.get_text.side_effect = RuntimeError("fitz error")
        page.rect.width = PAGE_WIDTH
        analyzer = _make_analyzer()

        result = analyzer.detect_columns(page)

        assert result.is_multi_column is False
        assert result.column_count == 1


# ---------------------------------------------------------------------------
# T17-T18: Column boundaries validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestColumnBoundaries:
    """T17-T18: Column boundary correctness."""

    def test_t17_two_column_boundaries_non_overlapping(self) -> None:
        """T17: Two-column boundaries are non-overlapping and correctly ordered."""
        blocks = [
            _make_block(72, 100, 280, 200, "L1", 0),
            _make_block(72, 250, 280, 350, "L2", 1),
            _make_block(320, 100, 540, 200, "R1", 2),
            _make_block(320, 250, 540, 350, "R2", 3),
        ]
        page = _make_mock_page(blocks)
        analyzer = _make_analyzer()

        result = analyzer.detect_columns(page)

        assert len(result.column_boundaries) == 2
        left_start, left_end = result.column_boundaries[0]
        right_start, right_end = result.column_boundaries[1]
        # Left column ends before right column starts.
        assert left_end < right_start
        # Sorted left-to-right.
        assert left_start < right_start

    def test_t18_three_column_boundaries_sorted(self) -> None:
        """T18: Three-column boundaries are sorted left-to-right."""
        blocks = [
            _make_block(50, 100, 180, 200, "C1-A", 0),
            _make_block(50, 250, 180, 350, "C1-B", 1),
            _make_block(220, 100, 360, 200, "C2-A", 2),
            _make_block(220, 250, 360, 350, "C2-B", 3),
            _make_block(400, 100, 560, 200, "C3-A", 4),
            _make_block(400, 250, 560, 350, "C3-B", 5),
        ]
        page = _make_mock_page(blocks)
        analyzer = _make_analyzer()

        result = analyzer.detect_columns(page)

        assert len(result.column_boundaries) == 3
        starts = [b[0] for b in result.column_boundaries]
        assert starts == sorted(starts)
        # Each boundary's start < end.
        for start, end in result.column_boundaries:
            assert start < end
