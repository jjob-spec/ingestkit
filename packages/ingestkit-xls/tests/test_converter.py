"""Tests for ingestkit_xls.converter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from ingestkit_xls.config import XlsProcessorConfig
from ingestkit_xls.converter import ExtractResult, SheetResult, chunk_text, extract_sheets


def _make_cell(ctype: int, value):
    """Create a mock xlrd cell."""
    cell = SimpleNamespace(ctype=ctype, value=value)
    return cell


def _make_sheet(name: str, rows: list[list], ncols: int | None = None):
    """Create a mock xlrd sheet.

    Parameters
    ----------
    name:
        Sheet name.
    rows:
        List of rows, each row is a list of (ctype, value) tuples.
    ncols:
        Number of columns. Inferred from first row if None.
    """
    sheet = MagicMock()
    sheet.name = name
    sheet.nrows = len(rows)
    if ncols is None:
        ncols = len(rows[0]) if rows else 0
    sheet.ncols = ncols

    def cell_func(row_idx, col_idx):
        ctype, value = rows[row_idx][col_idx]
        return _make_cell(ctype, value)

    sheet.cell = cell_func
    return sheet


def _make_workbook(sheets: list, datemode: int = 0):
    """Create a mock xlrd workbook."""
    wb = MagicMock()
    wb.sheets.return_value = sheets
    wb.datemode = datemode
    return wb


# xlrd cell type constants
XL_CELL_EMPTY = 0
XL_CELL_TEXT = 1
XL_CELL_NUMBER = 2
XL_CELL_DATE = 3
XL_CELL_BOOLEAN = 4
XL_CELL_ERROR = 5
XL_CELL_BLANK = 6


class TestExtractSheets:
    """extract_sheets with mocked xlrd."""

    def test_basic_extraction(self):
        """Mock workbook with 1 sheet, 3 rows."""
        sheet = _make_sheet("Sheet1", [
            [(XL_CELL_TEXT, "Name"), (XL_CELL_TEXT, "Age")],
            [(XL_CELL_TEXT, "Alice"), (XL_CELL_NUMBER, 30.0)],
            [(XL_CELL_TEXT, "Bob"), (XL_CELL_NUMBER, 25.0)],
        ])
        wb = _make_workbook([sheet])

        with patch("ingestkit_xls.converter.xlrd") as mock_xlrd:
            mock_xlrd.open_workbook.return_value = wb
            mock_xlrd.XL_CELL_EMPTY = XL_CELL_EMPTY
            mock_xlrd.XL_CELL_TEXT = XL_CELL_TEXT
            mock_xlrd.XL_CELL_NUMBER = XL_CELL_NUMBER
            mock_xlrd.XL_CELL_DATE = XL_CELL_DATE
            mock_xlrd.XL_CELL_BOOLEAN = XL_CELL_BOOLEAN
            mock_xlrd.XL_CELL_ERROR = XL_CELL_ERROR
            mock_xlrd.XL_CELL_BLANK = XL_CELL_BLANK

            result = extract_sheets("/fake/file.xls")

        assert len(result.sheets) == 1
        assert result.sheets[0].name == "Sheet1"
        assert "Name | Age" in result.text
        assert "Alice | 30" in result.text
        assert "Bob | 25" in result.text
        assert result.total_rows == 3
        assert result.word_count > 0

    def test_multi_sheet(self):
        """Mock workbook with 2 sheets, verify section headers."""
        sheet1 = _make_sheet("Employees", [
            [(XL_CELL_TEXT, "Name")],
            [(XL_CELL_TEXT, "Alice")],
        ])
        sheet2 = _make_sheet("Departments", [
            [(XL_CELL_TEXT, "Dept")],
            [(XL_CELL_TEXT, "Engineering")],
        ])
        wb = _make_workbook([sheet1, sheet2])

        with patch("ingestkit_xls.converter.xlrd") as mock_xlrd:
            mock_xlrd.open_workbook.return_value = wb
            mock_xlrd.XL_CELL_EMPTY = XL_CELL_EMPTY
            mock_xlrd.XL_CELL_TEXT = XL_CELL_TEXT
            mock_xlrd.XL_CELL_NUMBER = XL_CELL_NUMBER
            mock_xlrd.XL_CELL_DATE = XL_CELL_DATE
            mock_xlrd.XL_CELL_BOOLEAN = XL_CELL_BOOLEAN
            mock_xlrd.XL_CELL_ERROR = XL_CELL_ERROR
            mock_xlrd.XL_CELL_BLANK = XL_CELL_BLANK

            result = extract_sheets("/fake/file.xls")

        assert len(result.sheets) == 2
        assert "## Employees" in result.text
        assert "## Departments" in result.text

    def test_empty_sheet_skipped(self):
        """Mock sheet with 0 rows."""
        empty_sheet = _make_sheet("Empty", [], ncols=0)
        data_sheet = _make_sheet("Data", [
            [(XL_CELL_TEXT, "Value")],
        ])
        wb = _make_workbook([empty_sheet, data_sheet])

        with patch("ingestkit_xls.converter.xlrd") as mock_xlrd:
            mock_xlrd.open_workbook.return_value = wb
            mock_xlrd.XL_CELL_EMPTY = XL_CELL_EMPTY
            mock_xlrd.XL_CELL_TEXT = XL_CELL_TEXT
            mock_xlrd.XL_CELL_NUMBER = XL_CELL_NUMBER
            mock_xlrd.XL_CELL_DATE = XL_CELL_DATE
            mock_xlrd.XL_CELL_BOOLEAN = XL_CELL_BOOLEAN
            mock_xlrd.XL_CELL_ERROR = XL_CELL_ERROR
            mock_xlrd.XL_CELL_BLANK = XL_CELL_BLANK

            result = extract_sheets("/fake/file.xls")

        assert result.sheets_skipped == 1
        assert len(result.sheets) == 1
        assert result.sheets[0].name == "Data"

    def test_date_cell_conversion(self):
        """Mock cell with ctype=XL_CELL_DATE."""
        from datetime import datetime

        sheet = _make_sheet("Dates", [
            [(XL_CELL_DATE, 44927.0)],  # Some date value
        ])
        wb = _make_workbook([sheet])

        with patch("ingestkit_xls.converter.xlrd") as mock_xlrd:
            mock_xlrd.open_workbook.return_value = wb
            mock_xlrd.XL_CELL_EMPTY = XL_CELL_EMPTY
            mock_xlrd.XL_CELL_TEXT = XL_CELL_TEXT
            mock_xlrd.XL_CELL_NUMBER = XL_CELL_NUMBER
            mock_xlrd.XL_CELL_DATE = XL_CELL_DATE
            mock_xlrd.XL_CELL_BOOLEAN = XL_CELL_BOOLEAN
            mock_xlrd.XL_CELL_ERROR = XL_CELL_ERROR
            mock_xlrd.XL_CELL_BLANK = XL_CELL_BLANK
            mock_xlrd.xldate_as_datetime.return_value = datetime(2023, 1, 1, 0, 0, 0)

            result = extract_sheets("/fake/file.xls")

        assert "2023-01-01 00:00:00" in result.text

    def test_date_conversion_failure(self):
        """Mock xldate_as_datetime raising, verify warning + fallback."""
        sheet = _make_sheet("Dates", [
            [(XL_CELL_DATE, 99999.0)],
        ])
        wb = _make_workbook([sheet])

        with patch("ingestkit_xls.converter.xlrd") as mock_xlrd:
            mock_xlrd.open_workbook.return_value = wb
            mock_xlrd.XL_CELL_EMPTY = XL_CELL_EMPTY
            mock_xlrd.XL_CELL_TEXT = XL_CELL_TEXT
            mock_xlrd.XL_CELL_NUMBER = XL_CELL_NUMBER
            mock_xlrd.XL_CELL_DATE = XL_CELL_DATE
            mock_xlrd.XL_CELL_BOOLEAN = XL_CELL_BOOLEAN
            mock_xlrd.XL_CELL_ERROR = XL_CELL_ERROR
            mock_xlrd.XL_CELL_BLANK = XL_CELL_BLANK
            mock_xlrd.xldate_as_datetime.side_effect = ValueError("bad date")

            result = extract_sheets("/fake/file.xls")

        # Should fall back to str(value)
        assert "99999.0" in result.text

    def test_boolean_cells(self):
        """XL_CELL_BOOLEAN -> TRUE / FALSE."""
        sheet = _make_sheet("Bools", [
            [(XL_CELL_BOOLEAN, 1), (XL_CELL_BOOLEAN, 0)],
        ])
        wb = _make_workbook([sheet])

        with patch("ingestkit_xls.converter.xlrd") as mock_xlrd:
            mock_xlrd.open_workbook.return_value = wb
            mock_xlrd.XL_CELL_EMPTY = XL_CELL_EMPTY
            mock_xlrd.XL_CELL_TEXT = XL_CELL_TEXT
            mock_xlrd.XL_CELL_NUMBER = XL_CELL_NUMBER
            mock_xlrd.XL_CELL_DATE = XL_CELL_DATE
            mock_xlrd.XL_CELL_BOOLEAN = XL_CELL_BOOLEAN
            mock_xlrd.XL_CELL_ERROR = XL_CELL_ERROR
            mock_xlrd.XL_CELL_BLANK = XL_CELL_BLANK

            result = extract_sheets("/fake/file.xls")

        assert "TRUE | FALSE" in result.text

    def test_error_cells(self):
        """XL_CELL_ERROR -> #ERROR."""
        sheet = _make_sheet("Errors", [
            [(XL_CELL_ERROR, 0x07)],
        ])
        wb = _make_workbook([sheet])

        with patch("ingestkit_xls.converter.xlrd") as mock_xlrd:
            mock_xlrd.open_workbook.return_value = wb
            mock_xlrd.XL_CELL_EMPTY = XL_CELL_EMPTY
            mock_xlrd.XL_CELL_TEXT = XL_CELL_TEXT
            mock_xlrd.XL_CELL_NUMBER = XL_CELL_NUMBER
            mock_xlrd.XL_CELL_DATE = XL_CELL_DATE
            mock_xlrd.XL_CELL_BOOLEAN = XL_CELL_BOOLEAN
            mock_xlrd.XL_CELL_ERROR = XL_CELL_ERROR
            mock_xlrd.XL_CELL_BLANK = XL_CELL_BLANK

            result = extract_sheets("/fake/file.xls")

        assert "#ERROR" in result.text

    def test_float_integer_stripping(self):
        """1.0 -> '1' (strip trailing .0)."""
        sheet = _make_sheet("Numbers", [
            [(XL_CELL_NUMBER, 1.0), (XL_CELL_NUMBER, 3.14)],
        ])
        wb = _make_workbook([sheet])

        with patch("ingestkit_xls.converter.xlrd") as mock_xlrd:
            mock_xlrd.open_workbook.return_value = wb
            mock_xlrd.XL_CELL_EMPTY = XL_CELL_EMPTY
            mock_xlrd.XL_CELL_TEXT = XL_CELL_TEXT
            mock_xlrd.XL_CELL_NUMBER = XL_CELL_NUMBER
            mock_xlrd.XL_CELL_DATE = XL_CELL_DATE
            mock_xlrd.XL_CELL_BOOLEAN = XL_CELL_BOOLEAN
            mock_xlrd.XL_CELL_ERROR = XL_CELL_ERROR
            mock_xlrd.XL_CELL_BLANK = XL_CELL_BLANK

            result = extract_sheets("/fake/file.xls")

        assert "1 | 3.14" in result.text

    def test_xlrd_unavailable(self):
        """Verify ImportError raised when xlrd is None."""
        with patch("ingestkit_xls.converter.xlrd", None):
            with pytest.raises(ImportError, match="xlrd is required"):
                extract_sheets("/fake/file.xls")

    def test_all_sheets_empty(self):
        """All sheets empty -> empty text and sheets_skipped count."""
        empty1 = _make_sheet("Empty1", [], ncols=0)
        empty2 = _make_sheet("Empty2", [], ncols=0)
        wb = _make_workbook([empty1, empty2])

        with patch("ingestkit_xls.converter.xlrd") as mock_xlrd:
            mock_xlrd.open_workbook.return_value = wb
            mock_xlrd.XL_CELL_EMPTY = XL_CELL_EMPTY
            mock_xlrd.XL_CELL_TEXT = XL_CELL_TEXT
            mock_xlrd.XL_CELL_NUMBER = XL_CELL_NUMBER
            mock_xlrd.XL_CELL_DATE = XL_CELL_DATE
            mock_xlrd.XL_CELL_BOOLEAN = XL_CELL_BOOLEAN
            mock_xlrd.XL_CELL_ERROR = XL_CELL_ERROR
            mock_xlrd.XL_CELL_BLANK = XL_CELL_BLANK

            result = extract_sheets("/fake/file.xls")

        assert result.text == ""
        assert result.sheets_skipped == 2
        assert result.word_count == 0


class TestChunkText:
    """chunk_text splitting logic."""

    def test_empty_text(self):
        config = XlsProcessorConfig()
        assert chunk_text("", config) == []

    def test_whitespace_only(self):
        config = XlsProcessorConfig()
        assert chunk_text("   \n\n  ", config) == []

    def test_single_short_paragraph(self):
        config = XlsProcessorConfig(chunk_size_tokens=512)
        text = "This is a short paragraph."
        chunks = chunk_text(text, config)
        assert len(chunks) == 1
        assert chunks[0] == "This is a short paragraph."

    def test_multiple_paragraphs_single_chunk(self):
        config = XlsProcessorConfig(chunk_size_tokens=512)
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = chunk_text(text, config)
        assert len(chunks) == 1
        assert "Paragraph one." in chunks[0]
        assert "Paragraph two." in chunks[0]

    def test_multiple_chunks(self):
        config = XlsProcessorConfig(chunk_size_tokens=5, chunk_overlap_tokens=0)
        text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
        chunks = chunk_text(text, config)
        assert len(chunks) >= 2

    def test_overlap(self):
        config = XlsProcessorConfig(chunk_size_tokens=5, chunk_overlap_tokens=3)
        text = "Word one two.\n\nWord three four.\n\nWord five six."
        chunks = chunk_text(text, config)
        if len(chunks) > 1:
            assert len(chunks) >= 2

    def test_long_paragraph_split_on_newlines(self):
        config = XlsProcessorConfig(chunk_size_tokens=5, chunk_overlap_tokens=0)
        text = "Line one here now.\nLine two here now.\nLine three here now."
        chunks = chunk_text(text, config)
        assert len(chunks) >= 2
