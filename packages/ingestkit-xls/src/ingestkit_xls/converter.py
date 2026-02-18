"""Core .xls conversion logic -- extract sheets and chunk.

Provides ``extract_sheets()`` to pull structured text from legacy Excel
workbooks using xlrd, and ``chunk_text()`` to split the result into
token-sized chunks suitable for embedding.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from ingestkit_xls.config import XlsProcessorConfig

logger = logging.getLogger("ingestkit_xls")

# Import guard: xlrd is an optional dependency
try:
    import xlrd  # type: ignore[import-untyped]
except ImportError:
    xlrd = None  # type: ignore[assignment]


class SheetResult(BaseModel):
    """Extraction result for a single worksheet."""

    name: str
    text: str
    row_count: int
    col_count: int


class ExtractResult(BaseModel):
    """Internal model for the output of ``extract_sheets()``."""

    sheets: list[SheetResult]
    text: str
    word_count: int
    total_rows: int
    sheets_skipped: int
    warnings: list[str] = []


def _format_cell(cell, workbook, config: XlsProcessorConfig) -> str:
    """Convert an xlrd cell to its string representation.

    Parameters
    ----------
    cell:
        An xlrd Cell object.
    workbook:
        The xlrd Workbook (needed for datemode).
    config:
        Pipeline config (needed for date_format).

    Returns
    -------
    str
        The formatted cell value.
    """
    if cell.ctype == xlrd.XL_CELL_EMPTY or cell.ctype == xlrd.XL_CELL_BLANK:
        return ""
    elif cell.ctype == xlrd.XL_CELL_DATE:
        try:
            dt = xlrd.xldate_as_datetime(cell.value, workbook.datemode)
            return dt.strftime(config.date_format)
        except Exception as exc:
            logger.warning(
                "ingestkit_xls | date conversion failed: %s | falling back to str",
                exc,
            )
            return str(cell.value)
    elif cell.ctype == xlrd.XL_CELL_BOOLEAN:
        return "TRUE" if cell.value else "FALSE"
    elif cell.ctype == xlrd.XL_CELL_ERROR:
        return "#ERROR"
    else:
        # XL_CELL_TEXT or XL_CELL_NUMBER
        value = cell.value
        # Strip .0 from float values that are integers
        if isinstance(value, float) and value == int(value):
            return str(int(value))
        return str(value)


def extract_sheets(
    file_path: str, config: XlsProcessorConfig | None = None
) -> ExtractResult:
    """Extract structured text from a .xls file using xlrd.

    Opens the workbook, iterates sheets and rows, formats cells as
    pipe-separated values, and produces section-headed text suitable
    for RAG chunking.

    Parameters
    ----------
    file_path:
        Filesystem path to the .xls file.
    config:
        Pipeline config.  Uses defaults when *None*.

    Returns
    -------
    ExtractResult
        The extracted text, sheet results, word count, and metadata.

    Raises
    ------
    ImportError
        If xlrd is not installed.
    """
    if xlrd is None:
        raise ImportError(
            "xlrd is required to process .xls files. "
            "Install it with: pip install xlrd"
        )

    if config is None:
        config = XlsProcessorConfig()

    workbook = xlrd.open_workbook(file_path)
    sheet_results: list[SheetResult] = []
    total_rows = 0
    sheets_skipped = 0
    warnings: list[str] = []

    for sheet in workbook.sheets():
        if sheet.nrows == 0:
            if config.skip_empty_sheets:
                logger.debug(
                    "ingestkit_xls | skipping empty sheet: %s", sheet.name
                )
                sheets_skipped += 1
                warnings.append(
                    f"W_XLS_EMPTY_SHEET_SKIPPED: {sheet.name}"
                )
                continue

        rows_text: list[str] = []
        for row_idx in range(sheet.nrows):
            cells: list[str] = []
            for col_idx in range(sheet.ncols):
                cell = sheet.cell(row_idx, col_idx)
                cells.append(_format_cell(cell, workbook, config))
            row_text = " | ".join(cells)
            rows_text.append(row_text)

        sheet_text = "\n".join(rows_text)

        # Check if the sheet has any non-empty content
        if not sheet_text.strip():
            if config.skip_empty_sheets:
                logger.debug(
                    "ingestkit_xls | skipping empty sheet (whitespace only): %s",
                    sheet.name,
                )
                sheets_skipped += 1
                warnings.append(
                    f"W_XLS_EMPTY_SHEET_SKIPPED: {sheet.name}"
                )
                continue

        total_rows += sheet.nrows
        sheet_results.append(
            SheetResult(
                name=sheet.name,
                text=sheet_text,
                row_count=sheet.nrows,
                col_count=sheet.ncols,
            )
        )

    # Concatenate sheets with section headers
    sections: list[str] = []
    for sr in sheet_results:
        sections.append(f"## {sr.name}\n\n{sr.text}")

    full_text = "\n\n".join(sections)
    word_count = len(full_text.split()) if full_text.strip() else 0

    return ExtractResult(
        sheets=sheet_results,
        text=full_text,
        word_count=word_count,
        total_rows=total_rows,
        sheets_skipped=sheets_skipped,
        warnings=warnings,
    )


def chunk_text(text: str, config: XlsProcessorConfig) -> list[str]:
    """Split extracted text into token-sized chunks with overlap.

    Uses word-count token estimation (``len(text.split())``), consistent
    with ingestkit-doc and ingestkit-json chunkers.  Splits on paragraph
    boundaries (``\\n\\n``) first, then on single newlines for paragraphs
    that exceed ``chunk_size_tokens``.

    Parameters
    ----------
    text:
        The raw text extracted from the .xls file.
    config:
        Configuration for ``chunk_size_tokens`` and ``chunk_overlap_tokens``.

    Returns
    -------
    list[str]
        A list of text chunks ready for embedding.
    """
    if not text or not text.strip():
        return []

    chunk_size = config.chunk_size_tokens
    overlap = config.chunk_overlap_tokens

    # Split on paragraph boundaries first
    paragraphs = text.split("\n\n")

    # Further split paragraphs that exceed chunk_size on single newlines
    lines: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_tokens = len(para.split())
        if para_tokens > chunk_size:
            # Split on single newlines
            sub_lines = para.split("\n")
            for sub in sub_lines:
                sub = sub.strip()
                if sub:
                    lines.append(sub)
        else:
            lines.append(para)

    if not lines:
        return []

    # Build chunks from lines
    chunks: list[str] = []
    current_lines: list[str] = []
    current_tokens = 0

    for line in lines:
        line_tokens = len(line.split())

        # If a single line exceeds chunk_size, flush current and emit it alone
        if line_tokens > chunk_size:
            if current_lines:
                chunks.append("\n\n".join(current_lines))
            chunks.append(line)
            current_lines = []
            current_tokens = 0
            continue

        # If adding this line would exceed chunk_size, flush current chunk
        if current_tokens + line_tokens > chunk_size and current_lines:
            chunks.append("\n\n".join(current_lines))

            # Compute overlap: take trailing lines up to overlap token count
            overlap_lines: list[str] = []
            overlap_tokens = 0
            for prev_line in reversed(current_lines):
                prev_tokens = len(prev_line.split())
                if overlap_tokens + prev_tokens > overlap:
                    break
                overlap_lines.insert(0, prev_line)
                overlap_tokens += prev_tokens

            current_lines = overlap_lines
            current_tokens = overlap_tokens

        current_lines.append(line)
        current_tokens += line_tokens

    # Flush remaining
    if current_lines:
        chunks.append("\n\n".join(current_lines))

    return chunks
