"""Unit tests for TableExtractor (SPEC 11.3 steps 2-3)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ingestkit_pdf.config import PDFProcessorConfig
from ingestkit_pdf.errors import ErrorCode
from ingestkit_pdf.processors.table_extractor import TableExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_pdf(pages_tables: dict[int, list[list[list[str | None]]]]):
    """Build a mock pdfplumber PDF object.

    Args:
        pages_tables: {page_index_0based: [table1_data, table2_data, ...]}
            Each table_data is list-of-lists (first row = headers).
    """
    mock_pdf = MagicMock()
    max_idx = max(pages_tables.keys()) if pages_tables else 0
    mock_pages = []
    for i in range(max_idx + 1):
        page = MagicMock()
        page.extract_tables.return_value = pages_tables.get(i, [])
        mock_pages.append(page)
    mock_pdf.pages = mock_pages
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)
    return mock_pdf


# ---------------------------------------------------------------------------
# R-PC-2: Table Extraction Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@patch("ingestkit_pdf.processors.table_extractor.pdfplumber")
def test_single_table_small_nl_serialization(
    mock_plumber, pdf_config, mock_structured_db, mock_vector_store, mock_embedder
):
    """Table with few rows -> NL serialization path. One chunk per row."""
    table_data = [["Name", "Age"], ["Alice", "30"], ["Bob", "25"]]
    mock_plumber.open.return_value = _make_mock_pdf({0: [table_data]})

    extractor = TableExtractor(
        config=pdf_config,
        structured_db=mock_structured_db,
        vector_store=mock_vector_store,
        embedder=mock_embedder,
    )
    result = extractor.extract_tables(
        file_path="/tmp/test.pdf",
        page_numbers=[1],
        ingest_key="abc12345deadbeef",
        ingest_run_id="run-001",
    )

    assert len(result.tables) == 1
    assert result.tables[0].row_count == 2
    assert len(result.chunks) == 2  # one per row
    assert result.chunks[0].metadata.content_type == "table"
    assert result.chunks[0].metadata.table_index == 0
    assert result.chunks[0].metadata.page_numbers == [1]
    # No DB writes for small table
    assert len(result.table_names) == 0
    assert len(mock_structured_db.tables) == 0


@pytest.mark.unit
@patch("ingestkit_pdf.processors.table_extractor.pdfplumber")
def test_single_table_large_db_routing(
    mock_plumber, pdf_config, mock_structured_db, mock_vector_store, mock_embedder
):
    """Table with >20 rows -> StructuredDB path. Schema chunk created."""
    headers = ["Col_A", "Col_B"]
    rows = [[f"val_{i}_a", f"val_{i}_b"] for i in range(25)]
    table_data = [headers] + rows
    mock_plumber.open.return_value = _make_mock_pdf({0: [table_data]})

    extractor = TableExtractor(
        config=pdf_config,
        structured_db=mock_structured_db,
        vector_store=mock_vector_store,
        embedder=mock_embedder,
    )
    result = extractor.extract_tables(
        file_path="/tmp/test.pdf",
        page_numbers=[1],
        ingest_key="abc12345deadbeef",
        ingest_run_id="run-001",
    )

    assert len(result.tables) == 1
    assert result.tables[0].row_count == 25
    # DB write happened
    assert len(result.table_names) == 1
    assert len(mock_structured_db.tables) == 1
    # One schema chunk
    assert len(result.chunks) == 1
    assert result.chunks[0].metadata.content_type == "table"
    assert 'contains 25 rows' in result.chunks[0].text


@pytest.mark.unit
@patch("ingestkit_pdf.processors.table_extractor.pdfplumber")
def test_boundary_20_rows_nl_path(
    mock_plumber, pdf_config, mock_structured_db, mock_vector_store, mock_embedder
):
    """Exactly 20 rows -> NL serialization (boundary condition)."""
    headers = ["X"]
    rows = [[f"v{i}"] for i in range(20)]
    table_data = [headers] + rows
    mock_plumber.open.return_value = _make_mock_pdf({0: [table_data]})

    extractor = TableExtractor(
        config=pdf_config,
        structured_db=mock_structured_db,
        vector_store=mock_vector_store,
        embedder=mock_embedder,
    )
    result = extractor.extract_tables(
        file_path="/tmp/test.pdf",
        page_numbers=[1],
        ingest_key="abc12345deadbeef",
        ingest_run_id="run-001",
    )

    assert len(result.chunks) == 20  # one per row, NL path
    assert len(result.table_names) == 0  # no DB write
    assert len(mock_structured_db.tables) == 0


@pytest.mark.unit
@patch("ingestkit_pdf.processors.table_extractor.pdfplumber")
def test_boundary_21_rows_db_path(
    mock_plumber, pdf_config, mock_structured_db, mock_vector_store, mock_embedder
):
    """Exactly 21 rows -> StructuredDB (boundary condition)."""
    headers = ["X"]
    rows = [[f"v{i}"] for i in range(21)]
    table_data = [headers] + rows
    mock_plumber.open.return_value = _make_mock_pdf({0: [table_data]})

    extractor = TableExtractor(
        config=pdf_config,
        structured_db=mock_structured_db,
        vector_store=mock_vector_store,
        embedder=mock_embedder,
    )
    result = extractor.extract_tables(
        file_path="/tmp/test.pdf",
        page_numbers=[1],
        ingest_key="abc12345deadbeef",
        ingest_run_id="run-001",
    )

    assert len(result.chunks) == 1  # schema chunk only
    assert len(result.table_names) == 1  # DB write
    assert len(mock_structured_db.tables) == 1


@pytest.mark.unit
@patch("ingestkit_pdf.processors.table_extractor.pdfplumber")
def test_empty_table_skipped(
    mock_plumber, pdf_config, mock_structured_db, mock_vector_store, mock_embedder
):
    """Table with header only (0 data rows) -> skipped, no chunks."""
    table_data = [["Name", "Age"]]  # header only
    mock_plumber.open.return_value = _make_mock_pdf({0: [table_data]})

    extractor = TableExtractor(
        config=pdf_config,
        structured_db=mock_structured_db,
        vector_store=mock_vector_store,
        embedder=mock_embedder,
    )
    result = extractor.extract_tables(
        file_path="/tmp/test.pdf",
        page_numbers=[1],
        ingest_key="abc12345deadbeef",
        ingest_run_id="run-001",
    )

    assert len(result.tables) == 0
    assert len(result.chunks) == 0


@pytest.mark.unit
@patch("ingestkit_pdf.processors.table_extractor.pdfplumber")
def test_multiple_tables_single_page(
    mock_plumber, pdf_config, mock_structured_db, mock_vector_store, mock_embedder
):
    """Page with 2 tables -> both extracted with correct table_index."""
    table1 = [["A"], ["v1"], ["v2"]]
    table2 = [["B"], ["w1"], ["w2"], ["w3"]]
    mock_plumber.open.return_value = _make_mock_pdf({0: [table1, table2]})

    extractor = TableExtractor(
        config=pdf_config,
        structured_db=mock_structured_db,
        vector_store=mock_vector_store,
        embedder=mock_embedder,
    )
    result = extractor.extract_tables(
        file_path="/tmp/test.pdf",
        page_numbers=[1],
        ingest_key="abc12345deadbeef",
        ingest_run_id="run-001",
    )

    assert len(result.tables) == 2
    assert result.tables[0].table_index == 0
    assert result.tables[1].table_index == 1
    assert result.tables[0].row_count == 2
    assert result.tables[1].row_count == 3


@pytest.mark.unit
@patch("ingestkit_pdf.processors.table_extractor.pdfplumber")
def test_multiple_pages_no_stitching(
    mock_plumber, pdf_config, mock_structured_db, mock_vector_store, mock_embedder
):
    """Tables on different pages with different column counts -> no stitching."""
    table_p1 = [["A", "B"], ["1", "2"]]
    table_p2 = [["X", "Y", "Z"], ["a", "b", "c"]]
    mock_plumber.open.return_value = _make_mock_pdf({0: [table_p1], 1: [table_p2]})

    extractor = TableExtractor(
        config=pdf_config,
        structured_db=mock_structured_db,
        vector_store=mock_vector_store,
        embedder=mock_embedder,
    )
    result = extractor.extract_tables(
        file_path="/tmp/test.pdf",
        page_numbers=[1, 2],
        ingest_key="abc12345deadbeef",
        ingest_run_id="run-001",
    )

    assert len(result.tables) == 2
    assert result.tables[0].is_continuation is False
    assert result.tables[1].is_continuation is False
    assert result.tables[0].continuation_group_id is None


@pytest.mark.unit
@patch("ingestkit_pdf.processors.table_extractor.pdfplumber")
def test_none_headers_fallback(
    mock_plumber, pdf_config, mock_structured_db, mock_vector_store, mock_embedder
):
    """pdfplumber returns None in header cells -> replaced with column_N."""
    table_data = [[None, "Age"], ["Alice", "30"]]
    mock_plumber.open.return_value = _make_mock_pdf({0: [table_data]})

    extractor = TableExtractor(
        config=pdf_config,
        structured_db=mock_structured_db,
        vector_store=mock_vector_store,
        embedder=mock_embedder,
    )
    result = extractor.extract_tables(
        file_path="/tmp/test.pdf",
        page_numbers=[1],
        ingest_key="abc12345deadbeef",
        ingest_run_id="run-001",
    )

    assert len(result.tables) == 1
    # First header should be column_0 (fallback for None)
    assert result.tables[0].headers[0] == "column_0"


@pytest.mark.unit
@patch("ingestkit_pdf.processors.table_extractor.pdfplumber")
def test_extraction_error_per_table(
    mock_plumber, pdf_config, mock_structured_db, mock_vector_store, mock_embedder
):
    """pdfplumber raises on one page -> error recorded, other pages still processed."""
    table_p2 = [["A"], ["v1"]]

    mock_pdf = MagicMock()
    page1 = MagicMock()
    page1.extract_tables.side_effect = RuntimeError("corrupt page")
    page2 = MagicMock()
    page2.extract_tables.return_value = [table_p2]
    mock_pdf.pages = [page1, page2]
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)
    mock_plumber.open.return_value = mock_pdf

    extractor = TableExtractor(
        config=pdf_config,
        structured_db=mock_structured_db,
        vector_store=mock_vector_store,
        embedder=mock_embedder,
    )
    result = extractor.extract_tables(
        file_path="/tmp/test.pdf",
        page_numbers=[1, 2],
        ingest_key="abc12345deadbeef",
        ingest_run_id="run-001",
    )

    # Page 1 errored, page 2 succeeded
    assert len(result.errors) == 1
    assert result.errors[0].code == ErrorCode.E_PROCESS_TABLE_EXTRACT
    assert result.errors[0].page_number == 1
    assert len(result.tables) == 1
    assert result.tables[0].page_number == 2


@pytest.mark.unit
@patch("ingestkit_pdf.processors.table_extractor.pdfplumber")
def test_no_backends_pure_extraction(mock_plumber, pdf_config):
    """All backends None -> tables extracted, TableResult populated, no chunks."""
    table_data = [["Name", "Age"], ["Alice", "30"]]
    mock_plumber.open.return_value = _make_mock_pdf({0: [table_data]})

    extractor = TableExtractor(
        config=pdf_config,
        structured_db=None,
        vector_store=None,
        embedder=None,
    )
    result = extractor.extract_tables(
        file_path="/tmp/test.pdf",
        page_numbers=[1],
        ingest_key="abc12345deadbeef",
        ingest_run_id="run-001",
    )

    assert len(result.tables) == 1
    assert result.tables[0].row_count == 1
    assert result.tables[0].headers is not None
    # Chunks are created but vectors remain empty (no embedding backend)
    assert len(result.chunks) == 1
    assert result.chunks[0].vector == []
    assert result.texts_embedded == 0


# ---------------------------------------------------------------------------
# R-PC-3: Multi-Page Table Stitching Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@patch("ingestkit_pdf.processors.table_extractor.pdfplumber")
def test_continuation_stitching_basic(
    mock_plumber, pdf_config, mock_structured_db, mock_vector_store, mock_embedder
):
    """Same headers on page 1 and page 2 -> stitched into one table."""
    table_p1 = [["Name", "Age"], ["Alice", "30"], ["Bob", "25"]]
    table_p2 = [["Name", "Age"], ["Carol", "35"], ["Dave", "40"]]
    mock_plumber.open.return_value = _make_mock_pdf({0: [table_p1], 1: [table_p2]})

    extractor = TableExtractor(
        config=pdf_config,
        structured_db=mock_structured_db,
        vector_store=mock_vector_store,
        embedder=mock_embedder,
    )
    result = extractor.extract_tables(
        file_path="/tmp/test.pdf",
        page_numbers=[1, 2],
        ingest_key="abc12345deadbeef",
        ingest_run_id="run-001",
    )

    assert len(result.tables) == 1
    assert result.tables[0].is_continuation is True
    assert result.tables[0].continuation_group_id is not None
    assert ErrorCode.W_TABLE_CONTINUATION.value in result.warnings
    # Should have stitched: 2 + 2 = 4 rows (continuation header skipped)
    assert result.tables[0].row_count == 4


@pytest.mark.unit
@patch("ingestkit_pdf.processors.table_extractor.pdfplumber")
def test_continuation_skip_repeated_header(
    mock_plumber, pdf_config, mock_structured_db, mock_vector_store, mock_embedder
):
    """Continuation table starts with repeated header row -> skipped in concat."""
    table_p1 = [["Name", "Age"], ["Alice", "30"]]
    # Page 2 has repeated header as first data row
    table_p2 = [["Name", "Age"], ["Name", "Age"], ["Bob", "25"]]
    mock_plumber.open.return_value = _make_mock_pdf({0: [table_p1], 1: [table_p2]})

    extractor = TableExtractor(
        config=pdf_config,
        structured_db=mock_structured_db,
        vector_store=mock_vector_store,
        embedder=mock_embedder,
    )
    result = extractor.extract_tables(
        file_path="/tmp/test.pdf",
        page_numbers=[1, 2],
        ingest_key="abc12345deadbeef",
        ingest_run_id="run-001",
    )

    assert len(result.tables) == 1
    # 1 from page 1 + 1 from page 2 (repeated header skipped) = 2
    assert result.tables[0].row_count == 2


@pytest.mark.unit
@patch("ingestkit_pdf.processors.table_extractor.pdfplumber")
def test_continuation_below_threshold(
    mock_plumber, pdf_config, mock_structured_db, mock_vector_store, mock_embedder
):
    """Header similarity below 0.8 -> NOT stitched."""
    # Same column count but very different header text
    table_p1 = [["Name", "Age"], ["Alice", "30"]]
    table_p2 = [["City", "Population"], ["NYC", "8000000"]]
    mock_plumber.open.return_value = _make_mock_pdf({0: [table_p1], 1: [table_p2]})

    extractor = TableExtractor(
        config=pdf_config,
        structured_db=mock_structured_db,
        vector_store=mock_vector_store,
        embedder=mock_embedder,
    )
    result = extractor.extract_tables(
        file_path="/tmp/test.pdf",
        page_numbers=[1, 2],
        ingest_key="abc12345deadbeef",
        ingest_run_id="run-001",
    )

    assert len(result.tables) == 2
    assert result.tables[0].is_continuation is False
    assert result.tables[1].is_continuation is False


@pytest.mark.unit
@patch("ingestkit_pdf.processors.table_extractor.pdfplumber")
def test_continuation_column_count_mismatch(
    mock_plumber, pdf_config, mock_structured_db, mock_vector_store, mock_embedder
):
    """Different column counts -> NOT stitched even if header text similar."""
    table_p1 = [["Name", "Age"], ["Alice", "30"]]
    table_p2 = [["Name", "Age", "City"], ["Bob", "25", "NYC"]]
    mock_plumber.open.return_value = _make_mock_pdf({0: [table_p1], 1: [table_p2]})

    extractor = TableExtractor(
        config=pdf_config,
        structured_db=mock_structured_db,
        vector_store=mock_vector_store,
        embedder=mock_embedder,
    )
    result = extractor.extract_tables(
        file_path="/tmp/test.pdf",
        page_numbers=[1, 2],
        ingest_key="abc12345deadbeef",
        ingest_run_id="run-001",
    )

    assert len(result.tables) == 2
    assert result.tables[0].is_continuation is False
    assert result.tables[1].is_continuation is False


@pytest.mark.unit
@patch("ingestkit_pdf.processors.table_extractor.pdfplumber")
def test_continuation_three_pages(
    mock_plumber, pdf_config, mock_structured_db, mock_vector_store, mock_embedder
):
    """Table spans pages 1, 2, 3 -> all stitched into one."""
    table_p1 = [["ID", "Val"], ["1", "a"], ["2", "b"]]
    table_p2 = [["ID", "Val"], ["3", "c"], ["4", "d"]]
    table_p3 = [["ID", "Val"], ["5", "e"], ["6", "f"]]
    mock_plumber.open.return_value = _make_mock_pdf(
        {0: [table_p1], 1: [table_p2], 2: [table_p3]}
    )

    extractor = TableExtractor(
        config=pdf_config,
        structured_db=mock_structured_db,
        vector_store=mock_vector_store,
        embedder=mock_embedder,
    )
    result = extractor.extract_tables(
        file_path="/tmp/test.pdf",
        page_numbers=[1, 2, 3],
        ingest_key="abc12345deadbeef",
        ingest_run_id="run-001",
    )

    assert len(result.tables) == 1
    assert result.tables[0].is_continuation is True
    assert result.tables[0].continuation_group_id is not None
    # 2 + 2 + 2 = 6 rows (repeated headers on p2 and p3 skipped)
    assert result.tables[0].row_count == 6
    # Two W_TABLE_CONTINUATION warnings (p1->p2 and p2->p3)
    continuation_warnings = [
        w for w in result.warnings if w == ErrorCode.W_TABLE_CONTINUATION.value
    ]
    assert len(continuation_warnings) == 2


@pytest.mark.unit
@patch("ingestkit_pdf.processors.table_extractor.pdfplumber")
def test_stitched_table_routing(
    mock_plumber, pdf_config, mock_structured_db, mock_vector_store, mock_embedder
):
    """Stitched table with 30 total rows (15 per page) -> DB path. Multi-page metadata."""
    headers = ["Col"]
    rows_p1 = [[f"v{i}"] for i in range(15)]
    rows_p2 = [[f"v{i}"] for i in range(15, 30)]
    table_p1 = [headers] + rows_p1
    table_p2 = [headers] + rows_p2
    mock_plumber.open.return_value = _make_mock_pdf({0: [table_p1], 1: [table_p2]})

    extractor = TableExtractor(
        config=pdf_config,
        structured_db=mock_structured_db,
        vector_store=mock_vector_store,
        embedder=mock_embedder,
    )
    result = extractor.extract_tables(
        file_path="/tmp/test.pdf",
        page_numbers=[1, 2],
        ingest_key="abc12345deadbeef",
        ingest_run_id="run-001",
    )

    assert len(result.tables) == 1
    assert result.tables[0].row_count == 30
    # DB path (> 20 rows)
    assert len(result.table_names) == 1
    assert len(mock_structured_db.tables) == 1
    # Schema chunk has multi-page metadata
    assert len(result.chunks) == 1
    assert result.chunks[0].metadata.page_numbers == [1, 2]
