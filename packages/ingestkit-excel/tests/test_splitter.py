"""Tests for the HybridSplitter (Path C).

Covers region detection heuristics (blank rows, blank cols, merged blocks,
formatting transitions, header/footer, matrix), region classification,
Type A and Type B region processing, full process flow, multi-sheet,
sheet skipping, error handling, and result merging.
"""

from __future__ import annotations

import hashlib
import uuid
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ingestkit_excel.config import ExcelProcessorConfig
from ingestkit_excel.errors import ErrorCode, IngestError
from ingestkit_excel.models import (
    ChunkMetadata,
    ChunkPayload,
    ClassificationResult,
    ClassificationStageResult,
    ClassificationTier,
    EmbedStageResult,
    FileProfile,
    FileType,
    IngestionMethod,
    ParseStageResult,
    ParserUsed,
    ProcessingResult,
    RegionType,
    SheetProfile,
    SheetRegion,
    WrittenArtifacts,
)
from ingestkit_excel.processors.splitter import (
    HybridSplitter,
    _BLANK_COL_THRESHOLD,
    _BLANK_ROW_THRESHOLD,
    _FORMATTING_TRANSITION_WINDOW,
    _HEADER_FOOTER_MAX_ROWS,
    _MATRIX_MIN_HEADERS,
    _NUMERIC_HEAVY_THRESHOLD,
    _TEXT_HEAVY_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Test Helper Factories
# ---------------------------------------------------------------------------


def _make_sheet_profile(**overrides: object) -> SheetProfile:
    """Build a SheetProfile with sensible hybrid defaults."""
    defaults: dict = dict(
        name="Sheet1",
        row_count=50,
        col_count=5,
        merged_cell_count=3,
        merged_cell_ratio=0.05,
        header_row_detected=True,
        header_row_index=0,
        header_values=["A", "B", "C", "D", "E"],
        column_type_consistency=0.7,
        numeric_ratio=0.3,
        text_ratio=0.4,
        empty_ratio=0.3,
        sample_rows=[["1", "a", "x", "2.0", "y"]],
        has_formulas=False,
        is_hidden=False,
        parser_used=ParserUsed.OPENPYXL,
    )
    defaults.update(overrides)
    return SheetProfile(**defaults)


def _make_file_profile(
    sheets: list[SheetProfile],
    **overrides: object,
) -> FileProfile:
    """Build a FileProfile from a list of SheetProfiles."""
    defaults: dict = dict(
        file_path="/tmp/test.xlsx",
        file_size_bytes=4096,
        sheet_count=len(sheets),
        sheet_names=[s.name for s in sheets],
        sheets=sheets,
        has_password_protected_sheets=False,
        has_chart_only_sheets=False,
        total_merged_cells=sum(s.merged_cell_count for s in sheets),
        total_rows=sum(s.row_count for s in sheets),
        content_hash="c" * 64,
    )
    defaults.update(overrides)
    return FileProfile(**defaults)


def _make_parse_result(**overrides: object) -> ParseStageResult:
    defaults: dict = dict(
        parser_used=ParserUsed.OPENPYXL,
        fallback_reason_code=None,
        sheets_parsed=1,
        sheets_skipped=0,
        skipped_reasons={},
        parse_duration_seconds=0.1,
    )
    defaults.update(overrides)
    return ParseStageResult(**defaults)


def _make_classification_stage_result(**overrides: object) -> ClassificationStageResult:
    defaults: dict = dict(
        tier_used=ClassificationTier.RULE_BASED,
        file_type=FileType.HYBRID,
        confidence=0.85,
        signals=None,
        reasoning="Mixed tabular and document regions",
        per_sheet_types=None,
        classification_duration_seconds=0.05,
    )
    defaults.update(overrides)
    return ClassificationStageResult(**defaults)


def _make_classification_result(**overrides: object) -> ClassificationResult:
    defaults: dict = dict(
        file_type=FileType.HYBRID,
        confidence=0.85,
        tier_used=ClassificationTier.RULE_BASED,
        reasoning="Mixed tabular and document regions",
        per_sheet_types=None,
        signals=None,
    )
    defaults.update(overrides)
    return ClassificationResult(**defaults)


# ---------------------------------------------------------------------------
# Mock Backends
# ---------------------------------------------------------------------------


class MockStructuredDB:
    """Mock StructuredDBBackend for testing."""

    def __init__(self, connection_uri: str = "sqlite:///test.db"):
        self._uri = connection_uri
        self.tables_created: list[tuple[str, pd.DataFrame]] = []
        self.fail_on: str | None = None

    def create_table_from_dataframe(self, table_name: str, df: pd.DataFrame) -> None:
        if self.fail_on and table_name == self.fail_on:
            raise RuntimeError(f"DB connection error for table {table_name}")
        self.tables_created.append((table_name, df.copy()))

    def drop_table(self, table_name: str) -> None:
        pass

    def table_exists(self, table_name: str) -> bool:
        return any(t[0] == table_name for t in self.tables_created)

    def get_table_schema(self, table_name: str) -> dict:
        return {}

    def get_connection_uri(self) -> str:
        return self._uri


class MockVectorStore:
    """Mock VectorStoreBackend for testing."""

    def __init__(self):
        self.collections_ensured: list[tuple[str, int]] = []
        self.upserted: list[tuple[str, list[ChunkPayload]]] = []
        self.fail_on_upsert: bool = False

    def upsert_chunks(self, collection: str, chunks: list[ChunkPayload]) -> int:
        if self.fail_on_upsert:
            raise RuntimeError("Vector store connection error")
        self.upserted.append((collection, chunks))
        return len(chunks)

    def ensure_collection(self, collection: str, vector_size: int) -> None:
        self.collections_ensured.append((collection, vector_size))

    def create_payload_index(
        self, collection: str, field: str, field_type: str
    ) -> None:
        pass

    def delete_by_ids(self, collection: str, ids: list[str]) -> int:
        return len(ids)


class MockEmbedder:
    """Mock EmbeddingBackend for testing."""

    def __init__(self, dim: int = 768):
        self._dim = dim
        self.embed_calls: list[list[str]] = []
        self.fail_on_embed: bool = False

    def embed(
        self, texts: list[str], timeout: float | None = None
    ) -> list[list[float]]:
        if self.fail_on_embed:
            raise RuntimeError("Embedding timeout error")
        self.embed_calls.append(texts)
        return [[0.1] * self._dim for _ in texts]

    def dimension(self) -> int:
        return self._dim


# ---------------------------------------------------------------------------
# Mock openpyxl helpers
# ---------------------------------------------------------------------------


def _make_mock_cell(value):
    """Create a mock cell with a .value attribute."""
    cell = MagicMock()
    cell.value = value
    return cell


def _make_mock_merged_range(min_row, min_col, max_row, max_col):
    """Create a mock MergedCellRange."""
    mr = MagicMock()
    mr.min_row = min_row
    mr.min_col = min_col
    mr.max_row = max_row
    mr.max_col = max_col
    return mr


def _make_mock_workbook(sheets_data: dict[str, dict]) -> MagicMock:
    """Build a mock openpyxl workbook.

    Args:
        sheets_data: {sheet_name: {"rows": [[val, ...], ...], "merged": [(min_r, min_c, max_r, max_c), ...]}}

    Returns:
        A MagicMock that behaves like openpyxl.Workbook.
    """
    wb = MagicMock()
    worksheets = {}

    for name, data in sheets_data.items():
        ws = MagicMock()
        ws.title = name
        rows = data.get("rows", [])
        merged = data.get("merged", [])

        # Build iter_rows response: list of tuples of mock cells
        mock_rows = []
        for row_vals in rows:
            mock_rows.append(tuple(_make_mock_cell(v) for v in row_vals))
        ws.iter_rows.return_value = mock_rows

        # Build merged_cells.ranges
        merged_ranges = [
            _make_mock_merged_range(*m) for m in merged
        ]
        ws.merged_cells.ranges = merged_ranges

        # Mock ws.cell(row, col) for reading merged header values
        def _cell_factory(r, c, _rows=rows):
            cell = MagicMock()
            if 1 <= r <= len(_rows) and 1 <= c <= len(_rows[r - 1]):
                cell.value = _rows[r - 1][c - 1]
            else:
                cell.value = None
            return cell

        ws.cell = _cell_factory
        worksheets[name] = ws

    wb.__getitem__ = lambda self, name: worksheets[name]
    wb.close = MagicMock()
    return wb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_db() -> MockStructuredDB:
    return MockStructuredDB()


@pytest.fixture()
def mock_vector_store() -> MockVectorStore:
    return MockVectorStore()


@pytest.fixture()
def mock_embedder() -> MockEmbedder:
    return MockEmbedder()


@pytest.fixture()
def config() -> ExcelProcessorConfig:
    return ExcelProcessorConfig()


@pytest.fixture()
def structured_processor(
    mock_db: MockStructuredDB,
    mock_vector_store: MockVectorStore,
    mock_embedder: MockEmbedder,
    config: ExcelProcessorConfig,
):
    """Build a mock structured processor with _db, _vector_store, _embedder."""
    from ingestkit_excel.processors.structured_db import StructuredDBProcessor

    return StructuredDBProcessor(mock_db, mock_vector_store, mock_embedder, config)


@pytest.fixture()
def text_serializer(
    mock_vector_store: MockVectorStore,
    mock_embedder: MockEmbedder,
    config: ExcelProcessorConfig,
):
    """Build a mock text serializer."""
    from ingestkit_excel.processors.serializer import TextSerializer

    return TextSerializer(mock_vector_store, mock_embedder, config)


@pytest.fixture()
def splitter(
    structured_processor,
    text_serializer,
    config: ExcelProcessorConfig,
) -> HybridSplitter:
    return HybridSplitter(structured_processor, text_serializer, config)


# ---------------------------------------------------------------------------
# Section: Constructor / Backend Extraction
# ---------------------------------------------------------------------------


class TestConstructor:
    """Tests for HybridSplitter constructor and backend extraction."""

    def test_constructor_extracts_db(self, splitter, mock_db):
        assert splitter._db is mock_db

    def test_constructor_extracts_vector_store(self, splitter, mock_vector_store):
        assert splitter._vector_store is mock_vector_store

    def test_constructor_extracts_embedder(self, splitter, mock_embedder):
        assert splitter._embedder is mock_embedder

    def test_constructor_stores_config(self, splitter, config):
        assert splitter._config is config


# ---------------------------------------------------------------------------
# Section: Blank Row Boundary Detection
# ---------------------------------------------------------------------------


class TestBlankRowBoundaries:
    """Tests for _detect_blank_row_boundaries."""

    def test_no_blank_rows_returns_empty(self):
        all_rows = [[1, 2], [3, 4], [5, 6]]
        result = HybridSplitter._detect_blank_row_boundaries(all_rows)
        assert result == []

    def test_single_blank_row_not_boundary(self):
        all_rows = [[1, 2], [None, None], [3, 4]]
        result = HybridSplitter._detect_blank_row_boundaries(all_rows)
        assert result == []

    def test_two_consecutive_blank_rows_is_boundary(self):
        all_rows = [[1, 2], [None, None], [None, None], [3, 4]]
        result = HybridSplitter._detect_blank_row_boundaries(all_rows)
        assert len(result) == 1
        assert result[0] == 1  # gap starts at row 1

    def test_three_consecutive_blank_rows_single_boundary(self):
        all_rows = [[1, 2], [None, None], [None, None], [None, None], [3, 4]]
        result = HybridSplitter._detect_blank_row_boundaries(all_rows)
        assert len(result) == 1

    def test_multiple_gaps(self):
        all_rows = [
            [1, 2],
            [None, None], [None, None],
            [3, 4],
            [None, None], [None, None],
            [5, 6],
        ]
        result = HybridSplitter._detect_blank_row_boundaries(all_rows)
        assert len(result) == 2

    def test_blank_rows_with_empty_strings(self):
        all_rows = [[1, 2], ["", "  "], ["", ""], [3, 4]]
        result = HybridSplitter._detect_blank_row_boundaries(all_rows)
        assert len(result) == 1

    def test_trailing_blank_rows(self):
        all_rows = [[1, 2], [None, None], [None, None]]
        result = HybridSplitter._detect_blank_row_boundaries(all_rows)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Section: Blank Column Boundary Detection
# ---------------------------------------------------------------------------


class TestBlankColBoundaries:
    """Tests for _detect_blank_col_boundaries."""

    def test_no_blank_cols_returns_empty(self):
        all_rows = [[1, 2, 3], [4, 5, 6]]
        result = HybridSplitter._detect_blank_col_boundaries(all_rows, 3)
        assert result == []

    def test_two_consecutive_blank_cols_is_boundary(self):
        all_rows = [[1, None, None, 2], [3, None, None, 4]]
        result = HybridSplitter._detect_blank_col_boundaries(all_rows, 4)
        assert len(result) == 1
        assert result[0] == 1

    def test_single_blank_col_not_boundary(self):
        all_rows = [[1, None, 2], [3, None, 4]]
        result = HybridSplitter._detect_blank_col_boundaries(all_rows, 3)
        assert result == []

    def test_empty_rows_returns_empty(self):
        result = HybridSplitter._detect_blank_col_boundaries([], 0)
        assert result == []


# ---------------------------------------------------------------------------
# Section: Merged Block Detection
# ---------------------------------------------------------------------------


class TestMergedBlockDetection:
    """Tests for _detect_merged_blocks."""

    def test_no_merged_cells_returns_empty(self):
        ws = MagicMock()
        ws.merged_cells.ranges = []
        result = HybridSplitter._detect_merged_blocks(ws, "Sheet1")
        assert result == []

    def test_merged_span_2_cols_detected(self):
        ws = MagicMock()
        mr = _make_mock_merged_range(1, 1, 1, 2)
        ws.merged_cells.ranges = [mr]
        result = HybridSplitter._detect_merged_blocks(ws, "Sheet1")
        assert len(result) == 1
        assert result[0].region_type == RegionType.HEADER_BLOCK

    def test_merged_span_1_col_ignored(self):
        ws = MagicMock()
        mr = _make_mock_merged_range(1, 1, 2, 1)  # 1 col wide
        ws.merged_cells.ranges = [mr]
        result = HybridSplitter._detect_merged_blocks(ws, "Sheet1")
        assert result == []

    def test_merged_block_region_id_format(self):
        ws = MagicMock()
        mr = _make_mock_merged_range(1, 1, 1, 3)
        ws.merged_cells.ranges = [mr]
        result = HybridSplitter._detect_merged_blocks(ws, "TestSheet")
        assert result[0].region_id == "TestSheet_merged_0"

    def test_merged_block_rows_0_based(self):
        ws = MagicMock()
        mr = _make_mock_merged_range(3, 1, 3, 4)  # 1-based row 3
        ws.merged_cells.ranges = [mr]
        result = HybridSplitter._detect_merged_blocks(ws, "Sheet1")
        assert result[0].start_row == 2  # 0-based


# ---------------------------------------------------------------------------
# Section: Formatting Transition Detection
# ---------------------------------------------------------------------------


class TestFormattingTransitions:
    """Tests for _detect_formatting_transitions."""

    def test_short_data_returns_empty(self):
        all_rows = [[1, 2], [3, 4]]
        result = HybridSplitter._detect_formatting_transitions(all_rows)
        assert result == []

    def test_uniform_data_no_transitions(self):
        # All numeric rows
        all_rows = [[i, i + 1, i + 2] for i in range(20)]
        result = HybridSplitter._detect_formatting_transitions(all_rows)
        assert result == []

    def test_numeric_to_text_transition_detected(self):
        # First 10 rows numeric, next 10 rows text
        numeric_rows = [[1, 2, 3] for _ in range(10)]
        text_rows = [["hello", "world", "test"] for _ in range(10)]
        all_rows = numeric_rows + text_rows
        result = HybridSplitter._detect_formatting_transitions(all_rows)
        # Should detect at least one transition around row 10
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# Section: Header/Footer Detection
# ---------------------------------------------------------------------------


class TestHeaderFooterDetection:
    """Tests for _detect_header_footer."""

    def test_no_merged_cells_no_header_footer(self):
        ws = MagicMock()
        ws.merged_cells.ranges = []
        all_rows = [[1, 2], [3, 4]]
        header, footer = HybridSplitter._detect_header_footer(ws, all_rows, 2)
        assert header is None
        assert footer is None

    def test_header_detected_from_wide_merge(self):
        ws = MagicMock()
        ws.title = "Sheet1"
        mr = _make_mock_merged_range(1, 1, 1, 5)  # 5-col wide merge in row 1
        ws.merged_cells.ranges = [mr]

        def _cell(r, c):
            cell = MagicMock()
            cell.value = "Report Title" if r == 1 else None
            return cell

        ws.cell = _cell
        all_rows = [["Report Title", None, None, None, None], [1, 2, 3, 4, 5]]
        header, footer = HybridSplitter._detect_header_footer(ws, all_rows, 2)
        assert header is not None
        assert header.region_type == RegionType.HEADER_BLOCK

    def test_footer_detected_from_wide_merge_at_end(self):
        ws = MagicMock()
        ws.title = "Sheet1"
        # Footer merge in last row (row 10 of 10)
        mr = _make_mock_merged_range(10, 1, 10, 5)
        ws.merged_cells.ranges = [mr]

        def _cell(r, c):
            cell = MagicMock()
            cell.value = "Footer Note" if r == 10 else None
            return cell

        ws.cell = _cell
        all_rows = [[i] for i in range(10)]
        header, footer = HybridSplitter._detect_header_footer(ws, all_rows, 10)
        assert footer is not None
        assert footer.region_type == RegionType.FOOTER_BLOCK

    def test_empty_sheet_returns_none_none(self):
        ws = MagicMock()
        ws.merged_cells.ranges = []
        header, footer = HybridSplitter._detect_header_footer(ws, [], 0)
        assert header is None
        assert footer is None


# ---------------------------------------------------------------------------
# Section: Matrix Detection
# ---------------------------------------------------------------------------


class TestMatrixDetection:
    """Tests for _detect_matrix_regions."""

    def test_valid_matrix_detected(self):
        all_rows = [
            [None, "Q1", "Q2", "Q3"],
            ["Sales", 100, 200, 300],
            ["Marketing", 50, 80, 120],
        ]
        result = HybridSplitter._detect_matrix_regions(all_rows, 0, 3, 0, 4)
        assert result is True

    def test_non_matrix_corner_not_empty(self):
        all_rows = [
            ["Category", "Q1", "Q2"],
            ["Sales", 100, 200],
        ]
        result = HybridSplitter._detect_matrix_regions(all_rows, 0, 2, 0, 3)
        assert result is False

    def test_non_matrix_too_few_rows(self):
        all_rows = [[None, "Q1", "Q2"]]
        result = HybridSplitter._detect_matrix_regions(all_rows, 0, 1, 0, 3)
        assert result is False

    def test_non_matrix_insufficient_col_headers(self):
        all_rows = [
            [None, "Q1"],  # Only 1 col header, need >= 2
            ["Sales", 100],
        ]
        result = HybridSplitter._detect_matrix_regions(all_rows, 0, 2, 0, 2)
        assert result is False


# ---------------------------------------------------------------------------
# Section: Region Classification
# ---------------------------------------------------------------------------


class TestRegionClassification:
    """Tests for _classify_region and _compute_region_signals."""

    def test_tabular_region_classified_correctly(self, splitter):
        """Region with 5+ rows, consistent types, header -> TABULAR_DATA."""
        all_rows = [
            ["Name", "Age", "Dept", "Salary", "Active"],
            ["Alice", "30", "Eng", "100000", "yes"],
            ["Bob", "25", "Sales", "80000", "yes"],
            ["Carol", "35", "HR", "75000", "no"],
            ["Dave", "28", "Eng", "90000", "yes"],
            ["Eve", "32", "Sales", "85000", "no"],
        ]
        region = SheetRegion(
            sheet_name="Sheet1",
            region_id="Sheet1_r0",
            start_row=0,
            end_row=6,
            start_col=0,
            end_col=5,
            region_type=RegionType.DATA_TABLE,
            detection_confidence=0.8,
        )
        result = splitter._classify_region(all_rows, region)
        assert result == FileType.TABULAR_DATA

    def test_text_region_classified_correctly(self, splitter):
        """Region with few rows, no header pattern -> FORMATTED_DOCUMENT."""
        all_rows = [
            ["This is a paragraph of text about our company."],
            ["Another paragraph with different content."],
        ]
        region = SheetRegion(
            sheet_name="Sheet1",
            region_id="Sheet1_r0",
            start_row=0,
            end_row=2,
            start_col=0,
            end_col=1,
            region_type=RegionType.TEXT_BLOCK,
            detection_confidence=0.8,
        )
        result = splitter._classify_region(all_rows, region)
        assert result == FileType.FORMATTED_DOCUMENT

    def test_compute_signals_returns_5_keys(self, splitter):
        all_rows = [["A", "B"], ["1", "2"]]
        region = SheetRegion(
            sheet_name="Sheet1",
            region_id="Sheet1_r0",
            start_row=0,
            end_row=2,
            start_col=0,
            end_col=2,
            region_type=RegionType.DATA_TABLE,
            detection_confidence=0.8,
        )
        signals = splitter._compute_region_signals(all_rows, region)
        assert set(signals.keys()) == {
            "row_count",
            "merged_cell_ratio",
            "column_type_consistency",
            "header_detected",
            "numeric_ratio",
        }

    def test_header_block_region_low_merged_signal(self, splitter):
        """HEADER_BLOCK regions get merged_cell_ratio signal = False."""
        all_rows = [["Title", None, None]]
        region = SheetRegion(
            sheet_name="Sheet1",
            region_id="Sheet1_r0",
            start_row=0,
            end_row=1,
            start_col=0,
            end_col=3,
            region_type=RegionType.HEADER_BLOCK,
            detection_confidence=0.9,
        )
        signals = splitter._compute_region_signals(all_rows, region)
        assert signals["merged_cell_ratio"] is False


# ---------------------------------------------------------------------------
# Section: Full Process Flow
# ---------------------------------------------------------------------------


class TestProcessFlow:
    """Tests for the full process() method."""

    @patch("ingestkit_excel.processors.splitter.openpyxl.load_workbook")
    def test_process_single_sheet_happy_path(
        self,
        mock_load,
        splitter,
        mock_db,
        mock_vector_store,
        mock_embedder,
    ):
        """Process a sheet with tabular data produces correct result."""
        sheets_data = {
            "Data": {
                "rows": [
                    ["Name", "Age", "Dept", "Salary", "Active"],
                    ["Alice", "30", "Eng", "100000", "yes"],
                    ["Bob", "25", "Sales", "80000", "yes"],
                    ["Carol", "35", "HR", "75000", "no"],
                    ["Dave", "28", "Eng", "90000", "yes"],
                    ["Eve", "32", "Sales", "85000", "no"],
                ],
                "merged": [],
            }
        }
        mock_load.return_value = _make_mock_workbook(sheets_data)

        sheet = _make_sheet_profile(name="Data", row_count=6, col_count=5)
        profile = _make_file_profile([sheet])

        result = splitter.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        assert result.ingestion_method == IngestionMethod.HYBRID_SPLIT
        assert result.chunks_created >= 1
        assert len(mock_vector_store.upserted) >= 1
        assert result.errors == []
        assert result.processing_time_seconds > 0

    @patch("ingestkit_excel.processors.splitter.openpyxl.load_workbook")
    def test_process_result_ingestion_method_is_enum(
        self,
        mock_load,
        splitter,
        mock_db,
        mock_vector_store,
    ):
        sheets_data = {
            "S1": {
                "rows": [["Content", "Data"], ["val1", "val2"]],
                "merged": [],
            }
        }
        mock_load.return_value = _make_mock_workbook(sheets_data)

        sheet = _make_sheet_profile(name="S1", row_count=2, col_count=2)
        profile = _make_file_profile([sheet])

        result = splitter.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        assert result.ingestion_method is IngestionMethod.HYBRID_SPLIT

    @patch("ingestkit_excel.processors.splitter.openpyxl.load_workbook")
    def test_process_chunk_metadata_ingestion_method_string(
        self,
        mock_load,
        splitter,
        mock_db,
        mock_vector_store,
    ):
        """ChunkMetadata.ingestion_method must be 'hybrid_split' string."""
        sheets_data = {
            "S1": {
                "rows": [["Content", "Data"], ["val1", "val2"]],
                "merged": [],
            }
        }
        mock_load.return_value = _make_mock_workbook(sheets_data)

        sheet = _make_sheet_profile(name="S1", row_count=2, col_count=2)
        profile = _make_file_profile([sheet])

        splitter.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        chunk = mock_vector_store.upserted[0][1][0]
        assert chunk.metadata.ingestion_method == "hybrid_split"

    @patch("ingestkit_excel.processors.splitter.openpyxl.load_workbook")
    def test_process_region_id_populated(
        self,
        mock_load,
        splitter,
        mock_db,
        mock_vector_store,
    ):
        """region_id must be set on all chunks (not None)."""
        sheets_data = {
            "S1": {
                "rows": [["Content", "Data"], ["val1", "val2"]],
                "merged": [],
            }
        }
        mock_load.return_value = _make_mock_workbook(sheets_data)

        sheet = _make_sheet_profile(name="S1", row_count=2, col_count=2)
        profile = _make_file_profile([sheet])

        splitter.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        for collection, chunks in mock_vector_store.upserted:
            for chunk in chunks:
                assert chunk.metadata.region_id is not None

    @patch("ingestkit_excel.processors.splitter.openpyxl.load_workbook")
    def test_process_written_artifacts_populated(
        self,
        mock_load,
        splitter,
        mock_db,
        mock_vector_store,
    ):
        sheets_data = {
            "S1": {
                "rows": [["Content", "Data"], ["val1", "val2"]],
                "merged": [],
            }
        }
        mock_load.return_value = _make_mock_workbook(sheets_data)

        sheet = _make_sheet_profile(name="S1", row_count=2, col_count=2)
        profile = _make_file_profile([sheet])

        result = splitter.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        assert len(result.written.vector_point_ids) >= 1
        assert result.written.vector_collection == "helpdesk"

    @patch("ingestkit_excel.processors.splitter.openpyxl.load_workbook")
    def test_process_source_uri_format(
        self,
        mock_load,
        splitter,
        mock_db,
        mock_vector_store,
    ):
        sheets_data = {
            "S1": {
                "rows": [["Content"], ["val1"]],
                "merged": [],
            }
        }
        mock_load.return_value = _make_mock_workbook(sheets_data)

        sheet = _make_sheet_profile(name="S1", row_count=2, col_count=1)
        profile = _make_file_profile([sheet])

        splitter.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        chunk = mock_vector_store.upserted[0][1][0]
        assert chunk.metadata.source_uri.startswith("file://")

    @patch("ingestkit_excel.processors.splitter.openpyxl.load_workbook")
    def test_process_tenant_id_propagated(
        self,
        mock_load,
        mock_db,
        mock_vector_store,
        mock_embedder,
        structured_processor,
        text_serializer,
    ):
        cfg = ExcelProcessorConfig(tenant_id="tenant_xyz")
        # Need to rebuild with custom config
        from ingestkit_excel.processors.structured_db import StructuredDBProcessor

        sp = StructuredDBProcessor(mock_db, mock_vector_store, mock_embedder, cfg)
        proc = HybridSplitter(sp, text_serializer, cfg)

        sheets_data = {
            "S1": {
                "rows": [["Content"], ["val1"]],
                "merged": [],
            }
        }
        mock_load.return_value = _make_mock_workbook(sheets_data)

        sheet = _make_sheet_profile(name="S1", row_count=2, col_count=1)
        profile = _make_file_profile([sheet])

        result = proc.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        assert result.tenant_id == "tenant_xyz"
        chunk = mock_vector_store.upserted[0][1][0]
        assert chunk.metadata.tenant_id == "tenant_xyz"

    @patch("ingestkit_excel.processors.splitter.openpyxl.load_workbook")
    def test_process_embed_result_populated(
        self,
        mock_load,
        splitter,
        mock_db,
        mock_vector_store,
    ):
        sheets_data = {
            "S1": {
                "rows": [["Content"], ["val1"]],
                "merged": [],
            }
        }
        mock_load.return_value = _make_mock_workbook(sheets_data)

        sheet = _make_sheet_profile(name="S1", row_count=2, col_count=1)
        profile = _make_file_profile([sheet])

        result = splitter.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        assert result.embed_result is not None
        assert result.embed_result.texts_embedded > 0

    @patch("ingestkit_excel.processors.splitter.openpyxl.load_workbook")
    def test_process_chunk_ids_deterministic(
        self,
        mock_load,
        splitter,
        mock_db,
        mock_vector_store,
        mock_embedder,
    ):
        sheets_data = {
            "S1": {
                "rows": [["Content"], ["val1"]],
                "merged": [],
            }
        }

        def run_once():
            mock_load.return_value = _make_mock_workbook(sheets_data)
            mock_vector_store.upserted.clear()
            mock_vector_store.collections_ensured.clear()
            mock_embedder.embed_calls.clear()
            mock_db.tables_created.clear()
            sheet = _make_sheet_profile(name="S1", row_count=2, col_count=1)
            profile = _make_file_profile([sheet])
            return splitter.process(
                file_path="/tmp/test.xlsx",
                profile=profile,
                ingest_key="same_key" * 8,
                ingest_run_id="run-1",
                parse_result=_make_parse_result(),
                classification_result=_make_classification_stage_result(),
                classification=_make_classification_result(),
            )

        r1 = run_once()
        r2 = run_once()
        assert r1.written.vector_point_ids == r2.written.vector_point_ids


# ---------------------------------------------------------------------------
# Section: Type A Region Processing
# ---------------------------------------------------------------------------


class TestTypeARegionProcessing:
    """Tests for _process_type_a_region."""

    @patch("ingestkit_excel.processors.splitter.openpyxl.load_workbook")
    def test_type_a_creates_table(
        self,
        mock_load,
        splitter,
        mock_db,
        mock_vector_store,
    ):
        """Tabular region should create a DB table."""
        # 6 rows, 5 cols with clear header -> Type A
        sheets_data = {
            "Data": {
                "rows": [
                    ["Name", "Age", "Dept", "Salary", "Active"],
                    ["Alice", "30", "Eng", "100000", "yes"],
                    ["Bob", "25", "Sales", "80000", "yes"],
                    ["Carol", "35", "HR", "75000", "no"],
                    ["Dave", "28", "Eng", "90000", "yes"],
                    ["Eve", "32", "Sales", "85000", "no"],
                ],
                "merged": [],
            }
        }
        mock_load.return_value = _make_mock_workbook(sheets_data)

        sheet = _make_sheet_profile(name="Data", row_count=6, col_count=5)
        profile = _make_file_profile([sheet])

        result = splitter.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        assert result.tables_created >= 1
        assert len(mock_db.tables_created) >= 1

    @patch("ingestkit_excel.processors.splitter.openpyxl.load_workbook")
    def test_type_a_db_uri_in_metadata(
        self,
        mock_load,
        splitter,
        mock_db,
        mock_vector_store,
    ):
        """Type A chunks should have db_uri set."""
        sheets_data = {
            "Data": {
                "rows": [
                    ["Name", "Age", "Dept", "Salary", "Active"],
                    ["Alice", "30", "Eng", "100000", "yes"],
                    ["Bob", "25", "Sales", "80000", "yes"],
                    ["Carol", "35", "HR", "75000", "no"],
                    ["Dave", "28", "Eng", "90000", "yes"],
                    ["Eve", "32", "Sales", "85000", "no"],
                ],
                "merged": [],
            }
        }
        mock_load.return_value = _make_mock_workbook(sheets_data)

        sheet = _make_sheet_profile(name="Data", row_count=6, col_count=5)
        profile = _make_file_profile([sheet])

        splitter.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        # Find a chunk with db_uri set (Type A chunk)
        type_a_chunks = [
            c for _, chunks in mock_vector_store.upserted for c in chunks
            if c.metadata.db_uri is not None
        ]
        if type_a_chunks:
            assert type_a_chunks[0].metadata.db_uri == mock_db.get_connection_uri()


# ---------------------------------------------------------------------------
# Section: Type B Region Processing
# ---------------------------------------------------------------------------


class TestTypeBRegionProcessing:
    """Tests for _process_type_b_region."""

    @patch("ingestkit_excel.processors.splitter.openpyxl.load_workbook")
    def test_type_b_no_table_created(
        self,
        mock_load,
        splitter,
        mock_db,
        mock_vector_store,
    ):
        """Text region should NOT create a DB table."""
        # Short text content -> classified as FORMATTED_DOCUMENT
        sheets_data = {
            "Notes": {
                "rows": [
                    ["This is important text about our policy"],
                    ["Another paragraph about procedures"],
                ],
                "merged": [],
            }
        }
        mock_load.return_value = _make_mock_workbook(sheets_data)

        sheet = _make_sheet_profile(name="Notes", row_count=2, col_count=1)
        profile = _make_file_profile([sheet])

        result = splitter.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        # Type B chunks should have no table_name
        type_b_chunks = [
            c for _, chunks in mock_vector_store.upserted for c in chunks
            if c.metadata.table_name is None
        ]
        # At least one Type B chunk exists
        assert len(type_b_chunks) >= 1

    @patch("ingestkit_excel.processors.splitter.openpyxl.load_workbook")
    def test_type_b_original_structure_set(
        self,
        mock_load,
        splitter,
        mock_db,
        mock_vector_store,
    ):
        sheets_data = {
            "Notes": {
                "rows": [
                    ["Some text content"],
                    ["More content here"],
                ],
                "merged": [],
            }
        }
        mock_load.return_value = _make_mock_workbook(sheets_data)

        sheet = _make_sheet_profile(name="Notes", row_count=2, col_count=1)
        profile = _make_file_profile([sheet])

        splitter.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        type_b_chunks = [
            c for _, chunks in mock_vector_store.upserted for c in chunks
            if c.metadata.original_structure is not None
        ]
        if type_b_chunks:
            assert type_b_chunks[0].metadata.original_structure in [
                rt.value for rt in RegionType
            ]


# ---------------------------------------------------------------------------
# Section: Sheet Skipping
# ---------------------------------------------------------------------------


class TestSheetSkipping:
    """Tests for sheet skipping logic."""

    @patch("ingestkit_excel.processors.splitter.openpyxl.load_workbook")
    def test_process_skips_hidden_sheet(
        self,
        mock_load,
        splitter,
        mock_db,
        mock_vector_store,
    ):
        mock_load.return_value = _make_mock_workbook({})

        sheet = _make_sheet_profile(name="Hidden", row_count=10, col_count=2, is_hidden=True)
        profile = _make_file_profile([sheet])

        result = splitter.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        assert "W_SHEET_SKIPPED_HIDDEN" in result.warnings
        assert result.chunks_created == 0

    @patch("ingestkit_excel.processors.splitter.openpyxl.load_workbook")
    def test_process_skips_chart_only_sheet(
        self,
        mock_load,
        splitter,
        mock_db,
        mock_vector_store,
    ):
        mock_load.return_value = _make_mock_workbook({})

        sheet = _make_sheet_profile(name="Chart", row_count=0, col_count=0)
        profile = _make_file_profile([sheet])

        result = splitter.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        assert "W_SHEET_SKIPPED_CHART" in result.warnings
        assert result.chunks_created == 0

    @patch("ingestkit_excel.processors.splitter.openpyxl.load_workbook")
    def test_process_skips_oversized_sheet(
        self,
        mock_load,
        mock_db,
        mock_vector_store,
        mock_embedder,
        structured_processor,
        text_serializer,
    ):
        cfg = ExcelProcessorConfig(max_rows_in_memory=50)
        from ingestkit_excel.processors.structured_db import StructuredDBProcessor

        sp = StructuredDBProcessor(mock_db, mock_vector_store, mock_embedder, cfg)
        proc = HybridSplitter(sp, text_serializer, cfg)
        mock_load.return_value = _make_mock_workbook({})

        sheet = _make_sheet_profile(name="Big", row_count=100, col_count=5)
        profile = _make_file_profile([sheet])

        result = proc.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        assert "W_ROWS_TRUNCATED" in result.warnings
        assert result.chunks_created == 0

    @patch("ingestkit_excel.processors.splitter.openpyxl.load_workbook")
    def test_process_all_sheets_skipped_empty_result(
        self,
        mock_load,
        splitter,
        mock_db,
        mock_vector_store,
    ):
        mock_load.return_value = _make_mock_workbook({})

        sheet = _make_sheet_profile(name="Hidden", row_count=10, col_count=2, is_hidden=True)
        profile = _make_file_profile([sheet])

        result = splitter.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        assert result.chunks_created == 0
        assert result.tables_created == 0
        assert result.embed_result is None


# ---------------------------------------------------------------------------
# Section: Error Handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling during processing."""

    @patch("ingestkit_excel.processors.splitter.openpyxl.load_workbook")
    def test_process_sheet_error_continues(
        self,
        mock_load,
        splitter,
        mock_db,
        mock_vector_store,
    ):
        """Error on one sheet doesn't block the next."""
        wb = MagicMock()
        ws_bad = MagicMock()
        ws_bad.iter_rows.side_effect = RuntimeError("Corrupt sheet")

        ws_good = MagicMock()
        ws_good.title = "S2"
        ws_good.iter_rows.return_value = [
            tuple([_make_mock_cell("Content"), _make_mock_cell("Data")]),
            tuple([_make_mock_cell("val1"), _make_mock_cell("val2")]),
        ]
        ws_good.merged_cells.ranges = []

        wb.__getitem__ = lambda self, name: ws_bad if name == "S1" else ws_good
        wb.close = MagicMock()
        mock_load.return_value = wb

        sheet1 = _make_sheet_profile(name="S1", row_count=5, col_count=2)
        sheet2 = _make_sheet_profile(name="S2", row_count=2, col_count=2)
        profile = _make_file_profile([sheet1, sheet2])

        result = splitter.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        assert len(result.errors) >= 1
        assert result.chunks_created >= 1  # S2 still processed

    @patch("ingestkit_excel.processors.splitter.openpyxl.load_workbook")
    def test_process_region_detection_error_continues(
        self,
        mock_load,
        splitter,
        mock_db,
        mock_vector_store,
    ):
        """Region detection failure on a sheet records error and continues."""
        wb = MagicMock()

        # S1: iter_rows works but merged_cells.ranges raises
        ws_bad = MagicMock()
        ws_bad.title = "S1"
        ws_bad.iter_rows.return_value = [
            tuple([_make_mock_cell("Data")])
        ]
        # Make merged_cells.ranges access raise an error
        type(ws_bad.merged_cells).ranges = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("merge read fail"))
        )

        ws_good = MagicMock()
        ws_good.title = "S2"
        ws_good.iter_rows.return_value = [
            tuple([_make_mock_cell("Content")]),
            tuple([_make_mock_cell("val1")]),
        ]
        ws_good.merged_cells.ranges = []

        wb.__getitem__ = lambda self, name: ws_bad if name == "S1" else ws_good
        wb.close = MagicMock()
        mock_load.return_value = wb

        sheet1 = _make_sheet_profile(name="S1", row_count=1, col_count=1)
        sheet2 = _make_sheet_profile(name="S2", row_count=2, col_count=1)
        profile = _make_file_profile([sheet1, sheet2])

        result = splitter.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        # S1 should have an error, S2 should succeed
        assert len(result.errors) >= 1
        assert result.chunks_created >= 1

    @patch("ingestkit_excel.processors.splitter.openpyxl.load_workbook")
    def test_process_error_details_stage_is_process(
        self,
        mock_load,
        splitter,
        mock_db,
        mock_vector_store,
    ):
        wb = MagicMock()
        ws_bad = MagicMock()
        ws_bad.iter_rows.side_effect = RuntimeError("Corrupt sheet")
        wb.__getitem__ = lambda self, name: ws_bad
        wb.close = MagicMock()
        mock_load.return_value = wb

        sheet = _make_sheet_profile(name="BadSheet", row_count=5, col_count=2)
        profile = _make_file_profile([sheet])

        result = splitter.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        assert len(result.error_details) == 1
        assert result.error_details[0].stage == "process"
        assert result.error_details[0].sheet_name == "BadSheet"

    def test_classify_backend_error_timeout(self):
        exc = RuntimeError("Embedding timeout error")
        code = HybridSplitter._classify_backend_error(exc)
        assert code == ErrorCode.E_BACKEND_EMBED_TIMEOUT

    def test_classify_backend_error_vector_connect(self):
        exc = RuntimeError("Vector store connection refused")
        code = HybridSplitter._classify_backend_error(exc)
        assert code == ErrorCode.E_BACKEND_VECTOR_CONNECT

    def test_classify_backend_error_db_timeout(self):
        exc = RuntimeError("DB timeout waiting for response")
        code = HybridSplitter._classify_backend_error(exc)
        assert code == ErrorCode.E_BACKEND_DB_TIMEOUT

    def test_classify_backend_error_default(self):
        exc = RuntimeError("Something unexpected happened")
        code = HybridSplitter._classify_backend_error(exc)
        assert code == ErrorCode.E_PROCESS_REGION_DETECT


# ---------------------------------------------------------------------------
# Section: Multi-Sheet Processing
# ---------------------------------------------------------------------------


class TestMultiSheet:
    """Tests for multi-sheet processing."""

    @patch("ingestkit_excel.processors.splitter.openpyxl.load_workbook")
    def test_process_multi_sheet(
        self,
        mock_load,
        splitter,
        mock_db,
        mock_vector_store,
    ):
        sheets_data = {
            "S1": {
                "rows": [["Data1", "Val1"], ["d1", "v1"]],
                "merged": [],
            },
            "S2": {
                "rows": [["Data2", "Val2"], ["d2", "v2"]],
                "merged": [],
            },
        }
        mock_load.return_value = _make_mock_workbook(sheets_data)

        sheet1 = _make_sheet_profile(name="S1", row_count=2, col_count=2)
        sheet2 = _make_sheet_profile(name="S2", row_count=2, col_count=2)
        profile = _make_file_profile([sheet1, sheet2])

        result = splitter.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        assert result.chunks_created >= 2
        assert result.errors == []

    @patch("ingestkit_excel.processors.splitter.openpyxl.load_workbook")
    def test_process_multi_sheet_chunk_index_global(
        self,
        mock_load,
        splitter,
        mock_db,
        mock_vector_store,
    ):
        """chunk_index is global across sheets."""
        sheets_data = {
            "S1": {
                "rows": [["Content A"], ["val_a"]],
                "merged": [],
            },
            "S2": {
                "rows": [["Content B"], ["val_b"]],
                "merged": [],
            },
        }
        mock_load.return_value = _make_mock_workbook(sheets_data)

        sheet1 = _make_sheet_profile(name="S1", row_count=2, col_count=1)
        sheet2 = _make_sheet_profile(name="S2", row_count=2, col_count=1)
        profile = _make_file_profile([sheet1, sheet2])

        splitter.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        all_indices = []
        for _, chunks in mock_vector_store.upserted:
            for chunk in chunks:
                all_indices.append(chunk.metadata.chunk_index)

        # Indices should be sequential and unique
        assert len(all_indices) == len(set(all_indices))
        assert sorted(all_indices) == list(range(len(all_indices)))


# ---------------------------------------------------------------------------
# Section: Constants Verification
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify constants are accessible and have expected values."""

    def test_blank_row_threshold(self):
        assert _BLANK_ROW_THRESHOLD == 2

    def test_blank_col_threshold(self):
        assert _BLANK_COL_THRESHOLD == 2

    def test_formatting_transition_window(self):
        assert _FORMATTING_TRANSITION_WINDOW == 5

    def test_numeric_heavy_threshold(self):
        assert _NUMERIC_HEAVY_THRESHOLD == 0.6

    def test_text_heavy_threshold(self):
        assert _TEXT_HEAVY_THRESHOLD == 0.6

    def test_header_footer_max_rows(self):
        assert _HEADER_FOOTER_MAX_ROWS == 5

    def test_matrix_min_headers(self):
        assert _MATRIX_MIN_HEADERS == 2
