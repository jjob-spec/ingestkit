"""Tests for the TextSerializer (Path B).

Covers section detection, sub-structure classification, all four
serialization formats, merged cell handling, ChunkMetadata correctness,
process flow, multi-sheet processing, sheet skipping, error handling,
and embedding batching.
"""

from __future__ import annotations

import hashlib
import uuid
from unittest.mock import MagicMock, PropertyMock, patch

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
    SheetProfile,
    WrittenArtifacts,
)
from ingestkit_excel.processors.serializer import Section, TextSerializer


# ---------------------------------------------------------------------------
# Test Helper Factories
# ---------------------------------------------------------------------------


def _make_sheet_profile(**overrides: object) -> SheetProfile:
    """Build a SheetProfile with sensible formatted-document defaults."""
    defaults: dict = dict(
        name="Sheet1",
        row_count=50,
        col_count=5,
        merged_cell_count=5,
        merged_cell_ratio=0.1,
        header_row_detected=False,
        header_row_index=None,
        header_values=[],
        column_type_consistency=0.4,
        numeric_ratio=0.2,
        text_ratio=0.6,
        empty_ratio=0.2,
        sample_rows=[["Task", "Status", "Due"]],
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
        file_size_bytes=2048,
        sheet_count=len(sheets),
        sheet_names=[s.name for s in sheets],
        sheets=sheets,
        has_password_protected_sheets=False,
        has_chart_only_sheets=False,
        total_merged_cells=sum(s.merged_cell_count for s in sheets),
        total_rows=sum(s.row_count for s in sheets),
        content_hash="b" * 64,
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
        file_type=FileType.FORMATTED_DOCUMENT,
        confidence=0.90,
        signals=None,
        reasoning="High merged cell ratio, document-like structure",
        per_sheet_types=None,
        classification_duration_seconds=0.05,
    )
    defaults.update(overrides)
    return ClassificationStageResult(**defaults)


def _make_classification_result(**overrides: object) -> ClassificationResult:
    defaults: dict = dict(
        file_type=FileType.FORMATTED_DOCUMENT,
        confidence=0.90,
        tier_used=ClassificationTier.RULE_BASED,
        reasoning="High merged cell ratio, document-like structure",
        per_sheet_types=None,
        signals=None,
    )
    defaults.update(overrides)
    return ClassificationResult(**defaults)


# ---------------------------------------------------------------------------
# Mock Backends
# ---------------------------------------------------------------------------


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
def mock_vector_store() -> MockVectorStore:
    return MockVectorStore()


@pytest.fixture()
def mock_embedder() -> MockEmbedder:
    return MockEmbedder()


@pytest.fixture()
def config() -> ExcelProcessorConfig:
    return ExcelProcessorConfig()


@pytest.fixture()
def serializer(
    mock_vector_store: MockVectorStore,
    mock_embedder: MockEmbedder,
    config: ExcelProcessorConfig,
) -> TextSerializer:
    return TextSerializer(mock_vector_store, mock_embedder, config)


# ---------------------------------------------------------------------------
# Section: Section Detection
# ---------------------------------------------------------------------------


class TestSectionDetection:
    """Tests for _detect_sections logic."""

    def test_blank_row_splits_sections(self):
        """Two data blocks separated by blank rows produce 2 sections."""
        sheet = _make_sheet_profile(name="TestSheet")
        sheets_data = {
            "TestSheet": {
                "rows": [
                    ["Header1", "Col1"],
                    ["Data1", "Val1"],
                    [None, None],  # blank separator
                    ["Header2", "Col2"],
                    ["Data2", "Val2"],
                ],
                "merged": [],
            }
        }
        wb = _make_mock_workbook(sheets_data)
        ws = wb["TestSheet"]
        sections = TextSerializer._detect_sections(ws, sheet)
        assert len(sections) == 2

    def test_merged_header_creates_section(self):
        """A merged cell row starts a new section with that title."""
        sheet = _make_sheet_profile(name="TestSheet")
        sheets_data = {
            "TestSheet": {
                "rows": [
                    ["IT Setup", None, None],  # merged header
                    ["Task", "Status", "Due"],
                    ["Install OS", "Done", "Jan"],
                ],
                "merged": [(1, 1, 1, 3)],  # row 1, cols 1-3
            }
        }
        wb = _make_mock_workbook(sheets_data)
        ws = wb["TestSheet"]
        sections = TextSerializer._detect_sections(ws, sheet)
        assert len(sections) == 1
        assert sections[0].title == "IT Setup"

    def test_no_boundaries_single_section(self):
        """No blank rows or merged headers -> 1 section with sheet name."""
        sheet = _make_sheet_profile(name="MySheet")
        sheets_data = {
            "MySheet": {
                "rows": [
                    ["A", "B"],
                    ["1", "2"],
                    ["3", "4"],
                ],
                "merged": [],
            }
        }
        wb = _make_mock_workbook(sheets_data)
        ws = wb["MySheet"]
        sections = TextSerializer._detect_sections(ws, sheet)
        assert len(sections) == 1
        assert sections[0].title == "MySheet"

    def test_empty_sheet_produces_no_sections(self):
        """All blank rows produce no sections."""
        sheet = _make_sheet_profile(name="Empty")
        sheets_data = {
            "Empty": {
                "rows": [
                    [None, None],
                    [None, None],
                ],
                "merged": [],
            }
        }
        wb = _make_mock_workbook(sheets_data)
        ws = wb["Empty"]
        sections = TextSerializer._detect_sections(ws, sheet)
        assert len(sections) == 0

    def test_section_title_from_merged_cell(self):
        """Title comes from merged cell value."""
        sheet = _make_sheet_profile(name="TestSheet")
        sheets_data = {
            "TestSheet": {
                "rows": [
                    ["Requirements", None, None],  # merged
                    ["Item", "Priority", "Owner"],
                    ["Auth", "High", "Alice"],
                ],
                "merged": [(1, 1, 1, 3)],
            }
        }
        wb = _make_mock_workbook(sheets_data)
        ws = wb["TestSheet"]
        sections = TextSerializer._detect_sections(ws, sheet)
        assert sections[0].title == "Requirements"

    def test_section_title_fallback_numbering(self):
        """No clear title -> 'Section N' fallback (multiple sections)."""
        sheet = _make_sheet_profile(name="TestSheet")
        sheets_data = {
            "TestSheet": {
                "rows": [
                    ["Data1", "A"],
                    [None, None],
                    ["Data2", "B"],
                ],
                "merged": [],
            }
        }
        wb = _make_mock_workbook(sheets_data)
        ws = wb["TestSheet"]
        sections = TextSerializer._detect_sections(ws, sheet)
        assert len(sections) == 2
        assert sections[0].title == "Section 1"
        assert sections[1].title == "Section 2"


# ---------------------------------------------------------------------------
# Section: Sub-Structure Classification
# ---------------------------------------------------------------------------


class TestSubStructureClassification:
    """Tests for _classify_sub_structure heuristics."""

    def test_classify_checklist(self):
        """Status column header triggers checklist classification."""
        sec = Section(
            title="Tasks",
            sub_structure="free_text",
            rows=[
                ["Task", "Status", "Due"],
                ["Install", "Done", "Jan"],
                ["Config", "Pending", "Feb"],
            ],
            start_row=1,
            end_row=3,
            col_count=3,
        )
        TextSerializer._classify_sub_structure(sec)
        assert sec.sub_structure == "checklist"

    def test_classify_matrix(self):
        """Row + column headers trigger matrix classification."""
        sec = Section(
            title="Coverage",
            sub_structure="free_text",
            rows=[
                [None, "Q1", "Q2", "Q3"],
                ["Sales", 100, 200, 300],
                ["Marketing", 50, 80, 120],
            ],
            start_row=1,
            end_row=3,
            col_count=4,
        )
        TextSerializer._classify_sub_structure(sec)
        assert sec.sub_structure == "matrix"

    def test_classify_table(self):
        """Consistent header + data rows trigger table classification."""
        sec = Section(
            title="Employees",
            sub_structure="free_text",
            rows=[
                ["Name", "Department", "Salary"],
                ["Alice", "Engineering", "100000"],
                ["Bob", "Sales", "80000"],
                ["Carol", "HR", "75000"],
            ],
            start_row=1,
            end_row=4,
            col_count=3,
        )
        TextSerializer._classify_sub_structure(sec)
        assert sec.sub_structure == "table"

    def test_classify_free_text_default(self):
        """Ambiguous structure defaults to free_text."""
        sec = Section(
            title="Notes",
            sub_structure="free_text",
            rows=[
                ["This is a long paragraph of text about something important", None, None],
                ["Another paragraph follows here with more information", None, None],
                [None, None, "sparse"],
            ],
            start_row=1,
            end_row=3,
            col_count=3,
        )
        TextSerializer._classify_sub_structure(sec)
        assert sec.sub_structure == "free_text"

    def test_classify_few_rows_defaults_free_text(self):
        """< 2 rows -> free_text immediately."""
        sec = Section(
            title="Single",
            sub_structure="table",  # even if pre-set
            rows=[["Only one row", "of data"]],
            start_row=1,
            end_row=1,
            col_count=2,
        )
        TextSerializer._classify_sub_structure(sec)
        assert sec.sub_structure == "free_text"


# ---------------------------------------------------------------------------
# Section: Serialization Formats
# ---------------------------------------------------------------------------


class TestSerializationFormats:
    """Tests for the four serialization format methods."""

    def test_serialize_checklist_format(self):
        """Checklist format matches 'Item X: status is Y, due date is Z...'."""
        sec = Section(
            title="Tasks",
            sub_structure="checklist",
            rows=[
                ["Task", "Status", "Due Date"],
                ["Install OS", "Done", "Jan 15"],
                ["Configure", "Pending", "Feb 1"],
            ],
            start_row=1,
            end_row=3,
            col_count=3,
            header_row=["Task", "Status", "Due Date"],
        )
        text = TextSerializer._serialize_checklist(sec)
        assert "Install OS:" in text
        assert "status is Done" in text
        assert "due date is Jan 15" in text

    def test_serialize_matrix_format(self):
        """Matrix format matches 'For {row_header}, {col_header} is {value}.'."""
        sec = Section(
            title="Coverage",
            sub_structure="matrix",
            rows=[
                [None, "Q1", "Q2"],
                ["Sales", 100, 200],
                ["Marketing", 50, 80],
            ],
            start_row=1,
            end_row=3,
            col_count=3,
        )
        text = TextSerializer._serialize_matrix(sec)
        assert "For Sales, Q1 is 100." in text
        assert "For Sales, Q2 is 200." in text
        assert "For Marketing, Q1 is 50." in text

    def test_serialize_table_format(self):
        """Table format matches 'In section '{title}', {col} is {val}...'."""
        sec = Section(
            title="Team",
            sub_structure="table",
            rows=[
                ["Name", "Role"],
                ["Alice", "Engineer"],
                ["Bob", "Manager"],
            ],
            start_row=1,
            end_row=3,
            col_count=2,
            header_row=["Name", "Role"],
        )
        text = TextSerializer._serialize_table(sec)
        assert "In section 'Team', Name is Alice, Role is Engineer." in text
        assert "In section 'Team', Name is Bob, Role is Manager." in text

    def test_serialize_free_text_preserves_paragraphs(self):
        """Free text uses double newlines between rows."""
        sec = Section(
            title="Notes",
            sub_structure="free_text",
            rows=[
                ["First paragraph content"],
                ["Second paragraph content"],
            ],
            start_row=1,
            end_row=2,
            col_count=1,
        )
        text = TextSerializer._serialize_free_text(sec)
        assert "Notes" in text
        assert "First paragraph content" in text
        assert "Second paragraph content" in text
        # Double newline between paragraphs
        assert "\n\n" in text

    def test_serialize_handles_none_values(self):
        """None values become 'N/A' in table serialization."""
        sec = Section(
            title="Data",
            sub_structure="table",
            rows=[
                ["Name", "Value"],
                ["Item1", None],
            ],
            start_row=1,
            end_row=2,
            col_count=2,
            header_row=["Name", "Value"],
        )
        text = TextSerializer._serialize_table(sec)
        assert "N/A" in text


# ---------------------------------------------------------------------------
# Section: Merged Cell Handling
# ---------------------------------------------------------------------------


class TestMergedCellHandling:
    """Tests for merged cell detection and usage."""

    def test_merged_cells_detected_from_openpyxl(self):
        """Merged cell ranges are parsed correctly."""
        sheet = _make_sheet_profile(name="TestSheet")
        sheets_data = {
            "TestSheet": {
                "rows": [
                    ["Section Title", None, None],
                    ["Data", "Value", "Extra"],
                ],
                "merged": [(1, 1, 1, 3)],
            }
        }
        wb = _make_mock_workbook(sheets_data)
        ws = wb["TestSheet"]
        sections = TextSerializer._detect_sections(ws, sheet)
        assert len(sections) == 1
        assert sections[0].title == "Section Title"

    def test_merged_header_value_used_as_title(self):
        """Merged cell value becomes the section title."""
        sheet = _make_sheet_profile(name="TestSheet")
        sheets_data = {
            "TestSheet": {
                "rows": [
                    ["IT Setup Requirements", None, None, None],
                    ["Task", "Status", "Due", "Owner"],
                    ["Install", "Done", "Jan", "Alice"],
                ],
                "merged": [(1, 1, 1, 4)],
            }
        }
        wb = _make_mock_workbook(sheets_data)
        ws = wb["TestSheet"]
        sections = TextSerializer._detect_sections(ws, sheet)
        assert sections[0].title == "IT Setup Requirements"


# ---------------------------------------------------------------------------
# Section: ChunkMetadata
# ---------------------------------------------------------------------------


class TestChunkMetadata:
    """Tests for ChunkMetadata fields in Path B processing."""

    @patch("ingestkit_excel.processors.serializer.openpyxl.load_workbook")
    def test_metadata_ingestion_method(
        self, mock_load, serializer, mock_vector_store
    ):
        sheets_data = {
            "S1": {
                "rows": [["Text content here"]],
                "merged": [],
            }
        }
        mock_load.return_value = _make_mock_workbook(sheets_data)

        sheet = _make_sheet_profile(name="S1", row_count=1, col_count=1)
        profile = _make_file_profile([sheet])

        serializer.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        chunk = mock_vector_store.upserted[0][1][0]
        assert chunk.metadata.ingestion_method == "text_serialization"

    @patch("ingestkit_excel.processors.serializer.openpyxl.load_workbook")
    def test_metadata_section_title_populated(
        self, mock_load, serializer, mock_vector_store
    ):
        sheets_data = {
            "S1": {
                "rows": [
                    ["My Section", None],
                    ["Data", "Value"],
                ],
                "merged": [(1, 1, 1, 2)],
            }
        }
        mock_load.return_value = _make_mock_workbook(sheets_data)

        sheet = _make_sheet_profile(name="S1", row_count=2, col_count=2)
        profile = _make_file_profile([sheet])

        serializer.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        chunk = mock_vector_store.upserted[0][1][0]
        assert chunk.metadata.section_title == "My Section"

    @patch("ingestkit_excel.processors.serializer.openpyxl.load_workbook")
    def test_metadata_original_structure_populated(
        self, mock_load, serializer, mock_vector_store
    ):
        sheets_data = {
            "S1": {
                "rows": [["Just some text content"]],
                "merged": [],
            }
        }
        mock_load.return_value = _make_mock_workbook(sheets_data)

        sheet = _make_sheet_profile(name="S1", row_count=1, col_count=1)
        profile = _make_file_profile([sheet])

        serializer.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        chunk = mock_vector_store.upserted[0][1][0]
        assert chunk.metadata.original_structure in ("table", "checklist", "matrix", "free_text")

    @patch("ingestkit_excel.processors.serializer.openpyxl.load_workbook")
    def test_metadata_no_table_fields(
        self, mock_load, serializer, mock_vector_store
    ):
        sheets_data = {
            "S1": {
                "rows": [["Content"]],
                "merged": [],
            }
        }
        mock_load.return_value = _make_mock_workbook(sheets_data)

        sheet = _make_sheet_profile(name="S1", row_count=1, col_count=1)
        profile = _make_file_profile([sheet])

        serializer.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        chunk = mock_vector_store.upserted[0][1][0]
        assert chunk.metadata.table_name is None
        assert chunk.metadata.db_uri is None

    @patch("ingestkit_excel.processors.serializer.openpyxl.load_workbook")
    def test_metadata_tenant_id_propagated(self, mock_load, mock_vector_store, mock_embedder):
        cfg = ExcelProcessorConfig(tenant_id="tenant_xyz")
        proc = TextSerializer(mock_vector_store, mock_embedder, cfg)

        sheets_data = {
            "S1": {
                "rows": [["Content"]],
                "merged": [],
            }
        }
        mock_load.return_value = _make_mock_workbook(sheets_data)

        sheet = _make_sheet_profile(name="S1", row_count=1, col_count=1)
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

    @patch("ingestkit_excel.processors.serializer.openpyxl.load_workbook")
    def test_metadata_source_uri_format(
        self, mock_load, serializer, mock_vector_store
    ):
        sheets_data = {
            "S1": {
                "rows": [["Content"]],
                "merged": [],
            }
        }
        mock_load.return_value = _make_mock_workbook(sheets_data)

        sheet = _make_sheet_profile(name="S1", row_count=1, col_count=1)
        profile = _make_file_profile([sheet])

        serializer.process(
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

    @patch("ingestkit_excel.processors.serializer.openpyxl.load_workbook")
    def test_metadata_parser_used(
        self, mock_load, serializer, mock_vector_store
    ):
        sheets_data = {
            "S1": {
                "rows": [["Content"]],
                "merged": [],
            }
        }
        mock_load.return_value = _make_mock_workbook(sheets_data)

        sheet = _make_sheet_profile(
            name="S1", row_count=1, col_count=1, parser_used=ParserUsed.OPENPYXL
        )
        profile = _make_file_profile([sheet])

        serializer.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        chunk = mock_vector_store.upserted[0][1][0]
        assert chunk.metadata.parser_used == "openpyxl"


# ---------------------------------------------------------------------------
# Section: Process Flow
# ---------------------------------------------------------------------------


class TestProcessFlow:
    """Tests for the full process() method."""

    @patch("ingestkit_excel.processors.serializer.openpyxl.load_workbook")
    def test_process_single_sheet_happy_path(
        self, mock_load, serializer, mock_vector_store, mock_embedder
    ):
        sheets_data = {
            "Checklist": {
                "rows": [
                    ["Task", "Status"],
                    ["Install", "Done"],
                    ["Configure", "Pending"],
                ],
                "merged": [],
            }
        }
        mock_load.return_value = _make_mock_workbook(sheets_data)

        sheet = _make_sheet_profile(name="Checklist", row_count=3, col_count=2)
        profile = _make_file_profile([sheet])

        result = serializer.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        assert result.chunks_created >= 1
        assert len(mock_vector_store.upserted) >= 1
        assert result.errors == []
        assert result.processing_time_seconds > 0

    @patch("ingestkit_excel.processors.serializer.openpyxl.load_workbook")
    def test_process_result_ingestion_method(
        self, mock_load, serializer, mock_vector_store
    ):
        sheets_data = {
            "S1": {
                "rows": [["Content"]],
                "merged": [],
            }
        }
        mock_load.return_value = _make_mock_workbook(sheets_data)

        sheet = _make_sheet_profile(name="S1", row_count=1, col_count=1)
        profile = _make_file_profile([sheet])

        result = serializer.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        assert result.ingestion_method == IngestionMethod.TEXT_SERIALIZATION
        assert result.ingestion_method is IngestionMethod.TEXT_SERIALIZATION

    @patch("ingestkit_excel.processors.serializer.openpyxl.load_workbook")
    def test_process_tables_created_zero(
        self, mock_load, serializer, mock_vector_store
    ):
        sheets_data = {
            "S1": {
                "rows": [["Content"]],
                "merged": [],
            }
        }
        mock_load.return_value = _make_mock_workbook(sheets_data)

        sheet = _make_sheet_profile(name="S1", row_count=1, col_count=1)
        profile = _make_file_profile([sheet])

        result = serializer.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        assert result.tables_created == 0
        assert result.tables == []

    @patch("ingestkit_excel.processors.serializer.openpyxl.load_workbook")
    def test_process_written_artifacts_no_db_tables(
        self, mock_load, serializer, mock_vector_store
    ):
        sheets_data = {
            "S1": {
                "rows": [["Content"]],
                "merged": [],
            }
        }
        mock_load.return_value = _make_mock_workbook(sheets_data)

        sheet = _make_sheet_profile(name="S1", row_count=1, col_count=1)
        profile = _make_file_profile([sheet])

        result = serializer.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        assert result.written.db_table_names == []

    @patch("ingestkit_excel.processors.serializer.openpyxl.load_workbook")
    def test_process_written_artifacts_vector_ids_populated(
        self, mock_load, serializer, mock_vector_store
    ):
        sheets_data = {
            "S1": {
                "rows": [["Content"]],
                "merged": [],
            }
        }
        mock_load.return_value = _make_mock_workbook(sheets_data)

        sheet = _make_sheet_profile(name="S1", row_count=1, col_count=1)
        profile = _make_file_profile([sheet])

        result = serializer.process(
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

    @patch("ingestkit_excel.processors.serializer.openpyxl.load_workbook")
    def test_process_embed_result_populated(
        self, mock_load, serializer, mock_vector_store
    ):
        sheets_data = {
            "S1": {
                "rows": [["Content"]],
                "merged": [],
            }
        }
        mock_load.return_value = _make_mock_workbook(sheets_data)

        sheet = _make_sheet_profile(name="S1", row_count=1, col_count=1)
        profile = _make_file_profile([sheet])

        result = serializer.process(
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

    @patch("ingestkit_excel.processors.serializer.openpyxl.load_workbook")
    def test_process_chunk_ids_deterministic(
        self, mock_load, serializer, mock_vector_store, mock_embedder
    ):
        sheets_data = {
            "S1": {
                "rows": [["Content"]],
                "merged": [],
            }
        }

        def run_once():
            mock_load.return_value = _make_mock_workbook(sheets_data)
            mock_vector_store.upserted.clear()
            mock_embedder.embed_calls.clear()
            sheet = _make_sheet_profile(name="S1", row_count=1, col_count=1)
            profile = _make_file_profile([sheet])
            return serializer.process(
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
# Section: Multi-Sheet
# ---------------------------------------------------------------------------


class TestMultiSheet:
    """Tests for multi-sheet processing."""

    @patch("ingestkit_excel.processors.serializer.openpyxl.load_workbook")
    def test_process_multi_sheet_chunk_index_global(
        self, mock_load, serializer, mock_vector_store
    ):
        sheets_data = {
            "S1": {
                "rows": [["Content A"]],
                "merged": [],
            },
            "S2": {
                "rows": [["Content B"]],
                "merged": [],
            },
        }
        mock_load.return_value = _make_mock_workbook(sheets_data)

        sheet1 = _make_sheet_profile(name="S1", row_count=1, col_count=1)
        sheet2 = _make_sheet_profile(name="S2", row_count=1, col_count=1)
        profile = _make_file_profile([sheet1, sheet2])

        serializer.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        chunk0 = mock_vector_store.upserted[0][1][0]
        chunk1 = mock_vector_store.upserted[1][1][0]
        assert chunk0.metadata.chunk_index == 0
        assert chunk1.metadata.chunk_index == 1

    @patch("ingestkit_excel.processors.serializer.openpyxl.load_workbook")
    def test_process_multi_sheet_different_structures(
        self, mock_load, serializer, mock_vector_store
    ):
        sheets_data = {
            "Checklist": {
                "rows": [
                    ["Task", "Status"],
                    ["Install", "Done"],
                ],
                "merged": [],
            },
            "Notes": {
                "rows": [["Just free text content here"]],
                "merged": [],
            },
        }
        mock_load.return_value = _make_mock_workbook(sheets_data)

        sheet1 = _make_sheet_profile(name="Checklist", row_count=2, col_count=2)
        sheet2 = _make_sheet_profile(name="Notes", row_count=1, col_count=1)
        profile = _make_file_profile([sheet1, sheet2])

        result = serializer.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        assert result.chunks_created == 2


# ---------------------------------------------------------------------------
# Section: Sheet Skipping
# ---------------------------------------------------------------------------


class TestSheetSkipping:
    """Tests for sheet skipping logic."""

    @patch("ingestkit_excel.processors.serializer.openpyxl.load_workbook")
    def test_process_skips_hidden_sheet(
        self, mock_load, serializer, mock_vector_store
    ):
        mock_load.return_value = _make_mock_workbook({})

        sheet = _make_sheet_profile(name="Hidden", row_count=10, col_count=2, is_hidden=True)
        profile = _make_file_profile([sheet])

        result = serializer.process(
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

    @patch("ingestkit_excel.processors.serializer.openpyxl.load_workbook")
    def test_process_skips_chart_only_sheet(
        self, mock_load, serializer, mock_vector_store
    ):
        mock_load.return_value = _make_mock_workbook({})

        sheet = _make_sheet_profile(name="Chart", row_count=0, col_count=0)
        profile = _make_file_profile([sheet])

        result = serializer.process(
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

    @patch("ingestkit_excel.processors.serializer.openpyxl.load_workbook")
    def test_process_skips_oversized_sheet(
        self, mock_load, mock_vector_store, mock_embedder
    ):
        cfg = ExcelProcessorConfig(max_rows_in_memory=50)
        proc = TextSerializer(mock_vector_store, mock_embedder, cfg)
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

    @patch("ingestkit_excel.processors.serializer.openpyxl.load_workbook")
    def test_process_all_sheets_skipped_empty_result(
        self, mock_load, serializer, mock_vector_store
    ):
        mock_load.return_value = _make_mock_workbook({})

        sheet = _make_sheet_profile(name="Hidden", row_count=10, col_count=2, is_hidden=True)
        profile = _make_file_profile([sheet])

        result = serializer.process(
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

    @patch("ingestkit_excel.processors.serializer.openpyxl.load_workbook")
    def test_process_sheet_error_continues(
        self, mock_load, mock_vector_store, mock_embedder
    ):
        """Error on one sheet doesn't block the next."""
        cfg = ExcelProcessorConfig()
        proc = TextSerializer(mock_vector_store, mock_embedder, cfg)

        # Build a workbook where S1 raises, S2 works
        wb = MagicMock()
        ws_bad = MagicMock()
        ws_bad.iter_rows.side_effect = RuntimeError("Corrupt sheet")

        ws_good = MagicMock()
        ws_good.iter_rows.return_value = [
            tuple([_make_mock_cell("Content")])
        ]
        ws_good.merged_cells.ranges = []
        ws_good.cell = lambda r, c: _make_mock_cell("Content")

        wb.__getitem__ = lambda self, name: ws_bad if name == "S1" else ws_good
        wb.close = MagicMock()
        mock_load.return_value = wb

        sheet1 = _make_sheet_profile(name="S1", row_count=5, col_count=2)
        sheet2 = _make_sheet_profile(name="S2", row_count=1, col_count=1)
        profile = _make_file_profile([sheet1, sheet2])

        result = proc.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        assert len(result.errors) == 1
        assert result.chunks_created >= 1  # S2 still processed

    @patch("ingestkit_excel.processors.serializer.openpyxl.load_workbook")
    def test_process_error_recorded_in_details(
        self, mock_load, mock_vector_store, mock_embedder
    ):
        cfg = ExcelProcessorConfig()
        proc = TextSerializer(mock_vector_store, mock_embedder, cfg)

        wb = MagicMock()
        ws_bad = MagicMock()
        ws_bad.iter_rows.side_effect = RuntimeError("Test error")
        wb.__getitem__ = lambda self, name: ws_bad
        wb.close = MagicMock()
        mock_load.return_value = wb

        sheet = _make_sheet_profile(name="BadSheet", row_count=5, col_count=2)
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

        assert len(result.error_details) == 1
        assert result.error_details[0].stage == "process"
        assert result.error_details[0].sheet_name == "BadSheet"

    def test_classify_backend_error_timeout(self):
        exc = RuntimeError("Embedding timeout error")
        code = TextSerializer._classify_backend_error(exc)
        assert code == ErrorCode.E_BACKEND_EMBED_TIMEOUT

    def test_classify_backend_error_connect(self):
        exc = RuntimeError("Vector store connection refused")
        code = TextSerializer._classify_backend_error(exc)
        assert code == ErrorCode.E_BACKEND_VECTOR_CONNECT

    def test_classify_backend_error_default(self):
        exc = RuntimeError("Something unexpected happened")
        code = TextSerializer._classify_backend_error(exc)
        assert code == ErrorCode.E_PROCESS_SERIALIZE


# ---------------------------------------------------------------------------
# Section: Embedding Batching
# ---------------------------------------------------------------------------


class TestEmbeddingBatching:
    """Tests for embedding batch size handling."""

    @patch("ingestkit_excel.processors.serializer.openpyxl.load_workbook")
    def test_embedding_respects_batch_size(
        self, mock_load, mock_vector_store, mock_embedder
    ):
        cfg = ExcelProcessorConfig(embedding_batch_size=2)
        proc = TextSerializer(mock_vector_store, mock_embedder, cfg)

        # Create a sheet with 5 sections (via blank row separators)
        rows = []
        for i in range(5):
            rows.append([f"Section {i+1} data", f"Value {i+1}"])
            rows.append([None, None])
        # Remove last blank row
        rows.pop()

        sheets_data = {
            "S1": {
                "rows": rows,
                "merged": [],
            }
        }
        mock_load.return_value = _make_mock_workbook(sheets_data)

        sheet = _make_sheet_profile(name="S1", row_count=len(rows), col_count=2)
        profile = _make_file_profile([sheet])

        proc.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        # 5 sections with batch_size=2: ceil(5/2) = 3 embed calls
        assert len(mock_embedder.embed_calls) == 3
        assert len(mock_embedder.embed_calls[0]) == 2
        assert len(mock_embedder.embed_calls[1]) == 2
        assert len(mock_embedder.embed_calls[2]) == 1
