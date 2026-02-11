"""Tests for the StructuredDBProcessor (Path A).

Covers column cleaning, date detection, schema description generation,
row serialization, full process flow, multi-sheet processing, error
handling, and WrittenArtifacts tracking.
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
    SheetProfile,
    WrittenArtifacts,
)
from ingestkit_excel.processors.structured_db import (
    StructuredDBProcessor,
    clean_name,
    deduplicate_names,
)


# ---------------------------------------------------------------------------
# Test Helper Factories
# ---------------------------------------------------------------------------


def _make_sheet_profile(**overrides: object) -> SheetProfile:
    """Build a SheetProfile with sensible tabular defaults."""
    defaults: dict = dict(
        name="Sheet1",
        row_count=100,
        col_count=5,
        merged_cell_count=0,
        merged_cell_ratio=0.0,
        header_row_detected=True,
        header_row_index=0,
        header_values=["A", "B", "C", "D", "E"],
        column_type_consistency=0.9,
        numeric_ratio=0.4,
        text_ratio=0.5,
        empty_ratio=0.1,
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
        file_size_bytes=1024,
        sheet_count=len(sheets),
        sheet_names=[s.name for s in sheets],
        sheets=sheets,
        has_password_protected_sheets=False,
        has_chart_only_sheets=False,
        total_merged_cells=sum(s.merged_cell_count for s in sheets),
        total_rows=sum(s.row_count for s in sheets),
        content_hash="a" * 64,
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
        file_type=FileType.TABULAR_DATA,
        confidence=0.95,
        signals=None,
        reasoning="High column consistency, low merged cells",
        per_sheet_types=None,
        classification_duration_seconds=0.05,
    )
    defaults.update(overrides)
    return ClassificationStageResult(**defaults)


def _make_classification_result(**overrides: object) -> ClassificationResult:
    defaults: dict = dict(
        file_type=FileType.TABULAR_DATA,
        confidence=0.95,
        tier_used=ClassificationTier.RULE_BASED,
        reasoning="High column consistency, low merged cells",
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
        self.fail_on: str | None = None  # table name to fail on

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
def processor(
    mock_db: MockStructuredDB,
    mock_vector_store: MockVectorStore,
    mock_embedder: MockEmbedder,
    config: ExcelProcessorConfig,
) -> StructuredDBProcessor:
    return StructuredDBProcessor(mock_db, mock_vector_store, mock_embedder, config)


# ---------------------------------------------------------------------------
# Section: Column Name Cleaning (clean_name and deduplicate_names)
# ---------------------------------------------------------------------------


class TestCleanName:
    """Tests for the clean_name helper."""

    def test_clean_name_lowercase(self):
        assert clean_name("Hello World") == "hello_world"

    def test_clean_name_special_chars(self):
        assert clean_name("Col #1 (USD)") == "col_1_usd"

    def test_clean_name_consecutive_underscores(self):
        assert clean_name("a___b") == "a_b"

    def test_clean_name_leading_trailing_underscores(self):
        assert clean_name("__name__") == "name"

    def test_clean_name_empty_string(self):
        assert clean_name("") == ""

    def test_clean_name_unicode(self):
        # Umlaut characters get replaced with underscores
        result = clean_name("Geb\u00fchrenstatus")
        assert "_" in result or result.isascii()
        # The umlaut 'u' should be replaced
        assert "geb" in result

    def test_clean_name_numeric_only(self):
        assert clean_name("123") == "123"


class TestDeduplicateNames:
    """Tests for the deduplicate_names helper."""

    def test_deduplicate_names_no_dupes(self):
        assert deduplicate_names(["a", "b", "c"]) == ["a", "b", "c"]

    def test_deduplicate_names_with_dupes(self):
        assert deduplicate_names(["a", "a", "a"]) == ["a", "a_1", "a_2"]

    def test_deduplicate_names_empty_becomes_column_n(self):
        assert deduplicate_names(["", "", "a"]) == ["column_0", "column_1", "a"]

    def test_deduplicate_names_mixed(self):
        assert deduplicate_names(["id", "", "id", "name"]) == [
            "id",
            "column_1",
            "id_1",
            "name",
        ]


# ---------------------------------------------------------------------------
# Section: Date Detection (_auto_detect_dates)
# ---------------------------------------------------------------------------


class TestAutoDetectDates:
    """Tests for the _auto_detect_dates private method."""

    def test_auto_detect_excel_serial_dates(self, processor):
        df = pd.DataFrame({"date_col": [44197, 44228, 44256]})  # ~2021 dates
        result = processor._auto_detect_dates(df)
        assert pd.api.types.is_datetime64_any_dtype(result["date_col"])

    def test_auto_detect_string_dates(self, processor):
        df = pd.DataFrame(
            {"date_col": ["2023-01-15", "2023-02-20", "2023-03-10"]}
        )
        result = processor._auto_detect_dates(df)
        assert pd.api.types.is_datetime64_any_dtype(result["date_col"])

    def test_auto_detect_skips_integers_outside_range(self, processor):
        df = pd.DataFrame({"ids": [1, 2, 3, 50, 100]})
        result = processor._auto_detect_dates(df)
        assert pd.api.types.is_integer_dtype(result["ids"])

    def test_auto_detect_mixed_dates_below_threshold(self, processor):
        # Less than 50% parse as dates -> should NOT convert
        df = pd.DataFrame(
            {"mixed": ["2023-01-15", "not_a_date", "also_not", "nope"]}
        )
        result = processor._auto_detect_dates(df)
        assert not pd.api.types.is_datetime64_any_dtype(result["mixed"])

    def test_auto_detect_empty_column_ignored(self, processor):
        df = pd.DataFrame({"empty": [None, None, None]})
        result = processor._auto_detect_dates(df)
        # Should not raise and column should remain as-is
        assert len(result) == 3

    def test_auto_detect_preserves_non_date_columns(self, processor):
        df = pd.DataFrame(
            {
                "ids": [1, 2, 3],
                "names": ["Alice", "Bob", "Carol"],
                "serial_dates": [44197, 44228, 44256],
            }
        )
        result = processor._auto_detect_dates(df)
        assert pd.api.types.is_integer_dtype(result["ids"])
        assert not pd.api.types.is_datetime64_any_dtype(result["names"])
        assert pd.api.types.is_datetime64_any_dtype(result["serial_dates"])


# ---------------------------------------------------------------------------
# Section: Schema Description Generation
# ---------------------------------------------------------------------------


class TestSchemaDescription:
    """Tests for the _generate_schema_description method."""

    def test_schema_description_contains_table_name(self, processor):
        df = pd.DataFrame({"a": [1, 2, 3]})
        text = processor._generate_schema_description("my_table", df)
        assert 'Table "my_table"' in text

    def test_schema_description_integer_column_range(self, processor):
        df = pd.DataFrame({"count": [10, 20, 30]})
        text = processor._generate_schema_description("t", df)
        assert "range 10 to 30" in text

    def test_schema_description_float_column_range(self, processor):
        df = pd.DataFrame({"price": [1.5, 2.5, 3.5]})
        text = processor._generate_schema_description("t", df)
        assert "range 1.5 to 3.5" in text

    def test_schema_description_text_low_cardinality(self, processor):
        df = pd.DataFrame({"dept": ["HR", "Sales", "Eng"]})
        text = processor._generate_schema_description("t", df)
        assert "one of" in text
        assert "HR" in text
        assert "Sales" in text

    def test_schema_description_text_high_cardinality(self, processor):
        df = pd.DataFrame({"names": [f"name_{i}" for i in range(25)]})
        text = processor._generate_schema_description("t", df)
        assert "25 unique values" in text

    def test_schema_description_date_column_range(self, processor):
        df = pd.DataFrame(
            {"date": pd.to_datetime(["2023-01-01", "2023-06-15", "2023-12-31"])}
        )
        text = processor._generate_schema_description("t", df)
        assert "ranges from" in text

    def test_schema_description_boolean_column(self, processor):
        df = pd.DataFrame({"active": [True, False, True]})
        text = processor._generate_schema_description("t", df)
        assert "true/false" in text

    def test_schema_description_row_count(self, processor):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        text = processor._generate_schema_description("t", df)
        assert "contains 5 rows" in text


# ---------------------------------------------------------------------------
# Section: Row Serialization
# ---------------------------------------------------------------------------


class TestSerializeRows:
    """Tests for the _serialize_rows private method."""

    def test_serialize_rows_format(self, processor):
        df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        sheet = _make_sheet_profile(name="Sheet1")
        chunks = processor._serialize_rows(
            table_name="employees",
            df=df,
            start_chunk_index=1,
            sheet=sheet,
            source_uri="file:///tmp/test.xlsx",
            db_uri="sqlite:///test.db",
            ingest_key="abc123",
            ingest_run_id="run-1",
        )
        assert len(chunks) == 2
        assert "In table 'employees', row 1:" in chunks[0].text
        assert "id is 1" in chunks[0].text
        assert "name is Alice" in chunks[0].text

    def test_serialize_rows_handles_nan(self, processor):
        df = pd.DataFrame({"id": [1], "name": [None]})
        sheet = _make_sheet_profile(name="Sheet1")
        chunks = processor._serialize_rows(
            table_name="t",
            df=df,
            start_chunk_index=0,
            sheet=sheet,
            source_uri="file:///tmp/test.xlsx",
            db_uri="sqlite:///test.db",
            ingest_key="key",
            ingest_run_id="run",
        )
        assert "name is N/A" in chunks[0].text

    def test_serialize_rows_chunk_metadata_correct(self, processor):
        df = pd.DataFrame({"id": [1]})
        sheet = _make_sheet_profile(name="MySheet")
        chunks = processor._serialize_rows(
            table_name="my_table",
            df=df,
            start_chunk_index=5,
            sheet=sheet,
            source_uri="file:///tmp/test.xlsx",
            db_uri="sqlite:///test.db",
            ingest_key="key",
            ingest_run_id="run",
        )
        meta = chunks[0].metadata
        assert meta.table_name == "my_table"
        assert meta.chunk_index == 5
        assert meta.sheet_name == "MySheet"

    def test_serialize_rows_chunk_ids_deterministic(self, processor):
        df = pd.DataFrame({"id": [1]})
        sheet = _make_sheet_profile(name="Sheet1")
        kwargs = dict(
            table_name="t",
            df=df,
            start_chunk_index=0,
            sheet=sheet,
            source_uri="file:///tmp/test.xlsx",
            db_uri="sqlite:///test.db",
            ingest_key="same_key",
            ingest_run_id="run",
        )
        chunks1 = processor._serialize_rows(**kwargs)
        chunks2 = processor._serialize_rows(**kwargs)
        assert chunks1[0].id == chunks2[0].id

    def test_serialize_rows_chunk_index_continues(self, processor):
        df = pd.DataFrame({"id": [1, 2, 3]})
        sheet = _make_sheet_profile(name="Sheet1")
        chunks = processor._serialize_rows(
            table_name="t",
            df=df,
            start_chunk_index=10,
            sheet=sheet,
            source_uri="file:///tmp/test.xlsx",
            db_uri="sqlite:///test.db",
            ingest_key="key",
            ingest_run_id="run",
        )
        assert chunks[0].metadata.chunk_index == 10
        assert chunks[1].metadata.chunk_index == 11
        assert chunks[2].metadata.chunk_index == 12


# ---------------------------------------------------------------------------
# Section: Full Process Flow (with pd.read_excel mocked)
# ---------------------------------------------------------------------------


class TestProcessFlow:
    """Tests for the full process() method with mocked pd.read_excel."""

    @patch("ingestkit_excel.processors.structured_db.pd.read_excel")
    def test_process_single_sheet_happy_path(
        self,
        mock_read_excel,
        processor,
        mock_db,
        mock_vector_store,
        mock_embedder,
    ):
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Carol"]})
        mock_read_excel.return_value = df

        sheet = _make_sheet_profile(name="Employees", row_count=3, col_count=2)
        profile = _make_file_profile([sheet])
        parse_result = _make_parse_result()
        classification_stage = _make_classification_stage_result()
        classification = _make_classification_result()

        result = processor.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="abc123" * 10 + "abcd",
            ingest_run_id="550e8400-e29b-41d4-a716-446655440000",
            parse_result=parse_result,
            classification_result=classification_stage,
            classification=classification,
        )

        # Verify DB interaction
        assert len(mock_db.tables_created) == 1
        assert mock_db.tables_created[0][0] == "employees"

        # Verify vector store interaction
        assert len(mock_vector_store.upserted) >= 1
        assert mock_vector_store.collections_ensured[0] == ("helpdesk", 768)

        # Verify result structure
        assert result.ingestion_method == IngestionMethod.SQL_AGENT
        assert result.tables_created == 1
        assert result.tables == ["employees"]
        assert result.written.db_table_names == ["employees"]
        assert len(result.written.vector_point_ids) >= 1
        assert result.written.vector_collection == "helpdesk"
        assert result.errors == []
        assert result.processing_time_seconds > 0

    @patch("ingestkit_excel.processors.structured_db.pd.read_excel")
    def test_process_result_ingestion_method(
        self,
        mock_read_excel,
        processor,
        mock_db,
        mock_vector_store,
        mock_embedder,
    ):
        df = pd.DataFrame({"a": [1]})
        mock_read_excel.return_value = df

        sheet = _make_sheet_profile(name="S1", row_count=1, col_count=1)
        profile = _make_file_profile([sheet])

        result = processor.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )
        # Must be the enum member, NOT the string
        assert result.ingestion_method == IngestionMethod.SQL_AGENT
        assert result.ingestion_method is IngestionMethod.SQL_AGENT

    @patch("ingestkit_excel.processors.structured_db.pd.read_excel")
    def test_process_chunk_metadata_ingestion_method(
        self,
        mock_read_excel,
        processor,
        mock_db,
        mock_vector_store,
        mock_embedder,
    ):
        df = pd.DataFrame({"a": [1]})
        mock_read_excel.return_value = df

        sheet = _make_sheet_profile(name="S1", row_count=1, col_count=1)
        profile = _make_file_profile([sheet])

        processor.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        chunk = mock_vector_store.upserted[0][1][0]
        # ChunkMetadata.ingestion_method is a str field -> must be "sql_agent"
        assert chunk.metadata.ingestion_method == "sql_agent"

    @patch("ingestkit_excel.processors.structured_db.pd.read_excel")
    def test_process_chunk_metadata_parser_used(
        self,
        mock_read_excel,
        processor,
        mock_db,
        mock_vector_store,
        mock_embedder,
    ):
        df = pd.DataFrame({"a": [1]})
        mock_read_excel.return_value = df

        sheet = _make_sheet_profile(
            name="S1", row_count=1, col_count=1, parser_used=ParserUsed.OPENPYXL
        )
        profile = _make_file_profile([sheet])

        processor.process(
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

    @patch("ingestkit_excel.processors.structured_db.pd.read_excel")
    def test_process_written_artifacts_populated(
        self,
        mock_read_excel,
        processor,
        mock_db,
        mock_vector_store,
        mock_embedder,
    ):
        df = pd.DataFrame({"a": [1]})
        mock_read_excel.return_value = df

        sheet = _make_sheet_profile(name="S1", row_count=1, col_count=1)
        profile = _make_file_profile([sheet])

        result = processor.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        assert len(result.written.db_table_names) > 0
        assert len(result.written.vector_point_ids) > 0

    @patch("ingestkit_excel.processors.structured_db.pd.read_excel")
    def test_process_written_artifacts_collection(
        self,
        mock_read_excel,
        processor,
        mock_db,
        mock_vector_store,
        mock_embedder,
    ):
        df = pd.DataFrame({"a": [1]})
        mock_read_excel.return_value = df

        sheet = _make_sheet_profile(name="S1", row_count=1, col_count=1)
        profile = _make_file_profile([sheet])

        result = processor.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        assert result.written.vector_collection == "helpdesk"

    @patch("ingestkit_excel.processors.structured_db.pd.read_excel")
    def test_process_source_uri_format(
        self,
        mock_read_excel,
        processor,
        mock_db,
        mock_vector_store,
        mock_embedder,
    ):
        df = pd.DataFrame({"a": [1]})
        mock_read_excel.return_value = df

        sheet = _make_sheet_profile(name="S1", row_count=1, col_count=1)
        profile = _make_file_profile([sheet])

        processor.process(
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

    @patch("ingestkit_excel.processors.structured_db.pd.read_excel")
    def test_process_db_uri_from_backend(
        self,
        mock_read_excel,
        processor,
        mock_db,
        mock_vector_store,
        mock_embedder,
    ):
        df = pd.DataFrame({"a": [1]})
        mock_read_excel.return_value = df

        sheet = _make_sheet_profile(name="S1", row_count=1, col_count=1)
        profile = _make_file_profile([sheet])

        processor.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        chunk = mock_vector_store.upserted[0][1][0]
        assert chunk.metadata.db_uri == mock_db.get_connection_uri()

    @patch("ingestkit_excel.processors.structured_db.pd.read_excel")
    def test_process_embed_result_populated(
        self,
        mock_read_excel,
        processor,
        mock_db,
        mock_vector_store,
        mock_embedder,
    ):
        df = pd.DataFrame({"a": [1]})
        mock_read_excel.return_value = df

        sheet = _make_sheet_profile(name="S1", row_count=1, col_count=1)
        profile = _make_file_profile([sheet])

        result = processor.process(
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

    @patch("ingestkit_excel.processors.structured_db.pd.read_excel")
    def test_process_tables_created_count(
        self,
        mock_read_excel,
        processor,
        mock_db,
        mock_vector_store,
        mock_embedder,
    ):
        df = pd.DataFrame({"a": [1]})
        mock_read_excel.return_value = df

        sheet = _make_sheet_profile(name="S1", row_count=1, col_count=1)
        profile = _make_file_profile([sheet])

        result = processor.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        assert result.tables_created == 1

    @patch("ingestkit_excel.processors.structured_db.pd.read_excel")
    def test_process_chunks_created_count_schema_only(
        self,
        mock_read_excel,
        mock_db,
        mock_vector_store,
        mock_embedder,
    ):
        """With row_serialization_limit=0, only schema chunk counted."""
        cfg = ExcelProcessorConfig(row_serialization_limit=0)
        proc = StructuredDBProcessor(
            mock_db, mock_vector_store, mock_embedder, cfg
        )
        df = pd.DataFrame({"a": [1, 2, 3]})
        mock_read_excel.return_value = df

        sheet = _make_sheet_profile(name="S1", row_count=3, col_count=1)
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

        assert result.chunks_created == 1  # schema only

    @patch("ingestkit_excel.processors.structured_db.pd.read_excel")
    def test_process_tenant_id_passed_through(
        self,
        mock_read_excel,
        mock_db,
        mock_vector_store,
        mock_embedder,
    ):
        cfg = ExcelProcessorConfig(tenant_id="tenant_abc")
        proc = StructuredDBProcessor(
            mock_db, mock_vector_store, mock_embedder, cfg
        )
        df = pd.DataFrame({"a": [1]})
        mock_read_excel.return_value = df

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

        assert result.tenant_id == "tenant_abc"
        chunk = mock_vector_store.upserted[0][1][0]
        assert chunk.metadata.tenant_id == "tenant_abc"


# ---------------------------------------------------------------------------
# Section: Row Serialization Integration (triggers / skips)
# ---------------------------------------------------------------------------


class TestRowSerializationIntegration:
    """Tests for row serialization triggering/skipping during process()."""

    @patch("ingestkit_excel.processors.structured_db.pd.read_excel")
    def test_process_row_serialization_below_limit(
        self,
        mock_read_excel,
        processor,
        mock_db,
        mock_vector_store,
        mock_embedder,
    ):
        df = pd.DataFrame({"id": [1, 2, 3]})
        mock_read_excel.return_value = df

        sheet = _make_sheet_profile(name="S1", row_count=3, col_count=1)
        profile = _make_file_profile([sheet])

        result = processor.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        # 1 schema chunk + 3 row chunks = 4 total
        assert result.chunks_created == 4

    @patch("ingestkit_excel.processors.structured_db.pd.read_excel")
    def test_process_row_serialization_above_limit(
        self,
        mock_read_excel,
        mock_db,
        mock_vector_store,
        mock_embedder,
    ):
        cfg = ExcelProcessorConfig(row_serialization_limit=50)
        proc = StructuredDBProcessor(
            mock_db, mock_vector_store, mock_embedder, cfg
        )
        df = pd.DataFrame({"id": list(range(100))})
        mock_read_excel.return_value = df

        sheet = _make_sheet_profile(name="S1", row_count=100, col_count=1)
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

        # Only schema chunk, no row serialization
        assert result.chunks_created == 1

    @patch("ingestkit_excel.processors.structured_db.pd.read_excel")
    def test_process_row_serialization_batch_embedding(
        self,
        mock_read_excel,
        mock_db,
        mock_vector_store,
        mock_embedder,
    ):
        cfg = ExcelProcessorConfig(embedding_batch_size=64)
        proc = StructuredDBProcessor(
            mock_db, mock_vector_store, mock_embedder, cfg
        )
        df = pd.DataFrame({"id": list(range(130))})
        mock_read_excel.return_value = df

        sheet = _make_sheet_profile(name="S1", row_count=130, col_count=1)
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

        # 1 schema embed call + 2 row batch embed calls (64 + 64 + 2) = 3 calls
        # Actually: 1 schema + ceil(130/64) = 1 + 3 = 4 calls (64 + 64 + 2)
        assert len(mock_embedder.embed_calls) == 4  # 1 schema + 3 row batches

    @patch("ingestkit_excel.processors.structured_db.pd.read_excel")
    def test_process_row_serialization_batch_embedding_correct_batch_count(
        self,
        mock_read_excel,
        mock_db,
        mock_vector_store,
        mock_embedder,
    ):
        """130 rows with batch_size=64: ceil(130/64) = 3 row batches."""
        cfg = ExcelProcessorConfig(embedding_batch_size=64)
        proc = StructuredDBProcessor(
            mock_db, mock_vector_store, mock_embedder, cfg
        )
        df = pd.DataFrame({"id": list(range(130))})
        mock_read_excel.return_value = df

        sheet = _make_sheet_profile(name="S1", row_count=130, col_count=1)
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

        # First call: schema embed (1 text)
        assert len(mock_embedder.embed_calls[0]) == 1
        # Row batch calls
        assert len(mock_embedder.embed_calls[1]) == 64
        assert len(mock_embedder.embed_calls[2]) == 64
        assert len(mock_embedder.embed_calls[3]) == 2


# ---------------------------------------------------------------------------
# Section: Multi-Sheet Processing
# ---------------------------------------------------------------------------


class TestMultiSheetProcessing:
    """Tests for multi-sheet processing flows."""

    @patch("ingestkit_excel.processors.structured_db.pd.read_excel")
    def test_process_multi_sheet(
        self,
        mock_read_excel,
        mock_db,
        mock_vector_store,
        mock_embedder,
    ):
        cfg = ExcelProcessorConfig(row_serialization_limit=0)
        proc = StructuredDBProcessor(
            mock_db, mock_vector_store, mock_embedder, cfg
        )

        df1 = pd.DataFrame({"id": [1, 2]})
        df2 = pd.DataFrame({"name": ["A", "B"]})
        mock_read_excel.side_effect = [df1, df2]

        sheet1 = _make_sheet_profile(name="Sales", row_count=2, col_count=1)
        sheet2 = _make_sheet_profile(name="Inventory", row_count=2, col_count=1)
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

        assert result.tables_created == 2
        assert len(result.written.db_table_names) == 2
        assert "sales" in result.tables
        assert "inventory" in result.tables
        assert result.chunks_created >= 2

    @patch("ingestkit_excel.processors.structured_db.pd.read_excel")
    def test_process_multi_sheet_chunk_index_global(
        self,
        mock_read_excel,
        mock_db,
        mock_vector_store,
        mock_embedder,
    ):
        """chunk_index is global across sheets."""
        cfg = ExcelProcessorConfig(row_serialization_limit=0)
        proc = StructuredDBProcessor(
            mock_db, mock_vector_store, mock_embedder, cfg
        )

        df1 = pd.DataFrame({"id": [1]})
        df2 = pd.DataFrame({"name": ["A"]})
        mock_read_excel.side_effect = [df1, df2]

        sheet1 = _make_sheet_profile(name="S1", row_count=1, col_count=1)
        sheet2 = _make_sheet_profile(name="S2", row_count=1, col_count=1)
        profile = _make_file_profile([sheet1, sheet2])

        proc.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        # First sheet schema chunk: index 0
        chunk0 = mock_vector_store.upserted[0][1][0]
        assert chunk0.metadata.chunk_index == 0
        # Second sheet schema chunk: index 1
        chunk1 = mock_vector_store.upserted[1][1][0]
        assert chunk1.metadata.chunk_index == 1

    @patch("ingestkit_excel.processors.structured_db.pd.read_excel")
    def test_process_duplicate_sheet_names_deduped(
        self,
        mock_read_excel,
        mock_db,
        mock_vector_store,
        mock_embedder,
    ):
        cfg = ExcelProcessorConfig(row_serialization_limit=0)
        proc = StructuredDBProcessor(
            mock_db, mock_vector_store, mock_embedder, cfg
        )

        df = pd.DataFrame({"a": [1]})
        mock_read_excel.return_value = df

        sheet1 = _make_sheet_profile(name="Data", row_count=1, col_count=1)
        sheet2 = _make_sheet_profile(name="Data", row_count=1, col_count=1)
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

        assert "data" in result.tables
        assert "data_1" in result.tables
        assert result.tables_created == 2


# ---------------------------------------------------------------------------
# Section: Sheet Skipping
# ---------------------------------------------------------------------------


class TestSheetSkipping:
    """Tests for sheet skipping logic."""

    @patch("ingestkit_excel.processors.structured_db.pd.read_excel")
    def test_process_skips_hidden_sheet(
        self,
        mock_read_excel,
        processor,
        mock_db,
        mock_vector_store,
        mock_embedder,
    ):
        sheet = _make_sheet_profile(name="Hidden", row_count=10, col_count=2, is_hidden=True)
        profile = _make_file_profile([sheet])

        result = processor.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        assert "W_SHEET_SKIPPED_HIDDEN" in result.warnings
        assert result.tables_created == 0
        mock_read_excel.assert_not_called()

    @patch("ingestkit_excel.processors.structured_db.pd.read_excel")
    def test_process_skips_oversized_sheet(
        self,
        mock_read_excel,
        mock_db,
        mock_vector_store,
        mock_embedder,
    ):
        cfg = ExcelProcessorConfig(max_rows_in_memory=50)
        proc = StructuredDBProcessor(
            mock_db, mock_vector_store, mock_embedder, cfg
        )

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
        assert result.tables_created == 0

    @patch("ingestkit_excel.processors.structured_db.pd.read_excel")
    def test_process_all_sheets_skipped_empty_result(
        self,
        mock_read_excel,
        processor,
        mock_db,
        mock_vector_store,
        mock_embedder,
    ):
        sheet = _make_sheet_profile(
            name="Hidden", row_count=10, col_count=2, is_hidden=True
        )
        profile = _make_file_profile([sheet])

        result = processor.process(
            file_path="/tmp/test.xlsx",
            profile=profile,
            ingest_key="k" * 64,
            ingest_run_id="run-1",
            parse_result=_make_parse_result(),
            classification_result=_make_classification_stage_result(),
            classification=_make_classification_result(),
        )

        assert result.tables_created == 0
        assert result.chunks_created == 0
        assert result.embed_result is None


# ---------------------------------------------------------------------------
# Section: Error Handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling during processing."""

    @patch("ingestkit_excel.processors.structured_db.pd.read_excel")
    def test_process_db_failure_records_error(
        self,
        mock_read_excel,
        mock_db,
        mock_vector_store,
        mock_embedder,
        config,
    ):
        mock_db.fail_on = "sheet1"
        proc = StructuredDBProcessor(
            mock_db, mock_vector_store, mock_embedder, config
        )
        df = pd.DataFrame({"a": [1]})
        mock_read_excel.return_value = df

        sheet = _make_sheet_profile(name="Sheet1", row_count=1, col_count=1)
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

        assert len(result.errors) == 1
        assert len(result.error_details) == 1

    @patch("ingestkit_excel.processors.structured_db.pd.read_excel")
    def test_process_db_failure_continues_to_next_sheet(
        self,
        mock_read_excel,
        mock_db,
        mock_vector_store,
        mock_embedder,
        config,
    ):
        mock_db.fail_on = "sheet1"
        proc = StructuredDBProcessor(
            mock_db, mock_vector_store, mock_embedder, config
        )

        df = pd.DataFrame({"a": [1]})
        mock_read_excel.return_value = df

        sheet1 = _make_sheet_profile(name="Sheet1", row_count=1, col_count=1)
        sheet2 = _make_sheet_profile(name="Sheet2", row_count=1, col_count=1)
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

        # Sheet1 fails, Sheet2 succeeds
        assert result.tables_created == 1
        assert len(result.errors) == 1

    @patch("ingestkit_excel.processors.structured_db.pd.read_excel")
    def test_process_embed_failure_records_error(
        self,
        mock_read_excel,
        mock_db,
        mock_vector_store,
        mock_embedder,
        config,
    ):
        mock_embedder.fail_on_embed = True
        proc = StructuredDBProcessor(
            mock_db, mock_vector_store, mock_embedder, config
        )
        df = pd.DataFrame({"a": [1]})
        mock_read_excel.return_value = df

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

        assert len(result.errors) == 1

    @patch("ingestkit_excel.processors.structured_db.pd.read_excel")
    def test_process_error_details_stage_is_process(
        self,
        mock_read_excel,
        mock_db,
        mock_vector_store,
        mock_embedder,
        config,
    ):
        mock_db.fail_on = "sheet1"
        proc = StructuredDBProcessor(
            mock_db, mock_vector_store, mock_embedder, config
        )
        df = pd.DataFrame({"a": [1]})
        mock_read_excel.return_value = df

        sheet = _make_sheet_profile(name="Sheet1", row_count=1, col_count=1)
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

        assert result.error_details[0].stage == "process"

    @patch("ingestkit_excel.processors.structured_db.pd.read_excel")
    def test_process_error_details_has_sheet_name(
        self,
        mock_read_excel,
        mock_db,
        mock_vector_store,
        mock_embedder,
        config,
    ):
        mock_db.fail_on = "sheet1"
        proc = StructuredDBProcessor(
            mock_db, mock_vector_store, mock_embedder, config
        )
        df = pd.DataFrame({"a": [1]})
        mock_read_excel.return_value = df

        sheet = _make_sheet_profile(name="Sheet1", row_count=1, col_count=1)
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

        assert result.error_details[0].sheet_name == "Sheet1"


# ---------------------------------------------------------------------------
# Section: Config Interactions
# ---------------------------------------------------------------------------


class TestConfigInteractions:
    """Tests for config-driven behavior."""

    @patch("ingestkit_excel.processors.structured_db.pd.read_excel")
    def test_process_clean_column_names_disabled(
        self,
        mock_read_excel,
        mock_db,
        mock_vector_store,
        mock_embedder,
    ):
        cfg = ExcelProcessorConfig(clean_column_names=False, row_serialization_limit=0)
        proc = StructuredDBProcessor(
            mock_db, mock_vector_store, mock_embedder, cfg
        )
        df = pd.DataFrame({"My Column": [1, 2, 3]})
        mock_read_excel.return_value = df

        sheet = _make_sheet_profile(name="S1", row_count=3, col_count=1)
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

        # Column names should NOT be cleaned
        stored_df = mock_db.tables_created[0][1]
        assert "My Column" in stored_df.columns

    @patch("ingestkit_excel.processors.structured_db.pd.read_excel")
    def test_process_custom_collection(
        self,
        mock_read_excel,
        mock_db,
        mock_vector_store,
        mock_embedder,
    ):
        cfg = ExcelProcessorConfig(
            default_collection="custom_collection", row_serialization_limit=0
        )
        proc = StructuredDBProcessor(
            mock_db, mock_vector_store, mock_embedder, cfg
        )
        df = pd.DataFrame({"a": [1]})
        mock_read_excel.return_value = df

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

        assert mock_vector_store.collections_ensured[0][0] == "custom_collection"
        assert mock_vector_store.upserted[0][0] == "custom_collection"
        assert result.written.vector_collection == "custom_collection"
