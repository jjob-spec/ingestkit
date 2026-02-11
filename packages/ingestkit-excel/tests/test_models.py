"""Tests for all enums, IngestKey, stage artifacts, core models, and protocols."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

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
    IngestKey,
    IngestionMethod,
    ParseStageResult,
    ParserUsed,
    ProcessingResult,
    RegionType,
    SheetProfile,
    SheetRegion,
    WrittenArtifacts,
)
from ingestkit_excel.protocols import (
    EmbeddingBackend,
    LLMBackend,
    StructuredDBBackend,
    VectorStoreBackend,
)


# ---------------------------------------------------------------------------
# Enum value tests -- ENUM_VALUE pattern prevention
# ---------------------------------------------------------------------------


class TestFileType:
    """Assert every FileType member value matches the spec string exactly."""

    def test_tabular_data(self) -> None:
        assert FileType.TABULAR_DATA.value == "tabular_data"

    def test_formatted_document(self) -> None:
        assert FileType.FORMATTED_DOCUMENT.value == "formatted_document"

    def test_hybrid(self) -> None:
        assert FileType.HYBRID.value == "hybrid"

    def test_member_count(self) -> None:
        assert len(FileType) == 3


class TestClassificationTier:
    """Assert every ClassificationTier member value matches the spec."""

    def test_rule_based(self) -> None:
        assert ClassificationTier.RULE_BASED.value == "rule_based"

    def test_llm_basic(self) -> None:
        assert ClassificationTier.LLM_BASIC.value == "llm_basic"

    def test_llm_reasoning(self) -> None:
        assert ClassificationTier.LLM_REASONING.value == "llm_reasoning"

    def test_member_count(self) -> None:
        assert len(ClassificationTier) == 3


class TestIngestionMethod:
    """Assert every IngestionMethod member value matches the spec."""

    def test_sql_agent(self) -> None:
        assert IngestionMethod.SQL_AGENT.value == "sql_agent"

    def test_text_serialization(self) -> None:
        assert IngestionMethod.TEXT_SERIALIZATION.value == "text_serialization"

    def test_hybrid_split(self) -> None:
        assert IngestionMethod.HYBRID_SPLIT.value == "hybrid_split"

    def test_member_count(self) -> None:
        assert len(IngestionMethod) == 3


class TestRegionType:
    """Assert every RegionType member value matches the spec."""

    def test_data_table(self) -> None:
        assert RegionType.DATA_TABLE.value == "data_table"

    def test_text_block(self) -> None:
        assert RegionType.TEXT_BLOCK.value == "text_block"

    def test_header_block(self) -> None:
        assert RegionType.HEADER_BLOCK.value == "header_block"

    def test_footer_block(self) -> None:
        assert RegionType.FOOTER_BLOCK.value == "footer_block"

    def test_matrix_block(self) -> None:
        assert RegionType.MATRIX_BLOCK.value == "matrix_block"

    def test_chart_only(self) -> None:
        assert RegionType.CHART_ONLY.value == "chart_only"

    def test_empty(self) -> None:
        assert RegionType.EMPTY.value == "empty"

    def test_member_count(self) -> None:
        assert len(RegionType) == 7


class TestParserUsed:
    """Assert every ParserUsed member value matches the spec."""

    def test_openpyxl(self) -> None:
        assert ParserUsed.OPENPYXL.value == "openpyxl"

    def test_pandas_fallback(self) -> None:
        assert ParserUsed.PANDAS_FALLBACK.value == "pandas_fallback"

    def test_raw_text_fallback(self) -> None:
        assert ParserUsed.RAW_TEXT_FALLBACK.value == "raw_text_fallback"

    def test_member_count(self) -> None:
        assert len(ParserUsed) == 3


class TestEnumsAreStrEnum:
    """Verify all enums subclass str so .value is always a string."""

    @pytest.mark.parametrize(
        "enum_cls",
        [FileType, ClassificationTier, IngestionMethod, RegionType, ParserUsed],
    )
    def test_str_subclass(self, enum_cls: type) -> None:
        for member in enum_cls:
            assert isinstance(member, str)
            assert isinstance(member.value, str)


# ---------------------------------------------------------------------------
# IngestKey tests
# ---------------------------------------------------------------------------


class TestIngestKey:
    """Tests for deterministic key computation."""

    def test_key_is_deterministic(self) -> None:
        ik = IngestKey(
            content_hash="abc",
            source_uri="file:///test.xlsx",
            parser_version="1.0.0",
        )
        assert ik.key == ik.key

    def test_key_is_hex_string(self) -> None:
        ik = IngestKey(
            content_hash="abc",
            source_uri="file:///test.xlsx",
            parser_version="1.0.0",
        )
        assert len(ik.key) == 64
        int(ik.key, 16)  # should not raise

    def test_different_content_hash_different_key(self) -> None:
        ik1 = IngestKey(content_hash="aaa", source_uri="u", parser_version="v")
        ik2 = IngestKey(content_hash="bbb", source_uri="u", parser_version="v")
        assert ik1.key != ik2.key

    def test_different_source_uri_different_key(self) -> None:
        ik1 = IngestKey(content_hash="h", source_uri="a", parser_version="v")
        ik2 = IngestKey(content_hash="h", source_uri="b", parser_version="v")
        assert ik1.key != ik2.key

    def test_different_parser_version_different_key(self) -> None:
        ik1 = IngestKey(content_hash="h", source_uri="u", parser_version="v1")
        ik2 = IngestKey(content_hash="h", source_uri="u", parser_version="v2")
        assert ik1.key != ik2.key

    def test_tenant_id_none_vs_set(self) -> None:
        ik1 = IngestKey(content_hash="h", source_uri="u", parser_version="v")
        ik2 = IngestKey(
            content_hash="h", source_uri="u", parser_version="v", tenant_id="t"
        )
        assert ik1.key != ik2.key

    def test_same_fields_same_key(self) -> None:
        kwargs = dict(
            content_hash="h", source_uri="u", parser_version="v", tenant_id="t"
        )
        assert IngestKey(**kwargs).key == IngestKey(**kwargs).key

    def test_fixture_key(self, sample_ingest_key: IngestKey) -> None:
        assert len(sample_ingest_key.key) == 64


# ---------------------------------------------------------------------------
# Stage Artifact tests
# ---------------------------------------------------------------------------


class TestParseStageResult:
    """Test ParseStageResult instantiation and validation."""

    def test_valid(self) -> None:
        r = ParseStageResult(
            parser_used=ParserUsed.OPENPYXL,
            sheets_parsed=3,
            sheets_skipped=0,
            skipped_reasons={},
            parse_duration_seconds=1.5,
        )
        assert r.parser_used == ParserUsed.OPENPYXL
        assert r.fallback_reason_code is None

    def test_missing_required(self) -> None:
        with pytest.raises(ValidationError):
            ParseStageResult()  # type: ignore[call-arg]


class TestClassificationStageResult:
    """Test ClassificationStageResult instantiation."""

    def test_valid(self) -> None:
        r = ClassificationStageResult(
            tier_used=ClassificationTier.RULE_BASED,
            file_type=FileType.TABULAR_DATA,
            confidence=0.9,
            reasoning="High signal match",
            classification_duration_seconds=0.01,
        )
        assert r.tier_used == ClassificationTier.RULE_BASED
        assert r.signals is None
        assert r.per_sheet_types is None

    def test_missing_required(self) -> None:
        with pytest.raises(ValidationError):
            ClassificationStageResult()  # type: ignore[call-arg]


class TestEmbedStageResult:
    """Test EmbedStageResult instantiation."""

    def test_valid(self) -> None:
        r = EmbedStageResult(
            texts_embedded=10,
            embedding_dimension=768,
            embed_duration_seconds=2.0,
        )
        assert r.texts_embedded == 10

    def test_missing_required(self) -> None:
        with pytest.raises(ValidationError):
            EmbedStageResult()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Core Model tests
# ---------------------------------------------------------------------------


def _make_sheet_profile(**overrides: object) -> SheetProfile:
    """Helper to build a SheetProfile with sensible defaults."""
    defaults: dict = dict(
        name="Sheet1",
        row_count=100,
        col_count=5,
        merged_cell_count=0,
        merged_cell_ratio=0.0,
        header_row_detected=True,
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


class TestSheetProfile:
    """Test SheetProfile instantiation."""

    def test_valid(self) -> None:
        sp = _make_sheet_profile()
        assert sp.name == "Sheet1"

    def test_missing_required(self) -> None:
        with pytest.raises(ValidationError):
            SheetProfile()  # type: ignore[call-arg]


class TestFileProfile:
    """Test FileProfile instantiation."""

    def test_valid(self) -> None:
        sp = _make_sheet_profile()
        fp = FileProfile(
            file_path="/tmp/test.xlsx",
            file_size_bytes=1024,
            sheet_count=1,
            sheet_names=["Sheet1"],
            sheets=[sp],
            has_password_protected_sheets=False,
            has_chart_only_sheets=False,
            total_merged_cells=0,
            total_rows=100,
            content_hash="deadbeef",
        )
        assert fp.sheet_count == 1

    def test_missing_required(self) -> None:
        with pytest.raises(ValidationError):
            FileProfile()  # type: ignore[call-arg]


class TestClassificationResult:
    """Test ClassificationResult instantiation."""

    def test_valid(self) -> None:
        cr = ClassificationResult(
            file_type=FileType.TABULAR_DATA,
            confidence=0.9,
            tier_used=ClassificationTier.RULE_BASED,
            reasoning="Looks tabular",
        )
        assert cr.file_type == FileType.TABULAR_DATA
        assert cr.per_sheet_types is None
        assert cr.signals is None


class TestChunkMetadata:
    """Test ChunkMetadata instantiation."""

    def test_valid(self) -> None:
        cm = ChunkMetadata(
            source_uri="file:///tmp/test.xlsx",
            sheet_name="Sheet1",
            ingestion_method="sql_agent",
            parser_used="openpyxl",
            parser_version="ingestkit_excel:1.0.0",
            chunk_index=0,
            chunk_hash="abc123",
            ingest_key="key123",
            ingest_run_id="run456",
        )
        assert cm.source_format == "xlsx"
        assert cm.tenant_id is None
        assert cm.table_name is None

    def test_missing_required(self) -> None:
        with pytest.raises(ValidationError):
            ChunkMetadata()  # type: ignore[call-arg]


class TestChunkPayload:
    """Test ChunkPayload instantiation."""

    def test_valid(self) -> None:
        meta = ChunkMetadata(
            source_uri="file:///test.xlsx",
            sheet_name="Sheet1",
            ingestion_method="sql_agent",
            parser_used="openpyxl",
            parser_version="ingestkit_excel:1.0.0",
            chunk_index=0,
            chunk_hash="abc",
            ingest_key="key",
            ingest_run_id="run",
        )
        cp = ChunkPayload(
            id="uuid-1",
            text="some text",
            vector=[0.1, 0.2, 0.3],
            metadata=meta,
        )
        assert cp.id == "uuid-1"
        assert len(cp.vector) == 3


class TestSheetRegion:
    """Test SheetRegion instantiation."""

    def test_valid(self) -> None:
        sr = SheetRegion(
            sheet_name="Sheet1",
            region_id="r1",
            start_row=0,
            end_row=10,
            start_col=0,
            end_col=5,
            region_type=RegionType.DATA_TABLE,
            detection_confidence=0.85,
        )
        assert sr.classified_as is None

    def test_with_classified_as(self) -> None:
        sr = SheetRegion(
            sheet_name="Sheet1",
            region_id="r2",
            start_row=11,
            end_row=20,
            start_col=0,
            end_col=5,
            region_type=RegionType.TEXT_BLOCK,
            detection_confidence=0.7,
            classified_as=FileType.FORMATTED_DOCUMENT,
        )
        assert sr.classified_as == FileType.FORMATTED_DOCUMENT


class TestWrittenArtifacts:
    """Test WrittenArtifacts defaults and instantiation."""

    def test_defaults(self) -> None:
        wa = WrittenArtifacts()
        assert wa.vector_point_ids == []
        assert wa.vector_collection is None
        assert wa.db_table_names == []

    def test_populated(self) -> None:
        wa = WrittenArtifacts(
            vector_point_ids=["p1", "p2"],
            vector_collection="helpdesk",
            db_table_names=["employees"],
        )
        assert len(wa.vector_point_ids) == 2


class TestProcessingResult:
    """Test ProcessingResult instantiation."""

    def test_valid(self) -> None:
        pr = ProcessingResult(
            file_path="/tmp/test.xlsx",
            ingest_key="key123",
            ingest_run_id="run456",
            parse_result=ParseStageResult(
                parser_used=ParserUsed.OPENPYXL,
                sheets_parsed=1,
                sheets_skipped=0,
                skipped_reasons={},
                parse_duration_seconds=0.5,
            ),
            classification_result=ClassificationStageResult(
                tier_used=ClassificationTier.RULE_BASED,
                file_type=FileType.TABULAR_DATA,
                confidence=0.9,
                reasoning="signals",
                classification_duration_seconds=0.01,
            ),
            classification=ClassificationResult(
                file_type=FileType.TABULAR_DATA,
                confidence=0.9,
                tier_used=ClassificationTier.RULE_BASED,
                reasoning="signals",
            ),
            ingestion_method=IngestionMethod.SQL_AGENT,
            chunks_created=1,
            tables_created=1,
            tables=["employees"],
            written=WrittenArtifacts(db_table_names=["employees"]),
            errors=[],
            warnings=[],
            processing_time_seconds=2.0,
        )
        assert pr.chunks_created == 1
        assert pr.error_details == []

    def test_with_error_details(self) -> None:
        err = IngestError(
            code=ErrorCode.W_PARSER_FALLBACK,
            message="openpyxl failed",
            sheet_name="Sheet2",
            stage="parse",
            recoverable=True,
        )
        pr = ProcessingResult(
            file_path="/tmp/test.xlsx",
            ingest_key="key123",
            ingest_run_id="run456",
            parse_result=ParseStageResult(
                parser_used=ParserUsed.PANDAS_FALLBACK,
                fallback_reason_code="E_PARSE_OPENPYXL_FAIL",
                sheets_parsed=1,
                sheets_skipped=0,
                skipped_reasons={},
                parse_duration_seconds=0.5,
            ),
            classification_result=ClassificationStageResult(
                tier_used=ClassificationTier.RULE_BASED,
                file_type=FileType.TABULAR_DATA,
                confidence=0.9,
                reasoning="signals",
                classification_duration_seconds=0.01,
            ),
            classification=ClassificationResult(
                file_type=FileType.TABULAR_DATA,
                confidence=0.9,
                tier_used=ClassificationTier.RULE_BASED,
                reasoning="signals",
            ),
            ingestion_method=IngestionMethod.SQL_AGENT,
            chunks_created=0,
            tables_created=0,
            tables=[],
            written=WrittenArtifacts(),
            errors=[],
            warnings=[ErrorCode.W_PARSER_FALLBACK.value],
            error_details=[err],
            processing_time_seconds=1.0,
        )
        assert len(pr.error_details) == 1
        assert pr.error_details[0].code == ErrorCode.W_PARSER_FALLBACK

    def test_missing_required(self) -> None:
        with pytest.raises(ValidationError):
            ProcessingResult()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Protocol runtime_checkable tests
# ---------------------------------------------------------------------------


class TestProtocolRuntimeCheckable:
    """Verify that classes implementing protocol methods pass isinstance checks."""

    def test_vector_store_backend(self) -> None:
        class FakeVectorStore:
            def upsert_chunks(self, collection, chunks):
                return 0

            def ensure_collection(self, collection, vector_size):
                pass

            def create_payload_index(self, collection, field, field_type):
                pass

            def delete_by_ids(self, collection, ids):
                return 0

        assert isinstance(FakeVectorStore(), VectorStoreBackend)

    def test_structured_db_backend(self) -> None:
        class FakeStructuredDB:
            def create_table_from_dataframe(self, table_name, df):
                pass

            def drop_table(self, table_name):
                pass

            def table_exists(self, table_name):
                return False

            def get_table_schema(self, table_name):
                return {}

            def get_connection_uri(self):
                return "sqlite:///:memory:"

        assert isinstance(FakeStructuredDB(), StructuredDBBackend)

    def test_llm_backend(self) -> None:
        class FakeLLM:
            def classify(self, prompt, model, temperature=0.1, timeout=None):
                return {}

            def generate(self, prompt, model, temperature=0.7, timeout=None):
                return ""

        assert isinstance(FakeLLM(), LLMBackend)

    def test_embedding_backend(self) -> None:
        class FakeEmbedding:
            def embed(self, texts, timeout=None):
                return [[0.0] * 768 for _ in texts]

            def dimension(self):
                return 768

        assert isinstance(FakeEmbedding(), EmbeddingBackend)

    def test_non_conforming_class_fails(self) -> None:
        class Empty:
            pass

        assert not isinstance(Empty(), VectorStoreBackend)
        assert not isinstance(Empty(), StructuredDBBackend)
        assert not isinstance(Empty(), LLMBackend)
        assert not isinstance(Empty(), EmbeddingBackend)
