"""Tests for ingestkit_excel.router — ExcelRouter orchestrator and public API.

All internal modules (ParserChain, ExcelInspector, LLMClassifier, processors)
are mocked to test router logic in isolation.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from ingestkit_excel.config import ExcelProcessorConfig
from ingestkit_excel.errors import ErrorCode, IngestError
from ingestkit_excel.models import (
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
    SheetProfile,
    WrittenArtifacts,
)
from ingestkit_excel.router import ExcelRouter, create_default_router


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> ExcelProcessorConfig:
    """Return a config with defaults for testing."""
    return ExcelProcessorConfig(tenant_id="test_tenant")


@pytest.fixture()
def mock_vector_store():
    """Return a mock VectorStoreBackend."""
    vs = MagicMock()
    vs.upsert_chunks.return_value = 1
    vs.ensure_collection.return_value = None
    return vs


@pytest.fixture()
def mock_structured_db():
    """Return a mock StructuredDBBackend."""
    db = MagicMock()
    db.create_table_from_dataframe.return_value = None
    db.get_connection_uri.return_value = "sqlite:///test.db"
    return db


@pytest.fixture()
def mock_llm():
    """Return a mock LLMBackend."""
    llm = MagicMock()
    llm.classify.return_value = {
        "type": "tabular_data",
        "confidence": 0.85,
        "reasoning": "test reasoning",
    }
    return llm


@pytest.fixture()
def mock_embedder():
    """Return a mock EmbeddingBackend."""
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * 768]
    embedder.dimension.return_value = 768
    return embedder


@pytest.fixture()
def sample_sheet_profile() -> SheetProfile:
    """Return a sample SheetProfile for tests."""
    return SheetProfile(
        name="Sheet1",
        row_count=100,
        col_count=5,
        merged_cell_count=0,
        merged_cell_ratio=0.0,
        header_row_detected=True,
        header_row_index=0,
        header_values=["Name", "Age", "Salary", "Department", "ID"],
        column_type_consistency=0.95,
        numeric_ratio=0.4,
        text_ratio=0.5,
        empty_ratio=0.1,
        sample_rows=[
            ["Name", "Age", "Salary", "Department", "ID"],
            ["Alice", "30", "50000", "Engineering", "1"],
            ["Bob", "25", "45000", "Marketing", "2"],
        ],
        has_formulas=False,
        is_hidden=False,
        parser_used=ParserUsed.OPENPYXL,
    )


@pytest.fixture()
def sample_file_profile(sample_sheet_profile: SheetProfile) -> FileProfile:
    """Return a sample FileProfile with one tabular sheet."""
    return FileProfile(
        file_path="/tmp/test.xlsx",
        file_size_bytes=1024,
        sheet_count=1,
        sheet_names=["Sheet1"],
        sheets=[sample_sheet_profile],
        has_password_protected_sheets=False,
        has_chart_only_sheets=False,
        total_merged_cells=0,
        total_rows=100,
        content_hash="abc123",
    )


@pytest.fixture()
def tabular_classification() -> ClassificationResult:
    """Return a high-confidence tabular classification."""
    return ClassificationResult(
        file_type=FileType.TABULAR_DATA,
        confidence=0.9,
        tier_used=ClassificationTier.RULE_BASED,
        reasoning="All sheets classified as tabular_data.",
        signals={"per_sheet": {}},
    )


@pytest.fixture()
def formatted_classification() -> ClassificationResult:
    """Return a high-confidence formatted document classification."""
    return ClassificationResult(
        file_type=FileType.FORMATTED_DOCUMENT,
        confidence=0.9,
        tier_used=ClassificationTier.RULE_BASED,
        reasoning="All sheets classified as formatted_document.",
        signals={"per_sheet": {}},
    )


@pytest.fixture()
def hybrid_classification() -> ClassificationResult:
    """Return a high-confidence hybrid classification."""
    return ClassificationResult(
        file_type=FileType.HYBRID,
        confidence=0.9,
        tier_used=ClassificationTier.RULE_BASED,
        reasoning="Sheets disagree: hybrid classification.",
        signals={"per_sheet": {}},
    )


@pytest.fixture()
def inconclusive_classification() -> ClassificationResult:
    """Return an inconclusive classification (confidence 0.0)."""
    return ClassificationResult(
        file_type=FileType.HYBRID,
        confidence=0.0,
        tier_used=ClassificationTier.RULE_BASED,
        reasoning="Inconclusive: could not classify.",
        signals={"per_sheet": {}},
    )


@pytest.fixture()
def sample_processing_result() -> ProcessingResult:
    """Return a sample ProcessingResult for processor mock return values."""
    return ProcessingResult(
        file_path="/tmp/test.xlsx",
        ingest_key="abc123",
        ingest_run_id="run-123",
        tenant_id="test_tenant",
        parse_result=ParseStageResult(
            parser_used=ParserUsed.OPENPYXL,
            fallback_reason_code=None,
            sheets_parsed=1,
            sheets_skipped=0,
            skipped_reasons={},
            parse_duration_seconds=0.1,
        ),
        classification_result=ClassificationStageResult(
            tier_used=ClassificationTier.RULE_BASED,
            file_type=FileType.TABULAR_DATA,
            confidence=0.9,
            signals=None,
            reasoning="Classified as tabular.",
            per_sheet_types=None,
            classification_duration_seconds=0.05,
        ),
        embed_result=EmbedStageResult(
            texts_embedded=1,
            embedding_dimension=768,
            embed_duration_seconds=0.02,
        ),
        classification=ClassificationResult(
            file_type=FileType.TABULAR_DATA,
            confidence=0.9,
            tier_used=ClassificationTier.RULE_BASED,
            reasoning="Tabular data.",
        ),
        ingestion_method=IngestionMethod.SQL_AGENT,
        chunks_created=2,
        tables_created=1,
        tables=["sheet1"],
        written=WrittenArtifacts(
            vector_point_ids=["id1", "id2"],
            vector_collection="helpdesk",
            db_table_names=["sheet1"],
        ),
        errors=[],
        warnings=[],
        error_details=[],
        processing_time_seconds=0.5,
    )


def _make_ingest_key():
    """Build a deterministic IngestKey for patching."""
    return IngestKey(
        content_hash="abc123",
        source_uri="file:///tmp/test.xlsx",
        parser_version="ingestkit_excel:1.0.0",
        tenant_id="test_tenant",
    )


# ---------------------------------------------------------------------------
# Tests: Constructor
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExcelRouterInit:
    """Test ExcelRouter construction."""

    def test_creates_with_all_backends(
        self, mock_vector_store, mock_structured_db, mock_llm, mock_embedder, config
    ):
        router = ExcelRouter(
            vector_store=mock_vector_store,
            structured_db=mock_structured_db,
            llm=mock_llm,
            embedder=mock_embedder,
            config=config,
        )
        assert router._config is config
        assert router._parser_chain is not None
        assert router._inspector is not None
        assert router._llm_classifier is not None
        assert router._structured_db_processor is not None
        assert router._text_serializer is not None
        assert router._hybrid_splitter is not None

    def test_creates_with_default_config(
        self, mock_vector_store, mock_structured_db, mock_llm, mock_embedder
    ):
        router = ExcelRouter(
            vector_store=mock_vector_store,
            structured_db=mock_structured_db,
            llm=mock_llm,
            embedder=mock_embedder,
        )
        assert router._config is not None
        assert isinstance(router._config, ExcelProcessorConfig)


# ---------------------------------------------------------------------------
# Tests: Happy path — Path A (tabular)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRouterPathA:
    """Test routing to Path A (StructuredDBProcessor) for tabular files."""

    @patch("ingestkit_excel.router.compute_ingest_key")
    @patch.object(ExcelRouter, "_select_processor")
    def test_tabular_routes_to_path_a(
        self,
        mock_select,
        mock_ingest_key,
        mock_vector_store,
        mock_structured_db,
        mock_llm,
        mock_embedder,
        config,
        sample_file_profile,
        tabular_classification,
        sample_processing_result,
    ):
        """A file classified as TABULAR_DATA should be routed to StructuredDBProcessor."""
        mock_ingest_key.return_value = _make_ingest_key()

        # Mock processor
        mock_processor = MagicMock()
        mock_processor.process.return_value = sample_processing_result
        mock_select.return_value = mock_processor

        router = ExcelRouter(
            vector_store=mock_vector_store,
            structured_db=mock_structured_db,
            llm=mock_llm,
            embedder=mock_embedder,
            config=config,
        )

        with patch.object(router._parser_chain, "parse") as mock_parse, \
             patch.object(router._inspector, "classify") as mock_classify:
            mock_parse.return_value = (sample_file_profile, [])
            mock_classify.return_value = tabular_classification

            result = router.process("/tmp/test.xlsx")

        assert result.chunks_created == 2
        assert result.tables_created == 1
        mock_processor.process.assert_called_once()

    @patch("ingestkit_excel.router.compute_ingest_key")
    def test_tabular_full_flow(
        self,
        mock_ingest_key,
        mock_vector_store,
        mock_structured_db,
        mock_llm,
        mock_embedder,
        config,
        sample_file_profile,
        tabular_classification,
        sample_processing_result,
    ):
        """End-to-end test of full tabular flow with mocked internals."""
        mock_ingest_key.return_value = _make_ingest_key()

        router = ExcelRouter(
            vector_store=mock_vector_store,
            structured_db=mock_structured_db,
            llm=mock_llm,
            embedder=mock_embedder,
            config=config,
        )

        with patch.object(router._parser_chain, "parse") as mock_parse, \
             patch.object(router._inspector, "classify") as mock_classify, \
             patch.object(router._structured_db_processor, "process") as mock_proc:
            mock_parse.return_value = (sample_file_profile, [])
            mock_classify.return_value = tabular_classification
            mock_proc.return_value = sample_processing_result

            result = router.process("/tmp/test.xlsx")

        assert isinstance(result, ProcessingResult)
        assert result.ingest_key == "abc123"  # from sample_processing_result
        assert result.tenant_id == "test_tenant"
        mock_proc.assert_called_once()

        # Verify processor was called with correct args
        call_kwargs = mock_proc.call_args
        assert call_kwargs[1]["file_path"] == "/tmp/test.xlsx"
        assert call_kwargs[1]["ingest_key"] == _make_ingest_key().key
        assert call_kwargs[1]["profile"] is sample_file_profile


# ---------------------------------------------------------------------------
# Tests: Happy path — Path B (formatted document)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRouterPathB:
    """Test routing to Path B (TextSerializer) for formatted documents."""

    @patch("ingestkit_excel.router.compute_ingest_key")
    def test_formatted_routes_to_path_b(
        self,
        mock_ingest_key,
        mock_vector_store,
        mock_structured_db,
        mock_llm,
        mock_embedder,
        config,
        sample_file_profile,
        formatted_classification,
        sample_processing_result,
    ):
        """A file classified as FORMATTED_DOCUMENT routes to TextSerializer."""
        mock_ingest_key.return_value = _make_ingest_key()

        # Adjust the processing result for path B
        path_b_result = sample_processing_result.model_copy(
            update={
                "ingestion_method": IngestionMethod.TEXT_SERIALIZATION,
                "tables_created": 0,
                "tables": [],
            }
        )

        router = ExcelRouter(
            vector_store=mock_vector_store,
            structured_db=mock_structured_db,
            llm=mock_llm,
            embedder=mock_embedder,
            config=config,
        )

        with patch.object(router._parser_chain, "parse") as mock_parse, \
             patch.object(router._inspector, "classify") as mock_classify, \
             patch.object(router._text_serializer, "process") as mock_proc:
            mock_parse.return_value = (sample_file_profile, [])
            mock_classify.return_value = formatted_classification
            mock_proc.return_value = path_b_result

            result = router.process("/tmp/test.xlsx")

        assert isinstance(result, ProcessingResult)
        mock_proc.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: Happy path — Path C (hybrid)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRouterPathC:
    """Test routing to Path C (HybridSplitter) for hybrid files."""

    @patch("ingestkit_excel.router.compute_ingest_key")
    def test_hybrid_routes_to_path_c(
        self,
        mock_ingest_key,
        mock_vector_store,
        mock_structured_db,
        mock_llm,
        mock_embedder,
        config,
        sample_file_profile,
        hybrid_classification,
        sample_processing_result,
    ):
        """A file classified as HYBRID routes to HybridSplitter."""
        mock_ingest_key.return_value = _make_ingest_key()

        path_c_result = sample_processing_result.model_copy(
            update={
                "ingestion_method": IngestionMethod.HYBRID_SPLIT,
            }
        )

        router = ExcelRouter(
            vector_store=mock_vector_store,
            structured_db=mock_structured_db,
            llm=mock_llm,
            embedder=mock_embedder,
            config=config,
        )

        with patch.object(router._parser_chain, "parse") as mock_parse, \
             patch.object(router._inspector, "classify") as mock_classify, \
             patch.object(router._hybrid_splitter, "process") as mock_proc:
            mock_parse.return_value = (sample_file_profile, [])
            mock_classify.return_value = hybrid_classification
            mock_proc.return_value = path_c_result

            result = router.process("/tmp/test.xlsx")

        assert isinstance(result, ProcessingResult)
        mock_proc.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: Tier escalation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTierEscalation:
    """Test classification tier escalation logic."""

    @patch("ingestkit_excel.router.compute_ingest_key")
    def test_tier1_inconclusive_escalates_to_tier2(
        self,
        mock_ingest_key,
        mock_vector_store,
        mock_structured_db,
        mock_llm,
        mock_embedder,
        config,
        sample_file_profile,
        inconclusive_classification,
        tabular_classification,
        sample_processing_result,
    ):
        """When Tier 1 returns confidence 0.0, should escalate to Tier 2."""
        mock_ingest_key.return_value = _make_ingest_key()

        router = ExcelRouter(
            vector_store=mock_vector_store,
            structured_db=mock_structured_db,
            llm=mock_llm,
            embedder=mock_embedder,
            config=config,
        )

        # Tier 2 returns a confident result
        tier2_result = ClassificationResult(
            file_type=FileType.TABULAR_DATA,
            confidence=0.85,
            tier_used=ClassificationTier.LLM_BASIC,
            reasoning="LLM classified as tabular.",
        )

        with patch.object(router._parser_chain, "parse") as mock_parse, \
             patch.object(router._inspector, "classify") as mock_tier1, \
             patch.object(router._llm_classifier, "classify") as mock_tier2, \
             patch.object(router._structured_db_processor, "process") as mock_proc:
            mock_parse.return_value = (sample_file_profile, [])
            mock_tier1.return_value = inconclusive_classification
            mock_tier2.return_value = tier2_result
            mock_proc.return_value = sample_processing_result

            result = router.process("/tmp/test.xlsx")

        # Tier 2 was called with LLM_BASIC
        mock_tier2.assert_called_once_with(
            sample_file_profile, ClassificationTier.LLM_BASIC
        )
        mock_proc.assert_called_once()

    @patch("ingestkit_excel.router.compute_ingest_key")
    def test_tier2_low_confidence_escalates_to_tier3(
        self,
        mock_ingest_key,
        mock_vector_store,
        mock_structured_db,
        mock_llm,
        mock_embedder,
        config,
        sample_file_profile,
        inconclusive_classification,
        sample_processing_result,
    ):
        """When Tier 2 confidence < threshold, should escalate to Tier 3."""
        mock_ingest_key.return_value = _make_ingest_key()

        # Tier 2 returns low confidence
        tier2_result = ClassificationResult(
            file_type=FileType.FORMATTED_DOCUMENT,
            confidence=0.4,  # below default threshold of 0.6
            tier_used=ClassificationTier.LLM_BASIC,
            reasoning="Uncertain classification.",
        )

        # Tier 3 returns confident result
        tier3_result = ClassificationResult(
            file_type=FileType.FORMATTED_DOCUMENT,
            confidence=0.8,
            tier_used=ClassificationTier.LLM_REASONING,
            reasoning="Reasoning model classified as formatted.",
        )

        router = ExcelRouter(
            vector_store=mock_vector_store,
            structured_db=mock_structured_db,
            llm=mock_llm,
            embedder=mock_embedder,
            config=config,
        )

        path_b_result = sample_processing_result.model_copy(
            update={
                "ingestion_method": IngestionMethod.TEXT_SERIALIZATION,
                "tables_created": 0,
                "tables": [],
            }
        )

        with patch.object(router._parser_chain, "parse") as mock_parse, \
             patch.object(router._inspector, "classify") as mock_tier1, \
             patch.object(router._llm_classifier, "classify") as mock_llm_classify, \
             patch.object(router._text_serializer, "process") as mock_proc:
            mock_parse.return_value = (sample_file_profile, [])
            mock_tier1.return_value = inconclusive_classification
            mock_llm_classify.side_effect = [tier2_result, tier3_result]
            mock_proc.return_value = path_b_result

            result = router.process("/tmp/test.xlsx")

        # LLM classifier called twice (Tier 2, then Tier 3)
        assert mock_llm_classify.call_count == 2
        mock_llm_classify.assert_any_call(
            sample_file_profile, ClassificationTier.LLM_BASIC
        )
        mock_llm_classify.assert_any_call(
            sample_file_profile, ClassificationTier.LLM_REASONING
        )

    @patch("ingestkit_excel.router.compute_ingest_key")
    def test_tier3_disabled_stops_at_tier2(
        self,
        mock_ingest_key,
        mock_vector_store,
        mock_structured_db,
        mock_llm,
        mock_embedder,
        sample_file_profile,
        inconclusive_classification,
        sample_processing_result,
    ):
        """When enable_tier3=False, should not escalate beyond Tier 2."""
        config = ExcelProcessorConfig(
            tenant_id="test_tenant", enable_tier3=False
        )
        mock_ingest_key.return_value = _make_ingest_key()

        tier2_result = ClassificationResult(
            file_type=FileType.TABULAR_DATA,
            confidence=0.4,  # below threshold but Tier 3 disabled
            tier_used=ClassificationTier.LLM_BASIC,
            reasoning="Low confidence classification.",
        )

        router = ExcelRouter(
            vector_store=mock_vector_store,
            structured_db=mock_structured_db,
            llm=mock_llm,
            embedder=mock_embedder,
            config=config,
        )

        with patch.object(router._parser_chain, "parse") as mock_parse, \
             patch.object(router._inspector, "classify") as mock_tier1, \
             patch.object(router._llm_classifier, "classify") as mock_llm_classify, \
             patch.object(router._structured_db_processor, "process") as mock_proc:
            mock_parse.return_value = (sample_file_profile, [])
            mock_tier1.return_value = inconclusive_classification
            mock_llm_classify.return_value = tier2_result
            mock_proc.return_value = sample_processing_result

            result = router.process("/tmp/test.xlsx")

        # LLM classifier called only once (Tier 2)
        mock_llm_classify.assert_called_once_with(
            sample_file_profile, ClassificationTier.LLM_BASIC
        )

    @patch("ingestkit_excel.router.compute_ingest_key")
    def test_tier1_confident_skips_llm(
        self,
        mock_ingest_key,
        mock_vector_store,
        mock_structured_db,
        mock_llm,
        mock_embedder,
        config,
        sample_file_profile,
        tabular_classification,
        sample_processing_result,
    ):
        """When Tier 1 is confident (>0.0), should NOT call LLM classifier."""
        mock_ingest_key.return_value = _make_ingest_key()

        router = ExcelRouter(
            vector_store=mock_vector_store,
            structured_db=mock_structured_db,
            llm=mock_llm,
            embedder=mock_embedder,
            config=config,
        )

        with patch.object(router._parser_chain, "parse") as mock_parse, \
             patch.object(router._inspector, "classify") as mock_tier1, \
             patch.object(router._llm_classifier, "classify") as mock_llm_classify, \
             patch.object(router._structured_db_processor, "process") as mock_proc:
            mock_parse.return_value = (sample_file_profile, [])
            mock_tier1.return_value = tabular_classification
            mock_proc.return_value = sample_processing_result

            result = router.process("/tmp/test.xlsx")

        mock_llm_classify.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: Fail-closed
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFailClosed:
    """Test fail-closed behavior when classification is inconclusive."""

    @patch("ingestkit_excel.router.compute_ingest_key")
    def test_all_tiers_fail_returns_inconclusive(
        self,
        mock_ingest_key,
        mock_vector_store,
        mock_structured_db,
        mock_llm,
        mock_embedder,
        config,
        sample_file_profile,
        inconclusive_classification,
    ):
        """When all tiers return confidence 0.0, should return fail-closed result."""
        mock_ingest_key.return_value = _make_ingest_key()

        # All tiers fail
        tier2_fail = ClassificationResult(
            file_type=FileType.TABULAR_DATA,
            confidence=0.0,
            tier_used=ClassificationTier.LLM_BASIC,
            reasoning="LLM failed.",
        )
        tier3_fail = ClassificationResult(
            file_type=FileType.TABULAR_DATA,
            confidence=0.0,
            tier_used=ClassificationTier.LLM_REASONING,
            reasoning="Reasoning model failed.",
        )

        router = ExcelRouter(
            vector_store=mock_vector_store,
            structured_db=mock_structured_db,
            llm=mock_llm,
            embedder=mock_embedder,
            config=config,
        )

        with patch.object(router._parser_chain, "parse") as mock_parse, \
             patch.object(router._inspector, "classify") as mock_tier1, \
             patch.object(router._llm_classifier, "classify") as mock_llm_classify:
            mock_parse.return_value = (sample_file_profile, [])
            mock_tier1.return_value = inconclusive_classification
            mock_llm_classify.side_effect = [tier2_fail, tier3_fail]

            result = router.process("/tmp/test.xlsx")

        assert result.chunks_created == 0
        assert result.tables_created == 0
        assert result.tables == []
        assert ErrorCode.E_CLASSIFY_INCONCLUSIVE.value in result.errors
        assert any(
            e.code == ErrorCode.E_CLASSIFY_INCONCLUSIVE
            for e in result.error_details
        )
        assert result.ingest_key == _make_ingest_key().key
        assert result.ingest_run_id  # should have a UUID

    @patch("ingestkit_excel.router.compute_ingest_key")
    def test_fail_closed_has_correct_structure(
        self,
        mock_ingest_key,
        mock_vector_store,
        mock_structured_db,
        mock_llm,
        mock_embedder,
        config,
        sample_file_profile,
        inconclusive_classification,
    ):
        """Fail-closed result should have all required fields populated."""
        mock_ingest_key.return_value = _make_ingest_key()

        tier2_fail = ClassificationResult(
            file_type=FileType.TABULAR_DATA,
            confidence=0.0,
            tier_used=ClassificationTier.LLM_BASIC,
            reasoning="Failed.",
        )
        tier3_fail = ClassificationResult(
            file_type=FileType.TABULAR_DATA,
            confidence=0.0,
            tier_used=ClassificationTier.LLM_REASONING,
            reasoning="Also failed.",
        )

        router = ExcelRouter(
            vector_store=mock_vector_store,
            structured_db=mock_structured_db,
            llm=mock_llm,
            embedder=mock_embedder,
            config=config,
        )

        with patch.object(router._parser_chain, "parse") as mock_parse, \
             patch.object(router._inspector, "classify") as mock_tier1, \
             patch.object(router._llm_classifier, "classify") as mock_llm_classify:
            mock_parse.return_value = (sample_file_profile, [])
            mock_tier1.return_value = inconclusive_classification
            mock_llm_classify.side_effect = [tier2_fail, tier3_fail]

            result = router.process("/tmp/test.xlsx")

        # Verify all required fields present
        assert result.file_path == "/tmp/test.xlsx"
        assert result.ingest_key is not None
        assert result.ingest_run_id is not None
        assert result.tenant_id == "test_tenant"
        assert result.parse_result is not None
        assert result.classification_result is not None
        assert result.classification is not None
        assert result.processing_time_seconds > 0


# ---------------------------------------------------------------------------
# Tests: Parse error merging
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParseErrorMerging:
    """Test that parse-stage errors are merged into the final result."""

    @patch("ingestkit_excel.router.compute_ingest_key")
    def test_parse_warnings_merged_into_result(
        self,
        mock_ingest_key,
        mock_vector_store,
        mock_structured_db,
        mock_llm,
        mock_embedder,
        config,
        sample_file_profile,
        tabular_classification,
        sample_processing_result,
    ):
        """Parse fallback warnings should appear in result.warnings."""
        mock_ingest_key.return_value = _make_ingest_key()

        parse_errors = [
            IngestError(
                code=ErrorCode.W_PARSER_FALLBACK,
                message="Sheet 'Sheet1' parsed via pandas fallback.",
                sheet_name="Sheet1",
                stage="parse",
                recoverable=True,
            ),
        ]

        router = ExcelRouter(
            vector_store=mock_vector_store,
            structured_db=mock_structured_db,
            llm=mock_llm,
            embedder=mock_embedder,
            config=config,
        )

        with patch.object(router._parser_chain, "parse") as mock_parse, \
             patch.object(router._inspector, "classify") as mock_classify, \
             patch.object(router._structured_db_processor, "process") as mock_proc:
            mock_parse.return_value = (sample_file_profile, parse_errors)
            mock_classify.return_value = tabular_classification
            mock_proc.return_value = sample_processing_result

            result = router.process("/tmp/test.xlsx")

        assert ErrorCode.W_PARSER_FALLBACK.value in result.warnings
        assert any(
            e.code == ErrorCode.W_PARSER_FALLBACK
            for e in result.error_details
        )

    @patch("ingestkit_excel.router.compute_ingest_key")
    def test_parse_errors_merged_into_result(
        self,
        mock_ingest_key,
        mock_vector_store,
        mock_structured_db,
        mock_llm,
        mock_embedder,
        config,
        sample_file_profile,
        tabular_classification,
        sample_processing_result,
    ):
        """Parse errors (E_*) should appear in result.errors."""
        mock_ingest_key.return_value = _make_ingest_key()

        parse_errors = [
            IngestError(
                code=ErrorCode.E_PARSE_CORRUPT,
                message="All parsers failed for sheet 'Sheet2'.",
                sheet_name="Sheet2",
                stage="parse",
                recoverable=False,
            ),
        ]

        router = ExcelRouter(
            vector_store=mock_vector_store,
            structured_db=mock_structured_db,
            llm=mock_llm,
            embedder=mock_embedder,
            config=config,
        )

        with patch.object(router._parser_chain, "parse") as mock_parse, \
             patch.object(router._inspector, "classify") as mock_classify, \
             patch.object(router._structured_db_processor, "process") as mock_proc:
            mock_parse.return_value = (sample_file_profile, parse_errors)
            mock_classify.return_value = tabular_classification
            mock_proc.return_value = sample_processing_result

            result = router.process("/tmp/test.xlsx")

        assert ErrorCode.E_PARSE_CORRUPT.value in result.errors


# ---------------------------------------------------------------------------
# Tests: ParseStageResult building
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildParseStageResult:
    """Test _build_parse_stage_result helper."""

    def test_openpyxl_primary_parser(self, sample_file_profile):
        """When sheets use openpyxl, ParseStageResult should reflect that."""
        result = ExcelRouter._build_parse_stage_result(
            sample_file_profile, [], 0.1
        )
        assert result.parser_used == ParserUsed.OPENPYXL
        assert result.fallback_reason_code is None
        assert result.sheets_parsed == 1
        assert result.sheets_skipped == 0
        assert result.parse_duration_seconds == 0.1

    def test_pandas_fallback_parser(self, sample_file_profile):
        """When first sheet used pandas fallback, record the fallback reason."""
        profile = sample_file_profile.model_copy(deep=True)
        profile.sheets[0] = profile.sheets[0].model_copy(
            update={"parser_used": ParserUsed.PANDAS_FALLBACK}
        )
        result = ExcelRouter._build_parse_stage_result(profile, [], 0.2)
        assert result.parser_used == ParserUsed.PANDAS_FALLBACK
        assert result.fallback_reason_code == ErrorCode.E_PARSE_OPENPYXL_FAIL.value

    def test_raw_text_fallback_parser(self, sample_file_profile):
        """When first sheet used raw_text fallback, record the fallback reason."""
        profile = sample_file_profile.model_copy(deep=True)
        profile.sheets[0] = profile.sheets[0].model_copy(
            update={"parser_used": ParserUsed.RAW_TEXT_FALLBACK}
        )
        result = ExcelRouter._build_parse_stage_result(profile, [], 0.3)
        assert result.parser_used == ParserUsed.RAW_TEXT_FALLBACK
        assert result.fallback_reason_code == ErrorCode.E_PARSE_PANDAS_FAIL.value

    def test_skipped_sheets_tracked(self, sample_file_profile):
        """Skipped sheets should be counted and reasons recorded."""
        errors = [
            IngestError(
                code=ErrorCode.W_SHEET_SKIPPED_CHART,
                message="Chart sheet skipped.",
                sheet_name="Charts",
                stage="parse",
                recoverable=True,
            ),
            IngestError(
                code=ErrorCode.W_SHEET_SKIPPED_HIDDEN,
                message="Hidden sheet skipped.",
                sheet_name="Hidden",
                stage="parse",
                recoverable=True,
            ),
        ]
        result = ExcelRouter._build_parse_stage_result(
            sample_file_profile, errors, 0.1
        )
        assert result.sheets_skipped == 2
        assert "Charts" in result.skipped_reasons
        assert "Hidden" in result.skipped_reasons
        assert result.skipped_reasons["Charts"] == ErrorCode.W_SHEET_SKIPPED_CHART.value

    def test_empty_profile_defaults(self):
        """An empty profile (no sheets) should default to openpyxl parser."""
        profile = FileProfile(
            file_path="/tmp/empty.xlsx",
            file_size_bytes=0,
            sheet_count=0,
            sheet_names=[],
            sheets=[],
            has_password_protected_sheets=False,
            has_chart_only_sheets=False,
            total_merged_cells=0,
            total_rows=0,
            content_hash="empty",
        )
        result = ExcelRouter._build_parse_stage_result(profile, [], 0.01)
        assert result.parser_used == ParserUsed.OPENPYXL
        assert result.sheets_parsed == 0


# ---------------------------------------------------------------------------
# Tests: Batch processing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProcessBatch:
    """Test process_batch."""

    @patch("ingestkit_excel.router.compute_ingest_key")
    def test_batch_processes_all_files(
        self,
        mock_ingest_key,
        mock_vector_store,
        mock_structured_db,
        mock_llm,
        mock_embedder,
        config,
        sample_file_profile,
        tabular_classification,
        sample_processing_result,
    ):
        """process_batch should return one result per input file."""
        mock_ingest_key.return_value = _make_ingest_key()

        router = ExcelRouter(
            vector_store=mock_vector_store,
            structured_db=mock_structured_db,
            llm=mock_llm,
            embedder=mock_embedder,
            config=config,
        )

        with patch.object(router._parser_chain, "parse") as mock_parse, \
             patch.object(router._inspector, "classify") as mock_classify, \
             patch.object(router._structured_db_processor, "process") as mock_proc:
            mock_parse.return_value = (sample_file_profile, [])
            mock_classify.return_value = tabular_classification
            mock_proc.return_value = sample_processing_result

            results = router.process_batch(["/tmp/a.xlsx", "/tmp/b.xlsx", "/tmp/c.xlsx"])

        assert len(results) == 3
        assert mock_proc.call_count == 3

    @patch("ingestkit_excel.router.compute_ingest_key")
    def test_batch_empty_list(
        self,
        mock_ingest_key,
        mock_vector_store,
        mock_structured_db,
        mock_llm,
        mock_embedder,
        config,
    ):
        """process_batch with empty list should return empty list."""
        router = ExcelRouter(
            vector_store=mock_vector_store,
            structured_db=mock_structured_db,
            llm=mock_llm,
            embedder=mock_embedder,
            config=config,
        )
        results = router.process_batch([])
        assert results == []


# ---------------------------------------------------------------------------
# Tests: PII-safe logging
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPIISafeLogging:
    """Test that logging is PII-safe."""

    @patch("ingestkit_excel.router.compute_ingest_key")
    def test_info_log_after_processing(
        self,
        mock_ingest_key,
        mock_vector_store,
        mock_structured_db,
        mock_llm,
        mock_embedder,
        config,
        sample_file_profile,
        tabular_classification,
        sample_processing_result,
        caplog,
    ):
        """INFO log after processing should include truncated key, not raw data."""
        mock_ingest_key.return_value = _make_ingest_key()

        router = ExcelRouter(
            vector_store=mock_vector_store,
            structured_db=mock_structured_db,
            llm=mock_llm,
            embedder=mock_embedder,
            config=config,
        )

        with patch.object(router._parser_chain, "parse") as mock_parse, \
             patch.object(router._inspector, "classify") as mock_classify, \
             patch.object(router._structured_db_processor, "process") as mock_proc:
            mock_parse.return_value = (sample_file_profile, [])
            mock_classify.return_value = tabular_classification
            mock_proc.return_value = sample_processing_result

            import logging
            with caplog.at_level(logging.INFO, logger="ingestkit_excel"):
                result = router.process("/tmp/test.xlsx")

        # Should have an info log with "Processed"
        info_messages = [r.message for r in caplog.records if r.levelname == "INFO"]
        processed_msgs = [m for m in info_messages if "Processed" in m]
        assert len(processed_msgs) >= 1

        # The key should be truncated to 16 chars
        key_16 = _make_ingest_key().key[:16]
        assert any(key_16 in m for m in processed_msgs)


# ---------------------------------------------------------------------------
# Tests: Processor selection
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProcessorSelection:
    """Test _select_processor picks the right processor."""

    def test_tabular_selects_structured_db(
        self, mock_vector_store, mock_structured_db, mock_llm, mock_embedder, config
    ):
        router = ExcelRouter(
            vector_store=mock_vector_store,
            structured_db=mock_structured_db,
            llm=mock_llm,
            embedder=mock_embedder,
            config=config,
        )
        processor = router._select_processor(FileType.TABULAR_DATA)
        assert processor is router._structured_db_processor

    def test_formatted_selects_text_serializer(
        self, mock_vector_store, mock_structured_db, mock_llm, mock_embedder, config
    ):
        router = ExcelRouter(
            vector_store=mock_vector_store,
            structured_db=mock_structured_db,
            llm=mock_llm,
            embedder=mock_embedder,
            config=config,
        )
        processor = router._select_processor(FileType.FORMATTED_DOCUMENT)
        assert processor is router._text_serializer

    def test_hybrid_selects_hybrid_splitter(
        self, mock_vector_store, mock_structured_db, mock_llm, mock_embedder, config
    ):
        router = ExcelRouter(
            vector_store=mock_vector_store,
            structured_db=mock_structured_db,
            llm=mock_llm,
            embedder=mock_embedder,
            config=config,
        )
        processor = router._select_processor(FileType.HYBRID)
        assert processor is router._hybrid_splitter


# ---------------------------------------------------------------------------
# Tests: source_uri passthrough
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSourceUri:
    """Test that source_uri is passed through to compute_ingest_key."""

    @patch("ingestkit_excel.router.compute_ingest_key")
    def test_source_uri_passed(
        self,
        mock_ingest_key,
        mock_vector_store,
        mock_structured_db,
        mock_llm,
        mock_embedder,
        config,
        sample_file_profile,
        tabular_classification,
        sample_processing_result,
    ):
        """When source_uri is provided, it should be passed to compute_ingest_key."""
        mock_ingest_key.return_value = _make_ingest_key()

        router = ExcelRouter(
            vector_store=mock_vector_store,
            structured_db=mock_structured_db,
            llm=mock_llm,
            embedder=mock_embedder,
            config=config,
        )

        with patch.object(router._parser_chain, "parse") as mock_parse, \
             patch.object(router._inspector, "classify") as mock_classify, \
             patch.object(router._structured_db_processor, "process") as mock_proc:
            mock_parse.return_value = (sample_file_profile, [])
            mock_classify.return_value = tabular_classification
            mock_proc.return_value = sample_processing_result

            result = router.process("/tmp/test.xlsx", source_uri="s3://bucket/test.xlsx")

        mock_ingest_key.assert_called_once_with(
            file_path="/tmp/test.xlsx",
            parser_version=config.parser_version,
            tenant_id=config.tenant_id,
            source_uri="s3://bucket/test.xlsx",
        )


# ---------------------------------------------------------------------------
# Tests: create_default_router factory
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreateDefaultRouter:
    """Test the create_default_router factory function."""

    def test_factory_with_all_overrides(
        self, mock_vector_store, mock_structured_db, mock_llm, mock_embedder, config
    ):
        """When all backends are overridden, should use them directly."""
        router = create_default_router(
            vector_store=mock_vector_store,
            structured_db=mock_structured_db,
            llm=mock_llm,
            embedder=mock_embedder,
            config=config,
        )
        assert isinstance(router, ExcelRouter)
        assert router._config is config

    def test_factory_with_config_kwargs(
        self, mock_vector_store, mock_structured_db, mock_llm, mock_embedder
    ):
        """Config kwargs passed as flat arguments should build ExcelProcessorConfig."""
        router = create_default_router(
            vector_store=mock_vector_store,
            structured_db=mock_structured_db,
            llm=mock_llm,
            embedder=mock_embedder,
            tenant_id="from_kwargs",
        )
        assert router._config.tenant_id == "from_kwargs"


# ---------------------------------------------------------------------------
# Tests: file_type_to_path mapping
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFileTypeToPath:
    """Test the _file_type_to_path helper."""

    def test_tabular(self):
        assert ExcelRouter._file_type_to_path(FileType.TABULAR_DATA) == "Path A (sql_agent)"

    def test_formatted(self):
        assert ExcelRouter._file_type_to_path(FileType.FORMATTED_DOCUMENT) == "Path B (text_serialization)"

    def test_hybrid(self):
        assert ExcelRouter._file_type_to_path(FileType.HYBRID) == "Path C (hybrid_split)"


# ---------------------------------------------------------------------------
# Tests: IngestKey and ingest_run_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIngestKeyAndRunId:
    """Test that ingest_key and ingest_run_id are correctly propagated."""

    @patch("ingestkit_excel.router.compute_ingest_key")
    def test_ingest_key_uses_key_property(
        self,
        mock_ingest_key,
        mock_vector_store,
        mock_structured_db,
        mock_llm,
        mock_embedder,
        config,
        sample_file_profile,
        tabular_classification,
        sample_processing_result,
    ):
        """The hex string from IngestKey.key should be used, not the object."""
        ik = _make_ingest_key()
        mock_ingest_key.return_value = ik

        router = ExcelRouter(
            vector_store=mock_vector_store,
            structured_db=mock_structured_db,
            llm=mock_llm,
            embedder=mock_embedder,
            config=config,
        )

        with patch.object(router._parser_chain, "parse") as mock_parse, \
             patch.object(router._inspector, "classify") as mock_classify, \
             patch.object(router._structured_db_processor, "process") as mock_proc:
            mock_parse.return_value = (sample_file_profile, [])
            mock_classify.return_value = tabular_classification
            mock_proc.return_value = sample_processing_result

            result = router.process("/tmp/test.xlsx")

        # Verify processor received the .key hex string
        call_kwargs = mock_proc.call_args[1]
        assert call_kwargs["ingest_key"] == ik.key
        # ingest_run_id should be a UUID string
        assert len(call_kwargs["ingest_run_id"]) == 36  # UUID4 format


# ---------------------------------------------------------------------------
# Tests: Classification result propagation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClassificationResultPropagation:
    """Test that classification stage results are properly built and passed."""

    @patch("ingestkit_excel.router.compute_ingest_key")
    def test_classification_stage_result_built_correctly(
        self,
        mock_ingest_key,
        mock_vector_store,
        mock_structured_db,
        mock_llm,
        mock_embedder,
        config,
        sample_file_profile,
        tabular_classification,
        sample_processing_result,
    ):
        """ClassificationStageResult should capture tier, type, confidence."""
        mock_ingest_key.return_value = _make_ingest_key()

        router = ExcelRouter(
            vector_store=mock_vector_store,
            structured_db=mock_structured_db,
            llm=mock_llm,
            embedder=mock_embedder,
            config=config,
        )

        with patch.object(router._parser_chain, "parse") as mock_parse, \
             patch.object(router._inspector, "classify") as mock_classify, \
             patch.object(router._structured_db_processor, "process") as mock_proc:
            mock_parse.return_value = (sample_file_profile, [])
            mock_classify.return_value = tabular_classification
            mock_proc.return_value = sample_processing_result

            result = router.process("/tmp/test.xlsx")

        call_kwargs = mock_proc.call_args[1]
        cr = call_kwargs["classification_result"]
        assert isinstance(cr, ClassificationStageResult)
        assert cr.tier_used == ClassificationTier.RULE_BASED
        assert cr.file_type == FileType.TABULAR_DATA
        assert cr.confidence == 0.9
        assert cr.classification_duration_seconds > 0
