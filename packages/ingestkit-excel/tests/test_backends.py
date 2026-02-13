"""Tests for concrete backend implementations.

Covers SQLite (real in-memory DB), Qdrant (mocked client), Ollama LLM and
Embedding (mocked httpx), stubs (Milvus, Postgres), and runtime protocol
conformance checks.
"""

from __future__ import annotations

import json
import sqlite3
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from ingestkit_core.models import BaseChunkMetadata, ChunkPayload
from ingestkit_excel.backends.milvus import MilvusVectorStore
from ingestkit_excel.backends.postgres import PostgresStructuredDB
from ingestkit_excel.backends.sqlite import SQLiteStructuredDB
from ingestkit_excel.config import ExcelProcessorConfig
from ingestkit_excel.protocols import (
    EmbeddingBackend,
    LLMBackend,
    StructuredDBBackend,
    VectorStoreBackend,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    id_: str = "chunk-001",
    text: str = "sample text",
    vector: list[float] | None = None,
) -> ChunkPayload:
    """Create a ChunkPayload for testing."""
    return ChunkPayload(
        id=id_,
        text=text,
        vector=vector or [0.1, 0.2, 0.3],
        metadata=BaseChunkMetadata(
            source_uri="file:///tmp/test.xlsx",
            source_format="xlsx",
            ingestion_method="sql_agent",
            parser_version="ingestkit_excel:1.0.0",
            chunk_index=0,
            chunk_hash="abc123",
            ingest_key="key123",
            ingest_run_id="run-001",
            tenant_id="test_tenant",
            table_name="employees",
        ),
    )


def _make_config(**overrides) -> ExcelProcessorConfig:
    """Create a config with fast retry settings for tests."""
    defaults = {
        "backend_timeout_seconds": 5.0,
        "backend_max_retries": 1,
        "backend_backoff_base": 0.01,  # near-zero for fast tests
    }
    defaults.update(overrides)
    return ExcelProcessorConfig(**defaults)


# ===========================================================================
# TestSQLiteStructuredDB
# ===========================================================================


class TestSQLiteStructuredDB:
    """Tests for SQLiteStructuredDB using a real in-memory SQLite database."""

    @pytest.fixture()
    def db(self) -> SQLiteStructuredDB:
        return SQLiteStructuredDB(db_path=":memory:")

    @pytest.fixture()
    def sample_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "age": [30, 25, 35],
                "salary": [70000.0, 55000.0, 90000.0],
            }
        )

    # --- create_table_from_dataframe ---

    @pytest.mark.unit
    def test_create_table_writes_data(
        self, db: SQLiteStructuredDB, sample_df: pd.DataFrame
    ) -> None:
        db.create_table_from_dataframe("employees", sample_df)

        assert db.table_exists("employees")

    @pytest.mark.unit
    def test_create_table_replaces_existing(
        self, db: SQLiteStructuredDB, sample_df: pd.DataFrame
    ) -> None:
        db.create_table_from_dataframe("employees", sample_df)
        # Create again with different data
        new_df = pd.DataFrame({"name": ["Dave"], "age": [40], "salary": [80000.0]})
        db.create_table_from_dataframe("employees", new_df)

        # Should have only 1 row now (replaced, not appended)
        cursor = db._conn.execute("SELECT COUNT(*) FROM employees")
        assert cursor.fetchone()[0] == 1

    @pytest.mark.unit
    def test_create_table_preserves_all_rows(
        self, db: SQLiteStructuredDB, sample_df: pd.DataFrame
    ) -> None:
        db.create_table_from_dataframe("employees", sample_df)

        cursor = db._conn.execute("SELECT COUNT(*) FROM employees")
        assert cursor.fetchone()[0] == 3

    # --- drop_table ---

    @pytest.mark.unit
    def test_drop_table_removes_table(
        self, db: SQLiteStructuredDB, sample_df: pd.DataFrame
    ) -> None:
        db.create_table_from_dataframe("employees", sample_df)
        db.drop_table("employees")

        assert not db.table_exists("employees")

    @pytest.mark.unit
    def test_drop_table_nonexistent_is_noop(
        self, db: SQLiteStructuredDB
    ) -> None:
        # Should not raise
        db.drop_table("nonexistent")

    # --- table_exists ---

    @pytest.mark.unit
    def test_table_exists_returns_false_for_missing(
        self, db: SQLiteStructuredDB
    ) -> None:
        assert not db.table_exists("nonexistent")

    @pytest.mark.unit
    def test_table_exists_returns_true_after_create(
        self, db: SQLiteStructuredDB, sample_df: pd.DataFrame
    ) -> None:
        db.create_table_from_dataframe("employees", sample_df)
        assert db.table_exists("employees")

    # --- get_table_schema ---

    @pytest.mark.unit
    def test_get_table_schema_returns_columns(
        self, db: SQLiteStructuredDB, sample_df: pd.DataFrame
    ) -> None:
        db.create_table_from_dataframe("employees", sample_df)

        schema = db.get_table_schema("employees")

        assert "name" in schema
        assert "age" in schema
        assert "salary" in schema

    @pytest.mark.unit
    def test_get_table_schema_empty_for_missing_table(
        self, db: SQLiteStructuredDB
    ) -> None:
        schema = db.get_table_schema("nonexistent")
        assert schema == {}

    # --- get_connection_uri ---

    @pytest.mark.unit
    def test_get_connection_uri_memory(self, db: SQLiteStructuredDB) -> None:
        assert db.get_connection_uri() == "sqlite:///:memory:"

    @pytest.mark.unit
    def test_get_connection_uri_file_path(self) -> None:
        db = SQLiteStructuredDB(db_path="/tmp/test.db")
        assert db.get_connection_uri() == "sqlite:////tmp/test.db"
        db.close()

    # --- close ---

    @pytest.mark.unit
    def test_close_prevents_further_queries(
        self, db: SQLiteStructuredDB, sample_df: pd.DataFrame
    ) -> None:
        db.create_table_from_dataframe("employees", sample_df)
        db.close()

        with pytest.raises(sqlite3.ProgrammingError):
            db.table_exists("employees")


# ===========================================================================
# TestQdrantVectorStore
# ===========================================================================


class TestQdrantVectorStore:
    """Tests for QdrantVectorStore using a mocked qdrant_client."""

    @pytest.fixture()
    def mock_qdrant_client(self):
        """Create a mock QdrantClient."""
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        return mock_client

    @pytest.fixture()
    def store(self, mock_qdrant_client):
        """Create a QdrantVectorStore with mocked client."""
        config = _make_config()
        with patch("qdrant_client.QdrantClient", return_value=mock_qdrant_client):
            from ingestkit_excel.backends.qdrant import QdrantVectorStore

            store = QdrantVectorStore(
                url="http://localhost:6333",
                collection_prefix="test",
                config=config,
            )
        return store

    # --- ensure_collection ---

    @pytest.mark.unit
    def test_ensure_collection_creates_when_not_exists(
        self, store, mock_qdrant_client
    ) -> None:
        store.ensure_collection("helpdesk", vector_size=768)

        mock_qdrant_client.collection_exists.assert_called_with("test_helpdesk")
        mock_qdrant_client.create_collection.assert_called_once()

    @pytest.mark.unit
    def test_ensure_collection_skips_when_exists(
        self, store, mock_qdrant_client
    ) -> None:
        mock_qdrant_client.collection_exists.return_value = True

        store.ensure_collection("helpdesk", vector_size=768)

        mock_qdrant_client.create_collection.assert_not_called()

    @pytest.mark.unit
    def test_ensure_collection_uses_prefix(
        self, store, mock_qdrant_client
    ) -> None:
        store.ensure_collection("mydata", vector_size=128)

        mock_qdrant_client.collection_exists.assert_called_with("test_mydata")

    # --- upsert_chunks ---

    @pytest.mark.unit
    def test_upsert_chunks_returns_count(
        self, store, mock_qdrant_client
    ) -> None:
        chunks = [_make_chunk(id_="c1"), _make_chunk(id_="c2")]

        count = store.upsert_chunks("helpdesk", chunks)

        assert count == 2
        mock_qdrant_client.upsert.assert_called_once()

    @pytest.mark.unit
    def test_upsert_chunks_empty_returns_zero(
        self, store, mock_qdrant_client
    ) -> None:
        count = store.upsert_chunks("helpdesk", [])

        assert count == 0
        mock_qdrant_client.upsert.assert_not_called()

    @pytest.mark.unit
    def test_upsert_chunks_passes_correct_collection_name(
        self, store, mock_qdrant_client
    ) -> None:
        chunks = [_make_chunk()]

        store.upsert_chunks("helpdesk", chunks)

        call_kwargs = mock_qdrant_client.upsert.call_args
        assert call_kwargs.kwargs["collection_name"] == "test_helpdesk"

    @pytest.mark.unit
    def test_upsert_chunks_converts_to_point_structs(
        self, store, mock_qdrant_client
    ) -> None:
        chunk = _make_chunk(id_="pt-1", text="hello", vector=[1.0, 2.0, 3.0])

        store.upsert_chunks("helpdesk", [chunk])

        call_kwargs = mock_qdrant_client.upsert.call_args
        points = call_kwargs.kwargs["points"]
        assert len(points) == 1
        assert points[0].id == "pt-1"
        assert points[0].vector == [1.0, 2.0, 3.0]
        assert points[0].payload["text"] == "hello"

    @pytest.mark.unit
    def test_upsert_chunks_retry_on_failure(
        self, store, mock_qdrant_client
    ) -> None:
        mock_qdrant_client.upsert.side_effect = [
            RuntimeError("network error"),
            None,  # success on retry
        ]

        count = store.upsert_chunks("helpdesk", [_make_chunk()])

        assert count == 1
        assert mock_qdrant_client.upsert.call_count == 2

    @pytest.mark.unit
    def test_upsert_chunks_raises_after_retries_exhausted(
        self, store, mock_qdrant_client
    ) -> None:
        mock_qdrant_client.upsert.side_effect = RuntimeError("persistent failure")

        with pytest.raises(ConnectionError, match="vector collection"):
            store.upsert_chunks("helpdesk", [_make_chunk()])

    # --- create_payload_index ---

    @pytest.mark.unit
    def test_create_payload_index_keyword(
        self, store, mock_qdrant_client
    ) -> None:
        store.create_payload_index("helpdesk", "tenant_id", "keyword")

        mock_qdrant_client.create_payload_index.assert_called_once()

    @pytest.mark.unit
    def test_create_payload_index_integer(
        self, store, mock_qdrant_client
    ) -> None:
        store.create_payload_index("helpdesk", "chunk_index", "integer")

        mock_qdrant_client.create_payload_index.assert_called_once()

    @pytest.mark.unit
    def test_create_payload_index_invalid_type_raises(
        self, store
    ) -> None:
        with pytest.raises(ValueError, match="Unsupported field_type"):
            store.create_payload_index("helpdesk", "field", "boolean")

    # --- delete_by_ids ---

    @pytest.mark.unit
    def test_delete_by_ids_returns_count(
        self, store, mock_qdrant_client
    ) -> None:
        count = store.delete_by_ids("helpdesk", ["id1", "id2", "id3"])

        assert count == 3
        mock_qdrant_client.delete.assert_called_once()

    @pytest.mark.unit
    def test_delete_by_ids_empty_returns_zero(
        self, store, mock_qdrant_client
    ) -> None:
        count = store.delete_by_ids("helpdesk", [])

        assert count == 0
        mock_qdrant_client.delete.assert_not_called()

    # --- prefix behavior ---

    @pytest.mark.unit
    def test_no_prefix_uses_raw_collection_name(self, mock_qdrant_client) -> None:
        config = _make_config()
        with patch("qdrant_client.QdrantClient", return_value=mock_qdrant_client):
            from ingestkit_excel.backends.qdrant import QdrantVectorStore

            store = QdrantVectorStore(
                url="http://localhost:6333",
                collection_prefix="",
                config=config,
            )
        store.ensure_collection("helpdesk", vector_size=768)

        mock_qdrant_client.collection_exists.assert_called_with("helpdesk")


# ===========================================================================
# TestOllamaLLM
# ===========================================================================


class TestOllamaLLM:
    """Tests for OllamaLLM using mocked httpx."""

    @pytest.fixture()
    def config(self) -> ExcelProcessorConfig:
        return _make_config()

    @pytest.fixture()
    def mock_response(self):
        """Create a mock httpx response."""
        resp = Mock()
        resp.status_code = 200
        resp.raise_for_status = Mock()
        return resp

    # --- classify ---

    @pytest.mark.unit
    def test_classify_returns_parsed_json(self, config, mock_response) -> None:
        classify_result = {"type": "tabular_data", "confidence": 0.9, "reasoning": "test"}
        mock_response.json.return_value = {"response": json.dumps(classify_result)}

        with patch("httpx.post", return_value=mock_response):
            from ingestkit_excel.backends.ollama import OllamaLLM

            llm = OllamaLLM(config=config)
            result = llm.classify("test prompt", "qwen2.5:7b")

        assert result == classify_result

    @pytest.mark.unit
    def test_classify_posts_to_correct_endpoint(self, config, mock_response) -> None:
        mock_response.json.return_value = {"response": '{"type": "tabular_data", "confidence": 0.9, "reasoning": "ok"}'}

        with patch("httpx.post", return_value=mock_response) as mock_post:
            from ingestkit_excel.backends.ollama import OllamaLLM

            llm = OllamaLLM(base_url="http://myhost:11434", config=config)
            llm.classify("prompt", "model")

        call_args = mock_post.call_args
        assert call_args.args[0] == "http://myhost:11434/api/generate"

    @pytest.mark.unit
    def test_classify_sends_json_format(self, config, mock_response) -> None:
        mock_response.json.return_value = {"response": '{"type": "tabular_data", "confidence": 0.9, "reasoning": "ok"}'}

        with patch("httpx.post", return_value=mock_response) as mock_post:
            from ingestkit_excel.backends.ollama import OllamaLLM

            llm = OllamaLLM(config=config)
            llm.classify("prompt", "model")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["format"] == "json"
        assert payload["stream"] is False

    @pytest.mark.unit
    def test_classify_retries_on_malformed_json(self, config) -> None:
        resp1 = Mock()
        resp1.raise_for_status = Mock()
        resp1.json.return_value = {"response": "not json {{{"}

        resp2 = Mock()
        resp2.raise_for_status = Mock()
        resp2.json.return_value = {"response": '{"type": "tabular_data", "confidence": 0.9, "reasoning": "ok"}'}

        with patch("httpx.post", side_effect=[resp1, resp2]):
            from ingestkit_excel.backends.ollama import OllamaLLM

            llm = OllamaLLM(config=config)
            result = llm.classify("prompt", "model")

        assert result["type"] == "tabular_data"

    @pytest.mark.unit
    def test_classify_raises_after_two_malformed_json(self, config) -> None:
        resp = Mock()
        resp.raise_for_status = Mock()
        resp.json.return_value = {"response": "not json"}

        with patch("httpx.post", return_value=resp):
            from ingestkit_excel.backends.ollama import OllamaLLM

            llm = OllamaLLM(config=config)

            with pytest.raises(json.JSONDecodeError):
                llm.classify("prompt", "model")

    @pytest.mark.unit
    def test_classify_passes_temperature(self, config, mock_response) -> None:
        mock_response.json.return_value = {"response": '{"type": "tabular_data", "confidence": 0.9, "reasoning": "ok"}'}

        with patch("httpx.post", return_value=mock_response) as mock_post:
            from ingestkit_excel.backends.ollama import OllamaLLM

            llm = OllamaLLM(config=config)
            llm.classify("prompt", "model", temperature=0.3)

        payload = mock_post.call_args.kwargs["json"]
        assert payload["options"]["temperature"] == 0.3

    # --- generate ---

    @pytest.mark.unit
    def test_generate_returns_raw_text(self, config, mock_response) -> None:
        mock_response.json.return_value = {"response": "Generated text here"}

        with patch("httpx.post", return_value=mock_response):
            from ingestkit_excel.backends.ollama import OllamaLLM

            llm = OllamaLLM(config=config)
            result = llm.generate("prompt", "model")

        assert result == "Generated text here"

    @pytest.mark.unit
    def test_generate_does_not_use_json_format(self, config, mock_response) -> None:
        mock_response.json.return_value = {"response": "text"}

        with patch("httpx.post", return_value=mock_response) as mock_post:
            from ingestkit_excel.backends.ollama import OllamaLLM

            llm = OllamaLLM(config=config)
            llm.generate("prompt", "model")

        payload = mock_post.call_args.kwargs["json"]
        assert "format" not in payload

    # --- timeout handling ---

    @pytest.mark.unit
    def test_timeout_raises_timeout_error(self, config) -> None:
        import httpx

        with patch("httpx.post", side_effect=httpx.TimeoutException("timeout")):
            from ingestkit_excel.backends.ollama import OllamaLLM

            llm = OllamaLLM(config=config)

            with pytest.raises(TimeoutError, match="timed out"):
                llm.generate("prompt", "model")

    @pytest.mark.unit
    def test_connection_error_raises_connection_error(self, config) -> None:
        import httpx

        with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
            from ingestkit_excel.backends.ollama import OllamaLLM

            llm = OllamaLLM(config=config)

            with pytest.raises(ConnectionError, match="connection failed"):
                llm.generate("prompt", "model")


# ===========================================================================
# TestOllamaEmbedding
# ===========================================================================


class TestOllamaEmbedding:
    """Tests for OllamaEmbedding using mocked httpx."""

    @pytest.fixture()
    def config(self) -> ExcelProcessorConfig:
        return _make_config()

    # --- embed ---

    @pytest.mark.unit
    def test_embed_returns_vectors(self, config) -> None:
        mock_resp = Mock()
        mock_resp.raise_for_status = Mock()
        mock_resp.json.return_value = {
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        }

        with patch("httpx.post", return_value=mock_resp):
            from ingestkit_excel.backends.ollama import OllamaEmbedding

            emb = OllamaEmbedding(model="nomic-embed-text", embedding_dimension=3, config=config)
            result = emb.embed(["hello", "world"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]

    @pytest.mark.unit
    def test_embed_empty_returns_empty(self, config) -> None:
        from ingestkit_excel.backends.ollama import OllamaEmbedding

        emb = OllamaEmbedding(config=config)
        result = emb.embed([])

        assert result == []

    @pytest.mark.unit
    def test_embed_posts_to_correct_endpoint(self, config) -> None:
        mock_resp = Mock()
        mock_resp.raise_for_status = Mock()
        mock_resp.json.return_value = {"embeddings": [[0.1]]}

        with patch("httpx.post", return_value=mock_resp) as mock_post:
            from ingestkit_excel.backends.ollama import OllamaEmbedding

            emb = OllamaEmbedding(
                base_url="http://myhost:11434",
                model="nomic-embed-text",
                config=config,
            )
            emb.embed(["hello"])

        call_args = mock_post.call_args
        assert call_args.args[0] == "http://myhost:11434/api/embed"

    @pytest.mark.unit
    def test_embed_sends_model_name(self, config) -> None:
        mock_resp = Mock()
        mock_resp.raise_for_status = Mock()
        mock_resp.json.return_value = {"embeddings": [[0.1]]}

        with patch("httpx.post", return_value=mock_resp) as mock_post:
            from ingestkit_excel.backends.ollama import OllamaEmbedding

            emb = OllamaEmbedding(model="custom-embed", config=config)
            emb.embed(["hello"])

        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "custom-embed"

    @pytest.mark.unit
    def test_embed_timeout_raises_timeout_error(self, config) -> None:
        import httpx

        with patch("httpx.post", side_effect=httpx.TimeoutException("timeout")):
            from ingestkit_excel.backends.ollama import OllamaEmbedding

            emb = OllamaEmbedding(config=config)

            with pytest.raises(TimeoutError, match="embed timed out"):
                emb.embed(["hello"])

    @pytest.mark.unit
    def test_embed_connection_error_raises(self, config) -> None:
        import httpx

        with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
            from ingestkit_excel.backends.ollama import OllamaEmbedding

            emb = OllamaEmbedding(config=config)

            with pytest.raises(ConnectionError, match="embed connection failed"):
                emb.embed(["hello"])

    # --- dimension ---

    @pytest.mark.unit
    def test_dimension_returns_configured_value(self, config) -> None:
        from ingestkit_excel.backends.ollama import OllamaEmbedding

        emb = OllamaEmbedding(embedding_dimension=768, config=config)
        assert emb.dimension() == 768

    @pytest.mark.unit
    def test_dimension_default_is_768(self, config) -> None:
        from ingestkit_excel.backends.ollama import OllamaEmbedding

        emb = OllamaEmbedding(config=config)
        assert emb.dimension() == 768

    @pytest.mark.unit
    def test_dimension_custom_value(self, config) -> None:
        from ingestkit_excel.backends.ollama import OllamaEmbedding

        emb = OllamaEmbedding(embedding_dimension=384, config=config)
        assert emb.dimension() == 384


# ===========================================================================
# TestStubs
# ===========================================================================


class TestStubs:
    """Tests for stub backends that raise NotImplementedError."""

    # --- MilvusVectorStore ---

    @pytest.mark.unit
    def test_milvus_upsert_raises(self) -> None:
        store = MilvusVectorStore()
        with pytest.raises(NotImplementedError, match="stub"):
            store.upsert_chunks("col", [])

    @pytest.mark.unit
    def test_milvus_ensure_collection_raises(self) -> None:
        store = MilvusVectorStore()
        with pytest.raises(NotImplementedError, match="stub"):
            store.ensure_collection("col", 768)

    @pytest.mark.unit
    def test_milvus_create_payload_index_raises(self) -> None:
        store = MilvusVectorStore()
        with pytest.raises(NotImplementedError, match="stub"):
            store.create_payload_index("col", "field", "keyword")

    @pytest.mark.unit
    def test_milvus_delete_by_ids_raises(self) -> None:
        store = MilvusVectorStore()
        with pytest.raises(NotImplementedError, match="stub"):
            store.delete_by_ids("col", ["id1"])

    # --- PostgresStructuredDB ---

    @pytest.mark.unit
    def test_postgres_create_table_raises(self) -> None:
        db = PostgresStructuredDB()
        with pytest.raises(NotImplementedError, match="stub"):
            db.create_table_from_dataframe("tbl", pd.DataFrame())

    @pytest.mark.unit
    def test_postgres_drop_table_raises(self) -> None:
        db = PostgresStructuredDB()
        with pytest.raises(NotImplementedError, match="stub"):
            db.drop_table("tbl")

    @pytest.mark.unit
    def test_postgres_table_exists_raises(self) -> None:
        db = PostgresStructuredDB()
        with pytest.raises(NotImplementedError, match="stub"):
            db.table_exists("tbl")

    @pytest.mark.unit
    def test_postgres_get_table_schema_raises(self) -> None:
        db = PostgresStructuredDB()
        with pytest.raises(NotImplementedError, match="stub"):
            db.get_table_schema("tbl")

    @pytest.mark.unit
    def test_postgres_get_connection_uri_raises(self) -> None:
        db = PostgresStructuredDB()
        with pytest.raises(NotImplementedError, match="stub"):
            db.get_connection_uri()


# ===========================================================================
# TestProtocolConformance
# ===========================================================================


class TestProtocolConformance:
    """Tests that concrete backends satisfy their corresponding Protocol."""

    @pytest.mark.unit
    def test_sqlite_satisfies_structured_db_protocol(self) -> None:
        db = SQLiteStructuredDB()
        assert isinstance(db, StructuredDBBackend)

    @pytest.mark.unit
    def test_qdrant_satisfies_vector_store_protocol(self) -> None:
        config = _make_config()
        mock_client = MagicMock()
        with patch("qdrant_client.QdrantClient", return_value=mock_client):
            from ingestkit_excel.backends.qdrant import QdrantVectorStore

            store = QdrantVectorStore(config=config)

        assert isinstance(store, VectorStoreBackend)

    @pytest.mark.unit
    def test_ollama_llm_satisfies_llm_protocol(self) -> None:
        from ingestkit_excel.backends.ollama import OllamaLLM

        llm = OllamaLLM()
        assert isinstance(llm, LLMBackend)

    @pytest.mark.unit
    def test_ollama_embedding_satisfies_embedding_protocol(self) -> None:
        from ingestkit_excel.backends.ollama import OllamaEmbedding

        emb = OllamaEmbedding()
        assert isinstance(emb, EmbeddingBackend)

    @pytest.mark.unit
    def test_milvus_stub_satisfies_vector_store_protocol(self) -> None:
        store = MilvusVectorStore()
        assert isinstance(store, VectorStoreBackend)

    @pytest.mark.unit
    def test_postgres_stub_satisfies_structured_db_protocol(self) -> None:
        db = PostgresStructuredDB()
        assert isinstance(db, StructuredDBBackend)


# ===========================================================================
# TestBackendsPackageImport
# ===========================================================================


class TestBackendsPackageImport:
    """Tests for the backends package __init__.py imports."""

    @pytest.mark.unit
    def test_sqlite_importable(self) -> None:
        from ingestkit_excel.backends import SQLiteStructuredDB

        assert SQLiteStructuredDB is not None

    @pytest.mark.unit
    def test_stubs_importable(self) -> None:
        from ingestkit_excel.backends import MilvusVectorStore, PostgresStructuredDB

        assert MilvusVectorStore is not None
        assert PostgresStructuredDB is not None

    @pytest.mark.unit
    def test_qdrant_importable_when_available(self) -> None:
        from ingestkit_excel.backends import QdrantVectorStore

        # May be None if qdrant-client not installed, but import should not fail
        # If installed, should be the class
        if QdrantVectorStore is not None:
            assert hasattr(QdrantVectorStore, "upsert_chunks")

    @pytest.mark.unit
    def test_ollama_importable_when_available(self) -> None:
        from ingestkit_excel.backends import OllamaEmbedding, OllamaLLM

        # May be None if httpx not installed, but import should not fail
        if OllamaLLM is not None:
            assert hasattr(OllamaLLM, "classify")
        if OllamaEmbedding is not None:
            assert hasattr(OllamaEmbedding, "embed")

    @pytest.mark.unit
    def test_main_package_exports_backends(self) -> None:
        from ingestkit_excel import (
            MilvusVectorStore,
            PostgresStructuredDB,
            SQLiteStructuredDB,
        )

        assert SQLiteStructuredDB is not None
        assert MilvusVectorStore is not None
        assert PostgresStructuredDB is not None
