"""Unit tests for ingestkit_json.router -- orchestration."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from ingestkit_json.config import JSONProcessorConfig
from ingestkit_json.errors import ErrorCode
from ingestkit_json.router import JSONRouter


class TestCanHandle:
    """Tests for can_handle method."""

    def test_json_extension(self, mock_vector_store, mock_embedder):
        router = JSONRouter(mock_vector_store, mock_embedder)
        assert router.can_handle("data.json") is True

    def test_json_uppercase(self, mock_vector_store, mock_embedder):
        router = JSONRouter(mock_vector_store, mock_embedder)
        assert router.can_handle("data.JSON") is True

    def test_txt_extension(self, mock_vector_store, mock_embedder):
        router = JSONRouter(mock_vector_store, mock_embedder)
        assert router.can_handle("data.txt") is False

    def test_no_extension(self, mock_vector_store, mock_embedder):
        router = JSONRouter(mock_vector_store, mock_embedder)
        assert router.can_handle("data") is False


class TestProcessHappyPath:
    """Tests for process() on valid JSON files."""

    def test_simple_json(self, mock_vector_store, mock_embedder, tmp_json_file):
        mock_embedder.embed.return_value = [[0.1] * 768]
        router = JSONRouter(mock_vector_store, mock_embedder)
        fp = tmp_json_file({"name": "test", "value": 42})

        result = router.process(fp)

        assert result.chunks_created > 0
        assert result.ingest_key != ""
        assert result.total_keys == 2
        assert len(result.errors) == 0
        mock_vector_store.upsert_chunks.assert_called_once()
        mock_embedder.embed.assert_called_once()

    def test_nested_json(self, mock_vector_store, mock_embedder, tmp_json_file):
        mock_embedder.embed.return_value = [[0.1] * 768]
        data = {"user": {"name": "Alice", "address": {"city": "Springfield"}}}
        router = JSONRouter(mock_vector_store, mock_embedder)
        fp = tmp_json_file(data)

        result = router.process(fp)

        assert result.chunks_created > 0
        assert result.total_keys == 2  # name + city
        assert result.max_depth >= 2

    def test_ingest_key_computed(self, mock_vector_store, mock_embedder, tmp_json_file):
        mock_embedder.embed.return_value = [[0.1] * 768]
        router = JSONRouter(mock_vector_store, mock_embedder)
        fp = tmp_json_file({"a": 1})

        result = router.process(fp)

        assert len(result.ingest_key) == 64  # SHA-256 hex digest

    def test_tenant_id_propagation(self, mock_vector_store, mock_embedder, tmp_json_file):
        mock_embedder.embed.return_value = [[0.1] * 768]
        config = JSONProcessorConfig(tenant_id="tenant-42")
        router = JSONRouter(mock_vector_store, mock_embedder, config=config)
        fp = tmp_json_file({"a": 1})

        result = router.process(fp)

        assert result.tenant_id == "tenant-42"

    def test_source_uri_override(self, mock_vector_store, mock_embedder, tmp_json_file):
        mock_embedder.embed.return_value = [[0.1] * 768]
        router = JSONRouter(mock_vector_store, mock_embedder)
        fp = tmp_json_file({"a": 1})

        result = router.process(fp, source_uri="s3://bucket/data.json")

        assert result.ingest_key != ""  # Should still compute key

    def test_embed_result_populated(self, mock_vector_store, mock_embedder, tmp_json_file):
        mock_embedder.embed.return_value = [[0.1] * 768]
        router = JSONRouter(mock_vector_store, mock_embedder)
        fp = tmp_json_file({"a": 1})

        result = router.process(fp)

        assert result.embed_result is not None
        assert result.embed_result.texts_embedded > 0

    def test_written_artifacts(self, mock_vector_store, mock_embedder, tmp_json_file):
        mock_embedder.embed.return_value = [[0.1] * 768]
        router = JSONRouter(mock_vector_store, mock_embedder)
        fp = tmp_json_file({"a": 1})

        result = router.process(fp)

        assert len(result.written.vector_point_ids) > 0
        assert result.written.vector_collection == "helpdesk"


class TestProcessSecurityFailure:
    """Tests for process() when security scan fails."""

    def test_invalid_json_file(self, mock_vector_store, mock_embedder, tmp_path):
        router = JSONRouter(mock_vector_store, mock_embedder)
        fp = tmp_path / "bad.json"
        fp.write_text("{invalid}")

        result = router.process(str(fp))

        assert result.chunks_created == 0
        assert len(result.errors) > 0
        assert any("INVALID_JSON" in e for e in result.errors)

    def test_empty_file(self, mock_vector_store, mock_embedder, tmp_path):
        router = JSONRouter(mock_vector_store, mock_embedder)
        fp = tmp_path / "empty.json"
        fp.write_text("")

        result = router.process(str(fp))

        assert result.chunks_created == 0
        assert any("EMPTY" in e for e in result.errors)


class TestProcessBackendFailure:
    """Tests for process() when backend fails."""

    def test_embed_timeout(self, mock_vector_store, mock_embedder, tmp_json_file):
        mock_embedder.embed.side_effect = TimeoutError("timeout")
        router = JSONRouter(mock_vector_store, mock_embedder)
        fp = tmp_json_file({"a": 1})

        result = router.process(fp)

        assert result.chunks_created == 0
        assert any("EMBED" in e for e in result.errors)

    def test_vector_store_failure(self, mock_vector_store, mock_embedder, tmp_json_file):
        mock_embedder.embed.return_value = [[0.1] * 768]
        mock_vector_store.ensure_collection.side_effect = ConnectionError("refused")
        router = JSONRouter(mock_vector_store, mock_embedder)
        fp = tmp_json_file({"a": 1})

        result = router.process(fp)

        assert result.chunks_created == 0
        assert any("VECTOR" in e for e in result.errors)


class TestAprocess:
    """Tests for async aprocess() method."""

    @pytest.mark.asyncio
    async def test_aprocess_returns_result(self, mock_vector_store, mock_embedder, tmp_json_file):
        mock_embedder.embed.return_value = [[0.1] * 768]
        router = JSONRouter(mock_vector_store, mock_embedder)
        fp = tmp_json_file({"a": 1})

        result = await router.aprocess(fp)

        assert result.chunks_created > 0
        assert result.ingest_key != ""
