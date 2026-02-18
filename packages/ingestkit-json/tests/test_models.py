"""Unit tests for ingestkit_json.models."""

from __future__ import annotations

from ingestkit_json.models import FlattenResult, JSONChunkMetadata, ProcessingResult
from ingestkit_core.models import WrittenArtifacts


class TestJSONChunkMetadata:
    """Tests for JSONChunkMetadata."""

    def test_default_source_format(self):
        meta = JSONChunkMetadata(
            source_uri="file.json",
            source_format="json",
            ingestion_method="json_flatten",
            parser_version="ingestkit_json:1.0.0",
            chunk_index=0,
            chunk_hash="abc123",
            ingest_key="key",
        )
        assert meta.source_format == "json"

    def test_extra_fields(self):
        meta = JSONChunkMetadata(
            source_uri="file.json",
            source_format="json",
            ingestion_method="json_flatten",
            parser_version="ingestkit_json:1.0.0",
            chunk_index=0,
            chunk_hash="abc123",
            ingest_key="key",
            total_keys=100,
            nesting_depth=5,
            key_path_prefix="data",
        )
        assert meta.total_keys == 100
        assert meta.nesting_depth == 5
        assert meta.key_path_prefix == "data"


class TestFlattenResult:
    """Tests for FlattenResult."""

    def test_construction(self):
        result = FlattenResult(
            lines=["a: 1", "b: 2"],
            total_keys=2,
            max_depth=0,
            truncated=False,
        )
        assert result.lines == ["a: 1", "b: 2"]
        assert result.total_keys == 2
        assert result.max_depth == 0
        assert result.truncated is False


class TestProcessingResult:
    """Tests for ProcessingResult."""

    def test_serialization_roundtrip(self):
        result = ProcessingResult(
            file_path="test.json",
            ingest_key="abc123",
            ingest_run_id="run-1",
            tenant_id="t1",
            chunks_created=5,
            total_keys=10,
            max_depth=3,
            written=WrittenArtifacts(
                vector_point_ids=["p1", "p2"],
                vector_collection="helpdesk",
            ),
        )
        data = result.model_dump()
        restored = ProcessingResult.model_validate(data)
        assert restored.file_path == "test.json"
        assert restored.chunks_created == 5
        assert restored.written.vector_collection == "helpdesk"
        assert len(restored.written.vector_point_ids) == 2

    def test_defaults(self):
        result = ProcessingResult(
            file_path="test.json",
            ingest_key="abc",
            ingest_run_id="run-1",
        )
        assert result.chunks_created == 0
        assert result.errors == []
        assert result.warnings == []
        assert result.processing_time_seconds == 0.0
