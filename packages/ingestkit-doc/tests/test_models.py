"""Tests for ingestkit_doc.models."""

from __future__ import annotations

from ingestkit_doc.models import DocChunkMetadata, ProcessingResult


class TestDocChunkMetadata:
    """DocChunkMetadata default values and serialisation."""

    def test_default_source_format(self):
        meta = DocChunkMetadata(
            source_uri="test.doc",
            source_format="doc",
            ingestion_method="doc_mammoth",
            parser_version="ingestkit_doc:1.0.0",
            chunk_index=0,
            chunk_hash="abc123",
            ingest_key="key123",
        )
        assert meta.source_format == "doc"

    def test_default_word_count(self):
        meta = DocChunkMetadata(
            source_uri="test.doc",
            source_format="doc",
            ingestion_method="doc_mammoth",
            parser_version="ingestkit_doc:1.0.0",
            chunk_index=0,
            chunk_hash="abc123",
            ingest_key="key123",
        )
        assert meta.word_count == 0
        assert meta.mammoth_warnings == 0

    def test_round_trip(self):
        meta = DocChunkMetadata(
            source_uri="test.doc",
            source_format="doc",
            ingestion_method="doc_mammoth",
            parser_version="ingestkit_doc:1.0.0",
            chunk_index=0,
            chunk_hash="abc123",
            ingest_key="key123",
            word_count=500,
            mammoth_warnings=2,
        )
        data = meta.model_dump()
        restored = DocChunkMetadata(**data)
        assert restored.word_count == 500
        assert restored.mammoth_warnings == 2


class TestProcessingResult:
    """ProcessingResult default values and serialisation."""

    def test_defaults(self):
        result = ProcessingResult(
            file_path="test.doc",
            ingest_key="key123",
            ingest_run_id="run123",
        )
        assert result.chunks_created == 0
        assert result.word_count == 0
        assert result.errors == []
        assert result.warnings == []
        assert result.processing_time_seconds == 0.0

    def test_round_trip(self):
        result = ProcessingResult(
            file_path="test.doc",
            ingest_key="key123",
            ingest_run_id="run123",
            tenant_id="tenant-a",
            chunks_created=5,
            word_count=1000,
            errors=["E_DOC_EXTRACT_FAILED"],
        )
        data = result.model_dump()
        restored = ProcessingResult(**data)
        assert restored.tenant_id == "tenant-a"
        assert restored.chunks_created == 5
        assert restored.word_count == 1000
