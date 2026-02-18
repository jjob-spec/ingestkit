"""Tests for ingestkit_rtf.models."""

from __future__ import annotations

from ingestkit_rtf.models import RTFChunkMetadata, ProcessingResult


class TestRTFChunkMetadata:
    """RTFChunkMetadata default values and serialisation."""

    def test_default_source_format(self):
        meta = RTFChunkMetadata(
            source_uri="test.rtf",
            source_format="rtf",
            ingestion_method="rtf_striprtf",
            parser_version="ingestkit_rtf:1.0.0",
            chunk_index=0,
            chunk_hash="abc123",
            ingest_key="key123",
        )
        assert meta.source_format == "rtf"

    def test_default_word_count(self):
        meta = RTFChunkMetadata(
            source_uri="test.rtf",
            source_format="rtf",
            ingestion_method="rtf_striprtf",
            parser_version="ingestkit_rtf:1.0.0",
            chunk_index=0,
            chunk_hash="abc123",
            ingest_key="key123",
        )
        assert meta.word_count == 0

    def test_round_trip(self):
        meta = RTFChunkMetadata(
            source_uri="test.rtf",
            source_format="rtf",
            ingestion_method="rtf_striprtf",
            parser_version="ingestkit_rtf:1.0.0",
            chunk_index=0,
            chunk_hash="abc123",
            ingest_key="key123",
            word_count=500,
        )
        data = meta.model_dump()
        restored = RTFChunkMetadata(**data)
        assert restored.word_count == 500


class TestProcessingResult:
    """ProcessingResult default values and serialisation."""

    def test_defaults(self):
        result = ProcessingResult(
            file_path="test.rtf",
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
            file_path="test.rtf",
            ingest_key="key123",
            ingest_run_id="run123",
            tenant_id="tenant-a",
            chunks_created=5,
            word_count=1000,
            errors=["E_RTF_EXTRACT_FAILED"],
        )
        data = result.model_dump()
        restored = ProcessingResult(**data)
        assert restored.tenant_id == "tenant-a"
        assert restored.chunks_created == 5
        assert restored.word_count == 1000
