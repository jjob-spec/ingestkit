"""Tests for ingestkit_xls.models."""

from __future__ import annotations

from ingestkit_xls.models import ProcessingResult, XlsChunkMetadata


class TestXlsChunkMetadata:
    """XlsChunkMetadata default values and serialisation."""

    def test_default_source_format(self):
        meta = XlsChunkMetadata(
            source_uri="test.xls",
            source_format="xls",
            ingestion_method="xls_xlrd",
            parser_version="ingestkit_xls:1.0.0",
            chunk_index=0,
            chunk_hash="abc123",
            ingest_key="key123",
        )
        assert meta.source_format == "xls"

    def test_default_values(self):
        meta = XlsChunkMetadata(
            source_uri="test.xls",
            source_format="xls",
            ingestion_method="xls_xlrd",
            parser_version="ingestkit_xls:1.0.0",
            chunk_index=0,
            chunk_hash="abc123",
            ingest_key="key123",
        )
        assert meta.word_count == 0
        assert meta.sheet_count == 0
        assert meta.total_rows == 0
        assert meta.sheets_skipped == 0

    def test_round_trip(self):
        meta = XlsChunkMetadata(
            source_uri="test.xls",
            source_format="xls",
            ingestion_method="xls_xlrd",
            parser_version="ingestkit_xls:1.0.0",
            chunk_index=0,
            chunk_hash="abc123",
            ingest_key="key123",
            word_count=500,
            sheet_count=3,
            total_rows=100,
            sheets_skipped=1,
        )
        data = meta.model_dump()
        restored = XlsChunkMetadata(**data)
        assert restored.word_count == 500
        assert restored.sheet_count == 3
        assert restored.total_rows == 100
        assert restored.sheets_skipped == 1

    def test_inherits_from_base(self):
        from ingestkit_core.models import BaseChunkMetadata

        meta = XlsChunkMetadata(
            source_uri="test.xls",
            source_format="xls",
            ingestion_method="xls_xlrd",
            parser_version="ingestkit_xls:1.0.0",
            chunk_index=0,
            chunk_hash="abc123",
            ingest_key="key123",
        )
        assert isinstance(meta, BaseChunkMetadata)


class TestProcessingResult:
    """ProcessingResult default values and serialisation."""

    def test_defaults(self):
        result = ProcessingResult(
            file_path="test.xls",
            ingest_key="key123",
            ingest_run_id="run123",
        )
        assert result.chunks_created == 0
        assert result.word_count == 0
        assert result.sheet_count == 0
        assert result.total_rows == 0
        assert result.errors == []
        assert result.warnings == []
        assert result.processing_time_seconds == 0.0

    def test_round_trip(self):
        result = ProcessingResult(
            file_path="test.xls",
            ingest_key="key123",
            ingest_run_id="run123",
            tenant_id="tenant-a",
            chunks_created=5,
            word_count=1000,
            sheet_count=2,
            total_rows=50,
            errors=["E_XLS_EXTRACT_FAILED"],
        )
        data = result.model_dump()
        restored = ProcessingResult(**data)
        assert restored.tenant_id == "tenant-a"
        assert restored.chunks_created == 5
        assert restored.word_count == 1000
        assert restored.sheet_count == 2
        assert restored.total_rows == 50
