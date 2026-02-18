"""Unit tests for ingestkit_xml.models -- data models."""

from __future__ import annotations

from ingestkit_xml.models import ExtractResult, ProcessingResult, XMLChunkMetadata


class TestXMLChunkMetadata:
    """Tests for XMLChunkMetadata model."""

    def test_source_format_default(self):
        meta = XMLChunkMetadata(
            source_uri="test.xml",
            ingestion_method="xml_extract",
            parser_version="ingestkit_xml:1.0.0",
            ingest_key="abc",
            ingest_run_id="run-1",
            chunk_index=0,
            chunk_hash="hash",
        )
        assert meta.source_format == "xml"

    def test_root_tag_field(self):
        meta = XMLChunkMetadata(
            source_uri="test.xml",
            ingestion_method="xml_extract",
            parser_version="ingestkit_xml:1.0.0",
            ingest_key="abc",
            ingest_run_id="run-1",
            chunk_index=0,
            chunk_hash="hash",
            root_tag="document",
        )
        assert meta.root_tag == "document"

    def test_defaults(self):
        meta = XMLChunkMetadata(
            source_uri="test.xml",
            ingestion_method="xml_extract",
            parser_version="ingestkit_xml:1.0.0",
            ingest_key="abc",
            ingest_run_id="run-1",
            chunk_index=0,
            chunk_hash="hash",
        )
        assert meta.total_elements == 0
        assert meta.max_depth == 0
        assert meta.namespace_count == 0


class TestExtractResult:
    """Tests for ExtractResult model."""

    def test_construction(self):
        result = ExtractResult(
            lines=["line1", "line2"],
            total_elements=5,
            max_depth=3,
            namespaces=["http://example.com"],
            root_tag="root",
            truncated=False,
            fallback_used=False,
        )
        assert len(result.lines) == 2
        assert result.total_elements == 5
        assert result.max_depth == 3
        assert result.root_tag == "root"
        assert result.truncated is False
        assert result.fallback_used is False


class TestProcessingResult:
    """Tests for ProcessingResult model."""

    def test_construction(self):
        result = ProcessingResult(
            file_path="test.xml",
            ingest_key="abc123",
            ingest_run_id="run-1",
        )
        assert result.file_path == "test.xml"
        assert result.chunks_created == 0
        assert result.errors == []
        assert result.warnings == []
        assert result.total_elements == 0
        assert result.max_depth == 0
