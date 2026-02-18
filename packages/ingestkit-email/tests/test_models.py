"""Tests for ingestkit_email.models."""

from ingestkit_email.models import (
    EmailChunkMetadata,
    EmailContentType,
    EmailMetadata,
    EmailType,
    ProcessingResult,
    WrittenArtifacts,
)


class TestEmailType:
    def test_email_type_values(self):
        """EML='eml', MSG='msg'."""
        assert EmailType.EML.value == "eml"
        assert EmailType.MSG.value == "msg"


class TestEmailMetadata:
    def test_email_metadata_creation(self):
        """All fields populate correctly."""
        meta = EmailMetadata(
            from_address="a@b.com",
            to_address="c@d.com",
            cc_address="e@f.com",
            date="2026-02-17",
            subject="Test",
            message_id="<123>",
            content_type=EmailContentType.PLAIN_TEXT,
            attachment_count=2,
            has_html_body=False,
            has_plain_body=True,
        )
        assert meta.from_address == "a@b.com"
        assert meta.attachment_count == 2
        assert meta.has_plain_body is True


class TestEmailChunkMetadata:
    def test_email_chunk_metadata_defaults(self):
        """source_format defaults to 'email'."""
        chunk = EmailChunkMetadata(
            source_uri="/test.eml",
            ingestion_method="email_conversion",
            parser_version="ingestkit_email:1.0.0",
            chunk_index=0,
            chunk_hash="abc123",
            ingest_key="key123",
        )
        assert chunk.source_format == "email"
        assert chunk.email_type is None


class TestProcessingResult:
    def test_processing_result_creation(self):
        """Full model roundtrip."""
        result = ProcessingResult(
            file_path="/test.eml",
            ingest_key="key123",
            ingest_run_id="run-001",
            tenant_id="acme",
            chunks_created=1,
            written=WrittenArtifacts(
                vector_point_ids=["pt1"],
                vector_collection="helpdesk",
            ),
            errors=[],
            warnings=["W_EMAIL_NO_DATE"],
            processing_time_seconds=0.5,
        )
        assert result.file_path == "/test.eml"
        assert result.chunks_created == 1
        assert result.written.vector_point_ids == ["pt1"]
        assert "W_EMAIL_NO_DATE" in result.warnings
