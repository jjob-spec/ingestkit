"""Tests for ingestkit_rtf.router."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from ingestkit_rtf.config import RTFProcessorConfig
from ingestkit_rtf.errors import ErrorCode
from ingestkit_rtf.router import RTFRouter

_RTF_MAGIC = b"{\\rtf"


class TestCanHandle:
    """can_handle extension check."""

    def test_rtf_returns_true(self, mock_vector_store, mock_embedder):
        router = RTFRouter(mock_vector_store, mock_embedder)
        assert router.can_handle("report.rtf") is True

    def test_rtf_uppercase_returns_true(self, mock_vector_store, mock_embedder):
        router = RTFRouter(mock_vector_store, mock_embedder)
        assert router.can_handle("REPORT.RTF") is True

    def test_doc_returns_false(self, mock_vector_store, mock_embedder):
        router = RTFRouter(mock_vector_store, mock_embedder)
        assert router.can_handle("report.doc") is False

    def test_pdf_returns_false(self, mock_vector_store, mock_embedder):
        router = RTFRouter(mock_vector_store, mock_embedder)
        assert router.can_handle("report.pdf") is False

    def test_txt_returns_false(self, mock_vector_store, mock_embedder):
        router = RTFRouter(mock_vector_store, mock_embedder)
        assert router.can_handle("report.txt") is False


class TestProcessHappyPath:
    """process() happy path with mocked striprtf and backends."""

    def test_full_pipeline(self, mock_vector_store, mock_embedder, tmp_rtf_file):
        path = tmp_rtf_file()
        config = RTFProcessorConfig()
        router = RTFRouter(mock_vector_store, mock_embedder, config)

        with patch("ingestkit_rtf.converter.rtf_to_text") as mock_rtf_to_text:
            mock_rtf_to_text.return_value = (
                "This is a test document with enough words to create a chunk."
            )
            result = router.process(path)

        assert result.errors == []
        assert result.chunks_created > 0
        assert result.ingest_key != ""
        assert result.word_count > 0
        mock_vector_store.upsert_chunks.assert_called_once()

    def test_source_uri_override(self, mock_vector_store, mock_embedder, tmp_rtf_file):
        path = tmp_rtf_file()
        config = RTFProcessorConfig()
        router = RTFRouter(mock_vector_store, mock_embedder, config)

        with patch("ingestkit_rtf.converter.rtf_to_text") as mock_rtf_to_text:
            mock_rtf_to_text.return_value = "Document text here for testing source URI."
            result = router.process(path, source_uri="custom://rtf/123")

        assert result.errors == []
        call_args = mock_vector_store.upsert_chunks.call_args
        payloads = call_args[0][1]
        assert payloads[0].metadata.source_uri == "custom://rtf/123"


class TestProcessSecurityFailure:
    """process() with security scan failures."""

    def test_bad_extension(self, mock_vector_store, mock_embedder, tmp_path):
        path = tmp_path / "test.txt"
        path.write_bytes(b"dummy")
        router = RTFRouter(mock_vector_store, mock_embedder)
        result = router.process(str(path))
        assert ErrorCode.E_SECURITY_BAD_EXTENSION in result.errors

    def test_missing_file(self, mock_vector_store, mock_embedder):
        router = RTFRouter(mock_vector_store, mock_embedder)
        result = router.process("/nonexistent/file.rtf")
        assert any(e.startswith("E_") for e in result.errors)


class TestProcessExtractFailure:
    """process() with extraction failures."""

    def test_extract_failed(self, mock_vector_store, mock_embedder, tmp_rtf_file):
        path = tmp_rtf_file()
        router = RTFRouter(mock_vector_store, mock_embedder)

        with patch("ingestkit_rtf.converter.rtf_to_text") as mock_rtf_to_text:
            mock_rtf_to_text.side_effect = ValueError("Not a valid RTF file")
            result = router.process(path)

        assert ErrorCode.E_RTF_EXTRACT_FAILED in result.errors
        assert result.chunks_created == 0

    def test_empty_text(self, mock_vector_store, mock_embedder, tmp_rtf_file):
        path = tmp_rtf_file()
        router = RTFRouter(mock_vector_store, mock_embedder)

        with patch("ingestkit_rtf.converter.rtf_to_text") as mock_rtf_to_text:
            mock_rtf_to_text.return_value = ""
            result = router.process(path)

        assert ErrorCode.E_RTF_EMPTY_TEXT in result.errors
        assert result.chunks_created == 0


class TestProcessBackendFailure:
    """process() with backend failures."""

    def test_embed_timeout(self, mock_vector_store, mock_embedder, tmp_rtf_file):
        path = tmp_rtf_file()
        mock_embedder.embed.side_effect = TimeoutError("embed timeout")
        router = RTFRouter(mock_vector_store, mock_embedder)

        with patch("ingestkit_rtf.converter.rtf_to_text") as mock_rtf_to_text:
            mock_rtf_to_text.return_value = "Some document text for embedding."
            result = router.process(path)

        assert ErrorCode.E_BACKEND_EMBED_TIMEOUT in result.errors
        assert result.chunks_created == 0

    def test_upsert_failure(self, mock_vector_store, mock_embedder, tmp_rtf_file):
        path = tmp_rtf_file()
        mock_vector_store.upsert_chunks.side_effect = RuntimeError("connection refused")
        router = RTFRouter(mock_vector_store, mock_embedder)

        with patch("ingestkit_rtf.converter.rtf_to_text") as mock_rtf_to_text:
            mock_rtf_to_text.return_value = "Some document text for upsert test."
            result = router.process(path)

        assert ErrorCode.E_BACKEND_VECTOR_CONNECT in result.errors


class TestAprocess:
    """aprocess() async wrapper."""

    def test_aprocess_calls_process(self, mock_vector_store, mock_embedder, tmp_rtf_file):
        path = tmp_rtf_file()
        router = RTFRouter(mock_vector_store, mock_embedder)

        with patch("ingestkit_rtf.converter.rtf_to_text") as mock_rtf_to_text:
            mock_rtf_to_text.return_value = "Async test document text."
            result = asyncio.get_event_loop().run_until_complete(
                router.aprocess(path)
            )

        assert result.file_path == path


class TestTenantPropagation:
    """tenant_id flows through the full pipeline."""

    def test_tenant_id_in_result(self, mock_vector_store, mock_embedder, tmp_rtf_file):
        path = tmp_rtf_file()
        config = RTFProcessorConfig(tenant_id="tenant-abc")
        router = RTFRouter(mock_vector_store, mock_embedder, config)

        with patch("ingestkit_rtf.converter.rtf_to_text") as mock_rtf_to_text:
            mock_rtf_to_text.return_value = "Tenant propagation test document text."
            result = router.process(path)

        assert result.tenant_id == "tenant-abc"
        call_args = mock_vector_store.upsert_chunks.call_args
        payloads = call_args[0][1]
        assert payloads[0].metadata.tenant_id == "tenant-abc"
