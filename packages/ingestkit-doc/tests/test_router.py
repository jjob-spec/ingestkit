"""Tests for ingestkit_doc.router."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ingestkit_doc.config import DocProcessorConfig
from ingestkit_doc.errors import ErrorCode
from ingestkit_doc.router import DocRouter

_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


class TestCanHandle:
    """can_handle extension check."""

    def test_doc_returns_true(self, mock_vector_store, mock_embedder):
        router = DocRouter(mock_vector_store, mock_embedder)
        assert router.can_handle("report.doc") is True

    def test_doc_uppercase_returns_true(self, mock_vector_store, mock_embedder):
        router = DocRouter(mock_vector_store, mock_embedder)
        assert router.can_handle("REPORT.DOC") is True

    def test_docx_returns_false(self, mock_vector_store, mock_embedder):
        router = DocRouter(mock_vector_store, mock_embedder)
        assert router.can_handle("report.docx") is False

    def test_pdf_returns_false(self, mock_vector_store, mock_embedder):
        router = DocRouter(mock_vector_store, mock_embedder)
        assert router.can_handle("report.pdf") is False


class TestProcessHappyPath:
    """process() happy path with mocked mammoth and backends."""

    def test_full_pipeline(self, mock_vector_store, mock_embedder, tmp_doc_file):
        path = tmp_doc_file()
        config = DocProcessorConfig()
        router = DocRouter(mock_vector_store, mock_embedder, config)

        mock_result = SimpleNamespace(
            value="This is a test document with enough words to create a chunk.",
            messages=[],
        )
        with patch("ingestkit_doc.converter.mammoth") as mock_mammoth:
            mock_mammoth.extract_raw_text.return_value = mock_result
            result = router.process(path)

        assert result.errors == []
        assert result.chunks_created > 0
        assert result.ingest_key != ""
        assert result.word_count > 0
        mock_vector_store.upsert_chunks.assert_called_once()

    def test_source_uri_override(self, mock_vector_store, mock_embedder, tmp_doc_file):
        path = tmp_doc_file()
        config = DocProcessorConfig()
        router = DocRouter(mock_vector_store, mock_embedder, config)

        mock_result = SimpleNamespace(
            value="Document text here for testing source URI.",
            messages=[],
        )
        with patch("ingestkit_doc.converter.mammoth") as mock_mammoth:
            mock_mammoth.extract_raw_text.return_value = mock_result
            result = router.process(path, source_uri="custom://doc/123")

        assert result.errors == []
        # Check that the custom source_uri was used in chunk metadata
        call_args = mock_vector_store.upsert_chunks.call_args
        payloads = call_args[0][1]
        assert payloads[0].metadata.source_uri == "custom://doc/123"


class TestProcessSecurityFailure:
    """process() with security scan failures."""

    def test_bad_extension(self, mock_vector_store, mock_embedder, tmp_path):
        path = tmp_path / "test.txt"
        path.write_bytes(b"dummy")
        router = DocRouter(mock_vector_store, mock_embedder)
        result = router.process(str(path))
        assert ErrorCode.E_SECURITY_BAD_EXTENSION in result.errors

    def test_missing_file(self, mock_vector_store, mock_embedder):
        router = DocRouter(mock_vector_store, mock_embedder)
        result = router.process("/nonexistent/file.doc")
        assert any(e.startswith("E_") for e in result.errors)


class TestProcessExtractFailure:
    """process() with extraction failures."""

    def test_extract_failed(self, mock_vector_store, mock_embedder, tmp_doc_file):
        path = tmp_doc_file()
        router = DocRouter(mock_vector_store, mock_embedder)

        with patch("ingestkit_doc.converter.mammoth") as mock_mammoth:
            mock_mammoth.extract_raw_text.side_effect = ValueError("Not a Word doc")
            result = router.process(path)

        assert ErrorCode.E_DOC_EXTRACT_FAILED in result.errors
        assert result.chunks_created == 0

    def test_empty_text(self, mock_vector_store, mock_embedder, tmp_doc_file):
        path = tmp_doc_file()
        router = DocRouter(mock_vector_store, mock_embedder)

        mock_result = SimpleNamespace(value="", messages=[])
        with patch("ingestkit_doc.converter.mammoth") as mock_mammoth:
            mock_mammoth.extract_raw_text.return_value = mock_result
            result = router.process(path)

        assert ErrorCode.E_DOC_EMPTY_TEXT in result.errors
        assert result.chunks_created == 0


class TestProcessBackendFailure:
    """process() with backend failures."""

    def test_embed_timeout(self, mock_vector_store, mock_embedder, tmp_doc_file):
        path = tmp_doc_file()
        mock_embedder.embed.side_effect = TimeoutError("embed timeout")
        router = DocRouter(mock_vector_store, mock_embedder)

        mock_result = SimpleNamespace(
            value="Some document text for embedding.",
            messages=[],
        )
        with patch("ingestkit_doc.converter.mammoth") as mock_mammoth:
            mock_mammoth.extract_raw_text.return_value = mock_result
            result = router.process(path)

        assert ErrorCode.E_BACKEND_EMBED_TIMEOUT in result.errors
        assert result.chunks_created == 0

    def test_upsert_failure(self, mock_vector_store, mock_embedder, tmp_doc_file):
        path = tmp_doc_file()
        mock_vector_store.upsert_chunks.side_effect = RuntimeError("connection refused")
        router = DocRouter(mock_vector_store, mock_embedder)

        mock_result = SimpleNamespace(
            value="Some document text for upsert test.",
            messages=[],
        )
        with patch("ingestkit_doc.converter.mammoth") as mock_mammoth:
            mock_mammoth.extract_raw_text.return_value = mock_result
            result = router.process(path)

        assert ErrorCode.E_BACKEND_VECTOR_CONNECT in result.errors


class TestAprocess:
    """aprocess() async wrapper."""

    def test_aprocess_calls_process(self, mock_vector_store, mock_embedder, tmp_doc_file):
        path = tmp_doc_file()
        router = DocRouter(mock_vector_store, mock_embedder)

        mock_result = SimpleNamespace(
            value="Async test document text.",
            messages=[],
        )
        with patch("ingestkit_doc.converter.mammoth") as mock_mammoth:
            mock_mammoth.extract_raw_text.return_value = mock_result
            result = asyncio.get_event_loop().run_until_complete(
                router.aprocess(path)
            )

        assert result.file_path == path


class TestTenantPropagation:
    """tenant_id flows through the full pipeline."""

    def test_tenant_id_in_result(self, mock_vector_store, mock_embedder, tmp_doc_file):
        path = tmp_doc_file()
        config = DocProcessorConfig(tenant_id="tenant-abc")
        router = DocRouter(mock_vector_store, mock_embedder, config)

        mock_result = SimpleNamespace(
            value="Tenant propagation test document text.",
            messages=[],
        )
        with patch("ingestkit_doc.converter.mammoth") as mock_mammoth:
            mock_mammoth.extract_raw_text.return_value = mock_result
            result = router.process(path)

        assert result.tenant_id == "tenant-abc"
        # Check tenant_id in chunk metadata
        call_args = mock_vector_store.upsert_chunks.call_args
        payloads = call_args[0][1]
        assert payloads[0].metadata.tenant_id == "tenant-abc"
