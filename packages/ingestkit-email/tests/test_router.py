"""Tests for ingestkit_email.router."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ingestkit_email.config import EmailProcessorConfig
from ingestkit_email.errors import ErrorCode
from ingestkit_email.router import EmailRouter


class TestEmailRouter:
    def test_can_handle_eml(self, mock_vector_store, mock_embedder):
        router = EmailRouter(mock_vector_store, mock_embedder)
        assert router.can_handle("/path/to/email.eml") is True

    def test_can_handle_msg(self, mock_vector_store, mock_embedder):
        router = EmailRouter(mock_vector_store, mock_embedder)
        assert router.can_handle("/path/to/email.msg") is True

    def test_can_handle_pdf(self, mock_vector_store, mock_embedder):
        router = EmailRouter(mock_vector_store, mock_embedder)
        assert router.can_handle("/path/to/document.pdf") is False

    def test_process_eml(self, sample_eml_file, mock_vector_store, mock_embedder):
        """Full end-to-end with mock backends."""
        router = EmailRouter(mock_vector_store, mock_embedder)
        result = router.process(sample_eml_file)

        assert result.chunks_created == 1
        assert result.ingest_key != ""
        assert result.errors == []
        assert result.email_metadata is not None
        assert result.email_metadata.from_address == "sender@example.com"
        assert result.written.vector_point_ids
        mock_vector_store.upsert_chunks.assert_called_once()
        mock_embedder.embed.assert_called_once()

    def test_process_empty_body(self, tmp_path, mock_vector_store, mock_embedder):
        """Email with empty body produces E_EMAIL_EMPTY_BODY."""
        from tests.conftest import _build_eml_bytes

        eml_bytes = _build_eml_bytes(plain="", html=None)
        p = tmp_path / "empty.eml"
        p.write_bytes(eml_bytes)

        router = EmailRouter(mock_vector_store, mock_embedder)
        result = router.process(str(p))

        assert ErrorCode.E_EMAIL_EMPTY_BODY.value in result.errors

    def test_process_includes_headers(self, sample_eml_file, mock_vector_store, mock_embedder):
        """Default config includes headers in chunk text."""
        router = EmailRouter(mock_vector_store, mock_embedder)
        result = router.process(sample_eml_file)

        # Get the text that was embedded
        call_args = mock_embedder.embed.call_args[0][0]
        chunk_text = call_args[0]
        assert "From:" in chunk_text
        assert "Subject:" in chunk_text

    def test_process_excludes_headers(self, sample_eml_file, mock_vector_store, mock_embedder):
        """include_headers=False -> no headers in chunk text."""
        config = EmailProcessorConfig(include_headers=False)
        router = EmailRouter(mock_vector_store, mock_embedder, config=config)
        result = router.process(sample_eml_file)

        call_args = mock_embedder.embed.call_args[0][0]
        chunk_text = call_args[0]
        assert "From:" not in chunk_text

    def test_process_html_warning(self, sample_eml_html_file, mock_vector_store, mock_embedder):
        """HTML-only email produces W_EMAIL_HTML_ONLY in warnings."""
        router = EmailRouter(mock_vector_store, mock_embedder)
        result = router.process(sample_eml_html_file)

        assert ErrorCode.W_EMAIL_HTML_ONLY.value in result.warnings

    def test_ingest_key_deterministic(self, sample_eml_file, mock_vector_store, mock_embedder):
        """Same file produces same ingest key."""
        router = EmailRouter(mock_vector_store, mock_embedder)
        result1 = router.process(sample_eml_file)
        result2 = router.process(sample_eml_file)
        assert result1.ingest_key == result2.ingest_key

    def test_tenant_id_propagated(self, sample_eml_file, mock_vector_store, mock_embedder):
        """tenant_id flows from config to result and chunk metadata."""
        config = EmailProcessorConfig(tenant_id="acme-corp")
        router = EmailRouter(mock_vector_store, mock_embedder, config=config)
        result = router.process(sample_eml_file)

        assert result.tenant_id == "acme-corp"

        # Check chunk metadata via the upsert call
        call_args = mock_vector_store.upsert_chunks.call_args
        chunks = call_args[0][1]
        assert chunks[0].metadata.tenant_id == "acme-corp"
