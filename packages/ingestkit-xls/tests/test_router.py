"""Tests for ingestkit_xls.router."""

from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ingestkit_xls.config import XlsProcessorConfig
from ingestkit_xls.converter import ExtractResult, SheetResult
from ingestkit_xls.errors import ErrorCode
from ingestkit_xls.router import XlsRouter

_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def _make_extract_result(text: str = "Name | Age\nAlice | 30", sheets: int = 1):
    """Build a minimal ExtractResult for router tests."""
    sheet_results = [
        SheetResult(name=f"Sheet{i+1}", text=text, row_count=2, col_count=2)
        for i in range(sheets)
    ]
    sections = [f"## {sr.name}\n\n{sr.text}" for sr in sheet_results]
    full_text = "\n\n".join(sections)
    return ExtractResult(
        sheets=sheet_results,
        text=full_text,
        word_count=len(full_text.split()),
        total_rows=2 * sheets,
        sheets_skipped=0,
        warnings=[],
    )


class TestCanHandle:
    """can_handle extension check."""

    def test_xls_returns_true(self, mock_vector_store, mock_embedder):
        router = XlsRouter(mock_vector_store, mock_embedder)
        assert router.can_handle("report.xls") is True

    def test_xls_uppercase_returns_true(self, mock_vector_store, mock_embedder):
        router = XlsRouter(mock_vector_store, mock_embedder)
        assert router.can_handle("REPORT.XLS") is True

    def test_xlsx_returns_false(self, mock_vector_store, mock_embedder):
        router = XlsRouter(mock_vector_store, mock_embedder)
        assert router.can_handle("report.xlsx") is False

    def test_pdf_returns_false(self, mock_vector_store, mock_embedder):
        router = XlsRouter(mock_vector_store, mock_embedder)
        assert router.can_handle("report.pdf") is False


class TestProcessHappyPath:
    """process() happy path with mocked xlrd and backends."""

    def test_full_pipeline(self, mock_vector_store, mock_embedder, tmp_xls_file):
        path = tmp_xls_file()
        config = XlsProcessorConfig()
        router = XlsRouter(mock_vector_store, mock_embedder, config)

        extract_result = _make_extract_result()
        with patch("ingestkit_xls.router.extract_sheets", return_value=extract_result):
            result = router.process(path)

        assert result.errors == []
        assert result.chunks_created > 0
        assert result.ingest_key != ""
        assert result.word_count > 0
        mock_vector_store.upsert_chunks.assert_called_once()

    def test_source_uri_override(self, mock_vector_store, mock_embedder, tmp_xls_file):
        path = tmp_xls_file()
        config = XlsProcessorConfig()
        router = XlsRouter(mock_vector_store, mock_embedder, config)

        extract_result = _make_extract_result()
        with patch("ingestkit_xls.router.extract_sheets", return_value=extract_result):
            result = router.process(path, source_uri="custom://xls/123")

        assert result.errors == []
        call_args = mock_vector_store.upsert_chunks.call_args
        payloads = call_args[0][1]
        assert payloads[0].metadata.source_uri == "custom://xls/123"

    def test_tenant_id_in_result(self, mock_vector_store, mock_embedder, tmp_xls_file):
        path = tmp_xls_file()
        config = XlsProcessorConfig(tenant_id="tenant-abc")
        router = XlsRouter(mock_vector_store, mock_embedder, config)

        extract_result = _make_extract_result()
        with patch("ingestkit_xls.router.extract_sheets", return_value=extract_result):
            result = router.process(path)

        assert result.tenant_id == "tenant-abc"
        call_args = mock_vector_store.upsert_chunks.call_args
        payloads = call_args[0][1]
        assert payloads[0].metadata.tenant_id == "tenant-abc"

    def test_multi_sheet_result(self, mock_vector_store, mock_embedder, tmp_xls_file):
        path = tmp_xls_file()
        config = XlsProcessorConfig()
        router = XlsRouter(mock_vector_store, mock_embedder, config)

        extract_result = _make_extract_result(sheets=3)
        with patch("ingestkit_xls.router.extract_sheets", return_value=extract_result):
            result = router.process(path)

        assert result.errors == []
        assert result.sheet_count == 3
        assert result.total_rows == 6


class TestProcessSecurityFailure:
    """process() with security scan failures."""

    def test_bad_extension(self, mock_vector_store, mock_embedder, tmp_path):
        path = tmp_path / "test.txt"
        path.write_bytes(b"dummy")
        router = XlsRouter(mock_vector_store, mock_embedder)
        result = router.process(str(path))
        assert ErrorCode.E_SECURITY_BAD_EXTENSION in result.errors

    def test_missing_file(self, mock_vector_store, mock_embedder):
        router = XlsRouter(mock_vector_store, mock_embedder)
        result = router.process("/nonexistent/file.xls")
        assert any(e.startswith("E_") for e in result.errors)


class TestProcessExtractFailure:
    """process() with extraction failures."""

    def test_extract_failed(self, mock_vector_store, mock_embedder, tmp_xls_file):
        path = tmp_xls_file()
        router = XlsRouter(mock_vector_store, mock_embedder)

        with patch(
            "ingestkit_xls.router.extract_sheets",
            side_effect=ValueError("Not a valid XLS file"),
        ):
            result = router.process(path)

        assert ErrorCode.E_XLS_EXTRACT_FAILED in result.errors
        assert result.chunks_created == 0

    def test_empty_text(self, mock_vector_store, mock_embedder, tmp_xls_file):
        path = tmp_xls_file()
        router = XlsRouter(mock_vector_store, mock_embedder)

        empty_result = ExtractResult(
            sheets=[],
            text="",
            word_count=0,
            total_rows=0,
            sheets_skipped=2,
            warnings=[],
        )
        with patch("ingestkit_xls.router.extract_sheets", return_value=empty_result):
            result = router.process(path)

        assert ErrorCode.E_XLS_EMPTY_TEXT in result.errors
        assert result.chunks_created == 0

    def test_password_protected(self, mock_vector_store, mock_embedder, tmp_xls_file):
        path = tmp_xls_file()
        router = XlsRouter(mock_vector_store, mock_embedder)

        with patch(
            "ingestkit_xls.router.extract_sheets",
            side_effect=Exception("Workbook is password-protected"),
        ):
            result = router.process(path)

        assert ErrorCode.E_XLS_PASSWORD_PROTECTED in result.errors
        assert result.chunks_created == 0

    def test_xlrd_unavailable(self, mock_vector_store, mock_embedder, tmp_xls_file):
        path = tmp_xls_file()
        router = XlsRouter(mock_vector_store, mock_embedder)

        with patch(
            "ingestkit_xls.router.extract_sheets",
            side_effect=ImportError("xlrd not available"),
        ):
            result = router.process(path)

        assert ErrorCode.E_XLS_XLRD_UNAVAILABLE in result.errors
        assert result.chunks_created == 0


class TestProcessBackendFailure:
    """process() with backend failures."""

    def test_embed_timeout(self, mock_vector_store, mock_embedder, tmp_xls_file):
        path = tmp_xls_file()
        mock_embedder.embed.side_effect = TimeoutError("embed timeout")
        router = XlsRouter(mock_vector_store, mock_embedder)

        extract_result = _make_extract_result()
        with patch("ingestkit_xls.router.extract_sheets", return_value=extract_result):
            result = router.process(path)

        assert ErrorCode.E_BACKEND_EMBED_TIMEOUT in result.errors
        assert result.chunks_created == 0

    def test_upsert_failure(self, mock_vector_store, mock_embedder, tmp_xls_file):
        path = tmp_xls_file()
        mock_vector_store.upsert_chunks.side_effect = RuntimeError("connection refused")
        router = XlsRouter(mock_vector_store, mock_embedder)

        extract_result = _make_extract_result()
        with patch("ingestkit_xls.router.extract_sheets", return_value=extract_result):
            result = router.process(path)

        assert ErrorCode.E_BACKEND_VECTOR_CONNECT in result.errors


class TestAprocess:
    """aprocess() async wrapper."""

    def test_aprocess_calls_process(self, mock_vector_store, mock_embedder, tmp_xls_file):
        path = tmp_xls_file()
        router = XlsRouter(mock_vector_store, mock_embedder)

        extract_result = _make_extract_result()
        with patch("ingestkit_xls.router.extract_sheets", return_value=extract_result):
            result = asyncio.get_event_loop().run_until_complete(
                router.aprocess(path)
            )

        assert result.file_path == path
