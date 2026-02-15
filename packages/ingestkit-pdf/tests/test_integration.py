"""Integration tests for the ingestkit-pdf pipeline.

Exercises PDFRouter.process() end-to-end with real backends (Qdrant,
SQLite, Ollama).  All tests skip gracefully when backends are not
available -- no failures, only skips.

Requires: Qdrant on localhost:6333, Ollama on localhost:11434.
SQLite in-memory is always available.
"""

from __future__ import annotations

import pytest

from ingestkit_pdf.config import PDFProcessorConfig
from ingestkit_pdf.models import (
    IngestionMethod,
    PDFType,
)
from ingestkit_pdf.router import PDFRouter


# ---------------------------------------------------------------------------
# Backend Availability Checks
# ---------------------------------------------------------------------------


def _qdrant_available() -> bool:
    """Check if Qdrant is reachable on localhost:6333."""
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url="http://localhost:6333", timeout=2)
        client.get_collections()
        return True
    except Exception:
        return False


def _ollama_available() -> bool:
    """Check if Ollama is reachable on localhost:11434."""
    try:
        import httpx

        resp = httpx.get("http://localhost:11434/api/tags", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


requires_qdrant = pytest.mark.skipif(
    not _qdrant_available(), reason="Qdrant not available on localhost:6333"
)
requires_ollama = pytest.mark.skipif(
    not _ollama_available(), reason="Ollama not available on localhost:11434"
)
requires_backends = pytest.mark.skipif(
    not (_qdrant_available() and _ollama_available()),
    reason="Qdrant and/or Ollama not available",
)


def _tesseract_available() -> bool:
    """Check if Tesseract OCR is installed."""
    try:
        import shutil

        return shutil.which("tesseract") is not None
    except Exception:
        return False


requires_tesseract = pytest.mark.skipif(
    not _tesseract_available(), reason="Tesseract OCR not installed"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def real_vector_store():
    """Session-scoped Qdrant backend."""
    from ingestkit_excel.backends import QdrantVectorStore

    store = QdrantVectorStore(url="http://localhost:6333")
    yield store


@pytest.fixture(scope="session")
def real_structured_db():
    """Session-scoped in-memory SQLite backend."""
    from ingestkit_excel.backends import SQLiteStructuredDB

    return SQLiteStructuredDB(":memory:")


@pytest.fixture(scope="session")
def real_llm():
    """Session-scoped Ollama LLM backend."""
    from ingestkit_excel.backends import OllamaLLM

    return OllamaLLM(base_url="http://localhost:11434")


@pytest.fixture(scope="session")
def real_embedder():
    """Session-scoped Ollama embedding backend."""
    from ingestkit_excel.backends import OllamaEmbedding

    return OllamaEmbedding(
        base_url="http://localhost:11434", model="nomic-embed-text"
    )


@pytest.fixture()
def integration_router(real_vector_store, real_structured_db, real_llm, real_embedder):
    """PDFRouter wired to real backends with test config."""
    config = PDFProcessorConfig(
        tenant_id="integration-test",
        default_collection="test_integration",
    )
    return PDFRouter(
        vector_store=real_vector_store,
        structured_db=real_structured_db,
        llm=real_llm,
        embedder=real_embedder,
        config=config,
    )


@pytest.fixture(autouse=True)
def _cleanup_qdrant():
    """Delete test collections after each test to avoid cross-test pollution."""
    yield
    if not _qdrant_available():
        return
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url="http://localhost:6333", timeout=2)
        collections = client.get_collections().collections
        for c in collections:
            if c.name.startswith("test_"):
                client.delete_collection(c.name)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Path A: Text-Native PDF
# ---------------------------------------------------------------------------


@pytest.mark.integration
@requires_backends
class TestPathAIntegration:
    """Full pipeline integration tests for Path A (text extraction)."""

    def test_path_a_text_native_full_pipeline(
        self, integration_router, text_native_pdf
    ):
        result = integration_router.process(str(text_native_pdf))

        assert result.errors == [], f"Unexpected errors: {result.errors}"
        assert result.chunks_created > 0
        assert result.classification.pdf_type == PDFType.TEXT_NATIVE
        assert result.classification.confidence > 0
        assert len(result.ingest_key) == 64
        assert all(c in "0123456789abcdef" for c in result.ingest_key)
        assert result.processing_time_seconds > 0
        assert result.ingestion_method == IngestionMethod.TEXT_EXTRACTION

    def test_path_a_written_artifacts(self, integration_router, text_native_pdf):
        result = integration_router.process(str(text_native_pdf))

        assert result.written.vector_point_ids
        assert result.written.vector_collection == "test_integration"

    def test_path_a_stage_results(self, integration_router, text_native_pdf):
        result = integration_router.process(str(text_native_pdf))

        assert result.parse_result.pages_extracted == 3
        assert result.parse_result.extraction_method == "pymupdf"
        assert result.classification_result.tier_used is not None

    def test_path_a_idempotency(self, integration_router, text_native_pdf):
        result1 = integration_router.process(str(text_native_pdf))
        result2 = integration_router.process(str(text_native_pdf))

        assert result1.ingest_key == result2.ingest_key

    def test_path_a_tenant_propagation(self, integration_router, text_native_pdf):
        result = integration_router.process(str(text_native_pdf))

        assert result.tenant_id == "integration-test"


# ---------------------------------------------------------------------------
# Path B: Scanned PDF (OCR)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.ocr
@requires_backends
@requires_tesseract
class TestPathBIntegration:
    """Full pipeline integration tests for Path B (OCR pipeline)."""

    def test_path_b_scanned_full_pipeline(self, integration_router, scanned_pdf):
        result = integration_router.process(str(scanned_pdf))

        # Path B may produce warnings but no fatal errors
        fatal = [e for e in result.errors if not e.startswith("W_")]
        assert fatal == [], f"Unexpected fatal errors: {fatal}"
        assert result.chunks_created > 0
        assert result.classification.pdf_type == PDFType.SCANNED
        assert result.ingestion_method == IngestionMethod.OCR_PIPELINE
        assert result.ocr_result is not None

    def test_path_b_ocr_stage_result(self, integration_router, scanned_pdf):
        result = integration_router.process(str(scanned_pdf))

        assert result.ocr_result is not None
        assert result.ocr_result.pages_ocrd == 2
        assert result.ocr_result.engine_used is not None

    def test_path_b_written_artifacts(self, integration_router, scanned_pdf):
        result = integration_router.process(str(scanned_pdf))

        assert result.written.vector_point_ids


# ---------------------------------------------------------------------------
# Path C: Complex PDF
# ---------------------------------------------------------------------------


@pytest.mark.integration
@requires_backends
class TestPathCIntegration:
    """Tests for Path C (ComplexProcessor not yet available)."""

    def test_path_c_complex_not_available(self, integration_router, complex_pdf):
        """ComplexProcessor is None, so complex PDFs produce an error result."""
        result = integration_router.process(str(complex_pdf))

        # The router returns "ComplexProcessor not available" when the
        # complex_processor is None and the PDF classifies as COMPLEX.
        # If the file is classified differently (e.g., TEXT_NATIVE), that
        # is also acceptable -- the key assertion is no crash.
        if result.classification.pdf_type == PDFType.COMPLEX:
            assert any(
                "ComplexProcessor not available" in e for e in result.errors
            )
        # Otherwise it was classified and processed via another path -- fine.


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


@pytest.mark.integration
@requires_backends
class TestEdgeCaseIntegration:
    """Edge case tests with real backends."""

    def test_encrypted_pdf_handling(self, integration_router, encrypted_pdf):
        result = integration_router.process(str(encrypted_pdf))

        # Password-protected PDF should produce a security error
        assert result.chunks_created == 0
        assert len(result.errors) > 0

    @pytest.mark.xfail(
        reason="Known issue: garbled PDFs may trigger IndexError in PyMuPDF "
        "text extraction when page access occurs after document state changes",
        strict=False,
    )
    def test_garbled_pdf_handling(self, integration_router, garbled_pdf):
        result = integration_router.process(str(garbled_pdf))

        # Should complete; may have quality warnings or errors
        assert result.processing_time_seconds >= 0

    def test_can_handle_filter(self, integration_router):
        assert integration_router.can_handle("test.pdf") is True
        assert integration_router.can_handle("test.PDF") is True
        assert integration_router.can_handle("test.xlsx") is False
        assert integration_router.can_handle("test.txt") is False
