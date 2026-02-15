"""Benchmark tests with SLO assertions for ingestkit-pdf.

Per-stage latency tests run with mock backends (unit-level, no external
services).  Throughput SLO tests require real backends and are marked
as integration tests.

SLO targets are drawn from the issue specification:
- Security scan: < 500ms (max)
- Profile extraction: < 5s (max)
- Tier 1 classification: < 500ms (max)
- Path A throughput: >= 50 pages/sec
- Path B throughput: >= 10 pages/sec
"""

from __future__ import annotations

import time

import pathlib
import sys

import pytest

from ingestkit_pdf.config import PDFProcessorConfig
from ingestkit_pdf.inspector import PDFInspector
from ingestkit_pdf.router import PDFRouter
from ingestkit_pdf.security import PDFSecurityScanner

# Add tests directory to path so conftest helpers can be imported
_tests_dir = str(pathlib.Path(__file__).resolve().parent)
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)

from conftest import (  # noqa: E402
    MockEmbeddingBackend,
    MockLLMBackend,
    MockStructuredDBBackend,
    MockVectorStoreBackend,
    _make_document_profile,
    _make_page_profile,
)


# ---------------------------------------------------------------------------
# Backend Availability (reused from test_integration)
# ---------------------------------------------------------------------------


def _qdrant_available() -> bool:
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url="http://localhost:6333", timeout=2)
        client.get_collections()
        return True
    except Exception:
        return False


def _ollama_available() -> bool:
    try:
        import httpx

        resp = httpx.get("http://localhost:11434/api/tags", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


requires_backends = pytest.mark.skipif(
    not (_qdrant_available() and _ollama_available()),
    reason="Qdrant and/or Ollama not available",
)


def _tesseract_available() -> bool:
    try:
        import shutil

        return shutil.which("tesseract") is not None
    except Exception:
        return False


requires_tesseract = pytest.mark.skipif(
    not _tesseract_available(), reason="Tesseract OCR not installed"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _measure_throughput(
    router: PDFRouter,
    pdf_path: str,
    page_count: int,
    iterations: int,
) -> dict:
    """Run process() N times and compute throughput stats."""
    times = []
    for _ in range(iterations):
        start = time.monotonic()
        result = router.process(pdf_path)
        elapsed = time.monotonic() - start
        times.append(elapsed)
        # Allow warnings but not fatal errors
        fatal = [e for e in result.errors if not e.startswith("W_")]
        assert not fatal, f"Processing failed: {fatal}"

    total_pages = page_count * iterations
    total_time = sum(times)
    return {
        "pages_per_sec": total_pages / total_time if total_time > 0 else 0,
        "avg_time_sec": total_time / iterations,
        "min_time_sec": min(times),
        "max_time_sec": max(times),
        "total_pages": total_pages,
        "iterations": iterations,
    }


# ---------------------------------------------------------------------------
# Per-Stage Latency Benchmarks (unit-level, no external services)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.benchmark
class TestPerStageBenchmarks:
    """Per-stage latency budget tests using local-only operations."""

    def test_security_scan_latency(self, text_native_pdf):
        """Security scan must complete in < 500ms (SPEC max budget)."""
        config = PDFProcessorConfig()
        scanner = PDFSecurityScanner(config)

        start = time.monotonic()
        _metadata, _errors = scanner.scan(str(text_native_pdf))
        elapsed = time.monotonic() - start

        assert elapsed < 0.5, (
            f"Security scan took {elapsed:.3f}s, exceeds 500ms budget"
        )

    def test_profile_extraction_latency(self, text_native_pdf):
        """Profile extraction must complete in < 5s (SPEC max budget).

        Uses the router's internal _build_document_profile method via
        a lightweight router with mock backends.
        """
        config = PDFProcessorConfig()
        router = PDFRouter(
            vector_store=MockVectorStoreBackend(),
            structured_db=MockStructuredDBBackend(),
            llm=MockLLMBackend(),
            embedder=MockEmbeddingBackend(),
            config=config,
        )

        # Build profile by calling the full process and measuring parse_duration
        # We use process() to measure the actual profile building time as
        # reported in parse_result, since _build_document_profile is private.
        # The mock LLM will fail, but parse_result is built before classification.
        # Instead, we just time the security + profile stage directly.
        import fitz

        scanner = PDFSecurityScanner(config)
        metadata, _ = scanner.scan(str(text_native_pdf))

        doc = fitz.open(str(text_native_pdf))
        try:
            start = time.monotonic()
            router._build_document_profile(
                str(text_native_pdf), doc, metadata, []
            )
            elapsed = time.monotonic() - start
        finally:
            doc.close()

        assert elapsed < 5.0, (
            f"Profile extraction took {elapsed:.3f}s, exceeds 5s budget"
        )

    def test_tier1_classification_latency(self):
        """Tier 1 (rule-based) classification must complete in < 500ms."""
        config = PDFProcessorConfig()
        inspector = PDFInspector(config)

        # Build a representative 3-page text-native profile
        pages = [
            _make_page_profile(page_number=i, text_length=1500, word_count=300)
            for i in range(1, 4)
        ]
        profile = _make_document_profile(pages=pages)

        start = time.monotonic()
        _result = inspector.classify(profile)
        elapsed = time.monotonic() - start

        assert elapsed < 0.5, (
            f"Tier 1 classification took {elapsed:.3f}s, exceeds 500ms budget"
        )


# ---------------------------------------------------------------------------
# Throughput SLO Benchmarks (integration-level, real backends)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.benchmark
@requires_backends
class TestThroughputBenchmarks:
    """End-to-end throughput SLO tests with real backends."""

    @pytest.fixture()
    def benchmark_router(self):
        """Router wired to real backends for benchmarking."""
        from ingestkit_excel.backends import (
            OllamaEmbedding,
            OllamaLLM,
            QdrantVectorStore,
            SQLiteStructuredDB,
        )

        config = PDFProcessorConfig(
            tenant_id="benchmark-test",
            default_collection="test_benchmark",
        )
        return PDFRouter(
            vector_store=QdrantVectorStore(url="http://localhost:6333"),
            structured_db=SQLiteStructuredDB(":memory:"),
            llm=OllamaLLM(base_url="http://localhost:11434"),
            embedder=OllamaEmbedding(
                base_url="http://localhost:11434", model="nomic-embed-text"
            ),
            config=config,
        )

    def test_path_a_throughput_slo(self, benchmark_router, text_native_pdf):
        """Path A must achieve >= 50 pages/sec on text-native PDFs."""
        stats = _measure_throughput(
            router=benchmark_router,
            pdf_path=str(text_native_pdf),
            page_count=3,
            iterations=5,
        )

        assert stats["pages_per_sec"] >= 50, (
            f"Path A throughput {stats['pages_per_sec']:.1f} pages/sec "
            f"is below SLO of 50 pages/sec"
        )

    @pytest.mark.ocr
    @requires_tesseract
    def test_path_b_throughput_slo(self, benchmark_router, scanned_pdf):
        """Path B must achieve >= 10 pages/sec on scanned PDFs."""
        stats = _measure_throughput(
            router=benchmark_router,
            pdf_path=str(scanned_pdf),
            page_count=2,
            iterations=3,
        )

        assert stats["pages_per_sec"] >= 10, (
            f"Path B throughput {stats['pages_per_sec']:.1f} pages/sec "
            f"is below SLO of 10 pages/sec"
        )
