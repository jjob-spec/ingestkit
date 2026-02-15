"""Tests for ExecutionBackend protocol, LocalExecutionBackend, and DistributedExecutionBackend.

Covers: Protocol conformance, local backend submit/execute/get_result cycle,
distributed backend stub behavior, router integration with execution backend,
error handling (timeout, not found), and config-driven backend selection.
"""

from __future__ import annotations

import time
from typing import Any
import pytest

from ingestkit_core.models import ClassificationTier, WrittenArtifacts
from ingestkit_pdf.config import PDFProcessorConfig
from ingestkit_pdf.errors import ErrorCode, IngestError
from ingestkit_pdf.execution import (
    DistributedExecutionBackend,
    ExecutionBackend,
    ExecutionError,
    LocalExecutionBackend,
)
from ingestkit_pdf.models import (
    ClassificationResult,
    ClassificationStageResult,
    ExtractionQuality,
    IngestionMethod,
    PDFType,
    ParseStageResult,
    ProcessingResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_result(file_path: str = "/tmp/test.pdf") -> ProcessingResult:
    """Build a minimal ProcessingResult for testing."""
    empty_quality = ExtractionQuality(
        printable_ratio=0.0,
        avg_words_per_page=0.0,
        pages_with_text=0,
        total_pages=0,
        extraction_method="none",
    )
    return ProcessingResult(
        file_path=file_path,
        ingest_key="test-key-abc",
        ingest_run_id="test-run-001",
        tenant_id=None,
        parse_result=ParseStageResult(
            pages_extracted=0,
            pages_skipped=0,
            skipped_reasons={},
            extraction_method="none",
            overall_quality=empty_quality,
            parse_duration_seconds=0.0,
        ),
        classification_result=ClassificationStageResult(
            tier_used=ClassificationTier.RULE_BASED,
            pdf_type=PDFType.TEXT_NATIVE,
            confidence=0.8,
            signals=None,
            reasoning="Test classification.",
            per_page_types={},
            classification_duration_seconds=0.0,
        ),
        ocr_result=None,
        embed_result=None,
        classification=ClassificationResult(
            pdf_type=PDFType.TEXT_NATIVE,
            confidence=0.8,
            tier_used=ClassificationTier.RULE_BASED,
            reasoning="Test classification.",
            per_page_types={},
        ),
        ingestion_method=IngestionMethod.TEXT_EXTRACTION,
        chunks_created=0,
        tables_created=0,
        tables=[],
        written=WrittenArtifacts(),
        errors=[],
        warnings=[],
        error_details=[],
        processing_time_seconds=0.1,
    )


class MockExecutionBackend:
    """Mock that satisfies the ExecutionBackend protocol."""

    def __init__(self, results: dict[str, ProcessingResult] | None = None) -> None:
        self._results = results or {}
        self._counter = 0
        self.submitted: list[tuple[str, Any]] = []

    def submit(self, file_path: str, config: PDFProcessorConfig) -> str:
        self._counter += 1
        job_id = f"mock-{self._counter}"
        self.submitted.append((file_path, config))
        return job_id

    def get_result(self, job_id: str, timeout: float | None = None) -> ProcessingResult:
        if job_id in self._results:
            return self._results[job_id]
        return _make_minimal_result()


# ---------------------------------------------------------------------------
# Module-level picklable functions (required by ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def _picklable_process_fn(file_path: str, config_dict: dict) -> ProcessingResult:
    """Module-level callable that ProcessPoolExecutor can pickle."""
    return _make_minimal_result(file_path)


def _picklable_slow_fn(file_path: str, config_dict: dict) -> ProcessingResult:
    """Module-level callable that sleeps (for timeout tests)."""
    time.sleep(10)
    return _make_minimal_result(file_path)


def _picklable_failing_fn(file_path: str, config_dict: dict) -> ProcessingResult:
    """Module-level callable that always raises."""
    raise RuntimeError("Simulated processing failure")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_process_fn():
    """A picklable callable that returns a minimal ProcessingResult."""
    return _picklable_process_fn


@pytest.fixture
def failing_process_fn():
    """A picklable callable that always raises."""
    return _picklable_failing_fn


@pytest.fixture
def local_backend(mock_process_fn):
    return LocalExecutionBackend(process_fn=mock_process_fn, max_workers=2)


@pytest.fixture
def test_config():
    return PDFProcessorConfig()


# ---------------------------------------------------------------------------
# Protocol Conformance Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProtocolConformance:
    """Verify that all backends satisfy the ExecutionBackend protocol."""

    def test_local_backend_satisfies_protocol(self, mock_process_fn):
        backend = LocalExecutionBackend(process_fn=mock_process_fn)
        assert isinstance(backend, ExecutionBackend)

    def test_distributed_backend_satisfies_protocol(self):
        backend = DistributedExecutionBackend(queue_url="redis://localhost:6379")
        assert isinstance(backend, ExecutionBackend)

    def test_mock_backend_satisfies_protocol(self):
        backend = MockExecutionBackend()
        assert isinstance(backend, ExecutionBackend)


# ---------------------------------------------------------------------------
# LocalExecutionBackend Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLocalExecutionBackend:
    """Tests for the local (ProcessPoolExecutor) execution backend."""

    def test_submit_returns_job_id(self, local_backend, test_config):
        job_id = local_backend.submit("/tmp/test.pdf", test_config)
        assert isinstance(job_id, str)
        assert len(job_id) > 0

    def test_submit_multiple_returns_unique_ids(self, local_backend, test_config):
        ids = [
            local_backend.submit(f"/tmp/test{i}.pdf", test_config)
            for i in range(5)
        ]
        assert len(set(ids)) == 5

    def test_execute_all_processes_pending(self, local_backend, test_config):
        job_id = local_backend.submit("/tmp/test.pdf", test_config)
        local_backend.execute_all(timeout=30.0)
        result = local_backend.get_result(job_id)
        assert isinstance(result, ProcessingResult)
        assert result.file_path == "/tmp/test.pdf"

    def test_execute_all_multiple_files(self, local_backend, test_config):
        job_ids = [
            local_backend.submit(f"/tmp/test{i}.pdf", test_config)
            for i in range(3)
        ]
        local_backend.execute_all(timeout=30.0)
        for i, job_id in enumerate(job_ids):
            result = local_backend.get_result(job_id)
            assert result.file_path == f"/tmp/test{i}.pdf"

    def test_get_result_returns_processing_result(self, local_backend, test_config):
        job_id = local_backend.submit("/tmp/test.pdf", test_config)
        local_backend.execute_all(timeout=30.0)
        result = local_backend.get_result(job_id)
        assert isinstance(result, ProcessingResult)

    def test_get_result_unknown_job_raises(self, local_backend):
        with pytest.raises(ExecutionError) as exc_info:
            local_backend.get_result("nonexistent-job-id")
        assert exc_info.value.code == ErrorCode.E_EXECUTION_NOT_FOUND

    def test_get_result_before_execute_raises(self, local_backend, test_config):
        job_id = local_backend.submit("/tmp/test.pdf", test_config)
        with pytest.raises(ExecutionError) as exc_info:
            local_backend.get_result(job_id)
        assert exc_info.value.code == ErrorCode.E_EXECUTION_NOT_FOUND

    def test_execute_all_exception_produces_error_result(self, failing_process_fn, test_config):
        backend = LocalExecutionBackend(process_fn=failing_process_fn, max_workers=1)
        job_id = backend.submit("/tmp/fail.pdf", test_config)
        backend.execute_all(timeout=30.0)
        result = backend.get_result(job_id)
        assert isinstance(result, ProcessingResult)
        assert ErrorCode.E_EXECUTION_SUBMIT.value in result.errors

    def test_execute_all_clears_pending(self, local_backend, test_config):
        local_backend.submit("/tmp/test.pdf", test_config)
        local_backend.execute_all(timeout=30.0)
        # Pending should be cleared -- second execute_all is a no-op
        local_backend.execute_all(timeout=30.0)
        # No error means pending was cleared

    def test_execute_all_no_pending_is_noop(self, local_backend):
        # Should not raise
        local_backend.execute_all(timeout=30.0)

    def test_max_workers_capped_by_pending_count(self, mock_process_fn, test_config):
        """When fewer files than max_workers, pool uses file count."""
        backend = LocalExecutionBackend(process_fn=mock_process_fn, max_workers=8)
        backend.submit("/tmp/test1.pdf", test_config)
        backend.submit("/tmp/test2.pdf", test_config)
        # This exercises the min(len(pending), max_workers) logic
        backend.execute_all(timeout=30.0)
        r1 = backend.get_result(list(backend._results.keys())[0])
        assert isinstance(r1, ProcessingResult)

    def test_submit_stores_config_object(self, test_config):
        """Verify submit stores PDFProcessorConfig, execute_all converts via model_dump."""
        backend = LocalExecutionBackend(
            process_fn=_picklable_process_fn, max_workers=1,
        )
        job_id = backend.submit("/tmp/test.pdf", test_config)
        # Before execute_all, pending stores (file_path, config) with config as object
        stored_fp, stored_config = backend._pending[job_id]
        assert stored_fp == "/tmp/test.pdf"
        assert isinstance(stored_config, PDFProcessorConfig)
        assert stored_config.execution_backend == "local"

        # execute_all calls model_dump() internally before passing to process_fn
        backend.execute_all(timeout=30.0)
        result = backend.get_result(job_id)
        assert isinstance(result, ProcessingResult)


# ---------------------------------------------------------------------------
# DistributedExecutionBackend Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDistributedExecutionBackend:
    """Tests for the distributed (stub) execution backend."""

    def test_distributed_submit_raises_not_implemented(self):
        backend = DistributedExecutionBackend(queue_url="redis://localhost:6379")
        with pytest.raises(NotImplementedError, match="planned feature"):
            backend.submit("/tmp/test.pdf", PDFProcessorConfig())

    def test_distributed_get_result_raises_not_implemented(self):
        backend = DistributedExecutionBackend(queue_url="redis://localhost:6379")
        with pytest.raises(NotImplementedError, match="planned feature"):
            backend.get_result("some-job-id")

    def test_distributed_init_stores_queue_url(self):
        url = "redis://myhost:6379/0"
        backend = DistributedExecutionBackend(queue_url=url)
        assert backend._queue_url == url


# ---------------------------------------------------------------------------
# Router Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRouterExecutionIntegration:
    """Tests verifying PDFRouter integration with ExecutionBackend."""

    def test_router_uses_injected_execution_backend(self):
        """Router with mock execution backend delegates to it."""
        from ingestkit_pdf.router import PDFRouter

        mock_backend = MockExecutionBackend()

        # Use mock backends for the router
        class _Stub:
            """Minimal stub satisfying protocol."""
            def store(self, *a, **kw): ...
            def query(self, *a, **kw): return []
            def execute(self, *a, **kw): return []
            def classify(self, *a, **kw): return {}
            def generate(self, *a, **kw): return ""
            def embed(self, *a, **kw): return [[0.0]]
            def embed_batch(self, *a, **kw): return [[0.0]]

        stub = _Stub()
        router = PDFRouter(
            vector_store=stub,
            structured_db=stub,
            llm=stub,
            embedder=stub,
            execution=mock_backend,
        )
        assert router._execution is mock_backend

    def test_router_defaults_to_local_backend(self):
        """Router without execution param uses LocalExecutionBackend."""
        from ingestkit_pdf.router import PDFRouter

        class _Stub:
            def store(self, *a, **kw): ...
            def query(self, *a, **kw): return []
            def execute(self, *a, **kw): return []
            def classify(self, *a, **kw): return {}
            def generate(self, *a, **kw): return ""
            def embed(self, *a, **kw): return [[0.0]]
            def embed_batch(self, *a, **kw): return [[0.0]]

        stub = _Stub()
        router = PDFRouter(
            vector_store=stub,
            structured_db=stub,
            llm=stub,
            embedder=stub,
        )
        assert isinstance(router._execution, LocalExecutionBackend)

    def test_process_batch_empty_returns_empty(self):
        """process_batch([]) returns [] without calling execution backend."""
        from ingestkit_pdf.router import PDFRouter

        mock_backend = MockExecutionBackend()

        class _Stub:
            def store(self, *a, **kw): ...
            def query(self, *a, **kw): return []
            def execute(self, *a, **kw): return []
            def classify(self, *a, **kw): return {}
            def generate(self, *a, **kw): return ""
            def embed(self, *a, **kw): return [[0.0]]
            def embed_batch(self, *a, **kw): return [[0.0]]

        stub = _Stub()
        router = PDFRouter(
            vector_store=stub,
            structured_db=stub,
            llm=stub,
            embedder=stub,
            execution=mock_backend,
        )
        result = router.process_batch([])
        assert result == []
        assert len(mock_backend.submitted) == 0

    def test_process_batch_delegates_to_execution(self):
        """process_batch() calls submit() + get_result() on the backend."""
        from ingestkit_pdf.router import PDFRouter

        mock_backend = MockExecutionBackend()

        class _Stub:
            def store(self, *a, **kw): ...
            def query(self, *a, **kw): return []
            def execute(self, *a, **kw): return []
            def classify(self, *a, **kw): return {}
            def generate(self, *a, **kw): return ""
            def embed(self, *a, **kw): return [[0.0]]
            def embed_batch(self, *a, **kw): return [[0.0]]

        stub = _Stub()
        router = PDFRouter(
            vector_store=stub,
            structured_db=stub,
            llm=stub,
            embedder=stub,
            execution=mock_backend,
        )
        results = router.process_batch(["/tmp/a.pdf", "/tmp/b.pdf"])
        assert len(results) == 2
        assert len(mock_backend.submitted) == 2
        assert mock_backend.submitted[0][0] == "/tmp/a.pdf"
        assert mock_backend.submitted[1][0] == "/tmp/b.pdf"


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExecutionConfig:
    """Tests for execution-related config fields."""

    def test_config_defaults(self):
        config = PDFProcessorConfig()
        assert config.execution_backend == "local"
        assert config.execution_max_workers == 4
        assert config.execution_queue_url is None

    def test_config_distributed(self):
        config = PDFProcessorConfig(
            execution_backend="distributed",
            execution_queue_url="redis://localhost:6379/0",
        )
        assert config.execution_backend == "distributed"
        assert config.execution_queue_url == "redis://localhost:6379/0"

    def test_config_custom_max_workers(self):
        config = PDFProcessorConfig(execution_max_workers=8)
        assert config.execution_max_workers == 8


# ---------------------------------------------------------------------------
# Error Code Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExecutionErrorCodes:
    """Tests verifying the new execution error codes exist."""

    def test_error_codes_exist(self):
        assert ErrorCode.E_EXECUTION_TIMEOUT.value == "E_EXECUTION_TIMEOUT"
        assert ErrorCode.E_EXECUTION_SUBMIT.value == "E_EXECUTION_SUBMIT"
        assert ErrorCode.E_EXECUTION_NOT_FOUND.value == "E_EXECUTION_NOT_FOUND"

    def test_ingest_error_with_execution_code(self):
        err = IngestError(
            code=ErrorCode.E_EXECUTION_TIMEOUT,
            message="Timed out",
            stage="execution",
            recoverable=False,
        )
        assert err.code == ErrorCode.E_EXECUTION_TIMEOUT
