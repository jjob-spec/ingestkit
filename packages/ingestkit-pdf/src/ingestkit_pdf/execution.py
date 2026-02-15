"""Pluggable execution backends for the ingestkit-pdf pipeline (SPEC section 18.3).

Provides:
- ExecutionBackend: Protocol defining the submit/get_result interface.
- LocalExecutionBackend: Default backend using ProcessPoolExecutor.
- DistributedExecutionBackend: Stub for queue-based distributed processing.
"""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

from ingestkit_pdf.errors import ErrorCode, IngestError

if TYPE_CHECKING:
    from ingestkit_pdf.config import PDFProcessorConfig
    from ingestkit_pdf.models import ProcessingResult

logger = logging.getLogger("ingestkit_pdf")


class ExecutionError(Exception):
    """Raised when an execution backend operation fails.

    Wraps an :class:`ErrorCode` and human-readable message so callers
    can distinguish between different failure modes (not found, timeout,
    submit failure).
    """

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


@runtime_checkable
class ExecutionBackend(Protocol):
    """Pluggable execution backend for document processing."""

    def submit(self, file_path: str, config: PDFProcessorConfig) -> str:
        """Submit a document for processing. Returns a job_id."""
        ...

    def get_result(self, job_id: str, timeout: float | None = None) -> ProcessingResult:
        """Block until result is available or timeout."""
        ...


class LocalExecutionBackend:
    """v1.0 default: process via ProcessPoolExecutor with per-document isolation.

    Each submit() queues work lazily; execute_all() dispatches all pending
    jobs to a ProcessPoolExecutor (matching the existing process_batch
    batching behavior).
    """

    def __init__(
        self,
        process_fn: Callable[[str, dict], ProcessingResult],
        max_workers: int = 4,
    ) -> None:
        self._process_fn = process_fn
        self._max_workers = max_workers
        self._results: dict[str, ProcessingResult] = {}
        self._pending: dict[str, tuple[str, PDFProcessorConfig]] = {}

    def submit(self, file_path: str, config: PDFProcessorConfig) -> str:
        """Queue a document for processing. Returns a job_id.

        Execution is deferred until execute_all() is called, preserving
        the batching semantics of the original process_batch().
        """
        job_id = str(uuid.uuid4())
        self._pending[job_id] = (file_path, config)
        return job_id

    def get_result(self, job_id: str, timeout: float | None = None) -> ProcessingResult:
        """Retrieve the result for a submitted job.

        Raises ExecutionError with E_EXECUTION_NOT_FOUND if the job_id is
        unknown or has not yet been executed.
        """
        if job_id in self._results:
            return self._results[job_id]
        raise ExecutionError(
            code=ErrorCode.E_EXECUTION_NOT_FOUND,
            message=f"Job {job_id} not found or not yet executed",
        )

    def execute_all(self, timeout: float | None = None) -> None:
        """Run all pending jobs via ProcessPoolExecutor.

        Converts PDFProcessorConfig to dict via model_dump() before
        passing to the process_fn (which expects (str, dict) signature,
        matching _process_single_file).
        """
        if not self._pending:
            return

        num_workers = min(len(self._pending), self._max_workers)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_job_id = {}
            for job_id, (file_path, config) in self._pending.items():
                # PLAN-CHECK Issue #1: convert config to dict for _process_single_file
                config_dict = config.model_dump()
                future = executor.submit(self._process_fn, file_path, config_dict)
                future_to_job_id[future] = (job_id, file_path, config)

            for future in as_completed(future_to_job_id):
                job_id, file_path, config = future_to_job_id[future]
                try:
                    result = future.result(timeout=timeout)
                    self._results[job_id] = result
                except (FuturesTimeoutError, TimeoutError):
                    logger.error(
                        "ingestkit_pdf | execution | job=%s | file=%s | "
                        "code=%s | detail=Processing timed out",
                        job_id,
                        file_path,
                        ErrorCode.E_EXECUTION_TIMEOUT.value,
                    )
                    self._results[job_id] = self._build_error_result(
                        file_path=file_path,
                        error_code=ErrorCode.E_EXECUTION_TIMEOUT.value,
                        message=f"Processing timed out after {timeout}s",
                        elapsed=float(timeout) if timeout else 0.0,
                    )
                except Exception as exc:
                    logger.error(
                        "ingestkit_pdf | execution | job=%s | file=%s | "
                        "code=%s | detail=%s",
                        job_id,
                        file_path,
                        ErrorCode.E_EXECUTION_SUBMIT.value,
                        str(exc),
                    )
                    self._results[job_id] = self._build_error_result(
                        file_path=file_path,
                        error_code=ErrorCode.E_EXECUTION_SUBMIT.value,
                        message=f"Processing failed: {exc}",
                        elapsed=0.0,
                    )

        self._pending.clear()

    @staticmethod
    def _build_error_result(
        file_path: str,
        error_code: str,
        message: str,
        elapsed: float,
    ) -> ProcessingResult:
        """Build a minimal error ProcessingResult for execution failures."""
        from ingestkit_pdf.models import (
            ClassificationResult,
            ClassificationStageResult,
            ExtractionQuality,
            IngestionMethod,
            PDFType,
            ParseStageResult,
            ProcessingResult,
        )
        from ingestkit_core.models import ClassificationTier, WrittenArtifacts

        empty_quality = ExtractionQuality(
            printable_ratio=0.0,
            avg_words_per_page=0.0,
            pages_with_text=0,
            total_pages=0,
            extraction_method="none",
        )
        return ProcessingResult(
            file_path=file_path,
            ingest_key="",
            ingest_run_id=str(uuid.uuid4()),
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
                confidence=0.0,
                signals=None,
                reasoning="Error before classification.",
                per_page_types={},
                classification_duration_seconds=0.0,
            ),
            ocr_result=None,
            embed_result=None,
            classification=ClassificationResult(
                pdf_type=PDFType.TEXT_NATIVE,
                confidence=0.0,
                tier_used=ClassificationTier.RULE_BASED,
                reasoning="Error before classification.",
                per_page_types={},
            ),
            ingestion_method=IngestionMethod.TEXT_EXTRACTION,
            chunks_created=0,
            tables_created=0,
            tables=[],
            written=WrittenArtifacts(),
            errors=[error_code],
            warnings=[],
            error_details=[
                IngestError(
                    code=ErrorCode(error_code),
                    message=message,
                    stage="execution",
                    recoverable=False,
                ),
            ],
            processing_time_seconds=elapsed,
        )


class DistributedExecutionBackend:
    """v1.1+: submit to a queue (Redis, RabbitMQ, etc.), workers process.

    This is a stub implementation. The full distributed backend requires
    queue infrastructure (Redis, RabbitMQ) and a worker process that
    dequeues jobs and calls router.process(). See SPEC section 18.3 and
    the Phase 2 roadmap (section 24.3) for details.
    """

    def __init__(self, queue_url: str) -> None:
        self._queue_url = queue_url

    def submit(self, file_path: str, config: PDFProcessorConfig) -> str:
        """Submit a document for distributed processing.

        Raises NotImplementedError -- distributed backend is not yet implemented.
        """
        raise NotImplementedError(
            "DistributedExecutionBackend is a planned feature (SPEC section 18.3, "
            "Phase 2). Requires queue infrastructure. Use LocalExecutionBackend."
        )

    def get_result(self, job_id: str, timeout: float | None = None) -> ProcessingResult:
        """Retrieve result from distributed processing.

        Raises NotImplementedError -- distributed backend is not yet implemented.
        """
        raise NotImplementedError(
            "DistributedExecutionBackend is a planned feature (SPEC section 18.3, "
            "Phase 2). Requires queue infrastructure. Use LocalExecutionBackend."
        )
