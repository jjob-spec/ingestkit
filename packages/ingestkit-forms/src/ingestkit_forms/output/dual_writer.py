"""Dual-write orchestrator for form output.

Coordinates DB row and RAG chunk writes with consistency modes
(best_effort or strict_atomic), configurable PII redaction, and
rollback of partially-written artifacts.

See spec sections 8.0 and 8.5 for authoritative definitions.
"""

from __future__ import annotations

import logging
import re
import time

from ingestkit_forms.config import FormProcessorConfig
from ingestkit_forms.errors import FormErrorCode, FormIngestError, FormIngestException
from ingestkit_forms.models import (
    EmbedStageResult,
    FormExtractionResult,
    FormTemplate,
    FormWrittenArtifacts,
    RollbackResult,
)
from ingestkit_forms.output.chunk_writer import FormChunkWriter
from ingestkit_forms.output.db_writer import FormDBWriter
from ingestkit_forms.protocols import FormDBBackend, VectorStoreBackend

logger = logging.getLogger("ingestkit_forms")


# ---------------------------------------------------------------------------
# Redaction helpers
# ---------------------------------------------------------------------------


def apply_redaction(value: str, patterns: list[re.Pattern[str]]) -> str:
    """Apply compiled regex patterns to redact matching substrings."""
    for pattern in patterns:
        value = pattern.sub("[REDACTED]", value)
    return value


def redact_extraction(
    extraction: FormExtractionResult,
    config: FormProcessorConfig,
    destination: str,
) -> FormExtractionResult:
    """Return a (possibly deep-copied) extraction with redacted field values.

    Args:
        extraction: The original extraction result.
        config: Configuration with ``redact_patterns`` and ``redact_target``.
        destination: Either ``"db"`` or ``"chunks"`` -- the write target.

    Returns:
        The original extraction if no redaction needed, or a deep copy
        with field values redacted. The original is never mutated.
    """
    if not config.redact_patterns:
        return extraction

    target = config.redact_target  # "both", "chunks_only", "db_only"
    if target == "chunks_only" and destination == "db":
        return extraction
    if target == "db_only" and destination == "chunks":
        return extraction

    # Deep copy to avoid mutating the original
    redacted = extraction.model_copy(deep=True)
    compiled = [re.compile(p) for p in config.redact_patterns]

    for field in redacted.fields:
        if isinstance(field.value, str):
            new_value = apply_redaction(field.value, compiled)
            if new_value != field.value:
                field.value = new_value
                field.redacted = True
                if field.raw_value is not None:
                    field.raw_value = "[REDACTED]"

    return redacted


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


def rollback_written_artifacts(
    written: FormWrittenArtifacts,
    vector_backend: VectorStoreBackend | None,
    db_backend: FormDBBackend | None,
    config: FormProcessorConfig,
) -> RollbackResult:
    """Roll back written artifacts: delete vector points first, then DB rows.

    Intentional extension over spec 8.5: accepts ``config`` parameter
    to access ``backend_max_retries`` and ``backend_backoff_base`` for
    retry logic during rollback.

    Returns a ``RollbackResult`` indicating how many items were deleted
    and whether rollback was complete.
    """
    result = RollbackResult()
    max_attempts = config.backend_max_retries + 1

    # 1. Delete vector points first (less critical, faster)
    if (
        vector_backend is not None
        and written.vector_point_ids
        and written.vector_collection
    ):
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                count = vector_backend.delete_by_ids(
                    written.vector_collection, written.vector_point_ids
                )
                result.vector_points_deleted = count
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if attempt < max_attempts - 1:
                    sleep_time = config.backend_backoff_base * (2 ** attempt)
                    time.sleep(sleep_time)
        if last_exc is not None:
            result.errors.append(
                f"Vector rollback failed after {max_attempts} attempts: {last_exc}"
            )
            result.fully_rolled_back = False

    # 2. Delete DB rows second
    if db_backend is not None and written.db_row_ids and written.db_table_names:
        for table_name in written.db_table_names:
            last_exc = None
            for attempt in range(max_attempts):
                try:
                    count = db_backend.delete_rows(
                        table_name, "_form_id", written.db_row_ids
                    )
                    result.db_rows_deleted += count
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_attempts - 1:
                        sleep_time = config.backend_backoff_base * (2 ** attempt)
                        time.sleep(sleep_time)
            if last_exc is not None:
                result.errors.append(
                    f"DB rollback failed for {table_name} after {max_attempts} "
                    f"attempts: {last_exc}"
                )
                result.fully_rolled_back = False

    return result


# ---------------------------------------------------------------------------
# FormDualWriter
# ---------------------------------------------------------------------------


class FormDualWriter:
    """Orchestrates DB row + RAG chunk writes with consistency enforcement.

    Takes pre-constructed ``FormDBWriter`` and ``FormChunkWriter`` via
    dependency injection. Does not inherit from either writer, keeping
    each component independently testable.
    """

    def __init__(
        self,
        db_writer: FormDBWriter,
        chunk_writer: FormChunkWriter,
        config: FormProcessorConfig,
    ) -> None:
        self._db_writer = db_writer
        self._chunk_writer = chunk_writer
        self._config = config

    def write(
        self,
        extraction: FormExtractionResult,
        template: FormTemplate,
        source_uri: str,
        ingest_key: str,
        ingest_run_id: str,
    ) -> tuple[
        FormWrittenArtifacts,
        list[str],
        list[str],
        list[FormIngestError],
        EmbedStageResult | None,
    ]:
        """Execute dual write: DB row + RAG chunks.

        Returns:
            ``(written, errors, warnings, error_details, embed_result)``
        """
        config = self._config
        written = FormWrittenArtifacts(vector_collection=config.default_collection)
        errors: list[str] = []
        warnings: list[str] = []
        error_details: list[FormIngestError] = []
        embed_result: EmbedStageResult | None = None

        # --- DB Write ---
        db_success = False
        db_extraction = redact_extraction(extraction, config, "db")
        try:
            table_name = self._db_writer.ensure_table(template)
            form_id = self._db_writer.write_row(
                table_name, db_extraction, ingest_key, ingest_run_id
            )
            written.db_table_names.append(table_name)
            written.db_row_ids.append(form_id)
            db_success = True
        except (FormIngestException, Exception) as exc:
            msg = f"DB write failed: {exc}"
            errors.append(msg)
            error_details.append(
                FormIngestError(
                    code=FormErrorCode.E_FORM_DB_WRITE_FAILED,
                    message=msg,
                    stage="output",
                    recoverable=True,
                )
            )

        # --- Chunk Write ---
        chunk_success = False
        chunk_extraction = redact_extraction(extraction, config, "chunks")
        try:
            point_ids, embed_result = self._chunk_writer.write_chunks(
                chunk_extraction, template, source_uri, ingest_key, ingest_run_id
            )
            written.vector_point_ids.extend(point_ids)
            chunk_success = True
        except (FormIngestException, Exception) as exc:
            msg = f"Chunk write failed: {exc}"
            errors.append(msg)
            error_details.append(
                FormIngestError(
                    code=FormErrorCode.E_FORM_CHUNK_WRITE_FAILED,
                    message=msg,
                    stage="output",
                    recoverable=True,
                )
            )

        # --- Consistency enforcement ---
        if config.dual_write_mode == "strict_atomic":
            self._enforce_strict_atomic(
                db_success, chunk_success, written, warnings
            )
        elif config.dual_write_mode == "best_effort":
            if db_success != chunk_success:
                # Partial write
                warnings.append("W_FORM_PARTIAL_WRITE")
                error_details.append(
                    FormIngestError(
                        code=FormErrorCode.E_FORM_DUAL_WRITE_PARTIAL,
                        message="Partial dual write: one backend succeeded, the other failed.",
                        stage="output",
                        recoverable=True,
                    )
                )

        return written, errors, warnings, error_details, embed_result

    def _enforce_strict_atomic(
        self,
        db_success: bool,
        chunk_success: bool,
        written: FormWrittenArtifacts,
        warnings: list[str],
    ) -> None:
        """Roll back the successful side when the other fails (strict_atomic mode)."""
        if db_success and not chunk_success:
            # Rollback DB
            rb = rollback_written_artifacts(
                written,
                vector_backend=None,
                db_backend=self._db_writer._db,
                config=self._config,
            )
            if not rb.fully_rolled_back:
                warnings.append("W_FORM_ROLLBACK_FAILED")
            # Clear DB artifacts
            written.db_table_names.clear()
            written.db_row_ids.clear()

        elif chunk_success and not db_success:
            # Rollback vectors
            rb = rollback_written_artifacts(
                written,
                vector_backend=self._chunk_writer._vector_store,
                db_backend=None,
                config=self._config,
            )
            if not rb.fully_rolled_back:
                warnings.append("W_FORM_ROLLBACK_FAILED")
            # Clear vector artifacts
            written.vector_point_ids.clear()
