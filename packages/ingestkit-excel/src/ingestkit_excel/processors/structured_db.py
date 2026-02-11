"""Path A processor: structured DB ingestion with optional row serialization."""

from __future__ import annotations

import hashlib
import logging
import re
import time
import uuid
from pathlib import Path

import pandas as pd

from ingestkit_excel.config import ExcelProcessorConfig
from ingestkit_excel.errors import ErrorCode, IngestError
from ingestkit_excel.models import (
    ChunkMetadata,
    ChunkPayload,
    ClassificationResult,
    ClassificationStageResult,
    EmbedStageResult,
    FileProfile,
    IngestionMethod,
    ParseStageResult,
    ProcessingResult,
    SheetProfile,
    WrittenArtifacts,
)
from ingestkit_excel.protocols import (
    EmbeddingBackend,
    StructuredDBBackend,
    VectorStoreBackend,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def clean_name(raw: str) -> str:
    """Clean a raw name (column or sheet) for use as a DB identifier.

    Rules:
    1. Lowercase
    2. Replace non-alphanumeric/underscore with underscore
    3. Collapse consecutive underscores
    4. Strip leading/trailing underscores
    """
    name = raw.lower()
    name = re.sub(r"[^a-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    return name


def deduplicate_names(names: list[str]) -> list[str]:
    """Deduplicate a list of cleaned names by appending _1, _2, etc.

    Empty names after cleaning become ``column_N`` where N is the 0-based index.
    """
    seen: dict[str, int] = {}
    result: list[str] = []
    for i, name in enumerate(names):
        if not name:
            name = f"column_{i}"
        if name in seen:
            seen[name] += 1
            result.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            result.append(name)
    return result


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


class StructuredDBProcessor:
    """Path A processor: writes each sheet to a structured DB table,
    generates NL schema descriptions, embeds them, and optionally
    serializes rows for direct vector search.
    """

    def __init__(
        self,
        structured_db: StructuredDBBackend,
        vector_store: VectorStoreBackend,
        embedder: EmbeddingBackend,
        config: ExcelProcessorConfig,
    ) -> None:
        self._db = structured_db
        self._vector_store = vector_store
        self._embedder = embedder
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        file_path: str,
        profile: FileProfile,
        ingest_key: str,
        ingest_run_id: str,
        parse_result: ParseStageResult,
        classification_result: ClassificationStageResult,
        classification: ClassificationResult,
    ) -> ProcessingResult:
        """Process an Excel file via Path A (structured DB ingestion).

        Args:
            file_path: Absolute path to the Excel file on disk.
            profile: Pre-computed structural profile of the file.
            ingest_key: Deterministic SHA-256 hex string from ``IngestKey.key``.
            ingest_run_id: UUID4 string unique to this processing run.
            parse_result: Typed output of the parsing stage.
            classification_result: Typed output of the classification stage.
            classification: Simplified classification result.

        Returns:
            A fully-assembled ``ProcessingResult``.
        """
        start_time = time.monotonic()

        config = self._config
        collection = config.default_collection
        source_uri = f"file://{Path(file_path).resolve().as_posix()}"
        db_uri = self._db.get_connection_uri()

        errors: list[str] = []
        warnings: list[str] = []
        error_details: list[IngestError] = []

        written = WrittenArtifacts(vector_collection=collection)
        tables: list[str] = []
        total_chunks = 0
        total_texts_embedded = 0
        embed_duration = 0.0

        # Ensure vector collection exists
        vector_size = self._embedder.dimension()
        self._vector_store.ensure_collection(collection, vector_size)

        chunk_index_counter = 0  # global chunk index across all sheets

        for sheet in profile.sheets:
            # --- Skip logic ---
            if sheet.is_hidden:
                warnings.append(ErrorCode.W_SHEET_SKIPPED_HIDDEN.value)
                logger.info("Skipping hidden sheet: %s", sheet.name)
                continue
            if sheet.row_count == 0 and sheet.col_count == 0:
                # chart-only heuristic
                warnings.append(ErrorCode.W_SHEET_SKIPPED_CHART.value)
                logger.info("Skipping chart-only sheet: %s", sheet.name)
                continue
            if sheet.row_count > config.max_rows_in_memory:
                warnings.append(ErrorCode.W_ROWS_TRUNCATED.value)
                logger.warning(
                    "Sheet %s has %d rows, exceeds max_rows_in_memory (%d), skipping",
                    sheet.name,
                    sheet.row_count,
                    config.max_rows_in_memory,
                )
                continue

            try:
                # --- Step 1: Load DataFrame ---
                header_arg = sheet.header_row_index if sheet.header_row_detected else None
                df = pd.read_excel(
                    file_path,
                    sheet_name=sheet.name,
                    header=header_arg,
                )

                # --- Step 2: Clean column names ---
                if config.clean_column_names:
                    cleaned = [clean_name(str(c)) for c in df.columns]
                    cleaned = deduplicate_names(cleaned)
                    df.columns = cleaned

                # --- Step 3: Auto-detect dates ---
                df = self._auto_detect_dates(df)

                # --- Step 4: Write to DB ---
                table_name = clean_name(sheet.name)
                if not table_name:
                    table_name = f"sheet_{profile.sheets.index(sheet)}"
                # Deduplicate table names across sheets
                if table_name in tables:
                    suffix = 1
                    while f"{table_name}_{suffix}" in tables:
                        suffix += 1
                    table_name = f"{table_name}_{suffix}"

                self._db.create_table_from_dataframe(table_name, df)
                written.db_table_names.append(table_name)
                tables.append(table_name)

                # --- Step 5: Generate schema description ---
                schema_text = self._generate_schema_description(table_name, df)

                # --- Step 6: Embed schema + upsert ---
                embed_start = time.monotonic()
                vectors = self._embedder.embed(
                    [schema_text],
                    timeout=config.backend_timeout_seconds,
                )
                embed_duration += time.monotonic() - embed_start
                total_texts_embedded += 1

                chunk_hash = hashlib.sha256(schema_text.encode()).hexdigest()
                chunk_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{ingest_key}:{chunk_hash}",
                    )
                )
                columns_list = list(df.columns)

                metadata = ChunkMetadata(
                    source_uri=source_uri,
                    source_format="xlsx",
                    sheet_name=sheet.name,
                    region_id=None,
                    ingestion_method=IngestionMethod.SQL_AGENT.value,  # "sql_agent"
                    parser_used=sheet.parser_used.value,  # "openpyxl"
                    parser_version=config.parser_version,
                    chunk_index=chunk_index_counter,
                    chunk_hash=chunk_hash,
                    ingest_key=ingest_key,
                    ingest_run_id=ingest_run_id,
                    tenant_id=config.tenant_id,
                    table_name=table_name,
                    db_uri=db_uri,
                    row_count=len(df),
                    columns=columns_list,
                )
                chunk = ChunkPayload(
                    id=chunk_id,
                    text=schema_text,
                    vector=vectors[0],
                    metadata=metadata,
                )

                self._vector_store.upsert_chunks(collection, [chunk])
                written.vector_point_ids.append(chunk_id)
                chunk_index_counter += 1
                total_chunks += 1

                # --- Step 7: Optional row serialization ---
                if len(df) < config.row_serialization_limit:
                    row_chunks = self._serialize_rows(
                        table_name=table_name,
                        df=df,
                        start_chunk_index=chunk_index_counter,
                        sheet=sheet,
                        source_uri=source_uri,
                        db_uri=db_uri,
                        ingest_key=ingest_key,
                        ingest_run_id=ingest_run_id,
                    )
                    # Embed in batches
                    for batch_start in range(
                        0, len(row_chunks), config.embedding_batch_size
                    ):
                        batch = row_chunks[
                            batch_start : batch_start + config.embedding_batch_size
                        ]
                        texts = [c.text for c in batch]
                        embed_start = time.monotonic()
                        vectors = self._embedder.embed(
                            texts,
                            timeout=config.backend_timeout_seconds,
                        )
                        embed_duration += time.monotonic() - embed_start
                        total_texts_embedded += len(texts)
                        for c, vec in zip(batch, vectors):
                            c.vector = vec

                        self._vector_store.upsert_chunks(collection, list(batch))
                        for c in batch:
                            written.vector_point_ids.append(c.id)
                        total_chunks += len(batch)

                    chunk_index_counter += len(row_chunks)

            except Exception as exc:
                # Per-sheet error handling: log, record, continue to next sheet
                error_code = self._classify_backend_error(exc)
                errors.append(error_code.value)
                error_details.append(
                    IngestError(
                        code=error_code,
                        message=str(exc),
                        sheet_name=sheet.name,
                        stage="process",
                        recoverable=False,
                    )
                )
                logger.exception(
                    "Error processing sheet %s: %s", sheet.name, exc
                )
                continue

        # --- Assemble EmbedStageResult ---
        embed_result = None
        if total_texts_embedded > 0:
            embed_result = EmbedStageResult(
                texts_embedded=total_texts_embedded,
                embedding_dimension=self._embedder.dimension(),
                embed_duration_seconds=embed_duration,
            )

        elapsed = time.monotonic() - start_time

        return ProcessingResult(
            file_path=file_path,
            ingest_key=ingest_key,
            ingest_run_id=ingest_run_id,
            tenant_id=config.tenant_id,
            parse_result=parse_result,
            classification_result=classification_result,
            embed_result=embed_result,
            classification=classification,
            ingestion_method=IngestionMethod.SQL_AGENT,  # enum member, NOT string
            chunks_created=total_chunks,
            tables_created=len(tables),
            tables=tables,
            written=written,
            errors=errors,
            warnings=warnings,
            error_details=error_details,
            processing_time_seconds=elapsed,
        )

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _auto_detect_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Auto-detect and convert date columns in-place.

        Two heuristics:
        1. Excel serial dates: numeric columns with values in 35000..55000 range
        2. String dates: object columns where >50% of non-null values parse as dates
        """
        SERIAL_MIN = 35_000  # ~1995
        SERIAL_MAX = 55_000  # ~2050
        DATE_PARSE_THRESHOLD = 0.5

        for col in df.columns:
            series = df[col]
            non_null = series.dropna()
            if len(non_null) == 0:
                continue

            # Heuristic 1: Excel serial dates
            if pd.api.types.is_numeric_dtype(series):
                if (non_null >= SERIAL_MIN).all() and (non_null <= SERIAL_MAX).all():
                    try:
                        df[col] = pd.to_datetime(
                            series,
                            origin="1899-12-30",
                            unit="D",
                            errors="coerce",
                        )
                        logger.debug(
                            "Converted column '%s' from Excel serial date", col
                        )
                    except Exception:
                        pass  # leave as-is if conversion fails
                continue

            # Heuristic 2: String dates (object or string dtype)
            if series.dtype == object or pd.api.types.is_string_dtype(series):
                try:
                    parsed = pd.to_datetime(non_null, format="mixed", errors="coerce")
                    success_ratio = parsed.notna().sum() / len(non_null)
                    if success_ratio >= DATE_PARSE_THRESHOLD:
                        df[col] = pd.to_datetime(
                            series,
                            format="mixed",
                            errors="coerce",
                        )
                        logger.debug(
                            "Converted column '%s' from string dates (%.0f%% parsed)",
                            col,
                            success_ratio * 100,
                        )
                except Exception:
                    pass  # leave as-is

        return df

    def _generate_schema_description(
        self, table_name: str, df: pd.DataFrame
    ) -> str:
        """Generate a natural language schema description for embedding.

        Format:
            Table "{table_name}" contains {N} rows with columns:
            - col_name (type): description
            ...
        """
        LOW_CARDINALITY_THRESHOLD = 20
        lines = [f'Table "{table_name}" contains {len(df)} rows with columns:']

        for col in df.columns:
            series = df[col].dropna()
            col_type = self._infer_type_label(df[col])
            desc = self._describe_column(series, col_type, LOW_CARDINALITY_THRESHOLD)
            lines.append(f"- {col} ({col_type}): {desc}")

        return "\n".join(lines)

    @staticmethod
    def _infer_type_label(series: pd.Series) -> str:
        """Map a pandas dtype to a human-readable type label."""
        dtype = series.dtype
        if pd.api.types.is_integer_dtype(dtype):
            return "integer"
        if pd.api.types.is_float_dtype(dtype):
            return "float"
        if pd.api.types.is_bool_dtype(dtype):
            return "boolean"
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return "date"
        return "text"

    @staticmethod
    def _describe_column(
        series: pd.Series,
        col_type: str,
        low_card_threshold: int,
    ) -> str:
        """Generate a human-readable description of a column's values."""
        if len(series) == 0:
            return "no data"

        if col_type in ("integer", "float"):
            return f"range {series.min()} to {series.max()}"

        if col_type == "date":
            return f"ranges from {series.min()} to {series.max()}"

        if col_type == "boolean":
            return "true/false"

        # text
        n_unique = series.nunique()
        if n_unique < low_card_threshold:
            unique_vals = ", ".join(
                str(v) for v in series.unique()[:low_card_threshold]
            )
            return f"one of {unique_vals}"
        return f"{n_unique} unique values"

    def _serialize_rows(
        self,
        table_name: str,
        df: pd.DataFrame,
        start_chunk_index: int,
        sheet: SheetProfile,
        source_uri: str,
        db_uri: str,
        ingest_key: str,
        ingest_run_id: str,
    ) -> list[ChunkPayload]:
        """Serialize each row into a natural language sentence.

        Returns ChunkPayload objects with vectors set to empty lists.
        Vectors are populated later during batch embedding.

        Format: "In table '{table_name}', row {N}: {col} is {val}, ..."
        """
        config = self._config
        columns_list = list(df.columns)
        chunks: list[ChunkPayload] = []

        for row_idx, row in df.iterrows():
            # Build row text
            parts = []
            for col in df.columns:
                val = row[col]
                if pd.isna(val):
                    parts.append(f"{col} is N/A")
                else:
                    parts.append(f"{col} is {val}")
            row_number = row_idx + 1 if isinstance(row_idx, int) else row_idx
            text = f"In table '{table_name}', row {row_number}: {', '.join(parts)}."

            chunk_hash = hashlib.sha256(text.encode()).hexdigest()
            chunk_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{ingest_key}:{chunk_hash}",
                )
            )
            chunk_index = start_chunk_index + len(chunks)

            metadata = ChunkMetadata(
                source_uri=source_uri,
                source_format="xlsx",
                sheet_name=sheet.name,
                region_id=None,
                ingestion_method=IngestionMethod.SQL_AGENT.value,  # "sql_agent"
                parser_used=sheet.parser_used.value,
                parser_version=config.parser_version,
                chunk_index=chunk_index,
                chunk_hash=chunk_hash,
                ingest_key=ingest_key,
                ingest_run_id=ingest_run_id,
                tenant_id=config.tenant_id,
                table_name=table_name,
                db_uri=db_uri,
                row_count=len(df),
                columns=columns_list,
            )
            chunks.append(
                ChunkPayload(
                    id=chunk_id,
                    text=text,
                    vector=[],  # placeholder; populated during batch embedding
                    metadata=metadata,
                )
            )

        return chunks

    @staticmethod
    def _classify_backend_error(exc: Exception) -> ErrorCode:
        """Map an exception to the most appropriate ErrorCode.

        Inspects exception type and message to differentiate timeout vs
        connection errors for each backend type.
        """
        msg = str(exc).lower()
        if "timeout" in msg or "timed out" in msg:
            if "embed" in msg:
                return ErrorCode.E_BACKEND_EMBED_TIMEOUT
            if "vector" in msg or "qdrant" in msg or "collection" in msg:
                return ErrorCode.E_BACKEND_VECTOR_TIMEOUT
            return ErrorCode.E_BACKEND_DB_TIMEOUT
        if "connect" in msg or "connection" in msg:
            if "embed" in msg:
                return ErrorCode.E_BACKEND_EMBED_CONNECT
            if "vector" in msg or "qdrant" in msg or "collection" in msg:
                return ErrorCode.E_BACKEND_VECTOR_CONNECT
            return ErrorCode.E_BACKEND_DB_CONNECT
        # Default to schema gen error for unknown exceptions during processing
        return ErrorCode.E_PROCESS_SCHEMA_GEN
