"""Table extractor: pdfplumber-based table detection, dual routing, and multi-page stitching."""

from __future__ import annotations

import hashlib
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import pandas as pd
import pdfplumber
from pydantic import BaseModel

from ingestkit_core.models import BaseChunkMetadata, ChunkPayload
from ingestkit_pdf.config import PDFProcessorConfig
from ingestkit_pdf.errors import ErrorCode, IngestError
from ingestkit_pdf.models import (
    ContentType,
    IngestionMethod,
    PDFChunkMetadata,
    TableResult,
)
from ingestkit_pdf.protocols import (
    EmbeddingBackend,
    StructuredDBBackend,
    VectorStoreBackend,
)

logger = logging.getLogger("ingestkit_pdf")


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class TableExtractionResult(BaseModel):
    """Result container for table extraction from a PDF."""

    tables: list[TableResult] = []
    chunks: list[ChunkPayload] = []
    table_names: list[str] = []  # DB table names written (large tables only)
    warnings: list[str] = []
    errors: list[IngestError] = []
    texts_embedded: int = 0
    embed_duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


@dataclass
class _RawTable:
    page_number: int
    table_index: int
    df: pd.DataFrame
    headers: list[str]


@dataclass
class _FinalTable:
    page_numbers: list[int]
    table_index: int
    df: pd.DataFrame
    headers: list[str]
    is_continuation: bool = False
    continuation_group_id: str | None = None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def clean_name(raw: str) -> str:
    """Clean a raw name for use as a DB identifier."""
    name = raw.lower()
    name = re.sub(r"[^a-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def deduplicate_names(names: list[str]) -> list[str]:
    """Deduplicate a list of cleaned names by appending _1, _2, etc."""
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
# TableExtractor
# ---------------------------------------------------------------------------


class TableExtractor:
    """Extracts tables from PDF pages using pdfplumber.

    Handles dual routing (NL serialization vs StructuredDB), multi-page
    table stitching, and chunk embedding with table metadata.
    """

    def __init__(
        self,
        config: PDFProcessorConfig,
        structured_db: StructuredDBBackend | None = None,
        vector_store: VectorStoreBackend | None = None,
        embedder: EmbeddingBackend | None = None,
    ) -> None:
        self._config = config
        self._db = structured_db
        self._vector_store = vector_store
        self._embedder = embedder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_tables(
        self,
        file_path: str,
        page_numbers: list[int],
        ingest_key: str,
        ingest_run_id: str,
    ) -> TableExtractionResult:
        """Extract tables from the given PDF pages.

        Args:
            file_path: Path to the PDF file on disk.
            page_numbers: 1-indexed page numbers to process.
            ingest_key: Deterministic ingest key for dedup/ID generation.
            ingest_run_id: Unique ID for this processing run.

        Returns:
            A ``TableExtractionResult`` with tables, chunks, and diagnostics.
        """
        config = self._config
        source_uri = f"file://{file_path}"
        raw_tables: list[_RawTable] = []
        errors: list[IngestError] = []
        warnings: list[str] = []

        # Step 1: Extract raw tables from each page via pdfplumber
        with pdfplumber.open(file_path) as pdf:
            for page_num in page_numbers:
                try:
                    page = pdf.pages[page_num - 1]  # pdfplumber is 0-indexed
                    page_tables = page.extract_tables()
                    if not page_tables:
                        continue

                    for t_idx, table_data in enumerate(page_tables):
                        if not table_data or len(table_data) < 2:
                            # Header-only or empty table: skip
                            continue

                        # First row is headers
                        raw_headers = table_data[0]
                        # Replace None/empty headers with column_N
                        headers = [
                            str(h).strip() if h else f"column_{i}"
                            for i, h in enumerate(raw_headers)
                        ]
                        cleaned_headers = deduplicate_names(
                            [clean_name(h) for h in headers]
                        )

                        # Build DataFrame from remaining rows
                        df = pd.DataFrame(table_data[1:], columns=cleaned_headers)
                        # Drop all-NaN rows
                        df = df.dropna(how="all").reset_index(drop=True)
                        if df.empty:
                            continue

                        raw_tables.append(
                            _RawTable(
                                page_number=page_num,
                                table_index=t_idx,
                                df=df,
                                headers=cleaned_headers,
                            )
                        )
                except Exception as exc:
                    error_code = ErrorCode.E_PROCESS_TABLE_EXTRACT
                    errors.append(
                        IngestError(
                            code=error_code,
                            message=f"Table extraction failed on page {page_num}: {exc}",
                            page_number=page_num,
                            stage="process",
                            recoverable=False,
                        )
                    )
                    logger.exception(
                        "Table extraction error on page %d: %s", page_num, exc
                    )
                    continue

        # Step 2: Multi-page table stitching
        final_tables, stitch_warnings = self._stitch_tables(raw_tables)
        warnings.extend(stitch_warnings)

        # Step 3: Route each final table and collect results
        all_chunks: list[ChunkPayload] = []
        all_table_names: list[str] = []
        table_results: list[TableResult] = []
        chunk_index = 0
        total_texts_embedded = 0
        total_embed_duration = 0.0

        for ftable in final_tables:
            table_results.append(self._build_table_result(ftable))

            chunks, db_names, next_idx, n_embedded, embed_dur = self._route_table(
                table=ftable,
                ingest_key=ingest_key,
                ingest_run_id=ingest_run_id,
                source_uri=source_uri,
                chunk_index_start=chunk_index,
            )
            all_chunks.extend(chunks)
            all_table_names.extend(db_names)
            chunk_index = next_idx
            total_texts_embedded += n_embedded
            total_embed_duration += embed_dur

        return TableExtractionResult(
            tables=table_results,
            chunks=all_chunks,
            table_names=all_table_names,
            warnings=warnings,
            errors=errors,
            texts_embedded=total_texts_embedded,
            embed_duration_seconds=total_embed_duration,
        )

    # ------------------------------------------------------------------
    # Multi-page table stitching
    # ------------------------------------------------------------------

    def _stitch_tables(
        self,
        raw_tables: list[_RawTable],
    ) -> tuple[list[_FinalTable], list[str]]:
        """Stitch tables that span consecutive pages.

        Returns:
            (final_tables, warnings)
        """
        if not raw_tables:
            return [], []

        threshold = self._config.table_continuation_column_match_threshold
        warnings: list[str] = []

        # Group raw tables by page number, preserving order
        pages_in_order: list[int] = []
        tables_by_page: dict[int, list[_RawTable]] = {}
        for rt in raw_tables:
            if rt.page_number not in tables_by_page:
                pages_in_order.append(rt.page_number)
                tables_by_page[rt.page_number] = []
            tables_by_page[rt.page_number].append(rt)

        # Build final tables, checking for continuations across consecutive pages
        finals: list[_FinalTable] = []

        # pending holds a _FinalTable being accumulated across pages
        pending: _FinalTable | None = None

        for page_idx, page_num in enumerate(pages_in_order):
            page_tables = tables_by_page[page_num]

            for t_pos, rt in enumerate(page_tables):
                is_first_on_page = t_pos == 0

                # Check if this is a continuation of pending
                if (
                    pending is not None
                    and is_first_on_page
                    and page_idx > 0
                    and pages_in_order[page_idx - 1] == pending.page_numbers[-1]
                ):
                    # Check column count match
                    if pending.df.shape[1] == rt.df.shape[1]:
                        # Check header similarity
                        similarity = SequenceMatcher(
                            None, pending.headers, rt.headers
                        ).ratio()
                        if similarity >= threshold:
                            # It's a continuation -- merge
                            continuation_df = rt.df.copy()

                            # Skip repeated header row if present
                            if len(continuation_df) > 0:
                                first_row = (
                                    continuation_df.iloc[0].astype(str).tolist()
                                )
                                base_headers_lower = [
                                    h.lower().strip() for h in pending.headers
                                ]
                                first_row_lower = [
                                    v.lower().strip() for v in first_row
                                ]
                                if first_row_lower == base_headers_lower:
                                    continuation_df = continuation_df.iloc[1:]

                            # Concatenate
                            pending.df = pd.concat(
                                [pending.df, continuation_df],
                                ignore_index=True,
                            )
                            pending.page_numbers.append(page_num)
                            if pending.continuation_group_id is None:
                                pending.continuation_group_id = str(uuid.uuid4())
                            pending.is_continuation = True

                            warnings.append(ErrorCode.W_TABLE_CONTINUATION.value)
                            logger.info(
                                "Table continuation detected: pages %s",
                                pending.page_numbers,
                            )
                            continue

                # No continuation match -- finalize pending if any
                if pending is not None:
                    finals.append(pending)
                    pending = None

                # Start a new pending from current raw table
                pending = _FinalTable(
                    page_numbers=[rt.page_number],
                    table_index=rt.table_index,
                    df=rt.df.copy(),
                    headers=list(rt.headers),
                )

            # If there are multiple tables on a page, only the last one
            # can be continued onto the next page. Finalize non-last tables.
            # Actually, the loop above already handles this: only the first
            # table on a page is checked for continuation with pending.
            # Non-first tables on a page break the chain and start fresh.

        # Finalize any remaining pending
        if pending is not None:
            finals.append(pending)

        return finals, warnings

    # ------------------------------------------------------------------
    # Dual routing
    # ------------------------------------------------------------------

    def _route_table(
        self,
        table: _FinalTable,
        ingest_key: str,
        ingest_run_id: str,
        source_uri: str,
        chunk_index_start: int,
    ) -> tuple[list[ChunkPayload], list[str], int, int, float]:
        """Route a table to NL serialization or StructuredDB.

        Returns:
            (chunks, db_table_names, next_chunk_index, texts_embedded, embed_duration)
        """
        config = self._config
        row_count = len(table.df)

        if row_count <= config.table_max_rows_for_serialization:
            return self._nl_serialize_table(
                table, ingest_key, ingest_run_id, source_uri, chunk_index_start
            )
        else:
            return self._db_write_table(
                table, ingest_key, ingest_run_id, source_uri, chunk_index_start
            )

    def _nl_serialize_table(
        self,
        table: _FinalTable,
        ingest_key: str,
        ingest_run_id: str,
        source_uri: str,
        chunk_index_start: int,
    ) -> tuple[list[ChunkPayload], list[str], int, int, float]:
        """Serialize small table rows as NL sentences and embed."""
        config = self._config
        table_name = self._make_table_name(table, ingest_key)
        chunks: list[ChunkPayload] = []
        embed_duration = 0.0
        texts_embedded = 0

        # Build row texts
        row_texts: list[str] = []
        for row_idx, row in table.df.iterrows():
            parts = []
            for col in table.df.columns:
                val = row[col]
                if pd.isna(val) or val is None or str(val).strip() == "":
                    parts.append(f"{col} is N/A")
                else:
                    parts.append(f"{col} is {val}")
            row_number = row_idx + 1 if isinstance(row_idx, int) else row_idx
            text = f"In table '{table_name}', row {row_number}: {', '.join(parts)}."
            row_texts.append(text)

        # Build chunk payloads (vectors as placeholders)
        for i, text in enumerate(row_texts):
            chunk_hash = hashlib.sha256(text.encode()).hexdigest()
            chunk_id = str(
                uuid.uuid5(uuid.NAMESPACE_URL, f"{ingest_key}:{chunk_hash}")
            )
            metadata = PDFChunkMetadata(
                source_uri=source_uri,
                source_format="pdf",
                page_numbers=list(table.page_numbers),
                ingestion_method=IngestionMethod.COMPLEX_PROCESSING.value,
                parser_version=config.parser_version,
                chunk_index=chunk_index_start + i,
                chunk_hash=chunk_hash,
                ingest_key=ingest_key,
                ingest_run_id=ingest_run_id,
                tenant_id=config.tenant_id,
                content_type=ContentType.TABLE.value,
                table_index=table.table_index,
                table_name=table_name,
                row_count=len(table.df),
                columns=list(table.df.columns),
            )
            chunks.append(
                ChunkPayload(
                    id=chunk_id,
                    text=text,
                    vector=[],
                    metadata=metadata,
                )
            )

        # Embed and upsert if backends available
        if self._embedder is not None and self._vector_store is not None and chunks:
            collection = config.default_collection
            self._vector_store.ensure_collection(
                collection, self._embedder.dimension()
            )
            for batch_start in range(0, len(chunks), config.embedding_batch_size):
                batch = chunks[batch_start : batch_start + config.embedding_batch_size]
                texts = [c.text for c in batch]
                embed_start = time.monotonic()
                vectors = self._embedder.embed(
                    texts, timeout=config.backend_timeout_seconds
                )
                embed_duration += time.monotonic() - embed_start
                texts_embedded += len(texts)
                for c, vec in zip(batch, vectors):
                    c.vector = vec
                self._vector_store.upsert_chunks(collection, list(batch))

        next_index = chunk_index_start + len(chunks)
        return chunks, [], next_index, texts_embedded, embed_duration

    def _db_write_table(
        self,
        table: _FinalTable,
        ingest_key: str,
        ingest_run_id: str,
        source_uri: str,
        chunk_index_start: int,
    ) -> tuple[list[ChunkPayload], list[str], int, int, float]:
        """Write large table to StructuredDB and embed schema description."""
        config = self._config
        table_name = self._make_table_name(table, ingest_key)
        db_names: list[str] = []
        chunks: list[ChunkPayload] = []
        embed_duration = 0.0
        texts_embedded = 0

        # Write to DB if available
        if self._db is not None:
            self._db.create_table_from_dataframe(table_name, table.df)
            db_names.append(table_name)

        # Generate schema description
        schema_text = self._generate_schema_description(table_name, table.df)

        # Build schema chunk
        chunk_hash = hashlib.sha256(schema_text.encode()).hexdigest()
        chunk_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"{ingest_key}:{chunk_hash}")
        )
        metadata = PDFChunkMetadata(
            source_uri=source_uri,
            source_format="pdf",
            page_numbers=list(table.page_numbers),
            ingestion_method=IngestionMethod.COMPLEX_PROCESSING.value,
            parser_version=config.parser_version,
            chunk_index=chunk_index_start,
            chunk_hash=chunk_hash,
            ingest_key=ingest_key,
            ingest_run_id=ingest_run_id,
            tenant_id=config.tenant_id,
            content_type=ContentType.TABLE.value,
            table_index=table.table_index,
            table_name=table_name,
            row_count=len(table.df),
            columns=list(table.df.columns),
        )
        schema_chunk = ChunkPayload(
            id=chunk_id,
            text=schema_text,
            vector=[],
            metadata=metadata,
        )
        chunks.append(schema_chunk)

        # Embed and upsert if backends available
        if self._embedder is not None and self._vector_store is not None:
            collection = config.default_collection
            self._vector_store.ensure_collection(
                collection, self._embedder.dimension()
            )
            embed_start = time.monotonic()
            vectors = self._embedder.embed(
                [schema_text], timeout=config.backend_timeout_seconds
            )
            embed_duration = time.monotonic() - embed_start
            texts_embedded = 1
            schema_chunk.vector = vectors[0]
            self._vector_store.upsert_chunks(collection, [schema_chunk])

        next_index = chunk_index_start + 1
        return chunks, db_names, next_index, texts_embedded, embed_duration

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_table_name(self, table: _FinalTable, ingest_key: str) -> str:
        """Generate a unique DB table name for a table."""
        prefix = ingest_key[:8] if len(ingest_key) >= 8 else ingest_key
        page = table.page_numbers[0]
        return f"pdf_{prefix}_p{page}_t{table.table_index}"

    def _build_table_result(self, table: _FinalTable) -> TableResult:
        """Build a TableResult model from a _FinalTable."""
        return TableResult(
            page_number=table.page_numbers[0],
            table_index=table.table_index,
            row_count=len(table.df),
            col_count=table.df.shape[1],
            headers=table.headers,
            is_continuation=table.is_continuation,
            continuation_group_id=table.continuation_group_id,
        )

    def _generate_schema_description(
        self, table_name: str, df: pd.DataFrame
    ) -> str:
        """Generate a natural language schema description for embedding."""
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
        n_unique = series.nunique()
        if n_unique < low_card_threshold:
            unique_vals = ", ".join(
                str(v) for v in series.unique()[:low_card_threshold]
            )
            return f"one of {unique_vals}"
        return f"{n_unique} unique values"

    @staticmethod
    def _classify_backend_error(exc: Exception) -> ErrorCode:
        """Map an exception to the most appropriate ErrorCode."""
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
        return ErrorCode.E_PROCESS_TABLE_EXTRACT
