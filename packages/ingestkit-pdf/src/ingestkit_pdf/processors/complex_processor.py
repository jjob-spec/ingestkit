"""Path C processor: complex/hybrid PDF with per-page routing.

Handles Type C PDFs by routing each page through the appropriate extraction
method based on its PageType, then assembling results into a single chunked,
embedded document.

Spec reference: SPEC.md section 11.3.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import fitz  # type: ignore[import-untyped]
import pymupdf4llm  # type: ignore[import-untyped]

from ingestkit_core.models import ChunkPayload, EmbedStageResult, WrittenArtifacts
from ingestkit_pdf.config import PDFProcessorConfig
from ingestkit_pdf.errors import ErrorCode, IngestError
from ingestkit_pdf.models import (
    ClassificationResult,
    ClassificationStageResult,
    DocumentProfile,
    IngestionMethod,
    OCRResult,
    OCRStageResult,
    PageType,
    ParseStageResult,
    PDFChunkMetadata,
    ProcessingResult,
)
from ingestkit_pdf.processors.ocr_processor import _ocr_single_page
from ingestkit_pdf.processors.table_extractor import TableExtractor
from ingestkit_pdf.utils.chunker import PDFChunker
from ingestkit_pdf.utils.header_footer import HeaderFooterDetector
from ingestkit_pdf.utils.heading_detector import HeadingDetector
from ingestkit_pdf.utils.language import detect_language
from ingestkit_pdf.utils.layout_analysis import LayoutAnalyzer, extract_text_blocks

if TYPE_CHECKING:
    from ingestkit_core.protocols import (
        EmbeddingBackend,
        LLMBackend,
        StructuredDBBackend,
        VectorStoreBackend,
    )

logger = logging.getLogger("ingestkit_pdf.processors.complex_processor")


class ComplexProcessor:
    """Path C processor: complex/hybrid PDF with per-page routing."""

    def __init__(
        self,
        vector_store: VectorStoreBackend,
        structured_db: StructuredDBBackend,
        embedder: EmbeddingBackend,
        llm: LLMBackend | None,
        config: PDFProcessorConfig,
    ) -> None:
        self._vector_store = vector_store
        self._structured_db = structured_db
        self._embedder = embedder
        self._llm = llm
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        file_path: str,
        profile: DocumentProfile,
        ingest_key: str,
        ingest_run_id: str,
        parse_result: ParseStageResult,
        classification_result: ClassificationStageResult,
        classification: ClassificationResult,
    ) -> ProcessingResult:
        """Process a complex/hybrid PDF through per-page routing.

        Args:
            file_path: Absolute path to the PDF file on disk.
            profile: Pre-computed structural profile of the document.
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

        errors: list[str] = []
        warnings: list[str] = []
        error_details: list[IngestError] = []
        written = WrittenArtifacts(vector_collection=collection)

        page_texts: dict[int, str] = {}
        scanned_pages: list[int] = []
        table_heavy_pages: list[int] = []
        ocr_results_map: dict[int, OCRResult] = {}

        with fitz.open(file_path) as doc:
            # Step 1: Header/footer detection (once across full doc)
            hf_detector = HeaderFooterDetector(config)
            try:
                header_patterns, footer_patterns = hf_detector.detect(doc)
            except Exception as exc:
                warnings.append(ErrorCode.E_PROCESS_HEADER_FOOTER.value)
                error_details.append(
                    IngestError(
                        code=ErrorCode.E_PROCESS_HEADER_FOOTER,
                        message=str(exc),
                        stage="process",
                        recoverable=True,
                    )
                )
                header_patterns, footer_patterns = [], []

            # Step 2: Heading detection (once across full doc)
            heading_detector = HeadingDetector(config)
            raw_headings = heading_detector.detect(doc)

            # Step 3: Layout analyzer initialization
            layout_analyzer = LayoutAnalyzer(config)

            # Step 4: Page-level routing
            for page_number in sorted(classification.per_page_types.keys()):
                page_type = classification.per_page_types[page_number]

                if page_type == PageType.BLANK:
                    warnings.append(
                        f"{ErrorCode.W_PAGE_SKIPPED_BLANK.value}: page {page_number}"
                    )
                    continue
                if page_type == PageType.TOC:
                    warnings.append(
                        f"{ErrorCode.W_PAGE_SKIPPED_TOC.value}: page {page_number}"
                    )
                    continue
                if page_type == PageType.VECTOR_ONLY:
                    warnings.append(
                        f"{ErrorCode.W_PAGE_SKIPPED_VECTOR_ONLY.value}: page {page_number}"
                    )
                    continue

                try:
                    if page_type == PageType.TEXT:
                        text = self._extract_text_page(doc, page_number)
                    elif page_type == PageType.SCANNED:
                        scanned_pages.append(page_number)
                        continue  # handled in batch below
                    elif page_type == PageType.TABLE_HEAVY:
                        table_heavy_pages.append(page_number)
                        # Also extract any surrounding text on the page
                        try:
                            text = self._extract_text_page(doc, page_number)
                            if text.strip():
                                # Only keep if there is meaningful text alongside tables
                                pass
                            else:
                                continue
                        except Exception:
                            continue
                    elif page_type == PageType.FORM:
                        text = self._extract_form_fields(doc, page_number)
                    elif page_type == PageType.MIXED:
                        text = self._extract_mixed_page(doc, page_number, file_path)
                    else:
                        continue

                    # Multi-column reorder (use layout_analyzer.detect_columns
                    # as the authoritative source rather than profile flag - W2)
                    reordered = self._apply_layout_reorder(
                        doc, page_number, layout_analyzer,
                    )
                    if reordered is not None:
                        text = reordered

                    # Header/footer stripping
                    text = hf_detector.strip(
                        text, page_number, header_patterns, footer_patterns,
                    )

                    if text.strip():
                        page_texts[page_number] = text

                except Exception as exc:
                    warnings.append(
                        f"Page {page_number} extraction failed: {exc}"
                    )
                    error_details.append(
                        IngestError(
                            code=ErrorCode.E_PROCESS_CHUNK,
                            message=f"Page {page_number} extraction failed: {exc}",
                            page_number=page_number,
                            stage="process",
                            recoverable=True,
                        )
                    )

        # Step 5: OCR for SCANNED pages
        for page_num in scanned_pages:
            try:
                result = _ocr_single_page(
                    file_path,
                    page_num,
                    config.ocr_dpi,
                    config.ocr_preprocessing_steps,
                    config.ocr_engine.value,
                    config.ocr_language,
                    config.enable_language_detection,
                    config.default_language,
                )
                if isinstance(result, OCRResult):
                    ocr_results_map[page_num] = result
                    text = result.text
                    # Header/footer stripping on OCR text
                    with fitz.open(file_path) as doc:
                        text = hf_detector.strip(
                            text, page_num, header_patterns, footer_patterns,
                        )
                    if text.strip():
                        page_texts[page_num] = text
                elif isinstance(result, tuple):
                    # Error tuple: (page_number, error_message)
                    errors.append(
                        f"{ErrorCode.E_OCR_FAILED.value}: page {result[0]} - {result[1]}"
                    )
                    error_details.append(
                        IngestError(
                            code=ErrorCode.E_OCR_FAILED,
                            message=f"OCR failed on page {result[0]}: {result[1]}",
                            page_number=result[0],
                            stage="ocr",
                            recoverable=True,
                        )
                    )
            except Exception as exc:
                errors.append(
                    f"{ErrorCode.E_OCR_FAILED.value}: page {page_num} - {exc}"
                )
                error_details.append(
                    IngestError(
                        code=ErrorCode.E_OCR_FAILED,
                        message=str(exc),
                        page_number=page_num,
                        stage="ocr",
                        recoverable=True,
                    )
                )

        # Step 6: Table extraction (batch)
        table_names: list[str] = []
        table_chunks: list[ChunkPayload] = []
        table_texts_embedded = 0
        table_embed_duration = 0.0

        if table_heavy_pages:
            try:
                table_extractor = TableExtractor(
                    config=config,
                    structured_db=self._structured_db,
                    vector_store=self._vector_store,
                    embedder=self._embedder,
                )
                table_result = table_extractor.extract_tables(
                    file_path, table_heavy_pages, ingest_key, ingest_run_id,
                )
                warnings.extend(table_result.warnings)
                # W1: table_result.errors is list[IngestError], route to error_details
                for table_err in table_result.errors:
                    error_details.append(table_err)
                    errors.append(f"{table_err.code.value}: {table_err.message}")
                table_names = table_result.table_names
                table_chunks = table_result.chunks
                table_texts_embedded = table_result.texts_embedded
                table_embed_duration = table_result.embed_duration_seconds
            except Exception as exc:
                error_code = ErrorCode.E_PROCESS_TABLE_EXTRACT
                errors.append(f"{error_code.value}: {exc}")
                error_details.append(
                    IngestError(
                        code=error_code,
                        message=str(exc),
                        stage="process",
                        recoverable=True,
                    )
                )

        # Early return if no usable text, no table chunks, and no table names
        if not page_texts and not table_chunks and not table_names:
            elapsed = time.monotonic() - start_time
            return ProcessingResult(
                file_path=file_path,
                ingest_key=ingest_key,
                ingest_run_id=ingest_run_id,
                tenant_id=config.tenant_id,
                parse_result=parse_result,
                classification_result=classification_result,
                embed_result=None,
                classification=classification,
                ingestion_method=IngestionMethod.COMPLEX_PROCESSING,
                chunks_created=0,
                tables_created=0,
                tables=[],
                written=written,
                errors=errors,
                warnings=warnings,
                error_details=error_details,
                processing_time_seconds=elapsed,
            )

        # Step 7: Assemble full text
        full_text, page_boundary_list, page_offset_map = self._concatenate_pages(
            page_texts,
        )

        # Step 8: Convert headings to character offsets
        headings = self._convert_headings_to_offsets(raw_headings, page_offset_map)

        # Step 9: Language detection
        language = config.default_language
        if config.enable_language_detection and page_texts:
            first_text = next((t for t in page_texts.values() if t.strip()), "")
            if first_text:
                language, _ = detect_language(
                    first_text, default_language=config.default_language,
                )

        # Document metadata from profile
        doc_title = profile.metadata.title
        doc_author = profile.metadata.author
        doc_date = profile.metadata.creation_date

        # Step 10: Chunk
        chunk_dicts: list[dict] = []
        if full_text.strip():
            chunker = PDFChunker(config)
            chunk_dicts = chunker.chunk(full_text, headings, page_boundary_list)

        # Step 11: Embed and upsert text chunks
        embed_result, text_chunk_count = self._embed_and_upsert(
            chunk_dicts=chunk_dicts,
            source_uri=source_uri,
            ingest_key=ingest_key,
            ingest_run_id=ingest_run_id,
            collection=collection,
            doc_title=doc_title,
            doc_author=doc_author,
            doc_date=doc_date,
            language=language,
            ocr_results_map=ocr_results_map,
            written=written,
            errors=errors,
            warnings=warnings,
            error_details=error_details,
        )

        # Combine embed results from text chunks and table extraction
        total_texts_embedded = (
            (embed_result.texts_embedded if embed_result else 0)
            + table_texts_embedded
        )
        total_embed_duration = (
            (embed_result.embed_duration_seconds if embed_result else 0.0)
            + table_embed_duration
        )
        combined_embed_result: EmbedStageResult | None = None
        if total_texts_embedded > 0:
            combined_embed_result = EmbedStageResult(
                texts_embedded=total_texts_embedded,
                embedding_dimension=self._embedder.dimension(),
                embed_duration_seconds=total_embed_duration,
            )

        # Add table chunk IDs to written artifacts
        for chunk in table_chunks:
            written.vector_point_ids.append(chunk.id)

        total_chunks = text_chunk_count + len(table_chunks)

        # Step 12: Build OCRStageResult if any SCANNED pages were processed
        ocr_stage_result: OCRStageResult | None = None
        if ocr_results_map:
            ocr_results = list(ocr_results_map.values())
            avg_confidence = (
                sum(r.confidence for r in ocr_results) / len(ocr_results)
            )
            low_confidence_pages = [
                r.page_number
                for r in ocr_results
                if r.confidence < config.ocr_confidence_threshold
            ]
            ocr_stage_result = OCRStageResult(
                pages_ocrd=len(ocr_results),
                engine_used=config.ocr_engine,
                avg_confidence=avg_confidence,
                low_confidence_pages=low_confidence_pages,
                ocr_duration_seconds=0.0,  # individual page OCR timing not tracked
            )

        elapsed = time.monotonic() - start_time

        return ProcessingResult(
            file_path=file_path,
            ingest_key=ingest_key,
            ingest_run_id=ingest_run_id,
            tenant_id=config.tenant_id,
            parse_result=parse_result,
            classification_result=classification_result,
            ocr_result=ocr_stage_result,
            embed_result=combined_embed_result,
            classification=classification,
            ingestion_method=IngestionMethod.COMPLEX_PROCESSING,
            chunks_created=total_chunks,
            tables_created=len(table_names),
            tables=table_names,
            written=written,
            errors=errors,
            warnings=warnings,
            error_details=error_details,
            processing_time_seconds=elapsed,
        )

    # ------------------------------------------------------------------
    # Private: Page extraction methods
    # ------------------------------------------------------------------

    def _extract_text_page(self, doc: fitz.Document, page_number: int) -> str:
        """Extract text from a single TEXT page using pymupdf4llm.

        Args:
            doc: Open fitz document.
            page_number: 1-based page number.

        Returns:
            Extracted markdown text for the page.
        """
        try:
            text = pymupdf4llm.to_markdown(doc, pages=[page_number - 1])
            return text
        except Exception:
            logger.warning(
                "pymupdf4llm failed on page %d, falling back to get_text()",
                page_number,
            )
            return doc[page_number - 1].get_text()

    def _extract_form_fields(
        self, doc: fitz.Document, page_number: int,
    ) -> str:
        """Extract AcroForm fields from a FORM page as 'Field: Value' pairs.

        Args:
            doc: Open fitz document.
            page_number: 1-based page number.

        Returns:
            Concatenated 'Field Name: Value' lines, one per widget.
            Empty fields are represented as 'Field Name: (empty)'.
        """
        page = doc[page_number - 1]
        lines: list[str] = []

        # Extract any regular text on the page
        regular_text = page.get_text().strip()
        if regular_text:
            lines.append(regular_text)

        # Extract form fields
        field_lines: list[str] = []
        for widget in page.widgets():
            field_type = widget.field_type

            # Skip signature fields (type 3)
            if field_type == 3:
                continue

            field_name = widget.field_name or "(unnamed)"
            field_value = widget.field_value

            # Button fields (checkbox/radio, type 1)
            if field_type == 1:
                if field_value in ("Yes", "On", "/Yes", "/On"):
                    value = "Yes"
                else:
                    value = "No"
            else:
                value = field_value if field_value else "(empty)"

            field_lines.append(f"{field_name}: {value}")

        if field_lines:
            lines.append("[Form Fields]")
            lines.extend(field_lines)

        return "\n".join(lines)

    def _extract_mixed_page(
        self,
        doc: fitz.Document,
        page_number: int,
        file_path: str,
    ) -> str:
        """Extract text + OCR image regions from a MIXED page.

        Args:
            doc: Open fitz document.
            page_number: 1-based page number.
            file_path: Path to PDF for _ocr_single_page.

        Returns:
            Combined native text and OCR text from image regions.
        """
        # Extract native text
        try:
            native_text = pymupdf4llm.to_markdown(doc, pages=[page_number - 1])
        except Exception:
            native_text = doc[page_number - 1].get_text()

        # Check if page has images
        page = doc[page_number - 1]
        images = page.get_images(full=True)

        if not images:
            return native_text

        # Run full-page OCR
        cfg = self._config
        try:
            ocr_result = _ocr_single_page(
                file_path,
                page_number,
                cfg.ocr_dpi,
                cfg.ocr_preprocessing_steps,
                cfg.ocr_engine.value,
                cfg.ocr_language,
                cfg.enable_language_detection,
                cfg.default_language,
            )

            if isinstance(ocr_result, tuple):
                # Error tuple - just return native text
                logger.warning(
                    "OCR failed on MIXED page %d: %s", page_number, ocr_result[1],
                )
                return native_text

            # Compare word counts to decide whether OCR adds value
            native_words = len(native_text.split())
            ocr_words = len(ocr_result.text.split())

            if ocr_words > native_words * 1.2:
                # OCR produced significantly more text - append it
                return f"{native_text}\n\n[OCR from image regions]\n{ocr_result.text}"
            else:
                return native_text

        except Exception as exc:
            logger.warning(
                "OCR failed on MIXED page %d: %s", page_number, exc,
            )
            return native_text

    def _apply_layout_reorder(
        self,
        doc: fitz.Document,
        page_number: int,
        layout_analyzer: LayoutAnalyzer,
    ) -> str | None:
        """Detect multi-column layout and return reordered text, or None if single-column.

        Args:
            doc: Open fitz document.
            page_number: 1-based page number.
            layout_analyzer: Pre-initialized LayoutAnalyzer.

        Returns:
            Reordered text if multi-column detected, None otherwise.
        """
        page = doc[page_number - 1]
        layout = layout_analyzer.detect_columns(page)
        if layout.is_multi_column:
            blocks = extract_text_blocks(page)
            reordered = layout_analyzer.reorder_blocks(blocks, layout)
            return "\n".join(b.text for b in reordered)
        return None

    # ------------------------------------------------------------------
    # Private: Page concatenation
    # ------------------------------------------------------------------

    @staticmethod
    def _concatenate_pages(
        page_texts: dict[int, str],
    ) -> tuple[str, list[int], dict[int, int]]:
        """Concatenate page texts and compute page boundary offsets.

        Returns:
            - full_text: The concatenated document text.
            - page_boundary_list: Character offsets where each page starts.
            - page_offset_map: Mapping of page_number -> char_offset.
        """
        sorted_pages = sorted(page_texts.keys())
        parts: list[str] = []
        page_boundary_list: list[int] = []
        page_offset_map: dict[int, int] = {}
        offset = 0

        for page_num in sorted_pages:
            page_boundary_list.append(offset)
            page_offset_map[page_num] = offset
            text = page_texts[page_num]
            parts.append(text)
            offset += len(text) + 1  # +1 for joining newline

        full_text = "\n".join(parts)
        return full_text, page_boundary_list, page_offset_map

    # ------------------------------------------------------------------
    # Private: Heading offset conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_headings_to_offsets(
        raw_headings: list[tuple[int, str, int]],
        page_offset_map: dict[int, int],
    ) -> list[tuple[int, str, int]]:
        """Map page-number-based headings to character-offset-based headings."""
        result: list[tuple[int, str, int]] = []
        for level, title, page_num in raw_headings:
            char_offset = page_offset_map.get(page_num, 0)
            result.append((level, title, char_offset))
        return result

    # ------------------------------------------------------------------
    # Private: Embed and upsert
    # ------------------------------------------------------------------

    def _embed_and_upsert(
        self,
        chunk_dicts: list[dict],
        source_uri: str,
        ingest_key: str,
        ingest_run_id: str,
        collection: str,
        doc_title: str | None,
        doc_author: str | None,
        doc_date: str | None,
        language: str,
        ocr_results_map: dict[int, OCRResult],
        written: WrittenArtifacts,
        errors: list[str],
        warnings: list[str],
        error_details: list[IngestError],
    ) -> tuple[EmbedStageResult | None, int]:
        """Embed chunks and upsert to vector store.

        Returns:
            Tuple of (embed_result, total_chunks_upserted).
        """
        if not chunk_dicts:
            return None, 0

        config = self._config
        total_chunks = 0
        total_texts_embedded = 0
        embed_duration = 0.0

        # Ensure collection exists
        vector_size = self._embedder.dimension()
        self._vector_store.ensure_collection(collection, vector_size)

        # Build ChunkPayload list
        payloads: list[ChunkPayload] = []
        for cd in chunk_dicts:
            chunk_hash = cd["chunk_hash"]
            chunk_id = str(
                uuid.uuid5(uuid.NAMESPACE_URL, f"{ingest_key}:{chunk_hash}")
            )

            # Check if chunk pages overlap with OCR results
            chunk_page_nums = cd.get("page_numbers", [])
            ocr_page_overlap = [
                pn for pn in chunk_page_nums if pn in ocr_results_map
            ]

            ocr_engine = None
            ocr_confidence = None
            ocr_dpi = None
            ocr_preprocessing = None

            if ocr_page_overlap:
                # Set OCR metadata from overlapping results
                ocr_engine = config.ocr_engine.value
                ocr_dpi = config.ocr_dpi
                ocr_preprocessing = config.ocr_preprocessing_steps
                page_confidences = [
                    ocr_results_map[pn].confidence for pn in ocr_page_overlap
                ]
                ocr_confidence = (
                    sum(page_confidences) / len(page_confidences)
                    if page_confidences
                    else None
                )

            metadata = PDFChunkMetadata(
                source_uri=source_uri,
                source_format="pdf",
                page_numbers=cd["page_numbers"],
                ingestion_method=IngestionMethod.COMPLEX_PROCESSING.value,
                parser_version=config.parser_version,
                chunk_index=cd["chunk_index"],
                chunk_hash=chunk_hash,
                ingest_key=ingest_key,
                ingest_run_id=ingest_run_id,
                tenant_id=config.tenant_id,
                heading_path=cd["heading_path"],
                content_type=cd["content_type"],
                doc_title=doc_title,
                doc_author=doc_author,
                doc_date=doc_date,
                language=language,
                ocr_engine=ocr_engine,
                ocr_confidence=ocr_confidence,
                ocr_dpi=ocr_dpi,
                ocr_preprocessing=ocr_preprocessing,
            )
            payloads.append(
                ChunkPayload(
                    id=chunk_id,
                    text=cd["text"],
                    vector=[],
                    metadata=metadata,
                )
            )

        # Batch embed and upsert
        for batch_start in range(0, len(payloads), config.embedding_batch_size):
            batch = payloads[batch_start : batch_start + config.embedding_batch_size]
            texts = [p.text for p in batch]

            try:
                embed_start = time.monotonic()
                vectors = self._embedder.embed(
                    texts, timeout=config.backend_timeout_seconds,
                )
                embed_duration += time.monotonic() - embed_start
                total_texts_embedded += len(texts)

                for payload, vec in zip(batch, vectors):
                    payload.vector = vec

                self._vector_store.upsert_chunks(collection, batch)
                for payload in batch:
                    written.vector_point_ids.append(payload.id)
                total_chunks += len(batch)

            except Exception as exc:
                error_code = self._classify_backend_error(exc)
                errors.append(error_code.value)
                error_details.append(
                    IngestError(
                        code=error_code,
                        message=str(exc),
                        stage="process",
                        recoverable=False,
                    )
                )
                logger.exception("Embedding/upsert batch failed: %s", exc)
                continue

        embed_result = None
        if total_texts_embedded > 0:
            embed_result = EmbedStageResult(
                texts_embedded=total_texts_embedded,
                embedding_dimension=self._embedder.dimension(),
                embed_duration_seconds=embed_duration,
            )

        return embed_result, total_chunks

    # ------------------------------------------------------------------
    # Error classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_backend_error(exc: Exception) -> ErrorCode:
        """Map a backend exception to the appropriate error code."""
        msg = str(exc).lower()
        if "timeout" in msg or "timed out" in msg:
            if "embed" in msg:
                return ErrorCode.E_BACKEND_EMBED_TIMEOUT
            return ErrorCode.E_BACKEND_VECTOR_TIMEOUT
        if "connect" in msg or "connection" in msg:
            if "embed" in msg:
                return ErrorCode.E_BACKEND_EMBED_CONNECT
            return ErrorCode.E_BACKEND_VECTOR_CONNECT
        return ErrorCode.E_PROCESS_CHUNK
