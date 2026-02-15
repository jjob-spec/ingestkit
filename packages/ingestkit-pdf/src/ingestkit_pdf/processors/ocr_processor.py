"""Path B OCR processor: scanned-PDF ingestion via parallel OCR pipeline.

Orchestrates the 9 SPEC 11.2 steps for scanned documents: page rendering,
preprocessing, language detection, parallel OCR, postprocessing, optional LLM
cleanup, low-confidence flagging, and downstream chunking/embedding/upsert.

The module-level ``_ocr_single_page()`` function handles ProcessPoolExecutor
pickling constraints by accepting only serializable primitives.

Note on ``process()`` signature: The SPEC (11.2) defines 5 parameters but
``ProcessingResult`` requires ``parse_result``, ``classification_result``,
and ``classification`` as non-optional fields.  The expanded 8-parameter
signature matches the Excel ``StructuredDBProcessor`` pattern -- the router
(caller) provides these upstream stage results.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

import fitz  # type: ignore[import-untyped]

from ingestkit_core.models import ChunkPayload, EmbedStageResult, WrittenArtifacts
from ingestkit_pdf.config import PDFProcessorConfig
from ingestkit_pdf.errors import ErrorCode, IngestError
from ingestkit_pdf.models import (
    ClassificationResult,
    ClassificationStageResult,
    DocumentProfile,
    IngestionMethod,
    OCREngine,
    OCRResult,
    OCRStageResult,
    PageType,
    ParseStageResult,
    PDFChunkMetadata,
    ProcessingResult,
)
from ingestkit_pdf.utils.chunker import PDFChunker
from ingestkit_pdf.utils.header_footer import HeaderFooterDetector
from ingestkit_pdf.utils.heading_detector import HeadingDetector
from ingestkit_pdf.utils.language import detect_language, map_language_to_ocr
from ingestkit_pdf.utils.ocr_engines import create_ocr_engine
from ingestkit_pdf.utils.ocr_postprocess import postprocess_ocr_text
from ingestkit_pdf.utils.page_renderer import PageRenderer

logger = logging.getLogger("ingestkit_pdf.processors.ocr_processor")


# ---------------------------------------------------------------------------
# Module-level worker function (must be picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def _ocr_single_page(
    file_path: str,
    page_number: int,
    ocr_dpi: int,
    preprocessing_steps: list[str],
    ocr_engine_name: str,
    ocr_language: str,
    enable_language_detection: bool,
    default_language: str,
) -> OCRResult | tuple[int, str]:
    """Process a single PDF page through OCR.

    Returns an ``OCRResult`` on success or ``(page_number, error_message)``
    on failure.  This function is module-level (not a method) so that
    ``ProcessPoolExecutor`` can pickle it.
    """
    try:
        # Build minimal config from primitives (avoids pickling PDFProcessorConfig)
        config = PDFProcessorConfig(
            ocr_dpi=ocr_dpi,
            ocr_preprocessing_steps=preprocessing_steps,
            ocr_engine=OCREngine(ocr_engine_name),
            ocr_language=ocr_language,
        )

        # Step 1: Open PDF and get page
        doc = fitz.open(file_path)
        try:
            page = doc[page_number - 1]  # 0-indexed

            # Step 2: Render page to image
            renderer = PageRenderer(config)
            image = renderer.render_page(page)

            # Step 3: Preprocess image
            image = renderer.preprocess(image)
        finally:
            doc.close()

        # Step 4: Language detection
        detected_lang: str | None = None
        ocr_lang = ocr_language
        if enable_language_detection:
            # Create engine for initial sample OCR
            engine, _ = create_ocr_engine(config)
            sample_result = engine.recognize(image, ocr_language)
            if sample_result.text.strip():
                lang, _conf = detect_language(
                    sample_result.text, default_language=default_language
                )
                detected_lang = lang
                ocr_lang = map_language_to_ocr(lang, config.ocr_engine)

        # Step 5: Run OCR with detected/configured language
        engine, _ = create_ocr_engine(config)
        ocr_page_result = engine.recognize(image, ocr_lang)

        # Step 6: Postprocess OCR text
        cleaned_text = postprocess_ocr_text(ocr_page_result.text)

        return OCRResult(
            page_number=page_number,
            text=cleaned_text,
            confidence=ocr_page_result.confidence,
            engine_used=OCREngine(ocr_engine_name),
            dpi=ocr_dpi,
            preprocessing_steps=preprocessing_steps,
            language_detected=detected_lang,
        )
    except Exception as exc:
        return (page_number, str(exc))


# ---------------------------------------------------------------------------
# OCRProcessor
# ---------------------------------------------------------------------------


class OCRProcessor:
    """Path B processor: scanned-PDF OCR pipeline.

    Parameters
    ----------
    vector_store:
        Vector store backend for chunk upsert.
    embedder:
        Embedding backend for text vectorization.
    llm:
        Optional LLM backend for OCR text cleanup.
    config:
        Pipeline configuration.
    """

    def __init__(
        self,
        vector_store: Any,
        embedder: Any,
        llm: Any | None,
        config: PDFProcessorConfig,
    ) -> None:
        self._vector_store = vector_store
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
        pages: list[int] | None,
        ingest_key: str,
        ingest_run_id: str,
        parse_result: ParseStageResult,
        classification_result: ClassificationStageResult,
        classification: ClassificationResult,
    ) -> ProcessingResult:
        """Run the full OCR pipeline on a scanned PDF.

        Parameters
        ----------
        file_path:
            Path to the PDF file on disk.
        profile:
            Document structural profile from the parser.
        pages:
            Explicit page numbers to process (1-based), or None for auto-selection.
        ingest_key:
            Deterministic dedup key from upstream.
        ingest_run_id:
            Unique run identifier.
        parse_result:
            Upstream parse stage result (passed through to ProcessingResult).
        classification_result:
            Upstream classification stage result.
        classification:
            Upstream classification result.

        Returns
        -------
        ProcessingResult
            Fully assembled result with all stage artifacts.
        """
        start_time = time.monotonic()

        errors: list[str] = []
        warnings: list[str] = []
        error_details: list[IngestError] = []
        written = WrittenArtifacts(vector_collection=self._config.default_collection)
        source_uri = f"file://{Path(file_path).resolve().as_posix()}"

        # Step 0.5: Select pages to process
        page_numbers, page_warnings = self._select_pages(profile, pages)
        warnings.extend(page_warnings)

        # Steps 1-6: OCR pages
        ocr_start = time.monotonic()
        ocr_results, ocr_errors, ocr_warnings = self._ocr_pages(page_numbers, file_path)
        ocr_elapsed = time.monotonic() - ocr_start
        errors.extend(ocr_errors)
        warnings.extend(ocr_warnings)
        for err_msg, page_num, code in self._extract_error_details(ocr_errors):
            error_details.append(
                IngestError(
                    code=code,
                    message=err_msg,
                    page_number=page_num,
                    stage="ocr",
                    recoverable=True,
                )
            )

        # Step 7: LLM cleanup
        ocr_results = self._llm_cleanup(ocr_results, warnings)

        # Step 8: Low-confidence flagging
        low_confidence_pages = self._flag_low_confidence(ocr_results, warnings)

        # Step 9: Downstream pipeline (chunk, embed, upsert)
        chunks_created = 0
        embed_result: EmbedStageResult | None = None

        if ocr_results:
            try:
                chunks_created, embed_result = self._chunk_and_embed(
                    ocr_results,
                    file_path,
                    profile,
                    source_uri,
                    ingest_key,
                    ingest_run_id,
                    written,
                    errors,
                    warnings,
                    error_details,
                )
            except Exception as exc:
                code = self._classify_backend_error(exc)
                errors.append(f"{code.value}: {exc}")
                error_details.append(
                    IngestError(
                        code=code,
                        message=str(exc),
                        stage="chunk_and_embed",
                        recoverable=True,
                    )
                )

        # Build OCRStageResult
        avg_confidence = 0.0
        if ocr_results:
            avg_confidence = sum(r.confidence for r in ocr_results) / len(ocr_results)

        ocr_stage_result = OCRStageResult(
            pages_ocrd=len(ocr_results),
            engine_used=self._config.ocr_engine,
            avg_confidence=avg_confidence,
            low_confidence_pages=low_confidence_pages,
            ocr_duration_seconds=ocr_elapsed,
            engine_fallback_used=any(
                ErrorCode.W_OCR_ENGINE_FALLBACK.value in w for w in warnings
            ),
        )

        elapsed = time.monotonic() - start_time

        return ProcessingResult(
            file_path=file_path,
            ingest_key=ingest_key,
            ingest_run_id=ingest_run_id,
            tenant_id=self._config.tenant_id,
            parse_result=parse_result,
            classification_result=classification_result,
            ocr_result=ocr_stage_result,
            embed_result=embed_result,
            classification=classification,
            ingestion_method=IngestionMethod.OCR_PIPELINE,
            chunks_created=chunks_created,
            tables_created=0,
            tables=[],
            written=written,
            errors=errors,
            warnings=warnings,
            error_details=error_details,
            processing_time_seconds=elapsed,
        )

    # ------------------------------------------------------------------
    # Page selection
    # ------------------------------------------------------------------

    def _select_pages(
        self, profile: DocumentProfile, pages: list[int] | None
    ) -> tuple[list[int], list[str]]:
        """Select which pages to OCR based on profile and explicit list.

        Returns (page_numbers, warnings).
        """
        if pages is not None:
            return (pages, [])

        selected: list[int] = []
        warnings: list[str] = []

        skip_map = {
            PageType.BLANK: ErrorCode.W_PAGE_SKIPPED_BLANK,
            PageType.TOC: ErrorCode.W_PAGE_SKIPPED_TOC,
            PageType.VECTOR_ONLY: ErrorCode.W_PAGE_SKIPPED_VECTOR_ONLY,
        }

        for page_profile in profile.pages:
            ptype = page_profile.page_type
            if ptype in skip_map:
                code = skip_map[ptype]
                warnings.append(
                    f"{code.value}: page {page_profile.page_number} skipped ({ptype.value})"
                )
            else:
                selected.append(page_profile.page_number)

        return (selected, warnings)

    # ------------------------------------------------------------------
    # OCR orchestration
    # ------------------------------------------------------------------

    def _ocr_pages(
        self, page_numbers: list[int], file_path: str
    ) -> tuple[list[OCRResult], list[str], list[str]]:
        """OCR the given pages, returning (results, errors, warnings)."""
        if not page_numbers:
            return ([], [], [])

        use_parallel = (
            self._config.ocr_max_workers > 1 and len(page_numbers) > 1
        )

        if use_parallel:
            return self._ocr_pages_parallel(page_numbers, file_path)
        return self._ocr_pages_sequential(page_numbers, file_path)

    def _ocr_pages_parallel(
        self, page_numbers: list[int], file_path: str
    ) -> tuple[list[OCRResult], list[str], list[str]]:
        """OCR pages in parallel via ProcessPoolExecutor."""
        results: list[OCRResult] = []
        errors: list[str] = []
        warnings: list[str] = []

        cfg = self._config
        with ProcessPoolExecutor(max_workers=cfg.ocr_max_workers) as executor:
            future_to_page = {}
            for page_num in page_numbers:
                future = executor.submit(
                    _ocr_single_page,
                    file_path,
                    page_num,
                    cfg.ocr_dpi,
                    cfg.ocr_preprocessing_steps,
                    cfg.ocr_engine.value,
                    cfg.ocr_language,
                    cfg.enable_language_detection,
                    cfg.default_language,
                )
                future_to_page[future] = page_num

            # PLAN-CHECK CORRECTION 1: Use future.result(timeout=...) per future,
            # NOT as_completed(timeout=T) for the whole iteration.
            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    result = future.result(
                        timeout=cfg.ocr_per_page_timeout_seconds
                    )
                    if isinstance(result, OCRResult):
                        results.append(result)
                    elif isinstance(result, tuple):
                        # Error tuple: (page_number, error_string)
                        errors.append(
                            f"{ErrorCode.E_OCR_FAILED.value}: page {result[0]} - {result[1]}"
                        )
                except TimeoutError:
                    errors.append(
                        f"{ErrorCode.E_OCR_TIMEOUT.value}: page {page_num} timed out"
                    )
                except Exception as exc:
                    errors.append(
                        f"{ErrorCode.E_OCR_FAILED.value}: page {page_num} - {exc}"
                    )

        return (results, errors, warnings)

    def _ocr_pages_sequential(
        self, page_numbers: list[int], file_path: str
    ) -> tuple[list[OCRResult], list[str], list[str]]:
        """OCR pages sequentially (used when max_workers <= 1 or single page)."""
        results: list[OCRResult] = []
        errors: list[str] = []
        warnings: list[str] = []

        cfg = self._config
        for page_num in page_numbers:
            try:
                result = _ocr_single_page(
                    file_path,
                    page_num,
                    cfg.ocr_dpi,
                    cfg.ocr_preprocessing_steps,
                    cfg.ocr_engine.value,
                    cfg.ocr_language,
                    cfg.enable_language_detection,
                    cfg.default_language,
                )
                if isinstance(result, OCRResult):
                    results.append(result)
                elif isinstance(result, tuple):
                    errors.append(
                        f"{ErrorCode.E_OCR_FAILED.value}: page {result[0]} - {result[1]}"
                    )
            except Exception as exc:
                errors.append(
                    f"{ErrorCode.E_OCR_FAILED.value}: page {page_num} - {exc}"
                )

        return (results, errors, warnings)

    # ------------------------------------------------------------------
    # LLM cleanup (Step 7)
    # ------------------------------------------------------------------

    def _llm_cleanup(
        self, ocr_results: list[OCRResult], warnings: list[str]
    ) -> list[OCRResult]:
        """Optionally clean OCR text via LLM.  Returns (possibly modified) results."""
        if not self._config.enable_ocr_cleanup:
            return ocr_results

        if self._llm is None:
            logger.warning("LLM cleanup enabled but no LLM backend provided; skipping")
            warnings.append("LLM cleanup enabled but no LLM backend provided")
            return ocr_results

        cleaned: list[OCRResult] = []
        for result in ocr_results:
            if not result.text.strip():
                cleaned.append(result)
                continue
            try:
                prompt = (
                    "Fix obvious OCR errors in the following text while preserving "
                    "its meaning and structure. Do not add, remove, or rephrase "
                    "content. Return only the corrected text.\n\n"
                    f"{result.text}"
                )
                response = self._llm.generate(
                    prompt=prompt,
                    model=self._config.ocr_cleanup_model,
                    temperature=0.1,
                    timeout=self._config.backend_timeout_seconds,
                )
                if response and response.strip():
                    result = result.model_copy(update={"text": response.strip()})
            except Exception as exc:
                logger.warning(
                    "LLM cleanup failed for page %d: %s", result.page_number, exc
                )
                warnings.append(f"LLM cleanup failed for page {result.page_number}: {exc}")

            cleaned.append(result)
        return cleaned

    # ------------------------------------------------------------------
    # Low-confidence flagging (Step 8)
    # ------------------------------------------------------------------

    def _flag_low_confidence(
        self, ocr_results: list[OCRResult], warnings: list[str]
    ) -> list[int]:
        """Flag pages below the confidence threshold.  Returns list of page numbers."""
        low_pages: list[int] = []
        for result in ocr_results:
            if result.confidence < self._config.ocr_confidence_threshold:
                low_pages.append(result.page_number)
                warnings.append(
                    f"{ErrorCode.W_PAGE_LOW_OCR_CONFIDENCE.value}: page {result.page_number} "
                    f"confidence {result.confidence:.2f} < {self._config.ocr_confidence_threshold}"
                )
        return low_pages

    # ------------------------------------------------------------------
    # Downstream pipeline (Step 9)
    # ------------------------------------------------------------------

    def _chunk_and_embed(
        self,
        ocr_results: list[OCRResult],
        file_path: str,
        profile: DocumentProfile,
        source_uri: str,
        ingest_key: str,
        ingest_run_id: str,
        written: WrittenArtifacts,
        errors: list[str],
        warnings: list[str],
        error_details: list[IngestError],
    ) -> tuple[int, EmbedStageResult | None]:
        """Heading detection -> chunking -> embedding -> upsert."""
        cfg = self._config

        # 9a: Header/footer detection and stripping
        doc = fitz.open(file_path)
        try:
            hf_detector = HeaderFooterDetector(cfg)
            headers, footers = hf_detector.detect(doc)

            stripped_results: list[OCRResult] = []
            for result in ocr_results:
                stripped_text = hf_detector.strip(
                    result.text, result.page_number, headers, footers
                )
                stripped_results.append(result.model_copy(update={"text": stripped_text}))

            # 9b: Text assembly with page boundaries
            page_texts: list[str] = []
            page_boundaries: list[int] = []
            offset = 0
            for result in stripped_results:
                page_boundaries.append(offset)
                page_texts.append(result.text)
                offset += len(result.text) + 2  # +2 for "\n\n" separator

            full_text = "\n\n".join(page_texts)

            if not full_text.strip():
                return (0, None)

            # 9c: Heading detection
            heading_detector = HeadingDetector(cfg)
            raw_headings = heading_detector.detect(doc)
        finally:
            doc.close()

        # Convert page-based headings to character-offset headings
        headings_with_offsets: list[tuple[int, str, int]] = []
        for level, title, page_num in raw_headings:
            # Map page_number to character offset
            for i, result in enumerate(stripped_results):
                if result.page_number == page_num:
                    char_offset = page_boundaries[i] if i < len(page_boundaries) else 0
                    headings_with_offsets.append((level, title, char_offset))
                    break

        # 9d: Chunking
        chunker = PDFChunker(cfg)
        chunk_dicts = chunker.chunk(full_text, headings_with_offsets, page_boundaries)

        if not chunk_dicts:
            return (0, None)

        # 9e: Build ChunkPayloads
        # Build a page_number -> OCRResult lookup for confidence averaging
        page_result_map = {r.page_number: r for r in ocr_results}

        payloads: list[ChunkPayload] = []
        for chunk in chunk_dicts:
            chunk_id = str(
                uuid.uuid5(uuid.NAMESPACE_URL, f"{ingest_key}:{chunk['chunk_hash']}")
            )

            # Compute average OCR confidence across the chunk's pages
            chunk_page_nums = chunk.get("page_numbers", [])
            page_confidences = [
                page_result_map[pn].confidence
                for pn in chunk_page_nums
                if pn in page_result_map
            ]
            avg_conf = (
                sum(page_confidences) / len(page_confidences)
                if page_confidences
                else None
            )

            # Determine language from OCR results for these pages
            chunk_langs = [
                page_result_map[pn].language_detected
                for pn in chunk_page_nums
                if pn in page_result_map and page_result_map[pn].language_detected
            ]
            chunk_lang = chunk_langs[0] if chunk_langs else None

            metadata = PDFChunkMetadata(
                source_uri=source_uri,
                source_format="pdf",
                ingestion_method=IngestionMethod.OCR_PIPELINE.value,
                parser_version=cfg.parser_version,
                chunk_index=chunk["chunk_index"],
                chunk_hash=chunk["chunk_hash"],
                ingest_key=ingest_key,
                ingest_run_id=ingest_run_id,
                tenant_id=cfg.tenant_id,
                page_numbers=chunk["page_numbers"],
                heading_path=chunk.get("heading_path") or None,
                content_type=chunk.get("content_type"),
                doc_title=profile.metadata.title,
                doc_author=profile.metadata.author,
                doc_date=profile.metadata.creation_date,
                ocr_engine=cfg.ocr_engine.value,
                ocr_confidence=avg_conf,
                ocr_dpi=cfg.ocr_dpi,
                ocr_preprocessing=cfg.ocr_preprocessing_steps,
                language=chunk_lang,
            )

            payloads.append(
                ChunkPayload(
                    id=chunk_id,
                    text=chunk["text"],
                    vector=[],
                    metadata=metadata,
                )
            )

        # 9f: Batch embed
        collection = cfg.default_collection
        self._vector_store.ensure_collection(collection, self._embedder.dimension())

        embed_start = time.monotonic()
        total_embedded = 0
        batch_size = cfg.embedding_batch_size

        for i in range(0, len(payloads), batch_size):
            batch = payloads[i : i + batch_size]
            texts = [p.text for p in batch]
            try:
                vectors = self._embedder.embed(
                    texts, timeout=cfg.backend_timeout_seconds
                )
                for j, payload in enumerate(batch):
                    payload.vector = vectors[j]
                total_embedded += len(batch)

                # 9g: Upsert batch
                self._vector_store.upsert_chunks(collection, batch)
                written.vector_point_ids.extend(p.id for p in batch)
            except Exception as exc:
                code = self._classify_backend_error(exc)
                errors.append(f"{code.value}: batch {i // batch_size} - {exc}")
                error_details.append(
                    IngestError(
                        code=code,
                        message=str(exc),
                        stage="embed_upsert",
                        recoverable=True,
                    )
                )

        embed_elapsed = time.monotonic() - embed_start
        embed_result: EmbedStageResult | None = None
        if total_embedded > 0:
            embed_result = EmbedStageResult(
                texts_embedded=total_embedded,
                embedding_dimension=self._embedder.dimension(),
                embed_duration_seconds=embed_elapsed,
            )

        if cfg.log_ocr_output:
            for r in ocr_results:
                logger.debug("OCR page %d text: %s", r.page_number, r.text[:200])

        return (len(payloads), embed_result)

    # ------------------------------------------------------------------
    # Error classification helper
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
        return ErrorCode.E_OCR_FAILED

    @staticmethod
    def _extract_error_details(
        error_strings: list[str],
    ) -> list[tuple[str, int | None, ErrorCode]]:
        """Parse structured error strings into (message, page_number, code) tuples."""
        details: list[tuple[str, int | None, ErrorCode]] = []
        for err in error_strings:
            page_num: int | None = None
            code = ErrorCode.E_OCR_FAILED
            if ErrorCode.E_OCR_TIMEOUT.value in err:
                code = ErrorCode.E_OCR_TIMEOUT
            # Try to extract page number from "page N" pattern
            import re

            match = re.search(r"page (\d+)", err)
            if match:
                page_num = int(match.group(1))
            details.append((err, page_num, code))
        return details
