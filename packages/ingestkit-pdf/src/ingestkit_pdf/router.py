"""PDFRouter -- orchestrator and public API for the ingestkit-pdf pipeline.

Routes PDF files through the full ingestion pipeline:

1. Security scan via :class:`PDFSecurityScanner`.
2. Compute deterministic :class:`IngestKey` for deduplication.
3. Open document with PyMuPDF and build :class:`DocumentProfile`.
4. Classify via :class:`PDFInspector` (Tier 1) with optional escalation to
   :class:`PDFLLMClassifier` (Tier 2/3) and LLM outage resilience (SPEC §5.2).
5. Route to the appropriate processor based on classification result.
6. Return a fully-assembled :class:`ProcessingResult`.

The router enforces **fail-closed** semantics: if classification is
inconclusive after all tiers, it returns a ``ProcessingResult`` with
``E_CLASSIFY_INCONCLUSIVE`` and zero chunks/tables.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import fitz  # type: ignore[import-untyped]

from ingestkit_core.idempotency import compute_ingest_key
from ingestkit_core.models import ClassificationTier, WrittenArtifacts

from ingestkit_pdf.config import PDFProcessorConfig
from ingestkit_pdf.errors import ErrorCode, IngestError
from ingestkit_pdf.inspector import PDFInspector
from ingestkit_pdf.llm_classifier import PDFLLMClassifier
from ingestkit_pdf.models import (
    ClassificationResult,
    ClassificationStageResult,
    DocumentMetadata,
    DocumentProfile,
    ExtractionQuality,
    IngestionMethod,
    PageProfile,
    PageType,
    ParseStageResult,
    PDFType,
    ProcessingResult,
)
from ingestkit_pdf.processors.ocr_processor import OCRProcessor
from ingestkit_pdf.processors.text_extractor import TextExtractor
from ingestkit_pdf.protocols import (
    EmbeddingBackend,
    LLMBackend,
    StructuredDBBackend,
    VectorStoreBackend,
)
from ingestkit_pdf.quality import QualityAssessor
from ingestkit_pdf.security import PDFSecurityScanner
from ingestkit_pdf.utils.language import detect_language
from ingestkit_pdf.utils.layout_analysis import LayoutAnalyzer

logger = logging.getLogger("ingestkit_pdf")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class PDFRouter:
    """Top-level orchestrator for the ingestkit-pdf pipeline.

    Builds all internal components (security scanner, inspector, LLM
    classifier, quality assessor, layout analyzer, and the two processor
    paths) from the injected backends and config, then exposes
    :meth:`can_handle`, :meth:`process`, and :meth:`process_batch` as
    the public API.

    Parameters
    ----------
    vector_store:
        Backend for vector storage (e.g. Qdrant).
    structured_db:
        Backend for structured/relational storage (e.g. SQLite).
    llm:
        Backend for LLM classification prompts.
    embedder:
        Backend for text embedding.
    config:
        Pipeline configuration. Uses defaults when *None*.
    """

    def __init__(
        self,
        vector_store: VectorStoreBackend,
        structured_db: StructuredDBBackend,
        llm: LLMBackend,
        embedder: EmbeddingBackend,
        config: PDFProcessorConfig | None = None,
    ) -> None:
        self._config = config or PDFProcessorConfig()

        # Security
        self._security_scanner = PDFSecurityScanner(self._config)

        # Classification
        self._inspector = PDFInspector(self._config)
        self._llm_classifier = PDFLLMClassifier(llm, self._config)

        # Quality
        self._quality_assessor = QualityAssessor(self._config)

        # Layout analysis
        self._layout_analyzer = LayoutAnalyzer(self._config)

        # Processors
        self._text_extractor = TextExtractor(
            vector_store=vector_store,
            embedder=embedder,
            config=self._config,
        )
        self._ocr_processor = OCRProcessor(
            vector_store=vector_store,
            embedder=embedder,
            llm=llm,
            config=self._config,
        )
        # ComplexProcessor not yet implemented -- accept None
        self._complex_processor: Any = None

        # Store backends for process_batch worker recreation
        self._vector_store = vector_store
        self._structured_db = structured_db
        self._llm = llm
        self._embedder = embedder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def can_handle(self, file_path: str) -> bool:
        """Return True if *file_path* ends with ``.pdf`` (case-insensitive)."""
        return file_path.lower().endswith(".pdf")

    def process(
        self,
        file_path: str,
        source_uri: str | None = None,
    ) -> ProcessingResult:
        """Classify and process a single PDF file. Synchronous.

        Implements the 15-step flow from SPEC section 17.1.

        Parameters
        ----------
        file_path:
            Filesystem path to the PDF file.
        source_uri:
            Optional override for the source URI stored in the ingest key.

        Returns
        -------
        ProcessingResult
            The fully-assembled result.
        """
        overall_start = time.monotonic()
        config = self._config
        filename = os.path.basename(file_path)
        ingest_run_id = str(uuid.uuid4())

        # ==============================================================
        # Step 1: Security Scan
        # ==============================================================
        metadata, security_errors = self._security_scanner.scan(file_path)
        fatal_errors = [
            e for e in security_errors if e.code.value.startswith("E_")
        ]
        security_warnings = [
            e for e in security_errors if not e.code.value.startswith("E_")
        ]

        if fatal_errors:
            elapsed = time.monotonic() - overall_start
            logger.error(
                "ingestkit_pdf | file=%s | code=%s | detail=%s",
                filename,
                fatal_errors[0].code.value,
                fatal_errors[0].message,
            )
            return self._build_error_result(
                file_path=file_path,
                ingest_key="",
                ingest_run_id=ingest_run_id,
                errors=[e.code.value for e in fatal_errors],
                warnings=[e.code.value for e in security_warnings],
                error_details=security_errors,
                elapsed=elapsed,
            )

        # ==============================================================
        # Step 2: Compute Ingest Key
        # ==============================================================
        ingest_key_obj = compute_ingest_key(
            file_path=file_path,
            parser_version=config.parser_version,
            tenant_id=config.tenant_id,
            source_uri=source_uri,
        )
        ingest_key = ingest_key_obj.key

        # ==============================================================
        # Step 4: Open Document with PyMuPDF
        # ==============================================================
        parse_start = time.monotonic()
        try:
            doc = fitz.open(file_path)
        except Exception:
            # Attempt repair by forcing PDF interpretation
            try:
                doc = fitz.open(file_path, filetype="pdf")
            except Exception as exc:
                elapsed = time.monotonic() - overall_start
                logger.error(
                    "ingestkit_pdf | file=%s | code=%s | detail=PyMuPDF open failed, "
                    "repair attempt also failed",
                    filename,
                    ErrorCode.E_PARSE_CORRUPT.value,
                )
                return self._build_error_result(
                    file_path=file_path,
                    ingest_key=ingest_key,
                    ingest_run_id=ingest_run_id,
                    errors=[ErrorCode.E_PARSE_CORRUPT.value],
                    warnings=[e.code.value for e in security_warnings],
                    error_details=[
                        IngestError(
                            code=ErrorCode.E_PARSE_CORRUPT,
                            message=f"PyMuPDF open failed, repair attempt also failed: {exc}",
                            stage="parse",
                            recoverable=False,
                        ),
                    ],
                    elapsed=elapsed,
                )

        try:
            # ==============================================================
            # Step 5: Extract Document Profile
            # ==============================================================
            profile = self._build_document_profile(
                file_path, doc, metadata,
                [e.code.value for e in security_warnings],
            )
            parse_duration = time.monotonic() - parse_start

            # ==============================================================
            # Steps 7-9: Tiered Classification
            # ==============================================================
            classify_start = time.monotonic()
            classification, classify_errors, classify_warnings, classify_error_details = (
                self._classify(profile)
            )
            classify_duration = time.monotonic() - classify_start

            # ==============================================================
            # Step 10: Fail-Closed Check
            # ==============================================================
            if classification.confidence == 0.0:
                elapsed = time.monotonic() - overall_start
                logger.error(
                    "ingestkit_pdf | file=%s | code=%s | detail=Classification "
                    "inconclusive after all tiers",
                    filename,
                    ErrorCode.E_CLASSIFY_INCONCLUSIVE.value,
                )
                all_errors = [ErrorCode.E_CLASSIFY_INCONCLUSIVE.value] + classify_errors
                all_warnings = [e.code.value for e in security_warnings] + classify_warnings
                all_error_details = classify_error_details + [
                    IngestError(
                        code=ErrorCode.E_CLASSIFY_INCONCLUSIVE,
                        message="Classification inconclusive after all tiers. Fail-closed.",
                        stage="classify",
                        recoverable=False,
                    ),
                ]
                return self._build_error_result(
                    file_path=file_path,
                    ingest_key=ingest_key,
                    ingest_run_id=ingest_run_id,
                    errors=all_errors,
                    warnings=all_warnings,
                    error_details=all_error_details,
                    elapsed=elapsed,
                )

            # ==============================================================
            # Build stage artifacts
            # ==============================================================
            parse_result = self._build_parse_stage_result(profile, parse_duration)

            classification_stage_result = ClassificationStageResult(
                tier_used=classification.tier_used,
                pdf_type=classification.pdf_type,
                confidence=classification.confidence,
                signals=classification.signals,
                reasoning=classification.reasoning,
                per_page_types=classification.per_page_types,
                classification_duration_seconds=classify_duration,
                degraded=classification.degraded,
            )

            # ==============================================================
            # Step 11: Route to Processor
            # ==============================================================
            if classification.pdf_type == PDFType.TEXT_NATIVE:
                result = self._text_extractor.process(
                    file_path=file_path,
                    profile=profile,
                    ingest_key=ingest_key,
                    ingest_run_id=ingest_run_id,
                    parse_result=parse_result,
                    classification_result=classification_stage_result,
                    classification=classification,
                )
            elif classification.pdf_type == PDFType.SCANNED:
                result = self._ocr_processor.process(
                    file_path=file_path,
                    profile=profile,
                    pages=None,  # full-document OCR
                    ingest_key=ingest_key,
                    ingest_run_id=ingest_run_id,
                    parse_result=parse_result,
                    classification_result=classification_stage_result,
                    classification=classification,
                )
            elif classification.pdf_type == PDFType.COMPLEX:
                if self._complex_processor is None:
                    elapsed = time.monotonic() - overall_start
                    logger.warning(
                        "ingestkit_pdf | file=%s | detail=ComplexProcessor not available, "
                        "COMPLEX type cannot be processed",
                        filename,
                    )
                    return self._build_error_result(
                        file_path=file_path,
                        ingest_key=ingest_key,
                        ingest_run_id=ingest_run_id,
                        errors=["ComplexProcessor not available"],
                        warnings=[e.code.value for e in security_warnings] + classify_warnings,
                        error_details=classify_error_details,
                        elapsed=elapsed,
                    )
                else:
                    result = self._complex_processor.process(
                        file_path=file_path,
                        profile=profile,
                        ingest_key=ingest_key,
                        ingest_run_id=ingest_run_id,
                        parse_result=parse_result,
                        classification_result=classification_stage_result,
                        classification=classification,
                    )
            else:
                # Should never happen -- fail-closed
                elapsed = time.monotonic() - overall_start
                return self._build_error_result(
                    file_path=file_path,
                    ingest_key=ingest_key,
                    ingest_run_id=ingest_run_id,
                    errors=[f"Unknown PDFType: {classification.pdf_type}"],
                    warnings=[],
                    error_details=[],
                    elapsed=elapsed,
                )

            # ==============================================================
            # Steps 12-13: Merge errors and assemble final result
            # ==============================================================
            self._merge_errors(
                result, classify_errors, classify_warnings, classify_error_details,
            )
            # Merge security warnings into result
            for sw in security_warnings:
                code_val = sw.code.value
                if code_val not in result.warnings:
                    result.warnings.append(code_val)
                result.error_details.append(sw)

            # Override processing_time to include full pipeline time
            elapsed = time.monotonic() - overall_start
            result = result.model_copy(
                update={"processing_time_seconds": elapsed}
            )

            # ==============================================================
            # Step 14: PII-Safe Logging
            # ==============================================================
            path_name = self._pdf_type_to_path_name(classification.pdf_type)
            ocr_pages = result.ocr_result.pages_ocrd if result.ocr_result else 0
            logger.info(
                "ingestkit_pdf | file=%s | ingest_key=%s | tier=%s | type=%s | "
                "confidence=%.2f | degraded=%s | path=%s | pages=%d | chunks=%d | "
                "tables=%d | ocr_pages=%d | time=%.1fs",
                filename,
                ingest_key[:8],
                classification.tier_used.value,
                classification.pdf_type.value,
                classification.confidence,
                str(classification.degraded).lower(),
                path_name,
                profile.page_count,
                result.chunks_created,
                result.tables_created,
                ocr_pages,
                elapsed,
            )

            # ==============================================================
            # Step 15: Return Result
            # ==============================================================
            return result

        finally:
            doc.close()

    def process_batch(
        self,
        file_paths: list[str],
    ) -> list[ProcessingResult]:
        """Process multiple PDFs with process isolation and per-document timeout.

        Uses ``ProcessPoolExecutor`` for process isolation per SPEC section 17.2.
        Each worker creates its own backends and router via
        :func:`create_default_router`.

        Parameters
        ----------
        file_paths:
            List of filesystem paths to PDF files.

        Returns
        -------
        list[ProcessingResult]
            One result per input file, in the same order.
        """
        if not file_paths:
            return []

        results: dict[int, ProcessingResult] = {}
        timeout = self._config.per_document_timeout_seconds

        with ProcessPoolExecutor(
            max_workers=min(len(file_paths), 4)
        ) as executor:
            future_to_idx = {
                executor.submit(
                    _process_single_file,
                    fp,
                    self._config.model_dump(),
                ): idx
                for idx, fp in enumerate(file_paths)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                fp = file_paths[idx]
                try:
                    result = future.result(timeout=timeout)
                    results[idx] = result
                except TimeoutError:
                    results[idx] = self._build_error_result(
                        file_path=fp,
                        ingest_key="",
                        ingest_run_id=str(uuid.uuid4()),
                        errors=["Processing timeout"],
                        warnings=[],
                        error_details=[],
                        elapsed=float(timeout),
                    )
                except Exception as exc:
                    results[idx] = self._build_error_result(
                        file_path=fp,
                        ingest_key="",
                        ingest_run_id=str(uuid.uuid4()),
                        errors=[f"Processing failed: {exc}"],
                        warnings=[],
                        error_details=[],
                        elapsed=0.0,
                    )

        return [results[i] for i in range(len(file_paths))]

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _build_document_profile(
        self,
        file_path: str,
        doc: fitz.Document,
        metadata: DocumentMetadata,
        security_warnings: list[str],
    ) -> DocumentProfile:
        """Build a :class:`DocumentProfile` from a PyMuPDF document.

        Includes per-page profiling, language detection, TOC extraction,
        and overall quality assessment.
        """
        content_hash = hashlib.sha256(Path(file_path).read_bytes()).hexdigest()

        pages: list[PageProfile] = []
        page_qualities: list[ExtractionQuality] = []
        for page_num in range(doc.page_count):
            page = doc[page_num]
            page_profile = self._build_page_profile(page, page_num + 1)
            pages.append(page_profile)
            page_qualities.append(page_profile.extraction_quality)

        # Page type distribution
        distribution: dict[str, int] = {}
        for p in pages:
            key = p.page_type.value
            distribution[key] = distribution.get(key, 0) + 1

        # Language detection
        detected_languages: list[str] = []
        if self._config.enable_language_detection:
            sample_texts: list[str] = []
            for i in range(min(5, doc.page_count)):
                text = doc[i].get_text()
                if len(text.strip()) > 50:
                    sample_texts.append(text)
            sample_text = " ".join(sample_texts)
            if sample_text:
                lang, _conf = detect_language(
                    sample_text, default_language=self._config.default_language,
                )
                detected_languages = [lang]

        # TOC extraction
        toc = doc.get_toc()
        has_toc = len(toc) > 0
        toc_entries: list[tuple[int, str, int]] | None = None
        if toc:
            toc_entries = [(lvl, title, page) for lvl, title, page in toc]

        # Overall quality
        overall_quality = self._quality_assessor.assess_document(page_qualities)

        return DocumentProfile(
            file_path=file_path,
            file_size_bytes=metadata.file_size_bytes,
            page_count=doc.page_count,
            content_hash=content_hash,
            metadata=metadata,
            pages=pages,
            page_type_distribution=distribution,
            detected_languages=detected_languages,
            has_toc=has_toc,
            toc_entries=toc_entries,
            overall_quality=overall_quality,
            security_warnings=list(security_warnings),
        )

    def _build_page_profile(
        self,
        page: fitz.Page,
        page_number: int,
    ) -> PageProfile:
        """Build a :class:`PageProfile` from a single PyMuPDF page."""
        text = page.get_text()
        words = text.split()

        # Images
        images = page.get_images(full=True)
        image_count = len(images)

        # Image coverage
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height
        image_area = 0.0
        for img_block in page.get_image_info():
            bbox = img_block.get("bbox", (0, 0, 0, 0))
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            image_area += w * h
        image_coverage = image_area / max(page_area, 1.0)

        # Fonts
        fonts = page.get_fonts()
        font_names = list({f[3] for f in fonts if f[3]})
        font_count = len(font_names)

        # Tables
        tables: Any = []
        if hasattr(page, "find_tables"):
            try:
                tables = page.find_tables()
            except Exception:
                tables = []
        table_count = len(tables) if tables else 0

        # Form fields
        has_form_fields = False
        try:
            widget_iter = page.widgets()
            if widget_iter is not None:
                has_form_fields = len(list(widget_iter)) > 0
        except Exception:
            pass

        # Multi-column detection
        layout_result = self._layout_analyzer.detect_columns(page)
        is_multi_column = layout_result.column_count > 1

        # Quality assessment
        quality = self._quality_assessor.assess_page(text, page_number)

        # Preliminary page type
        page_type = self._determine_page_type(
            text_length=len(text),
            word_count=len(words),
            image_count=image_count,
            image_coverage=image_coverage,
            font_count=font_count,
            table_count=table_count,
            has_form_fields=has_form_fields,
            is_multi_column=is_multi_column,
        )

        return PageProfile(
            page_number=page_number,
            text_length=len(text),
            word_count=len(words),
            image_count=image_count,
            image_coverage_ratio=min(image_coverage, 1.0),
            table_count=table_count,
            font_count=font_count,
            font_names=font_names,
            has_form_fields=has_form_fields,
            is_multi_column=is_multi_column,
            page_type=page_type,
            extraction_quality=quality,
        )

    @staticmethod
    def _determine_page_type(
        *,
        text_length: int,
        word_count: int,
        image_count: int,
        image_coverage: float,
        font_count: int,
        table_count: int,
        has_form_fields: bool,
        is_multi_column: bool,
    ) -> PageType:
        """Assign a preliminary page type based on simple heuristics."""
        if text_length < 10 and image_count == 0:
            return PageType.BLANK
        if text_length < 50 and image_coverage > 0.7:
            return PageType.SCANNED
        if has_form_fields:
            return PageType.FORM
        if table_count >= 1:
            return PageType.TABLE_HEAVY
        if is_multi_column or (table_count >= 1 and word_count > 50):
            return PageType.MIXED
        return PageType.TEXT

    def _classify(
        self,
        profile: DocumentProfile,
    ) -> tuple[ClassificationResult, list[str], list[str], list[IngestError]]:
        """Tiered classification with LLM outage resilience (SPEC section 5.2).

        Returns
        -------
        tuple
            ``(classification, errors, warnings, error_details)``
        """
        errors: list[str] = []
        warnings: list[str] = []
        error_details: list[IngestError] = []

        # Step 7: Tier 1 -- ALWAYS runs, zero external dependencies
        tier1_result = self._inspector.classify(profile)

        # High confidence threshold: tier1_high_confidence_signals / 5
        high_conf = self._config.tier1_high_confidence_signals / 5
        if tier1_result.confidence >= high_conf:
            return tier1_result, errors, warnings, error_details

        # Tier 2 attempt with LLM outage resilience
        try:
            tier2_result = self._llm_classifier.classify(
                profile, ClassificationTier.LLM_BASIC,
            )
            if tier2_result.confidence >= self._config.tier2_confidence_threshold:
                return tier2_result, errors, warnings, error_details

            # Tier 3 escalation
            if self._config.enable_tier3:
                try:
                    tier3_result = self._llm_classifier.classify(
                        profile, ClassificationTier.LLM_REASONING,
                    )
                    return tier3_result, errors, warnings, error_details
                except (ConnectionError, TimeoutError):
                    # Tier 3 failed -- fall through to degrade check
                    pass

            # Tier 2 returned but low confidence, Tier 3 disabled or failed
            if tier2_result.confidence > 0.0:
                return tier2_result, errors, warnings, error_details

            # Fall through to Tier 1 degraded
        except (ConnectionError, TimeoutError) as exc:
            # LLM outage -- degrade to Tier 1
            warnings.extend([
                ErrorCode.W_LLM_UNAVAILABLE.value,
                ErrorCode.W_CLASSIFICATION_DEGRADED.value,
            ])
            error_details.append(
                IngestError(
                    code=ErrorCode.W_LLM_UNAVAILABLE,
                    message=(
                        f"LLM backend unreachable ({type(exc).__name__}), "
                        "degraded to Tier 1"
                    ),
                    stage="classify",
                    recoverable=True,
                )
            )
            logger.warning(
                "ingestkit_pdf | file=%s | code=%s | detail=LLM backend "
                "unreachable (%s), classification degraded to Tier 1",
                os.path.basename(profile.file_path),
                ErrorCode.W_LLM_UNAVAILABLE.value,
                type(exc).__name__,
            )
            degraded = tier1_result.model_copy(update={"degraded": True})
            return degraded, errors, warnings, error_details

        # If we get here: Tier 1 was low-confidence AND LLM tiers didn't help
        return tier1_result, errors, warnings, error_details

    def _build_parse_stage_result(
        self,
        profile: DocumentProfile,
        parse_duration: float,
    ) -> ParseStageResult:
        """Build a :class:`ParseStageResult` from the document profile."""
        pages_with_text = sum(
            1 for p in profile.pages if p.page_type != PageType.BLANK
        )
        pages_skipped = profile.page_count - pages_with_text
        skipped_reasons: dict[int, str] = {
            p.page_number: "blank"
            for p in profile.pages
            if p.page_type == PageType.BLANK
        }

        return ParseStageResult(
            pages_extracted=pages_with_text,
            pages_skipped=pages_skipped,
            skipped_reasons=skipped_reasons,
            extraction_method="pymupdf",
            overall_quality=profile.overall_quality,
            parse_duration_seconds=parse_duration,
        )

    def _build_error_result(
        self,
        file_path: str,
        ingest_key: str,
        ingest_run_id: str,
        errors: list[str],
        warnings: list[str],
        error_details: list[IngestError],
        elapsed: float,
    ) -> ProcessingResult:
        """Build a :class:`ProcessingResult` representing a fatal error."""
        empty_quality = ExtractionQuality(
            printable_ratio=0.0,
            avg_words_per_page=0.0,
            pages_with_text=0,
            total_pages=0,
            extraction_method="none",
        )
        return ProcessingResult(
            file_path=file_path,
            ingest_key=ingest_key,
            ingest_run_id=ingest_run_id,
            tenant_id=self._config.tenant_id,
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
            errors=errors,
            warnings=warnings,
            error_details=error_details,
            processing_time_seconds=elapsed,
        )

    @staticmethod
    def _merge_errors(
        result: ProcessingResult,
        errors: list[str],
        warnings: list[str],
        error_details: list[IngestError],
    ) -> None:
        """Merge classification errors/warnings into a ProcessingResult."""
        for e in errors:
            if e not in result.errors:
                result.errors.append(e)
        for w in warnings:
            if w not in result.warnings:
                result.warnings.append(w)
        result.error_details.extend(error_details)

    @staticmethod
    def _pdf_type_to_ingestion_method(pdf_type: PDFType) -> IngestionMethod:
        """Map a PDFType to its corresponding IngestionMethod."""
        return {
            PDFType.TEXT_NATIVE: IngestionMethod.TEXT_EXTRACTION,
            PDFType.SCANNED: IngestionMethod.OCR_PIPELINE,
            PDFType.COMPLEX: IngestionMethod.COMPLEX_PROCESSING,
        }[pdf_type]

    @staticmethod
    def _pdf_type_to_path_name(pdf_type: PDFType) -> str:
        """Map a PDFType to a human-readable path name for logging."""
        return {
            PDFType.TEXT_NATIVE: "text_extraction",
            PDFType.SCANNED: "ocr_pipeline",
            PDFType.COMPLEX: "complex_processing",
        }.get(pdf_type, "unknown")


# ---------------------------------------------------------------------------
# Module-level worker function (must be picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def _process_single_file(
    file_path: str, config_dict: dict,
) -> ProcessingResult:
    """Worker function for :meth:`PDFRouter.process_batch`.

    Runs in a child process. Creates fresh backends and router per-worker
    to avoid cross-process sharing issues (SPEC section 18.1).
    """
    config = PDFProcessorConfig(**config_dict)
    router = create_default_router(config=config)
    return router.process(file_path)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_default_router(**overrides: Any) -> PDFRouter:
    """Create a PDFRouter with default backends (Qdrant, SQLite, Ollama).

    Convenience factory for local development and testing. All defaults
    can be overridden via keyword arguments:

    - ``vector_store``: VectorStoreBackend (default: QdrantVectorStore)
    - ``structured_db``: StructuredDBBackend (default: SQLiteStructuredDB)
    - ``llm``: LLMBackend (default: OllamaLLM)
    - ``embedder``: EmbeddingBackend (default: OllamaEmbedding)
    - ``config``: PDFProcessorConfig (default: PDFProcessorConfig())

    Any other keyword arguments are passed to PDFProcessorConfig.

    Returns
    -------
    PDFRouter
        A fully-configured router ready for ``process()`` calls.

    Raises
    ------
    ImportError
        If optional backend dependencies are not installed.
    """
    from ingestkit_excel.backends import (
        OllamaEmbedding,
        OllamaLLM,
        QdrantVectorStore,
        SQLiteStructuredDB,
    )

    # Separate known router kwargs from config overrides
    router_keys = {"vector_store", "structured_db", "llm", "embedder", "config"}
    router_kwargs = {k: v for k, v in overrides.items() if k in router_keys}
    config_kwargs = {k: v for k, v in overrides.items() if k not in router_keys}

    config = router_kwargs.pop("config", None)
    if config is None and config_kwargs:
        config = PDFProcessorConfig(**config_kwargs)
    elif config is None:
        config = PDFProcessorConfig()

    vector_store = router_kwargs.pop("vector_store", None)
    if vector_store is None:
        vector_store = QdrantVectorStore()

    structured_db = router_kwargs.pop("structured_db", None)
    if structured_db is None:
        structured_db = SQLiteStructuredDB()

    llm = router_kwargs.pop("llm", None)
    if llm is None:
        llm = OllamaLLM()

    embedder = router_kwargs.pop("embedder", None)
    if embedder is None:
        embedder = OllamaEmbedding(model=config.embedding_model)

    return PDFRouter(
        vector_store=vector_store,
        structured_db=structured_db,
        llm=llm,
        embedder=embedder,
        config=config,
    )
