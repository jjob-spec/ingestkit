"""FormRouter: orchestrates matching, extraction, and output.

Entry point for the form processing pipeline. Routes matched forms
through the appropriate extractor and output writer (spec section 9).
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING

from ingestkit_forms.confidence import (
    apply_confidence_actions,
    compute_overall_confidence,
)
from ingestkit_forms.errors import FormErrorCode, FormIngestException
from ingestkit_forms.extractors import (
    ExcelCellExtractor,
    NativePDFExtractor,
    OCROverlayExtractor,
    VLMFieldExtractor,
)
from ingestkit_forms.idempotency import compute_form_extraction_key, compute_ingest_key
from ingestkit_forms.matcher import FormMatcher, detect_source_format
from ingestkit_forms.models import (
    ExtractedField,
    FormExtractionResult,
    FormIngestRequest,
    FormProcessingResult,
    FormTemplate,
    FormWrittenArtifacts,
    SourceFormat,
    TemplateMatch,
)
from ingestkit_forms.output import FormChunkWriter, FormDBWriter, FormDualWriter

if TYPE_CHECKING:
    from ingestkit_forms.config import FormProcessorConfig
    from ingestkit_forms.protocols import (
        EmbeddingBackend,
        FormDBBackend,
        LayoutFingerprinter,
        FormTemplateStore,
        OCRBackend,
        PDFWidgetBackend,
        VectorStoreBackend,
        VLMBackend,
    )

logger = logging.getLogger("ingestkit_forms")


class FormRouter:
    """Orchestrates the full form extraction pipeline.

    Accepts all backend dependencies via DI. Routes incoming documents
    through: ingest key computation, template matching/resolution,
    source detection, field extraction, confidence scoring, optional
    VLM fallback, and dual-write output.

    The return type of ``extract_form`` is ``FormProcessingResult | None``.
    A ``None`` return signals graceful fallthrough (no template matched
    and no manual override specified).
    """

    def __init__(
        self,
        template_store: FormTemplateStore,
        fingerprinter: LayoutFingerprinter,
        form_db: FormDBBackend,
        vector_store: VectorStoreBackend,
        embedder: EmbeddingBackend,
        ocr_backend: OCRBackend | None = None,
        pdf_widget_backend: PDFWidgetBackend | None = None,
        vlm_backend: VLMBackend | None = None,
        config: FormProcessorConfig | None = None,
    ) -> None:
        from ingestkit_forms.config import FormProcessorConfig as _Cfg

        self._config = config or _Cfg()
        self._template_store = template_store
        self._form_db = form_db
        self._vector_store = vector_store
        self._embedder = embedder
        self._ocr_backend = ocr_backend
        self._pdf_widget_backend = pdf_widget_backend
        self._vlm_backend = vlm_backend

        # Internal components
        self._matcher = FormMatcher(template_store, fingerprinter, self._config)

        # Extractors
        self._excel_extractor = ExcelCellExtractor(self._config)

        self._native_pdf_extractor: NativePDFExtractor | None = None
        if pdf_widget_backend is not None:
            self._native_pdf_extractor = NativePDFExtractor(
                pdf_backend=pdf_widget_backend,
                config=self._config,
                ocr_backend=ocr_backend,
            )
        else:
            logger.warning(
                "%s: PDFWidgetBackend not provided — native PDF field "
                "extraction will be unavailable",
                FormErrorCode.W_FORM_NATIVE_FIELDS_UNAVAILABLE.value,
            )

        self._ocr_extractor: OCROverlayExtractor | None = None
        if ocr_backend is not None:
            self._ocr_extractor = OCROverlayExtractor(ocr_backend, self._config)

        self._vlm_extractor: VLMFieldExtractor | None = None
        if vlm_backend is not None:
            self._vlm_extractor = VLMFieldExtractor(vlm_backend, self._config)

        # Output writers
        db_writer = FormDBWriter(db=form_db, config=self._config)
        chunk_writer = FormChunkWriter(
            vector_store=vector_store,
            embedder=embedder,
            config=self._config,
        )
        self._dual_writer = FormDualWriter(db_writer, chunk_writer, self._config)

    # ------------------------------------------------------------------
    # Public matching interface
    # ------------------------------------------------------------------

    def match_document(
        self,
        file_path: str,
        tenant_id: str | None = None,
    ) -> list[TemplateMatch]:
        """Match a document against all active templates.

        Public wrapper around ``FormMatcher.match_document`` that adds
        structured logging per spec section 18.4.

        If ``config.form_match_enabled`` is ``False``, returns an empty
        list immediately.

        Returns:
            Ranked list of ``TemplateMatch`` objects with confidence >= 0.5
            (the warning floor). The caller is responsible for applying the
            higher ``form_match_confidence_threshold`` for gate decisions
            (see ``try_match``).
        """
        if not self._config.form_match_enabled:
            logger.info(
                "forms.match.disabled",
                extra={"template_candidates": 0, "confidence": 0.0},
            )
            return []

        t0 = time.monotonic()
        matches = self._matcher.match_document(file_path)
        duration_ms = (time.monotonic() - t0) * 1000.0

        top_confidence = matches[0].confidence if matches else 0.0
        match_result = "auto" if matches else "fallthrough"

        logger.info(
            "forms.match.%s",
            match_result,
            extra={
                "template_candidates": len(matches),
                "confidence": top_confidence,
                "match_duration_ms": duration_ms,
                "template_id": matches[0].template_id if matches else None,
            },
        )

        return matches

    def try_match(
        self,
        file_path: str,
        tenant_id: str | None = None,
    ) -> TemplateMatch | None:
        """Pipeline gate: probe whether a document matches a form template.

        This is a **convenience method for the orchestration layer** (not
        part of the ``FormPluginAPI`` protocol). The caller invokes this
        BEFORE deciding to route to Path F vs. the standard pipeline.

        Semantics:
            - Runs auto-match via ``match_document(file_path, tenant_id)``.
            - Returns the top match only when its confidence is >=
              ``config.form_match_confidence_threshold`` (default 0.8).
            - On any exception: log a warning, return ``None`` (graceful
              fallthrough -- zero state mutation).

        For manual template override, use ``extract_form()`` directly
        with ``FormIngestRequest(template_id=..., manual_override=True)``.

        Returns:
            ``TemplateMatch`` if a template was matched above threshold,
            ``None`` otherwise.
        """
        try:
            if not self._config.form_match_enabled:
                return None

            # Auto-match path
            matches = self.match_document(file_path, tenant_id=tenant_id)
            if not matches:
                return None

            top = matches[0]
            if top.confidence >= self._config.form_match_confidence_threshold:
                return top

            return None

        except Exception:
            logger.warning(
                "try_match fallthrough due to exception for '%s'",
                file_path,
                exc_info=True,
            )
            return None

    def extract_form(
        self,
        request: FormIngestRequest,
    ) -> FormProcessingResult | None:
        """Execute the full form extraction pipeline.

        Pipeline steps (spec 4.2):
            1. Compute global ingest key.
            2. Generate ingest_run_id.
            3. Template resolution (manual override or auto-match).
            4. Compute form extraction key.
            5. Detect source format.
            6. Select and run extractor.
            7. Per-field confidence check + VLM fallback.
            8. Overall confidence fail-closed check.
            9. Dual-write (DB + chunks).
           10. Assemble and return FormProcessingResult.

        Returns:
            FormProcessingResult on success, or None if no template matched
            and no manual template_id was provided (graceful fallthrough).
        """
        t0 = time.monotonic()
        errors: list[str] = []
        warnings: list[str] = []
        error_details: list = []
        cfg = self._config

        # Step 1: Compute global ingest key
        tenant_id = request.tenant_id or cfg.tenant_id
        ingest_key_obj = compute_ingest_key(
            file_path=request.file_path,
            parser_version=cfg.parser_version,
            tenant_id=tenant_id,
            source_uri=request.source_uri,
        )
        ingest_key = ingest_key_obj.key

        # Step 2: Generate ingest_run_id
        ingest_run_id = str(uuid.uuid4())

        # Step 3: Template resolution (with structured logging per 18.4)
        template: FormTemplate
        match_method: str
        match_confidence: float | None = None
        t_match = time.monotonic()

        if request.manual_override or request.template_id is not None:
            # Manual override (explicit flag or template_id presence)
            template = self._matcher.resolve_manual_override(request)
            match_method = "manual_override"
            match_duration_ms = (time.monotonic() - t_match) * 1000.0
            logger.info(
                "forms.match.manual",
                extra={
                    "template_id": template.template_id,
                    "template_version": template.version,
                    "confidence": 1.0,
                    "match_duration_ms": match_duration_ms,
                },
            )
        else:
            # Auto-match
            matches = self._matcher.match_document(request.file_path)
            above_threshold = [
                m
                for m in matches
                if m.confidence >= cfg.form_match_confidence_threshold
            ]
            match_duration_ms = (time.monotonic() - t_match) * 1000.0
            top_conf = matches[0].confidence if matches else 0.0
            match_result_str = "auto" if above_threshold else "fallthrough"

            logger.info(
                "forms.match.%s",
                match_result_str,
                extra={
                    "template_candidates": len(matches),
                    "confidence": top_conf,
                    "match_duration_ms": match_duration_ms,
                    "template_id": matches[0].template_id if matches else None,
                },
            )

            if not above_threshold:
                return None

            best = above_threshold[0]
            template_obj = self._template_store.get_template(
                best.template_id, version=best.template_version
            )
            if template_obj is None:
                logger.warning(
                    "forms.match.template_not_found",
                    extra={
                        "template_id": best.template_id,
                        "template_version": best.template_version,
                    },
                )
                return None
            template = template_obj
            match_method = "auto_detect"
            match_confidence = best.confidence

        # Step 4: Compute form extraction key (used for dedup; stored for future gate)
        _extraction_key = compute_form_extraction_key(
            ingest_key, template.template_id, template.version
        )

        # Step 5-6: Detect source, select extractor, run extraction
        extraction_method, fields, extraction_duration = self._run_extraction(
            request.file_path, template
        )

        # Step 7: Per-field confidence actions + VLM fallback
        fields, overall_confidence, field_warnings = (
            self._apply_confidence_and_vlm(fields, template, request.file_path)
        )
        warnings.extend(field_warnings)

        # Structured extract-stage logging per spec 18.4
        fields_failed = sum(1 for f in fields if f.value is None)
        logger.info(
            "forms.extract.completed",
            extra={
                "template_id": template.template_id,
                "template_version": template.version,
                "extraction_method": extraction_method,
                "duration_s": extraction_duration,
                "field_count": len(fields),
                "fields_extracted": len(fields) - fields_failed,
                "fields_failed": fields_failed,
            },
        )

        # Step 8: Fail-closed check
        if overall_confidence < cfg.form_extraction_min_overall_confidence:
            logger.warning(
                "forms.extract.low_confidence",
                extra={
                    "template_id": template.template_id,
                    "overall_confidence": overall_confidence,
                    "threshold": cfg.form_extraction_min_overall_confidence,
                },
            )
            extraction_result = self._build_extraction_result(
                template=template,
                fields=[],
                overall_confidence=overall_confidence,
                extraction_method=extraction_method,
                match_method=match_method,
                match_confidence=match_confidence,
                source_uri=request.source_uri or request.file_path,
                extraction_duration=extraction_duration,
            )
            errors.append(
                f"Overall confidence {overall_confidence:.3f} "
                f"below threshold {cfg.form_extraction_min_overall_confidence}"
            )
            from ingestkit_forms.errors import FormIngestError

            error_details.append(
                FormIngestError(
                    code=FormErrorCode.E_FORM_EXTRACTION_LOW_CONFIDENCE,
                    message=(
                        f"Overall confidence {overall_confidence:.3f} "
                        f"below {cfg.form_extraction_min_overall_confidence}"
                    ),
                    stage="extraction",
                    recoverable=False,
                    template_id=template.template_id,
                    template_version=template.version,
                )
            )
            return FormProcessingResult(
                file_path=request.file_path,
                ingest_key=ingest_key,
                ingest_run_id=ingest_run_id,
                tenant_id=tenant_id,
                extraction_result=extraction_result,
                embed_result=None,
                chunks_created=0,
                tables_created=0,
                tables=[],
                written=FormWrittenArtifacts(vector_collection=cfg.default_collection),
                errors=errors,
                warnings=warnings,
                error_details=error_details,
                processing_time_seconds=time.monotonic() - t0,
            )

        # Step 9: Build extraction result for dual-write
        extraction_result = self._build_extraction_result(
            template=template,
            fields=fields,
            overall_confidence=overall_confidence,
            extraction_method=extraction_method,
            match_method=match_method,
            match_confidence=match_confidence,
            source_uri=request.source_uri or request.file_path,
            extraction_duration=extraction_duration,
        )

        # Step 10: Dual write
        written, write_errors, write_warnings, write_error_details, embed_result = (
            self._dual_writer.write(
                extraction=extraction_result,
                template=template,
                source_uri=request.source_uri or request.file_path,
                ingest_key=ingest_key,
                ingest_run_id=ingest_run_id,
            )
        )
        errors.extend(write_errors)
        warnings.extend(write_warnings)
        error_details.extend(write_error_details)

        # Step 11: Assemble final result
        return FormProcessingResult(
            file_path=request.file_path,
            ingest_key=ingest_key,
            ingest_run_id=ingest_run_id,
            tenant_id=tenant_id,
            extraction_result=extraction_result,
            embed_result=embed_result,
            chunks_created=len(written.vector_point_ids),
            tables_created=len(written.db_table_names),
            tables=written.db_table_names,
            written=written,
            errors=errors,
            warnings=warnings,
            error_details=error_details,
            processing_time_seconds=time.monotonic() - t0,
        )

    def extract_form_batch(
        self,
        requests: list[FormIngestRequest],
    ) -> list[FormProcessingResult | None]:
        """Process multiple form requests sequentially.

        Returns:
            List of results, one per request. None entries indicate
            graceful fallthrough (no template match).
        """
        return [self.extract_form(req) for req in requests]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _select_extractor(
        self,
        file_path: str,
        template: FormTemplate,
    ) -> tuple[str, object]:
        """Select the appropriate extractor based on source format.

        Returns:
            (extraction_method_name, extractor_instance) tuple.

        Raises:
            FormIngestException if no suitable extractor is available.
        """
        fmt = detect_source_format(file_path)

        if fmt == SourceFormat.XLSX:
            return "cell_mapping", self._excel_extractor

        if fmt == SourceFormat.PDF:
            if (
                self._pdf_widget_backend is not None
                and self._pdf_widget_backend.has_form_fields(file_path)
            ):
                if self._native_pdf_extractor is None:
                    raise FormIngestException(
                        code=FormErrorCode.E_FORM_NATIVE_FIELDS_UNAVAILABLE,
                        message="NativePDFExtractor not initialized",
                        stage="extraction",
                        recoverable=False,
                    )
                return "native_fields", self._native_pdf_extractor
            # Fallback: OCR for scanned/flattened PDFs
            if self._ocr_extractor is not None:
                return "ocr_overlay", self._ocr_extractor
            raise FormIngestException(
                code=FormErrorCode.E_FORM_OCR_FAILED,
                message="No OCR backend available for scanned PDF",
                stage="extraction",
                recoverable=False,
            )

        if fmt == SourceFormat.IMAGE:
            if self._ocr_extractor is not None:
                return "ocr_overlay", self._ocr_extractor
            raise FormIngestException(
                code=FormErrorCode.E_FORM_OCR_FAILED,
                message="No OCR backend available for image",
                stage="extraction",
                recoverable=False,
            )

        raise FormIngestException(
            code=FormErrorCode.E_FORM_UNSUPPORTED_FORMAT,
            message=f"Unsupported source format: {fmt}",
            stage="extraction",
            recoverable=False,
        )

    def _run_extraction(
        self,
        file_path: str,
        template: FormTemplate,
    ) -> tuple[str, list[ExtractedField], float]:
        """Run the appropriate extractor and return results with timing.

        Returns:
            (extraction_method, extracted_fields, extraction_duration_seconds)
        """
        t0 = time.monotonic()
        method_name, extractor = self._select_extractor(file_path, template)
        fields = extractor.extract(file_path, template)
        duration = time.monotonic() - t0
        return method_name, fields, duration

    def _apply_confidence_and_vlm(
        self,
        fields: list[ExtractedField],
        template: FormTemplate,
        file_path: str,
    ) -> tuple[list[ExtractedField], float, list[str]]:
        """Apply per-field confidence actions and optional VLM fallback.

        Returns:
            (updated_fields, overall_confidence, warnings)
        """
        cfg = self._config
        all_warnings: list[str] = []

        # Per-field confidence actions
        updated_fields: list[ExtractedField] = []
        for field in fields:
            updated, field_warnings = apply_confidence_actions(field, cfg)
            updated_fields.append(updated)
            all_warnings.extend(field_warnings)

        # VLM fallback for fields marked as vlm_fallback_pending
        if (
            self._vlm_extractor is not None
            and cfg.form_vlm_enabled
        ):
            vlm_pending = [
                f for f in updated_fields
                if f.extraction_method == "vlm_fallback_pending"
            ]
            if vlm_pending:
                updated_fields = self._vlm_extractor.apply_vlm_fallback(
                    updated_fields, template, file_path
                )

        overall = compute_overall_confidence(updated_fields, template)
        return updated_fields, overall, all_warnings

    def _build_extraction_result(
        self,
        template: FormTemplate,
        fields: list[ExtractedField],
        overall_confidence: float,
        extraction_method: str,
        match_method: str,
        match_confidence: float | None,
        source_uri: str,
        extraction_duration: float,
    ) -> FormExtractionResult:
        """Assemble a FormExtractionResult from pipeline data."""
        return FormExtractionResult(
            template_id=template.template_id,
            template_name=template.name,
            template_version=template.version,
            source_uri=source_uri,
            source_format=template.source_format.value,
            fields=fields,
            overall_confidence=overall_confidence,
            extraction_method=extraction_method,
            match_method=match_method,
            match_confidence=match_confidence,
            pages_processed=template.page_count,
            extraction_duration_seconds=extraction_duration,
        )


def create_default_router(
    *,
    template_store: FormTemplateStore | None = None,
    fingerprinter: LayoutFingerprinter | None = None,
    form_db: FormDBBackend | None = None,
    vector_store: VectorStoreBackend | None = None,
    embedder: EmbeddingBackend | None = None,
    ocr_backend: OCRBackend | None = None,
    pdf_widget_backend: PDFWidgetBackend | None = None,
    vlm_backend: VLMBackend | None = None,
    config: FormProcessorConfig | None = None,
) -> FormRouter:
    """Create a FormRouter with sensible defaults where possible.

    Required backends (no defaults exist in ingestkit-forms):
        - ``template_store``
        - ``fingerprinter``
        - ``form_db``
        - ``vector_store``
        - ``embedder``

    Optional backends (default to None):
        - ``ocr_backend``
        - ``pdf_widget_backend``
        - ``vlm_backend``

    Raises:
        ValueError: If any required backend is not provided.
    """
    missing: list[str] = []
    if template_store is None:
        missing.append("template_store")
    if fingerprinter is None:
        missing.append("fingerprinter")
    if form_db is None:
        missing.append("form_db")
    if vector_store is None:
        missing.append("vector_store")
    if embedder is None:
        missing.append("embedder")

    if missing:
        raise ValueError(
            f"Required backend(s) not provided: {', '.join(missing)}. "
            "ingestkit-forms has no built-in concrete backends. "
            "Pass all required backends explicitly."
        )

    return FormRouter(
        template_store=template_store,  # type: ignore[arg-type]
        fingerprinter=fingerprinter,  # type: ignore[arg-type]
        form_db=form_db,  # type: ignore[arg-type]
        vector_store=vector_store,  # type: ignore[arg-type]
        embedder=embedder,  # type: ignore[arg-type]
        ocr_backend=ocr_backend,
        pdf_widget_backend=pdf_widget_backend,
        vlm_backend=vlm_backend,
        config=config,
    )
