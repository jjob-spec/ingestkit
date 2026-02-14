"""ExcelRouter -- orchestrator and public API for the ingestkit-excel pipeline.

Routes Excel files through the full ingestion pipeline:

1. Compute deterministic :class:`IngestKey` for deduplication.
2. Parse via :class:`ParserChain` (three-tier per-sheet fallback).
3. Classify via :class:`ExcelInspector` (Tier 1) with optional escalation to
   :class:`LLMClassifier` (Tier 2/3).
4. Route to the appropriate processor based on classification result.
5. Return a fully-assembled :class:`ProcessingResult`.

The router enforces **fail-closed** semantics: if classification is
inconclusive after all tiers, it returns a ``ProcessingResult`` with
``E_CLASSIFY_INCONCLUSIVE`` and zero chunks/tables.
"""

from __future__ import annotations

import logging
import os
import time
import uuid

from ingestkit_excel.config import ExcelProcessorConfig
from ingestkit_excel.errors import ErrorCode, IngestError
from ingestkit_excel.idempotency import compute_ingest_key
from ingestkit_excel.inspector import ExcelInspector
from ingestkit_excel.llm_classifier import LLMClassifier
from ingestkit_excel.models import (
    ClassificationResult,
    ClassificationStageResult,
    ClassificationTier,
    FileProfile,
    FileType,
    IngestionMethod,
    ParseStageResult,
    ParserUsed,
    ProcessingResult,
    WrittenArtifacts,
)
from ingestkit_excel.parser_chain import ParserChain
from ingestkit_excel.processors.serializer import TextSerializer
from ingestkit_excel.processors.splitter import HybridSplitter
from ingestkit_excel.processors.structured_db import StructuredDBProcessor
from ingestkit_excel.protocols import (
    EmbeddingBackend,
    LLMBackend,
    StructuredDBBackend,
    VectorStoreBackend,
)

logger = logging.getLogger("ingestkit_excel")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class ExcelRouter:
    """Orchestrator that drives the full Excel ingestion pipeline.

    Builds all internal components (parser chain, inspector, LLM classifier,
    and the three processor paths) from the injected backends and config,
    then exposes :meth:`process` and :meth:`process_batch` as the public API.

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
        config: ExcelProcessorConfig | None = None,
    ) -> None:
        self._config = config or ExcelProcessorConfig()

        # Build internal pipeline components
        self._parser_chain = ParserChain(self._config)
        self._inspector = ExcelInspector(self._config)
        self._llm_classifier = LLMClassifier(llm, self._config)

        # Build processors
        self._structured_db_processor = StructuredDBProcessor(
            structured_db=structured_db,
            vector_store=vector_store,
            embedder=embedder,
            config=self._config,
        )
        self._text_serializer = TextSerializer(
            vector_store=vector_store,
            embedder=embedder,
            config=self._config,
        )
        self._hybrid_splitter = HybridSplitter(
            structured_processor=self._structured_db_processor,
            text_serializer=self._text_serializer,
            config=self._config,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        file_path: str,
        source_uri: str | None = None,
    ) -> ProcessingResult:
        """Process a single Excel file through the full ingestion pipeline.

        Parameters
        ----------
        file_path:
            Filesystem path to the ``.xlsx`` file.
        source_uri:
            Optional override for the source URI stored in the ingest key.
            When *None*, the canonical absolute path is used.

        Returns
        -------
        ProcessingResult
            The fully-assembled result, including parse, classification,
            embedding, and processing stage outputs.
        """
        overall_start = time.monotonic()
        config = self._config
        filename = os.path.basename(file_path)

        # ----------------------------------------------------------
        # Step 1: Compute ingest key
        # ----------------------------------------------------------
        ingest_key_obj = compute_ingest_key(
            file_path=file_path,
            parser_version=config.parser_version,
            tenant_id=config.tenant_id,
            source_uri=source_uri,
        )
        ingest_key = ingest_key_obj.key

        # ----------------------------------------------------------
        # Step 2: Generate ingest run ID
        # ----------------------------------------------------------
        ingest_run_id = str(uuid.uuid4())

        # ----------------------------------------------------------
        # Step 3: Parse
        # ----------------------------------------------------------
        parse_start = time.monotonic()
        profile, parse_errors = self._parser_chain.parse(file_path)
        parse_duration = time.monotonic() - parse_start

        # ----------------------------------------------------------
        # Step 4: Build ParseStageResult
        # ----------------------------------------------------------
        parse_result = self._build_parse_stage_result(
            profile, parse_errors, parse_duration
        )

        # Log parse fallbacks as warnings
        fallback_errors = [
            e for e in parse_errors
            if e.code == ErrorCode.W_PARSER_FALLBACK
        ]
        for fe in fallback_errors:
            logger.warning(
                "Parse fallback for %s: %s", filename, fe.message
            )

        # ----------------------------------------------------------
        # Step 5: Classify (tiered escalation)
        # ----------------------------------------------------------
        classify_start = time.monotonic()

        # Tier 1: Rule-based inspector
        classification = self._inspector.classify(profile)
        tier_used = ClassificationTier.RULE_BASED

        if classification.confidence == 0.0:
            # Escalate to Tier 2: LLM Basic
            logger.warning(
                "Tier 1 inconclusive for %s, escalating to Tier 2.",
                filename,
            )
            classification = self._llm_classifier.classify(
                profile, ClassificationTier.LLM_BASIC
            )
            tier_used = ClassificationTier.LLM_BASIC

            if (
                classification.confidence < config.tier2_confidence_threshold
                and config.enable_tier3
            ):
                # Escalate to Tier 3: LLM Reasoning
                logger.warning(
                    "Tier 2 confidence %.2f < threshold %.2f for %s, "
                    "escalating to Tier 3.",
                    classification.confidence,
                    config.tier2_confidence_threshold,
                    filename,
                )
                classification = self._llm_classifier.classify(
                    profile, ClassificationTier.LLM_REASONING
                )
                tier_used = ClassificationTier.LLM_REASONING

        classify_duration = time.monotonic() - classify_start

        # ----------------------------------------------------------
        # Step 6: Fail-closed check
        # ----------------------------------------------------------
        if classification.confidence == 0.0:
            logger.error(
                "Classification inconclusive for %s after all tiers. "
                "Returning fail-closed result.",
                filename,
            )
            classification_result = ClassificationStageResult(
                tier_used=tier_used,
                file_type=classification.file_type,
                confidence=0.0,
                signals=classification.signals,
                reasoning=classification.reasoning,
                per_sheet_types=classification.per_sheet_types,
                classification_duration_seconds=classify_duration,
            )

            elapsed = time.monotonic() - overall_start
            fail_result = ProcessingResult(
                file_path=file_path,
                ingest_key=ingest_key,
                ingest_run_id=ingest_run_id,
                tenant_id=config.tenant_id,
                parse_result=parse_result,
                classification_result=classification_result,
                embed_result=None,
                classification=classification,
                ingestion_method=IngestionMethod.SQL_AGENT,
                chunks_created=0,
                tables_created=0,
                tables=[],
                written=WrittenArtifacts(),
                errors=[ErrorCode.E_CLASSIFY_INCONCLUSIVE.value],
                warnings=[],
                error_details=[
                    IngestError(
                        code=ErrorCode.E_CLASSIFY_INCONCLUSIVE,
                        message="Classification inconclusive after all tiers. Fail-closed.",
                        stage="classify",
                        recoverable=False,
                    )
                ],
                processing_time_seconds=elapsed,
            )

            # Merge parse errors into result
            self._merge_parse_errors(fail_result, parse_errors)

            logger.info(
                "Processed %s: key=%s tier=%s type=%s confidence=%.2f "
                "path=NONE chunks=0 tables=0 time=%.3fs",
                filename,
                ingest_key[:16],
                tier_used.value,
                classification.file_type.value,
                classification.confidence,
                elapsed,
            )
            return fail_result

        # ----------------------------------------------------------
        # Build ClassificationStageResult
        # ----------------------------------------------------------
        classification_result = ClassificationStageResult(
            tier_used=tier_used,
            file_type=classification.file_type,
            confidence=classification.confidence,
            signals=classification.signals,
            reasoning=classification.reasoning,
            per_sheet_types=classification.per_sheet_types,
            classification_duration_seconds=classify_duration,
        )

        # Log classification escalations
        if tier_used != ClassificationTier.RULE_BASED:
            logger.warning(
                "Classification for %s required escalation to %s.",
                filename,
                tier_used.value,
            )

        # ----------------------------------------------------------
        # Step 7: Route to processor
        # ----------------------------------------------------------
        processor = self._select_processor(classification.file_type)
        processing_path = self._file_type_to_path(classification.file_type)

        logger.info(
            "Routing %s to %s (type=%s, confidence=%.2f).",
            filename,
            processing_path,
            classification.file_type.value,
            classification.confidence,
        )

        # ----------------------------------------------------------
        # Step 8: Process
        # ----------------------------------------------------------
        result = processor.process(
            file_path=file_path,
            profile=profile,
            ingest_key=ingest_key,
            ingest_run_id=ingest_run_id,
            parse_result=parse_result,
            classification_result=classification_result,
            classification=classification,
        )

        # ----------------------------------------------------------
        # Step 9: Merge parse errors and finalize
        # ----------------------------------------------------------
        self._merge_parse_errors(result, parse_errors)

        elapsed = time.monotonic() - overall_start
        # Override the processing_time to include full pipeline time
        result = result.model_copy(
            update={"processing_time_seconds": elapsed}
        )

        # PII-safe INFO log
        logger.info(
            "Processed %s: key=%s tier=%s type=%s confidence=%.2f "
            "path=%s chunks=%d tables=%d time=%.3fs",
            filename,
            ingest_key[:16],
            tier_used.value,
            classification.file_type.value,
            classification.confidence,
            processing_path,
            result.chunks_created,
            result.tables_created,
            elapsed,
        )

        return result

    def process_batch(
        self,
        file_paths: list[str],
    ) -> list[ProcessingResult]:
        """Process multiple Excel files sequentially.

        Parameters
        ----------
        file_paths:
            List of filesystem paths to ``.xlsx`` files.

        Returns
        -------
        list[ProcessingResult]
            One result per input file, in the same order.
        """
        results: list[ProcessingResult] = []
        for fp in file_paths:
            results.append(self.process(fp))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_parse_stage_result(
        profile: FileProfile,
        parse_errors: list[IngestError],
        parse_duration: float,
    ) -> ParseStageResult:
        """Build a ParseStageResult from a FileProfile and parse timing."""
        # Determine primary parser from the first sheet that was parsed
        primary_parser = ParserUsed.OPENPYXL
        if profile.sheets:
            primary_parser = profile.sheets[0].parser_used

        # Compute skipped sheets info
        skipped_reasons: dict[str, str] = {}
        for error in parse_errors:
            if error.code in (
                ErrorCode.W_SHEET_SKIPPED_CHART,
                ErrorCode.W_SHEET_SKIPPED_HIDDEN,
                ErrorCode.W_SHEET_SKIPPED_PASSWORD,
                ErrorCode.W_ROWS_TRUNCATED,
            ) and error.sheet_name:
                skipped_reasons[error.sheet_name] = error.code.value

        # Determine fallback reason if the primary parser is not openpyxl
        fallback_reason_code: str | None = None
        if primary_parser == ParserUsed.PANDAS_FALLBACK:
            fallback_reason_code = ErrorCode.E_PARSE_OPENPYXL_FAIL.value
        elif primary_parser == ParserUsed.RAW_TEXT_FALLBACK:
            fallback_reason_code = ErrorCode.E_PARSE_PANDAS_FAIL.value

        sheets_parsed = len(profile.sheets)
        sheets_skipped = len(skipped_reasons)

        return ParseStageResult(
            parser_used=primary_parser,
            fallback_reason_code=fallback_reason_code,
            sheets_parsed=sheets_parsed,
            sheets_skipped=sheets_skipped,
            skipped_reasons=skipped_reasons,
            parse_duration_seconds=parse_duration,
        )

    @staticmethod
    def _merge_parse_errors(
        result: ProcessingResult,
        parse_errors: list[IngestError],
    ) -> None:
        """Merge parse-stage errors/warnings into the ProcessingResult.

        Errors (E_*) go into ``result.errors``; warnings (W_*) go into
        ``result.warnings``.  All go into ``result.error_details``.
        """
        for error in parse_errors:
            if error.code.value.startswith("E_"):
                if error.code.value not in result.errors:
                    result.errors.append(error.code.value)
            else:
                if error.code.value not in result.warnings:
                    result.warnings.append(error.code.value)
            result.error_details.append(error)

    def _select_processor(
        self, file_type: FileType
    ) -> StructuredDBProcessor | TextSerializer | HybridSplitter:
        """Select the appropriate processor based on file type."""
        if file_type == FileType.TABULAR_DATA:
            return self._structured_db_processor
        elif file_type == FileType.FORMATTED_DOCUMENT:
            return self._text_serializer
        else:
            return self._hybrid_splitter

    @staticmethod
    def _file_type_to_path(file_type: FileType) -> str:
        """Map file type to the processing path name for logging."""
        return {
            FileType.TABULAR_DATA: "Path A (sql_agent)",
            FileType.FORMATTED_DOCUMENT: "Path B (text_serialization)",
            FileType.HYBRID: "Path C (hybrid_split)",
        }.get(file_type, "unknown")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_default_router(**overrides) -> ExcelRouter:
    """Create an ExcelRouter with default SQLite + Ollama backends.

    Convenience factory for local development and testing.  All defaults
    can be overridden via keyword arguments:

    - ``vector_store``: VectorStoreBackend (default: QdrantVectorStore)
    - ``structured_db``: StructuredDBBackend (default: SQLiteStructuredDB)
    - ``llm``: LLMBackend (default: OllamaLLM)
    - ``embedder``: EmbeddingBackend (default: OllamaEmbedding)
    - ``config``: ExcelProcessorConfig (default: ExcelProcessorConfig())

    Any other keyword arguments are passed to ExcelProcessorConfig.

    Returns
    -------
    ExcelRouter
        A fully-configured router ready for ``process()`` calls.

    Raises
    ------
    ImportError
        If optional backend dependencies (e.g. ``httpx``, ``qdrant-client``)
        are not installed.
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
        config = ExcelProcessorConfig(**config_kwargs)
    elif config is None:
        config = ExcelProcessorConfig()

    vector_store = router_kwargs.pop("vector_store", None)
    if vector_store is None:
        if QdrantVectorStore is None:
            raise ImportError(
                "qdrant-client is required for the default vector store. "
                "Install with: pip install qdrant-client"
            )
        vector_store = QdrantVectorStore()

    structured_db = router_kwargs.pop("structured_db", None)
    if structured_db is None:
        structured_db = SQLiteStructuredDB()

    llm = router_kwargs.pop("llm", None)
    if llm is None:
        if OllamaLLM is None:
            raise ImportError(
                "httpx is required for the default LLM backend. "
                "Install with: pip install httpx"
            )
        llm = OllamaLLM()

    embedder = router_kwargs.pop("embedder", None)
    if embedder is None:
        if OllamaEmbedding is None:
            raise ImportError(
                "httpx is required for the default embedding backend. "
                "Install with: pip install httpx"
            )
        embedder = OllamaEmbedding(model=config.embedding_model)

    return ExcelRouter(
        vector_store=vector_store,
        structured_db=structured_db,
        llm=llm,
        embedder=embedder,
        config=config,
    )
