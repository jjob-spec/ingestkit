"""Path A processor: text-native PDF extraction via pymupdf4llm.

Extracts markdown from text-native PDFs (Type A) through a 10-step pipeline:
text extraction, quality assessment with block-level fallback, header/footer
stripping, TOC/blank page filtering, heading detection, metadata extraction,
chunking, embedding, and vector store upsert.

Spec reference: SPEC.md section 11.1.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from ingestkit_core.models import ChunkPayload, EmbedStageResult, WrittenArtifacts
from ingestkit_pdf.config import PDFProcessorConfig
from ingestkit_pdf.errors import ErrorCode, IngestError
from ingestkit_pdf.models import (
    ClassificationResult,
    ClassificationStageResult,
    DocumentProfile,
    ExtractionQualityGrade,
    IngestionMethod,
    ParseStageResult,
    PDFChunkMetadata,
    ProcessingResult,
)
from ingestkit_pdf.quality import QualityAssessor
from ingestkit_pdf.utils.chunker import PDFChunker
from ingestkit_pdf.utils.header_footer import HeaderFooterDetector
from ingestkit_pdf.utils.heading_detector import HeadingDetector
from ingestkit_pdf.utils.language import detect_language

import fitz  # type: ignore[import-untyped]
import pymupdf4llm  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from ingestkit_core.protocols import EmbeddingBackend, VectorStoreBackend

logger = logging.getLogger("ingestkit_pdf.processors.text_extractor")

_TOC_PATTERN = re.compile(r"\.{2,}\s*\d+\s*$")


class TextExtractor:
    """Path A processor: text-native PDF extraction via pymupdf4llm."""

    def __init__(
        self,
        vector_store: VectorStoreBackend,
        embedder: EmbeddingBackend,
        config: PDFProcessorConfig,
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
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
        """Process a text-native PDF through the 10-step extraction pipeline.

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

        # Steps 1-5: Extract text per page (with quality fallback, h/f strip, TOC/blank skip)
        page_texts = self._extract_pages(
            file_path, config, warnings, error_details,
        )

        # Early return if no usable text
        if not page_texts:
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
                ingestion_method=IngestionMethod.TEXT_EXTRACTION,
                chunks_created=0,
                tables_created=0,
                tables=[],
                written=WrittenArtifacts(vector_collection=collection),
                errors=errors,
                warnings=warnings,
                error_details=error_details,
                processing_time_seconds=elapsed,
            )

        # Step 6: Heading detection
        with fitz.open(file_path) as doc:
            heading_detector = HeadingDetector(config)
            raw_headings = heading_detector.detect(doc)  # (level, title, page_number)

        # Concatenate page texts and compute page boundary offsets
        full_text, page_boundary_list, page_offset_map = self._concatenate_pages(
            page_texts,
        )

        # Convert page-number-based headings to char-offset-based headings
        headings = self._convert_headings_to_offsets(raw_headings, page_offset_map)

        # Step 7: Language detection
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

        # Step 8: Chunking
        chunker = PDFChunker(config)
        chunk_dicts = chunker.chunk(full_text, headings, page_boundary_list)

        # Steps 9-10: Embed and upsert
        embed_result, chunk_count = self._embed_and_upsert(
            chunk_dicts=chunk_dicts,
            source_uri=source_uri,
            ingest_key=ingest_key,
            ingest_run_id=ingest_run_id,
            collection=collection,
            doc_title=doc_title,
            doc_author=doc_author,
            doc_date=doc_date,
            language=language,
            written=written,
            errors=errors,
            warnings=warnings,
            error_details=error_details,
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
            ingestion_method=IngestionMethod.TEXT_EXTRACTION,
            chunks_created=chunk_count,
            tables_created=0,
            tables=[],
            written=written,
            errors=errors,
            warnings=warnings,
            error_details=error_details,
            processing_time_seconds=elapsed,
        )

    # ------------------------------------------------------------------
    # Steps 1-5: Page extraction
    # ------------------------------------------------------------------

    def _extract_pages(
        self,
        file_path: str,
        config: PDFProcessorConfig,
        warnings: list[str],
        error_details: list[IngestError],
    ) -> dict[int, str]:
        """Extract and clean text per page (steps 1-5).

        Returns a dict mapping 1-based page number to cleaned text.
        Pages skipped (TOC, blank) are excluded.
        """
        page_texts: dict[int, str] = {}
        quality_assessor = QualityAssessor(config)

        with fitz.open(file_path) as doc:
            # Step 1: Extract markdown per page
            page_chunks = pymupdf4llm.to_markdown(
                doc, page_chunks=True, header=False, footer=False,
            )

            # Step 3: Header/footer detection (needs full doc)
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

            for chunk in page_chunks:
                metadata = chunk.get("metadata", {})
                page_number = metadata.get("page", 0) + 1  # pymupdf4llm uses 0-based
                text = chunk.get("text", "")

                # Step 2: Quality assessment with block-level fallback
                page_quality = quality_assessor.assess_page(text, page_number - 1)
                if (
                    quality_assessor.needs_ocr_fallback(page_quality)
                    and config.auto_ocr_fallback
                ):
                    warnings.append(ErrorCode.W_QUALITY_LOW_NATIVE.value)
                    logger.info("Low quality on page %d, attempting block fallback", page_number)

                    # Block-level fallback (MAP D5)
                    page_obj = doc[page_number - 1]
                    blocks = page_obj.get_text("blocks")
                    block_texts = [
                        b[4].strip()
                        for b in blocks
                        if b[6] == 0 and b[4].strip()  # type 0 = text
                    ]
                    fallback_text = "\n".join(block_texts)

                    # Re-assess quality of fallback text
                    fb_quality = quality_assessor.assess_page(
                        fallback_text, page_number - 1,
                    )
                    if fb_quality.grade != ExtractionQualityGrade.LOW:
                        text = fallback_text
                    else:
                        warnings.append(ErrorCode.W_OCR_FALLBACK.value)
                        # Keep original text even if still low quality

                    if config.log_sample_text:
                        logger.debug(
                            "Page %d text sample: %s...", page_number, text[:100],
                        )

                # Step 3 (continued): Strip headers/footers from this page
                text = hf_detector.strip(
                    text, page_number, header_patterns, footer_patterns,
                )

                # Step 4: TOC page detection
                if self._is_toc_page(text):
                    warnings.append(ErrorCode.W_PAGE_SKIPPED_TOC.value)
                    logger.info("Skipping TOC page %d", page_number)
                    continue

                # Step 5: Blank page detection
                if self._is_blank_page(text, config):
                    warnings.append(ErrorCode.W_PAGE_SKIPPED_BLANK.value)
                    logger.info("Skipping blank page %d", page_number)
                    continue

                page_texts[page_number] = text

        return page_texts

    # ------------------------------------------------------------------
    # TOC / Blank detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_toc_page(text: str) -> bool:
        """Return True if the page appears to be a Table of Contents.

        A page is TOC if >30% of non-empty lines match dot-leader + page-number
        patterns (e.g. "Chapter 1 .............. 42").
        """
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 3:
            return False
        toc_lines = sum(1 for line in lines if _TOC_PATTERN.search(line))
        return toc_lines / len(lines) > 0.3

    @staticmethod
    def _is_blank_page(text: str, config: PDFProcessorConfig) -> bool:
        """Return True if the page is effectively blank."""
        stripped = text.strip()
        if not stripped:
            return True
        return len(stripped.split()) < config.quality_min_words_per_page

    # ------------------------------------------------------------------
    # Heading offset conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_headings_to_offsets(
        raw_headings: list[tuple[int, str, int]],
        page_offset_map: dict[int, int],
    ) -> list[tuple[int, str, int]]:
        """Map page-number-based headings to character-offset-based headings.

        Args:
            raw_headings: ``(level, title, page_number)`` from HeadingDetector.
            page_offset_map: Mapping of page_number -> char_offset in the
                concatenated document text.

        Returns:
            ``(level, title, char_offset)`` tuples for PDFChunker.
        """
        result: list[tuple[int, str, int]] = []
        for level, title, page_num in raw_headings:
            char_offset = page_offset_map.get(page_num, 0)
            result.append((level, title, char_offset))
        return result

    # ------------------------------------------------------------------
    # Page concatenation
    # ------------------------------------------------------------------

    @staticmethod
    def _concatenate_pages(
        page_texts: dict[int, str],
    ) -> tuple[str, list[int], dict[int, int]]:
        """Concatenate page texts and compute page boundary offsets.

        Returns:
            - full_text: The concatenated document text.
            - page_boundary_list: Character offsets where each page starts
              (ordered list for the chunker).
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
    # Steps 9-10: Embed and upsert
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
        written: WrittenArtifacts,
        errors: list[str],
        warnings: list[str],
        error_details: list[IngestError],
    ) -> tuple[EmbedStageResult | None, int]:
        """Embed chunks and upsert to vector store (steps 9-10).

        Returns:
            Tuple of (embed_result, total_chunks_upserted).
        """
        config = self._config
        total_chunks = 0
        total_texts_embedded = 0
        embed_duration = 0.0

        # Ensure collection exists
        vector_size = self._embedder.dimension()
        self._vector_store.ensure_collection(collection, vector_size)

        # Build ChunkPayload list (vectors empty initially)
        payloads: list[ChunkPayload] = []
        for cd in chunk_dicts:
            chunk_hash = cd["chunk_hash"]
            chunk_id = str(
                uuid.uuid5(uuid.NAMESPACE_URL, f"{ingest_key}:{chunk_hash}")
            )

            metadata = PDFChunkMetadata(
                source_uri=source_uri,
                source_format="pdf",
                page_numbers=cd["page_numbers"],
                ingestion_method=IngestionMethod.TEXT_EXTRACTION.value,
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
            )
            payloads.append(
                ChunkPayload(
                    id=chunk_id,
                    text=cd["text"],
                    vector=[],  # placeholder
                    metadata=metadata,
                )
            )

        if config.log_chunk_previews:
            logger.debug("Built %d chunk payloads for embedding", len(payloads))

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
                continue  # skip this batch, try next

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
        """Map an exception to the most appropriate ErrorCode."""
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
