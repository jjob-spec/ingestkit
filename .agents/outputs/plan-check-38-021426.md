---
issue: 38
agent: PLAN-CHECK
date: 2026-02-14
status: PASS
---

# PLAN-CHECK: Issue #38 -- Path A Text Extractor

## Validation Checklist

### 1. SPEC 11.1 Step Coverage

All 10 processing steps from SPEC 11.1 are mapped to PLAN implementation sections:

| SPEC Step | PLAN Section | Covered |
|-----------|-------------|---------|
| 1. `pymupdf4llm.to_markdown()` with `header=False, footer=False` | 1c `_extract_pages()` lines 220-222 | [x] |
| 2. Quality LOW -> block-level fallback via `page.get_text("blocks")` | 1c lines 244-261 | [x] |
| 3. Header/footer stripping via `HeaderFooterDetector` | 1c lines 225-236 (detect), line 264 (strip) | [x] |
| 4. TOC page detection and skip | 1c lines 267-270, 1d `_is_toc_page()` | [x] |
| 5. Blank page detection and skip | 1c lines 273-276, 1e `_is_blank_page()` | [x] |
| 6. Heading hierarchy via `HeadingDetector` + offset conversion | 1b lines 130-135, 1f `_convert_headings_to_offsets()` | [x] |
| 7. Document metadata extraction from `profile.metadata` | 1b lines 146-148 | [x] |
| 8. Chunking via `PDFChunker.chunk()` | 1b lines 153-154 | [x] |
| 9. Embed chunks via `EmbeddingBackend` | 1h `_embed_and_upsert()` lines 437-440 | [x] |
| 10. Upsert to `VectorStoreBackend` with `PDFChunkMetadata` | 1h lines 445-447 | [x] |

### 2. ProcessingResult and PDFChunkMetadata Fields

**ProcessingResult** (verified against models.py lines 297-322):

| Field | PLAN Source | Correct |
|-------|-----------|---------|
| `file_path` | Passed through from `process()` arg | [x] |
| `ingest_key` | Passed through | [x] |
| `ingest_run_id` | Passed through | [x] |
| `tenant_id` | From `config.tenant_id` | [x] |
| `parse_result: ParseStageResult` | Passed in via extended signature (D1) | [x] |
| `classification_result: ClassificationStageResult` | Passed in via extended signature | [x] |
| `ocr_result` | `None` (correct for Path A) | [x] |
| `embed_result: EmbedStageResult` | Constructed in `_embed_and_upsert()` | [x] |
| `classification: ClassificationResult` | Passed in via extended signature | [x] |
| `ingestion_method` | `IngestionMethod.TEXT_EXTRACTION` | [x] |
| `chunks_created` | From `_embed_and_upsert()` return | [x] |
| `tables_created` | Hardcoded `0` (correct for Path A) | [x] |
| `tables` | Hardcoded `[]` (correct for Path A) | [x] |
| `written: WrittenArtifacts` | Tracked during embedding loop | [x] |
| `errors`, `warnings`, `error_details` | Accumulated throughout | [x] |
| `processing_time_seconds` | `time.monotonic()` elapsed | [x] |

**PDFChunkMetadata** (verified against models.py lines 230-249):

| Field | PLAN Source | Correct |
|-------|-----------|---------|
| `source_uri` | `f"file://{Path(file_path).resolve().as_posix()}"` | [x] |
| `source_format` | `"pdf"` (default in model) | [x] |
| `page_numbers` | From chunk dict `cd["page_numbers"]` | [x] |
| `ingestion_method` | `IngestionMethod.TEXT_EXTRACTION.value` | [x] |
| `parser_version` | `config.parser_version` | [x] |
| `chunk_index` | From chunk dict | [x] |
| `chunk_hash` | From chunk dict | [x] |
| `ingest_key` | Passed through | [x] |
| `ingest_run_id` | Passed through | [x] |
| `tenant_id` | `config.tenant_id` | [x] |
| `heading_path` | From chunk dict | [x] |
| `content_type` | From chunk dict | [x] |
| `doc_title` | From `profile.metadata.title` | [x] |
| `doc_author` | From `profile.metadata.author` | [x] |
| `doc_date` | From `profile.metadata.creation_date` | [x] |
| `language` | From language detection or default | [x] |

Note: OCR-specific fields (`ocr_engine`, `ocr_confidence`, `ocr_dpi`, `ocr_preprocessing`) are correctly omitted (default to `None` for Path A).

### 3. Utility Dependency Wiring

| Utility | Constructor | Method Calls | Correct |
|---------|------------|-------------|---------|
| `HeaderFooterDetector(config)` | [x] Matches `header_footer.py:36` | `detect(doc)` -> `strip(text, page_number, headers, footers)` per page | [x] |
| `HeadingDetector(config)` | [x] Matches `heading_detector.py:37` | `detect(doc)` returns `(level, title, page_number)` tuples | [x] |
| `PDFChunker(config)` | [x] Matches `chunker.py:195` | `chunk(text, headings, page_boundaries)` with `(level, title, char_offset)` headings | [x] |
| `QualityAssessor(config)` | [x] Matches `quality.py:21` | `assess_page(text, page_number)`, `needs_ocr_fallback(quality)` | [x] |
| `detect_language(text, default_language=...)` | N/A (function) | Returns `tuple[str, float]` | [x] |

**Heading offset conversion** (MAP D4): PLAN section 1f correctly converts `(level, title, page_number)` from `HeadingDetector` to `(level, title, char_offset)` for `PDFChunker`. The revised flow in section 1g correctly builds `page_offset_map` during concatenation.

### 4. Test Coverage

37 tests across 12 classes covering:

| Area | Tests | Steps/Features Covered |
|------|-------|----------------------|
| Constructor | 2 | DI pattern |
| Happy path | 5 | Steps 1-10 end-to-end, result correctness |
| TOC detection | 3 | Step 4, threshold boundary |
| Blank detection | 3 | Step 5, empty/whitespace/low-word-count |
| Header/footer | 2 | Step 3, graceful error handling |
| Quality fallback | 3 | Step 2, block fallback, OCR warning |
| Heading detection | 2 | Step 6, offset conversion |
| Chunk metadata | 4 | Step 7, all PDFChunkMetadata fields, tenant_id, language |
| Embedding/upsert | 4 | Steps 9-10, batch size, ensure_collection |
| Error handling | 4 | Backend error classification, batch continuation |
| Result assembly | 3 | Empty document, field passthrough, tables always empty |
| Language detection | 2 | Enabled/disabled paths |

All tests use `@pytest.mark.unit` with mocked `fitz`, `pymupdf4llm`, and backends. No real PDFs or external services needed.

**SPEC 22.3 coverage check** -- the SPEC lists: markdown extraction, heading hierarchy, header/footer stripping, TOC skipping, chunk metadata correctness. All are covered by the test plan.

### 5. Scope Creep Check

- [x] No Path B (OCR) logic included
- [x] No Path C (complex) logic included
- [x] No Router implementation
- [x] No ROADMAP items introduced
- [x] No concrete backends inside ingestkit_pdf
- [x] No ABC base classes -- uses Protocol throughout
- [x] Files limited to 2 created + 2 modified (matches MAP inventory)

## Issues Found

**ISSUE-1 (LOW): `_concatenate_pages()` return type inconsistency.** PLAN section 1g first defines the method returning `tuple[str, list[int]]`, then the "Important" note and "Revised flow" change it to also return `dict[int, int]` for the page offset map. The revised version is correct. PATCH agent should implement the revised three-return-value version directly and skip the first definition.

**ISSUE-2 (LOW): `_extract_pages()` returns `page_boundaries` as `list(page_texts.keys())`.** This is a list of page numbers, not character offsets. The actual page boundary offsets needed by the chunker are computed later in `_concatenate_pages()`. The naming in the return type (`page_boundaries`) is misleading since it is really `page_numbers`. PATCH agent should rename or clarify. This does not affect correctness since the values are only used to build the offset map later.

**ISSUE-3 (INFO): Process signature extends SPEC.** The PLAN extends the SPEC's `process(file_path, profile, ingest_key, ingest_run_id)` signature with three additional parameters (`parse_result`, `classification_result`, `classification`). This is justified by MAP Decision D1 and follows the Excel `StructuredDBProcessor` pattern. The Router (future issue) will supply these. No action needed.

**ISSUE-4 (INFO): `fitz.open()` called twice.** The PLAN opens the PDF in `_extract_pages()` (for pymupdf4llm extraction and header/footer detection) and again in `process()` (for heading detection). This is intentional -- the first context manager closes the doc before heading detection runs. PATCH agent could optimize by extracting headings inside `_extract_pages()`, but the current approach is correct and simpler to reason about. No blocker.

## Verdict

**PASS** -- The PLAN covers all 10 SPEC 11.1 steps, correctly populates all `ProcessingResult` and `PDFChunkMetadata` fields, properly wires all utility dependencies with correct API signatures, provides comprehensive test coverage (37 tests across all steps and edge cases), and introduces no scope creep. The two LOW issues are naming/documentation concerns that do not affect correctness.

AGENT_RETURN: .agents/outputs/plan-check-38-021426.md
