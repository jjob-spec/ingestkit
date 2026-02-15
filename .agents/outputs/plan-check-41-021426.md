---
issue: 41
agent: PLAN-CHECK
date: 2026-02-14
status: CONDITIONAL PASS
plan_artifact: plan-41-021426.md
map_artifact: map-41-021426.md
---

# PLAN-CHECK: Issue #41 -- Implement Path C Complex Processor

## Verdict: CONDITIONAL PASS (2 blocking issues, 2 warnings)

The plan is thorough and well-structured. It correctly identifies all 8 PageType routes, reuses existing utilities, and covers ProcessingResult assembly. Two issues must be fixed before PATCH proceeds.

---

## Blocking Issues

### B1: `process()` parameter order mismatch with sibling processors

The plan proposes:
```python
def process(self, file_path, profile, classification, ingest_key, ingest_run_id, parse_result, classification_result)
```

Both sibling processors place `classification` as the **last** parameter:
- **TextExtractor**: `(file_path, profile, ingest_key, ingest_run_id, parse_result, classification_result, classification)`
- **OCRProcessor**: `(file_path, profile, pages, ingest_key, ingest_run_id, parse_result, classification_result, classification)`

**Fix**: Move `classification` to the last position. ComplexProcessor does not need a `pages` parameter (it routes internally via `classification.per_page_types`), so the signature should be:
```python
def process(self, file_path, profile, ingest_key, ingest_run_id, parse_result, classification_result, classification)
```

### B2: `_ocr_single_page()` requires 8 primitive parameters, not just (doc, page_number)

The plan describes calling `_ocr_single_page()` but does not show the full calling convention. The actual signature is:
```python
_ocr_single_page(file_path, page_number, ocr_dpi, preprocessing_steps, ocr_engine_name, ocr_language, enable_language_detection, default_language)
```

It accepts primitives (not a config object) because it is designed for `ProcessPoolExecutor` pickling. The plan must pass all 8 args from `self._config` when calling this function. Additionally, it returns `OCRResult | tuple[int, str]` (not just `OCRResult`), so the caller must handle the error tuple case.

**Fix**: PATCH must unpack config primitives when calling `_ocr_single_page()` and handle the `tuple[int, str]` error return.

---

## Warnings

### W1: `TableExtractionResult.errors` is `list[IngestError]`, not `list[str]`

Plan step 8 says to accumulate `table_result.errors` into the main `errors` list. However, `TableExtractionResult.errors` is `list[IngestError]` while `ProcessingResult.errors` is `list[str]` and `ProcessingResult.error_details` is `list[IngestError]`. PATCH should route `table_result.errors` to `error_details`, and convert error codes/messages to strings for `errors`.

### W2: Multi-column reorder uses `profile.pages[page_idx].is_multi_column` but plan also calls `layout_analyzer.detect_columns()`

The plan (step 6) checks `profile.pages[page_idx].is_multi_column` AND runs `layout_analyzer.detect_columns()`. Since the profile already has the flag, the detect_columns call is redundant for gating. PATCH should either: (a) trust the profile flag and only call `reorder_blocks()`, or (b) ignore the profile flag and rely solely on `detect_columns()`. Doing both wastes cycles.

---

## Acceptance Criteria Coverage

| Criterion | Planned Task | Status |
|-----------|-------------|--------|
| ComplexProcessor class created with SPEC 11.3 constructor | Task 1b | OK -- constructor matches SPEC exactly |
| process() expanded 7-param signature | Task 1b | NEEDS FIX (B1: param order) |
| Page routing for all 8 PageType values | Task 1b step 6 | OK |
| TEXT pages via pymupdf4llm | Task 1b _extract_text_page | OK |
| SCANNED pages via _ocr_single_page | Task 1b step 7 | NEEDS FIX (B2: calling convention) |
| TABLE_HEAVY pages via TableExtractor.extract_tables() | Task 1b step 8 | OK -- signature matches actual API |
| FORM pages via page.widgets() | Task 1b _extract_form_fields | OK -- correct API usage |
| MIXED pages native + OCR | Task 1b _extract_mixed_page | OK |
| BLANK/TOC/VECTOR_ONLY skip with warnings | Task 1b step 6 | OK -- codes exist in errors.py |
| Multi-column reorder via LayoutAnalyzer | Task 1b _apply_layout_reorder | OK (see W2) |
| Header/footer detection + stripping | Task 1b steps 3, 6 | OK -- API matches actual code |
| Heading detection on full document | Task 1b step 4 | OK -- API matches actual code |
| Text assembled, chunked, embedded, upserted | Task 1b steps 9-14 | OK |
| ProcessingResult with COMPLEX_PROCESSING | Task 1b step 17 | OK |
| OCRStageResult when SCANNED pages present | Task 1b step 15 | OK |
| Exported from processors/__init__.py | Task 2 | OK |
| Tests with mocked backends | Task 3 | OK -- mock strategy is sound |
| Per-page recoverable errors | Task 1b _classify_backend_error | OK |

---

## API Signature Verification

| Component | Plan API | Actual API | Match |
|-----------|----------|------------|-------|
| TextExtractor.__init__ | (vector_store, embedder, config) | (vector_store, embedder, config) | OK |
| OCRProcessor.__init__ | (vector_store, embedder, llm, config) | (vector_store, embedder, llm, config) | OK |
| TableExtractor.__init__ | (config, structured_db, vector_store, embedder) | (config, structured_db=None, vector_store=None, embedder=None) | OK |
| TableExtractor.extract_tables | (file_path, page_numbers, ingest_key, ingest_run_id) | (file_path, page_numbers, ingest_key, ingest_run_id) | OK |
| HeaderFooterDetector.detect | (doc) -> (headers, footers) | (doc) -> tuple[list[str], list[str]] | OK |
| HeaderFooterDetector.strip | (text, page_number, headers, footers) | (text, page_number, headers, footers) | OK |
| HeadingDetector.detect | (doc) -> list[tuple] | (doc) -> list[tuple[int, str, int]] | OK |
| LayoutAnalyzer.detect_columns | (page) -> LayoutResult | (page) -> LayoutResult | OK |
| LayoutAnalyzer.reorder_blocks | (blocks, layout) -> list[TextBlock] | (blocks, layout) -> list[TextBlock] | OK |
| extract_text_blocks | (page) -> list[TextBlock] | (page) -> list[TextBlock] | OK |
| PDFChunker.chunk | (text, headings, page_boundaries) | (text, headings, page_boundaries) -> list[dict] | OK |
| _ocr_single_page | (file_path, page_number) implied | 8 primitive params, returns OCRResult or tuple | MISMATCH (B2) |

---

## Enum/Error Code Verification

All referenced enums and error codes exist in the source:

- `PageType`: TEXT, SCANNED, TABLE_HEAVY, FORM, MIXED, BLANK, VECTOR_ONLY, TOC -- all 8 confirmed in `models.py:48-58`
- `IngestionMethod.COMPLEX_PROCESSING` -- confirmed in `models.py:66`
- `ContentType.FORM_FIELD` -- confirmed in `models.py:91`
- `ErrorCode.W_PAGE_SKIPPED_BLANK` -- confirmed in `errors.py:58`
- `ErrorCode.W_PAGE_SKIPPED_TOC` -- confirmed in `errors.py:59`
- `ErrorCode.W_PAGE_SKIPPED_VECTOR_ONLY` -- confirmed in `errors.py:60`
- `ErrorCode.E_PROCESS_HEADER_FOOTER` -- confirmed in `errors.py:55`
- `ErrorCode.E_BACKEND_EMBED_TIMEOUT` -- confirmed in `errors.py:49`
- `ErrorCode.E_BACKEND_VECTOR_TIMEOUT` -- confirmed in `errors.py:45`
- `ErrorCode.E_BACKEND_EMBED_CONNECT` -- confirmed in `errors.py:50`
- `ErrorCode.E_BACKEND_VECTOR_CONNECT` -- confirmed in `errors.py:46`
- `ErrorCode.E_PROCESS_CHUNK` -- confirmed in `errors.py:54`

---

## Scope Check

- No scope creep detected. Plan stays within SPEC 11.3 boundaries.
- No ROADMAP items are being implemented.
- No new backend implementations inside the package.
- Form field extraction uses `page.widgets()` which is the correct PyMuPDF API.
- Files created/modified (3 total) is appropriate for the COMPLEX classification.

---

## Recommendation

Fix **B1** (parameter order) and **B2** (full _ocr_single_page calling convention with error handling) before proceeding to PATCH. Warnings W1 and W2 should be addressed during PATCH but are not blocking.

AGENT_RETURN: .agents/outputs/plan-check-41-021426.md
