---
issue: 39
agent: PLAN-CHECK
date: 2026-02-14
complexity: COMPLEX
stack: [python, pydantic, pytest, ocr, multiprocessing]
---

# PLAN-CHECK: Issue #39 -- Path B OCR Processor

**Plan artifact**: `.agents/outputs/plan-39-021426.md`
**MAP artifact**: `.agents/outputs/map-39-021426.md`
**Issue**: #39 -- Implement Path B OCR Processor (processors/ocr_processor.py)
**Date**: 2026-02-14

---

## Executive Summary

The plan is **APPROVED with corrections**. All 9 SPEC steps are covered. The worker function design is correct (module-level, primitive args, serializable return). Per-page error isolation is properly designed. Test coverage is comprehensive (16 test classes, 40+ cases). Three corrections are required before PATCH proceeds.

---

## Check 1: SPEC 11.2 Step Coverage

| SPEC Step | Plan Location | Status |
|-----------|--------------|--------|
| 1. Render page to high-DPI image | Worker steps 2-3 (`render_page`, `preprocess`) | Pass |
| 2. Preprocess images (deskew/denoise/binarize/contrast) | Worker step 4 (`renderer.preprocess()`) | Pass |
| 3. Detect language per page | Worker step 5 (`detect_language()`) | Pass |
| 4. Run OCR via configured engine | Worker steps 6-7 (`create_ocr_engine()`, `engine.recognize()`) | Pass |
| 5. Collect per-page OCRResult | Worker step 9, `_ocr_pages()` result collection | Pass |
| 6. Postprocess OCR text | Worker step 8 (`postprocess_ocr_text()`) | Pass |
| 7. Optional LLM cleanup | `_llm_cleanup()` method | Pass |
| 8. Flag low-confidence pages | `_flag_low_confidence()` method | Pass |
| 9. Heading detection -> chunking -> embedding -> upsert | `_chunk_and_embed()` sub-steps 9a-9g | Pass |

All 9 processing steps are covered.

---

## Check 2: ProcessPoolExecutor Parallelism Design

| Aspect | Assessment | Status |
|--------|-----------|--------|
| Worker is module-level function | `_ocr_single_page()` at module scope | Pass |
| Args are all primitives/serializable | file_path(str), page_number(int), ocr_dpi(int), etc. | Pass |
| No unpicklable objects passed | Config reconstructed from primitives inside worker | Pass |
| Each worker creates own OCR engine | `create_ocr_engine()` called inside worker | Pass |
| fitz.Page NOT passed to worker | Worker opens PDF from file_path | Pass |
| PIL Images stay in worker | Render + OCR inside worker, only OCRResult returned | Pass |
| Sequential fallback | `ocr_max_workers <= 1` or single page | Pass |

**CORRECTION REQUIRED -- `as_completed()` timeout semantics:**

The plan states: "Uses `as_completed()` with per-page timeout `config.ocr_per_page_timeout_seconds`". This is **incorrect**. `concurrent.futures.as_completed(fs, timeout=T)` applies timeout `T` to the **entire iteration** (time from the call to `as_completed`), NOT per-future. If 10 pages are submitted and each takes 50s with a 60s per-page timeout, the iteration would time out at 60s total, not 600s.

**Fix:** Use `future.result(timeout=config.ocr_per_page_timeout_seconds)` on each future returned by `as_completed()`. The `as_completed()` call itself should have no timeout (or a generous total timeout like `ocr_per_page_timeout_seconds * len(pages)`). The per-page timeout is enforced when calling `.result()` on individual futures.

---

## Check 3: Worker Function Picklability

| Requirement | Status |
|-------------|--------|
| Module-level function (not method/lambda/closure) | Pass |
| All arguments are primitive types | Pass -- str, int, list[str], bool |
| Return type is serializable | Pass -- OCRResult (Pydantic v2, picklable) or tuple[int, str] |
| No references to unpicklable objects | Pass -- no config object, no fitz objects in args |
| Config reconstructed inside worker | Pass -- `PDFProcessorConfig(...)` from primitives |

Note: Pydantic v2 `BaseModel` instances are picklable. `OCRResult` returning across process boundaries is valid.

---

## Check 4: Per-Page Error Isolation

| Scenario | Handling | Status |
|----------|---------|--------|
| Single page OCR failure | Worker returns `(page_number, error_str)`, parent records `E_OCR_FAILED`, continues | Pass |
| Single page timeout | `TimeoutError` on `future.result()`, records `E_OCR_TIMEOUT`, continues | Pass (with fix from Check 2) |
| All pages fail | Zero OCRResults, zero chunks, ProcessingResult with errors | Pass |
| Worker crashes (unhandled exception) | `future.result()` raises, caught as `E_OCR_FAILED` | Pass |
| LLM cleanup failure | try/except per page, keeps original text | Pass |
| Embed/upsert failure | try/except per batch, records backend error code | Pass |

---

## Check 5: OCR Config Fields Usage

| Config Field | Used In Plan | Matches `config.py` Default | Status |
|-------------|-------------|----------------------------|--------|
| `ocr_engine` | Engine selection in worker | `OCREngine.TESSERACT` | Pass |
| `ocr_dpi` | Worker arg, metadata | `300` | Pass |
| `ocr_language` | Worker arg | `"en"` | Pass |
| `ocr_confidence_threshold` | `_flag_low_confidence()` | `0.7` | Pass |
| `ocr_preprocessing_steps` | Worker arg, metadata | `["deskew"]` | Pass |
| `ocr_max_workers` | Sequential vs parallel decision | `4` | Pass |
| `ocr_per_page_timeout_seconds` | Per-page timeout | `60` | Pass |
| `enable_ocr_cleanup` | `_llm_cleanup()` gate | `False` | Pass |
| `ocr_cleanup_model` | LLM generate model arg | `"qwen2.5:7b"` | Pass |
| `enable_language_detection` | Worker arg | `True` | Pass |
| `default_language` | Worker arg fallback | `"en"` | Pass |
| `embedding_batch_size` | Batch embed loop | `64` | Pass |
| `backend_timeout_seconds` | Backend call timeout | `30.0` | Pass |
| `default_collection` | Vector store collection | `"helpdesk"` | Pass |
| `chunk_size_tokens` | Chunker config | `512` | Pass |
| `chunk_overlap_tokens` | Chunker config | `50` | Pass |
| `log_ocr_output` | PII-safe logging | `False` | Pass |
| `log_sample_text` | PII-safe logging | `False` | Pass |
| `parser_version` | Metadata | `"ingestkit_pdf:1.0.0"` | Pass |
| `tenant_id` | Metadata propagation | `None` | Pass |

All relevant config fields are accounted for.

---

## Check 6: Test Coverage Assessment

| Area | Test Classes | Key Cases | Status |
|------|-------------|-----------|--------|
| Worker function | TestOCRSinglePageWorker (4 cases) | Success, failure, lang on/off | Pass |
| Constructor | TestOCRProcessorInit (2 cases) | Backends, None LLM | Pass |
| Page filtering | TestPageFiltering (6 cases) | BLANK/TOC/VECTOR_ONLY skip, explicit pages | Pass |
| Single page E2E | TestSinglePageOCR (3 cases) | Full pipeline, metadata, source_uri | Pass |
| Multi page | TestMultiPageOCR (2 cases) | Boundaries, metadata propagation | Pass |
| Parallel execution | TestSequentialVsParallel (3 cases) | Sequential/parallel selection | Pass |
| Per-page errors | TestPerPageErrorIsolation (4 cases) | Failure, timeout, all-fail | Pass |
| Low confidence | TestLowConfidenceWarning (3 cases) | Below/above threshold, stage result | Pass |
| LLM cleanup | TestLLMCleanup (5 cases) | Enabled/disabled, None LLM, failure, model | Pass |
| Stage result | TestOCRStageResult (3 cases) | Fields, fallback, avg confidence | Pass |
| Header/footer | TestHeaderFooterStripping (2 cases) | Strip, no patterns | Pass |
| Headings | TestHeadingDetection (2 cases) | From outline, empty | Pass |
| Batch embed | TestBatchEmbedding (2 cases) | Batching, ensure_collection | Pass |
| Processing result | TestProcessingResult (3 cases) | Assembly, ingestion method, time | Pass |
| Engine unavailable | TestEngineUnavailable (1 case) | E_OCR_ENGINE_UNAVAILABLE | Pass |
| Empty output | TestEmptyOCROutput (2 cases) | All empty, no pages | Pass |

**Total: 16 test classes, ~45 test cases.** Coverage is comprehensive.

**Missing test case (minor):** No explicit test for `_classify_backend_error()` helper. This is a low-priority gap since the method is tested indirectly via embed/upsert failure tests.

---

## Corrections Required

### CORRECTION 1 (MUST FIX): `as_completed()` timeout semantics

**Problem:** Plan says "Uses `as_completed()` with per-page timeout". `as_completed(timeout=T)` applies `T` to the entire iteration, not per future.

**Fix:** In `_ocr_pages_parallel()`, call `as_completed()` without timeout (or with `total_timeout = ocr_per_page_timeout_seconds * len(pages)`). Apply per-page timeout on each `future.result(timeout=config.ocr_per_page_timeout_seconds)`.

### CORRECTION 2 (SHOULD FIX): `process()` signature deviation from SPEC

**Problem:** SPEC 11.2 defines `process(self, file_path, profile, pages, ingest_key, ingest_run_id)` with 5 parameters. Plan adds `parse_result`, `classification_result`, `classification` (3 extra). This is justified because `ProcessingResult` requires these as non-optional fields (verified: `models.py:305-310`), and the Excel StructuredDBProcessor uses the same expanded signature.

**Action:** This is a correct deviation. Document in the PATCH that the SPEC public interface is incomplete -- the router (caller) must provide these upstream stage results. No code change needed; the plan's decision is sound.

### CORRECTION 3 (MINOR): MockLLMBackend.generate() response type

**Problem:** Plan proposes `generate()` returns `str(response)` from the response queue, but `classify()` returns `dict`. The plan correctly uses separate response types, but the existing `_responses: list[dict | Exception]` type hint in conftest.py would need updating to `list[dict | str | Exception]` to support both `classify()` (dict) and `generate()` (str) responses from the same queue.

**Fix:** Update `_responses` type hint to `list[dict | str | Exception]` when modifying `MockLLMBackend`.

---

## Scope Containment

| Check | Status |
|-------|--------|
| No concrete backends inside ingestkit_pdf | Pass -- uses Protocol types |
| No ABC base classes | Pass -- structural subtyping |
| No ROADMAP items implemented | Pass |
| tenant_id propagated through metadata | Pass -- PDFChunkMetadata.tenant_id from config |
| PII-safe logging checked | Pass -- log_ocr_output gate |

---

## Verdict

**APPROVED with corrections.** Correction 1 (timeout semantics) is a must-fix for PATCH. Corrections 2-3 are documentation/minor type hint fixes. The plan is otherwise thorough, well-structured, and ready for implementation.

AGENT_RETURN: .agents/outputs/plan-check-39-021426.md
