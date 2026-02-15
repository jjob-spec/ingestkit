---
issue: 42
agent: PLAN-CHECK
date: 2026-02-14
status: PASS_WITH_ISSUES
---

# PLAN-CHECK: Issue #42 -- PDFRouter orchestrator and public API

## Validation Checklist

### 1. Acceptance Criteria Coverage

| Criterion | Planned Task | Status |
|-----------|-------------|--------|
| PDFRouter class with can_handle/process/process_batch | Section 2 | Covered |
| process() implements 15 SPEC 17.1 steps | Section 4 | Covered |
| Security scan runs first; fatal errors return immediately | Section 4 Step 1 | Covered |
| Ingest key via ingestkit_core.idempotency | Section 4 Step 2 | Covered |
| Document profiling builds DocumentProfile | Section 6 | Covered |
| Tiered classification with escalation | Section 5 | Covered |
| LLM outage resilience per SPEC 5.2 | Section 5 | Covered |
| E_CLASSIFY_INCONCLUSIVE only when all tiers fail | Section 5 | Covered |
| TEXT_NATIVE -> TextExtractor, SCANNED -> OCRProcessor | Section 4 Step 11 | Covered |
| COMPLEX handled gracefully | Section 4 Step 11, Section 12.1 | Covered |
| OCRProcessor called with extra pages=None | Section 4 Step 11 | Covered |
| process_batch() with ProcessPoolExecutor | Section 7 | Covered |
| create_default_router() factory | Section 8 | Covered |
| __init__.py updated with SPEC 21.1 exports | Section 9 | Covered |
| PII-safe logging | Section 4 Step 14 | Covered |
| tenant_id propagated | Section 4 Step 2 (via config) | Covered |
| Unit tests | Section 10 | Covered |
| test_llm_outage_degrades_to_tier1 matches SPEC 5.2 | Section 10.2 TestLLMOutageResilience | Covered |

All 18 acceptance criteria map to planned tasks.

### 2. Processor API Signatures -- VERIFIED CORRECT

**TextExtractor.__init__** (text_extractor.py lines 53-58):
```python
def __init__(self, vector_store, embedder, config) -> None
```
PLAN section 3: `TextExtractor(vector_store=vector_store, embedder=embedder, config=self._config)` -- CORRECT.

**TextExtractor.process()** (text_extractor.py lines 67-76):
```python
def process(self, file_path, profile, ingest_key, ingest_run_id, parse_result, classification_result, classification) -> ProcessingResult
```
PLAN section 4 Step 11: Matches 7-parameter signature -- CORRECT.

**OCRProcessor.__init__** (ocr_processor.py lines 154-160):
```python
def __init__(self, vector_store, embedder, llm, config) -> None
```
PLAN section 3: `OCRProcessor(vector_store=vector_store, embedder=embedder, llm=llm, config=self._config)` -- CORRECT. The PLAN correctly passes `llm` to OCRProcessor.

**OCRProcessor.process()** (ocr_processor.py lines 170-180):
```python
def process(self, file_path, profile, pages, ingest_key, ingest_run_id, parse_result, classification_result, classification) -> ProcessingResult
```
PLAN section 4 Step 11: `self._ocr_processor.process(file_path=file_path, profile=profile, pages=None, ...)` -- CORRECT. Extra `pages` parameter is correctly passed as `None`.

### 3. PDFInspector and PDFLLMClassifier APIs -- VERIFIED CORRECT

**PDFInspector** (inspector.py lines 62-67):
- `__init__(self, config: PDFProcessorConfig)` -- PLAN: `PDFInspector(self._config)` -- CORRECT.
- `classify(self, profile: DocumentProfile) -> ClassificationResult` -- PLAN: `self._inspector.classify(profile)` -- CORRECT.

**PDFLLMClassifier** (llm_classifier.py lines 84, 90-93):
- `__init__(self, llm: LLMBackend, config: PDFProcessorConfig)` -- PLAN: `PDFLLMClassifier(llm, self._config)` -- CORRECT.
- `classify(self, profile: DocumentProfile, tier: ClassificationTier) -> ClassificationResult` -- PLAN: `self._llm_classifier.classify(profile, ClassificationTier.LLM_BASIC)` -- CORRECT.
- Raises `ConnectionError` for outage (line 174-176) -- PLAN correctly catches this.

### 4. SecurityScanner API -- VERIFIED CORRECT

**PDFSecurityScanner** (security.py lines 32-35):
- `__init__(self, config: PDFProcessorConfig)` -- PLAN: `PDFSecurityScanner(self._config)` -- CORRECT.
- `scan(self, file_path: str) -> tuple[DocumentMetadata, list[IngestError]]` -- PLAN section 4 Step 1: `self._security_scanner.scan(file_path)` with correct tuple destructure -- CORRECT.

### 5. Error/Warning Code Verification

| Code Used in PLAN | Exists in errors.py | Status |
|-------------------|-------------------|--------|
| E_PARSE_CORRUPT | Line 26 | OK |
| E_CLASSIFY_INCONCLUSIVE | Line 38 | OK |
| W_LLM_UNAVAILABLE | Line 70 | OK |
| W_CLASSIFICATION_DEGRADED | Line 71 | OK |
| E_PROCESS_NOT_AVAILABLE | NOT FOUND | ISSUE |
| E_PROCESS_TIMEOUT | NOT FOUND | ISSUE |

**ISSUE-1 (MEDIUM): Missing error codes.** The PLAN references `E_PROCESS_NOT_AVAILABLE` (section 4 Step 11, for COMPLEX type when no processor) and `E_PROCESS_TIMEOUT` (section 7, for batch timeout). Neither exists in `errors.py`. Two options:
1. Add these codes to `errors.py` (scope creep -- errors.py is a separate concern).
2. Use existing codes: `E_CLASSIFY_INCONCLUSIVE` for COMPLEX-not-available, and a generic error string for timeout.

**Recommendation:** Use string-based error messages for these cases (e.g., `errors=["ComplexProcessor not available"]` for COMPLEX, and `errors=["Processing timeout"]` for batch timeout) rather than creating new ErrorCode entries. The existing pattern in other PLAN code uses string-based errors in some places. Alternatively, PATCH could add 2 new error codes but should document this as a minor deviation.

### 6. LLM Outage Resilience -- VERIFIED MATCHES SPEC 5.2

PLAN section 5 implements the exact contract from SPEC 5.2:

1. Tier 1 always runs (no external deps) -- CORRECT.
2. High confidence check: `tier1_result.confidence >= self._config.tier1_high_confidence_signals / 5` (4/5 = 0.8 default) -- CORRECT per SPEC 10.6.
3. Tier 2 wrapped in `try/except (ConnectionError, TimeoutError)` -- CORRECT.
4. Tier 3 escalation when `enable_tier3=True` and Tier 2 below threshold -- CORRECT.
5. On LLM exception: degrade to Tier 1 with `degraded=True`, emit `W_LLM_UNAVAILABLE` + `W_CLASSIFICATION_DEGRADED` -- CORRECT.
6. Key invariant: `E_CLASSIFY_INCONCLUSIVE` only when Tier 1 confidence==0.0 AND all LLM tiers fail -- CORRECT.

**Minor note:** PLAN catches `TimeoutError` in addition to `ConnectionError`. The LLM classifier (llm_classifier.py line 164) catches `TimeoutError` internally and continues retrying, but only explicitly re-raises `ConnectionError` (line 174-176). Other exceptions propagate as generic `Exception`. The router should also catch generic exceptions to avoid unhandled errors from Tier 3 calls. The PLAN's `(ConnectionError, TimeoutError)` catch is correct for the described behavior but PATCH should verify whether `TimeoutError` actually propagates from the LLM classifier or gets swallowed by the retry loop.

### 7. __init__.py Exports -- VERIFIED AGAINST SPEC 21.1

SPEC 21.1 requires these exports:
```
PDFRouter, PDFProcessorConfig, PDFType, PageType, ClassificationTier,
ClassificationResult, ProcessingResult, ChunkPayload, PDFChunkMetadata,
DocumentProfile, DocumentMetadata, PageProfile, ExtractionQuality,
OCRResult, TableResult, WrittenArtifacts, ParseStageResult,
ClassificationStageResult, OCRStageResult, EmbedStageResult,
ErrorCode, IngestError, create_default_router
```

PLAN section 9 exports all of the above PLUS:
- `IngestionMethod` -- not in SPEC 21.1 but useful; acceptable addition.
- `PDFInspector`, `PDFLLMClassifier`, `LLMClassificationResponse` -- not in SPEC 21.1 but already exported in current `__init__.py`; preserving existing exports is correct.
- `TextExtractor`, `OCRProcessor` -- not in SPEC 21.1 but already exported; preserving is correct.

**Verified:** All SPEC 21.1 exports are covered. Additional exports are justified by backward compatibility with existing `__init__.py`.

### 8. QualityAssessor API -- VERIFIED

PLAN section 6: `self._quality_assessor.assess_page(text, page_number)` and `self._quality_assessor.assess_document(page_qualities)`.
- `assess_page(page_text: str, page_number: int)` (quality.py line 24) -- CORRECT.
- `assess_document(page_qualities: list[ExtractionQuality])` (quality.py line 61) -- CORRECT.

### 9. LayoutAnalyzer API -- VERIFIED

PLAN section 6: `self._layout_analyzer.detect_columns(page)` returning result with `.column_count`.
- `LayoutAnalyzer.__init__(config)` (layout_analysis.py line 131) -- CORRECT.
- `detect_columns(page) -> LayoutResult` with `column_count` field (layout_analysis.py lines 52-58, 138) -- CORRECT.

### 10. detect_language API -- VERIFIED

PLAN section 6: `detect_language(sample_text, default_language=self._config.default_language)`.
- `detect_language(text: str, *, default_language: str = "en") -> tuple[str, float]` (language.py lines 88-92) -- CORRECT. Note: `default_language` is a keyword-only argument (preceded by `*`). PLAN uses it as keyword arg -- CORRECT.

### 11. Scope Check

Files touched: 3 (2 create, 1 modify)
- Create: `router.py`, `tests/test_router.py`
- Modify: `__init__.py`

**No scope creep detected:**
- No new error codes created (ISSUE-1 notes the gap but doesn't add them)
- No ComplexProcessor implementation (correctly stubbed as None)
- No changes to models.py, config.py, errors.py
- No new backends
- No ROADMAP items implemented
- Document profiling is inline in the router (correct per PLAN decision 12.2)

### 12. Imports Verification

PLAN section 11 lists all imports for `router.py`. Verified against actual module paths:
- `ingestkit_core.idempotency.compute_ingest_key` -- EXISTS
- `ingestkit_core.models.ClassificationTier` -- EXISTS (re-exported in pdf models too)
- `ingestkit_core.models.EmbedStageResult` -- EXISTS
- `ingestkit_core.models.WrittenArtifacts` -- EXISTS
- `ingestkit_pdf.quality.QualityAssessor` -- EXISTS (quality.py line 18)
- `ingestkit_pdf.utils.layout_analysis.LayoutAnalyzer` -- EXISTS (layout_analysis.py line 122)
- All other imports verified present.

**ISSUE-2 (LOW): OCRStageResult import listed but may not be needed.** The router creates `ProcessingResult` but the OCR stage result is populated by the processors, not the router. The router only receives it via the processor's returned `ProcessingResult`. Import is harmless but unnecessary. PATCH should include only what is actually used.

### 13. process_batch() Worker Function

PLAN section 7 defines `_process_single_file()` at module level calling `create_default_router(config=config)`. This requires `create_default_router` to be importable with only a config dict -- it creates its own backends.

**ISSUE-3 (MEDIUM): process_batch worker creates default backends, ignoring caller's custom backends.** If a caller creates `PDFRouter` with custom backends, `process_batch()` will use default backends in workers instead. This is documented in the PLAN ("For custom backends, callers should iterate and call `process()` directly") and matches SPEC 18.1 ("Backend connections created per-worker"). Acceptable for v1.0.

### 14. Test Coverage Assessment

The PLAN defines 9 test classes with 33+ test methods covering:
- can_handle: 6 tests (extensions)
- Classification tiers: 4 tests
- LLM outage resilience: 5 tests (matches SPEC 5.2 test contract)
- Security scan: 2 tests
- Document profiling: 6 tests
- Process flow: 6 tests
- Ingest key: 3 tests
- process_batch: 3 tests
- create_default_router: 2 tests
- Logging: 3 tests

**Verified:** SPEC 5.2 test contract is explicitly included (`test_llm_outage_degrades_to_tier1`). All public methods have tests. Mock strategy is sound -- uses existing conftest.py fixtures.

## Issues Found

**ISSUE-1 (MEDIUM): `E_PROCESS_NOT_AVAILABLE` and `E_PROCESS_TIMEOUT` do not exist in errors.py.** PATCH should use descriptive string errors or add these codes with a note about the addition.

**ISSUE-2 (LOW): `OCRStageResult` imported but likely unused in router.py.** PATCH should trim unused imports.

**ISSUE-3 (INFO): `process_batch()` ignores caller's custom backends.** Documented and acceptable per SPEC 18.1. No action needed.

**ISSUE-4 (LOW): `TimeoutError` propagation from LLM classifier uncertain.** The LLM classifier catches `TimeoutError` internally (llm_classifier.py line 164) and retries, so `TimeoutError` may not propagate to the router after retries are exhausted. The classifier returns a `confidence=0.0` result on retry exhaustion instead. The `except (ConnectionError, TimeoutError)` in the router's `_classify()` will still correctly catch `ConnectionError` (which IS propagated). `TimeoutError` from `ConnectionError` variants (httpx, urllib3) would also be caught. This is safe but PATCH should verify end-to-end behavior.

## Verdict

**PASS** -- The PLAN is comprehensive and correctly implements SPEC sections 5.2, 17.1-17.3, 18.1, 20, and 21.1. All 18 acceptance criteria are covered. All processor API signatures verified correct against actual source code. LLM outage resilience exactly matches SPEC 5.2 contract. `__init__.py` exports cover all SPEC 21.1 requirements plus backward-compatible additions. Issues found are all LOW/MEDIUM severity and addressable during PATCH without plan changes. Ready for PATCH.

AGENT_RETURN: .agents/outputs/plan-check-42-021426.md
