---
issue: 69
agent: PLAN-CHECK
date: 2026-02-15
complexity: COMPLEX
stack: backend
status: REVISE
blocking_issues: 3
non_blocking_issues: 5
---

# PLAN-CHECK - Issue #69: FormRouter and Plugin API Surface

## Executive Summary

The PLAN is structurally sound and covers the core deliverables: FormRouter orchestrator, 4 new API methods on FormTemplateAPI, factory function, and tests. However, 3 blocking issues must be resolved before PATCH: (1) the PLAN incorrectly scopes idempotency implementation as new work when `idempotency.py` is already fully implemented, creating overlap with #71; (2) the PLAN proposes a `compute_form_ingest_key` wrapper function that does not exist in the spec or codebase and duplicates the already-exported `compute_ingest_key`; (3) the DI constructor lists 9 backends but does not include `FormDBBackend` in the correct protocol name position -- `protocols.py` defines it as `FormDBBackend`, not `StructuredDBBackend`. Additionally, 5 non-blocking issues need attention.

## Blocking Issues

### B1: Idempotency Module Already Implemented -- Remove from Scope

**PLAN says (File 1, D6):** "Replace stub with implementation" for `idempotency.py`, implementing `compute_form_ingest_key()` and `compute_form_extraction_key()`.

**Actual state:** `idempotency.py` (line 1-77) is fully implemented with:
- `compute_form_extraction_key()` (lines 31-54) -- exactly as the PLAN describes
- `compute_vector_point_id()` (lines 57-76) -- additional function not in the PLAN
- Re-exports `compute_ingest_key` from `ingestkit_core.idempotency` (line 20)
- `__init__.py` already exports all three (lines 34-38, 126-128)

The PLAN's `compute_form_ingest_key()` wrapper function does NOT exist in the codebase, the spec, or the existing `__init__.py` exports. It would be redundant with the already-exported `compute_ingest_key`.

**Resolution:** Remove File 1 entirely from the PLAN. The router should call `compute_ingest_key()` directly (already available via `from ingestkit_forms.idempotency import compute_ingest_key`) and `compute_form_extraction_key()` (already implemented). This also eliminates the overlap concern with #71 -- there is no overlap because the work is already done.

### B2: Router Pipeline Step 1 References Non-Existent Function

**PLAN D2 step 1:** "Compute global ingest key -- call `compute_form_ingest_key()`"

**Actual:** This function does not exist. The router should call `compute_ingest_key(file_path, parser_version, tenant_id, source_uri)` from `ingestkit_core.idempotency` (re-exported by `ingestkit_forms.idempotency`).

**Resolution:** Update D2 step 1 to use `compute_ingest_key()` directly. The function signature is:
```python
compute_ingest_key(
    file_path: str,
    parser_version: str,
    tenant_id: str | None = None,
    source_uri: str | None = None,
) -> IngestKey
```

### B3: PLAN D1 Constructor Signature Missing FormDBBackend Protocol Import

**PLAN D1:** Lists `form_db: FormDBBackend` as a constructor param.

**Actual protocols.py:** `FormDBBackend` is defined at line 88. This is correct, but the PLAN's File 2 implementation section (line 194) says "FormDBWriter(form_db, config)" -- the `FormDBWriter.__init__` signature is `(db: FormDBBackend, config: FormProcessorConfig)`, parameter named `db` not `form_db`.

**Resolution:** Minor but could cause a naming bug. In the router constructor, store as `self._form_db` but pass to `FormDBWriter(db=form_db, config=config)` using the keyword argument.

## Non-Blocking Issues

### N1: PLAN Acceptance Criteria References `compute_form_ingest_key` (Remove)

Acceptance criteria items reference `compute_form_ingest_key()` which should not be created. Remove these criteria:
- "compute_form_ingest_key() delegates to core and returns IngestKey"
- "compute_form_extraction_key() produces deterministic SHA-256..."

Replace with: "Router correctly calls `compute_ingest_key()` from core and `compute_form_extraction_key()` from idempotency module."

### N2: `__init__.py` Export Plan Is Partially Redundant

**PLAN File 4:** Add exports for `FormRouter`, `create_default_router`, `compute_form_ingest_key`, `compute_form_extraction_key`.

**Actual:** `compute_form_extraction_key` and `compute_ingest_key` are already exported. Only `FormRouter` and `create_default_router` need adding. Remove the idempotency additions from File 4.

### N3: FormDualWriter.write() Return Type Verification

**PLAN D2 step 11:** "FormDualWriter.write() returns (written, errors, warnings, error_details, embed_result)"

**Actual (dual_writer.py line 186-192):** Signature is:
```python
def write(self, extraction, template, source_uri, ingest_key, ingest_run_id)
    -> tuple[FormWrittenArtifacts, list[str], list[str], list[FormIngestError], EmbedStageResult | None]
```
This matches the PLAN's 5-tuple description. Confirmed correct.

### N4: Extractor Constructors -- Verify Plan Matches Actual Signatures

| Extractor | PLAN Constructor | Actual Constructor | Match? |
|-----------|-----------------|-------------------|--------|
| ExcelCellExtractor | `(config)` | `(config: FormProcessorConfig)` | Yes |
| NativePDFExtractor | `(pdf_widget_backend, config, ocr_backend)` | `(pdf_backend: PDFWidgetBackend, config: FormProcessorConfig, ocr_backend: OCRBackend \| None = None)` | Yes |
| OCROverlayExtractor | `(ocr_backend, config)` | `(ocr_backend: OCRBackend, config: FormProcessorConfig)` | Yes |
| VLMFieldExtractor | `(vlm_backend, config)` | `(vlm_backend: VLMBackend, config: FormProcessorConfig)` | Yes |

All match. No issues.

### N5: VLMFieldExtractor API -- `apply_vlm_fallback` Not `extract`

**PLAN D2 step 8:** "VLMFieldExtractor.apply_vlm_fallback() on fields marked vlm_fallback_pending"

**Actual (vlm_fallback.py line 106):** `apply_vlm_fallback(fields, template, file_path) -> list[ExtractedField]`

The PLAN correctly references `apply_vlm_fallback` (not `extract`), and the VLMFieldExtractor does NOT have an `extract()` method. The PLAN's D3 `_select_extractor` should NOT include VLM as a selectable extractor -- it is a post-processing step, not a primary extractor. The PLAN handles this correctly in D2 (step 8 is separate from step 6). Confirmed correct.

## Scope Overlap Check (Issues #71, #72)

### Issue #71 (Idempotency)

**Concern:** #71 was originally scoped to implement `compute_form_ingest_key` and `compute_form_extraction_key`.

**Finding:** Both functions are already implemented in `idempotency.py`. The PLAN's D6 section proposes re-implementing them, which would be duplicate work. Since the code already exists, neither #69 nor #71 needs to implement these functions.

**Verdict:** No overlap. Remove idempotency implementation from #69's scope. If #71 adds additional logic (e.g., dedup gate checks), that remains separate.

### Issue #72 (Path F Pipeline Gate)

**Concern:** #72 implements `try_match()` pipeline gate and structured logging.

**Finding:** The PLAN does not implement `try_match()` or any pipeline gate function. The router's `extract_form()` calls `matcher.match_document()` internally, which is the matching logic (not the gate). The gate (`try_match`) would be the orchestration layer's decision point that calls the router. No overlap.

**Verdict:** No overlap confirmed.

## 12-Step Pipeline vs. Spec 4.2

| PLAN Step | Spec 4.2 Stage | Match? |
|-----------|---------------|--------|
| 1. Compute ingest key | "Compute ingest_key (idempotency)" | Yes (with B2 fix) |
| 2. Generate ingest_run_id | Not explicit in spec diagram | Acceptable addition |
| 3. Template resolution (match/manual) | "Form Matcher (Path F gate)" | Yes |
| 4. Compute form extraction key | Implied by spec 4.3 | Yes |
| 5. Detect source format | Spec 3.1 flow | Yes |
| 6. Select and run extractor | Spec 3.1 extractor selection | Yes |
| 7. Confidence scoring | Per-field confidence (spec 7.4) | Yes |
| 8. VLM fallback | Spec 3.1 VLM tier | Yes |
| 9. Fail-closed check | Spec 4.1 principle 5 | Yes |
| 10. Build FormExtractionResult | Assembly step | Yes |
| 11. Dual write | "Dual Write: DB row + Chunks" | Yes |
| 12. Assemble FormProcessingResult | Return result | Yes |

**Missing from PLAN:** The spec 4.2 diagram shows "Pre-flight Security Scan" as step 0 before ingest key. The PLAN correctly does NOT implement this (it's an existing stage, not part of the router). Confirmed appropriate.

## Source Detection vs. Spec 3.1

PLAN D3 maps:
- `.xlsx` -> ExcelCellExtractor: Matches spec
- `.pdf` + `has_form_fields()=True` -> NativePDFExtractor: Matches spec
- `.pdf` + `has_form_fields()=False` -> OCROverlayExtractor: Matches spec
- Image files -> OCROverlayExtractor: Matches spec
- Unsupported -> `E_FORM_UNSUPPORTED_FORMAT`: Matches spec

**Confirmed correct.**

## FormPluginAPI Coverage (Spec 9.1 -- 10 Operations)

| # | Operation | Existing? | PLAN Adds? |
|---|-----------|-----------|------------|
| 1 | list_templates | Yes (api.py:237) | -- |
| 2 | get_template | Yes (api.py:214) | -- |
| 3 | list_template_versions | Yes (api.py:252) | -- |
| 4 | create_template | Yes (api.py:56) | -- |
| 5 | update_template | Yes (api.py:111) | -- |
| 6 | delete_template | Yes (api.py:200) | -- |
| 7 | render_document | No | Yes (D4) |
| 8 | preview_extraction | No | Yes (D4) |
| 9 | match_document | No | Yes (D4) |
| 10 | extract_form | No | Yes (D4) |

All 10 operations covered.

## DI Backend Count Check

PLAN D1 lists 9 DI params. Actual protocols requiring injection:

| # | Backend | Protocol Source | Required? | PLAN? |
|---|---------|----------------|-----------|-------|
| 1 | template_store | FormTemplateStore | Yes | Yes |
| 2 | fingerprinter | LayoutFingerprinter | Yes | Yes |
| 3 | form_db | FormDBBackend | Yes | Yes |
| 4 | vector_store | VectorStoreBackend | Yes | Yes |
| 5 | embedder | EmbeddingBackend | Yes | Yes |
| 6 | ocr_backend | OCRBackend | Optional | Yes |
| 7 | pdf_widget_backend | PDFWidgetBackend | Optional | Yes |
| 8 | vlm_backend | VLMBackend | Optional | Yes |
| 9 | config | FormProcessorConfig | Optional | Yes |

9 params confirmed. All match.

## Test Coverage Assessment

18 tests planned across 2 test files:
- `test_router.py`: 18 tests (constructor, extractor selection, pipeline, errors, factory, idempotency)
- `test_api.py`: 5 tests (4 new methods + dependency guard)

The idempotency integration test (PLAN item 6.1) should be simplified since the functions already exist and are tested. The test should verify the router correctly uses them, not test the functions themselves.

Mock backends: All 8 needed mocks already exist in `conftest.py` (lines 31-414). No new mocks needed. Confirmed.

## Recommendation

**Status: REVISE** -- Fix 3 blocking issues before proceeding to PATCH:

1. Remove File 1 (idempotency.py) entirely from scope -- already implemented.
2. Update router pipeline to call `compute_ingest_key()` directly (not `compute_form_ingest_key()`).
3. Update `__init__.py` plan to only add `FormRouter` and `create_default_router`.

After these fixes, the PLAN is ready for PATCH.

---
AGENT_RETURN: plan-check-69-021526.md
