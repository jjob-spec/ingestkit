---
issue: 57
agent: PLAN-CHECK
date: 2026-02-15
status: PASS_WITH_ISSUES
depends_on: plan-57-021526.md
---

# PLAN-CHECK: Issue #57 -- Form Data Models (Pydantic v2)

## 1. Executive Summary

The PLAN is substantially correct and comprehensive. It covers all 22 types from the MAP, correctly applies Pydantic v2 patterns, and includes a thorough test plan. Three issues require attention before PATCH: (1) the PLAN creates `errors.py` which overlaps with issue #58's scope, (2) the PLAN states "37 codes" for `FormErrorCode` but the spec defines 41 codes, and (3) the spec's `FormChunkMetadata` has `ingest_run_id: str` (required) while core's `BaseChunkMetadata` has `ingest_run_id: str | None = None` -- the PLAN does not address this type narrowing.

---

## 2. Check Results

### 2.1 Are all 22 types from the MAP accounted for?

**PASS.** All 22 types from MAP section 7 are present in the PLAN:

| # | Type | PLAN Section | Status |
|---|------|-------------|--------|
| 1 | `SourceFormat` | 2.2 Section 1 | Present |
| 2 | `FieldType` | 2.2 Section 1 | Present |
| 3 | `DualWriteMode` | 2.2 Section 1 | Present |
| 4 | `BoundingBox` | 2.2 Section 2 | Present |
| 5 | `CellAddress` | 2.2 Section 2 | Present |
| 6 | `FieldMapping` | 2.2 Section 2 | Present |
| 7 | `FormTemplate` | 2.2 Section 2 | Present |
| 8 | `TemplateMatch` | 2.2 Section 3 | Present |
| 9 | `FormIngestRequest` | 2.2 Section 3 | Present |
| 10 | `ExtractedField` | 2.2 Section 4 | Present |
| 11 | `FormExtractionResult` | 2.2 Section 4 | Present |
| 12 | `FormProcessingResult` | 2.2 Section 6 | Present |
| 13 | `FormWrittenArtifacts` | 2.2 Section 6 | Present |
| 14 | `FormTemplateCreateRequest` | 2.2 Section 7 | Present |
| 15 | `FormTemplateUpdateRequest` | 2.2 Section 7 | Present |
| 16 | `ExtractionPreview` | 2.2 Section 7 | Present |
| 17 | `FormChunkMetadata` | 2.2 Section 5 | Present |
| 18 | `FormChunkPayload` | 2.2 Section 5 | Present |
| 19 | `RollbackResult` | 2.2 Section 6 | Present |
| 20 | `FormIngestError` | 2.1 errors.py | Present |
| 21 | `IngestKey` | 2.2 imports | Reused from core |
| 22 | `EmbedStageResult` | 2.2 imports | Reused from core |

### 2.2 Do enum values match spec exactly?

**PASS.** All three enums use `(str, Enum)` pattern with correct lowercase string values:

- `SourceFormat`: `PDF="pdf"`, `XLSX="xlsx"`, `IMAGE="image"` -- matches spec §5.1 lines 308-310.
- `FieldType`: All 7 values lowercase -- matches spec §5.1 lines 316-322.
- `DualWriteMode`: `BEST_EFFORT="best_effort"`, `STRICT_ATOMIC="strict_atomic"` -- matches spec §8.0 lines 988-989.

ENUM_VALUE risk properly mitigated.

### 2.3 Is FieldMapping model_validator correct?

**PASS.** The PLAN's validator matches the spec §5.2 lines 470-492 exactly:
- Uses `@model_validator(mode="after")`
- Checks both-None case (raises ValueError)
- Checks both-set case (raises ValueError)
- Returns `self`
- Error messages match spec text

### 2.4 Are core imports correct?

**PASS.** The PLAN imports:
- `BaseChunkMetadata` from `ingestkit_core.models` -- exists at line 78 of core models.py
- `WrittenArtifacts` from `ingestkit_core.models` -- exists at line 115 of core models.py
- `EmbedStageResult` from `ingestkit_core.models` -- exists at line 65 of core models.py
- `IngestKey` from `ingestkit_core.models` -- exists at line 38 of core models.py
- `BaseIngestError` from `ingestkit_core.errors` -- exists at line 48 of core errors.py

All import paths verified against actual source files.

### 2.5 Is Pydantic v2 pattern used (no v1 json_encoders)?

**PASS.** The PLAN correctly identifies the spec's v1 pattern (`class Config: json_encoders`) and replaces it with:
- `@field_serializer("layout_fingerprint", "thumbnail")` for bytes-to-hex serialization
- `datetime.now(timezone.utc)` instead of deprecated `datetime.utcnow`
- No `ConfigDict` needed (bytes is a standard type)

### 2.6 Are all model fields matching the spec?

**PASS with one issue.** Field-by-field verification against spec:

- `BoundingBox`: 4 fields match spec §5.1 lines 333-336. Constraints (ge/le/gt) correct.
- `CellAddress`: 2 fields match spec §5.1 lines 346-350.
- `FieldMapping`: 12 fields match spec §5.1 lines 360-399. (Note: PLAN table lists 12 rows including field_id, the MAP says 11 -- PLAN is correct, counted field_id.)
- `FormTemplate`: 13 fields match spec §5.1 lines 410-449.
- `TemplateMatch`: 6 fields match spec §6.2 lines 622-638.
- `FormIngestRequest`: 6 fields match spec §6.3 lines 661-669.
- `ExtractedField`: 11 fields match spec §10.1 lines 1410-1440.
- `FormExtractionResult`: 14 fields match spec §10.1 lines 1450-1473.
- `FormProcessingResult`: 14 fields match spec §10.2 lines 1485-1502.
- `FormChunkMetadata`: Fields match spec §8.3 lines 1124-1155.
- `FormChunkPayload`: 4 fields match spec §8.4 lines 1164-1167.
- `RollbackResult`: 4 fields match spec §8.5 lines 1178-1181.
- `FormTemplateCreateRequest`: 8 fields match spec §9.2 lines 1363-1372.
- `FormTemplateUpdateRequest`: 5 fields match spec §9.2 lines 1378-1385.
- `ExtractionPreview`: 7 fields match spec §9.2 lines 1391-1397.

**Issue:** The spec §8.3 defines `FormChunkMetadata` with `ingest_run_id: str` (no default, required), but core `BaseChunkMetadata` has `ingest_run_id: str | None = None` (optional). If `FormChunkMetadata` extends `BaseChunkMetadata`, the inherited `ingest_run_id` will be `str | None = None`. The PLAN does not override this field to make it required. This is acceptable (Pydantic allows callers to pass `str`), but PATCH should document this intentional relaxation.

### 2.7 Are test cases comprehensive?

**PASS.** The PLAN specifies ~30 tests across 11 categories:
- Enum value correctness (3 tests)
- BoundingBox validation (3 tests)
- FieldMapping validator (4 tests -- covers both error paths + both valid paths)
- FormTemplate defaults and serialization (4 tests)
- Matching model bounds (2 tests)
- Extraction result types and defaults (3 tests)
- Chunk model inheritance (2 tests)
- Result model structure (3 tests)
- Request/Response validation (2 tests)
- Error model inheritance (2 tests)
- Serialization round-trips (2 tests)

Coverage is comprehensive. Every model/enum has at least one test. Both validator paths for FieldMapping are covered.

---

## 3. Scope Overlap Analysis (#57 vs #58)

**ISSUE FOUND.** The PLAN for #57 (section 2.1) creates `errors.py` containing:
- `FormErrorCode` (41 enum members)
- `FormIngestError` (extends `BaseIngestError`)

Issue #58 (`map-plan-58-021526.md`) is explicitly scoped to create this same `errors.py` file with `FormErrorCode` and `FormIngestError`.

**Recommendation:** The PLAN for #57 should NOT create `errors.py`. Instead:
- If #58 is implemented first: #57 imports from the existing `errors.py`
- If #57 is implemented first: #57 should stub a minimal `errors.py` with just `FormIngestError` (needed by `FormProcessingResult.error_details`), or use `BaseIngestError` directly and let #58 refine it

**Preferred approach:** Implement #58 before #57, then #57 simply imports `FormErrorCode` and `FormIngestError` from `ingestkit_forms.errors`. The PLAN's `models.py` already imports from `ingestkit_forms.errors` (line 108), so the dependency direction is correct.

Similarly, the PLAN creates `pyproject.toml` and `__init__.py` which overlap with issue #56 (scaffold). The PLAN should assume #56 runs first and provides these files.

---

## 4. Error Code Count Discrepancy

**ISSUE FOUND.** The PLAN states "37 codes from spec section 12.1" (section 2.1). Actual count from spec §12.1:

| Category | Count |
|----------|-------|
| Template errors | 4 |
| Matching errors | 2 |
| Extraction errors | 6 |
| Output errors | 3 |
| Dual-write errors | 1 |
| Manual override errors | 1 |
| VLM errors | 2 |
| Security errors | 2 |
| Backend errors (reused) | 6 |
| Warnings | 14 |
| **Total** | **41** |

The PLAN's own listing (section 2.1) actually enumerates all 41 codes correctly -- only the summary count "37" is wrong. This is a documentation error, not a logic error. PATCH should use the correct count of 41.

Note: the #58 MAP-PLAN says "42 enum members" which appears to be its own counting error (or counts the class definition itself). The spec clearly shows 41 distinct enum values.

---

## 5. Intentional Spec Deviations (Approved)

The PLAN makes three deliberate deviations from the spec, all well-justified:

1. **`FormProcessingResult.written` typed as `FormWrittenArtifacts`** instead of spec's `WrittenArtifacts`. Reason: core's `WrittenArtifacts` lacks `db_row_ids`. The PLAN creates `FormWrittenArtifacts(WrittenArtifacts)` to add it. This is correct -- the spec §10.2 shows `db_row_ids` on `WrittenArtifacts` but that contradicts the actual core definition.

2. **`FormProcessingResult.error_details` typed as `list[FormIngestError]`** instead of spec's `list[IngestError]`. Reason: forms needs form-specific error context fields. Using `FormIngestError` is more type-safe.

3. **`FormChunkMetadata` extends `BaseChunkMetadata`** instead of standalone `BaseModel`. Reason: avoids field duplication for the 9 overlapping standard fields. Spec §8.3 says "Extends the standard chunk metadata pattern" in its docstring, supporting this approach.

4. **`FormIngestError` extends `BaseIngestError`** instead of spec's standalone `BaseModel`. Reason: follows Excel's established pattern and enables polymorphic error handling.

---

## 6. Verdict

**PASS WITH ISSUES.** The PLAN is approved for PATCH with the following required fixes:

| # | Severity | Issue | Required Action |
|---|----------|-------|-----------------|
| 1 | HIGH | Scope overlap with #58 (errors.py) | Remove `errors.py` creation from #57 scope. Assume #58 runs first, or use `BaseIngestError` directly as a temporary measure. |
| 2 | HIGH | Scope overlap with #56 (pyproject.toml, __init__.py) | Remove scaffolding from #57 scope. Assume #56 runs first. |
| 3 | LOW | Error code count says "37" but actual count is 41 | Correct to 41 in PATCH. |
| 4 | LOW | `FormChunkMetadata.ingest_run_id` type relaxation | Document that inherited `str | None` is acceptable; callers should always provide a value. |

If issues 1 and 2 are addressed (by either reordering execution to #56 -> #58 -> #57, or by removing the overlapping files from #57's scope), the PLAN is ready for PATCH.

AGENT_RETURN: .agents/outputs/plan-check-57-021526.md
