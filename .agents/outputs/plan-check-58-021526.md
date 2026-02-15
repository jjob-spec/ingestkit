---
issue: 58
agent: PLAN-CHECK
date: 2026-02-15
plan_artifact: map-plan-58-021526.md
spec_section: "docs/specs/form-ingestor-plugin.md section 12"
verdict: PASS_WITH_FINDINGS
---

# PLAN-CHECK: Issue #58 -- Form Error Taxonomy and Structured Error Model

## Executive Summary

The MAP-PLAN for issue #58 is well-structured and covers the spec requirements with high fidelity. All 41 error/warning codes from section 12.1 are present and correctly transcribed. The FormIngestError model includes all fields from section 12.2. Two minor findings were identified: (1) the plan states a total of 42 enum members but the actual count is 41, and (2) the plan lacks a formal acceptance criteria checklist. Neither finding blocks implementation.

## Check Results

### 1. Error/Warning Code Coverage (section 12.1)

| Category | Spec Count | Plan Count | Status |
|----------|-----------|------------|--------|
| Template errors (E_FORM_TEMPLATE_*) | 4 | 4 | PASS |
| Matching errors (E_FORM_NO_MATCH, E_FORM_FINGERPRINT_FAILED) | 2 | 2 | PASS |
| Extraction errors (E_FORM_EXTRACTION_*, E_FORM_OCR_*, etc.) | 6 | 6 | PASS |
| Output errors (E_FORM_DB_*, E_FORM_CHUNK_*) | 3 | 3 | PASS |
| Dual-write errors (E_FORM_DUAL_WRITE_PARTIAL) | 1 | 1 | PASS |
| Manual override errors (E_FORM_FORMAT_MISMATCH) | 1 | 1 | PASS |
| VLM errors (E_FORM_VLM_*) | 2 | 2 | PASS |
| Security errors (E_FORM_FILE_*) | 2 | 2 | PASS |
| Backend errors (E_BACKEND_*) | 6 | 6 | PASS |
| Warnings (W_FORM_*) | 14 | 14 | PASS |
| **Total** | **41** | **41** | **PASS** |

Every code in the spec appears in the plan's enum listing. No codes are missing. No spurious codes were added.

**Finding F1 (LOW):** The plan's investigation summary (line 22) states "42 enum members" and the total count test (Step 5, test #2) asserts `== 42`. The correct count is **41**: 21 form-specific E_ codes + 6 backend E_ codes + 14 W_ codes = 41. The plan miscounted form-specific E_ codes as 22 instead of 21. The PATCH agent must use count 41 in the test assertion.

### 2. FormIngestError Fields (section 12.2)

| Field | Spec | Plan | Status |
|-------|------|------|--------|
| `code: FormErrorCode` | Yes | Yes (narrows from base `str`) | PASS |
| `message: str` | Yes | Inherited from BaseIngestError | PASS |
| `template_id: str \| None` | Yes | Yes | PASS |
| `template_version: int \| None` | Yes | Yes | PASS |
| `field_name: str \| None` | Yes | Yes | PASS |
| `page_number: int \| None` | Yes | Yes | PASS |
| `stage: str \| None` | Yes | Inherited from BaseIngestError | PASS |
| `recoverable: bool = False` | Yes | Inherited from BaseIngestError | PASS |
| `candidate_matches: list[dict] \| None` | Yes | Yes (with Field descriptor) | PASS |
| `backend_operation_id: str \| None` | Yes | Yes (with Field descriptor) | PASS |
| `fallback_reason: str \| None` | Yes | Yes (with Field descriptor) | PASS |

All 11 fields accounted for. The plan's decision to extend `BaseIngestError` rather than use a standalone `BaseModel` is sound -- it follows the established ingestkit-excel pattern and inherits `code`, `message`, `stage`, `recoverable` from the base.

Note: The spec shows `FormIngestError(BaseModel)` directly, while the plan uses `FormIngestError(BaseIngestError)`. This is an intentional improvement that maintains consistency with the sibling package pattern. Functionally equivalent since all base fields are present.

### 3. E_BACKEND_* Code Reuse from ingestkit-core

All 6 backend codes in the plan match `CoreErrorCode` in `packages/ingestkit-core/src/ingestkit_core/errors.py` (lines 37-42):

- `E_BACKEND_VECTOR_TIMEOUT` -- matches
- `E_BACKEND_VECTOR_CONNECT` -- matches
- `E_BACKEND_DB_TIMEOUT` -- matches
- `E_BACKEND_DB_CONNECT` -- matches
- `E_BACKEND_EMBED_TIMEOUT` -- matches
- `E_BACKEND_EMBED_CONNECT` -- matches

The plan correctly duplicates these as `FormErrorCode` enum members (same pattern as ingestkit-excel) rather than importing them, and includes a unit test (Step 5, test #7) to verify the values stay in sync. **PASS**.

### 4. ENUM_VALUE Check (name == value)

Every enum member in the plan's code listing has its name equal to its string value (e.g., `E_FORM_NO_MATCH = "E_FORM_NO_MATCH"`). The plan also includes a unit test (Step 5, test #1) that asserts `member.name == member.value` for all members. **PASS**.

### 5. Acceptance Criteria Completeness

**Finding F2 (MEDIUM):** The plan does not include a formal "Acceptance Criteria" section as expected by the orchestrate workflow. Step 5 (unit tests) implicitly covers the acceptance criteria, but the PATCH and PROVE agents expect an explicit checklist to reference.

**Recommended acceptance criteria for PATCH agent:**

1. `FormErrorCode` enum has exactly 41 members
2. All enum member names equal their string values
3. All 21 form-specific E_ codes present per spec section 12.1
4. All 6 E_BACKEND_* codes present and match `CoreErrorCode` values
5. All 14 W_FORM_* warning codes present per spec section 12.1
6. `FormIngestError` extends `BaseIngestError` with all 7 form-specific fields
7. `FormIngestError` serializes and deserializes correctly (Pydantic round-trip)
8. Diagnostic context fields default to `None`
9. `recoverable` defaults to `False`
10. Package installs with `pip install -e` and imports succeed
11. All unit tests pass

## Findings Summary

| ID | Severity | Description | Action |
|----|----------|-------------|--------|
| F1 | LOW | Enum count stated as 42, actual is 41 | PATCH must use 41 in test assertion |
| F2 | MEDIUM | No formal acceptance criteria section | PATCH should use criteria from this check |

## Verdict

**PASS_WITH_FINDINGS** -- The plan is approved for PATCH with the two corrections noted above. No blocking issues. The plan correctly covers all spec requirements, follows established patterns, and includes appropriate test coverage.

AGENT_RETURN: .agents/outputs/plan-check-58-021526.md
