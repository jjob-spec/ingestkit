---
issue: 66
agent: PLAN-CHECK
date: 2026-02-15
complexity: COMPLEX
stack: backend
---

# PLAN-CHECK: Issue #66 -- VLM Fallback Extraction for Low-Confidence OCR Fields

**Plan artifact**: `.agents/outputs/plan-66-021526.md`
**MAP artifact**: `.agents/outputs/map-66-021526.md`
**Issue**: #66 -- Implement VLM fallback extraction for low-confidence OCR fields
**Date**: 2026-02-15

---

## Executive Summary

Plan is **APPROVED with minor corrections**. The plan accurately reflects the spec (section 7.5), correctly maps to existing protocols, models, config fields, and error codes. All 14 acceptance criteria are covered by implementation steps and test cases. Three corrections are needed before PATCH: (1) `make_field_mapping` in conftest.py lacks `extraction_hint` and `field_id` parameters -- tests need to either use `FieldMapping()` directly or extend the factory; (2) the plan's `_get_page_number_for_field` comment states page_number comes from `FieldMapping` but the lookup is by `field_id` on the `ExtractedField` -- the `ExtractedField` model does NOT have a `page_number` field, so the page must come from the matched `FieldMapping.page_number`; (3) `form_vlm_timeout_seconds` is typed `int` in config.py (line 117), but `VLMBackend.extract_field` expects `timeout: float | None` -- the plan correctly casts with `float()`.

---

## Check 1: VLMBackend Protocol Match

**Source**: `protocols.py` lines 231-271

| Protocol Method | Plan Coverage | Signature Match |
|-----------------|--------------|-----------------|
| `extract_field(self, image_bytes: bytes, field_type: str, field_name: str, extraction_hint: str \| None = None, timeout: float \| None = None) -> VLMFieldResult` | Plan section 1.5 step 6e, MockVLMBackend section 3.1 | PASS |
| `model_name(self) -> str` | MockVLMBackend section 3.1 | PASS |
| `is_available(self) -> bool` | Plan section 1.5 step 2, MockVLMBackend section 3.1 | PASS |

**MockVLMBackend** (plan section 3.1) matches all 3 protocol methods with correct signatures. PASS.

---

## Check 2: VLMFieldResult Model Fields

**Source**: `protocols.py` lines 71-78

| Field | Type | Plan Usage | Status |
|-------|------|------------|--------|
| `value` | `str \| bool \| None` | Plan section 1.5 step 6f: `vlm_result.value` | PASS |
| `confidence` | `float` (ge=0.0, le=1.0) | Plan section 1.5 step 6f: `vlm_result.confidence` | PASS |
| `model` | `str` | MockVLMBackend returns `self._model` | PASS |
| `prompt_tokens` | `int \| None` | MockVLMBackend hardcodes 100 | PASS |
| `completion_tokens` | `int \| None` | MockVLMBackend hardcodes 20 | PASS |

All fields covered. PASS.

---

## Check 3: Budget Enforcement

**Config field**: `form_vlm_max_fields_per_document` (default 10, ge=1) at config.py line 121-125.

**Plan coverage**:
- Step 5 (plan section 1.5): takes first N candidates from sorted list, emits `W_FORM_VLM_BUDGET_EXHAUSTED` for overflow. PASS.
- Test 5 (`test_vlm_fallback_budget_exhausted`): 5 fields, budget=3, asserts call_count==3, 2 overflow get warning. PASS.

---

## Check 4: Priority Sorting

**Spec requirement** (section 7.5): "required fields are prioritized over optional fields, and fields with the lowest OCR confidence are processed first."

**Plan coverage** (step 4): Sort by `(not mapping.required, candidate.confidence)`. This gives:
- `required=True` -> `not True = False` (sorts first, since False < True)
- Within required group: lowest confidence first
- `required=False` -> `not False = True` (sorts second)

Correct per spec. PASS.

**Test 6** (`test_vlm_fallback_priority_required_first`): 4 fields, budget=2, asserts required fields C (0.15) then B (0.3) are processed first. PASS.

---

## Check 5: Graceful Degradation

| Scenario | Plan Step | Test | Error Code | Status |
|----------|-----------|------|------------|--------|
| VLM timeout | Step 6h: catch `TimeoutError`, log `E_FORM_VLM_TIMEOUT`, retain original | Test 7 | `E_FORM_VLM_TIMEOUT` (errors.py L56) | PASS |
| VLM generic error | Step 6i: catch `Exception`, log `E_FORM_VLM_UNAVAILABLE`, retain original | Test 8 | `E_FORM_VLM_UNAVAILABLE` (errors.py L55) | PASS |
| VLM unavailable | Step 2: `is_available()` returns False, return unchanged | Test 9 | `E_FORM_VLM_UNAVAILABLE` (errors.py L55) | PASS |
| VLM disabled | Step 1: guard clause returns unchanged | Test 1 | N/A | PASS |

---

## Check 6: 10% Padding on Crop

**Plan coverage**: `_crop_field_region_with_padding()` (section 1.2) computes `pad_x = px_w * 0.10`, `pad_y = px_h * 0.10`, expands box, clamps to bounds.

**Tests**: Test 12 (`test_crop_with_padding`) and Test 13 (`test_crop_with_padding_edge`) verify correct math and edge clamping.

PASS.

---

## Check 7: Warning Codes in errors.py

| Code | errors.py Line | Plan Usage | Status |
|------|---------------|------------|--------|
| `W_FORM_VLM_FALLBACK_USED` | L83 | Appended to all VLM-processed fields (step 6j) | PASS |
| `W_FORM_VLM_BUDGET_EXHAUSTED` | L84 | Appended to overflow fields (step 5) | PASS |
| `E_FORM_VLM_UNAVAILABLE` | L55 | Logged on `is_available()` False and generic errors (steps 2, 6i) | PASS |
| `E_FORM_VLM_TIMEOUT` | L56 | Logged on TimeoutError (step 6h) | PASS |

All 4 VLM error/warning codes exist and are correctly referenced. PASS.

---

## Check 8: Test Coverage vs. Acceptance Criteria

| Acceptance Criterion | Test(s) | Status |
|---------------------|---------|--------|
| VLMFieldExtractor class with apply_vlm_fallback() | All tests instantiate it | PASS |
| Trigger: confidence < 0.4 AND vlm_enabled=True | Tests 1-3 | PASS |
| Budget guard with W_FORM_VLM_BUDGET_EXHAUSTED | Test 5 | PASS |
| Priority: required first, lowest confidence first | Test 6 | PASS |
| Graceful degradation on timeout/error | Tests 7, 8 | PASS |
| 10% padding on crop | Tests 12, 13 | PASS |
| extraction_method="vlm_fallback" on improved fields | Test 3 | PASS |
| W_FORM_VLM_FALLBACK_USED on all processed fields | Tests 3, 4, 7, 8, 11 | PASS |
| is_available() check, log E_FORM_VLM_UNAVAILABLE | Test 9 | PASS |
| PII-safe logging | Plan step 6 PII-safe logging note | PASS (no dedicated test) |
| All VLM via VLMBackend protocol only | Plan imports only Protocol type | PASS |
| Unit tests cover all 13 cases | Tests 1-13 listed | PASS |
| No regressions | Verification gates section | PASS |
| VLMFieldExtractor in extractors/__init__.py | Plan file 2 | PASS |
| MockVLMBackend in conftest.py | Plan file 3 | PASS |

All 15 criteria covered. PASS.

---

## Issues Found

### Issue 1 (MINOR): `make_field_mapping` lacks `extraction_hint` and `field_id` parameters

The existing `make_field_mapping()` factory in conftest.py (line 225-248) does not expose `extraction_hint` or `field_id` parameters. The plan's `VLMFieldExtractor` looks up fields by `field_id` and passes `mapping.extraction_hint` to the VLM backend.

**Impact**: Tests that need specific `field_id` or `extraction_hint` values must either:
- Extend `make_field_mapping()` with those parameters (recommended), or
- Construct `FieldMapping` directly in each test.

**Recommendation**: PATCH should add `field_id` and `extraction_hint` params to `make_field_mapping()` as optional kwargs. This is a backward-compatible change.

### Issue 2 (INFORMATIONAL): `ExtractedField` has no `page_number` field

The `ExtractedField` model (models.py L271-304) does not have a `page_number` attribute. The plan correctly uses `FieldMapping.page_number` from the template lookup (section 1.6 `_get_field_mapping`, then access `mapping.page_number`). Plan section 1.7 confirms this approach.

No action needed -- just confirming the plan is correct.

### Issue 3 (INFORMATIONAL): `form_vlm_timeout_seconds` is `int`, protocol expects `float | None`

Config field `form_vlm_timeout_seconds` is typed `int` (config.py L117). Plan section 1.5 step 6e correctly passes `float(self._config.form_vlm_timeout_seconds)` for the cast. No action needed.

### Issue 4 (MINOR): Plan references `FormErrorCode.W_FORM_VLM_FALLBACK_USED.value` in ExtractedField warnings

The plan (section 1.5, step 6f) uses `FormErrorCode.W_FORM_VLM_FALLBACK_USED.value` when appending to `warnings: list[str]`. Since `FormErrorCode` is a `str` enum where name equals value (e.g., `"W_FORM_VLM_FALLBACK_USED"`), using `.value` is correct. Using the enum member directly would also work since `str(member) == member.value` for `str` enums. Either approach is fine, but `.value` is explicit and matches the `list[str]` type annotation.

No action needed.

### Issue 5 (MINOR): Plan's `vlm_enabled_config` fixture should set `form_extraction_min_field_confidence`

The plan's `vlm_enabled_config` fixture (section 3.3) sets `form_vlm_fallback_threshold=0.4` but relies on the default for `form_extraction_min_field_confidence` (0.5). The config validator (config.py L241-245) enforces `form_vlm_fallback_threshold < form_extraction_min_field_confidence`. Since 0.4 < 0.5, this passes. No issue, but PATCH should be aware that changing either value in the fixture could trigger validation errors.

No action needed.

---

## Recommendations

1. **Proceed with implementation.** Plan is solid and comprehensive.
2. **Extend `make_field_mapping()`** in conftest.py to accept `field_id: str | None = None` and `extraction_hint: str | None = None`. This keeps test code clean without needing raw `FieldMapping()` constructors.
3. **Verify `get_page_image` mock in tests.** Tests 3-8 and 10-11 need to mock `get_page_image` to return a test image. The plan mentions this (Test 3 setup: "Mock `get_page_image` to return a test image") but does not detail the mock mechanism. PATCH should use `unittest.mock.patch` on `ingestkit_forms.extractors.vlm_fallback.get_page_image`.
4. **Consider a PII-safe logging test.** The plan notes PII-safe logging (step 6) but has no dedicated test for it. This is acceptable for P0 but could be added as P2.

---

## Summary

**Plan status**: APPROVED (with minor corrections noted above)

| Check | Result |
|-------|--------|
| VLMBackend protocol match | PASS |
| VLMFieldResult model fields | PASS |
| Budget enforcement | PASS |
| Priority sorting | PASS |
| Graceful degradation | PASS |
| 10% padding | PASS |
| Warning codes in errors.py | PASS |
| Test coverage vs acceptance criteria | PASS |

**Issues**: 2 minor (factory extension, mock patching clarity), 3 informational
**Risk level**: LOW
**Ready for PATCH**: YES

---

AGENT_RETURN: .agents/outputs/plan-check-66-021526.md
