---
issue: 63
title: "Implement native PDF form field extraction"
agent: plan-check
timestamp: 2026-02-15
status: APPROVED
plan_artifact: .agents/outputs/plan-63-021526.md
map_artifact: .agents/outputs/map-63-021526.md
---

# PLAN-CHECK: Issue #63 -- Native PDF Form Field Extraction

## Executive Summary

The PLAN is well-structured, spec-aligned, and ready for PATCH. All seven validation checks pass. The `extract()` signature correctly adapts the spec's standalone function to an instance method. IoU computation is mathematically correct. Widget coercion covers all seven `FieldType` values. Flattened form detection matches the spec. Validation pattern enforcement is sound. No direct PyMuPDF imports. Test coverage is comprehensive (16 tests). The conftest changes are safe -- zero existing tests reference the placeholder fixtures being replaced.

---

## Validation Checks

### 1. Does `extract()` method signature match the spec?

**PASS**

| Aspect | Spec (section 7.1, line 698) | PLAN (section 1.3) |
|--------|-----|------|
| Input: `file_path` | `str` | `str` |
| Input: `template` | `FormTemplate` | `FormTemplate` |
| Return type | `list[ExtractedField]` | `list[ExtractedField]` |
| Method name | `extract_native_pdf_fields` (standalone) | `extract` (instance method) |

The PLAN correctly adapts the spec's standalone function to a `NativePDFExtractor.extract()` instance method. The constructor injects `PDFWidgetBackend` (required), `FormProcessorConfig` (required), and `OCRBackend` (optional) -- all protocol types, no concrete imports.

### 2. Is IoU computation correct (axis-aligned, normalized coords)?

**PASS**

The `_compute_iou()` algorithm in PLAN section 1.5 is mathematically correct:
- Converts `(x, y, width, height)` to `(x1, y1, x2, y2)` correctly: `x2 = x + width`, `y2 = y + height`
- Intersection: `max(0, min(x2a,x2b) - max(x1a,x1b)) * max(0, min(y2a,y2b) - max(y1a,y1b))`
- Union: `area_a + area_b - intersection`
- Division guard: returns `0.0` when `union_area == 0.0`
- Operates entirely in normalized 0.0-1.0 coordinate space (backend handles PDF-to-normalized conversion per protocol contract)

### 3. Does widget coercion handle all FieldTypes?

**PASS**

All 7 `FieldType` enum values covered in `_coerce_widget_value()` (PLAN section 1.6):

| FieldType | Coercion | None handling |
|-----------|----------|---------------|
| `TEXT` | Return as-is | `None` |
| `DATE` | Return as-is | `None` |
| `NUMBER` | `float()` with fallback | `None` |
| `CHECKBOX` | Truthy set (`"yes"`, `"on"`, `"true"`, `"1"`) | `False` |
| `RADIO` | Same truthy set | `False` |
| `DROPDOWN` | Return as-is | `None` |
| `SIGNATURE` | `bool(raw_value.strip())` | `False` |

Verified against `FieldType` enum at `models.py:41-50`: `TEXT`, `NUMBER`, `DATE`, `CHECKBOX`, `RADIO`, `SIGNATURE`, `DROPDOWN` -- all present.

### 4. Is flattened form detection correct?

**PASS**

PLAN section 1.3, step 1: calls `self._pdf_backend.has_form_fields(file_path)`. When `False`, returns empty list `[]` immediately and logs a warning. This matches the spec (line 726): "If a PDF has zero widgets ... the extractor detects this ... logging a `W_FORM_FIELDS_FLATTENED` warning."

The PLAN correctly delegates the `W_FORM_FIELDS_FLATTENED` warning emission to the caller (FormProcessor orchestrator) since the extractor returns an empty list as the signal. The extractor logs via `logger.warning()` for observability.

### 5. Is `validation_pattern` enforcement correct?

**PASS**

PLAN section 1.7:
- Uses `re.fullmatch()` (not `re.match()` or `re.search()`) -- correct for whole-value validation
- Returns `(value, True, [])` on match, `(value, False, [W_FORM_FIELD_VALIDATION_FAILED])` on failure
- Handles invalid regex via `re.error` exception -> `(value, None, [warning_msg])`
- Validation failure keeps the value (does not null it) -- correct design, caller decides policy
- Handles `None` values gracefully: `(None, None, [])`

### 6. No direct PyMuPDF imports in extractor?

**PASS**

PLAN section 1.1 imports only: `logging`, `re`, `collections.defaultdict` (stdlib), and `ingestkit_forms.*` internal modules (`config`, `errors`, `models`, `protocols`). No `fitz`, `pymupdf`, or any concrete backend. This matches spec section 7.1 (line 728): "The `NativePDFExtractor` does not import PyMuPDF directly."

### 7. Are test cases comprehensive (16 tests)?

**PASS**

All 16 tests are well-specified with concrete assertions:

| Category | Tests | Coverage |
|----------|-------|----------|
| Happy path (text) | #1, #2 | Exact match, IoU above threshold |
| IoU boundary | #3, #14, #15 | Below threshold, configurable threshold, direct IoU unit test |
| Field types | #4, #5, #6, #7, #8 | Checkbox checked/unchecked, radio, dropdown, number coercion |
| Flattened form | #9 | Zero widgets detection |
| Validation | #10, #11 | Pattern pass and fail |
| Multi-page | #12 | Fields across pages 0 and 1 |
| OCR fallback | #13, #16 | No OCR backend + unmatched, extraction method labels |

**IoU test math verified** (test #2): template (0.1, 0.1, 0.3, 0.05), widget (0.12, 0.1, 0.3, 0.05). Intersection x: max(0.1,0.12)=0.12 to min(0.40,0.42)=0.40, width=0.28. Height=0.05. Inter=0.014. Union=0.03-0.014=0.016. IoU=0.014/0.016=0.875. Correct.

---

## Conftest Safety Check

**SAFE** -- No regressions expected.

Verified via `grep` that none of the 152 existing passing tests (`test_models.py`, `test_config.py`, `test_errors.py`, `test_protocols.py`, `test_router.py`, `test_output.py`, `test_api.py`, `test_matcher.py`) reference any conftest fixture. The fixtures (`form_config`, `mock_template_store`, `mock_ocr_backend`, `mock_pdf_widget_backend`, `sample_pdf_form`, `sample_excel_form`, `sample_scanned_form`) are only defined in conftest, never imported or used by any existing test.

The PLAN replaces `form_config` (returns `{}`) with one that returns `FormProcessorConfig()`. Since no existing test uses this fixture, zero risk. The `MockPDFWidgetBackend` and `MockOCRBackend` classes are additions, not replacements.

---

## Minor Observations (Non-Blocking)

1. **Test #14 IoU math**: The PLAN says IoU ~ 0.6 for widget at (0.15, 0.1, 0.3, 0.05) vs field at (0.1, 0.1, 0.3, 0.05). Actual: intersection x = 0.15 to 0.40, width=0.25, height=0.05, inter=0.0125. Union=0.03-0.0125=0.0175. IoU=0.0125/0.0175=0.714. Above 0.5 (match), below 0.8 (no match). The PLAN's logic is correct even though the "~0.6" approximation should be "~0.71".

2. **`FormTemplate` requires `fields` with `min_length=1`**: The `make_template()` factory defaults to a single field -- correct.

3. **`FieldMapping` validator**: Requires exactly one of `region` or `cell_address`. The `make_field_mapping()` factory sets `region` and leaves `cell_address` as None -- correct.

---

## Verdict

**APPROVED** -- PLAN is ready for PATCH execution. No blockers found. All 7 validation checks pass. Conftest changes are safe against the existing 152 tests.

---

AGENT_RETURN: .agents/outputs/plan-check-63-021526.md
