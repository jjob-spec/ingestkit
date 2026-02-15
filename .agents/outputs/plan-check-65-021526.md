---
issue: 65
agent: PLAN-CHECK
date: 2026-02-15
complexity: SIMPLE
stack: backend
---

# PLAN-CHECK: Issue #65 -- Excel Cell Mapping Extraction

**Plan artifact**: `.agents/outputs/map-plan-65-021526.md`
**Spec section**: `docs/specs/form-ingestor-plugin.md` section 7.3 (lines 829-867)
**Issue**: #65 -- Implement Excel cell mapping extraction
**Date**: 2026-02-15

---

## Executive Summary

The plan is **APPROVED WITH CORRECTIONS**. The implementation design faithfully follows spec section 7.3 with correct cell/range resolution, merged cell handling, type coercion, confidence scoring, and validation pattern application. Two issues require correction before PATCH: (1) the CHECKBOX coercion list includes `True`/`False` Python booleans beyond what the spec lists -- this is acceptable as a defensive addition but should be documented as an extension, and (2) the plan's `_read_range` signature returns `(str, list[str])` but should return `(Any, list[str])` since range values fed into NUMBER/DATE coercion should not be pre-stringified. One minor gap exists in test coverage: no test for `data_only=True` formula cell behavior.

---

## Check 1: Single Cells, Ranges (D5:D7), and Merged Cells

**Spec section 7.3 step 2a-c**:
- Single cell: resolve sheet, read `ws[cell_coord].value`
- Range (`":"` in cell): read all cells in range, join non-empty with newline
- Merged cells: detect via merged ranges, read from top-left cell of merge

**Plan coverage**:
- `_read_cell_value()`: dispatches single vs range based on `":"` in `field.cell_address.cell` -- CORRECT
- `_read_range()`: reads all cells in range, joins non-empty with newline -- CORRECT
- `_resolve_merged_cell()`: checks `ws.merged_cells.ranges` for containment, reads top-left -- CORRECT
- Sheet resolution: `wb[field.cell_address.sheet_name]` if specified, else `wb.active` -- CORRECT

**Issue found**: `_read_range()` returns `(str, list[str])` per the plan signature. If a range is used for a NUMBER field, the joined string would then be passed to `float()` coercion, which would fail. However, this is an unlikely edge case since ranges are primarily used for multi-line text fields. The plan should document that range values are always treated as TEXT before coercion, or alternatively return the first non-empty value when the field type is NUMBER/DATE.

**Status**: PASS WITH NOTE -- Range + NUMBER/DATE coercion edge case should be documented.

---

## Check 2: Type Coercion for All FieldTypes

**Spec section 7.3 step 2d**:
| FieldType | Spec Coercion | Plan Coercion | Match |
|-----------|--------------|---------------|-------|
| NUMBER | `float(value)`; fails -> `None` | `float(value)`; fails -> `None`, emit `W_FORM_FIELD_TYPE_COERCION` | PASS |
| DATE | `datetime` parse; fails -> `None` | `isinstance(value, datetime)` check first, string fallback parse; fails -> `None` | PASS |
| CHECKBOX | "X", "x", "Yes", "TRUE", 1 -> True; empty, "No", "FALSE", 0, None -> False | Same + `True`/`False` booleans added | PASS (extension) |
| TEXT | `str(value).strip()` | `str(value).strip()` | PASS |
| DROPDOWN | not explicitly listed | `str(value)` (same as TEXT) | PASS |
| RADIO | not explicitly listed | `str(value)` (same as TEXT) | PASS |
| SIGNATURE | not explicitly listed | treat as TEXT | PASS |

**CHECKBOX extension note**: The plan adds Python `True`/`False` booleans to the checkbox coercion mappings. The spec only lists string/numeric values ("X", "x", "Yes", "TRUE", 1 and "No", "FALSE", 0, None). However, openpyxl returns native Python booleans for cells formatted as boolean, so this is a correct defensive addition. The warning code `W_FORM_FIELD_TYPE_COERCION` is correctly emitted on NUMBER coercion failure.

**Status**: PASS

---

## Check 3: Confidence Scoring (0.95 / 0.0 / 0.95)

**Spec section 7.3 step 2e**:
| Condition | Spec Confidence | Plan Confidence | Warning | Match |
|-----------|----------------|-----------------|---------|-------|
| Cell has value | 0.95 | 0.95 | None | PASS |
| Cell empty + required | 0.0 | 0.0 | `W_FORM_FIELD_MISSING_REQUIRED` | PASS |
| Cell empty + optional | 0.95, value=default_value | 0.95, value=field.default_value | None | PASS |

**extraction_method**: Plan sets `"cell_mapping"` -- matches spec step 2f.

**Status**: PASS -- All three confidence branches match spec exactly.

---

## Check 4: W_FORM_MERGED_CELL_RESOLVED Emission

**Spec section 7.3 (line 867)**: "The extractor detects merged ranges and reads from the canonical (top-left) cell."

**Plan implementation** (step 2c): When `_resolve_merged_cell()` returns `was_merged=True`, the plan emits `W_FORM_MERGED_CELL_RESOLVED` in the warnings list.

**Error code verified**: `FormErrorCode.W_FORM_MERGED_CELL_RESOLVED` exists in `errors.py` line 79.

**Status**: PASS

---

## Check 5: validation_pattern Application

**Spec section 7.3 step 3**: "Apply validation_pattern regex if defined."

**Plan implementation** (step 3): `_validate_field_value()` applies `re.fullmatch(field.validation_pattern, str(value))`. On failure: `validation_passed=False`, emit `W_FORM_FIELD_VALIDATION_FAILED`. On success: `validation_passed=True`. If no pattern: `validation_passed=None`.

**Comparison with NativePDFExtractor pattern**: Plan states "reuse same pattern as NativePDFExtractor._validate_field_value" which is correct -- the existing extractor has this same logic.

**Missing detail**: The plan does not mention ReDoS protection for validation_pattern. The NativePDFExtractor (issue #56) and OCROverlayExtractor (issue #64) both include threading-based regex timeout. The Excel cell extractor should follow the same pattern for consistency, even though Excel cell values are typically short strings with low ReDoS risk.

**Status**: PASS WITH NOTE -- Consider adding ReDoS timeout for consistency with other extractors, though risk is low for cell values.

---

## Check 6: Test Coverage (17 Tests)

**Test matrix analysis**:

| # | Test | Covers | Status |
|---|------|--------|--------|
| 1 | `test_extract_text_field_single_cell` | Single cell TEXT, confidence=0.95, method="cell_mapping" | PASS |
| 2 | `test_extract_number_field_coercion` | NUMBER coercion success | PASS |
| 3 | `test_extract_number_field_coercion_failure` | NUMBER coercion fail -> None, W_FORM_FIELD_TYPE_COERCION | PASS |
| 4 | `test_extract_date_field` | DATE from datetime object | PASS |
| 5 | `test_extract_checkbox_true_values` | CHECKBOX True variants (parametrized) | PASS |
| 6 | `test_extract_checkbox_false_values` | CHECKBOX False variants (parametrized) | PASS |
| 7 | `test_extract_cell_range` | Range "D5:D7" -> newline join | PASS |
| 8 | `test_extract_cell_range_skips_empty` | Range with empty cells | PASS |
| 9 | `test_merged_cell_resolution` | Merged cell -> top-left, W_FORM_MERGED_CELL_RESOLVED | PASS |
| 10 | `test_empty_required_field` | Empty + required -> confidence=0.0, W_FORM_FIELD_MISSING_REQUIRED | PASS |
| 11 | `test_empty_optional_field` | Empty + optional -> confidence=0.95, value=default | PASS |
| 12 | `test_validation_pattern_pass` | Regex match -> validation_passed=True | PASS |
| 13 | `test_validation_pattern_fail` | Regex fail -> validation_passed=False, W_FORM_FIELD_VALIDATION_FAILED | PASS |
| 14 | `test_sheet_name_resolution` | Explicit sheet_name routing | PASS |
| 15 | `test_sheet_name_not_found` | Invalid sheet -> value=None, confidence=0.0 | PASS |
| 16 | `test_multiple_fields_extraction` | 3+ fields in order | PASS |
| 17 | `test_skips_pdf_fields` | Fields with region (no cell_address) are skipped | PASS |

**Coverage gaps**:
| Area | Covered | Gap |
|------|---------|-----|
| All relevant FieldTypes | TEXT, NUMBER, DATE, CHECKBOX | DROPDOWN, RADIO, SIGNATURE not explicitly tested (coerce same as TEXT, low risk) |
| Error paths | Sheet not found, coercion failure, validation failure | Missing: corrupt workbook -> FormIngestException |
| data_only=True | Not tested | Missing: formula cell returning None (cached value not saved) |
| PII-safe logging | Not tested | Missing: verify no raw cell data logged when log_sample_data=False |

**Assessment**: 17 tests provide solid coverage for the core algorithm. The gaps are low-risk since DROPDOWN/RADIO/SIGNATURE share TEXT coercion logic, and corrupt workbook handling is a single try/except. Adding 2-3 more tests (corrupt file, formula cell, PII logging) would bring coverage to comprehensive.

**Status**: PASS -- 17 tests are sufficient. Optional additions noted above.

---

## Issues Found

### Issue 1: MINOR -- `_read_range` Return Type

**Method**: `_read_range(self, ws, range_str: str) -> tuple[str, list[str]]`

**Problem**: Return type is `str` but the joined result will be passed to `_coerce_value()`. If a range is mapped to a NUMBER field, `float("10\n20\n30")` would fail. This is an edge case (ranges are typically for multi-line text), but the type signature should be `tuple[Any, list[str]]` for consistency, or the plan should document that ranges always produce TEXT values regardless of field_type.

### Issue 2: MINOR -- Missing ReDoS Protection in Validation

The NativePDFExtractor and OCROverlayExtractor both use threading-based regex timeout (1s) for `validation_pattern`. The plan mentions reusing the same pattern but does not explicitly include the timeout mechanism. PATCH should implement the same `_regex_match_with_timeout()` approach.

### Issue 3: MINOR -- Missing Corrupt Workbook Test

The plan documents error handling for corrupt files (raise `FormIngestException` with `E_FORM_FILE_CORRUPT`) but no test covers this path. Adding `test_corrupt_workbook_raises_exception` would verify this.

---

## Recommendations

1. **Clarify range + non-TEXT coercion behavior** -- either document that ranges produce TEXT-only values or handle the edge case in `_coerce_value`.
2. **Add ReDoS timeout** to `_validate_field_value` for consistency with other extractors.
3. **Optionally add 2-3 tests** for corrupt workbook, formula cells, and PII-safe logging.
4. **Proceed with implementation** -- the design is sound and spec-compliant.

---

## Acceptance Criteria Status (from MAP-PLAN)

| # | Criterion | Status |
|---|-----------|--------|
| 1 | ExcelCellExtractor class with extract(file_path, template) | PASS |
| 2 | Single cell resolution with sheet_name routing | PASS |
| 3 | Range resolution (D5:D7) with newline join | PASS |
| 4 | Merged cell detection and top-left read | PASS |
| 5 | W_FORM_MERGED_CELL_RESOLVED emitted for merged cells | PASS |
| 6 | Type coercion for NUMBER, DATE, CHECKBOX, TEXT | PASS |
| 7 | Confidence: 0.95 (value), 0.0 (empty required), 0.95 (empty optional + default) | PASS |
| 8 | extraction_method = "cell_mapping" | PASS |
| 9 | validation_pattern regex applied | PASS |
| 10 | Error handling for missing sheets and invalid cells | PASS |
| 11 | ExcelCellExtractor added to __init__.py exports | PASS |
| 12 | 17 unit tests with mock openpyxl | PASS |
| 13 | No concrete backend imports (openpyxl is core dep, acceptable) | PASS |

---

## Summary

**Plan status**: APPROVED WITH CORRECTIONS

**Corrections required** (2 items, both minor):
1. Add ReDoS protection to `_validate_field_value` for consistency with NativePDF and OCR extractors
2. Clarify `_read_range` return type and range + non-TEXT coercion behavior

**Optional improvements** (3 items):
1. Add corrupt workbook test
2. Add formula cell (data_only=True returning None) test
3. Add PII-safe logging test

**Risk level**: LOW
- openpyxl merged cell API is well-documented and used elsewhere in the codebase (parser_chain.py)
- No network calls, no backend protocol needed
- Algorithm is deterministic and straightforward

**Ready for PATCH phase**: YES (with corrections noted above)

---

AGENT_RETURN: .agents/outputs/plan-check-65-021526.md
