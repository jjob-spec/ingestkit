---
issue: 64
agent: PLAN-CHECK
date: 2026-02-15
complexity: COMPLEX
stack: backend
---

# PLAN-CHECK: Issue #64 -- OCR Overlay Extraction for Scanned/Image Forms

**Plan artifact**: `.agents/outputs/plan-64-021526.md`
**Map artifact**: `.agents/outputs/map-64-021526.md`
**Issue**: #64 -- Implement OCR overlay extraction for scanned/image forms
**Date**: 2026-02-15

---

## Executive Summary

The plan is **APPROVED WITH CORRECTIONS**. The implementation design faithfully follows spec section 7.2 with correct preprocessing pipeline, field-type dispatch, confidence calculations, and security controls. Three issues require correction before PATCH: (1) the `test_field_without_region_fails_closed` test is impossible due to the `FieldMapping` model validator requiring either `region` or `cell_address`, (2) spec section 7.2 step 2c lists only "deskew, contrast enhancement" for TEXT fields but the plan adds bilateral noise reduction and binarization from the preprocessing pipeline diagram -- this is acceptable as the diagram is more detailed, but the plan should acknowledge the distinction, and (3) the plan imports `signal` in `ocr_overlay.py` but never uses it (uses threading for ReDoS timeout instead).

---

## Check 1: Preprocessing Pipeline vs Spec section 7.2

**Spec section 7.2 preprocessing pipeline** (lines 806-827):
```
Deskew -> CLAHE -> Bilateral filter (noise reduction) -> Adaptive threshold (binarization) -> OCR
```

**Plan `preprocess_for_ocr` pipeline** (Plan section 1.6):
- TEXT/NUMBER/DATE: `deskew -> enhance_contrast (CLAHE) -> reduce_noise (bilateral) -> adaptive_threshold`
- CHECKBOX/RADIO/SIGNATURE: `adaptive_threshold` only

**Spec section 7.2 step 2c field-specific preprocessing**:
- TEXT: "deskew, contrast enhancement"
- NUMBER: "deskew, contrast, apply digits-only OCR config"
- DATE: "deskew, contrast, apply date-format hint"
- CHECKBOX: "convert to binary, check fill ratio"
- SIGNATURE: "convert to binary, check ink ratio"

**Analysis**: The step 2c bullet points for TEXT/NUMBER/DATE mention only deskew + contrast, but the full preprocessing pipeline diagram (lines 806-827) shows the complete 4-step chain (deskew, CLAHE, bilateral, binarization). The plan applies the full 4-step chain for text types, which aligns with the more detailed pipeline diagram. This is correct -- the step 2c bullets are abbreviated.

**Status**: PASS

---

## Check 2: Field-Type-Specific Processing Paths (7 FieldTypes)

| FieldType | Plan Handling | Spec Match | Status |
|-----------|--------------|------------|--------|
| TEXT | Full preprocessing -> OCR -> post-process -> confidence=mean(char_confidences) | section 7.2 step 2c-d | PASS |
| NUMBER | Full preprocessing -> OCR with digit whitelist -> strip non-numeric | section 7.2 step 2c | PASS |
| DATE | Full preprocessing -> OCR with date hint -> strip whitespace | section 7.2 step 2c | PASS |
| CHECKBOX | adaptive_threshold -> compute_fill_ratio -> compare to 0.3 threshold | section 7.2 step 2e | PASS |
| RADIO | Same as CHECKBOX (plan section 3.8 dispatches RADIO to `_extract_checkbox_field`) | section 7.2 step 2e | PASS |
| SIGNATURE | adaptive_threshold -> compute_ink_ratio -> compare to 0.05 threshold | section 7.2 step 2f | PASS |
| DROPDOWN | Fail-closed with `E_FORM_OCR_FAILED` | Correct -- DROPDOWN not applicable to OCR overlay | PASS |

**Status**: PASS -- All 7 FieldTypes handled correctly.

---

## Check 3: Checkbox Fill Ratio Threshold (0.3) and Signature Ink Ratio Threshold (0.05)

**Spec section 7.2**:
- Checkbox: `fill_ratio > 0.3 = checked` (line 776)
- Signature: `ink_ratio > 0.05 = signed` (line 778)
- Checkbox confidence: `abs(fill_ratio - 0.3) / 0.3, capped at 1.0` (line 787)
- Signature confidence: `abs(ink_ratio - 0.05) / 0.05, capped at 1.0` (line 791)

**Plan section 3.10-3.11**:
- Checkbox: `fill_ratio > threshold` where threshold = `config.checkbox_fill_threshold` (default 0.3)
- Signature: `ink_ratio > threshold` where threshold = `config.signature_ink_threshold` (default 0.05)
- Both: `min(abs(ratio - threshold) / threshold, 1.0)` with guard `if threshold > 0`

**Config defaults verified** (`config.py` lines 143-154):
- `checkbox_fill_threshold: float = 0.3`
- `signature_ink_threshold: float = 0.05`

**Status**: PASS -- Thresholds and confidence formulas match spec exactly.

---

## Check 4: Per-Field Timeout

**Spec section 18.1** (referenced in MAP): "<2s per OCR field, 10s timeout"
**Config** (`config.py` line 88): `form_ocr_per_field_timeout_seconds: int = 10`

**Plan implementation** (section 3.9):
- `timeout=float(self._config.form_ocr_per_field_timeout_seconds)` passed to `ocr_backend.ocr_region()`
- `TimeoutError` caught, returns `(None, 0.0, None)`

**Protocol signature** (`protocols.py` line 149): `timeout: float | None = None`

**Status**: PASS -- Timeout propagated correctly.

---

## Check 5: Validation Pattern Enforcement with ReDoS Protection

**Spec section 7.2 step 3**: "If validation fails: set value to None, confidence to 0.0, add W_FORM_FIELD_VALIDATION_FAILED warning."

**Spec section 13.5**: "Test that a ReDoS pattern times out within 1s and does not hang the extraction pipeline."

**Plan implementation** (section 3.3):
- `_regex_match_with_timeout()` uses `threading.Thread` with `join(timeout=1.0)`
- On timeout: returns `None`, triggering `value=None, confidence=0.0, W_FORM_FIELD_VALIDATION_FAILED`
- On match failure: same handling
- On match success: value kept

**Status**: PASS -- ReDoS protection and validation failure handling match spec.

---

## Check 6: Security -- Decompression Bomb Protection and Resolution Limits

**Spec section 13.4**:
1. "Images where `decompressed_size / compressed_size > 100` are rejected"
2. "Images wider or taller than 10000 pixels are rejected"
3. "The plugin only processes local files. It never fetches images from URLs."

**Plan implementation** (section 2.2-2.4):
- `MAX_DECOMPRESSION_RATIO = 100` -- matches spec
- `MAX_IMAGE_DIMENSION = 10_000` -- matches spec
- `validate_image_safety()` checks both before loading
- `render_pdf_page()` also validates rendered dimensions post-rendering
- No URL loading anywhere in the plan

**Status**: PASS -- All three security controls covered.

---

## Check 7: No Concrete Backend Imports in Extractor

**Plan `ocr_overlay.py` imports** (section 3.1):
- `OCRBackend` used only under `TYPE_CHECKING` guard
- No import of Tesseract, PaddleOCR, or any concrete engine
- `fitz` (PyMuPDF) imported only inside `render_pdf_page()` in `_rendering.py`, not in the extractor itself
- OCR engine accessed exclusively via `self._ocr.ocr_region()` protocol method
- Engine name accessed via `self._ocr.engine_name()` for config string selection

**Status**: PASS -- Protocol-only access throughout.

---

## Check 8: Test Coverage (46 Tests)

**Plan test count breakdown**:
- Preprocessing: 12 tests
- Rendering: 6 tests
- OCROverlayExtractor: 19 tests
- Security: 3 tests
- PII safety: 2 tests
- Helper functions: 4 tests
- **Total: 46 tests**

**Coverage gaps analysis**:

| Area | Covered | Gap |
|------|---------|-----|
| All 7 FieldTypes | TEXT, NUMBER, DATE, CHECKBOX, RADIO (via checkbox), SIGNATURE, DROPDOWN (fail-closed) | None |
| Error handling | OCR failure, timeout, page render failure | None |
| Security | Decompression bomb, resolution, ReDoS | Missing: magic byte mismatch (spec 13.5 lists it but it is a separate concern from this issue) |
| PII logging | Both modes tested | None |
| Edge cases | Empty region, multi-page, small images, partial fill | None |

**Issue found**: `test_field_without_region_fails_closed` (test 18 in section 6.5) creates a `FieldMapping` with `region=None`. However, the `FieldMapping` model validator (`models.py` lines 141-158) requires **exactly one** of `region` or `cell_address` to be set. Creating a `FieldMapping(region=None, cell_address=None)` will raise a `ValueError` during model construction, not during extraction.

**Correction required**: This test must either:
- (a) Use a `FieldMapping` with `cell_address` set (simulating an Excel field passed to OCR by mistake), OR
- (b) Use `model_construct()` to bypass validation, OR
- (c) Mock the field object to have `region=None` without triggering the validator.

Option (a) is most realistic -- it tests what happens when an Excel-addressed field reaches the OCR extractor.

**Status**: PASS WITH CORRECTION -- 1 test needs adjustment (see above).

---

## Check 9: Adding numpy to pyproject.toml

**Current `pyproject.toml` dependencies** (lines 10-15):
```toml
dependencies = [
    "ingestkit-core>=0.1.0",
    "pydantic>=2.0",
    "openpyxl>=3.1",
    "Pillow>=10.0",
]
```

**Plan**: Add `numpy>=1.24` to the dependencies list.

**Analysis**:
- numpy IS a transitive dependency of Pillow (via internal C extensions, not a pip-visible dependency) and of all OCR backends
- Making it explicit is the right call since `_preprocessing.py` imports it directly
- `numpy>=1.24` is compatible with Python >=3.10 (numpy 1.24 supports Python 3.8-3.11, but 1.24+ is fine for 3.10+)
- No version conflicts with Pillow>=10.0 or other deps
- numpy is NOT actually a pip dependency of Pillow -- Pillow uses its own C code. numpy is optional for Pillow's array interface. Making it explicit is necessary.

**Status**: PASS -- Adding numpy is correct and necessary.

---

## Issues Found

### Issue 1: CRITICAL -- Impossible Test Case

**Test**: `test_field_without_region_fails_closed` (section 6.5, test 18)

**Problem**: `FieldMapping` has a model validator that raises `ValueError` if both `region` and `cell_address` are `None`. The test cannot create such an object without bypassing Pydantic validation.

**Fix**: Use a `FieldMapping` with `cell_address=CellAddress(cell="B2")` and `region=None` to simulate an Excel field reaching the OCR extractor. This is a realistic scenario.

### Issue 2: MINOR -- Unused Import

**File**: `ocr_overlay.py` (section 3.1) imports `signal` and `time` but neither is used. The plan uses `threading.Thread` for ReDoS timeout and passes timeout to the OCR backend. Remove `signal` and `time` from imports.

### Issue 3: MINOR -- Spec Ambiguity Acknowledged

The spec step 2c lists abbreviated preprocessing for TEXT ("deskew, contrast") while the preprocessing pipeline diagram shows the full 4-step chain. The plan correctly follows the full pipeline diagram. No code change needed, but PATCH should add a comment noting this design decision.

---

## Recommendations

1. **Fix test_field_without_region_fails_closed** to use a `FieldMapping` with `cell_address` set instead of attempting to create one with both `None`.
2. **Remove unused `signal` and `time` imports** from `ocr_overlay.py`.
3. **Proceed with implementation** -- the design is thorough, spec-compliant, and security-conscious.

---

## Acceptance Criteria Status (from PLAN)

| # | Criterion | Status |
|---|-----------|--------|
| 1 | OCROverlayExtractor accepts OCRBackend protocol only | PASS |
| 2 | All image ops use Pillow + numpy, no OpenCV | PASS |
| 3 | PyMuPDF import is lazy, handles ImportError | PASS |
| 4 | Per-field timeout propagated to ocr_region() | PASS |
| 5 | Checkbox fill_ratio > 0.3 -> True | PASS |
| 6 | Signature ink_ratio > 0.05 -> True | PASS |
| 7 | Confidence = min(abs(ratio-threshold)/threshold, 1.0) | PASS |
| 8 | Text confidence = mean(char_confidences) | PASS |
| 9 | Validation failure -> value=None, confidence=0.0, warning | PASS |
| 10 | No raw OCR text in logs when log_ocr_output=False | PASS |
| 11 | Decompression bomb check + resolution limit 10000px | PASS |
| 12 | ReDoS protection with 1s thread timeout | PASS |
| 13 | All tests use mock OCRBackend | PASS |
| 14 | Tests marked @pytest.mark.unit | PASS |
| 15 | numpy>=1.24 added to pyproject.toml | PASS |
| 16 | OCROverlayExtractor exported from __init__.py | PASS |
| 17 | Page images released after processing | PASS |
| 18 | Fail-closed on extraction failure | PASS |

---

## Summary

**Plan status**: APPROVED WITH CORRECTIONS

**Corrections required** (2 items, both minor):
1. Fix `test_field_without_region_fails_closed` -- use `cell_address` instead of `region=None`
2. Remove unused `signal` and `time` imports from `ocr_overlay.py`

**Risk level**: LOW-MEDIUM
- numpy dependency addition is safe (already transitive)
- CLAHE and bilateral implementations are custom but have Pillow fallbacks
- Threading-based ReDoS timeout is platform-safe (unlike signal-based)

**Ready for PATCH phase**: YES (with corrections noted above)

---

AGENT_RETURN: .agents/outputs/plan-check-64-021526.md
