---
issue: 67
agent: PLAN-CHECK
date: 2026-02-15
plan_artifact: map-plan-67-021526.md
verdict: APPROVED with 2 WARNINGS
---

# PLAN-CHECK: Issue #67 -- Per-field confidence scoring and overall aggregation

## Executive Summary

The MAP-PLAN for issue #67 is well-structured and correctly identifies the four gaps between the current code and spec section 7.4. Investigation claims were verified against source code and the spec. Two warnings are raised: (1) the spec pseudocode uses `FieldMapping()` as a default but that would fail Pydantic validation, and the plan correctly identifies this but the proposed fix needs to be explicit in the implementation; (2) existing test assertions for hardcoded `confidence=0.95` in `test_extractors.py` will break and the plan mentions this but does not enumerate the specific lines that need updating.

## Check 1: Confidence scoring rules vs spec 7.4 table

| Spec Row | Plan Coverage | Status |
|---|---|---|
| Native PDF fields: 0.90-0.99 | `compute_field_confidence("native_fields", ...)` clamps to 0.90-0.99, deducts 0.02 for coercion | MATCH |
| OCR (clean/poor): 0.40-0.95 | Pass-through of raw OCR confidence (char averaging already in `ocr_overlay.py`) | MATCH |
| Excel cell mapping: 0.90-0.99 | Clamp to 0.90-0.99, deduct 0.02 for coercion | MATCH |
| Checkbox/radio (fill ratio): 0.60-0.99 | Pass-through (already correct in OCR extractor) | MATCH |
| VLM fallback | Pass-through of raw VLM confidence | MATCH |

Verdict: All 5 rows from the spec table are covered.

## Check 2: Confidence-based actions (4 tiers)

| Spec Action | Plan Tier | Status |
|---|---|---|
| conf >= min_field (0.5) -> accept | Tier 1: accept, no warnings | MATCH |
| conf >= vlm_threshold (0.4) AND < min_field -> accept + warning | Tier 2: accept + `W_FORM_FIELD_LOW_CONFIDENCE` | MATCH |
| conf < vlm_threshold AND vlm_enabled -> trigger VLM | Tier 3: mark field for VLM (flag, not invoke) | MATCH -- correctly defers actual VLM call to router |
| conf < vlm_threshold AND vlm not enabled -> value=None + warning | Tier 4: value=None + `W_FORM_FIELD_LOW_CONFIDENCE` | MATCH |

Verdict: All 4 actions are correct. The plan correctly notes that actual VLM invocation is out of scope -- it only marks the field.

## Check 3: `compute_overall_confidence` weighted mean (required 2x)

The plan states: "Verbatim from spec section 7.4 pseudocode."

Verified against spec lines 892-907: the spec pseudocode computes `weighted_sum / max(total_weight, 1.0)` where required fields get `weight = 2.0` and optional fields get `weight = 1.0`. The plan's test case confirms this:
- 2 required fields at 0.8 + 1 optional at 0.4 -> `(0.8*2 + 0.8*2 + 0.4*1) / (2+2+1) = 3.6/5 = 0.72`

WARNING: The spec pseudocode uses `FieldMapping()` as fallback default (line 903), but `FieldMapping` has a `model_validator` requiring either `region` or `cell_address` (models.py lines 141-158). The plan identifies this at line 177 and proposes defaulting to `weight=1.0` when field_id is not found. This is the correct fix, but the PATCH agent must ensure this deviation from spec pseudocode is clearly implemented (e.g., `weight = 2.0 if field_map.get(ef.field_id) and field_map[ef.field_id].required else 1.0`).

Verdict: MATCH with one WARNING about FieldMapping default handling.

## Check 4: Fail-closed gate (< 0.3 -> E_FORM_EXTRACTION_LOW_CONFIDENCE)

The plan states the gate check "will live in the router" and tests only that the function returns a value below threshold. This aligns with:
- Spec line 165: "If overall extraction quality is too low, return `ProcessingResult` with `E_FORM_EXTRACTION_LOW_CONFIDENCE` and zero chunks."
- Spec line 1947: "Extraction overall confidence < threshold -> Return result with error, zero chunks written -> `E_FORM_EXTRACTION_LOW_CONFIDENCE`"
- Config line 1636-1642: `form_extraction_min_overall_confidence: float = 0.3`

The plan correctly places the gate in the router (not in `confidence.py`) and provides the `compute_overall_confidence` function that produces the value the router will check. The error code `E_FORM_EXTRACTION_LOW_CONFIDENCE` already exists in `errors.py` (line 38).

Verdict: MATCH. The gate is correctly scoped -- function in `confidence.py`, gate check deferred to router.

## Check 5: Regression risk from extractor changes

**NativePDFExtractor (`native_pdf.py`):**
- Line 112 changes from `confidence=0.95` to variable confidence via `compute_field_confidence`. Two existing tests in `test_extractors.py` assert `confidence == 0.95` (lines 183 and 207). The plan mentions updating these in File 5 but does not list the specific assertions.
- Risk: LOW if tests are updated. The plan accounts for this.

**OCROverlayExtractor (`ocr_overlay.py`):**
- Lines 293-295 change from a simple warning append to a call to `apply_confidence_actions()`. This changes behavior: currently, low-confidence fields keep their value and get a warning. Under the new 4-tier system, fields below `vlm_threshold` with VLM disabled will have `value=None`. This is a behavioral change for fields with confidence < 0.4.
- Risk: MEDIUM. Existing OCR tests that set up low-confidence scenarios (< 0.4) will now see `value=None` instead of the OCR'd value. The plan must ensure test updates cover this.

WARNING: The plan's File 5 section is vague ("if needed" and "Update existing ... tests"). PATCH should enumerate the specific test functions that assert on confidence values or low-confidence field behavior in `test_extractors.py`.

Verified specific lines at risk:
- `test_extractors.py:183` -- asserts `confidence == 0.95` (native PDF)
- `test_extractors.py:207` -- asserts `confidence == 0.95` (native PDF)
- Any OCR tests with confidence < 0.4 that currently expect a non-None value

Verdict: Acceptable risk if PATCH updates tests. WARNING on vagueness of File 5.

## Check 6: Test case comprehensiveness

| Category | Test Cases in Plan | Coverage |
|---|---|---|
| `compute_field_confidence` | 5 cases (native, native+coercion, OCR pass-through, cell mapping, checkbox) | Adequate |
| `compute_overall_confidence` | 4 cases (equal, weighted, empty, None values) | Adequate |
| `apply_confidence_actions` | 4 cases (one per tier) | Adequate |
| Overall gate | 1 case (value < threshold) | Adequate |
| Extractor test updates | Mentioned but not enumerated | WARNING -- see Check 5 |

Missing test cases to consider (not blocking):
- `compute_field_confidence` with unknown extraction method (should pass through or raise?)
- `compute_overall_confidence` where all fields are optional (weight=1.0 only)
- `apply_confidence_actions` at exact boundary values (confidence == vlm_threshold, confidence == min_field)
- Interaction between validation failure (sets confidence=0.0) and confidence actions

Verdict: Core coverage is good. Boundary tests recommended but not blocking.

## Summary

| Check | Verdict |
|---|---|
| 1. Confidence ranges match spec table | PASS |
| 2. Four confidence-based actions correct | PASS |
| 3. Weighted mean with 2x required | PASS (WARNING: FieldMapping default) |
| 4. Fail-closed gate at < 0.3 | PASS |
| 5. Regression risk | PASS (WARNING: enumerate test updates) |
| 6. Test comprehensiveness | PASS |

**Overall: APPROVED** -- The plan is ready for PATCH. Two non-blocking warnings:

1. PATCH must handle the `FieldMapping()` default problem explicitly (cannot instantiate bare `FieldMapping`). Use `weight = 2.0 if (fm := field_map.get(ef.field_id)) and fm.required else 1.0`.
2. PATCH must update specific test assertions in `test_extractors.py` lines 183 and 207 (and any OCR tests affected by the value=None behavior change for confidence < 0.4).

AGENT_RETURN: .agents/outputs/plan-check-67-021526.md
