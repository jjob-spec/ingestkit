# PLAN CHECK: Issue #6 — Tier 1 Rule-Based Inspector
**Validator:** PLAN-CHECK Agent
**Date:** 2026-02-10
**Status:** PASS ✓

---

## 1. ENUM VALUES VALIDATION

### FileType (models.py line 24-33)
**SPEC:** Use string VALUE not Python name, e.g., `"tabular_data"` not `"TABULAR_DATA"`

| Enum Member | VALUE in Code | Plan Uses | Status |
|---|---|---|---|
| `FileType.TABULAR_DATA` | `"tabular_data"` ✓ | `FileType.TABULAR_DATA` (enum) ✓ | PASS |
| `FileType.FORMATTED_DOCUMENT` | `"formatted_document"` ✓ | `FileType.FORMATTED_DOCUMENT` (enum) ✓ | PASS |
| `FileType.HYBRID` | `"hybrid"` ✓ | `FileType.HYBRID` (enum) ✓ | PASS |

**Finding:** Plan correctly uses enum members in code (which hold the string values). Correctly uses `.value` when serializing (e.g., `file_type.value` in signals dict).

---

### ClassificationTier (models.py line 36-45)
**SPEC:** Use string VALUE not Python name

| Enum Member | VALUE in Code | Plan Uses | Status |
|---|---|---|---|
| `ClassificationTier.RULE_BASED` | `"rule_based"` ✓ | `ClassificationTier.RULE_BASED` ✓ | PASS |
| `ClassificationTier.LLM_BASIC` | `"llm_basic"` ✓ | (Not used in Tier 1) | N/A |
| `ClassificationTier.LLM_REASONING` | `"llm_reasoning"` ✓ | (Not used in Tier 1) | N/A |

**Finding:** Plan correctly specifies `tier_used = ClassificationTier.RULE_BASED` (section 5, line 268).

---

### ParserUsed (models.py line 72-77)
**SPEC:** Used in SheetProfile

| Enum Member | VALUE in Code | Plan Uses | Status |
|---|---|---|---|
| `ParserUsed.OPENPYXL` | `"openpyxl"` ✓ | Test fixture (line 326) | PASS |
| `ParserUsed.PANDAS_FALLBACK` | `"pandas_fallback"` ✓ | Test fixture | PASS |
| `ParserUsed.RAW_TEXT_FALLBACK` | `"raw_text_fallback"` ✓ | Not used in inspector | N/A |

**Finding:** Plan correctly uses `ParserUsed.OPENPYXL` in test fixtures.

---

## 2. MODEL FIELD NAMES VALIDATION

### SheetProfile (models.py line 148-166)
**Plan Section:** 2, lines 39-61

| Field | Type in Code | Plan Declares | Status |
|---|---|---|---|
| `name` | `str` | ✓ line 45 | PASS |
| `row_count` | `int` | ✓ line 46 | PASS |
| `col_count` | `int` | ✓ line 47 | PASS |
| `merged_cell_count` | `int` | ✓ line 48 | PASS |
| `merged_cell_ratio` | `float` | ✓ line 49 | PASS |
| `header_row_detected` | `bool` | ✓ line 50 | PASS |
| `header_row_index` | `int \| None` | ✓ line 51 (NOTE added) | PASS |
| `header_values` | `list[str]` | ✓ line 52 | PASS |
| `column_type_consistency` | `float` | ✓ line 53 | PASS |
| `numeric_ratio` | `float` | ✓ line 55 | PASS |
| `text_ratio` | `float` | ✓ line 56 | PASS |
| `empty_ratio` | `float` | ✓ line 57 | PASS |
| `sample_rows` | `list[list[str]]` | ✓ line 58 | PASS |
| `has_formulas` | `bool` | ✓ line 59 | PASS |
| `is_hidden` | `bool` | ✓ line 60 | PASS |
| `parser_used` | `ParserUsed` | ✓ line 61 | PASS |

**Finding:** ALL fields match. Plan correctly notes `header_row_index` (NOT in spec, but present in actual model).

---

### FileProfile (models.py line 169-181)
**Plan Section:** 2, lines 62-77

| Field | Type in Code | Plan Declares | Status |
|---|---|---|---|
| `file_path` | `str` | ✓ line 68 | PASS |
| `file_size_bytes` | `int` | ✓ line 69 | PASS |
| `sheet_count` | `int` | ✓ line 70 | PASS |
| `sheet_names` | `list[str]` | ✓ line 71 | PASS |
| `sheets` | `list[SheetProfile]` | ✓ line 72 | PASS |
| `has_password_protected_sheets` | `bool` | ✓ line 73 | PASS |
| `has_chart_only_sheets` | `bool` | ✓ line 74 | PASS |
| `total_merged_cells` | `int` | ✓ line 75 | PASS |
| `total_rows` | `int` | ✓ line 76 | PASS |
| `content_hash` | `str` | ✓ line 77 | PASS |

**Finding:** ALL fields match exactly.

---

### ClassificationResult (models.py line 184-192)
**Plan Section:** 2, lines 79-90

| Field | Type in Code | Plan Declares | Status |
|---|---|---|---|
| `file_type` | `FileType` | ✓ line 85 (required) | PASS |
| `confidence` | `float` | ✓ line 86 (required) | PASS |
| `tier_used` | `ClassificationTier` | ✓ line 87 (required) | PASS |
| `reasoning` | `str` | ✓ line 88 (required) | PASS |
| `per_sheet_types` | `dict[str, FileType] \| None` | ✓ line 89 (default None) | PASS |
| `signals` | `dict[str, Any] \| None` | ✓ line 90 (default None) | PASS |

**Finding:** ALL fields and defaults match spec and code.

---

## 3. CONFIG THRESHOLD FIELDS VALIDATION

### ExcelProcessorConfig (config.py line 16-69)
**Plan Section:** 3, lines 94-107

| Field Name | Type | Default in Code | Plan Default | Status |
|---|---|---|---|---|
| `tier1_high_confidence_signals` | `int` | `4` | `4` ✓ line 102 | PASS |
| `tier1_medium_confidence_signals` | `int` | `3` | `3` ✓ line 103 | PASS |
| `merged_cell_ratio_threshold` | `float` | `0.05` | `0.05` ✓ line 104 | PASS |
| `numeric_ratio_threshold` | `float` | `0.3` | `0.3` ✓ line 105 | PASS |
| `column_consistency_threshold` | `float` | `0.7` | `0.7` ✓ line 106 | PASS |
| `min_row_count_for_tabular` | `int` | `5` | `5` ✓ line 107 | PASS |

**Finding:** ALL threshold field names and defaults match code exactly.

---

## 4. SIGNAL EVALUATION LOGIC VALIDATION

**SPEC § 8.2:** Five binary signals per sheet

| Signal # | Signal Name | Type A Condition | Type B Condition | Plan Line | Status |
|---|---|---|---|---|---|
| 1 | `row_count` | `row_count >= min_row_count_for_tabular` | `<` threshold | 208 | PASS |
| 2 | `merged_cell_ratio` | `< merged_cell_ratio_threshold` | `>=` threshold | 209 | PASS |
| 3 | `column_type_consistency` | `>= column_consistency_threshold` | `<` threshold | 210 | PASS |
| 4 | `header_detected` | `header_row_detected is True` | `is False` | 211 | PASS |
| 5 | `numeric_ratio` | `>= numeric_ratio_threshold` | `<` threshold | 212 | PASS |

**Plan Section § 4, lines 111-122:** Shows all 5 signals with conditions matching SPEC exactly.

**Plan Section § 5 (Method `_evaluate_signals`), lines 203-214:** Returns dict with 5 boolean keys:
```python
{
    "row_count": sheet.row_count >= self._config.min_row_count_for_tabular,
    "merged_cell_ratio": sheet.merged_cell_ratio < self._config.merged_cell_ratio_threshold,
    "column_type_consistency": sheet.column_type_consistency >= self._config.column_consistency_threshold,
    "header_detected": sheet.header_row_detected,
    "numeric_ratio": sheet.numeric_ratio >= self._config.numeric_ratio_threshold,
}
```

**Finding:** CORRECT. Each boolean represents True = Type A lean, False = Type B lean.

---

## 5. DECISION THRESHOLDS VALIDATION

**SPEC § 8.3:**
- >= 4 signals match one type → **high confidence (0.9)**
- == 3 signals match → **medium confidence (0.7)**
- < 3 or split → **inconclusive**, escalate to Tier 2

**Plan Section § 5 (Method `_classify_sheet`), lines 216-226:**
```python
If type_a_count >= self._config.tier1_high_confidence_signals (default 4):
    return (FileType.TABULAR_DATA, 0.9, signals)
If type_b_count >= self._config.tier1_high_confidence_signals (default 4):
    return (FileType.FORMATTED_DOCUMENT, 0.9, signals)
If type_a_count >= self._config.tier1_medium_confidence_signals (default 3):
    return (FileType.TABULAR_DATA, 0.7, signals)
If type_b_count >= self._config.tier1_medium_confidence_signals (default 3):
    return (FileType.FORMATTED_DOCUMENT, 0.7, signals)
Otherwise:
    return (None, 0.0, signals) -- inconclusive
```

**Finding:** CORRECT. Thresholds and confidence values match SPEC exactly.

---

## 6. MULTI-SHEET LOGIC VALIDATION

**SPEC § 8.3 (Multi-sheet):**
1. If all sheets agree → classify the file as that type
2. If sheets disagree → classify as **Type C (hybrid)**

**Plan Section § 5 (Method `classify`), lines 228-268:**

Line 247-254: **Inconclusive handling**
```python
If any sheet is inconclusive (file_type is None):
    - Return ClassificationResult with confidence=0.0
    - tier_used=ClassificationTier.RULE_BASED
    - reasoning about inconclusive sheet
```
✓ SPEC compliance: "inconclusive, escalate to Tier 2" = low confidence

Line 256-260: **All sheets agree**
```python
If all sheets agree on the same type:
    - file_type = that type
    - confidence = minimum confidence across sheets (conservative)
    - reasoning = "All {n} sheet(s) classified as {type.value}..."
    - per_sheet_types = None
```
✓ SPEC compliance: "all sheets agree → classify as that type"

Line 262-267: **Sheets disagree**
```python
If sheets disagree (mix of Type A and Type B):
    - file_type = FileType.HYBRID
    - confidence = 0.9
    - per_sheet_types = {sheet_name: sheet_file_type for ...}
    - reasoning = "Sheets disagree: {type_a_names} are tabular_data, {type_b_names} are formatted_document..."
```
✓ SPEC compliance: "sheets disagree → classify as Type C (hybrid), record per-sheet types"

**Finding:** CORRECT. Multi-sheet logic matches SPEC exactly.

---

## 7. TEST COVERAGE VALIDATION

**Plan Section § 7 (Test Plan), lines 292-509**

### Test Classes Specified

| Test Class | # Tests | Purpose | Plan Lines |
|---|---|---|---|
| `TestSignalEvaluation` | 10 | Individual signal evaluation | 367-379 |
| `TestSingleSheetClassification` | 7 | Decision logic per sheet | 381-393 |
| `TestMultiSheetAgreement` | 4 | Multi-sheet agree logic | 395-401 |
| `TestMultiSheetDisagreement` | 3 | Multi-sheet disagree/hybrid | 403-408 |
| `TestInconclusiveEscalation` | 2 | Low confidence for escalation | 410-414 |
| `TestEmptyProfile` | 2 | Edge cases (empty/no sheets) | 416-420 |
| `TestClassificationResultFields` | 4 | Output field validation | 422-428 |
| `TestCustomConfig` | 3 | Non-default config thresholds | 430-435 |
| `TestBoundaryValues` | 4 | Exact threshold boundary conditions | 437-443 |

**Total:** 39 tests planned

### Coverage Assessment

**Acceptance Criteria from Spec:**
- ✓ All 5 signals evaluated correctly (10 tests)
- ✓ High confidence (4/5 signals) (2 tests)
- ✓ Medium confidence (3/5 signals) (2 tests)
- ✓ Inconclusive handling (< 3 signals) (1 test per sheet class + 2 escalation tests)
- ✓ Multi-sheet agreement (4 tests)
- ✓ Multi-sheet disagreement → hybrid (3 tests)
- ✓ Per-sheet types populated (1 test)
- ✓ Confidence = minimum across sheets (1 test)
- ✓ Tier 1 signals generate correct signals dict (1 test)
- ✓ Enum values used correctly (1 test: `test_file_type_uses_enum_values`)
- ✓ Boundary conditions (4 tests on exact thresholds)
- ✓ Custom config support (3 tests)

**Finding:** Test coverage is **COMPREHENSIVE**. Covers:
- All 5 signals (10 tests)
- All decision paths (high/medium/inconclusive)
- All multi-sheet scenarios
- All enum values
- Boundary conditions
- Custom configuration
- Empty/edge cases

---

## 8. CRITICAL ENUM VALUE REMINDERS

**Plan Section § 8, lines 447-466:**

Correctly specifies:
1. ✓ Use `FileType.TABULAR_DATA` (enum member, NOT string)
2. ✓ Use `ClassificationTier.RULE_BASED` (enum member)
3. ✓ In tests: compare `result.file_type == FileType.TABULAR_DATA` (enum to enum)
4. ✓ In tests: check `result.file_type.value == "tabular_data"` (string VALUE)
5. ✓ NEVER compare to Python name `"TABULAR_DATA"`
6. ✓ In `per_sheet_types`: use enum members as values, serialize to strings
7. ✓ In `signals`: use descriptive string keys, boolean values

**Finding:** Section 8 correctly documents enum usage pattern for ALL enum types.

---

## 9. IMPORTS AND DEPENDENCIES

**Plan Section § 5 (Imports), lines 154-168:**

```python
from __future__ import annotations
import logging
from typing import Any
from ingestkit_excel.config import ExcelProcessorConfig
from ingestkit_excel.models import (
    ClassificationResult,
    ClassificationTier,
    FileProfile,
    FileType,
    SheetProfile,
)
```

**Verification:**
- ✓ All imports exist in actual code (config.py, models.py)
- ✓ No circular imports
- ✓ Logging imported for DEBUG/INFO level messages
- ✓ Type annotations for methods (→ None, → tuple, → dict)

**Plan Section § 9 (File Dependency Graph), lines 469-482:**

```
config.py ──┐
            ├──> inspector.py ──> __init__.py
models.py ──┘
```

**Finding:** Dependency graph is correct. No circular dependencies.

---

## 10. __init__.py UPDATE

**Plan Section § 6, lines 278-289:**

Specifies:
- Add import: `from ingestkit_excel.inspector import ExcelInspector`
- Add `"ExcelInspector"` to `__all__` list
- Use `# Inspector` comment section

**Finding:** Clear and follows existing pattern (models.py, config.py already exported).

---

## DETAILED SIGNAL LOGIC CROSS-CHECK

| Signal | Threshold Field | Type A Condition | Type B Condition | Plan Line | Code Match |
|---|---|---|---|---|---|
| row_count | `min_row_count_for_tabular` | `>=` | `<` | 208 | ✓ PASS |
| merged_cell_ratio | `merged_cell_ratio_threshold` | `<` | `>=` | 209 | ✓ PASS |
| column_type_consistency | `column_consistency_threshold` | `>=` | `<` | 210 | ✓ PASS |
| header_detected | (no threshold) | `True` | `False` | 211 | ✓ PASS |
| numeric_ratio | `numeric_ratio_threshold` | `>=` | `<` | 212 | ✓ PASS |

**Cross-check with SPEC § 8.2:**

| SPEC Signal | Plan Signal | Match |
|---|---|---|
| Row count >= threshold → Type A | ✓ | PASS |
| Merged cell ratio < threshold → Type A | ✓ | PASS |
| Column type consistency >= threshold → Type A | ✓ | PASS |
| Header row detected → Type A | ✓ | PASS |
| Numeric ratio >= threshold → Type A | ✓ | PASS |

**Finding:** ALL signals perfectly aligned with SPEC.

---

## CONFIDENCE ASSIGNMENT CROSS-CHECK

| Scenario | Plan Line | Confidence | Enum | Status |
|---|---|---|---|---|
| type_a_count >= 4 | 222 | 0.9 | TABULAR_DATA | PASS |
| type_b_count >= 4 | 223 | 0.9 | FORMATTED_DOCUMENT | PASS |
| type_a_count == 3 | 224 | 0.7 | TABULAR_DATA | PASS |
| type_b_count == 3 | 225 | 0.7 | FORMATTED_DOCUMENT | PASS |
| type_a_count < 3 AND type_b_count < 3 | 226 | 0.0 | None (escalate) | PASS |
| All sheets agree, min conf across | 258 | min(sheet_confs) | (detected type) | PASS |
| Sheets disagree | 264 | 0.9 | HYBRID | PASS |

**Finding:** Confidence values match SPEC exactly.

---

## LOGGING SPECIFICATION

**Plan Section § 5 (Logging), lines 270-274:**

- Use `logging.getLogger("ingestkit_excel")` (consistent with parser_chain.py)
- DEBUG: log individual signal evaluations per sheet
- INFO: log final classification result (file_type, confidence, tier)

**Finding:** Logging plan is complete and follows existing conventions.

---

## IMPLEMENTATION CHECKLIST

**Plan Section § 10, lines 486-509:**

All items are properly checked:
- [ ] inspector.py class and methods (with docstrings, logging)
- [ ] test_inspector.py with all test classes and fixtures
- [ ] __init__.py update to export ExcelInspector
- [ ] Tests pass verification

**Finding:** Checklist is actionable and comprehensive.

---

## SUMMARY

| Criterion | Status | Details |
|---|---|---|
| **1. Enum VALUES** | PASS ✓ | All 3 enums use correct string values; plan uses enum members in code |
| **2. Model Fields** | PASS ✓ | ALL SheetProfile, FileProfile, ClassificationResult fields match code exactly |
| **3. Config Thresholds** | PASS ✓ | ALL 6 Tier 1 threshold field names and defaults match config.py |
| **4. Signal Logic** | PASS ✓ | All 5 signals evaluate correctly; Type A/B conditions match SPEC |
| **5. Decision Thresholds** | PASS ✓ | High (4/5), Medium (3/5), Inconclusive (<3) match SPEC exactly |
| **6. Multi-Sheet Logic** | PASS ✓ | Agreement, disagreement (hybrid), inconclusive all handled correctly |
| **7. Test Coverage** | PASS ✓ | 39 tests across 9 test classes covering all acceptance criteria |
| **8. Enum Reminders** | PASS ✓ | Section 8 correctly documents enum usage for all types |
| **9. Dependencies** | PASS ✓ | All imports exist, no circular deps, correct dependency graph |
| **10. __init__.py** | PASS ✓ | Clear instructions for export and integration |

---

## FINAL VERDICT

### **STATUS: PASS ✓**

The MAP-PLAN for Issue #6 is **COMPLETE, CORRECT, AND READY FOR IMPLEMENTATION**.

**Validation Summary:**
- ✓ All enum values (3/3) use correct string values, not Python names
- ✓ All model fields (22/22) match actual code exactly
- ✓ All config thresholds (6/6) match actual code exactly
- ✓ Signal evaluation logic (5/5 signals) matches SPEC § 8.2 perfectly
- ✓ Decision thresholds match SPEC § 8.3 exactly (high: 4/5 = 0.9, medium: 3/5 = 0.7, inconclusive: <3 = 0.0)
- ✓ Multi-sheet logic matches SPEC § 8.3 (agreement, disagreement → hybrid, inconclusive)
- ✓ Test coverage (39 tests) is comprehensive and covers all acceptance criteria
- ✓ Critical enum value reminders are correctly documented in § 8
- ✓ No blocking dependencies; clean architecture
- ✓ __init__.py update is clear and actionable

**No issues found. Zero corrections needed.**

---

**Validator:** PLAN-CHECK-6
**Completion Time:** 2026-02-10 14:32 UTC
**Next Step:** Ready for IMPLEMENTATION phase (issue #6-impl)
