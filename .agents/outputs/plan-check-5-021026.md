---
issue: 5
agent: PLAN-CHECK
date: 2026-02-10
complexity: SIMPLE
validation_result: PASS
---

# PLAN-CHECK - Issue #5

## Executive Summary

The MAP-PLAN artifact for Issue #5 is **VALID** and ready for implementation. All acceptance criteria are mapped to concrete test cases, file scope is within the SIMPLE tier (3 files: 1 create parser_chain.py, 1 create test_parser_chain.py, 1 modify __init__.py), and the approach correctly implements the three-tier fallback chain per SPEC.md section 7.

---

## VALIDATION RESULTS

### 1. Requirement Coverage ✓

All acceptance criteria from Issue #5 are mapped to planned tasks:

| Acceptance Criterion | Mapped Task | Evidence |
|---|---|---|
| Clean .xlsx parsed successfully with openpyxl (primary) | `test_clean_xlsx_uses_openpyxl_primary` | MAP-PLAN line 221 |
| Corrupted-for-openpyxl file falls back to pandas | `test_openpyxl_fail_falls_back_to_pandas` | MAP-PLAN line 223 |
| All-parsers-fail produces E_PARSE_CORRUPT | `test_all_parsers_fail_produces_e_parse_corrupt` | MAP-PLAN line 225 |
| Per-sheet fallback independence verified | `test_per_sheet_fallback_independence` | MAP-PLAN line 227 |
| fallback_reason_code correctly set per sheet | `test_fallback_reason_code_correctly_set` | MAP-PLAN line 229 |
| W_PARSER_FALLBACK in warnings when fallback used | `test_w_parser_fallback_in_warnings` | MAP-PLAN line 231 |
| Password-protected, chart-only, empty edge cases handled | `test_password_protected_file`, `test_chart_only_sheet_skipped`, `test_empty_file` | MAP-PLAN lines 233-237 |
| pytest tests/test_parser_chain.py -q passes | All 12 test cases | MAP-PLAN lines 219-243 |

**Status:** COMPLETE - All 8 acceptance criteria have explicit test case mappings.

---

### 2. Scope Containment (SIMPLE tier = max 5 files) ✓

**Files identified:**

1. **Create:** `/home/jjob/projects/ingestkit/packages/ingestkit-excel/src/ingestkit_excel/parser_chain.py` (NEW)
2. **Create:** `/home/jjob/projects/ingestkit/packages/ingestkit-excel/tests/test_parser_chain.py` (NEW)
3. **Modify:** `/home/jjob/projects/ingestkit/packages/ingestkit-excel/src/ingestkit_excel/__init__.py` (EXISTING)

**Count:** 3 files. **Result:** PASS (well below 5-file limit for SIMPLE)

**Impact Analysis Summary:**
- No modifications needed to `models.py`, `errors.py`, `config.py` -- all required enums, types, and constants already exist (verified by MAP-PLAN lines 22-30).
- No circular import risk: `parser_chain.py` imports from `models`, `errors`, `config` (all stable); `__init__.py` imports from `parser_chain` (standard public API pattern).
- Test fixtures can be created programmatically within `test_parser_chain.py` using openpyxl (no dependency on external fixture files).

**Status:** PASS - Scope is tight and focused.

---

### 3. Pattern Pre-Check ✓

**N/A for this issue.** This is a backend-only implementation with:
- No enum values being defined (only imported and used: `ParserUsed.OPENPYXL`, `ParserUsed.PANDAS_FALLBACK`, `ParserUsed.RAW_TEXT_FALLBACK`).
- No role/status/type field enums that could trigger ENUM_VALUE pattern (26%).
- No frontend component reuse (COMPONENT_API pattern, 17%).
- No assumption gaps requiring VERIFICATION_GAP checks (all code paths examined in MAP-PLAN).

**Status:** N/A - Core failure patterns do not apply.

---

### 4. Wiring Completeness ✓

#### 4a. ParserChain Export from __init__.py

**Requirement:** `ParserChain` must be exported from the public API.

**Mapped Solution:** MAP-PLAN lines 197-203:
```
Add import: from ingestkit_excel.parser_chain import ParserChain
Add "ParserChain" to __all__ list
```

**Status:** VERIFIED - Explicitly planned with clear instructions.

#### 4b. Test Coverage Completeness

**Acceptance Criterion:** `pytest tests/test_parser_chain.py -q` passes (all edge cases).

**Test Cases Planned (12 total):**

1. ✓ Primary parser success: `test_clean_xlsx_uses_openpyxl_primary` (line 221)
2. ✓ Openpyxl→Pandas fallback: `test_openpyxl_fail_falls_back_to_pandas` (line 223)
3. ✓ All parsers fail: `test_all_parsers_fail_produces_e_parse_corrupt` (line 225)
4. ✓ Per-sheet independence: `test_per_sheet_fallback_independence` (line 227)
5. ✓ Fallback reason code: `test_fallback_reason_code_correctly_set` (line 229)
6. ✓ Fallback warning: `test_w_parser_fallback_in_warnings` (line 231)
7. ✓ Password protection: `test_password_protected_file` (line 233)
8. ✓ Chart-only sheets: `test_chart_only_sheet_skipped` (line 235)
9. ✓ Empty files: `test_empty_file` (line 237)
10. ✓ Row truncation warning: `test_rows_truncated_warning` (line 239)
11. ✓ Content hash: `test_content_hash_computed` (line 241)
12. ✓ FileProfile aggregation: `test_file_profile_aggregate_fields` (line 243)

**Edge Cases Covered:**
- Password-protected files (line 233, 256)
- Chart-only sheets (line 235, 256)
- Empty files (line 237, 256)
- Row truncation (line 239, 256)

**Status:** COMPLETE - 12 test cases covering all acceptance criteria and edge cases.

#### 4c. Implementation Architecture Soundness

**ParserChain Class Structure** (MAP-PLAN lines 114-124):
- `__init__(self, config: ExcelProcessorConfig)` — configurable instance
- `parse(self, file_path: str) -> tuple[FileProfile, list[IngestError]]` — public interface matching SPEC section 7, line 523-562
- Private methods for each tier: `_parse_sheet_openpyxl`, `_parse_sheet_pandas`, `_parse_sheet_raw_text`
- Supporting methods: `_compute_content_hash`, `_build_sheet_profile_from_*`

**Fallback Chain Logic** (MAP-PLAN lines 126-150):
1. Compute content hash (line 129)
2. Validate file not empty (line 130)
3. Try openpyxl (line 131) → if fails, go to step 4
4. Try pandas (line 140) → if fails, go to step 5
5. Try raw text (line 143) → if fails, `E_PARSE_CORRUPT` (line 145)
6. Per-sheet independence: loop (line 134) runs the fallback chain independently for each sheet

**FileProfile Aggregation** (MAP-PLAN lines 147-149):
- Combines all `SheetProfile` objects
- Aggregates: `total_merged_cells` (sum), `total_rows` (sum), `sheet_count`, `sheet_names`
- Includes: `content_hash`, `file_size_bytes`, `has_password_protected_sheets`, `has_chart_only_sheets`

**Status:** VERIFIED - Architecture aligns with SPEC section 7 requirements.

#### 4d. Error & Warning Codes

**Verified in MAP-PLAN (lines 22-27):**
- `E_PARSE_CORRUPT`, `E_PARSE_OPENPYXL_FAIL`, `E_PARSE_PANDAS_FAIL`, `E_PARSE_PASSWORD`, `E_PARSE_EMPTY` defined in `errors.py`
- `W_SHEET_SKIPPED_CHART`, `W_SHEET_SKIPPED_PASSWORD`, `W_PARSER_FALLBACK`, `W_ROWS_TRUNCATED` defined in `errors.py`
- `IngestError` model with `code`, `message`, `sheet_name`, `stage`, `recoverable` fields exists

**Plan Usage:**
- `E_PARSE_CORRUPT` (line 225, 251)
- `E_PARSE_OPENPYXL_FAIL`, `E_PARSE_PANDAS_FAIL` (lines 139-144, fallback chain)
- `E_PARSE_PASSWORD` (line 132, 256)
- `E_PARSE_EMPTY` (line 130, 237, 256)
- `W_PARSER_FALLBACK` (line 141, 144, 231, 256)
- `W_SHEET_SKIPPED_CHART` (line 135, 235, 256)
- `W_ROWS_TRUNCATED` (line 136, 239, 256)

**Status:** VERIFIED - All error/warning codes used in plan are pre-defined in models/errors.

---

## GATE CHECKLIST

### Gate 1: Module Creation ✓
- ✓ `src/ingestkit_excel/parser_chain.py` creation planned (MAP-PLAN lines 50-56)
- ✓ `ParserChain` class with `__init__(config)` and `parse(file_path)` signatures (lines 115-116)
- ✓ Import strategy documented (lines 102-112)
- ✓ No circular import risk: imports are from stable leaf modules

### Gate 2: Core Functionality ✓
- ✓ Clean xlsx parsing with openpyxl as primary (lines 137-138)
- ✓ `SheetProfile` field population via `_parse_sheet_openpyxl` (lines 155-169)
- ✓ `FileProfile` aggregation logic detailed (lines 147-149)
- ✓ `content_hash` computation via SHA-256 (lines 151-153, pattern from idempotency.py line 63-64)

### Gate 3: Fallback Chain ✓
- ✓ Three-tier chain (openpyxl → pandas → raw_text) (lines 137-145)
- ✓ `E_PARSE_CORRUPT` on all failures (line 145)
- ✓ `W_PARSER_FALLBACK` warning added (lines 141, 144)
- ✓ `fallback_reason_code` recorded (lines 139, 142, 145)
- ✓ Per-sheet independence via loop (line 134)

### Gate 4: Edge Cases ✓
- ✓ Password-protected: `E_PARSE_PASSWORD` at line 132
- ✓ Chart-only sheets: `W_SHEET_SKIPPED_CHART` at line 135
- ✓ Empty files: `E_PARSE_EMPTY` at line 130
- ✓ Oversized sheets: `W_ROWS_TRUNCATED` at line 136

### Gate 5: Integration ✓
- ✓ `__init__.py` export planned (lines 197-203)
- ✓ Test suite with 12 test cases (lines 219-243)
- ✓ Test execution target: `pytest tests/test_parser_chain.py -q` (line 256)
- ✓ No modifications to existing modules (`models.py`, `errors.py`, `config.py`) — no regressions expected

---

## RISK ANALYSIS VALIDATION

The MAP-PLAN identifies 5 key risks at lines 69-90. Assessment:

| Risk | Mitigation Provided | Adequacy | Notes |
|---|---|---|---|
| **Risk 1: Password detection** | Catch openpyxl exceptions, handle both file-level and sheet-level (lines 72-73) | ADEQUATE | openpyxl raises specific exception; test will monkeypatch this (line 233) |
| **Risk 2: Chart-only detection** | Check `isinstance(sheet, Chartsheet)` (line 76-77) | ADEQUATE | Clear type check; test case provided (line 235) |
| **Risk 3: Pandas doesn't provide all fields** | Set reduced-fidelity fields to defaults (0 / 0.0 / False); `parser_used` signals reduced fidelity (line 80-81) | ADEQUATE | SheetProfile fields will have sensible defaults; fidelity tracked via `parser_used` enum |
| **Risk 4: Raw text fallback also fails** | This is expected terminal scenario; `E_PARSE_CORRUPT` is correct (line 84-85) | ADEQUATE | By design; test covers (line 225) |
| **Risk 5: Row count detection** | For openpyxl, check `ws.max_row` before iterate; for pandas, check `len(df)` after load (line 88-89) | ADEQUATE | Specific guidance; test case provided (line 239) |

**Status:** All risks mitigated with clear implementation guidance.

---

## DETAILED VERIFICATION

### Acceptance Criterion Traceability

1. **"Clean .xlsx parsed successfully with openpyxl (primary)"**
   - Test case: `test_clean_xlsx_uses_openpyxl_primary` (line 221)
   - Implementation: Lines 137-138 (try openpyxl first)
   - Verification: Fixture creates valid xlsx; parser used must be `ParserUsed.OPENPYXL`; no errors

2. **"Corrupted-for-openpyxl file falls back to pandas"**
   - Test case: `test_openpyxl_fail_falls_back_to_pandas` (line 223)
   - Implementation: Lines 139-141 (on openpyxl failure, try pandas)
   - Verification: Monkeypatch openpyxl to raise; verify pandas fallback; check `parser_used == ParserUsed.PANDAS_FALLBACK`

3. **"All-parsers-fail produces E_PARSE_CORRUPT"**
   - Test case: `test_all_parsers_fail_produces_e_parse_corrupt` (line 225)
   - Implementation: Lines 142-145 (on pandas failure, try raw_text; if all fail, error)
   - Verification: Random bytes as .xlsx; verify `E_PARSE_CORRUPT` in returned errors

4. **"Per-sheet fallback independence verified"**
   - Test case: `test_per_sheet_fallback_independence` (line 227)
   - Implementation: Lines 134-145 (for-each-sheet loop with independent fallback chain)
   - Verification: Multi-sheet workbook; sheet 1 parses with openpyxl, sheet 2 with pandas; verify different `parser_used` values

5. **"fallback_reason_code correctly set per sheet"**
   - Test case: `test_fallback_reason_code_correctly_set` (line 229)
   - Implementation: Lines 139, 142 (record failed parser error code)
   - Verification: Fallback error has `fallback_reason_code` matching the failed parser's error

6. **"W_PARSER_FALLBACK in warnings when fallback used"**
   - Test case: `test_w_parser_fallback_in_warnings` (line 231)
   - Implementation: Lines 141, 144 (add `W_PARSER_FALLBACK` warning)
   - Verification: Any fallback produces `W_PARSER_FALLBACK` in error list

7. **"Password-protected, chart-only, empty edge cases handled"**
   - Test cases: `test_password_protected_file` (line 233), `test_chart_only_sheet_skipped` (line 235), `test_empty_file` (line 237)
   - Implementation: Lines 130-136
   - Verification: Each edge case has explicit handling and test

8. **"pytest tests/test_parser_chain.py -q passes"**
   - Test suite: 12 test cases (lines 219-243)
   - Verification: All tests must pass; no failures

**Status:** All 8 acceptance criteria have explicit implementation sections and test case mappings.

---

## READINESS SUMMARY

| Aspect | Status | Notes |
|---|---|---|
| Requirement Coverage | ✓ COMPLETE | All 8 acceptance criteria mapped to test cases |
| Scope (SIMPLE tier) | ✓ PASS | 3 files (below 5-file limit) |
| Pattern Pre-Check | ✓ N/A | No enum/component/verification issues |
| Module Wiring | ✓ VERIFIED | ParserChain export planned; test coverage complete |
| Error Codes | ✓ VERIFIED | All error/warning codes pre-defined in models/errors |
| Architecture | ✓ SOUND | Three-tier chain, per-sheet independence, aggregation logic clear |
| Risk Mitigation | ✓ ADEQUATE | All 5 risks have documented mitigations |
| Implementation Clarity | ✓ HIGH | Detailed pseudocode, method signatures, field mapping provided |

**OVERALL VALIDATION: PASS** ✓

---

## FINAL RECOMMENDATION

The MAP-PLAN artifact is **ready for implementation**. All acceptance criteria are clearly mapped to concrete implementation steps and test cases. The scope is appropriate for SIMPLE complexity, and the architecture correctly implements the SPEC.md section 7 requirements (three-tier fallback chain, per-sheet independence, error/warning handling, edge cases).

**Next Step:** Proceed to IMPLEMENT phase.

---

**Validation Report Generated:** 2026-02-10
**Validator:** PLAN-CHECK Agent
**Model:** Claude Haiku 4.5
