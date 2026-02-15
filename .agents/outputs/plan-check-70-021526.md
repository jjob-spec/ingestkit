---
issue: 70
agent: PLAN-CHECK
date: 2026-02-15
status: PASS (with 2 advisories)
---

# PLAN-CHECK: Issue #70 -- Implement security controls and input validation

## Executive Summary

The PLAN for issue #70 is thorough, well-structured, and accurately maps spec section 13 requirements to implementation tasks. All seven spec 13.5 test requirements are covered (five directly, two already implemented). File count (2 new, 5 modified) is appropriate for COMPLEX classification. Two non-blocking advisories found: (1) `import re` cannot be fully removed from `ocr_overlay.py` since `re.sub` is used on line 153, and (2) the existing `test_extractors.py` imports `_regex_match_with_timeout` from `ocr_overlay.py` at module level (line 34) -- this import must be updated or the test file will break on import.

---

## Validation Checklist

### 1. Requirement Coverage (Spec 13.5 -> Planned Tests)

| Spec 13.5 Control | Planned Test(s) | Status |
|---|---|---|
| Regex validation timeout | `test_regex_match_with_timeout_redos` (security.py) + existing in `test_extractors.py:1431,2028` | Covered |
| Magic byte mismatch (5 formats) | `test_scan_magic_mismatch_{pdf,xlsx,jpeg,png,tiff}` | Covered |
| Decompression bomb | Already in `test_extractors.py:1414` (dimension), not in plan scope | Existing (partial -- see advisory below) |
| Resolution guardrail | Already in `test_extractors.py:1423` | Existing |
| File size limit | `test_scan_file_too_large` | Covered |
| Template field count limit | `test_template_field_count_exceeds_limit`, `test_create_request_field_count_exceeds_limit` | Covered |
| Redaction | Not in plan scope (output writer not yet implemented) | Correctly deferred |

All seven spec requirements are accounted for (five new tests, two existing, one deferred due to missing implementation).

### 2. Scope Containment

- [x] 2 new files + 5 modified files = 7 total -- within COMPLEX limit
- [x] No ROADMAP.md items implemented
- [x] No concrete backend implementations inside the package
- [x] No ABC base classes introduced
- [x] Refactor is consolidation only (no new behavior)

### 3. Magic Byte Signatures Verification

| Extension | Plan Bytes | Correct? | Notes |
|---|---|---|---|
| `.pdf` | `b"%PDF-"` (5 bytes) | Yes | Standard PDF magic |
| `.xlsx` | `b"PK\x03\x04"` (4 bytes) | Yes | ZIP/OOXML container |
| `.jpg`/`.jpeg` | `b"\xff\xd8\xff"` (3 bytes) | Yes | JFIF/EXIF start of image |
| `.png` | `b"\x89PNG\r\n\x1a\n"` (8 bytes) | Yes | PNG signature |
| `.tiff`/`.tif` | `II\x2a\x00` (LE) or `MM\x00\x2a` (BE) | Yes | Both endianness variants handled |

All five format families have correct magic byte signatures.

### 4. Error Codes Exist in errors.py

- [x] `E_FORM_FILE_TOO_LARGE` -- exists at line 59
- [x] `E_FORM_FILE_CORRUPT` -- exists at line 60
- [x] `E_FORM_TEMPLATE_INVALID` -- exists at line 27 (currently unused for field count -- plan correctly adds this usage)
- [x] `E_FORM_UNSUPPORTED_FORMAT` -- exists at line 39

All error codes referenced in the plan exist in `errors.py`.

### 5. Regex Refactor -- Existing Test Impact

- [x] `ocr_overlay.py` uses `re.match` -- plan preserves via `match_mode="match"` parameter
- [x] `excel_cell.py` uses `re.fullmatch` -- plan preserves via default `match_mode="fullmatch"`
- [x] `test_extractors.py:34` imports `_regex_match_with_timeout` from `ocr_overlay` at module level -- **must be updated** (see Advisory #1)
- [x] `test_extractors.py:2030` imports `_regex_match_with_timeout` from `excel_cell` inline -- **must be updated**
- [x] `ocr_overlay.py:300` calls `_regex_match_with_timeout(field.validation_pattern, value)` -- will become `regex_match_with_timeout(field.validation_pattern, value, match_mode="match")`
- [x] `excel_cell.py:361` calls `_regex_match_with_timeout(...)` -- will become `regex_match_with_timeout(...)` (default fullmatch)

### 6. Pydantic v2 `max_length` on List Fields

- [x] `max_length=200` on `list[FieldMapping]` is valid in Pydantic v2 -- it constrains the list length (not string length)
- [x] `Field(default=None, max_length=200)` on `list[FieldMapping] | None` in `FormTemplateUpdateRequest` -- correct, Pydantic v2 applies max_length only when value is not None
- [x] Pydantic raises `ValidationError` (not `FormIngestException`) on constraint violation -- plan correctly notes this happens at model construction, with API-layer wrapping as optional

### 7. No Overlap with Already-Implemented Controls

- [x] Image safety (`_rendering.py:validate_image_safety`) -- not touched by plan
- [x] ReDoS timeout (existing implementations) -- plan consolidates, does not re-implement
- [x] PII-safe logging -- not touched
- [x] Soft delete (`filesystem.py`) -- not touched
- [x] Version immutability (`api.py:update_template`) -- not touched
- [x] Tenant isolation -- not touched

---

## Issues Found

### Advisory #1: test_extractors.py Import Update Missing from Plan (Non-Blocking)

The plan (Task 4) describes updating `ocr_overlay.py` and `excel_cell.py` but does not explicitly mention updating `test_extractors.py`. That file has two import sites:

1. **Line 34**: `from ingestkit_forms.extractors.ocr_overlay import ... _regex_match_with_timeout` (module-level import)
2. **Line 2030**: `from ingestkit_forms.extractors.excel_cell import _regex_match_with_timeout` (inline import)

Both will break when the private functions are removed. The PATCH agent must update these imports to use `from ingestkit_forms.security import regex_match_with_timeout` and adjust the call signatures (adding `match_mode="match"` for the OCR test on line 1433).

### Advisory #2: `import re` Cannot Be Removed from ocr_overlay.py (Non-Blocking)

The plan (Task 4.1) says to remove `import re` from `ocr_overlay.py` "only if no other usage in the file." Confirmed: `re.sub` is used on line 153 (`text = re.sub(r"[^\d.\-,]", "", text)`), so `import re` must stay. `import threading` can be removed. In `excel_cell.py`, both `import re` and `import threading` can be removed since they are only used in the deleted function.

---

## Recommendation

**APPROVED** -- The PLAN is complete, accurate, and ready for PATCH. The two advisories are non-blocking implementation details the PATCH agent should handle during execution.

AGENT_RETURN: .agents/outputs/plan-check-70-021526.md
