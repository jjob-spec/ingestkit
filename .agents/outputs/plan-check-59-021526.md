---
issue: 59
agent: PLAN-CHECK
date: 2026-02-15
complexity: SIMPLE
stack: backend
---

# PLAN-CHECK: Issue #59 -- FormProcessorConfig with validation

**Plan artifact**: `.agents/outputs/map-plan-59-021526.md`
**Issue**: #59 -- Implement FormProcessorConfig with validation
**Date**: 2026-02-15

---

## Executive Summary

The plan is **APPROVED with minor corrections**.

All major checks passed:
- Field count verified: 43 fields across 16 groups, matching spec section 11 code block (lines 1526-1724) exactly.
- model_validator cross-field checks are complete (5 enum-like string fields + 1 threshold ordering constraint).
- `from_file()` follows the established PDF/Excel pattern.
- RedactTarget enum values use correct lowercase string values (ENUM_VALUE check passed).
- Default values match spec.
- Acceptance criteria are complete and testable.

**Minor corrections**:
1. `from_file()` import style: Spec uses lazy `import json as json_mod` inside the method body. Plan references the PDF pattern which imports `json` at module top-level. PATCH should follow the spec's version (local import inside method).
2. Plan groups fields into 14 categories; spec actually has 16 comment-delimited sections (plan merges "Native PDF Field Matching" into "Field Extraction" and omits the section count discrepancy). Cosmetic only -- no impact on implementation.

---

## Check 1: Config Parameter Coverage (43 fields)

### Field-by-field verification against spec lines 1534-1724

| # | Group | Field | Spec Default | In Plan? | Status |
|---|-------|-------|-------------|----------|--------|
| 1 | Identity | `parser_version` | `"ingestkit_forms:1.0.0"` | Yes | PASS |
| 2 | Identity | `tenant_id` | `None` | Yes | PASS |
| 3 | Form Matching | `form_match_enabled` | `True` | Yes | PASS |
| 4 | Form Matching | `form_match_confidence_threshold` | `0.8` | Yes | PASS |
| 5 | Form Matching | `form_match_per_page_minimum` | `0.6` | Yes | PASS |
| 6 | Form Matching | `form_match_extra_page_penalty` | `0.02` | Yes | PASS |
| 7 | Form Matching | `page_match_strategy` | `"windowed"` | Yes | PASS |
| 8 | Fingerprinting | `fingerprint_dpi` | `150` | Yes | PASS |
| 9 | Fingerprinting | `fingerprint_grid_rows` | `20` | Yes | PASS |
| 10 | Fingerprinting | `fingerprint_grid_cols` | `16` | Yes | PASS |
| 11 | OCR | `form_ocr_dpi` | `300` | Yes | PASS |
| 12 | OCR | `form_ocr_engine` | `"paddleocr"` | Yes | PASS |
| 13 | OCR | `form_ocr_language` | `"en"` | Yes | PASS |
| 14 | OCR | `form_ocr_per_field_timeout_seconds` | `10` | Yes | PASS |
| 15 | Native PDF | `pdf_widget_backend` | `"pymupdf"` | Yes | PASS |
| 16 | VLM Fallback | `form_vlm_enabled` | `False` | Yes | PASS |
| 17 | VLM Fallback | `form_vlm_model` | `"qwen2.5-vl:7b"` | Yes | PASS |
| 18 | VLM Fallback | `form_vlm_fallback_threshold` | `0.4` | Yes | PASS |
| 19 | VLM Fallback | `form_vlm_timeout_seconds` | `15` | Yes | PASS |
| 20 | VLM Fallback | `form_vlm_max_fields_per_document` | `10` | Yes | PASS |
| 21 | Field Extraction | `form_extraction_min_field_confidence` | `0.5` | Yes | PASS |
| 22 | Field Extraction | `form_extraction_min_overall_confidence` | `0.3` | Yes | PASS |
| 23 | Field Extraction | `checkbox_fill_threshold` | `0.3` | Yes | PASS |
| 24 | Field Extraction | `signature_ink_threshold` | `0.05` | Yes | PASS |
| 25 | Native PDF Matching | `native_pdf_iou_threshold` | `0.5` | Yes | PASS |
| 26 | Output DB | `form_db_table_prefix` | `"form_"` | Yes | PASS |
| 27 | Chunking | `chunk_max_fields` | `20` | Yes | PASS |
| 28 | Embedding | `embedding_model` | `"nomic-embed-text"` | Yes | PASS |
| 29 | Embedding | `embedding_dimension` | `768` | Yes | PASS |
| 30 | Embedding | `embedding_batch_size` | `64` | Yes | PASS |
| 31 | Vector Store | `default_collection` | `"helpdesk"` | Yes | PASS |
| 32 | Template Storage | `form_template_storage_path` | `"./form_templates"` | Yes | PASS |
| 33 | Resource Limits | `max_file_size_mb` | `100` | Yes | PASS |
| 34 | Resource Limits | `per_document_timeout_seconds` | `120` | Yes | PASS |
| 35 | Backend Resilience | `backend_timeout_seconds` | `30.0` | Yes | PASS |
| 36 | Backend Resilience | `backend_max_retries` | `2` | Yes | PASS |
| 37 | Backend Resilience | `backend_backoff_base` | `1.0` | Yes | PASS |
| 38 | Dual-Write | `dual_write_mode` | `"best_effort"` | Yes | PASS |
| 39 | Logging/PII | `log_sample_data` | `False` | Yes | PASS |
| 40 | Logging/PII | `log_ocr_output` | `False` | Yes | PASS |
| 41 | Logging/PII | `log_extraction_details` | `False` | Yes | PASS |
| 42 | Logging/PII | `redact_patterns` | `[]` | Yes | PASS |
| 43 | Logging/PII | `redact_target` | `"both"` | Yes | PASS |

**Result**: 43/43 fields covered. PASS.

---

## Check 2: model_validator Cross-Field Checks

Spec validator (lines 1726-1748) contains 6 checks:

| # | Validation | Spec Line | In Plan? | Status |
|---|-----------|-----------|----------|--------|
| 1 | `dual_write_mode` in `{"best_effort", "strict_atomic"}` | 1728-1730 | Yes (Step 3) | PASS |
| 2 | `redact_target` in `{"both", "chunks_only", "db_only"}` | 1731-1733 | Yes (Step 3) | PASS |
| 3 | `page_match_strategy` in `{"windowed"}` | 1734-1736 | Yes (Step 3) | PASS |
| 4 | `form_ocr_engine` in `{"paddleocr", "tesseract"}` | 1737-1739 | Yes (Step 3) | PASS |
| 5 | `pdf_widget_backend` in `{"pymupdf", "pdfplumber"}` | 1740-1742 | Yes (Step 3) | PASS |
| 6 | `form_vlm_fallback_threshold < form_extraction_min_field_confidence` | 1743-1747 | Yes (Step 3) | PASS |

**Result**: 6/6 validator checks covered. PASS.

---

## Check 3: from_file() Pattern

| Aspect | Spec (lines 1750-1789) | Plan (Step 4) | PDF Pattern | Status |
|--------|------------------------|---------------|-------------|--------|
| Detect `.yaml`/`.yml`/`.json` by extension | Yes | Yes | Yes | PASS |
| `FileNotFoundError` for missing file | Yes | Yes | Yes | PASS |
| `ValueError` for unsupported extension | Yes | Yes | Yes | PASS |
| `ImportError` for missing pyyaml | Yes | Yes | Yes | PASS |
| `None` data handled (empty YAML) | Yes | Yes | Yes | PASS |
| Return `cls(**data)` | Yes | Yes | Yes | PASS |

**Minor note**: Spec imports `json as json_mod` inside the method body (lazy import). The PDF config imports `json` at module top-level. Plan references the PDF pattern. Recommend PATCH follow the spec version (lazy import) for consistency with the spec code block, though either works.

**Result**: PASS (with minor note).

---

## Check 4: RedactTarget Enum -- ENUM_VALUE Check

| Aspect | Spec (lines 1984-1988) | Plan (Step 1) | Status |
|--------|------------------------|---------------|--------|
| `BOTH = "both"` | Yes | Yes | PASS |
| `CHUNKS_ONLY = "chunks_only"` | Yes | Yes | PASS |
| `DB_ONLY = "db_only"` | Yes | Yes | PASS |
| Inherits `(str, Enum)` | Yes | Yes | PASS |
| Config default uses lowercase `"both"` (not `"BOTH"`) | Spec line 1722: `default="both"` | Plan line 47: verified | PASS |

**ENUM_VALUE risk**: The plan explicitly identifies this risk (lines 47, 166-172) and proposes a test to verify `default redact_target == "both"` (lowercase). This is the correct mitigation.

**Plan decision**: Keep `redact_target` typed as `str` (matching spec code block line 1721) rather than `RedactTarget`. Define `RedactTarget` as a standalone enum for caller convenience. This is consistent with the spec's config code, where the validator checks the string value against the allowed set. The spec's PII section (line 1981) references `RedactTarget` in a narrative context but the implementation code block uses `str`. Plan correctly follows the code block.

**Result**: PASS.

---

## Check 5: Default Values Spot-Check

| Field | Spec Default | Plan Matches? | Status |
|-------|-------------|---------------|--------|
| `parser_version` | `"ingestkit_forms:1.0.0"` | Yes | PASS |
| `form_match_confidence_threshold` | `0.8` | Yes | PASS |
| `form_match_per_page_minimum` | `0.6` | Yes | PASS |
| `form_match_extra_page_penalty` | `0.02` | Yes | PASS |
| `fingerprint_dpi` | `150` | Yes | PASS |
| `form_ocr_dpi` | `300` | Yes | PASS |
| `form_ocr_engine` | `"paddleocr"` | Yes | PASS |
| `form_vlm_enabled` | `False` | Yes | PASS |
| `form_vlm_model` | `"qwen2.5-vl:7b"` | Yes | PASS |
| `form_vlm_fallback_threshold` | `0.4` | Yes | PASS |
| `form_extraction_min_field_confidence` | `0.5` | Yes | PASS |
| `checkbox_fill_threshold` | `0.3` | Yes | PASS |
| `signature_ink_threshold` | `0.05` | Yes | PASS |
| `embedding_dimension` | `768` | Yes | PASS |
| `redact_target` | `"both"` | Yes | PASS |
| `dual_write_mode` | `"best_effort"` | Yes | PASS |

**Result**: 16/16 spot-checked defaults correct. PASS.

---

## Check 6: Acceptance Criteria Completeness

| # | Plan AC (lines 158-166) | Covers Requirement? | Status |
|---|-------------------------|---------------------|--------|
| 1 | FormProcessorConfig is BaseModel with 43+ fields matching spec | Yes - field count | PASS |
| 2 | All Field() constraints (ge, le, description) match spec | Yes - constraint accuracy | PASS |
| 3 | Default values match spec table | Yes - default correctness | PASS |
| 4 | model_validator validates 5 enum fields + 1 threshold | Yes - validator completeness | PASS |
| 5 | from_file() loads YAML and JSON | Yes - file loading | PASS |
| 6 | RedactTarget(str, Enum) with correct members | Yes - enum definition | PASS |
| 7 | Unit tests cover validators, defaults, from_file, Field constraints | Yes - test coverage | PASS |
| 8 | No regressions in existing tests | Yes - regression safety | PASS |
| 9 | redact_target uses "both" not "BOTH" | Yes - ENUM_VALUE check | PASS |

**Missing AC**: None identified. The 9 acceptance criteria cover all implementation requirements.

**Test plan review** (20 tests in plan Step 6): Comprehensive. Covers default construction, field count, default values, all 5 enum validators, threshold cross-field check, from_file (JSON + YAML + error cases), Field constraints, RedactTarget enum members, serialization round-trip, and constructor overrides.

**Result**: PASS.

---

## Issues Found

### Minor (non-blocking)

1. **`from_file()` import style**: Spec uses `import json as json_mod` inside the method. Plan references the PDF pattern which uses top-level `import json`. PATCH should follow the spec's local import style. Severity: cosmetic.

2. **Field group count**: Plan says "14 categories" but spec has 16 comment-delimited sections. Plan merges "Native PDF Field Matching" into "Field Extraction" and counts "Logging/PII" as one group. No impact on implementation.

3. **`form_vlm_fallback_threshold` default validation edge case**: With defaults `form_vlm_fallback_threshold=0.4` and `form_extraction_min_field_confidence=0.5`, the constraint `0.4 < 0.5` passes. However, if a user sets both to the same value (e.g., 0.5), the `>=` check correctly rejects. The test plan (test #9) covers this. No issue.

---

## Summary

**Plan status**: **APPROVED**

| Check | Result |
|-------|--------|
| 1. Config parameter coverage (43 fields) | PASS (43/43) |
| 2. model_validator cross-field checks | PASS (6/6) |
| 3. from_file() pattern | PASS (minor: import style) |
| 4. RedactTarget ENUM_VALUE check | PASS (lowercase values confirmed) |
| 5. Default values | PASS (16/16 spot-checked) |
| 6. Acceptance criteria completeness | PASS (9 AC, 20 tests) |

**Risk level**: LOW
- Near-direct transcription from spec code block
- Follows established patterns from ingestkit-pdf and ingestkit-excel
- ENUM_VALUE risk explicitly identified and mitigated in plan

**Ready for PATCH phase**: YES

---

AGENT_RETURN: .agents/outputs/plan-check-59-021526.md
