---
issue: 28
agent: PLAN-CHECK
date: 2026-02-14
status: PASS
---

# PLAN-CHECK: Issue #28 -- Tier 1 Rule-Based PDF Inspector

## Validation Checklist

### Spec Compliance (SPEC.md §9.1--§9.4)

- ✅ **Purpose (§9.1)**: PLAN correctly describes rule-based, no-LLM, deterministic classifier handling ~80% of files. Zero external dependencies.
- ✅ **Signal 1 -- Text chars/page**: PLAN uses `page.text_length >= config.min_chars_per_page` (200) for text-native, `< _MAX_CHARS_FOR_SCANNED` (50) for scanned. Matches §9.2.
- ✅ **Signal 2 -- Image coverage ratio**: PLAN uses `page.image_coverage_ratio < config.max_image_coverage_for_text` (0.3) for text-native, `> _MIN_IMAGE_COVERAGE_FOR_SCANNED` (0.7) for scanned. Matches §9.2.
- ✅ **Signal 3 -- Font count**: PLAN uses `page.font_count >= config.min_font_count_for_digital` (1) for text-native, `== 0` for scanned. Matches §9.2.
- ✅ **Signal 4 -- Table count**: PLAN uses `page.table_count >= config.min_table_count_for_complex` (1). Matches §9.2.
- ✅ **Signal 5 -- Multi-column layout**: PLAN uses `page.is_multi_column`. Matches §9.2.
- ✅ **Signal 6 -- Form fields**: PLAN uses `page.has_form_fields`. Matches §9.2.
- ✅ **Signal 7 -- Page consistency**: PLAN applies at document level -- pages agree = that type, disagree = COMPLEX. Matches §9.2/§9.3.
- ✅ **Decision logic (§9.3)**: Complex check first (signals 4-6), then text-native (signals 1-3), then scanned (signals 1-3 inverted), then borderline. Matches spec.
- ✅ **Document aggregation (§9.3)**: All pages agree = that type; disagreement = COMPLEX. Matches spec.
- ✅ **Public interface (§9.4)**: `PDFInspector.__init__(config)` and `classify(profile) -> ClassificationResult`. Matches spec exactly.

### Model Field Names (models.py)

- ✅ `PageProfile.text_length` (int) -- PLAN references correctly (note: issue description says `text_char_count` but actual model uses `text_length`; PLAN uses the correct name)
- ✅ `PageProfile.image_coverage_ratio` (float) -- correct
- ✅ `PageProfile.font_count` (int) -- correct
- ✅ `PageProfile.table_count` (int) -- correct
- ✅ `PageProfile.is_multi_column` (bool) -- correct
- ✅ `PageProfile.has_form_fields` (bool) -- correct
- ✅ `PageProfile.page_type` (PageType) -- correct
- ✅ `DocumentProfile.pages` (list[PageProfile]) -- correct
- ✅ `DocumentProfile.page_count` (int) -- correct
- ✅ `ClassificationResult` fields: `pdf_type`, `confidence`, `tier_used`, `reasoning`, `per_page_types`, `signals`, `degraded` -- all match
- ✅ `PDFType` enum values: `TEXT_NATIVE`, `SCANNED`, `COMPLEX` -- correct
- ✅ `PageType` enum values: `TEXT`, `SCANNED`, `TABLE_HEAVY`, `FORM`, `MIXED`, `BLANK`, `VECTOR_ONLY`, `TOC` -- all present in model
- ✅ `ClassificationTier` re-exported from `ingestkit_core.models` -- correct

### Config Param Names (config.py)

- ✅ `min_chars_per_page` = 200 -- exists, default matches
- ✅ `max_image_coverage_for_text` = 0.3 -- exists, default matches
- ✅ `min_font_count_for_digital` = 1 -- exists, default matches
- ✅ `min_table_count_for_complex` = 1 -- exists, default matches
- ✅ `tier1_high_confidence_signals` = 4 -- exists (used by router, not inspector internally; PLAN correctly omits from inspector logic)
- ✅ `tier1_medium_confidence_signals` = 3 -- exists (same as above)

### Error Codes (errors.py)

- ✅ `E_CLASSIFY_INCONCLUSIVE` -- exists
- ✅ `W_PAGE_SKIPPED_BLANK` -- exists
- ✅ `W_PAGE_SKIPPED_TOC` -- exists
- ✅ `W_PAGE_SKIPPED_VECTOR_ONLY` -- exists
- ✅ `W_CLASSIFICATION_DEGRADED` -- exists (PLAN correctly notes it is not used by Tier 1)

### Confidence Calculation

- ✅ Per-page: 0.9 (all indicators match), 0.7 (2 of 3 match), 0.0 (borderline) -- sound approach
- ✅ Per-document: minimum of non-skippable page confidences -- consistent with Excel pattern
- ✅ Disagreement confidence: 0.9 (high confidence that the document IS complex) -- reasonable
- ✅ Confidence values work with router threshold: `tier1_high_confidence_signals / 5` = 0.8, so 0.9 passes, 0.7 escalates to Tier 2 -- coherent

### Test Coverage

- ✅ `TestPerPageSignalEvaluation` -- covers all 6 per-page signals with true/false cases (16 tests)
- ✅ `TestPerPageClassification` -- text-native, scanned, complex, borderline, priority (10 tests)
- ✅ `TestSpecialPages` -- blank, TOC, vector-only handling (6 tests)
- ✅ `TestDocumentLevelAgreement` -- all-same-type cases + min confidence (5 tests)
- ✅ `TestDocumentLevelDisagreement` -- mixed pages produce COMPLEX (6 tests)
- ✅ `TestInconclusiveEscalation` -- borderline triggers Tier 2 (3 tests)
- ✅ `TestEmptyProfile` -- no pages edge case (3 tests)
- ✅ `TestClassificationResultFields` -- output field validation (6 tests)
- ✅ `TestCustomConfig` -- custom thresholds respected (4 tests)
- ✅ `TestBoundaryValues` -- exact threshold boundaries (8 tests)
- ✅ `TestConfidenceCalculation` -- confidence model validation (8 tests)

### Acceptance Criteria Coverage

- ✅ PDFInspector class with `__init__` and `classify` -- covered in implementation plan
- ✅ 7 per-page signals -- all 7 covered
- ✅ Three-way classification -- text-native, scanned, complex all covered
- ✅ Borderline escalation -- confidence 0.0 triggers Tier 2
- ✅ Document-level aggregation -- agreement/disagreement logic covered
- ✅ Special page handling -- blank/TOC/vector-only excluded from aggregation
- ✅ ClassificationResult fully populated -- all fields planned
- ✅ tier_used always RULE_BASED -- stated in plan
- ✅ degraded always False -- stated in plan
- ✅ PII-safe logging -- logger name `ingestkit_pdf`, no raw content
- ✅ Config thresholds used -- all 4 configurable thresholds sourced from config
- ✅ Unit tests -- 11 test classes, ~75 test methods planned
- ✅ No concrete backend imports -- only imports from ingestkit_pdf and ingestkit_core
- ✅ No ABC usage -- structural subtyping only

### Scope Containment

- ✅ No modifications to `models.py`, `config.py`, or `errors.py` -- correct, all needed types exist
- ✅ No ROADMAP items implemented
- ✅ No concrete backends introduced
- ✅ Two files created (`inspector.py`, `test_inspector.py`) + minor `conftest.py` modification -- appropriate for scope

## Issues Found

### Minor (non-blocking)

1. **Dead code in step 6 of `_classify_page`**: Steps 4 and 5 already return when `text_count >= 2` or `scan_count >= 2`. Step 6's conditions `text_count >= 2` and `scan_count >= 2` are unreachable. The fallback will always hit the borderline case. This is harmless but the implementor could simplify step 6 to just return borderline directly. *Non-blocking -- does not affect correctness.*

2. **`tier1_high_confidence_signals` and `tier1_medium_confidence_signals` unused by inspector**: These config values exist but the PLAN's confidence model (0.9/0.7/0.0) doesn't reference them. This is actually correct -- the SPEC shows these are used by the *router* (`tier1_result.confidence >= tier1_high_confidence_signals / 5`), not the inspector itself. The PLAN's confidence values (0.9, 0.7, 0.0) align well with the router's threshold of 0.8. *Non-blocking -- coherent design.*

3. **Test helper `_make_page_profile` includes `ExtractionQuality` construction**: The helper constructs `ExtractionQuality` inline. This is correct but verbose; the implementor may want to extract an `_extraction_quality()` default factory. *Non-blocking -- style preference.*

## Recommendation

**APPROVED**

The PLAN is thorough, accurate, and well-aligned with the SPEC, models, config, and error codes. All 7 signals are covered. Decision logic matches the spec. Model field names and config param names are verified correct against actual source files. Test coverage is comprehensive with 11 test classes covering all public methods and edge cases. No blocking issues found.

AGENT_RETURN: .agents/outputs/plan-check-28-021426.md
