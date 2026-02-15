---
issue: 37
agent: PLAN-CHECK
date: 2026-02-14
status: PASS
---

# PLAN-CHECK: Issue #37 -- Layout Analysis for Multi-Column Detection

## Executive Summary

The MAP-PLAN for issue #37 is thorough and well-aligned with SPEC sections 2.3, 9.2, and 11.3 step 4. File count (1 created, 1 modified, 1 test file) is appropriate for SIMPLE complexity. All 13 acceptance criteria map to planned tasks. The gap-based clustering approach is sound and avoids external dependencies. The 18 test cases provide strong coverage across single/multi-column layouts, mixed layouts, edge cases, and error handling. One minor issue noted regarding the `reorder_blocks` page width reference, but it is non-blocking.

## Validation Checklist

### Spec Alignment

- [x] SPEC 2.3 lists `utils/layout_analysis.py` -- Multi-column detection, reading order (line 164)
- [x] SPEC 9.2 signal 5: `PageProfile.is_multi_column: bool` -- "detected via text block x-coordinate clustering" (line 330)
- [x] SPEC 11.3 step 4: Detect multi-column layouts via text block x-coordinate clustering, reorder into correct reading order (lines 1155-1157)
- [x] `PageProfile.is_multi_column` field already exists at models.py line 141 -- layout analyzer provides the detection logic

### fitz API Verification

- [x] `page.get_text("blocks")` returns `(x0, y0, x1, y1, text, block_no, block_type)` tuples -- confirmed from header_footer.py lines 75-76
- [x] `block_type == 0` for text, `block_type == 1` for image -- correctly used in plan's filter step
- [x] `page.rect.width` for page dimensions -- standard PyMuPDF API
- [x] Issue mentions both `get_text("dict")` and `get_text("blocks")` as options; plan correctly chose `get_text("blocks")` which is simpler and sufficient for x-coordinate clustering (consistent with header_footer.py pattern)

### File Count & Scope

- [x] Files to create: 1 (`utils/layout_analysis.py`) + 1 test (`tests/test_layout_analysis.py`) -- appropriate for SIMPLE
- [x] Files to modify: 1 (`utils/__init__.py` -- add LayoutAnalyzer, LayoutResult, TextBlock re-exports)
- [x] No existing `layout_analysis.py` (verified via glob search)
- [x] No scope creep -- plan stays within SPEC 11.3 step 4, does not touch Complex Processor integration, inspector wiring, or config changes
- [x] Correctly defers integration (callers populating `PageProfile.is_multi_column` and Complex Processor calling `reorder_blocks`) to downstream issues

### Pattern Compliance

- [x] Follows `header_footer.py` pattern: module docstring, `from __future__ import annotations`, `TYPE_CHECKING` guard for fitz
- [x] Logger name `ingestkit_pdf.utils.layout_analysis` follows convention (cf. `ingestkit_pdf.utils.header_footer`)
- [x] Config injected via constructor (`PDFProcessorConfig`) -- reserved for future thresholds
- [x] Pydantic v2 models for `TextBlock` and `LayoutResult` -- consistent with project conventions
- [x] No ABCs -- structural subtyping preserved
- [x] No concrete backend imports inside package
- [x] Fail-safe: all methods wrapped in try/except, return single-column default on error

### Gap-Based Clustering Algorithm Review

- [x] **Approach is sound**: Sort x0 values, find gaps > 10% page width, split into clusters. This is a well-known 1D gap-based clustering technique.
- [x] **No external dependencies**: Uses only stdlib sorting and comparison -- avoids sklearn/numpy requirement.
- [x] **Conservative threshold**: 10% page width gap is reasonable. Typical paragraph indentation is 2-5% of page width, so false positives from indentation are unlikely.
- [x] **Minimum block threshold**: Requires 3+ text blocks before attempting clustering -- prevents false positives on sparse pages.
- [x] **Cluster validation**: Each cluster must have 2+ blocks -- prevents single-block noise from creating phantom columns.
- [x] **Column cap**: Capped at 3 columns -- prevents over-segmentation from noisy documents.
- [x] **Full-width block handling**: Blocks spanning >75% of page width excluded from clustering and placed first in reading order -- correctly handles common header/title patterns.

### Minor Issue: `reorder_blocks` Page Width Reference

- The `reorder_blocks` method uses "75% of page width based on column boundaries" to identify full-width blocks, but it only receives `blocks` and `layout` (no `page` reference). The column boundaries in `LayoutResult` should provide enough info to infer page width (or the 75% threshold could be relative to total column span). This is an implementation detail the PATCH agent can resolve -- **non-blocking**.

## Acceptance Criteria Coverage

| # | MAP-PLAN Criterion | Mapped to Plan Section |
|---|-------------------|----------------------|
| 1 | detect_columns identifies single, two, three-column | `detect_columns()` method + clustering algorithm |
| 2 | Uses text block x-coordinate clustering per SPEC | Clustering algorithm detail (section 5) |
| 3 | reorder_blocks produces correct reading order | `reorder_blocks()` method |
| 4 | Mixed layouts handled | Full-width block filtering (>75% width) |
| 5 | < 3 text blocks returns single-column | Early return in `detect_columns()` |
| 6 | Image/empty blocks filtered | `extract_text_blocks()` helper |
| 7 | Fail-safe: exceptions caught, single-column default | Fail-safe section (7) |
| 8 | Re-exported from utils/__init__.py | File 2 plan |
| 9 | All unit tests pass | 18 test cases in test plan |
| 10 | Tests use only mock fitz.Page | Test helpers: `_make_mock_page` |
| 11 | Logger uses correct name | Plan specifies `ingestkit_pdf.utils.layout_analysis` |
| 12 | Module follows code conventions | Plan specifies `__future__`, `TYPE_CHECKING`, Pydantic |
| 13 | No external clustering dependencies | Stdlib-only gap-based algorithm |

### Issue Acceptance Criteria Coverage

| Issue Criterion | Covered |
|----------------|---------|
| Single-column layout detected correctly | AC #1 + tests T1-T3 |
| Two-column layout detected and reordered | AC #1, #3 + tests T4-T5, T9 |
| Mixed layout (full-width header + columns) handled | AC #4 + tests T7-T8, T11 |
| Reading order correct for standard layouts | AC #3 + tests T9-T11 |
| `pytest tests/test_layout_analysis.py -q` passes | AC #9 |
| `ruff check .` passes | Standard PROVE gate |

### Test Plan Coverage (18 tests)

- [x] Single-column detection: 3 tests (T1-T3: aligned blocks, no text blocks, fewer than 3 blocks)
- [x] Two-column detection: 2 tests (T4-T5: classic layout, unequal block counts)
- [x] Three-column detection: 1 test (T6)
- [x] Mixed layout: 2 tests (T7-T8: full-width + columns, all full-width)
- [x] Reading order: 3 tests (T9-T11: two-column, single-column, mixed reorder)
- [x] Edge cases: 5 tests (T12-T16: image blocks, empty text, narrow page, overlapping x-ranges, exception handling)
- [x] Column boundaries: 2 tests (T17-T18: boundary validation for 2 and 3 columns)
- [x] Total: 18 tests -- sufficient for SIMPLE complexity
- [x] Test helpers planned: `_make_block`, `_make_mock_page`

## Issues Found

**Minor (non-blocking):**

1. **`reorder_blocks` page width inference**: The method receives `blocks` and `layout` but not `page`. The 75% full-width threshold needs to be computed from available data (column boundaries or a stored page width in `LayoutResult`). PATCH agent should either add `page_width` to `LayoutResult` or compute the threshold from column boundary span. This is an implementation detail, not a design flaw.

2. **`ruff check .` not in plan acceptance criteria**: The issue requires `ruff check .` passing. This is a standard PROVE gate and will be checked there -- not a blocker.

## Recommendation

**APPROVED** -- The MAP-PLAN is accurate, complete, and correctly scoped. The gap-based clustering algorithm is sound and well-suited for column detection without external dependencies. All issue acceptance criteria are covered. Test coverage (18 cases) is strong across detection, reordering, mixed layouts, and edge cases. The one minor issue (page width in `reorder_blocks`) is an implementation detail the PATCH agent can resolve. Proceed to PATCH.
