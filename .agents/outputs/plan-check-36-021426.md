---
issue: 36
agent: PLAN-CHECK
date: 2026-02-14
status: PASS (with 1 advisory)
---

# PLAN-CHECK: Issue #36 — OCR Postprocessing

## Executive Summary

The MAP-PLAN for issue #36 is well-structured, accurately reflects the SPEC 11.2 step 6 requirements, and correctly scopes the implementation. All four postprocessing operations are covered. File count (3 files: 1 new, 1 modified, 1 test) is appropriate for SIMPLE complexity. Test coverage is thorough with 21 cases. One advisory issue found: the proposed regex for hyphen merging contradicts test case T5.

---

## Validation Checklist

### Requirement Coverage (Issue Acceptance Criteria)

- [x] Hyphenated line breaks merged correctly -- T1-T5, `_merge_hyphenated_breaks`
- [x] Multiple spaces collapsed to single space -- T8-T9, `_normalize_whitespace`
- [x] Unicode normalized (NFC) -- T6-T7, `_normalize_unicode`
- [x] Random isolated characters stripped -- T13-T14, `_strip_ocr_artifacts`
- [x] Repeated punctuation cleaned up -- T15-T17, `_strip_ocr_artifacts`
- [x] Clean text preserved unchanged -- T20
- [x] Tests planned -- 21 test cases in `test_ocr_postprocess.py`
- [x] No mention of ruff-incompatible patterns

### Spec Accuracy (SPEC 11.2 step 6)

- [x] All four operations from spec are present: hyphen merge, whitespace normalize, Unicode normalize, artifact strip
- [x] Module path matches spec file tree: `utils/ocr_postprocess.py`
- [x] Integration point correctly identified: after OCR engine output, before LLM cleanup (step 7) and chunking (step 9)
- [x] Config analysis correct: no new config fields needed; module is stateless

### Scope and Complexity

- [x] File count (3: 1 source + 1 `__init__.py` edit + 1 test) matches SIMPLE complexity
- [x] No scope creep: no changes to processors, router, or backends
- [x] No ROADMAP.md items implemented
- [x] No external dependencies added (stdlib only: `re`, `unicodedata`)
- [x] No concrete backend implementations inside the package

### Test Coverage (21 cases)

- [x] Hyphen merging: 5 cases (T1-T5) including preservation of same-line hyphens
- [x] Unicode: 2 cases (T6-T7) covering NFD-to-NFC and ASCII passthrough
- [x] Whitespace: 5 cases (T8-T12) covering spaces, tabs, line endings, blank lines, trailing whitespace
- [x] Artifact stripping: 5 cases (T13-T17) covering isolated chars, whitelist, repeated punctuation, ellipsis, periods
- [x] Integration: 4 cases (T18-T21) covering combined pipeline, empty input, clean passthrough, realistic sample
- [x] Edge cases: whitelist for "I", "a", "A" explicitly tested in T14
- [x] No external service dependencies in tests

### Architecture Compliance

- [x] Function-based design (no ABC or class hierarchy)
- [x] Logger namespace `ingestkit_pdf` -- plan mentions it
- [x] Module docstring references SPEC 11.2 step 6
- [x] Pure utility, no backend imports
- [x] `__init__.py` export update planned

---

## Issues Found

### Advisory: Regex vs. T5 Contradiction (Non-Blocking)

The plan specifies the regex `r'(\w)-\n(\w)'` for hyphen merging, but `\w` matches digits. Test case T5 expects `"123-\n456"` to remain unchanged (digits should not merge). The proposed regex would merge it into `"123456"`.

**Resolution for PATCH**: The implementer should use `r'([a-zA-Z])-\n([a-zA-Z])'` (or `r'([a-z])-\n([a-z])'` if only lowercase continuation is desired) instead of `\w` to match the intended behavior in T5. This is a minor implementation detail, not a plan-level blocker.

---

## Recommendation

**APPROVED** -- The MAP-PLAN is accurate, complete, and ready for PATCH. The implementer should note the regex advisory above when coding `_merge_hyphenated_breaks`.

AGENT_RETURN: .agents/outputs/plan-check-36-021426.md
