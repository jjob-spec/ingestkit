---
issue: 31
agent: PLAN-CHECK
date: 2026-02-14
status: PASS
---

# PLAN-CHECK: Issue #31 — Heading Hierarchy Detection

## Executive Summary

The MAP-PLAN for issue #31 is well-structured and accurately reflects the SPEC (sections 14.1-14.4), the existing config parameters, and model fields. File count (2 created, 1 modified) is appropriate for SIMPLE complexity. All acceptance criteria map to planned tasks. No scope creep detected.

## Validation Checklist

### Spec Alignment

- [x] Strategy 1 (PDF outline via `doc.get_toc()`) matches SPEC 14.1
- [x] Strategy 2 (font-based inference) matches SPEC 14.2 -- body size = most common, threshold = `body_size * ratio`, requires bold, top 3 sizes to H1/H2/H3
- [x] Strategy 3 (pymupdf4llm markdown parsing) matches SPEC 14.3
- [x] Public interface (`detect()`, `get_heading_path()`) matches SPEC 14.4 signatures exactly
- [x] Strategy cascade order (outline > fonts > markdown) is consistent with spec's "authoritative when present" language

### Config Parameter Verification

- [x] `heading_min_font_size_ratio: float = 1.2` exists at config.py line 83 -- plan references line 83, correct
- [x] Parameter name used in plan (`self._config.heading_min_font_size_ratio`) matches actual field name

### Model Field Verification

- [x] `DocumentProfile.has_toc: bool` exists at models.py line 183
- [x] `DocumentProfile.toc_entries: list[tuple[int, str, int]] | None` exists at models.py line 184 -- type matches `detect()` return type
- [x] `PDFChunkMetadata.heading_path: list[str] | None` exists at models.py line 239 -- type matches `get_heading_path()` return type

### File Count & Scope

- [x] Files to create: 2 (heading_detector.py, test_heading_detector.py) -- appropriate for SIMPLE
- [x] Files to modify: 1 (utils/__init__.py) -- currently empty, confirmed
- [x] No existing heading_detector.py or test_heading_detector.py (verified via glob)
- [x] No scope creep -- plan stays within section 14 of the spec, does not touch chunking (section 15) or other modules

### Acceptance Criteria Coverage

| Criterion | Mapped to Plan Section |
|-----------|----------------------|
| HeadingDetector class created | File 1: class definition |
| Constructor accepts PDFProcessorConfig | File 1: `__init__` |
| `detect()` returns correct type | File 1: `detect()` method |
| Strategy 1 implemented | File 1: `_detect_from_outline()` |
| Strategy 2 implemented | File 1: `_detect_from_fonts()` |
| Strategy 3 implemented | File 1: `_detect_from_markdown()` |
| `get_heading_path()` implemented | File 1: `get_heading_path()` |
| No concrete backend imports | Plan uses only PyMuPDF/pymupdf4llm |
| Logger uses `ingestkit_pdf` | Plan specifies this |
| Failures return empty lists | Plan specifies try/except + warning logging |
| `utils/__init__.py` re-exports | File 2: explicit re-export |
| Tests pass with mocked fitz | File 3: all test classes use mocks |
| Tests marked `@pytest.mark.unit` | Stated in test plan |
| No binary PDF fixtures | Stated in test plan |
| Config ratio respected | Test: `test_heading_min_font_size_ratio_configurable` |

### Test Plan Coverage

- [x] All 3 strategies tested individually (TestDetectFromOutline: 3 tests, TestDetectFromFonts: 8 tests, TestDetectFromMarkdown: 3 tests)
- [x] `get_heading_path()` tested (TestGetHeadingPath: 5 tests)
- [x] Strategy cascade tested (TestStrategyCascade: 4 tests)
- [x] Edge cases covered (TestEdgeCases: 4 tests)
- [x] Total: 27 tests covering both public methods and all 3 strategies

### Pattern Compliance

- [x] Uses `Protocol`-based architecture (no ABCs)
- [x] Logger name follows `ingestkit_pdf` convention
- [x] Pydantic config accepted (not dataclass)
- [x] No concrete backend imports inside package

## Issues Found

None.

## Recommendation

**APPROVED** -- The MAP-PLAN is accurate, complete, and correctly scoped. Proceed to PATCH.
