---
issue: 35
agent: PLAN-CHECK
date: 2026-02-14
status: PASS
---

# PLAN-CHECK: Issue #35 -- Page Renderer and OCR Preprocessing

## Executive Summary

The MAP-PLAN for issue #35 is well-structured and aligns with the GitHub issue requirements and SPEC 11.2 steps 1-2. File count (1 created, 1 modified, 1 test file) is appropriate for SIMPLE complexity. All 11 acceptance criteria map to planned tasks. Dependencies (pymupdf, Pillow, opencv-python-headless) are correctly identified and already declared in pyproject.toml. OpenCV optional import guard is planned. No scope creep detected.

## Validation Checklist

### Spec Alignment

- [x] Step 1 (render page to high-DPI image via `page.get_pixmap(dpi=...)`) matches SPEC 11.2 step 1
- [x] Step 2 (configurable preprocessing pipeline) matches SPEC 11.2 step 2
- [x] All four preprocessing algorithms match spec: deskew (Hough), denoise (fastNlMeansDenoisingColored), binarize (Otsu), contrast (CLAHE)
- [x] Default DPI 300 and default steps `["deskew"]` match config.py lines 63, 66

### Config Parameter Verification

- [x] `ocr_dpi: int = 300` exists at config.py line 63 -- plan references correctly
- [x] `ocr_preprocessing_steps: list[str] = ["deskew"]` exists at config.py line 66 -- plan references correctly
- [x] No new config fields needed -- all parameters already exist

### Model Field Verification

- [x] `OCRResult.preprocessing_steps: list[str]` exists at models.py line 214 -- confirms renderer should track applied steps
- [x] `PDFChunkMetadata.ocr_preprocessing: list[str] | None` exists at models.py line 247 -- downstream metadata field confirmed

### Dependency Verification

- [x] `pymupdf>=1.24` in pyproject.toml line 12 (core dependency) -- provides `fitz`
- [x] `Pillow>=10.0` in pyproject.toml line 17 (core dependency) -- provides `PIL.Image`
- [x] `opencv-python-headless>=4.8` in pyproject.toml line 29 under `[opencv]` extra -- optional, plan correctly treats as optional
- [x] No new dependencies needed

### Error Code Verification

- [x] `E_OCR_FAILED` exists at errors.py line 35 -- appropriate for rendering/preprocessing failures
- [x] Plan correctly uses `E_OCR_FAILED` with descriptive message (no new error code needed)

### File Count & Scope

- [x] Files to create: 1 (`utils/page_renderer.py`) + 1 test (`tests/test_page_renderer.py`) -- appropriate for SIMPLE
- [x] Files to modify: 1 (`utils/__init__.py` -- add PageRenderer re-export) -- currently has 2 re-exports, adding 1 more
- [x] No existing `page_renderer.py` (verified via glob search)
- [x] No scope creep -- plan stays within SPEC 11.2 steps 1-2, does not touch OCR engine integration (step 3+) or other modules

### Pattern Compliance

- [x] Follows `header_footer.py` pattern: module docstring, `from __future__ import annotations`, `TYPE_CHECKING` guard for fitz
- [x] Logger name `ingestkit_pdf.utils.page_renderer` follows convention (cf. `ingestkit_pdf.utils.header_footer`)
- [x] Config injected via constructor (`PDFProcessorConfig`)
- [x] No ABCs -- structural subtyping preserved
- [x] No concrete backend imports inside package

### OpenCV Import Guard

- [x] Plan explicitly states: "OpenCV import inside preprocessing methods or guarded at module level with a clear error message if `cv2` is not installed"
- [x] Test planned: `test_opencv_not_installed` -- verifies clear error when cv2 unavailable and steps are configured
- [x] Consistent with `opencv-python-headless` being an optional extra in pyproject.toml

## Acceptance Criteria Coverage

| # | Criterion | Mapped to Plan Section |
|---|-----------|----------------------|
| 1 | Renders fitz.Page to PIL Image at configured DPI | `render_page()` method |
| 2 | Preprocess applies steps in order from config | `preprocess()` method with dispatch dict |
| 3 | All four preprocessing steps implemented | Private methods: `_deskew`, `_denoise`, `_binarize`, `_contrast` |
| 4 | Unknown steps skipped with warning | `__init__` validation + skip logic |
| 5 | OpenCV guarded with clear error | Import guard (section 6 of plan) |
| 6 | Large page dimensions handled gracefully | `render_page()` threshold check + warning |
| 7 | Re-exported from utils/__init__.py | Explicit in plan (File 2) |
| 8 | All unit tests pass | 12 test cases in test plan |
| 9 | Tests use only synthetic images and mock fitz | Test helper functions + mocks |
| 10 | Logger uses correct name | Plan specifies `ingestkit_pdf.utils.page_renderer` |
| 11 | Module follows code conventions | Plan specifies `__future__`, `TYPE_CHECKING`, Pydantic config |

### Issue Acceptance Criteria Coverage

| Issue Criterion | Covered |
|----------------|---------|
| PDF page rendered to PIL Image at configured DPI | AC #1 |
| Deskew correction works on skewed images | AC #3 + test `test_preprocess_deskew_only` |
| Denoise reduces noise in scanned images | AC #3 + test `test_preprocess_denoise` |
| Binarize produces clean black/white images | AC #3 + test `test_preprocess_binarize` |
| CLAHE enhancement improves contrast | AC #3 + test `test_preprocess_contrast` |
| Preprocessing steps configurable and applied in order | AC #2 + test `test_preprocess_multiple_steps` |
| `pytest tests/test_page_renderer.py -q` passes | AC #8 |
| `ruff check .` passes | Not explicit in plan but standard PROVE gate |

### Test Plan Coverage

- [x] Rendering tests: 3 (default DPI, custom DPI, large dimensions)
- [x] Preprocessing pipeline: 7 (no steps, each of 4 steps individually, multiple steps, unknown step)
- [x] Integration: 1 (render + preprocess end-to-end)
- [x] Edge cases: 2 (grayscale input, opencv not installed)
- [x] Total: 13 tests covering both public methods and all 4 preprocessing algorithms
- [x] Test helpers planned: `_make_synthetic_image`, `_make_mock_pixmap`

## Issues Found

**Minor (non-blocking):**

1. **`ruff check .` not mentioned in plan**: The issue's acceptance criteria include `ruff check .` passing. This is a standard PROVE gate and will be checked there, so this is not a blocker -- but the PATCH agent should ensure new code is ruff-compliant.

## Recommendation

**APPROVED** -- The MAP-PLAN is accurate, complete, and correctly scoped. All acceptance criteria are covered, dependencies are verified in pyproject.toml, the OpenCV import guard is planned, and test coverage is sufficient. Proceed to PATCH.
