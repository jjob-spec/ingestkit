---
issue: 44
agent: PLAN-CHECK
date: 2026-02-14
status: PASS
plan_artifact: plan-44-021426.md
map_artifact: map-44-021426.md
---

# PLAN-CHECK Artifact: Issue #44 -- PaddleOCR Adapter and Engine Fallback Chain

## Verdict: PASS (3 minor findings, 0 blockers)

The PLAN is well-structured, matches SPEC sections 12.4 and 12.5, and is safe to proceed to PATCH.

---

## Check Results

### 1. PaddleOCR Result Parsing -- PASS

The nested structure `[[[bbox, (text, conf)], ...]]` is handled correctly in the PLAN:
- Outer `result` None check: present (line 117 of plan code)
- Inner `page_lines` None check: present (line 119)
- Individual `line` None check: present (line 122)
- Extraction of `line[1]` for `(text, conf)` tuple: correct
- Whitespace-only line filtering via `text_val.strip()`: correct
- Empty list `[[]]` case: handled (loop body simply doesn't execute)

No issues found.

### 2. PIL to numpy Conversion -- PASS

`np.array(image)` on a PIL Image produces an HWC numpy array, which is what PaddleOCR expects. The deferred `import numpy as np` inside `recognize()` is consistent with the lazy-import pattern used throughout the module.

### 3. Factory Fallback Chain vs SPEC 12.5 -- PASS

PLAN replaces the stub (lines 194-199) with:
```python
return PaddleOCREngine(lang=config.ocr_language), warnings
```

This matches SPEC 12.5 exactly (line 1249). The existing `except ImportError` block with `W_OCR_ENGINE_FALLBACK` warning and Tesseract fall-through (lines 200-205) is preserved unchanged. Fallback chain: PaddleOCR -> Tesseract -> `EngineUnavailableError` -- matches SPEC line 1267.

### 4. W_OCR_ENGINE_FALLBACK Warning Emission -- PASS

The existing `except ImportError` block at line 200-205 of `ocr_engines.py` already appends `"W_OCR_ENGINE_FALLBACK"` to `warnings` and logs a warning. The PLAN correctly preserves this block unchanged. When `PaddleOCREngine.__init__()` raises `ImportError` (paddleocr not installed), the fallback fires naturally.

### 5. Existing Tests Not Broken -- PASS

The PLAN:
- Does not modify any existing test class or test method
- Adds new test classes (`TestPaddleOCRLanguageMapping`, `TestPaddleOCREngine`) and new methods to existing classes (`TestOCREngineInterface`, `TestCreateOCREngine`)
- All new tests use mocks; no real PaddleOCR dependency for `@pytest.mark.unit` tests
- The factory change (removing artificial `raise ImportError`) could affect `test_paddleocr_config_falls_back_to_tesseract` (line 293-315), but that test already mocks `paddleocr` as missing via `_fake_import`, so the `ImportError` will now come from `PaddleOCREngine.__init__()` instead of the explicit raise -- same behavior, test still passes.

### 6. Method Name: recognize vs recognise -- PASS

Protocol uses `recognize` (American spelling) at `ocr_engines.py:63`. PLAN uses `recognize` consistently. SPEC 12.4 (line 1226) also uses `recognize`. No spelling mismatch.

### 7. Language Mapping Coverage -- PASS with MINOR FINDING

PLAN defines `_PADDLEOCR_LANG_MAP` with 4 exceptions: `zh->ch`, `ko->korean`, `ja->japan`, `de->german`. All other codes pass through as-is.

**Finding F1 (LOW)**: The existing `language.py:61-74` has a broader map including identity mappings (`en->en`, `fr->fr`, etc.) plus `ar->ar`, `hi->hi`. The PLAN's leaner map (exceptions only) is *more correct* -- identity mappings are redundant since `_to_paddleocr_lang()` falls back to the input. However, `ar` (Arabic) maps to `"ar"` in both PaddleOCR and ISO 639-1, so it's correctly omitted as an identity. No functional gap.

---

## Minor Findings

| ID | Severity | Description | Recommendation |
|----|----------|-------------|----------------|
| F1 | LOW | PLAN's `_PADDLEOCR_LANG_MAP` is leaner than `language.py`'s version (omits identity mappings). | Correct design. No action needed. |
| F2 | LOW | `test_init_raises_import_error_when_not_installed` (plan 3c) uses `__builtins__.__import__` pattern which is fragile across Python implementations. | Acceptable -- same pattern already used in existing tests at lines 158, 301. Consistency is more important. |
| F3 | LOW | Language re-creation in `recognize()` (lines 96-106 of plan) creates a temporary `PaddleOCR` instance but doesn't cache it, so repeated calls with the same non-default language will re-create each time. | Acceptable for now. Per-worker engines are short-lived (SPEC 12.4). Could be optimized later if profiling shows overhead. Not in scope for this issue. |

---

## Requirement Coverage Matrix

| Acceptance Criterion | Covered in PLAN | Notes |
|---------------------|-----------------|-------|
| PaddleOCREngine class with __init__, recognize, name | Section 1b | Matches SPEC 12.4 signatures |
| _PADDLEOCR_LANG_MAP with zh/ko/ja/de mappings | Section 1a | 4 exception mappings |
| _to_paddleocr_lang() function | Section 1a | Pass-through for unmapped |
| __init__ does import paddleocr | Section 1b | Lazy import in constructor |
| recognize() converts PIL->numpy, parses results | Section 1b | np.array() + nested parse |
| Empty/None handling | Section 1b, tests 3c | 3 empty-page test variants |
| Lines joined with \n | Section 1b | Line-level, not word-level |
| Confidence 0.0-1.0 native | Section 1b, test | No scaling (unlike Tesseract /100) |
| name() returns "paddleocr" | Section 1b | Matches SPEC |
| Satisfies OCREngineInterface Protocol | Test 3c, 3d | isinstance check |
| Factory returns PaddleOCREngine when available | Section 1c, test 3e | 3-line replacement |
| Factory fallback with W_OCR_ENGINE_FALLBACK | Preserved unchanged | Existing code, not modified |
| Exported from utils/__init__.py | Section 2 | Import + __all__ |
| Existing tests pass | Analysis in check 5 | No breaking changes |
| Unit tests without PaddleOCR | All tests use mocks | @pytest.mark.unit |
| Integration tests gated | Section 3f | @pytest.mark.ocr_paddle |

All 16 acceptance criteria are covered. No gaps.

---

## Scope Containment

- File count (3 files modified) matches SIMPLE complexity classification.
- No ROADMAP items implemented.
- No concrete backends added inside the package (PaddleOCREngine is an adapter wrapping an external package, consistent with existing TesseractEngine pattern).
- No ABC base classes introduced.

---

## Recommendation

**APPROVED** -- Proceed to PATCH. No blockers. Three low-severity findings documented for awareness but require no plan changes.

---

AGENT_RETURN: .agents/outputs/plan-check-44-021426.md
