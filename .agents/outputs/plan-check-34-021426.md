---
issue: 34
agent: PLAN-CHECK
date: 2026-02-14
status: PASS
---

# PLAN-CHECK: Issue #34 -- OCR Engine Abstraction

## Validation Checklist

### 1. Requirement Coverage

Every acceptance criterion maps to a planned implementation task:

- [x] `OCREngineInterface` Protocol with `@runtime_checkable` -- PLAN section 1c, uses `typing.Protocol` + `@runtime_checkable` decorator, two methods `recognize()` and `name()`
- [x] `OCRPageResult` Pydantic model with correct fields -- PLAN section 1b, fields: `text`, `confidence`, `word_confidences`, `language_detected`
- [x] `TesseractEngine` implementing the protocol -- PLAN sections 1g-1i, constructor + `recognize()` + `name()`
- [x] Language mapping ISO 639-1 to Tesseract ISO 639-3 -- PLAN section 1d (`_LANGUAGE_MAP`) + section 1e (`_to_tesseract_lang()`)
- [x] `_tesseract_available()` dual check -- PLAN section 1f, checks `shutil.which("tesseract")` then `import pytesseract`
- [x] `create_ocr_engine()` factory with correct return type -- PLAN section 1j, returns `tuple[OCREngineInterface, list[str]]`
- [x] PaddleOCR fallback with `W_OCR_ENGINE_FALLBACK` -- PLAN section 1j, `ImportError` catch triggers warning and fall-through
- [x] `EngineUnavailableError` as `Exception` subclass -- PLAN section 1a, simple `Exception` subclass
- [x] `OCRPageResult` in `ocr_engines.py` (not `models.py`) -- PLAN section 1b explicitly states "Lives here per SPEC 12.2, NOT in `models.py`"
- [x] Logger name `"ingestkit_pdf.utils.ocr_engines"` -- PLAN section 1 (after imports)
- [x] No ABC classes -- Protocol only throughout
- [x] Exports added to `utils/__init__.py` -- PLAN section 2
- [x] All unit tests mocked -- PLAN section 3 header, all tests `@pytest.mark.unit`

### 2. File Count vs Complexity

COMPLEX classification requires multi-file changes. PLAN specifies:
- 2 files to create (`ocr_engines.py` ~180 lines, `test_ocr_engines.py` ~250 lines)
- 1 file to modify (`utils/__init__.py`)
- Total: 3 files

**Verdict:** Acceptable for COMPLEX. The complexity stems from the protocol + adapter + factory pattern with multiple interacting components, not raw file count.

### 3. Protocol Uses @runtime_checkable (Not ABC)

Verified in PLAN section 1c:
```python
@runtime_checkable
class OCREngineInterface(Protocol):
```
No `ABC` or `ABCMeta` anywhere in the PLAN. Consistent with project convention (CLAUDE.md: "Do NOT introduce ABC base classes").

### 4. OCRPageResult Location

PLAN section 1b explicitly defines `OCRPageResult` in `utils/ocr_engines.py`. The PLAN correctly distinguishes it from the existing `OCRResult` in `models.py` (which is a higher-level pipeline result wrapping `OCRPageResult` with page number, DPI, etc.).

### 5. Language Code Mapping Completeness

PLAN section 1d provides 20 language mappings covering: en, fr, de, es, it, pt, nl, ru, zh, ja, ko, ar, hi, pl, tr, vi, th, uk, cs, ro. Unknown codes pass through via `_LANGUAGE_MAP.get(language, language)`.

**Verified:** All major ISO 639-1 codes that have non-obvious Tesseract counterparts are covered (e.g., `zh -> chi_sim`, `ko -> kor`, `cs -> ces`, `ro -> ron`). Pass-through handles codes where ISO 639-1 and 639-3 happen to match or where users provide Tesseract codes directly.

### 6. Factory Fallback Chain

PLAN section 1j implements the SPEC 12.5 fallback chain:
1. If `config.ocr_engine == OCREngine.PADDLEOCR`: try import, catch `ImportError` -> append `"W_OCR_ENGINE_FALLBACK"` -> fall through
2. Check `_tesseract_available()`: if False -> raise `EngineUnavailableError`
3. Return `TesseractEngine(lang=config.ocr_language)`

Matches SPEC pseudocode at lines 1247-1264 exactly.

**Minor note:** The PLAN shows three different approaches for the PaddleOCR stub (self-import, explicit raise, import-then-raise). The final approach (`import paddleocr` + raise because adapter not implemented) is reasonable but adds unnecessary complexity. The simplest correct approach is the first one shown (try to import `PaddleOCREngine` from the module, which will fail with `ImportError` since the class doesn't exist yet). Either way, the fallback behavior is correct. PATCH agent should pick one approach and stick with it.

### 7. `_tesseract_available()` Checks Both Binary and Import

PLAN section 1f:
1. `shutil.which("tesseract")` -- checks binary on PATH
2. `import pytesseract` -- checks Python binding importable

Both must pass. Binary check is first (cheaper). Confirmed correct per MAP decision D4.

### 8. Test Coverage

34 tests across 7 test classes:

| Class | Count | Covers |
|-------|-------|--------|
| `TestOCRPageResult` | 6 | Model validation |
| `TestOCREngineInterface` | 2 | Protocol `isinstance` checks |
| `TestLanguageMapping` | 5 | `_to_tesseract_lang()` + `_LANGUAGE_MAP` |
| `TestTesseractAvailable` | 4 | `_tesseract_available()` all branches |
| `TestTesseractEngine` | 8 | `recognize()`, `name()`, filtering, scaling |
| `TestCreateOCREngine` | 7 | Factory happy path, fallback, error |
| `TestEngineUnavailableError` | 2 | Exception basics |

All tests use mocked `pytesseract` and `shutil.which`. No real Tesseract binary required. Coverage is comprehensive -- every public function, happy path, edge cases, and error conditions are tested.

## Issues Found

**ISSUE-1 (LOW): PaddleOCR stub indecision.** The PLAN presents three different approaches for the PaddleOCR branch in the factory (sections show "Wait", "Correction", "Actually, better approach"). PATCH agent should use the simplest approach that matches the SPEC pseudocode pattern: attempt to import/instantiate `PaddleOCREngine`, which will naturally `ImportError` since the class doesn't exist. This avoids hard-coding `raise ImportError(...)`.

**ISSUE-2 (INFO): SPEC does not show `@runtime_checkable` on `OCREngineInterface`.** The PLAN adds it (correct per project convention), but the SPEC pseudocode at line 1192 omits it. This is a deliberate deviation, justified by the project rule requiring `@runtime_checkable` on all Protocol classes. No action needed.

## Verdict

**PASS** -- The PLAN is complete, correctly structured, and ready for PATCH. All 13 acceptance criteria map to implementation tasks. Protocol patterns, model placement, language mapping, factory chain, availability checking, and test coverage all meet requirements.

AGENT_RETURN: .agents/outputs/plan-check-34-021426.md
