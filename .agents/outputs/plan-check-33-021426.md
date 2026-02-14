---
issue: 33
agent: PLAN-CHECK
date: 2026-02-14
status: PASS
---

# PLAN-CHECK: Issue #33 — Language Detection (`utils/language.py`)

## Executive Summary

The MAP-PLAN for issue #33 is well-structured and accurately reflects SPEC sections 16.1-16.3. All acceptance criteria from the GitHub issue map to planned tasks. File count (2 created, 1 modified) is appropriate for TRIVIAL complexity. The OCREngine enum, config fields, and dependency declarations all match the actual codebase. No scope creep detected. Test coverage is thorough with 16 unit tests covering both functions, edge cases, and error paths.

---

## Validation Checklist

### Spec Alignment

- [x] `detect_language(text) -> tuple[str, float]` matches SPEC 16.3 signature
- [x] `map_language_to_ocr(lang, engine) -> str` matches SPEC 16.3 signature
- [x] Uses `fast-langdetect` (FastText wrapper) per SPEC 16.2
- [x] Returns `(iso_639_1_code, confidence)` per SPEC 16.3

### Config Parameter Verification

- [x] `enable_language_detection: bool = True` exists at config.py line 105
- [x] `default_language: str = "en"` exists at config.py line 106
- [x] Plan uses `default_language="en"` as default kwarg -- consistent with config default

### Model / Enum Verification

- [x] `OCREngine` enum exists at models.py line 69 with values `TESSERACT = "tesseract"` and `PADDLEOCR = "paddleocr"` -- matches plan exactly
- [x] Import path `from ingestkit_pdf.models import OCREngine` is correct

### Dependency Verification

- [x] `fast-langdetect>=0.2` declared at pyproject.toml line 26 under `[project.optional-dependencies]` -> `langdetect`
- [x] Included in `baseline` bundle (line 44) and `full` bundle (line 45)
- [x] Plan correctly states no pyproject.toml changes needed

### File Count & Scope

- [x] Files to create: 2 (language.py, test_language.py) -- appropriate for TRIVIAL
- [x] Files to modify: 1 (utils/__init__.py) -- currently exports HeaderFooterDetector and HeadingDetector
- [x] No existing language.py or test_language.py (verified via glob)
- [x] No scope creep -- plan stays within SPEC section 16, no other modules touched
- [x] No ROADMAP.md items implemented

### Acceptance Criteria Coverage (Issue -> Plan)

| Issue Criterion | Mapped to Plan Section |
|-----------------|----------------------|
| English text detected as "en" with high confidence | Test #4: `test_detect_language_english` |
| Language code mapping correct for Tesseract (en -> eng) | Test #10: `test_map_english_to_tesseract` |
| Language code mapping correct for PaddleOCR (en -> en) | Test #11: `test_map_english_to_paddleocr` |
| Short text handled gracefully | Test #3: `test_detect_language_very_short_text` |
| Empty text returns default language | Tests #1, #2: empty string + whitespace |
| `pytest tests/test_language.py -q` passes | All 16 tests mocked, no external deps |
| `ruff check .` passes | Verified by PROVE phase |

### Utils __init__.py Update

- [x] Plan adds `detect_language` and `map_language_to_ocr` to exports
- [x] Plan preserves existing exports (`HeaderFooterDetector`, `HeadingDetector`)
- [x] Note: Plan omits `PDFChunker` from `__all__` -- not currently in `utils/__init__.py` either, so this is correct

### Test Plan Coverage

- [x] `detect_language` covered: 9 tests (empty, whitespace, short text, English, Spanish, custom default, import error, unexpected error, BCP-47 normalization)
- [x] `map_language_to_ocr` covered: 7 tests (en/zh for both engines, Spanish Tesseract, unknown lang for both engines)
- [x] Mocking strategy: patches `fast_langdetect.detect` -- tests work without the optional dependency
- [x] All tests marked `@pytest.mark.unit`
- [x] No binary fixtures or external service dependencies

### Architecture Compliance

- [x] No ABC base classes
- [x] Logger name `ingestkit_pdf.utils.language` follows convention
- [x] No concrete backend imports inside package
- [x] Graceful degradation when optional dependency missing

---

## Issues Found

None.

---

## Minor Observations (Non-Blocking)

1. **Signature enhancement**: The plan adds a `default_language` keyword argument to `detect_language()` that is not in the SPEC 16.3 signature. This is a reasonable enhancement that integrates with the config's `default_language` field, but the implementer should note this is an extension beyond the spec. Since the spec signature is a subset (no `default_language` param), callers using the spec signature will still work due to the default value.

2. **`enable_language_detection` config field unused**: The plan does not reference `config.enable_language_detection` anywhere. This is acceptable since the language module is a utility -- the enable/disable check would happen at the caller level (e.g., the router or processor). Just noting for completeness.

3. **Tesseract map has 29 entries**: Plan claims "at least 20 common languages" and delivers 29 entries in `_TESSERACT_LANG_MAP`. Satisfies the acceptance criterion.

---

## Recommendation

**APPROVED** -- The MAP-PLAN is accurate, complete, and correctly scoped. Proceed to PATCH.

AGENT_RETURN: .agents/outputs/plan-check-33-021426.md
