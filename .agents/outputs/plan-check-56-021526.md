---
issue: 56
agent: PLAN-CHECK
date: 2026-02-15
plan_artifact: map-plan-56-021526.md
status: PASS (with advisory notes)
---

# PLAN-CHECK: Issue #56 -- Scaffold ingestkit-forms package

## Executive Summary

The MAP-PLAN for issue #56 is well-structured, covers all required modules from spec section 4.4, includes all 7 optional dependency groups from section 15.1, and follows sibling package patterns correctly. Two advisory notes are raised (version number deviation and omission of SPEC.md/ROADMAP.md) but neither blocks implementation.

---

## Check 1: Module Coverage (spec 4.4)

| Spec 4.4 Module | Plan File # | Status |
|---|---|---|
| `__init__.py` | File 2 | ✅ |
| `protocols.py` | File 3 | ✅ |
| `models.py` | File 4 | ✅ |
| `errors.py` | File 5 | ✅ |
| `config.py` | File 6 | ✅ |
| `idempotency.py` | File 7 | ✅ |
| `matcher.py` | File 8 | ✅ |
| `router.py` | File 9 | ✅ |
| `api.py` | File 10 | ✅ |
| `extractors/__init__.py` | File 11 | ✅ |
| `extractors/native_pdf.py` | File 12 | ✅ |
| `extractors/ocr_overlay.py` | File 13 | ✅ |
| `extractors/excel_cell.py` | File 14 | ✅ |
| `output/__init__.py` | File 15 | ✅ |
| `output/db_writer.py` | File 16 | ✅ |
| `output/chunk_writer.py` | File 17 | ✅ |
| `tests/__init__.py` | File 18 | ✅ |
| `tests/conftest.py` | File 19 | ✅ |
| `tests/test_matcher.py` | File 20 | ✅ |
| `tests/test_extractors.py` | File 21 | ✅ |
| `tests/test_output.py` | File 22 | ✅ |
| `tests/test_api.py` | File 23 | ✅ |
| `tests/test_router.py` | File 24 | ✅ |
| `tests/test_config.py` | File 25 | ✅ |
| `tests/fixtures/` | File 26 | ✅ |

**Result: 25/25 modules covered. PASS.**

---

## Check 2: Optional Dependency Groups (spec 15.1)

| Spec 15.1 Group | In Plan | Status |
|---|---|---|
| `pdf` | ✅ PyMuPDF>=1.24 | ✅ |
| `pdf-mit` | ✅ pdfplumber>=0.11, pypdf>=4.0 | ✅ |
| `ocr` | ✅ paddleocr>=2.7 with platform marker | ✅ |
| `ocr-lite` | ✅ pytesseract>=0.3.10 | ✅ |
| `vlm` | ✅ httpx>=0.27 | ✅ |
| `all` | ✅ ingestkit-forms[pdf,ocr,vlm] | ✅ |
| `dev` | ✅ pytest>=7.0, pytest-cov, pyyaml>=6.0 | ✅ |

**Result: 7/7 groups covered. PASS.**

---

## Check 3: pyproject.toml Sibling Pattern Compliance

| Element | Sibling Pattern | Plan | Status |
|---|---|---|---|
| `[build-system]` setuptools>=68 | ✅ | ✅ | ✅ |
| `[tool.setuptools.packages.find]` where=["src"] | ✅ | ✅ | ✅ |
| `[tool.pytest.ini_options]` testpaths + markers | ✅ | ✅ | ✅ |
| `ingestkit-core>=0.1.0` in core deps | ✅ (excel, pdf both have it) | ✅ | ✅ |
| `requires-python = ">=3.10"` | ✅ | ✅ | ✅ |

**Advisory -- Version number:** Spec 15.1 says `version = "1.0.0"` but plan uses `version = "0.1.0"` to match sibling packages. This is the correct choice for a scaffold -- sibling consistency takes priority over the spec's aspirational version. No action needed.

**Advisory -- ingestkit-core not in spec 15.1:** The spec's core deps are `pydantic`, `openpyxl`, `Pillow` only. The plan adds `ingestkit-core>=0.1.0` following sibling pattern. This is correct -- all sibling packages depend on core.

**Result: PASS.**

---

## Check 4: Acceptance Criteria Completeness

| Criterion | Verifiable | Maps to Plan | Status |
|---|---|---|---|
| Directory structure matches spec 4.4 | Yes -- file listing | Files 1-26 | ✅ |
| pyproject.toml has 7 optional groups | Yes -- file content | File 1 | ✅ |
| Core deps include core+pydantic+openpyxl+Pillow | Yes -- file content | File 1 | ✅ |
| All source modules exist with docstrings | Yes -- import check | Files 2-17 | ✅ |
| Subpackages have `__init__.py` | Yes -- file listing | Files 11, 15 | ✅ |
| conftest.py has fixture placeholders | Yes -- file content | File 19 | ✅ |
| All test stubs exist | Yes -- file listing | Files 20-25 | ✅ |
| fixtures/ directory exists | Yes -- file listing | File 26 | ✅ |
| pip install succeeds | Yes -- verification gate | Gates section | ✅ |
| import succeeds | Yes -- verification gate | Gates section | ✅ |

**Result: 10/10 criteria verifiable and mapped. PASS.**

---

## Check 5: Missing Files / Scope Issues

**SPEC.md and ROADMAP.md:** Spec 4.4 lists these in the tree. The plan explicitly omits them (plan line 92), arguing the spec lives at `docs/specs/` and ROADMAP is deferred. This is a reasonable scope choice for a scaffold issue -- creating placeholder SPEC.md/ROADMAP.md adds no value when the authoritative spec is already in `docs/specs/`. If the issue owner wants them, they can be added as a follow-up. **Not blocking.**

**No other missing files detected.** All 14 source modules, 2 subpackage `__init__.py` files, 6 test stubs, 1 conftest, 1 fixtures directory, and 1 pyproject.toml are accounted for.

---

## Verdict

**PASS** -- Plan is ready for PATCH. Two advisory notes (version 0.1.0 vs 1.0.0, omitted SPEC.md/ROADMAP.md) are documented but do not block implementation.

AGENT_RETURN: .agents/outputs/plan-check-56-021526.md
