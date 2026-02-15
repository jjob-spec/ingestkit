---
issue: 43
agent: PLAN-CHECK
date: 2026-02-14
status: PASS
plan_artifact: .agents/outputs/plan-43-021426.md
map_artifact: .agents/outputs/map-43-021426.md
---

# PLAN-CHECK: Issue #43 — Test Infrastructure, Mock Backends, and PDF Fixtures

## Verdict: PASS (with 3 advisory notes)

The plan is well-structured, backward-compatible, and implementable. No blocking issues found.

---

## 1. Backward Compatibility Check

**Consumers identified (3 files):**

| File | Imports from conftest |
|------|----------------------|
| `test_llm_classifier.py` | `MockLLMBackend`, `_make_document_profile`, `_make_page_profile`, `_valid_response` |
| `test_text_extractor.py` | `_make_document_profile`, `_make_page_profile` |
| `test_ocr_processor.py` | `MockEmbeddingBackend`, `MockLLMBackend`, `MockVectorStoreBackend`, `_make_document_profile`, `_make_extraction_quality`, `_make_page_profile` |

**Assessment:** SAFE. The plan explicitly states "Do NOT rename or restructure existing classes" (Section 2.1). All changes are additive:
- New methods added to existing classes (enqueue helpers, error injection, assertion properties)
- No existing method signatures modified
- No class renames
- Sentinel handling inserted BEFORE `isinstance(response, Exception)` check, preserving existing Exception-in-queue behavior

**Verified:** `test_ocr_processor.py` constructs `MockLLMBackend()` with no arguments and `MockVectorStoreBackend()` / `MockEmbeddingBackend()` with no arguments. New `__init__` fields use defaults (`_errors = {}`, `_error_on_next = None`), so zero-arg construction remains valid.

---

## 2. Mock Backend Enhancement Check

| Backend | Error Injection | Assertion Helpers | Enqueue Methods | Status |
|---------|----------------|-------------------|-----------------|--------|
| MockLLMBackend | Sentinels (timeout, connection, malformed JSON) | `call_count`, `assert_called_with_model` | `enqueue_classify`, `enqueue_generate`, `enqueue_timeout`, `enqueue_connection_error` | OK |
| MockVectorStoreBackend | `fail_next_upsert`, `fail_next_ensure` | `total_chunks_upserted` | N/A | OK |
| MockEmbeddingBackend | `fail_next_embed` | `total_texts_embedded` | N/A | OK |
| MockStructuredDBBackend | `fail_next_create` | N/A | N/A | OK |

**All 4 backends covered.**

---

## 3. PDF Fixtures Check

| Fixture | SPEC 22.2 Required | In Plan | Implementation Approach | Feasibility |
|---------|-------------------|---------|------------------------|-------------|
| `text_native_pdf(tmp_path)` | Yes | Section 2.2.1 | reportlab Canvas, 3 pages | OK |
| `scanned_pdf(tmp_path)` | Yes | Section 2.2.2 | PIL Image + reportlab drawImage | OK |
| `complex_pdf(tmp_path)` | Yes | Section 2.2.3 | reportlab platypus Table + Frame | OK |
| `encrypted_pdf(tmp_path)` | Yes | Section 2.2.4 | fitz AES-256 encryption | OK |
| `garbled_pdf(tmp_path)` | Yes | Section 2.2.5 | Non-printable Unicode chars | OK |

**All 5 fixtures planned. No binary files committed (all generated at test time via `tmp_path`).**

---

## 4. Dependency Check

| Dependency | Required By | Available | Status |
|------------|-------------|-----------|--------|
| `reportlab>=4.0` | All PDF fixtures | pyproject.toml line 52 (dev) | OK |
| `Pillow>=10.0` | scanned_pdf | pyproject.toml line 17 (core) | OK |
| `pymupdf>=1.24` (fitz) | encrypted_pdf | pyproject.toml line 12 (core) | OK |
| `pdfplumber>=0.10` | test_complex_pdf_has_tables | pyproject.toml line 14 (core) | OK |

---

## 5. Pytest Markers Check

All four markers already exist in `pyproject.toml` lines 62-67:
- `unit` - present
- `integration` - present
- `ocr` - present
- `ocr_paddle` - present

Plan correctly states "No new pytest markers needed" (Section 7).

---

## 6. SPEC Conformance

Plan aligns with SPEC 22.1-22.5:
- 22.1 (Mock Backends): Four protocols mocked, error injection added
- 22.2 (Programmatic PDF Generation): Five fixtures, reportlab-based, no binaries
- 22.3 (Test Coverage): `test_utils.py` covers infrastructure validation
- 22.4 (Markers): Already exist
- 22.5 (Release Gates): Not in scope for this issue (correct)

---

## 7. Advisory Notes (Non-Blocking)

**A1: Sentinel pattern diverges slightly from Excel.** The Excel `MockLLM` uses inline string literals (`"__TIMEOUT__"`, `"__MALFORMED_JSON__"`) without module-level constants. The PDF plan introduces `_SENTINEL_TIMEOUT` etc. as module-level constants. This is arguably better (avoids magic strings in test code), but the naming convention differs from the Excel package. Not blocking -- consistency within the PDF package is more important.

**A2: `enqueue_classify` and `enqueue_generate` share the same queue.** The Excel `MockLLM` has separate `classify_responses` and `generate_responses` queues. The PDF plan keeps the single `_responses` queue (matching current implementation) and adds `enqueue_classify`/`enqueue_generate` as aliases that both append to the same queue. This works but could confuse developers expecting separate queues. The plan should note this in a comment. Not blocking since existing tests already use the shared queue.

**A3: `garbled_pdf` feasibility risk acknowledged.** The plan correctly identifies that reportlab may sanitize non-printable characters (Section 6 Risks). The fallback to `fitz.TextWriter` is sound. PATCH should verify `printable_ratio < 0.5` immediately after generating the fixture.

---

## 8. Scope Containment

- No changes to production code (`models.py`, `errors.py`, `config.py`, `protocols.py`) -- correct
- Only 2 files modified/created (`conftest.py`, `test_utils.py`) -- appropriate for issue scope
- No ROADMAP items implemented -- correct
- No binary fixtures committed -- correct

---

## Recommendation

**PASS -- proceed to PATCH.**

AGENT_RETURN: .agents/outputs/plan-check-43-021426.md
