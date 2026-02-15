---
issue: 47
agent: PLAN-CHECK
date: 2026-02-14
status: PASS (with 4 advisories)
---

# PLAN-CHECK: Issue #47 — Integration Test Suite and Benchmark Report

## Executive Summary

The PLAN for issue #47 is comprehensive and well-structured. It correctly identifies the PDFRouter API, programmatic PDF fixtures, existing markers, and backend import paths. Four advisories found: (1) session-scoped SQLite fixture with `:memory:` will not survive across tests in session scope correctly, (2) `_cleanup_qdrant` fixture depends on `real_vector_store` but bypasses it with a raw QdrantClient, (3) the `PDFProcessorConfig` parameter in the plan uses `default_collection="test_integration"` but the actual config field is `default_collection` with default `"helpdesk"` -- this is fine but the plan's `written.vector_collection == "test_integration"` assertion (File 1, test_path_a_written_artifacts) must match the config value set, and (4) benchmark script duplicates PDF generation code from conftest.py rather than extracting a shared module.

---

## Validation Checklist

### Requirement Coverage

- [x] Integration tests for Path A (TextExtractor) -- 5 tests planned
- [x] Integration tests for Path B (OCRProcessor) -- 3 tests planned
- [x] Integration tests for Path C (ComplexProcessor unavailable) -- 1 test, correctly expects error
- [x] Edge case tests (encrypted, garbled, can_handle) -- 3 tests planned
- [x] Benchmark per-stage latency tests -- 3 tests with SLO max values
- [x] Benchmark throughput tests -- 2 tests (Path A >= 50 pp/s, Path B >= 10 pp/s)
- [x] Standalone benchmark script with JSON report -- scripts/benchmark.py planned
- [x] `benchmark` marker registration in pyproject.toml -- yes
- [x] No binary PDF fixtures committed -- all programmatic via reportlab/PyMuPDF
- [x] Skip decorators for unavailable backends -- `_qdrant_available()` and `_ollama_available()` helpers

### PDFRouter API Correctness

- [x] Constructor: `PDFRouter(vector_store, structured_db, llm, embedder, config)` -- matches `router.py:97-104`
- [x] `process(file_path: str, source_uri: str | None)` -- matches `router.py:149-153`
- [x] `can_handle(file_path: str) -> bool` -- matches `router.py:145-147`
- [x] Return type `ProcessingResult` -- matches `models.py:297-322`

### Backend Imports

- [x] Plan imports from `ingestkit_excel.backends` -- matches `router.py:949-954`
- [x] `QdrantVectorStore`, `SQLiteStructuredDB`, `OllamaLLM`, `OllamaEmbedding` -- all correct class names

### Fixture Availability (conftest.py)

- [x] `text_native_pdf` -- exists, function-scoped, 3-page digital PDF (line 446)
- [x] `scanned_pdf` -- exists, function-scoped, 2-page image-only PDF (line 504)
- [x] `complex_pdf` -- exists, function-scoped, 2-page mixed content (line 557)
- [x] `encrypted_pdf` -- exists, function-scoped, AES-256 encrypted (line 704)
- [x] `garbled_pdf` -- exists, function-scoped, low printable ratio (line 728)
- [x] Mock backends: `MockLLMBackend`, `MockVectorStoreBackend`, `MockEmbeddingBackend`, `MockStructuredDBBackend` -- all exist

### Marker Registration

- [x] `integration` already registered in pyproject.toml line 66
- [x] `ocr` already registered in pyproject.toml line 67
- [x] `benchmark` NOT registered -- plan correctly adds it (File 4)
- [ ] **Advisory**: `ocr_paddle` also exists (line 68) -- plan does not reference it, which is fine

### Scope Containment

- [x] No ROADMAP items implemented
- [x] No new Pydantic models added
- [x] No changes to core pipeline code
- [x] No concrete backends added inside ingestkit-pdf
- [x] File count (3 new + 1 modified = 4 files) appropriate for COMPLEX classification

---

## Issues Found

### Advisory 1: Session-scoped SQLite `:memory:` fixture

The plan defines `real_structured_db` as `session`-scoped returning `SQLiteStructuredDB(":memory:")`. In-memory SQLite databases are connection-bound. If the session fixture returns a single instance, this is fine as long as no test closes the connection. However, session scope means the same in-memory DB accumulates state across all tests. Consider function scope or adding cleanup logic.

**Severity**: Low -- SQLite is not actively used by Path A/B text/OCR processing, so state leakage is unlikely.

### Advisory 2: Cleanup fixture bypasses `real_vector_store`

The `_cleanup_qdrant` fixture declares `real_vector_store` as a dependency but then creates a raw `QdrantClient` for cleanup. This is redundant and fragile -- if `real_vector_store` changes its URL, the cleanup fixture won't follow. Recommend using the `real_vector_store` instance directly or at minimum sharing the URL constant.

**Severity**: Low -- functional but creates a maintenance coupling.

### Advisory 3: PDF fixtures are function-scoped, not session-scoped

All conftest.py PDF fixtures (`text_native_pdf`, `scanned_pdf`, etc.) use `@pytest.fixture()` (function scope) and write to `tmp_path` (also function-scoped). The plan's session-scoped `integration_router` fixture takes function-scoped PDF fixtures as peers (not dependencies), so this is fine as long as tests request the PDF fixture directly alongside the router. However, running N integration tests will regenerate PDFs N times. This is acceptable for the test counts in the plan (12 tests total).

**Severity**: None -- correct as designed.

### Advisory 4: Benchmark script duplicates conftest.py PDF generation

The plan proposes extracting PDF generation helpers into `scripts/benchmark.py` duplicating conftest.py logic. This creates two maintenance locations for the same code. Consider extracting shared helpers into a `tests/_pdf_generators.py` module that both conftest.py and the benchmark script import.

**Severity**: Low -- acceptable for initial implementation; can refactor later.

---

## Benchmark SLO Targets Assessment

| Metric | Plan Target | Plan Max | Reasonableness |
|--------|-------------|----------|----------------|
| Path A throughput | >= 50 pages/sec | -- | Reasonable for 3-page text PDFs on modern hardware |
| Path B throughput | >= 10 pages/sec | -- | Reasonable for OCR, depends on Tesseract availability |
| Security scan latency | < 100ms target | < 500ms assert | Sound -- `scan()` does file stat + PyMuPDF metadata only |
| Profile extraction | < 2s target | < 5s assert | Sound -- 3-page PDF profiling is lightweight |
| Tier 1 classification | < 200ms target | < 500ms assert | Sound -- rule-based, no I/O |

The plan correctly uses max values (not targets) for test assertions, providing headroom for CI variability.

---

## Cross-Check: Plan vs MAP Discrepancies

| Item | MAP | PLAN | Status |
|------|-----|------|--------|
| Package name | Says `ingestkit-excel` in section 3.1 | Correctly uses `ingestkit-pdf` | MAP error, PLAN correct |
| SPEC sections 22.5, 25 | MAP notes these don't exist | PLAN references them in summary but doesn't depend on them | Acceptable |
| `PDFRouter` vs `ExcelRouter` | MAP section 4 says `ExcelRouter` | PLAN correctly uses `PDFRouter` | MAP error, PLAN correct |
| "pages" metric | MAP flags ambiguity | PLAN uses actual PDF page counts | Resolved correctly |

---

## Recommendation

**PASS** -- The plan is implementable as written. The four advisories are all low-severity and do not block PATCH. The PDFRouter API usage, fixture references, skip decorators, and SLO targets are all sound.

---

AGENT_RETURN: .agents/outputs/plan-check-47-021426.md
