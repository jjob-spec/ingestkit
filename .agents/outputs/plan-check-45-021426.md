---
issue: 45
agent: PLAN-CHECK
date: 2026-02-14
status: PASS
---

# PLAN-CHECK: Issue #45 -- Implement async wrapper aprocess()

## Executive Summary

The MAP-PLAN for issue #45 is accurate, well-scoped, and aligned with SPEC 18.2. The `asyncio.to_thread()` approach is correct for Python 3.10+. The `process()` signature in the plan matches the actual code. Test approach is sound. One actionable finding: `pytest-asyncio` is missing from dev dependencies and must be added. No scope creep detected.

## Validation Checklist

### Spec Alignment (SPEC 18.2, lines 1501-1518)

- [x] `aprocess()` wraps `process()` via `asyncio.to_thread()` -- matches SPEC 18.2 line 1511
- [x] Sync `process()` remains canonical, not deprecated -- matches SPEC line 1516
- [x] Async is for caller integration (FastAPI, async job queues) -- matches SPEC line 1513
- [x] Test requirement R-EXE-1: `test_router::test_aprocess_matches_process` -- matches SPEC line 2045

### process() Signature Verification

- [x] Plan claims: `def process(self, file_path: str, source_uri: str | None = None) -> ProcessingResult:` at line 147
- [x] Actual: identical signature confirmed at router.py line 147-151
- [x] `aprocess()` mirrors this signature exactly -- correct

### asyncio.to_thread() Correctness

- [x] `asyncio.to_thread()` available since Python 3.9; project requires >= 3.10 -- safe
- [x] Usage `asyncio.to_thread(self.process, file_path, source_uri)` -- correct positional arg passing
- [x] Preferred over `loop.run_in_executor()` (simpler, modern API) -- correct choice
- [x] Thread safety: `process()` uses instance-level components; concurrent `aprocess()` calls on the same router instance could race. This is acceptable for v1.1 scope (SPEC does not require concurrent safety) but worth noting for future.

### Insertion Point Verification

- [x] Plan says: insert after line 423, before `process_batch()` at line 428
- [x] Actual: line 423 is `return result` inside a try block; line 425-426 is `finally: doc.close()`; line 428 is `def process_batch(`
- [x] Correct insertion point: between the end of `process()` method (line 427) and `process_batch()` (line 428)

### Test Approach

- [x] Test class `TestAprocess` follows existing pattern: `@pytest.mark.unit` class decorator, descriptive docstring
- [x] Plan correctly uses existing helpers: `_make_processing_result`, `_make_security_metadata`, `_make_ingest_key_obj`, `_make_document_profile`, `_make_mock_doc`
- [x] `_make_document_profile` imported from conftest (test_router.py line 45) -- plan's mocking approach is consistent
- [x] Plan correctly recommends the simplified `test_aprocess_matches_process` (mock `process()` directly) over the verbose version -- preferred
- [x] `test_aprocess_propagates_source_uri` verifies argument forwarding -- good coverage
- [x] Insertion after `TestProcessBatch` (line 1082-1153), before `TestCreateDefaultRouter` (line 1154) -- matches test file structure

### Dependency: pytest-asyncio

- [x] **NOT in pyproject.toml** -- confirmed by reading dev dependencies (lines 48-55)
- [x] Plan correctly identifies this as conditional and includes the fix: add `"pytest-asyncio>=0.23"` to dev extras
- [x] `asyncio` marker also not in `[tool.pytest.ini_options]` markers list -- PATCH should add it or use `pytest-asyncio` auto mode

### Scope Check

- [x] No changes to `process()` itself -- correct
- [x] No changes to `__init__.py` exports -- correct (method is on class)
- [x] No new error codes -- correct (wrapper re-raises whatever `process()` raises)
- [x] No ROADMAP items pulled in
- [x] File count: 2 modified (router.py, test_router.py) + 1 conditional (pyproject.toml) -- appropriate for SIMPLE

## Issues Found

**Minor (non-blocking):**

1. **pytest-asyncio mode configuration**: When adding `pytest-asyncio`, the PATCH agent should also consider adding `asyncio_mode = "auto"` to `[tool.pytest.ini_options]` in pyproject.toml, OR ensure each async test uses `@pytest.mark.asyncio`. The plan uses `@pytest.mark.asyncio` decorators, which works with the default "strict" mode -- this is fine as-is.

2. **Class docstring update**: Plan proposes updating the PDFRouter docstring at line 78 to mention `aprocess`. The exact text references "the public API." but the actual text at line 78-79 is `the public API.` followed by a blank line and `Parameters`. The PATCH agent should read the exact line content before editing.

## Recommendation

**APPROVED** -- The MAP-PLAN is accurate, complete, and correctly scoped. `asyncio.to_thread()` is the right approach. The simplified test pattern (mocking `process()` directly) is preferred. `pytest-asyncio>=0.23` must be added to dev dependencies. Proceed to PATCH.

AGENT_RETURN: .agents/outputs/plan-check-45-021426.md
