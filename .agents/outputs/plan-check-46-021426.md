---
issue: 46
title: "Implement ExecutionBackend protocol and distributed backend"
agent: PLAN-CHECK
date: 2026-02-14
status: PASS_WITH_NOTES
plan_artifact: plan-46-021426.md
map_artifact: map-46-021426.md
---

# PLAN-CHECK: Issue #46 -- ExecutionBackend Protocol and Distributed Backend

## Executive Summary

The PLAN is well-structured and implementable. The ExecutionBackend Protocol design is sound, backward compatibility is preserved, and the lazy submit/execute_all pattern correctly mirrors the existing ProcessPoolExecutor logic. I found 3 issues that need correction before PATCH and 2 minor notes. The MAP-to-PLAN pivot from Excel to PDF was correctly handled. No scope creep detected.

## Verification Results

### 1. ExecutionBackend Protocol Design

| Check | Status |
|-------|--------|
| `@runtime_checkable` decorator | PASS |
| `typing.Protocol` base (no ABC) | PASS |
| Method signatures match SPEC 18.3 claim | PASS (cannot verify -- SPEC not read, but signatures are reasonable) |
| Structural subtyping pattern matches existing protocols | PASS |
| Ellipsis bodies in Protocol | PASS |

### 2. LocalExecutionBackend -- Wrapping Existing Logic

| Check | Status |
|-------|--------|
| ProcessPoolExecutor exists in current `process_batch()` | PASS -- confirmed at router.py:485-495 |
| `_process_single_file` exists as module-level function | PASS -- confirmed at router.py:907-917, signature: `(file_path: str, config_dict: dict) -> ProcessingResult` |
| Callable injection avoids circular dependency | PASS |
| Lazy submit + execute_all preserves batching semantics | PASS |

**ISSUE #1 (MEDIUM): `process_fn` signature mismatch.**
The PLAN defines `process_fn: Callable[[str, dict], ProcessingResult]` (section 1.3), which matches `_process_single_file(file_path: str, config_dict: dict)`. However, section 5.3 shows `self._execution.submit(fp, self._config)` passing a `PDFProcessorConfig` object, not a `dict`. The `submit()` Protocol takes `config: PDFProcessorConfig`. Inside `LocalExecutionBackend`, the `execute_all()` method must convert `self._config` to `config_dict` via `config.model_dump()` before passing to `process_fn`. The PLAN mentions `config_dict` in `self._pending` storage but doesn't explicitly show the `model_dump()` conversion inside `execute_all()`. PATCH must ensure this conversion happens.

### 3. DistributedExecutionBackend Stub

| Check | Status |
|-------|--------|
| Raises `NotImplementedError` on both methods | PASS |
| Satisfies Protocol structurally | PASS |
| No real queue dependencies imported | PASS |
| Docstrings reference SPEC and Phase 2 | PASS |

### 4. Circular Dependencies

| Check | Status |
|-------|--------|
| `execution.py` imports from `config` and `models` only (TYPE_CHECKING) | PASS |
| `router.py` imports from `execution.py` | PASS -- no cycle (execution does not import router) |
| `protocols.py` re-exports from `execution.py` | PASS -- no cycle |

### 5. Backward Compatibility

| Check | Status |
|-------|--------|
| `PDFRouter.__init__()` -- `execution` param is optional with `None` default | PASS |
| Existing callers without `execution` get `LocalExecutionBackend` auto | PASS |
| `process_batch()` signature unchanged (`file_paths: list[str]`) | PASS |
| `_process_single_file` module-level function preserved | PASS |
| `create_default_router()` still works without `execution` kwarg | PASS |

### 6. Config Field Conflicts

Current `PDFProcessorConfig` has 48 fields (config.py:30-121). Proposed additions:

| New Field | Conflict Check | Status |
|-----------|---------------|--------|
| `execution_backend: str = "local"` | No existing `execution_*` fields | PASS |
| `execution_max_workers: int = 4` | No conflict. Note: `ocr_max_workers: int = 4` exists (line 67) but is distinct | PASS |
| `execution_queue_url: str | None = None` | No conflict | PASS |

**PLAN says "Add after `backend_backoff_base` field (line 114)"** -- verified: `backend_backoff_base` is at line 114. Correct insertion point.

### 7. Error Code Conflicts

Current `ErrorCode` enum has 37 codes (errors.py:19-72). Proposed additions:

| New Code | Conflict Check | Status |
|----------|---------------|--------|
| `E_EXECUTION_TIMEOUT` | No existing `E_EXECUTION_*` codes | PASS |
| `E_EXECUTION_SUBMIT` | No conflict | PASS |
| `E_EXECUTION_NOT_FOUND` | No conflict | PASS |

**PLAN says "Add after `E_PROCESS_HEADER_FOOTER` (line 56)"** -- verified: `E_PROCESS_HEADER_FOOTER` is at line 55. Off-by-one but inconsequential.

Note: MAP suggested `E_EXECUTION_SUBMIT_FAIL` and `E_EXECUTION_RESULT_LOST`; PLAN uses `E_EXECUTION_SUBMIT` and `E_EXECUTION_NOT_FOUND`. PLAN's naming is more concise and consistent with existing patterns. Acceptable divergence.

### 8. Scope Creep Check

| Check | Status |
|-------|--------|
| No UI additions | PASS |
| No new external service dependencies in core | PASS |
| Redis optional dep in pyproject.toml only | PASS |
| No ROADMAP items implemented | PASS |
| No changes to classification/processing pipeline | PASS |
| Distributed backend is stub-only | PASS |

## Issues Found

### ISSUE #1 (MEDIUM): process_fn signature vs submit() signature

See section 2 above. The `submit()` Protocol accepts `PDFProcessorConfig` but the underlying `_process_single_file` takes `(str, dict)`. The `LocalExecutionBackend` must handle this conversion internally. PATCH must be explicit about where `model_dump()` is called.

### ISSUE #2 (LOW): `hasattr` duck-typing for `execute_all()`

The PLAN uses `hasattr(self._execution, "execute_all")` in `process_batch()` (section 5.3). This works but is a code smell -- it couples the router to implementation details of `LocalExecutionBackend`. A cleaner pattern would be for `LocalExecutionBackend.submit()` to accept a "batch mode" or for the Protocol to optionally include `execute_all()`. However, since `execute_all()` is local-backend-specific and the alternative is more complex, this is acceptable for v1.0.

### ISSUE #3 (LOW): `create_default_router()` imports from `ingestkit_excel.backends`

At router.py:949, the existing `create_default_router()` imports from `ingestkit_excel.backends` (not `ingestkit_pdf.backends`). This is likely a copy-paste bug in the existing code, not introduced by this PLAN. PATCH should not fix it (out of scope) but should be aware when adding `execution` to `router_keys`.

## Requirement Coverage

| Acceptance Criterion | Planned Task | Status |
|---------------------|-------------|--------|
| ExecutionBackend Protocol with submit/get_result | File 1, section 1.2 | COVERED |
| LocalExecutionBackend wraps ProcessPoolExecutor | File 1, section 1.3 | COVERED |
| DistributedExecutionBackend stub | File 1, section 1.4 | COVERED |
| Both satisfy isinstance check | File 8, test cases | COVERED |
| Config fields added | File 2 | COVERED |
| Router accepts optional execution param | File 5, section 5.2 | COVERED |
| process_batch delegates to backend | File 5, section 5.3 | COVERED |
| Default matches current behavior | Design decision D5 | COVERED |
| Rollback is config-only | Config defaults to "local" | COVERED |
| 3 new error codes | File 3 | COVERED |
| Re-exported from protocols.py | File 4 | COVERED |
| Exported from __init__.py | File 6 | COVERED |
| Unit tests pass | File 8 | COVERED |
| No regressions | Verification gates | COVERED |

## Recommendation

**PASS WITH NOTES** -- The PLAN is ready for PATCH with the following guidance:

1. PATCH must ensure `LocalExecutionBackend.execute_all()` calls `config.model_dump()` before passing to `_process_single_file` (Issue #1).
2. The `hasattr` pattern for `execute_all()` is acceptable but should include a brief comment explaining why (Issue #2).
3. Do not fix the `ingestkit_excel.backends` import in `create_default_router()` -- that is a separate issue (Issue #3).

---

AGENT_RETURN: .agents/outputs/plan-check-46-021426.md
