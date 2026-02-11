---
issue: 4
agent: PLAN-CHECK
date: 2026-02-10
status: PASS
---

# PLAN-CHECK - Issue #4

## Status: PASS ✅

## Validation Results

| Check | Status | Notes |
|-------|--------|-------|
| Requirement Coverage | ✅ PASS | All 6 acceptance criteria from SPEC.md §6.3 mapped to planned tasks (AC1-AC5 in implementation, AC6 deferred to test issue) |
| Scope Containment | ✅ PASS | TRIVIAL classification accurate: 2 files identified, 1 file to create, 1 file already complete, 0 modifications to existing code |
| Pattern Pre-Check | ✅ N/A | No enums, no frontend components, no role/status fields — pattern checklist not applicable to backend utility function |
| Wiring Completeness | ✅ PASS | `compute_ingest_key` export must be added to `__init__.py` — plan mentions this is a new module but requires explicit export coordination |

## Detailed Validation

### Requirement Coverage
The MAP-PLAN correctly identifies and maps all acceptance criteria:

1. **AC1 — Identical files produce identical keys** → Implementation detail: function uses same file bytes → same hash
2. **AC2 — Modified files produce different keys** → Implementation detail: different bytes → different content_hash
3. **AC3 — Different parser_version produces different keys** → Implementation detail: version included in IngestKey.key computation
4. **AC4 — tenant_id changes key when present** → Implementation detail: IngestKey.key includes tenant_id in composite hash
5. **AC5 — IngestKey.key is a hex string** → Already verified in models.py lines 98-104; property returns `hexdigest()`
6. **AC6 — pytest tests/test_idempotency.py -q passes** → Correctly deferred to test implementation issue (not part of this implementation issue)

All criteria are verifiable via static code review or unit tests.

### Scope Containment
**TRIVIAL classification is accurate:**
- Files to create: 1 (`src/ingestkit_excel/idempotency.py`)
- Files to modify: 0 (IngestKey model already complete per models.py lines 85-104)
- Files to review: 1 (`models.py` for reference)
- **Total touched files: 2** ✅ Meets TRIVIAL threshold (≤3 files)

Implementation is pure — single function, no side effects beyond file I/O, no external API calls.

### Pattern Pre-Check
Not applicable to this issue:
- ✅ No enum values (no ENUM_VALUE pattern risk)
- ✅ No frontend components (no COMPONENT_API pattern risk)
- ✅ Backend utility function with simple string inputs (no complex polymorphism)

### Wiring Completeness — CRITICAL FINDING ⚠️

**Issue found:** Plan does not explicitly state that `compute_ingest_key` must be exported from `__init__.py`.

**Current state:**
- `__init__.py` lines 43-44 already imports and exports `IngestKey` model ✅
- `__init__.py` does NOT import `compute_ingest_key` function ❌

**Required action:**
After creating `src/ingestkit_excel/idempotency.py`, add export to `__init__.py`:
```python
from ingestkit_excel.idempotency import compute_ingest_key

# Add to __all__:
# "compute_ingest_key" (in appropriate section, e.g., after "IngestKey" or in new section)
```

**Why this matters:** Issue 4 is "Implement idempotency", but without export, downstream code cannot use `compute_ingest_key`. Per project patterns, public API functions must be exported from package `__init__.py`.

## Issues Found

### Issue #1: Missing Export Planning
**Severity:** MEDIUM
**Location:** MAP-PLAN Step 1, lines 65-90
**Problem:** Plan creates `idempotency.py` but does not explicitly mention adding `compute_ingest_key` to `__init__.py` exports
**Impact:** Function will be created but inaccessible from package public API without modification to `__init__.py`
**Fix:** Add explicit step to update `__init__.py` import and `__all__` list

**Proposed fix:**
- After Step 1 (create `idempotency.py`), add Step 1.5: Update `__init__.py`
  - Import: `from ingestkit_excel.idempotency import compute_ingest_key`
  - Export: Add `"compute_ingest_key"` to `__all__` list

## Recommendation

**Verdict:** PASS with minor clarification needed

**Action:** Approval to proceed with implementation with one additional step:

1. ✅ Create `src/ingestkit_excel/idempotency.py` with `compute_ingest_key()` function (per plan)
2. ✅ Verify `IngestKey.key` property in models.py (already complete)
3. ⚠️ **ADD:** Update `__init__.py` to export `compute_ingest_key`
   ```python
   # Line 8 (after ExcelProcessorConfig import):
   from ingestkit_excel.idempotency import compute_ingest_key

   # Update __all__ (after line 44, after "IngestKey"):
   "compute_ingest_key",
   ```
4. ✅ Verify implementation against acceptance criteria via static review
5. ✅ Defer unit tests to test implementation issue

**Risk level:** LOW
- Implementation is straightforward (pure function, stdlib only)
- IngestKey model already validated
- Acceptance criteria are binary/deterministic

---

**AGENT_RETURN:** plan-check-4-021026.md
