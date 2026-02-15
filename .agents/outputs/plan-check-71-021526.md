---
issue: 71
agent: PLAN-CHECK
date: 2026-02-15
status: PASS
plan_artifact: map-plan-71-021526.md
---

# PLAN-CHECK -- Issue #71: Implement Idempotency Keying for Form Extraction

## Executive Summary

The MAP-PLAN for issue #71 is well-structured and ready for PATCH. All eight acceptance criteria map to planned tasks. The file count (3 implementation + 1 test) is appropriate for SIMPLE classification. Key formulas match spec section 4.3. One minor concern: the separator character in `compute_form_extraction_key` should use `"|"` consistently with `IngestKey.key` (the plan shows this correctly). The decision to defer chunk_writer/db_writer integration is sound and correctly scoped.

## Check 1: Requirement Coverage

| Acceptance Criterion | Planned Task | Status |
|---|---|---|
| `compute_ingest_key` delegates to core | File 1, function 1 -- re-export from `ingestkit_core` | COVERED |
| `compute_form_extraction_key` = `sha256(global + template_id + version)` | File 1, function 2 | COVERED |
| `compute_vector_point_id` = `uuid5(NAMESPACE, extraction_key + index)` | File 1, function 3 | COVERED |
| Same inputs -> same outputs (idempotency) | File 3, tests in all 4 classes | COVERED |
| Different template version -> different extraction key, same global key | File 3, `TestEndToEndIdempotency` | COVERED |
| All public functions exported from `__init__` | File 2 -- adds imports and `__all__` entries | COVERED |
| Tests pass: `test_idempotency.py` | File 3, 19 test methods | COVERED |
| No regressions | File 3 -- no existing code modified (stub only) | COVERED |

Result: ALL COVERED

## Check 2: Scope Containment

| Metric | Expected (SIMPLE) | Planned | Status |
|---|---|---|---|
| Files modified | 1-3 | 2 (`idempotency.py`, `__init__.py`) | OK |
| Files created | 0-1 | 1 (`test_idempotency.py`) | OK |
| Total files touched | 1-3 | 3 | OK |
| Deferred integration | chunk_writer, db_writer | Explicitly deferred | OK |

Result: WITHIN SCOPE

## Check 3: Pattern Consistency with ingestkit-excel

| Aspect | ingestkit-excel | Plan for ingestkit-forms | Match? |
|---|---|---|---|
| Re-exports `compute_ingest_key` from core | Yes (line 7) | Yes, function 1 | YES |
| Re-exports `IngestKey` from core | Yes (line 8) | Yes, in exports | YES |
| `__all__` list | Yes (lines 10-13) | Yes, planned | YES |
| Test file naming | `test_idempotency.py` | `test_idempotency.py` | YES |
| New form-specific functions | N/A | `compute_form_extraction_key`, `compute_vector_point_id` | N/A (form-specific) |

Result: CONSISTENT

## Check 4: Key Formulas vs Spec Section 4.3

**Spec (lines 226-228):**
```
ingest_key_global = sha256(content_hash + source_uri + parser_version)
```
**Plan:** Delegates to `ingestkit_core.idempotency.compute_ingest_key` which produces `IngestKey` with `.key` = `sha256("|".join([content_hash, source_uri, parser_version, ?tenant_id]))`. The `+` in spec means concatenation; the core implementation uses `|` separator. This is the established pattern used by ingestkit-excel. MATCH.

**Spec (lines 233-234):**
```
form_extraction_key = sha256(ingest_key_global + template_id + template_version)
```
**Plan (line 82):**
```python
sha256(f"{ingest_key_global}|{template_id}|{template_version}").hexdigest()
```
Uses `|` separator consistent with `IngestKey.key` pattern. Returns 64-char hex. MATCH.

**Spec (line 244):**
```
uuid5(NAMESPACE, form_extraction_key + chunk_index)
```
**Plan (line 87-88):**
```python
str(uuid5(NAMESPACE_URL, f"{form_extraction_key}:{chunk_index}"))
```
Uses `NAMESPACE_URL` matching existing `chunk_writer.py:226`. Uses `:` separator matching existing chunk_writer pattern. MATCH.

Result: ALL FORMULAS MATCH

## Check 5: uuid5 Namespace and Format

- **Namespace:** `uuid.NAMESPACE_URL` -- consistent with existing `chunk_writer.py:226`.
- **Input format:** `f"{form_extraction_key}:{chunk_index}"` -- deterministic string concatenation.
- **Output format:** UUID string via `str(uuid.uuid5(...))` -- standard UUID format.
- **Determinism:** `uuid5` is inherently deterministic given same namespace + name. Correct.

Result: CORRECT

## Warnings (Non-Blocking)

1. **Stub docstring references wrong section:** The current stub says "spec section 4.2" but should be "4.3". Plan does not explicitly mention fixing this, but replacing the stub will naturally fix it.
2. **`IngestKey.key` includes optional `tenant_id`:** The spec formula `sha256(content_hash + source_uri + parser_version)` does not mention `tenant_id`, but the core `IngestKey.key` appends it when present. This is an existing design choice from ingestkit-core, not introduced by this plan. No action needed.

## Verdict

**PASS** -- Plan is complete, correctly scoped, pattern-consistent, and spec-compliant. Ready for PATCH.
