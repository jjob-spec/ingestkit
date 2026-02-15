---
issue: 68
agent: PLAN-CHECK
date: 2026-02-15
status: ISSUES_FOUND
plan_artifact: plan-68-021526.md
map_artifact: map-68-021526.md
spec_sections: "8.0-8.5, 13.2"
---

# PLAN-CHECK - Issue #68

## Status: ISSUES_FOUND

## Validation Results

| Check | Status | Notes |
|-------|--------|-------|
| Requirement Coverage | PASS | All 18 acceptance criteria map to implementation steps |
| Scope Containment | PASS | 7 files modified/created (COMPLEX limit: 10) |
| Pattern Pre-Check | PASS | Backend-only, no fullstack enum concerns |
| Wiring Completeness | PASS | `output/__init__.py` exports; `dual_writer` integrates `db_writer` + `chunk_writer`; conftest provides mocks |
| Spec Fidelity | ISSUES_FOUND | 3 issues identified below |

## Issues

### ISSUE-1 (MEDIUM): `ExtractedField` lacks `page_number` -- chunk splitting will fail

The plan's `split_fields_into_chunks` (Step 3a) groups fields by `field.page_number`, but `ExtractedField` in `models.py:271-304` has no `page_number` attribute. Only `FieldMapping` has `page_number` (line 111).

**Options for PATCH:**
- (a) Add `page_number: int` to `ExtractedField` model (clean, requires extractors to populate it)
- (b) Accept the `FormTemplate` in `split_fields_into_chunks` and look up page numbers by `field_id` from template fields
- (c) Default all fields to page 0 if page_number is missing (degraded but functional)

**Recommendation:** Option (a) is cleanest and consistent with spec intent. This means `models.py` becomes an additional modified file (8 files total, still within COMPLEX limit).

### ISSUE-2 (LOW): `evolve_schema` signature diverges from spec

Spec section 8.1 defines `evolve_table_schema(db, table_name, old_template, new_template)` taking both old and new templates. The plan's `FormDBWriter.evolve_schema(table_name, new_template)` queries existing columns from DB instead.

The plan's approach is actually **more robust** (DB is source of truth, not a possibly-stale old_template object), but PATCH should document this intentional deviation. Not a blocker.

### ISSUE-3 (LOW): `rollback_written_artifacts` signature adds `config` parameter

Spec section 8.5 defines `rollback_written_artifacts(written, vector_backend, db_backend)` without a config parameter. The plan adds `config: FormProcessorConfig` to access `backend_max_retries`. This is a reasonable extension (retries need config), but PATCH should note the deviation.

## Spec Verification Summary

| Spec Requirement | Plan Coverage |
|-----------------|---------------|
| DualWriteMode enum values (best_effort, strict_atomic) | Correct in models.py line 53-57 |
| FormWrittenArtifacts.db_row_ids field | Confirmed in models.py line 407 |
| Behavior matrix: 4 DB/Vector combinations x 2 modes | Step 4b covers all 4; tests cover all 8 |
| Schema evolution: ALTER TABLE ADD COLUMN, never drop | Step 2b `evolve_schema` |
| 10 metadata columns per spec 8.1 | Step 2a METADATA_COLUMNS dict, all 10 present |
| Chunk text format per spec 8.2 | Step 3a `serialize_form_to_text` |
| FormChunkMetadata 11 form-specific fields | Step 3a `build_chunk_metadata`, all 11 present |
| Rollback order: vector first, then DB | Step 4c explicitly ordered |
| Redaction: BOTH/CHUNKS_ONLY/DB_ONLY | Step 4a `redact_extraction` with destination check |
| Redaction deep copy, never mutate original | Step 4a uses `model_copy(deep=True)` |
| Retry: `backoff_base * 2**attempt` | Steps 2b, 3b, 4c retry loops |
| RollbackResult model fields | Confirmed in models.py lines 436-442 |
| FormDBBackend protocol (no overlap with StructuredDBBackend) | Step 1: 5 methods, all form-specific. No overlap except `table_exists` and `get_connection_uri` which are intentional mirrors |
| Test plan covers dual-write matrix | Step 7: 42 tests covering all combinations |

## Recommendation

**FIX ISSUE-1 before PATCH**, then PROCEED. ISSUE-1 will cause a runtime `AttributeError` in `split_fields_into_chunks`. The PATCH agent should either add `page_number` to `ExtractedField` or adjust the splitting function to accept the template for page lookup.

ISSUES 2-3 are intentional improvements over the spec signature and should be documented as deviations, not fixed.

---
AGENT_RETURN: .agents/outputs/plan-check-68-021526.md
