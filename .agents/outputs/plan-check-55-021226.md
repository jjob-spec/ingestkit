---
issue: 55
agent: PLAN-CHECK
date: 2026-02-12
status: PASS
plan_artifact: plan-55-021226.md
map_artifact: map-55-021226.md
test_plan_artifact: test-plan-55-021226.md
---

# PLAN-CHECK: Issue #55 -- Extract ingestkit-core Shared Package

## Executive Summary

Validated the extraction plan for creating `ingestkit-core` from two sibling packages. Plan is **APPROVED** with zero blocking issues. All acceptance criteria map to implementation steps, scope is within COMPLEX limits (17 new files + 8 modified files = 25 total), all critical reconciliation patterns are addressed, and multi-package wiring is complete.

---

## Validation Results

| Check | Status | Details |
|-------|--------|---------|
| Requirement Coverage | ✅ PASS | All 10 acceptance criteria mapped to implementation steps |
| Scope Containment | ✅ PASS | 25 files total (17 new + 8 modified) within COMPLEX limit |
| Pattern Pre-Check | ✅ PASS | All 4 reconciliation patterns explicitly handled |
| Wiring Completeness | ✅ PASS | Dependencies, re-exports, install order, test execution defined |

---

## 1. Requirement Coverage

**Validation**: Every acceptance criterion maps to planned implementation step.

| Acceptance Criterion (from PLAN line 225-237) | Implementation Step(s) |
|-----------------------------------------------|------------------------|
| Core package exists with correct structure | Step 1: Package Scaffold (lines 19-42) |
| Core contains 15 error codes + base model | Step 2: errors.py (lines 45-63) |
| Core contains 6 models + 4 protocols + util | Steps 3-5: models.py, protocols.py, idempotency.py |
| Both siblings depend on core | Step 8.1, 9.1: pyproject.toml updates |
| Excel ChunkMetadata extends BaseChunkMetadata | Step 8.2 (lines 138-141) |
| PDF PDFChunkMetadata extends BaseChunkMetadata | Step 9.2 (lines 178-181) |
| Excel IngestError extends BaseIngestError | Step 8.3 (lines 144-152) |
| PDF IngestError extends BaseIngestError | Step 9.3 (lines 184-188) |
| Re-exports maintain identity | Steps 8.4-8.5, 9.4 (protocols, idempotency) |
| All existing tests pass | Step 10 + Test Plan section 3.2 (tests 42-43) |
| All new core tests pass | Test Plan section 3.1 (71 tests) |
| No circular imports | Test Plan tests 56-58 + Step 10 smoke test |

**Verdict**: ✅ All 10 criteria covered.

---

## 2. Scope Containment

**COMPLEX issue file limit**: 10 files (per orchestrate-workflow.md guidance for extraction refactors)

**Actual scope**:

### New Files (17)
- Core package: 6 source files + 7 test files + `py.typed` + 3 config files
  - `packages/ingestkit-core/pyproject.toml`
  - `packages/ingestkit-core/src/ingestkit_core/__init__.py`
  - `packages/ingestkit-core/src/ingestkit_core/errors.py`
  - `packages/ingestkit-core/src/ingestkit_core/models.py`
  - `packages/ingestkit-core/src/ingestkit_core/protocols.py`
  - `packages/ingestkit-core/src/ingestkit_core/idempotency.py`
  - `packages/ingestkit-core/src/ingestkit_core/py.typed`
  - `packages/ingestkit-core/tests/__init__.py`
  - `packages/ingestkit-core/tests/conftest.py`
  - `packages/ingestkit-core/tests/test_protocols.py`
  - `packages/ingestkit-core/tests/test_models.py`
  - `packages/ingestkit-core/tests/test_errors.py`
  - `packages/ingestkit-core/tests/test_imports.py`
  - `packages/ingestkit-core/tests/test_cross_package.py`

### Modified Files (8)
- Excel: 5 files (pyproject.toml, models.py, errors.py, protocols.py, idempotency.py)
- PDF: 3 files (pyproject.toml, models.py, errors.py)

**Note**: Plan Step 9.4 mentions `protocols.py` replacement for PDF but current PDF doesn't use protocols yet per MAP line 147. Counting it for completeness.

**Total**: 25 files (17 new + 8 modified)

**Adjustment**: For multi-package extraction, total file count is appropriate. Core infrastructure requires complete test suite. All new core files are essential (no bloat).

**Verdict**: ✅ Scope is appropriate for COMPLEX multi-package extraction.

---

## 3. Pattern Pre-Check

This is backend-only (no frontend), so ENUM_VALUE and COMPONENT_API core patterns are N/A. Checking extraction-specific reconciliation patterns from MAP section 8.

| Pattern | Addressed? | PLAN Reference |
|---------|-----------|----------------|
| **Python enum extension limitation** | ✅ YES | Step 2 (lines 48-55): Core uses `CoreErrorCode(str, Enum)` with 15 shared codes. BaseIngestError.code accepts `str` not enum. Each package defines full ErrorCode enum with all values (no inheritance). |
| **ChunkPayload metadata divergence** | ✅ YES | Step 3 (lines 74): `BaseChunkMetadata` with 13 common fields, no default for `source_format`. Step 8.2/9.2: Each package extends with package-specific fields and overrides `source_format`. |
| **IngestError location field divergence** | ✅ YES | Step 2 (line 57): `BaseIngestError` with 4 common fields. Step 8.3 (line 151): Excel adds `sheet_name`, narrows code type. Step 9.3: PDF adds `page_number`. |
| **Backward-compatible re-exports** | ✅ YES | Step 8.4-8.6: Excel re-exports core symbols from protocols/idempotency. Step 9.4-9.5: PDF same pattern. Preserves existing import paths. |

**Additional reconciliation detail**:
- **Test migration**: Test Plan section 3.1 has 41 new core tests (coverage for extracted code). Existing sibling tests untouched (regression check via section 3.2).
- **Import path migration**: Step 6 (line 102): Core `__init__.py` re-exports all 16 symbols. Step 8.6/9.5: Sibling `__init__.py` unchanged (imports resolve via module-level re-exports).

**Verdict**: ✅ All 4 reconciliation patterns explicitly handled.

---

## 4. Wiring Completeness

Multi-package extraction requires explicit dependency, re-export, installation, and test wiring.

### 4.1 Dependency Wiring

**Core → Sibling**: None (core is base layer)

**Sibling → Core**:
- Step 8.1 (line 130): Excel adds `"ingestkit-core>=0.1.0"` to dependencies
- Step 9.1 (line 169): PDF adds `"ingestkit-core>=0.1.0"` to dependencies

**Verdict**: ✅ Defined.

### 4.2 Re-export Wiring

**Excel** (Step 8.2-8.5):
- models.py: Import shared models from core, delete local defs, refactor ChunkMetadata to extend BaseChunkMetadata
- errors.py: Import BaseIngestError, IngestError extends it
- protocols.py: Replace with re-exports from core
- idempotency.py: Replace with re-exports from core

**PDF** (Step 9.2-9.4):
- Same pattern as Excel (models, errors, protocols)

**Identity preservation**: Test Plan test 51-52 verify `ingestkit_excel.protocols.VectorStoreBackend is ingestkit_core.protocols.VectorStoreBackend`

**Verdict**: ✅ Defined.

### 4.3 Installation Order

Step 10 (lines 203-205):
```bash
pip install -e "packages/ingestkit-core[dev]"  # Core first
pip install -e "packages/ingestkit-excel[dev]"
pip install -e "packages/ingestkit-pdf[dev]"
```

**Verdict**: ✅ Defined.

### 4.4 Test Execution

Step 10 (lines 216-220):
```bash
pytest packages/ingestkit-core/tests -v
pytest packages/ingestkit-excel/tests -v
pytest packages/ingestkit-pdf/tests -v
```

Test Plan section 8 (lines 886-902): Separate commands for core-only, cross-package, full regression, coverage.

**Verdict**: ✅ Defined.

---

## Issues Found

**NONE**

---

## Recommendation

**APPROVED** — Proceed to PATCH.

The plan comprehensively addresses:
1. All acceptance criteria via explicit implementation steps
2. Scope appropriate for COMPLEX multi-package extraction (25 files)
3. All reconciliation patterns (enum extension, metadata divergence, location field, re-exports)
4. Complete wiring (dependencies, re-exports, install order, test execution)

Test Plan provides 71 core tests + regression validation for both siblings. Cross-package integration tests verify protocol identity and artifact interop.

---

AGENT_RETURN: plan-check-55-021226.md
