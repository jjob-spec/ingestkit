---
issue: 9
title: "Implement Path B text serializer (Backend)"
agent: plan-check
timestamp: 2026-02-12
status: PASS
complexity: COMPLEX
branch: feature/issue-9-path-b-serializer
---

# PLAN-CHECK: Issue #9 -- Path B Text Serializer

## Executive Summary

Plan validation for Path B text serializer implementation. All acceptance criteria mapped to implementation steps. Scope within COMPLEX limits (4 files). Pattern correctly follows Path A processor structure. Wiring completeness verified.

**Result**: ✅ PASS — Plan is complete and ready for PATCH.

---

## 1. Requirement Coverage

### 1.1 Acceptance Criteria Mapping

| Acceptance Criterion (Issue) | Plan Coverage | Location |
|------------------------------|---------------|----------|
| Merged cells preserved/contextualized | ✅ Covered | Plan §1.5 lines 93-112 (section detection with merged cell map) |
| Checklist serialization format | ✅ Covered | Plan §1.7 lines 135-143 (_serialize_checklist) |
| Matrix serialization format | ✅ Covered | Plan §1.7 lines 144-148 (_serialize_matrix) |
| Free text with paragraphs | ✅ Covered | Plan §1.7 lines 149-153 (_serialize_free_text) |
| Section detection (blank/merged) | ✅ Covered | Plan §1.5 lines 93-112 (algorithm steps 1-6) |
| ChunkMetadata.original_structure | ✅ Covered | Plan §1.9 lines 164-180 (metadata construction) |
| WrittenArtifacts populated | ✅ Covered | Plan §1.4 line 69, line 86 (tracking vector_point_ids) |
| Works with mock backends | ✅ Covered | Plan §4.1 lines 207-209 (MockVectorStore, MockEmbedder) |
| pytest tests pass | ✅ Covered | Plan §4.2 lines 222-287 (44 tests across 9 classes) |

**Additional Spec Requirements** (SPEC.md §10.2):

| Spec Requirement | Plan Coverage | Location |
|------------------|---------------|----------|
| openpyxl parse preserving merged cells | ✅ Covered | Plan §1.4 line 71 (openpyxl.load_workbook), §1.5 lines 98-100 (merged cell map) |
| Logical section detection | ✅ Covered | Plan §1.5 lines 93-112 |
| Sub-structure classification | ✅ Covered | Plan §1.6 lines 114-125 |
| Embed via EmbeddingBackend | ✅ Covered | Plan §1.4 lines 81-86 |
| Upsert to VectorStoreBackend | ✅ Covered | Plan §1.4 line 86, line 69 (ensure_collection) |
| Standardized ChunkMetadata | ✅ Covered | Plan §1.9 lines 164-180 |

**Coverage**: 9/9 acceptance criteria + 6/6 spec requirements = **15/15 (100%)**

### 1.2 Process Signature Reconciliation

**Issue Spec Says** (simplified):
```python
def process(file_path, profile, ingest_key, ingest_run_id) -> ProcessingResult
```

**Plan Uses** (reconciled to match Path A):
```python
def process(file_path, profile, ingest_key, ingest_run_id,
            parse_result, classification_result, classification) -> ProcessingResult
```

**Validation**: ✅ CORRECT. MAP document (line 120-140) explicitly documents this reconciliation. The extra 3 parameters are required because `ProcessingResult` model demands them. Plan follows actual Path A implementation (not the simplified spec signature). This is intentional and documented in MAP section 4.2 and Plan section 1.4 line 56.

---

## 2. Scope Containment

### 2.1 File Count

| File | Action | Purpose |
|------|--------|---------|
| `processors/serializer.py` | CREATE | Core TextSerializer class |
| `tests/test_serializer.py` | CREATE | 44 unit tests across 9 test classes |
| `processors/__init__.py` | MODIFY | Export TextSerializer |
| `__init__.py` | MODIFY | Add TextSerializer to package exports |

**Total**: 4 files (2 create, 2 modify)

**COMPLEX Limit**: 10 files

**Status**: ✅ WITHIN LIMITS (4/10 files = 40% utilization)

### 2.2 Test Coverage

**Planned Tests**: 44 unit tests organized into 9 test classes

**Test Classes**:
1. TestSectionDetection (6 tests)
2. TestSubStructureClassification (5 tests)
3. TestSerializationFormats (5 tests)
4. TestMergedCellHandling (2 tests)
5. TestChunkMetadata (7 tests)
6. TestProcessFlow (7 tests)
7. TestMultiSheet (2 tests)
8. TestSheetSkipping (4 tests)
9. TestErrorHandling (5 tests)
10. TestEmbeddingBatching (1 test)

**Coverage**: Section detection, all 4 serialization formats (table/checklist/matrix/free_text), metadata correctness, error handling, multi-sheet, sheet skipping logic.

**Status**: ✅ COMPREHENSIVE (covers all acceptance criteria)

---

## 3. Pattern Pre-Check

### 3.1 Backend-Only Validation

**Stack**: Backend only (processors module)

**Protocol Usage**:
- ✅ `VectorStoreBackend` (yes — required for chunk upsert)
- ✅ `EmbeddingBackend` (yes — required for embedding)
- ❌ `StructuredDBBackend` (no — Path B does not write to structured DB, correctly omitted)
- ❌ `LLMBackend` (no — Path B does not use LLM, correctly omitted)

**Status**: ✅ CORRECT — Path B uses only vector store + embedder, no DB backend.

### 3.2 Path A Processor Pattern Match

Comparing plan to actual Path A implementation (`structured_db.py`):

| Path A Pattern | Path B Plan | Status |
|----------------|-------------|--------|
| Constructor takes backends + config | Plan §1.3 line 51 | ✅ Match |
| process() 8-param signature | Plan §1.4 line 56-61 | ✅ Match |
| start_time, source_uri, errors, warnings initialization | Plan §1.4 lines 66-68 | ✅ Match |
| WrittenArtifacts tracking | Plan §1.4 line 69 | ✅ Match |
| ensure_collection call | Plan §1.4 line 70 | ✅ Match |
| chunk_index_counter global across sheets | Plan §1.4 line 72 | ✅ Match |
| Per-sheet skip logic (hidden/chart-only/oversized) | Plan §1.4 lines 73-77 | ✅ Match |
| Per-sheet try/except with continue | Plan §1.4 line 78, line 85 | ✅ Match |
| _classify_backend_error() static method | Plan §1.8 lines 154-160 | ✅ Match |
| Deterministic UUID5 chunk IDs | Plan §1.9 line 180 | ✅ Match |
| Embedding batch pattern | Plan §1.4 lines 81-86 | ✅ Match |
| EmbedStageResult if texts_embedded > 0 | Plan §1.4 line 87 | ✅ Match |
| ProcessingResult assembly with all fields | Plan §1.4 lines 88-92 | ✅ Match |

**Status**: ✅ EXCELLENT — 13/13 patterns match Path A structure exactly.

### 3.3 Path B-Specific Correctness

| Path B Requirement | Plan Implementation | Status |
|--------------------|---------------------|--------|
| ingestion_method = TEXT_SERIALIZATION | Plan §1.4 line 89 | ✅ Enum member used (not string) |
| tables_created = 0 | Plan §1.4 line 90 | ✅ Correct |
| tables = [] | Plan §1.4 line 90 | ✅ Correct |
| written.db_table_names = [] | Plan §1.4 line 91 | ✅ Correct |
| ChunkMetadata.section_title | Plan §1.9 line 175 | ✅ Set from section.title |
| ChunkMetadata.original_structure | Plan §1.9 line 176 | ✅ Set from section.sub_structure |
| ChunkMetadata.table_name = None | Plan §1.9 line 174 | ✅ Correct (no DB table) |
| Error code: E_PROCESS_SERIALIZE | Plan §1.8 line 159 | ✅ Correct (not E_PROCESS_SCHEMA_GEN) |
| openpyxl direct usage (not pandas) | Plan §1.4 line 71 | ✅ Correct (merged cell preservation) |

**Status**: ✅ EXCELLENT — All 9 Path B-specific requirements correctly implemented.

### 3.4 Structural Subtyping (Protocol Usage)

Plan correctly uses `Protocol` types from `protocols.py`:
- ✅ Constructor params: `vector_store: VectorStoreBackend`, `embedder: EmbeddingBackend` (Plan §1.3)
- ✅ No concrete backend references
- ✅ No ABC usage (structural subtyping via Protocol)

**Status**: ✅ CORRECT — Follows project architecture constraints.

---

## 4. Wiring Completeness

### 4.1 Processor Export

**Current State** (`processors/__init__.py`):
```python
from ingestkit_excel.processors.structured_db import StructuredDBProcessor
__all__ = ["StructuredDBProcessor"]
```

**Planned Change** (Plan §2 lines 182-193):
```python
from ingestkit_excel.processors.serializer import TextSerializer
from ingestkit_excel.processors.structured_db import StructuredDBProcessor
__all__ = ["StructuredDBProcessor", "TextSerializer"]
```

**Status**: ✅ COVERED

### 4.2 Package Export

**Current State** (`__init__.py` line 33, 71):
```python
from ingestkit_excel.processors import StructuredDBProcessor
...
__all__ = [..., "StructuredDBProcessor", ...]
```

**Planned Change** (Plan §3 lines 195-201):
- Import: Add `from ingestkit_excel.processors import TextSerializer` after line 33
- __all__: Add `"TextSerializer"` to __all__ after `"StructuredDBProcessor"` (line 71)

**Status**: ✅ COVERED

### 4.3 Test File Creation

**Planned**: `tests/test_serializer.py` with 44 tests across 9 classes

**Test Infrastructure**:
- ✅ MockVectorStore, MockEmbedder (Plan §4.1)
- ✅ Factory functions for profiles/results (Plan §4.1)
- ✅ openpyxl mock pattern with mock workbook builder (Plan §4.1 lines 217-220)

**Status**: ✅ COVERED

---

## 5. Risk Assessment

### 5.1 Known Complexity Points

| Risk | Mitigation in Plan | Status |
|------|-------------------|--------|
| Merged cell parsing | Explicit algorithm in §1.5 lines 98-100 (build merged_headers dict from ws.merged_cells.ranges) | ✅ Addressed |
| Section detection ambiguity | Fail-closed heuristic in §1.5 line 112 (edge case: no boundaries → single section) | ✅ Addressed |
| Sub-structure classification | Explicit priority order + default to "free_text" in §1.6 lines 114-125 | ✅ Addressed |
| openpyxl mock complexity | Helper function `_make_mock_workbook()` in §4.1 lines 217-220 | ✅ Addressed |

### 5.2 Deviations from Spec

**Deviation 1**: Process signature has 8 params instead of 4.

**Justification**: Documented in MAP (section 2.2) and Plan (section 1.4). Required by `ProcessingResult` model. Intentional reconciliation to match Path A actual implementation.

**Status**: ✅ ACCEPTABLE (documented and justified)

---

## 6. Verification Gates

Plan includes verification commands (Plan lines 309-315):

```bash
pytest packages/ingestkit-excel/tests -v
pytest packages/ingestkit-excel/tests/test_serializer.py -v
pytest packages/ingestkit-excel/tests/test_structured_db.py -v  # Regression check
python -c "from ingestkit_excel import TextSerializer; print('OK')"
pytest packages/ingestkit-excel/tests/test_serializer.py --cov=ingestkit_excel.processors.serializer --cov-report=term-missing
```

**Status**: ✅ COMPLETE (new tests + regression + import + coverage)

---

## Final Validation

| Check | Result | Notes |
|-------|--------|-------|
| Requirement coverage | ✅ PASS | 15/15 criteria mapped (100%) |
| Scope containment | ✅ PASS | 4/10 files (40% utilization) |
| Pattern adherence | ✅ PASS | 13/13 Path A patterns match |
| Backend-only correctness | ✅ PASS | Only vector store + embedder (no DB) |
| Wiring completeness | ✅ PASS | Export to processors/__init__.py + __init__.py |
| Test coverage | ✅ PASS | 44 tests across 9 classes |
| Verification gates | ✅ PASS | 5 verification commands defined |

---

## Recommendation

**Status**: ✅ APPROVED FOR PATCH

Plan is complete, well-structured, and ready for implementation. All acceptance criteria covered, scope within limits, pattern correctly follows Path A processor structure, wiring fully specified.

**Key Strengths**:
1. Explicit reconciliation of process() signature mismatch (documented in both MAP and PLAN)
2. Comprehensive section detection algorithm with merged cell handling
3. All 4 serialization formats (table/checklist/matrix/free_text) specified
4. 44 unit tests covering all edge cases
5. Path B-specific correctness (no DB backend, correct error codes, correct metadata fields)
6. Deterministic chunk ID generation matching Path A pattern

**No blockers identified.** Proceed to PATCH phase.

---

AGENT_RETURN: plan-check-9-021226.md
