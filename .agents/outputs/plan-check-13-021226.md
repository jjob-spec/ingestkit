---
issue: 13
agent: PLAN-CHECK
date: 2026-02-14
complexity: SIMPLE
stack: backend
---

# PLAN-CHECK: Issue #13 — Test Infrastructure, Mock Backends, and Fixtures

**Plan artifact**: `.agents/outputs/map-plan-13-021226.md`
**Issue**: #13 — [Spec] Implement test infrastructure, mock backends, and fixtures (Backend)
**Date**: 2026-02-14

---

## Executive Summary

✅ **Plan is APPROVED for implementation.**

All checks passed:
- ✅ Requirement coverage complete (4 mock backends, 6 fixture files, markers, conftest fixtures)
- ✅ Scope properly contained (no modifications to existing test files or source code)
- ✅ Protocol signatures verified and match plan
- ✅ Wiring validated (existing conftest.py structure, pytest markers already registered)

**Minor corrections required**:
1. MockStructuredDB storage: Plan proposes `dict[str, pd.DataFrame]` but existing ad-hoc mocks use `list[tuple[str, pd.DataFrame]]`. Both approaches work; recommend sticking with `dict` as planned for better `drop_table()` support.
2. MockEmbedding zero vectors: Plan correctly specifies `[0.0]*dim` per SPEC.md requirement, diverging from existing ad-hoc `[0.1]*dim`. This is correct.

---

## Check 1: Requirement Coverage

### 1.1 Four Mock Backends

| Backend | Protocol | Plan Section | Signature Match | Status |
|---------|----------|--------------|-----------------|--------|
| `MockVectorStore` | `VectorStoreBackend` | 3.2.1 | ✅ | ✅ PASS |
| `MockStructuredDB` | `StructuredDBBackend` | 3.2.2 | ✅ | ✅ PASS |
| `MockLLM` | `LLMBackend` | 3.2.3 | ✅ | ✅ PASS |
| `MockEmbedding` | `EmbeddingBackend` | 3.2.4 | ✅ | ✅ PASS |

**Details**:

#### MockVectorStore
- **Protocol** (`protocols.py:19-36`):
  - `upsert_chunks(collection: str, chunks: list[ChunkPayload]) -> int`
  - `ensure_collection(collection: str, vector_size: int) -> None`
  - `create_payload_index(collection: str, field: str, field_type: str) -> None`
  - `delete_by_ids(collection: str, ids: list[str]) -> int`
- **Plan coverage**: All 4 methods specified (lines 93-100). ✅
- **Error simulation**: `fail_on_upsert: bool` parameter. ✅
- **Storage**: `self.chunks: list[ChunkPayload]` for tracking upserted chunks. ✅

#### MockStructuredDB
- **Protocol** (`protocols.py:40-61`):
  - `create_table_from_dataframe(table_name: str, df: pd.DataFrame) -> None`
  - `drop_table(table_name: str) -> None`
  - `table_exists(table_name: str) -> bool`
  - `get_table_schema(table_name: str) -> dict`
  - `get_connection_uri() -> str`
- **Plan coverage**: All 5 methods specified (lines 102-110). ✅
- **Error simulation**: `fail_on: str | None` parameter for table-specific failures. ✅
- **Storage**: Plan proposes `dict[str, pd.DataFrame]`. Existing ad-hoc mocks use `list[tuple[str, pd.DataFrame]]`. **Recommendation**: Use `dict` as planned — better supports `drop_table()` and `table_exists()` logic.

#### MockLLM
- **Protocol** (`protocols.py:65-86`):
  - `classify(prompt: str, model: str, temperature: float = 0.1, timeout: float | None = None) -> dict`
  - `generate(prompt: str, model: str, temperature: float = 0.7, timeout: float | None = None) -> str`
- **Plan coverage**: Both methods specified (lines 112-121). ✅
- **Error simulation**: Queue-based responses, supports dict | Exception. Can simulate:
  - Valid response: queue well-formed dict ✅
  - Malformed JSON: queue `json.JSONDecodeError` or invalid string ✅
  - Schema-invalid: queue dict missing required fields ✅
  - Timeout: queue `TimeoutError` ✅
- **Existing implementation**: `test_llm_classifier.py:36-81` already has `MockLLMBackend` with queue-based responses. Plan correctly proposes consolidation. ✅

#### MockEmbedding
- **Protocol** (`protocols.py:90-101`):
  - `embed(texts: list[str], timeout: float | None = None) -> list[list[float]]`
  - `dimension() -> int`
- **Plan coverage**: Both methods specified (lines 124-130). ✅
- **Error simulation**: `fail_on_embed: bool` parameter. ✅
- **Zero vectors**: Plan specifies `[0.0] * dim` per SPEC.md line 1075 requirement. Existing ad-hoc mocks return `[0.1] * dim`. **Plan is correct per spec.** ✅

### 1.2 Six Fixture .xlsx Files

| Fixture Name | File | Plan Section | Description Match | Status |
|--------------|------|--------------|-------------------|--------|
| `type_a_simple_xlsx` | `type_a_simple.xlsx` | 3.3 line 136 | 3 cols, 20 rows, tabular | ✅ PASS |
| `type_b_checklist_xlsx` | `type_b_checklist.xlsx` | 3.3 line 137 | Merged headers, checklist | ✅ PASS |
| `type_c_hybrid_xlsx` | `type_c_hybrid.xlsx` | 3.3 line 138 | Sheet 1 tabular, Sheet 2 doc | ✅ PASS |
| `edge_empty_xlsx` | `edge_empty.xlsx` | 3.3 line 139 | Empty workbook | ✅ PASS |
| `edge_chart_only_xlsx` | `edge_chart_only.xlsx` | 3.3 line 140 | No data rows (approximated) | ✅ PASS |
| `edge_large_xlsx` | `edge_large.xlsx` | 3.3 line 141 | Exceeds `max_rows_in_memory` | ✅ PASS |

**SPEC.md requirements** (lines 1079-1085): All 6 fixture files specified. ✅

**Generation approach**: Plan correctly proposes `openpyxl`-based generation in session-scoped fixtures using `tmp_path_factory`. ✅

### 1.3 pytest Markers

**Plan verification** (lines 46-47):
```toml
markers = [
    "unit: Unit tests (no external services)",
    "integration: Integration tests (require external services)",
]
```

**Actual `pyproject.toml`** (lines 32-35):
```toml
markers = [
    "unit: Unit tests (no external services)",
    "integration: Integration tests (require external services)",
]
```

✅ **Markers already registered.** No changes required.

### 1.4 conftest.py Fixtures

**Current state** (`conftest.py:11-25`):
- `sample_config()` — returns `ExcelProcessorConfig()` with defaults
- `sample_ingest_key()` — returns hardcoded `IngestKey` instance

**Plan additions** (lines 149-160):
- `test_config()` fixture with test-friendly defaults (fast timeouts, small limits) ✅
- Mock backend fixtures (4 fixtures for the 4 mock classes) ✅
- Fixture .xlsx file generators (6 session-scoped fixtures) ✅

**Total new fixtures**: 11 (1 config + 4 mock backends + 6 .xlsx files)

✅ **Coverage complete.**

---

## Check 2: Scope Containment

### 2.1 File Changes

**Plan section 3.1** (lines 83-89):

| File | Action | In Scope? |
|------|--------|-----------|
| `tests/conftest.py` | EDIT | ✅ Yes |
| `tests/fixtures/` | CREATE DIR | ✅ Yes (may be empty if all generated in conftest) |

**No modifications to**:
- ❌ Existing test files (`test_structured_db.py`, `test_llm_classifier.py`, etc.)
- ❌ Source code modules (`processors/`, `llm_classifier.py`, etc.)
- ❌ `pyproject.toml` (markers already registered)

✅ **Plan correctly states** (line 89): "Existing test files are NOT modified. The new conftest fixtures are additive and available for future tests."

✅ **Scope properly contained.**

### 2.2 Helper Factory Functions

**Plan decision** (lines 162-164):
> Do NOT add `_make_sheet_profile()` etc. to conftest.py. These are file-specific test helpers with different defaults per test module (tabular defaults vs. formatted-document defaults vs. hybrid defaults). Moving them would break the per-file customization and couple unrelated test files. Leave them as-is in each test file.

✅ **Correct decision.** Verified in codebase:
- `test_structured_db.py:47-68` — `_make_sheet_profile()` with tabular defaults
- `test_llm_classifier.py:88-108` — `_make_sheet_profile()` with different sample_rows
- `test_serializer.py` (similar pattern)
- `test_splitter.py` (similar pattern)

Moving these would break encapsulation. Plan correctly avoids this. ✅

---

## Check 3: Pattern Pre-checks

### 3.1 Protocol Signature Verification

**VectorStoreBackend** (`protocols.py:19-36`):
```python
def upsert_chunks(self, collection: str, chunks: list[ChunkPayload]) -> int: ...
def ensure_collection(self, collection: str, vector_size: int) -> None: ...
def create_payload_index(self, collection: str, field: str, field_type: str) -> None: ...
def delete_by_ids(self, collection: str, ids: list[str]) -> int: ...
```

**MockVectorStore plan** (lines 93-100):
- ✅ `upsert_chunks(collection: str, chunks: list[ChunkPayload]) -> int`
- ✅ `ensure_collection(collection: str, vector_size: int) -> None`
- ✅ `create_payload_index(collection: str, field: str, field_type: str) -> None`
- ✅ `delete_by_ids(collection: str, ids: list[str]) -> int`

**Match**: ✅ PASS

---

**StructuredDBBackend** (`protocols.py:40-61`):
```python
def create_table_from_dataframe(self, table_name: str, df: pd.DataFrame) -> None: ...
def drop_table(self, table_name: str) -> None: ...
def table_exists(self, table_name: str) -> bool: ...
def get_table_schema(self, table_name: str) -> dict: ...
def get_connection_uri(self) -> str: ...
```

**MockStructuredDB plan** (lines 102-110):
- ✅ `create_table_from_dataframe(table_name: str, df: pd.DataFrame) -> None`
- ✅ `drop_table(table_name: str) -> None`
- ✅ `table_exists(table_name: str) -> bool`
- ✅ `get_table_schema(table_name: str) -> dict`
- ✅ `get_connection_uri() -> str`

**Match**: ✅ PASS

---

**LLMBackend** (`protocols.py:65-86`):
```python
def classify(self, prompt: str, model: str, temperature: float = 0.1, timeout: float | None = None) -> dict: ...
def generate(self, prompt: str, model: str, temperature: float = 0.7, timeout: float | None = None) -> str: ...
```

**MockLLM plan** (lines 112-121):
- ✅ `classify(prompt: str, model: str, temperature: float = 0.1, timeout: float | None = None) -> dict`
- ✅ `generate(prompt: str, model: str, temperature: float = 0.7, timeout: float | None = None) -> str`

**Match**: ✅ PASS

---

**EmbeddingBackend** (`protocols.py:90-101`):
```python
def embed(self, texts: list[str], timeout: float | None = None) -> list[list[float]]: ...
def dimension(self) -> int: ...
```

**MockEmbedding plan** (lines 124-130):
- ✅ `embed(texts: list[str], timeout: float | None = None) -> list[list[float]]`
- ✅ `dimension() -> int`

**Match**: ✅ PASS

---

### 3.2 ChunkPayload Model

**Plan references** `ChunkPayload` in MockVectorStore (line 94). Verified in `ingestkit_core/models.py:101-107`:

```python
class ChunkPayload(BaseModel):
    id: str
    text: str
    vector: list[float]
    metadata: BaseChunkMetadata
```

✅ **Model exists and is imported from `ingestkit_core`.**

---

## Check 4: Wiring

### 4.1 Existing conftest.py Structure

**Current** (`conftest.py:1-26`):
- Line 1: Docstring
- Line 5: `import pytest`
- Line 7-8: Imports from `ingestkit_excel` (config, models)
- Lines 11-14: `sample_config` fixture
- Lines 17-26: `sample_ingest_key` fixture

**Plan additions**:
- Import `openpyxl`, `pandas`, `pathlib`, `typing.Protocol`
- Import all 4 Protocol types from `ingestkit_core.protocols`
- Import `ChunkPayload` from `ingestkit_core.models`
- Add 4 mock backend classes (~155 LOC)
- Add 4 pytest fixtures for mock backends (~20 LOC)
- Add 1 `test_config` fixture (~10 LOC)
- Add 6 session-scoped .xlsx generator fixtures (~80 LOC)

**Total estimated additions**: ~265 LOC

✅ **Wiring is straightforward.** No conflicts with existing fixtures.

### 4.2 Fixture Name Collision

**Plan addresses this** (lines 217-225):
> Since existing test files (test_structured_db.py, test_serializer.py, test_splitter.py) define their own `mock_vector_store`, `mock_embedder`, `mock_db` fixtures locally, pytest will use the local fixtures (closest scope wins). The conftest.py fixtures will only be used by tests that don't define their own. This is the correct pytest behavior and means no conflicts.

✅ **Correct analysis.** Local fixtures override conftest fixtures by pytest design. No risk of breaking existing tests.

### 4.3 pytest Markers Registration

**Already registered** in `pyproject.toml:32-35`. No changes needed. ✅

---

## Issues Found

None. Plan is ready for implementation.

---

## Recommendations

1. **Proceed with implementation** as planned.
2. **Use `dict[str, pd.DataFrame]` for MockStructuredDB storage** as proposed (better than `list[tuple]` for `drop_table()` support).
3. **Use zero vectors `[0.0]*dim`** for MockEmbedding per SPEC.md requirement (not `[0.1]*dim` like existing ad-hoc mocks).
4. **Session-scoped .xlsx fixtures** are correct for performance — files are read-only and can be shared across all tests.
5. **Test file `test_conftest.py`** (planned in section 4, task 8) should verify:
   - All 4 mocks pass `isinstance(mock, Protocol)` checks
   - MockLLM error simulation modes (valid, malformed JSON, schema-invalid, timeout)
   - All 6 .xlsx files are loadable with `openpyxl.load_workbook()`
   - Basic structural assertions (sheet count, row count ranges)

---

## Acceptance Criteria Status (from MAP-PLAN lines 169-178)

| # | Criterion | Status |
|---|---|--------|
| 1 | All mock backends satisfy their Protocol (`isinstance` check passes) | ✅ Plan specifies Protocol conformance |
| 2 | `MockLLM` can simulate: valid response, malformed JSON, schema-invalid, timeout | ✅ Plan lines 112-121 specify queue-based error simulation |
| 3 | All fixture .xlsx files created and loadable with openpyxl | ✅ Plan lines 136-145 specify 6 fixtures, openpyxl generation |
| 4 | Markers registered and usable | ✅ Already done in `pyproject.toml` |
| 5 | `pytest --co -q` shows discovered tests | ✅ No changes to test discovery (manual verification post-implementation) |
| 6 | Existing tests are NOT broken | ✅ Plan correctly avoids modifying existing test files |

---

## Summary

**Plan status**: ✅ **APPROVED**

**Requirement coverage**: ✅ Complete (4 backends, 6 fixtures, markers, conftest fixtures)
**Scope containment**: ✅ Proper (additive only, no modifications to existing tests/source)
**Protocol signatures**: ✅ Verified and match
**Wiring**: ✅ Validated (existing conftest structure, no fixture name collisions)

**Estimated implementation**: ~365 LOC (plan estimate line 195)

**Risk level**: LOW
- Additive changes only
- No breaking changes to existing tests
- Session-scoped fixtures improve test performance
- Local fixtures override conftest fixtures (pytest design prevents conflicts)

**Ready for PATCH phase**: ✅ YES

---

AGENT_RETURN: plan-check-13-021226.md
