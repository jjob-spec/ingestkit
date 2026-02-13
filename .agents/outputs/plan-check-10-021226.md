---
issue: 10
agent: PLAN-CHECK
date: 2026-02-12
complexity: COMPLEX
stack: backend
---

# PLAN-CHECK -- Issue #10

## Requirement Coverage: PASS

| Requirement (Issue) | Plan Section | Status |
|---------------------|-------------|--------|
| Blank separators >= 2 rows/cols | `_detect_blank_row_boundaries`, `_detect_blank_col_boundaries`, constants `_BLANK_ROW_THRESHOLD=2`, `_BLANK_COL_THRESHOLD=2` | Covered |
| Merged cell blocks -> header/title | `_detect_merged_blocks(ws)` returns `SheetRegion` with `HEADER_BLOCK` | Covered |
| Formatting transitions (numeric/text shift) | `_detect_formatting_transitions` with sliding window, `_FORMATTING_TRANSITION_WINDOW=5` | Covered |
| Header/footer detection -> HEADER_BLOCK/FOOTER_BLOCK | `_detect_header_footer(ws, all_rows, total_rows)` returns tuple of optional regions | Covered |
| Matrix detection -> MATRIX_BLOCK | `_detect_matrix_regions` checks row/col headers, intersection population | Covered |
| SheetRegion with bounding coords + detection_confidence | Plan specifies per-heuristic confidence (0.7-0.9), region_id generation, bounding box coords | Covered |
| Classify each region as Type A/B using Tier 1 signals | `_classify_region` with 5 signals, threshold logic matching ExcelInspector | Covered |
| Route to StructuredDBProcessor or TextSerializer | `_process_type_a_region` and `_process_type_b_region` in process() flow | Covered |
| All chunks share ingest_key, linked via region_id | Plan sets region_id in ChunkMetadata, shared ingest_key from process() param | Covered |
| Constructor: HybridSplitter(structured_processor, text_serializer, config) | Plan constructor matches exactly | Covered |
| process() -> ProcessingResult | Plan returns ProcessingResult with all required fields | Covered |
| Merge WrittenArtifacts | Plan accumulates vector_point_ids and db_table_names across all regions | Covered |

**Notes:**
- The issue specifies `process(file_path, profile, classification, ingest_key, ingest_run_id)` (5 params), but the plan uses the 7-param signature matching Path A/B. This is the correct approach -- the issue description is a simplified summary, and the plan correctly matches the actual Path A/B signatures for router compatibility. PASS.

## Scope Containment: PASS

| Planned File | Action | Authorized |
|-------------|--------|------------|
| `processors/splitter.py` | Create | Yes -- issue requests this |
| `tests/test_splitter.py` | Create | Yes -- issue requests test file (issue says `test_processors.py` but plan uses `test_splitter.py` matching project convention of one test file per module) |
| `processors/__init__.py` | Modify (add import) | Yes -- required for export |
| `__init__.py` | Modify (add import + __all__) | Yes -- required for package export |

- No unauthorized files modified.
- No ROADMAP items included.
- Test file naming deviation (`test_splitter.py` vs issue's `test_processors.py`) follows the project convention in CLAUDE.md: "One test file per module (`test_<module>.py`)". This is acceptable.

## Pattern Pre-checks: PASS

### process() Signature Match

**Path A** (`structured_db.py:105-114`):
```python
def process(self, file_path: str, profile: FileProfile, ingest_key: str,
            ingest_run_id: str, parse_result: ParseStageResult,
            classification_result: ClassificationStageResult,
            classification: ClassificationResult) -> ProcessingResult
```

**Path B** (`serializer.py:86-95`):
```python
def process(self, file_path: str, profile: FileProfile, ingest_key: str,
            ingest_run_id: str, parse_result: ParseStageResult,
            classification_result: ClassificationStageResult,
            classification: ClassificationResult) -> ProcessingResult
```

**Plan** (`plan-10-021226.md` line 130-139):
```python
def process(self, file_path: str, profile: FileProfile, ingest_key: str,
            ingest_run_id: str, parse_result: ParseStageResult,
            classification_result: ClassificationStageResult,
            classification: ClassificationResult) -> ProcessingResult
```

All three signatures match exactly (7 params + self). PASS.

### Model Existence

| Model/Enum | Location | Exists |
|-----------|----------|--------|
| `SheetRegion` | `models.py:176-188` | Yes -- fields: sheet_name, region_id, start_row, end_row, start_col, end_col, region_type, detection_confidence, classified_as |
| `RegionType` | `models.py:59-68` | Yes -- values: DATA_TABLE, TEXT_BLOCK, HEADER_BLOCK, FOOTER_BLOCK, MATRIX_BLOCK, CHART_ONLY, EMPTY |
| `ErrorCode.E_PROCESS_REGION_DETECT` | `errors.py:42` | Yes |
| `IngestionMethod.HYBRID_SPLIT` | `models.py:56` | Yes -- value: "hybrid_split" |

All required models/enums exist. PASS.

## Wiring: PASS

### processors/__init__.py Current State
```python
from ingestkit_excel.processors.serializer import TextSerializer
from ingestkit_excel.processors.structured_db import StructuredDBProcessor
__all__ = ["StructuredDBProcessor", "TextSerializer"]
```

Currently exports StructuredDBProcessor and TextSerializer. Plan adds HybridSplitter import and export. Correct.

### Package __init__.py Current State
- Line 33: `from ingestkit_excel.processors import StructuredDBProcessor, TextSerializer`
- Line 71: `"StructuredDBProcessor"` in __all__
- Line 72: `"TextSerializer"` in __all__

Plan adds `HybridSplitter` to both the import line and __all__. Correct.

## Architecture: PASS (with caveats)

### Delegation Approach

The plan went through extensive deliberation (lines 17-67) before settling on: **HybridSplitter performs its own processing loop rather than delegating to sub-processors' `process()` methods.**

**Validation**: This decision is correct. Reading the actual source code:

- `StructuredDBProcessor` (`structured_db.py:232`): Sets `region_id=None` in ChunkMetadata.
- `TextSerializer` (`serializer.py:175`): Sets `region_id=None` in ChunkMetadata.

Since the spec requires `region_id` to be set in ChunkMetadata for every Path C chunk, and sub-processors hardcode `region_id=None`, delegation to `process()` would produce chunks without region_id. The only way to set region_id correctly is to handle the processing loop internally. The plan's decision is architecturally sound.

### Backend Extraction via Private Attributes

The plan extracts backends from sub-processors:
```python
self._db = structured_processor._db
self._vector_store = structured_processor._vector_store
self._embedder = structured_processor._embedder
```

**Validation**: These attributes exist in the actual constructors:

| Attribute | Source | Line |
|-----------|--------|------|
| `StructuredDBProcessor._db` | `structured_db.py:96` | `self._db = structured_db` |
| `StructuredDBProcessor._vector_store` | `structured_db.py:97` | `self._vector_store = vector_store` |
| `StructuredDBProcessor._embedder` | `structured_db.py:98` | `self._embedder = embedder` |

All three private attributes exist on StructuredDBProcessor. PASS.

**Caveat**: Accessing private attributes (`_db`, `_vector_store`, `_embedder`) is a code smell but acceptable here because:
1. The spec mandates the constructor takes `structured_processor` and `text_serializer` (not raw backends).
2. The only alternative is duplicating the constructor to also accept raw backends, which contradicts the spec.
3. All three classes are in the same package and tightly coupled by design.

**Recommendation for PATCH**: Add a brief comment explaining why private attributes are accessed, e.g., `# Access backends directly to set region_id on chunks (spec requirement)`.

### Additional Observation

The plan's `process()` pseudocode (line 394) recomputes `all_rows` inside the region loop: `all_rows = [[cell.value for cell in row] for row in ws.iter_rows()]`. This should be computed once per sheet, outside the region loop, to avoid redundant iteration. The PATCH agent should optimize this.

## Overall: APPROVED

All five checks pass. The plan is well-structured, covers all issue requirements, stays within scope, matches existing signatures and models, and makes a sound architectural decision on the delegation approach.

**Minor recommendations for PATCH**:
1. Move `all_rows` computation outside the region loop (optimization).
2. Add comment explaining private attribute access on sub-processors.
3. The plan's deliberation section (lines 17-67) is verbose but does not affect implementation correctness.
4. TextSerializer has `_vector_store` and `_embedder` but no `_db` -- the plan only extracts from `structured_processor`, which is correct since TextSerializer doesn't use a DB backend.

AGENT_RETURN: plan-check-10-021226.md
