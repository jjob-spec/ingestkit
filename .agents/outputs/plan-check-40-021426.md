---
issue: 40
agent: PLAN-CHECK
date: 2026-02-14
status: PASS
---

# PLAN-CHECK: Issue #40 -- Table Extractor

## Validation Checklist

### 1. Dual Routing Logic (SPEC 11.3 Step 2)

SPEC states:
- `<= table_max_rows_for_serialization` rows: serialize as NL sentences, embed as chunks.
- `> table_min_rows_for_db` rows: load into StructuredDB, embed schema description.

PLAN section 1f (`_route_table()`) implements:
- `len(df) <= config.table_max_rows_for_serialization` (default 20) -> NL serialization path
- `len(df) > config.table_min_rows_for_db` (default 20) -> StructuredDB path

**Verified:** Config defaults are both 20 (confirmed in `config.py` lines 86-87). Boundary is clean: exactly 20 rows -> NL, 21+ rows -> DB. PLAN correctly notes this.

**Verified:** NL serialization format `"In table '{name}', row {N}: {col} is {val}, ..."` matches Excel `structured_db.py` pattern per SPEC expectation.

**Verified:** StructuredDB path calls `create_table_from_dataframe()` and embeds a schema description -- matches SPEC step 2 exactly.

**Verified:** Both paths tag with `content_type="table"`, `table_index`, `page_numbers` per SPEC.

### 2. Multi-Page Table Stitching (SPEC 11.3 Step 3)

SPEC states:
- Compare column count and header text between last table on page N and first table on page N+1.
- If column count matches AND header similarity >= `table_continuation_column_match_threshold` -> concatenate (skip repeated header on N+1).
- Tag with `is_continuation=True` and shared `continuation_group_id`.
- Emit `W_TABLE_CONTINUATION` warning.

PLAN section 1e (`_stitch_tables()`) implements:
- Column count check: `last.df.shape[1] == first.df.shape[1]`
- Header similarity: `SequenceMatcher(None, last.headers, first.headers).ratio() >= config.table_continuation_column_match_threshold`
- Repeated header detection: compares first data row of continuation against header list (case-insensitive)
- Assigns `continuation_group_id = str(uuid.uuid4())`
- Sets `is_continuation=True`
- Records `W_TABLE_CONTINUATION` warning

**Verified:** All five SPEC requirements for stitching are covered. Algorithm is correct -- `difflib.SequenceMatcher` on header strings is a reasonable similarity measure for the 0.8 threshold.

### 3. TableResult Model Population

PLAN section 1h (`_build_table_result()`) populates:
- `page_number` = first page in group
- `table_index` = index of first table in group
- `row_count` = `len(table.df)`
- `col_count` = `table.df.shape[1]`
- `headers` = header list
- `is_continuation` = from stitching result
- `continuation_group_id` = from stitching result

**Verified:** All 7 fields of `TableResult` (models.py lines 218-227) are populated correctly.

### 4. pdfplumber API Usage

PLAN section 1d uses:
- `pdfplumber.open(file_path)` as context manager
- `pdf.pages[page_number - 1]` (correct 0-indexed access for 1-indexed page numbers)
- `page.extract_tables()` returning `list[list[list[str | None]]]`
- `pd.DataFrame(table_data[1:], columns=table_data[0])` for conversion

**Verified:** pdfplumber API usage is correct. Pages are 0-indexed in pdfplumber, and the PLAN correctly applies the `-1` offset. `extract_tables()` returns the documented format.

**Verified:** None/empty header fallback to `column_N` is handled.

### 5. Test Coverage

16 test cases covering both routing paths and stitching:

**R-PC-2 (Table Extraction) -- 10 tests:**

| Test | Validates |
|------|-----------|
| `test_single_table_small_nl_serialization` | NL path, chunk count, content_type tag |
| `test_single_table_large_db_routing` | DB path, `create_table_from_dataframe` called, schema chunk |
| `test_boundary_20_rows_nl_path` | Boundary: exactly 20 -> NL |
| `test_boundary_21_rows_db_path` | Boundary: exactly 21 -> DB |
| `test_empty_table_skipped` | Header-only table skipped |
| `test_multiple_tables_single_page` | Correct `table_index` (0, 1) |
| `test_multiple_pages_no_stitching` | Different columns -> no stitch |
| `test_none_headers_fallback` | None headers -> `column_N` |
| `test_extraction_error_per_table` | Per-table error isolation |
| `test_no_backends_pure_extraction` | None backends -> no crash, TableResult populated |

**R-PC-3 (Multi-Page Stitching) -- 6 tests:**

| Test | Validates |
|------|-----------|
| `test_continuation_stitching_basic` | `is_continuation`, `continuation_group_id`, `W_TABLE_CONTINUATION` |
| `test_continuation_skip_repeated_header` | Repeated header row skipped, correct row count |
| `test_continuation_below_threshold` | Similarity 0.7 < 0.8 -> NOT stitched |
| `test_continuation_column_count_mismatch` | Different col count -> NOT stitched |
| `test_continuation_three_pages` | Three-page stitching into single table |
| `test_stitched_table_routing` | Stitched 30-row table -> DB path, multi-page metadata |

**Verified:** Both routing paths tested (including boundary conditions at 20/21 rows). Both positive and negative stitching conditions tested. Error isolation and None-backend safety tested. Mock pdfplumber pattern is correct.

### 6. Scope Check

Files touched: 4 (1 new module, 1 new test file, 2 edits)
- Create: `processors/table_extractor.py`, `tests/test_table_extractor.py`
- Edit: `processors/__init__.py`, `tests/conftest.py`

**No scope creep detected:**
- No ComplexProcessor implementation (deferred, correctly noted as consumer)
- No TextExtractor or OCRProcessor logic
- No form field extraction
- No changes to models.py (TableResult already exists)
- No changes to config.py (config fields already exist)
- `TableExtractionResult` defined in-module (not promoted to models.py) -- appropriate for internal result type

### 7. Mock Backend Protocol Conformance

Verified each mock against `ingestkit_core.protocols`:

- `MockStructuredDBBackend`: implements all 5 protocol methods (`create_table_from_dataframe`, `drop_table`, `table_exists`, `get_table_schema`, `get_connection_uri`)
- `MockVectorStoreBackend`: implements all 4 protocol methods (`upsert_chunks`, `ensure_collection`, `create_payload_index`, `delete_by_ids`)
- `MockEmbeddingBackend`: implements both protocol methods (`embed`, `dimension`)

**Verified:** All mocks satisfy their respective `@runtime_checkable` Protocol interfaces.

### 8. Pattern Compliance

- [x] Backend-agnostic core: references Protocol types only, backends injected via constructor
- [x] Structural subtyping: no ABC classes
- [x] Pydantic v2: `TableExtractionResult` uses `BaseModel`
- [x] Error codes: `E_PROCESS_TABLE_EXTRACT` (line 53 of errors.py), `W_TABLE_CONTINUATION` (line 65)
- [x] Logger name: `"ingestkit_pdf"` (consistent with package convention)
- [x] Per-table error handling: one table failing does not block others

## Issues Found

**ISSUE-1 (INFO): `EmbedStageResult` and `WrittenArtifacts` imported but not used in PLAN.** The imports section includes `EmbedStageResult` and `WrittenArtifacts` from `ingestkit_core.models`, but the PLAN does not show how these are populated. The `TableExtractionResult` has `texts_embedded` and `embed_duration_seconds` fields as aggregation helpers. This is fine -- the caller (`ComplexProcessor`) will aggregate these. PATCH agent should only import what is actually used.

**ISSUE-2 (LOW): Test fixtures duplicated between conftest.py and test file.** The PLAN defines `config`, `mock_structured_db`, `mock_vector_store`, `mock_embedder` fixtures locally in `test_table_extractor.py`, while also adding them to `conftest.py`. PATCH agent should use the conftest.py fixtures only (they are shared and available to all test files). Remove the local duplicates in the test file.

## Verdict

**PASS** -- The PLAN is complete and correctly implements SPEC 11.3 steps 2-3. Dual routing boundary is clean (<=20 NL, >20 DB). Multi-page stitching algorithm correctly checks column count + header similarity with threshold. TableResult fields are fully populated. pdfplumber API usage is correct (0-indexed pages). Test coverage spans both routing paths, boundary conditions, stitching positive/negative cases, and error isolation. No scope creep. Ready for PATCH.

AGENT_RETURN: .agents/outputs/plan-check-40-021426.md
