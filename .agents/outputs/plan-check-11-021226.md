---
issue: 11
agent: PLAN-CHECK
date: 2026-02-12
plan_artifact: map-plan-11-021226.md
status: PASS
---

# PLAN-CHECK — Issue #11: Concrete Backends

## Executive Summary

The MAP-PLAN artifact for issue #11 is **well-formed and complete**. All issue requirements map to plan sections. Protocol signatures match the authoritative source exactly. Scope is correctly contained to `backends/` files, tests, and `__init__.py` exports. The `backends/` directory does not yet exist, confirming a clean starting point. Two minor findings noted below — neither is blocking.

## Check 1: Requirement Coverage

| Issue Requirement | Plan Section | Status |
|---|---|---|
| **QdrantVectorStore** | File 2: `backends/qdrant.py` | |
| `ensure_collection` (cosine, create if not exists) | Method 1 — creates with `Distance.COSINE` | ✅ |
| `upsert_chunks` (batch points with metadata) | Method 2 — converts to `PointStruct`, includes text in payload | ✅ |
| `create_payload_index` (keyword/integer) | Method 3 — maps `field_type` to `PayloadSchemaType` | ✅ |
| `delete_by_ids` | Method 4 — `PointIdsList` selector | ✅ |
| Timeout + retry with exponential backoff | Method 5 (`_with_retry`) + constructor params from config | ✅ |
| **SQLiteStructuredDB** | File 3: `backends/sqlite.py` | |
| `create_table_from_dataframe` (replace if exists) | Method 1 — `df.to_sql(if_exists="replace")` | ✅ |
| `drop_table` | Method 2 — `DROP TABLE IF EXISTS` | ✅ |
| `table_exists` (check `sqlite_master`) | Method 3 — `SELECT FROM sqlite_master` | ✅ |
| `get_table_schema` (`PRAGMA table_info`) | Method 4 — returns `{name: type}` dict | ✅ |
| `get_connection_uri` | Method 5 — `sqlite:///path` or `sqlite://` for memory | ✅ |
| **OllamaLLM** | File 4: `backends/ollama.py` | |
| `classify` (JSON parse + retry on malformed) | Method 1 — retry once with correction hint, `E_LLM_MALFORMED_JSON` | ✅ |
| `generate` (raw text) | Method 2 — returns `response["response"]` | ✅ |
| Timeout → `E_LLM_TIMEOUT` | Both methods handle `httpx.TimeoutException` | ✅ |
| **OllamaEmbedding** | File 4: `backends/ollama.py` | |
| `embed` (batch, float vectors, timeout) | Method 1 — `POST /api/embed` | ✅ |
| `dimension` (768 default) | Method 2 — returns `self._vector_dimension` | ✅ |
| **Stubs** | Files 5-6 | |
| `MilvusVectorStore` → `NotImplementedError` | File 5 — all 4 methods raise with message | ✅ |
| `PostgresStructuredDB` → `NotImplementedError` | File 6 — all 5 methods raise with message | ✅ |
| Module-level docstrings explaining how to implement | Both stubs have docstrings referencing the concrete impl | ✅ |

**Coverage: 20/20 requirements mapped.** No gaps.

## Check 2: Scope Containment

| Scope Rule | Status |
|---|---|
| Only creates files under `backends/` | ✅ 5 new files in `src/ingestkit_excel/backends/` |
| Only creates `tests/test_backends.py` | ✅ Single test file |
| Updates only `__init__.py` exports | ✅ File 8 adds imports + `__all__` entries |
| No changes to core processing (`parser_chain`, `inspector`, `llm_classifier`, `processors`) | ✅ Not touched |
| No changes to `protocols.py`, `models.py`, `config.py`, `errors.py` | ✅ Not touched |
| No ABC base classes introduced | ✅ All backends are concrete classes, no inheritance from protocols |

**Scope: Clean.** No overreach detected.

## Check 3: Protocol Signature Match

Compared plan's method signatures against the authoritative source at `/home/jjob/projects/ingestkit/packages/ingestkit-core/src/ingestkit_core/protocols.py`:

| Protocol | Method | Plan Signature | Actual Signature | Match |
|---|---|---|---|---|
| VectorStoreBackend | `upsert_chunks` | `(self, collection: str, chunks: list[ChunkPayload]) -> int` | `(self, collection: str, chunks: list[ChunkPayload]) -> int` | ✅ |
| VectorStoreBackend | `ensure_collection` | `(self, collection: str, vector_size: int) -> None` | `(self, collection: str, vector_size: int) -> None` | ✅ |
| VectorStoreBackend | `create_payload_index` | `(self, collection: str, field: str, field_type: str) -> None` | `(self, collection: str, field: str, field_type: str) -> None` | ✅ |
| VectorStoreBackend | `delete_by_ids` | `(self, collection: str, ids: list[str]) -> int` | `(self, collection: str, ids: list[str]) -> int` | ✅ |
| StructuredDBBackend | `create_table_from_dataframe` | `(self, table_name: str, df: pd.DataFrame) -> None` | `(self, table_name: str, df: pd.DataFrame) -> None` | ✅ |
| StructuredDBBackend | `drop_table` | `(self, table_name: str) -> None` | `(self, table_name: str) -> None` | ✅ |
| StructuredDBBackend | `table_exists` | `(self, table_name: str) -> bool` | `(self, table_name: str) -> bool` | ✅ |
| StructuredDBBackend | `get_table_schema` | `(self, table_name: str) -> dict` | `(self, table_name: str) -> dict` | ✅ |
| StructuredDBBackend | `get_connection_uri` | `(self) -> str` | `(self) -> str` | ✅ |
| LLMBackend | `classify` | `(self, prompt: str, model: str, temperature: float = 0.1, timeout: float \| None = None) -> dict` | `(self, prompt: str, model: str, temperature: float = 0.1, timeout: float \| None = None) -> dict` | ✅ |
| LLMBackend | `generate` | `(self, prompt: str, model: str, temperature: float = 0.7, timeout: float \| None = None) -> str` | `(self, prompt: str, model: str, temperature: float = 0.7, timeout: float \| None = None) -> str` | ✅ |
| EmbeddingBackend | `embed` | `(self, texts: list[str], timeout: float \| None = None) -> list[list[float]]` | `(self, texts: list[str], timeout: float \| None = None) -> list[list[float]]` | ✅ |
| EmbeddingBackend | `dimension` | `(self) -> int` | `(self) -> int` | ✅ |

**All 13 method signatures match exactly.**

## Check 4: Wiring

| Item | Status | Detail |
|---|---|---|
| `backends/` directory does not exist yet | ✅ Confirmed: `ls` returns "No such file or directory" |
| Current `__init__.py` exports | ✅ Reviewed — no backend imports present; plan adds them cleanly |
| `backends/__init__.py` imports all 6 classes | ✅ Plan File 1 shows complete `__all__` |
| Package `__init__.py` re-exports all 6 | ✅ Plan File 8 adds imports + `__all__` entries |
| Protocols re-exported from `ingestkit_excel.protocols` | ✅ Already in place (re-exports from `ingestkit_core.protocols`) |

## Check 5: ChunkPayload Model Reference

The plan references `ChunkPayload` with fields `id`, `text`, `vector`, `metadata` (of type `BaseChunkMetadata`). Verified against `/home/jjob/projects/ingestkit/packages/ingestkit-core/src/ingestkit_core/models.py`:
- `ChunkPayload.id: str` ✅
- `ChunkPayload.text: str` ✅
- `ChunkPayload.vector: list[float]` ✅
- `ChunkPayload.metadata: BaseChunkMetadata` ✅

Plan's `upsert_chunks` correctly accesses `chunk.metadata.model_dump()` and adds `chunk.text` to the Qdrant payload. This is consistent.

## Findings

### Finding 1 (MINOR): `backends/__init__.py` eager imports may fail without optional deps

The plan's `backends/__init__.py` (File 1) unconditionally imports from `qdrant.py` and `ollama.py`. If `qdrant-client` or `httpx` are not installed, `from ingestkit_excel.backends import SQLiteStructuredDB` will fail even though SQLite has no optional deps.

**Recommendation for PATCH**: Use lazy imports or conditional imports in `backends/__init__.py`. Alternatively, the import guards inside `qdrant.py` and `ollama.py` should raise `ImportError` only when the class is *instantiated*, not when the module is *imported*. The plan already mentions import guards but places them at module level — PATCH should ensure the guard fires in `__init__` (constructor), not at import time.

**Severity**: Minor — does not block plan acceptance but PATCH should handle it.

### Finding 2 (MINOR): `E_LLM_SCHEMA_INVALID` listed in error codes table but not used

The plan's error codes table (line 59) lists `E_LLM_SCHEMA_INVALID` but no method in the plan raises it. The issue mentions "validate against schema" for `classify`. This is actually fine — schema validation is done by `LLMClassifier` (a higher-level component), not by the backend itself. The backend `classify` only parses JSON. No action needed.

## Verdict

**PASS** — The plan is complete, correctly scoped, and protocol-accurate. Ready for PATCH.

AGENT_RETURN: plan-check-11-021226.md
