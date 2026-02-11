# ingestkit-excel — Technical Specification

**Version:** 2.0
**Status:** DRAFT — Pending final review
**Previous version:** 1.0 (initial draft)
**Changelog:** v2.0 incorporates team review feedback (P0/P1 gaps) and OSS benchmark findings.

## 1. Overview

**Package:** `ingestkit-excel`
**Python package name:** `ingestkit_excel`
**Parent ecosystem:** `ingestkit` — a plugin-based ingestion framework for the "AI Help Desk in a Box" on-premises RAG system.

This package classifies `.xlsx` files into one of three structural types and routes each to the appropriate ingestion path, producing chunks in a vector store and/or tables in a structured database.

### 1.1 Problem Statement

Excel files in HR and IT environments vary wildly in structure:
- Some are clean tabular data (employee rosters, inventory lists).
- Some abuse Excel as a layout tool (onboarding checklists, compliance matrices).
- Some are hybrids of both.

A single ingestion strategy fails on this diversity. This package solves that with tiered classification and type-specific processing.

### 1.2 Scope

**In scope:**
- Classification of Excel files into Type A (tabular), Type B (document-formatted), or Type C (hybrid).
- Three-tier detection: rule-based → lightweight LLM → reasoning model.
- Three processing paths: structured DB ingestion, text serialization, and hybrid split-and-route.
- Abstract backend interfaces (Protocols) for vector store, structured DB, LLM, and embeddings.
- Concrete backend implementations for Qdrant, SQLite, and Ollama.
- Stub implementations for Milvus, PostgreSQL.
- Deterministic idempotency keying for deduplication.
- Parser fallback chain with reason codes.
- PII-safe structured logging with configurable redaction.
- Normalized error taxonomy.
- Standardized chunk/table metadata schema.

**Out of scope:**
- Query-time SQL agent or retrieval logic.
- Full RAG pipeline (retrieval, reranking, generation).
- Cloud API backends (OpenAI, Pinecone, etc.).
- Non-Excel file formats (handled by sibling `ingestkit-*` packages).
- Durable state machines, dead-letter queues, or job orchestration (caller's responsibility).
- Multi-tenant isolation enforcement (package propagates `tenant_id`; caller enforces).
- Ingestion quality dashboards (package emits structured data; caller visualizes).

---

## 2. Architecture

### 2.1 Design Principles

1. **Backend-agnostic core.** All processing logic references Protocol types, never concrete backends.
2. **Dependency injection.** The `ExcelRouter` accepts backend instances; it never creates them.
3. **Structural subtyping.** Backends use `typing.Protocol` (not ABCs) — implement the methods and it works.
4. **Fail gracefully.** Corrupt files, password-protected sheets, chart-only sheets — return structured errors with normalized error codes, never crash.
5. **PII-safe observability.** Every file processed emits a structured log entry. No raw data, sample rows, or chunk content in logs at any level without explicit opt-in.
6. **Idempotent by default.** Every ingest produces a deterministic key; re-ingesting the same content with the same parser version produces the same key and can be deduplicated by the caller.
7. **Parser resilience.** Primary parser failure triggers a fallback chain, not an immediate error.

### 2.2 High-Level Flow

```
.xlsx file
    │
    ▼
┌──────────────────────────┐
│  Compute ingest_key       │  content_hash + source_uri + parser_version
│  (idempotency)            │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  Parse with fallback      │  openpyxl → pandas fallback → raw text fallback
│  chain                    │  Records fallback_reason_code per sheet
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  Tier 1: Inspector        │  Rule-based structural analysis
│  (no LLM)                │  Handles ~85% of files
└──────────┬───────────────┘
           │ confidence < threshold
           ▼
┌──────────────────────────┐
│  Tier 2: LLM Classifier  │  Lightweight classification model
│                          │  Handles ~12% of files
└──────────┬───────────────┘
           │ confidence < 0.6
           ▼
┌──────────────────────────┐
│  Tier 3: Reasoning Model  │  Larger reasoning model (optional)
│                          │  Handles ~3% of files
└──────────────────────────┘
           │
           ▼
   ClassificationResult
   {type, confidence, tier_used, per_sheet_types}
           │
           ├─ Type A ──▶ Path A: Structured DB ingestion
           ├─ Type B ──▶ Path B: Text serialization
           └─ Type C ──▶ Path C: Split regions → route to A or B
           │
           ▼
   ProcessingResult
   {stage_artifacts, written_ids, error_codes, ...}
```

### 2.3 Module Structure

```
ingestkit-excel/
├── pyproject.toml
├── SPEC.md
├── ROADMAP.md
└── src/
    └── ingestkit_excel/
        ├── __init__.py              # Exports: ExcelRouter, create_default_router
        ├── protocols.py             # VectorStoreBackend, StructuredDBBackend, LLMBackend, EmbeddingBackend
        ├── models.py                # Pydantic models for all data structures
        ├── errors.py                # Normalized error codes and error model
        ├── config.py                # ExcelProcessorConfig with all thresholds and defaults
        ├── idempotency.py           # Ingest key computation (content hash + source_uri + parser_version)
        ├── parser_chain.py          # Parser fallback chain: openpyxl → pandas → raw text
        ├── inspector.py             # Tier 1: rule-based structural analysis
        ├── llm_classifier.py        # Tier 2 & 3: LLM-based classification with schema validation
        ├── router.py                # ExcelRouter: orchestrates detection → processing
        ├── processors/
        │   ├── __init__.py
        │   ├── structured_db.py     # Path A: tabular → structured DB + schema embeddings
        │   ├── serializer.py        # Path B: document-formatted → natural language chunks
        │   └── splitter.py          # Path C: hybrid → detect regions → route to A or B
        ├── backends/
        │   ├── __init__.py
        │   ├── qdrant.py            # QdrantVectorStore
        │   ├── milvus.py            # MilvusVectorStore (stub)
        │   ├── sqlite.py            # SQLiteStructuredDB
        │   ├── postgres.py          # PostgresStructuredDB (stub)
        │   └── ollama.py            # OllamaLLM + OllamaEmbedding
        └── tests/
            ├── __init__.py
            ├── conftest.py          # Shared fixtures, mock backends
            ├── test_inspector.py
            ├── test_router.py
            ├── test_processors.py
            ├── test_backends.py
            ├── test_parser_chain.py
            ├── test_idempotency.py
            ├── test_llm_validation.py
            └── fixtures/            # Sample .xlsx files for Type A, B, C
```

---

## 3. Data Models (`models.py`)

All models use Pydantic v2 (`BaseModel`).

### 3.1 Enumerations

```python
class FileType(str, Enum):
    TABULAR_DATA = "tabular_data"              # Type A
    FORMATTED_DOCUMENT = "formatted_document"  # Type B
    HYBRID = "hybrid"                          # Type C

class ClassificationTier(str, Enum):
    RULE_BASED = "rule_based"        # Tier 1
    LLM_BASIC = "llm_basic"         # Tier 2
    LLM_REASONING = "llm_reasoning"  # Tier 3

class IngestionMethod(str, Enum):
    SQL_AGENT = "sql_agent"                    # Path A
    TEXT_SERIALIZATION = "text_serialization"   # Path B
    HYBRID_SPLIT = "hybrid_split"              # Path C

class RegionType(str, Enum):
    DATA_TABLE = "data_table"
    TEXT_BLOCK = "text_block"
    HEADER_BLOCK = "header_block"
    FOOTER_BLOCK = "footer_block"
    MATRIX_BLOCK = "matrix_block"
    CHART_ONLY = "chart_only"
    EMPTY = "empty"

class ParserUsed(str, Enum):
    OPENPYXL = "openpyxl"
    PANDAS_FALLBACK = "pandas_fallback"
    RAW_TEXT_FALLBACK = "raw_text_fallback"
```

### 3.2 Idempotency

```python
class IngestKey(BaseModel):
    """Deterministic key for deduplication."""
    content_hash: str          # SHA-256 of file bytes
    source_uri: str            # canonical path or URI of the source file
    parser_version: str        # e.g. "ingestkit_excel:1.0.0"
    tenant_id: str | None = None  # propagated from config if multi-tenant

    @property
    def key(self) -> str:
        """Deterministic string key for dedup lookups."""
        parts = [self.content_hash, self.source_uri, self.parser_version]
        if self.tenant_id:
            parts.append(self.tenant_id)
        return hashlib.sha256("|".join(parts).encode()).hexdigest()
```

### 3.3 Stage Artifacts

```python
class ParseStageResult(BaseModel):
    """Typed output of the parsing stage."""
    parser_used: ParserUsed
    fallback_reason_code: str | None = None  # e.g. "E_PARSE_OPENPYXL_CORRUPT"
    sheets_parsed: int
    sheets_skipped: int
    skipped_reasons: dict[str, str]          # {sheet_name: reason_code}
    parse_duration_seconds: float

class ClassificationStageResult(BaseModel):
    """Typed output of the classification stage."""
    tier_used: ClassificationTier
    file_type: FileType
    confidence: float
    signals: dict[str, Any] | None = None     # Tier 1 signal breakdown
    reasoning: str
    per_sheet_types: dict[str, FileType] | None = None
    classification_duration_seconds: float

class EmbedStageResult(BaseModel):
    """Typed output of the embedding stage."""
    texts_embedded: int
    embedding_dimension: int
    embed_duration_seconds: float
```

### 3.4 Core Models

```python
class SheetProfile(BaseModel):
    """Structural profile of a single worksheet."""
    name: str
    row_count: int
    col_count: int
    merged_cell_count: int
    merged_cell_ratio: float          # merged_cells / total_cells
    header_row_detected: bool
    header_values: list[str]
    column_type_consistency: float    # 0.0-1.0, how uniform column dtypes are
    numeric_ratio: float              # proportion of numeric cells
    text_ratio: float                 # proportion of text cells
    empty_ratio: float                # proportion of empty cells
    sample_rows: list[list[str]]      # first N rows as strings (redacted in logs)
    has_formulas: bool
    is_hidden: bool
    parser_used: ParserUsed           # which parser succeeded on this sheet

class FileProfile(BaseModel):
    """Aggregate structural profile of an Excel file."""
    file_path: str
    file_size_bytes: int
    sheet_count: int
    sheet_names: list[str]
    sheets: list[SheetProfile]
    has_password_protected_sheets: bool
    has_chart_only_sheets: bool
    total_merged_cells: int
    total_rows: int
    content_hash: str                 # SHA-256 of file bytes

class ClassificationResult(BaseModel):
    """Result of the tiered classification."""
    file_type: FileType
    confidence: float                 # 0.0-1.0
    tier_used: ClassificationTier
    reasoning: str
    per_sheet_types: dict[str, FileType] | None = None
    signals: dict[str, Any] | None = None

class ChunkMetadata(BaseModel):
    """Standardized metadata attached to every chunk. Canonical schema across all paths."""
    source_uri: str                   # canonical file path or URI
    source_format: str = "xlsx"
    sheet_name: str
    region_id: str | None = None      # for hybrid splits
    ingestion_method: str             # IngestionMethod value
    parser_used: str                  # ParserUsed value
    parser_version: str               # e.g. "ingestkit_excel:1.0.0"
    chunk_index: int                  # 0-based position within this source
    chunk_hash: str                   # SHA-256 of chunk text
    ingest_key: str                   # idempotency key
    ingest_run_id: str                # unique per process() invocation
    tenant_id: str | None = None
    # Path A specific
    table_name: str | None = None
    db_uri: str | None = None
    row_count: int | None = None
    columns: list[str] | None = None
    # Path B specific
    section_title: str | None = None
    original_structure: str | None = None  # "table", "checklist", "matrix", "free_text"

class ChunkPayload(BaseModel):
    """A single chunk ready for vector store upsert."""
    id: str                           # deterministic UUID from chunk_hash + ingest_key
    text: str
    vector: list[float]
    metadata: ChunkMetadata

class SheetRegion(BaseModel):
    """A detected region within a worksheet (used by Path C splitter)."""
    sheet_name: str
    region_id: str                    # unique within the file
    start_row: int
    end_row: int
    start_col: int
    end_col: int
    region_type: RegionType
    detection_confidence: float       # 0.0-1.0, confidence in the boundary detection
    classified_as: FileType | None = None

class WrittenArtifacts(BaseModel):
    """IDs of everything written to backends, enabling caller-side rollback."""
    vector_point_ids: list[str] = []       # IDs upserted to vector store
    vector_collection: str | None = None
    db_table_names: list[str] = []         # tables created in structured DB

class ProcessingResult(BaseModel):
    """Final result returned after processing a file."""
    file_path: str
    ingest_key: str                        # idempotency key
    ingest_run_id: str                     # unique per invocation
    tenant_id: str | None = None

    # Stage artifacts (typed, persisted)
    parse_result: ParseStageResult
    classification_result: ClassificationStageResult
    embed_result: EmbedStageResult | None = None

    # Legacy convenience fields
    classification: ClassificationResult
    ingestion_method: IngestionMethod

    # Outputs
    chunks_created: int
    tables_created: int
    tables: list[str]
    written: WrittenArtifacts              # for caller-side rollback

    # Errors and warnings (normalized codes)
    errors: list[str]                      # list of ErrorCode values
    warnings: list[str]                    # list of ErrorCode values
    error_details: list["IngestError"] = []  # structured error objects

    processing_time_seconds: float
```

---

## 4. Error Taxonomy (`errors.py`)

### 4.1 Normalized Error Codes

All errors use a normalized code from a defined enum. Error codes are stable strings suitable for metrics, alerting, and programmatic handling.

```python
class ErrorCode(str, Enum):
    """Normalized error codes for the ingestkit-excel pipeline."""

    # Parse errors
    E_PARSE_CORRUPT = "E_PARSE_CORRUPT"                  # file cannot be opened by any parser
    E_PARSE_OPENPYXL_FAIL = "E_PARSE_OPENPYXL_FAIL"     # openpyxl primary parser failed
    E_PARSE_PANDAS_FAIL = "E_PARSE_PANDAS_FAIL"          # pandas fallback parser failed
    E_PARSE_PASSWORD = "E_PARSE_PASSWORD"                # password-protected sheet
    E_PARSE_EMPTY = "E_PARSE_EMPTY"                      # file has no data sheets
    E_PARSE_TOO_LARGE = "E_PARSE_TOO_LARGE"              # exceeds max_rows_in_memory

    # Classification errors
    E_CLASSIFY_INCONCLUSIVE = "E_CLASSIFY_INCONCLUSIVE"  # all tiers failed to classify
    E_LLM_TIMEOUT = "E_LLM_TIMEOUT"                     # LLM backend timed out
    E_LLM_MALFORMED_JSON = "E_LLM_MALFORMED_JSON"       # LLM returned unparseable JSON
    E_LLM_SCHEMA_INVALID = "E_LLM_SCHEMA_INVALID"       # LLM JSON failed schema validation
    E_LLM_CONFIDENCE_OOB = "E_LLM_CONFIDENCE_OOB"       # confidence outside 0.0-1.0

    # Backend errors
    E_BACKEND_VECTOR_TIMEOUT = "E_BACKEND_VECTOR_TIMEOUT"
    E_BACKEND_VECTOR_CONNECT = "E_BACKEND_VECTOR_CONNECT"
    E_BACKEND_DB_TIMEOUT = "E_BACKEND_DB_TIMEOUT"
    E_BACKEND_DB_CONNECT = "E_BACKEND_DB_CONNECT"
    E_BACKEND_EMBED_TIMEOUT = "E_BACKEND_EMBED_TIMEOUT"
    E_BACKEND_EMBED_CONNECT = "E_BACKEND_EMBED_CONNECT"

    # Processing errors
    E_PROCESS_REGION_DETECT = "E_PROCESS_REGION_DETECT"  # hybrid region detection failed
    E_PROCESS_SERIALIZE = "E_PROCESS_SERIALIZE"          # text serialization failed
    E_PROCESS_SCHEMA_GEN = "E_PROCESS_SCHEMA_GEN"       # schema description generation failed

    # Warnings (non-fatal)
    W_SHEET_SKIPPED_CHART = "W_SHEET_SKIPPED_CHART"      # chart-only sheet skipped
    W_SHEET_SKIPPED_HIDDEN = "W_SHEET_SKIPPED_HIDDEN"    # hidden sheet skipped (if configured)
    W_SHEET_SKIPPED_PASSWORD = "W_SHEET_SKIPPED_PASSWORD" # password-protected sheet skipped
    W_PARSER_FALLBACK = "W_PARSER_FALLBACK"              # primary parser failed, fallback used
    W_LLM_RETRY = "W_LLM_RETRY"                         # LLM call retried
    W_ROWS_TRUNCATED = "W_ROWS_TRUNCATED"                # rows exceeded max_rows_in_memory

class IngestError(BaseModel):
    """Structured error with code, message, and context."""
    code: ErrorCode
    message: str
    sheet_name: str | None = None
    stage: str | None = None          # "parse", "classify", "process", "embed"
    recoverable: bool = False
```

### 4.2 Fail-Closed Default

The pipeline defaults to **fail-closed** behavior:
- If all classification tiers fail → return `ProcessingResult` with `E_CLASSIFY_INCONCLUSIVE`, zero chunks/tables. Do not guess.
- If LLM output fails schema validation after retry → fall back to Tier 1 result if available, otherwise fail with `E_LLM_SCHEMA_INVALID`.
- If primary and fallback parsers both fail → fail with `E_PARSE_CORRUPT`, do not produce partial output.

---

## 5. Configuration (`config.py`)

```python
class ExcelProcessorConfig(BaseModel):
    """All tunable parameters with sensible defaults."""

    # --- Identity ---
    parser_version: str = "ingestkit_excel:1.0.0"
    tenant_id: str | None = None       # propagated to all metadata and ingest keys

    # --- Tier 1 thresholds ---
    tier1_high_confidence_signals: int = 4     # out of 5 signals
    tier1_medium_confidence_signals: int = 3
    merged_cell_ratio_threshold: float = 0.05  # above this → leans Type B
    numeric_ratio_threshold: float = 0.3       # above this → leans Type A
    column_consistency_threshold: float = 0.7  # above this → leans Type A
    min_row_count_for_tabular: int = 5

    # --- Tier 2/3 LLM settings ---
    classification_model: str = "qwen2.5:7b"
    reasoning_model: str = "deepseek-r1:14b"
    tier2_confidence_threshold: float = 0.6    # below this → escalate to Tier 3
    llm_temperature: float = 0.1

    # --- Path A settings ---
    row_serialization_limit: int = 5000        # only serialize rows if table < this
    clean_column_names: bool = True

    # --- Embedding ---
    embedding_model: str = "nomic-embed-text"
    embedding_dimension: int = 768
    embedding_batch_size: int = 64

    # --- Vector store ---
    default_collection: str = "helpdesk"

    # --- General ---
    max_sample_rows: int = 3                   # rows to include in LLM summaries
    enable_tier3: bool = True                  # set False to skip reasoning model
    max_rows_in_memory: int = 100_000          # sheets exceeding this are skipped with W_ROWS_TRUNCATED

    # --- Backend resilience ---
    backend_timeout_seconds: float = 30.0      # per-call timeout for all backends
    backend_max_retries: int = 2               # retries with exponential backoff
    backend_backoff_base: float = 1.0          # base seconds for backoff (1s, 2s, 4s)

    # --- Logging / PII safety ---
    log_sample_data: bool = False              # if True, include sample rows in DEBUG logs
    log_llm_prompts: bool = False              # if True, include LLM prompts/responses in DEBUG logs
    log_chunk_previews: bool = False           # if True, include chunk text in DEBUG logs
    redact_patterns: list[str] = []            # regex patterns to redact from any logged text (e.g. SSN, email)
```

Supports loading from YAML or JSON:
```python
@classmethod
def from_file(cls, path: str) -> "ExcelProcessorConfig":
    ...
```

---

## 6. Idempotency (`idempotency.py`)

### 6.1 Purpose

Ensure that re-ingesting the same file with the same parser version produces a deterministic key. The caller uses this key to decide whether to skip, overwrite, or version the ingest.

### 6.2 Key Derivation

```python
def compute_ingest_key(
    file_path: str,
    parser_version: str,
    tenant_id: str | None = None,
) -> IngestKey:
    """Compute a deterministic ingest key for deduplication.

    Components:
    - content_hash: SHA-256 of the raw file bytes (content-addressed).
    - source_uri: canonical absolute path or URI.
    - parser_version: from config (e.g. "ingestkit_excel:1.0.0").
    - tenant_id: optional, included if multi-tenant.
    """
    ...
```

### 6.3 Usage Contract

- The `ingest_key` is included in every `ProcessingResult` and every `ChunkMetadata`.
- If the caller sees an existing ingest with the same key, it can skip processing.
- If the file content changes (different hash) or the parser version changes, a new key is generated.
- The package does NOT enforce deduplication — it provides the key. The caller decides the policy.

---

## 7. Parser Fallback Chain (`parser_chain.py`)

### 7.1 Purpose

Ensure that parser failures don't immediately fail the ingest. Fallback parsers trade fidelity for resilience.

### 7.2 Chain

```
Primary: openpyxl (full fidelity — merged cells, formulas, formatting)
    │
    │ fails with exception
    ▼
Fallback 1: pandas read_excel (loses merged cell info, gains robustness)
    │
    │ fails with exception
    ▼
Fallback 2: raw text extraction via openpyxl data_only=True (minimal fidelity)
    │
    │ fails with exception
    ▼
Error: E_PARSE_CORRUPT (all parsers failed)
```

### 7.3 Behavior

- Each fallback records a `fallback_reason_code` (the `ErrorCode` from the failed parser).
- The `ParserUsed` enum is set on each `SheetProfile` so downstream processors know what fidelity to expect.
- If a fallback is used, `W_PARSER_FALLBACK` is added to `ProcessingResult.warnings`.
- Fallback is per-sheet: if sheet 1 succeeds with openpyxl but sheet 2 needs pandas, both are recorded independently.

### 7.4 Public Interface

```python
class ParserChain:
    def __init__(self, config: ExcelProcessorConfig): ...
    def parse(self, file_path: str) -> tuple[FileProfile, list[IngestError]]:
        """Parse file with fallback chain. Returns profile and any non-fatal errors."""
        ...
```

---

## 8. Tier 1 — Rule-Based Inspector (`inspector.py`)

### 8.1 Purpose

Classify files without any LLM call. Fast, deterministic, handles the majority of files.

### 8.2 Signals

The inspector evaluates **5 binary signals** per sheet, scored toward Type A or Type B:

| # | Signal | Type A (tabular) | Type B (document) |
|---|--------|-------------------|---------------------|
| 1 | Row count | >= `min_row_count_for_tabular` | < threshold |
| 2 | Merged cell ratio | < `merged_cell_ratio_threshold` | >= threshold |
| 3 | Column type consistency | >= `column_consistency_threshold` | < threshold |
| 4 | Header row detected | Yes, with distinct typed columns | No, or headers are merged |
| 5 | Numeric vs. text ratio | Numeric ratio >= `numeric_ratio_threshold` | Text-dominant |

### 8.3 Decision Logic

For each sheet:
- Count signals matching Type A profile and Type B profile.
- If >= 4 signals match one type → **high confidence** (0.9).
- If 3 signals match → **medium confidence** (0.7), flag for optional review.
- If < 3 or split → **inconclusive**, escalate to Tier 2.

For multi-sheet files:
- If all sheets agree → classify the file as that type.
- If sheets disagree → classify as **Type C (hybrid)**, record per-sheet types.

### 8.4 Public Interface

```python
class ExcelInspector:
    def __init__(self, config: ExcelProcessorConfig): ...
    def classify(self, profile: FileProfile) -> ClassificationResult: ...
```

---

## 9. Tier 2 & 3 — LLM Classifier (`llm_classifier.py`)

### 9.1 Purpose

Handle ambiguous files that Tier 1 cannot classify with sufficient confidence.

### 9.2 Structural Summary Generation

Before calling the LLM, generate a text summary of the file structure. **Never send raw data values.** Summary includes only structural metadata:

```
File: benefits_matrix.xlsx
Sheets: 3 (Benefits Overview, Plan Comparison, Notes)

Sheet "Benefits Overview":
- Rows: 45, Columns: 8
- Merged cells: 12 (ratio: 0.033)
- Headers: [Employee Type, Medical, Dental, Vision, 401k Match, PTO Days, ...]
- Column types: [str, str, str, str, float, int, ...]
- Sample rows (structure only):
  Row 2: [str, str, str, str, float, int, ...]
  Row 3: [str, str, str, str, float, int, ...]
```

Note: If `config.log_sample_data` is True, actual sample values may be included in the summary sent to the LLM. The default is structure-only to prevent PII leakage through the LLM.

### 9.3 Classification Prompt

```
You are classifying an Excel file for a document ingestion system.
Based on the structural summary below, classify this file as one of:

- "tabular_data": Rows are records, columns are fields. Consistent structure. Suitable for SQL database import.
- "formatted_document": Excel used as a layout/formatting tool. Merged cells, irregular structure, text-heavy. Suitable for text extraction.
- "hybrid": Mix of tabular and document-formatted sections. Different sheets or regions serve different purposes.

Respond with JSON only:
{
  "type": "tabular_data" | "formatted_document" | "hybrid",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "brief explanation",
  "sheet_types": {"sheet_name": "type", ...}  // only if hybrid
}

Structural summary:
{summary}
```

### 9.4 Structured Output Validation

LLM responses are validated with a strict Pydantic model before acceptance:

```python
class LLMClassificationResponse(BaseModel):
    """Schema for validating LLM classification output."""
    type: Literal["tabular_data", "formatted_document", "hybrid"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1)
    sheet_types: dict[str, Literal["tabular_data", "formatted_document"]] | None = None

    @field_validator("confidence")
    @classmethod
    def confidence_in_bounds(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Confidence {v} outside [0.0, 1.0]")
        return v
```

**Validation flow:**
1. Parse LLM response as JSON. If JSON parsing fails → `E_LLM_MALFORMED_JSON`, retry once.
2. Validate against `LLMClassificationResponse`. If validation fails → `E_LLM_SCHEMA_INVALID`, retry once with a correction hint appended to the prompt.
3. Check confidence bounds. If out of range → `E_LLM_CONFIDENCE_OOB`, clamp and warn.
4. After 2 failed attempts → fall back to Tier 1 result (if available) or fail with structured error. **Never accept unvalidated LLM output.**

### 9.5 Tier Escalation

- **Tier 2:** Uses `config.classification_model` (default: `qwen2.5:7b`).
- **Tier 3:** Triggered when Tier 2 confidence < `config.tier2_confidence_threshold` (default: 0.6). Uses `config.reasoning_model` (default: `deepseek-r1:14b`). Can be disabled via `config.enable_tier3 = False`.

### 9.6 Public Interface

```python
class LLMClassifier:
    def __init__(self, llm: LLMBackend, config: ExcelProcessorConfig): ...
    def classify(self, profile: FileProfile, tier: ClassificationTier) -> ClassificationResult: ...
```

---

## 10. Processing Paths

### 10.1 Path A — Structured DB Processor (`processors/structured_db.py`)

**Input:** FileProfile + classification confirming Type A.

**Steps:**
1. Load each sheet into a pandas DataFrame.
2. Clean column names: lowercase, replace spaces/special chars with underscores, deduplicate.
3. Auto-detect date columns (Excel serial dates, string dates) and parse them.
4. Write each sheet as a table to `StructuredDBBackend` via `create_table_from_dataframe()`.
5. Generate a natural language schema description per table:
   ```
   Table "employee_roster" contains 342 rows with columns:
   - employee_id (integer): unique identifier, range 10042-10943
   - full_name (text): employee name
   - department (text): one of Engineering, Sales, HR, Finance, ...
   - hire_date (date): ranges from 2018-01-15 to 2024-11-30
   - salary (float): ranges from 45000.0 to 185000.0
   ```
6. Embed the schema description and upsert to `VectorStoreBackend` with standardized `ChunkMetadata`:
   ```python
   ChunkMetadata(
       source_uri="file:///path/to/roster.xlsx",
       sheet_name="Sheet1",
       ingestion_method="sql_agent",
       parser_used="openpyxl",
       parser_version="ingestkit_excel:1.0.0",
       chunk_index=0,
       chunk_hash="sha256:...",
       ingest_key="...",
       ingest_run_id="...",
       tenant_id="...",
       table_name="employee_roster",
       db_uri="sqlite:///helpdesk.db",
       row_count=342,
       columns=["employee_id", "full_name", "department", "hire_date", "salary"],
   )
   ```
7. **Optional row serialization** (tables < `config.row_serialization_limit` rows): Convert each row to a natural language sentence and embed as individual chunks.

**Public interface:**
```python
class StructuredDBProcessor:
    def __init__(self, structured_db: StructuredDBBackend, vector_store: VectorStoreBackend,
                 embedder: EmbeddingBackend, config: ExcelProcessorConfig): ...
    def process(self, file_path: str, profile: FileProfile,
                ingest_key: str, ingest_run_id: str) -> ProcessingResult: ...
```

### 10.2 Path B — Text Serializer (`processors/serializer.py`)

**Input:** FileProfile + classification confirming Type B.

**Steps:**
1. Parse with openpyxl, preserving merged cell structure.
2. Detect logical sections: look for merged header rows, blank row separators, indentation patterns.
3. For each section, determine its sub-structure:
   - **Small table** → serialize rows as sentences.
   - **Checklist** → "Item X: status is Y, due date is Z, responsible party is W."
   - **Matrix** → serialize with row/column header context.
   - **Free text** → extract as-is, preserving paragraph breaks.
4. Embed each section/chunk via `EmbeddingBackend`.
5. Upsert to `VectorStoreBackend` with standardized `ChunkMetadata`:
   ```python
   ChunkMetadata(
       source_uri="file:///path/to/onboarding.xlsx",
       sheet_name="Onboarding Checklist",
       ingestion_method="text_serialization",
       parser_used="openpyxl",
       parser_version="ingestkit_excel:1.0.0",
       chunk_index=3,
       chunk_hash="sha256:...",
       ingest_key="...",
       ingest_run_id="...",
       tenant_id="...",
       section_title="IT Setup Requirements",
       original_structure="checklist",
   )
   ```

**Public interface:**
```python
class TextSerializer:
    def __init__(self, vector_store: VectorStoreBackend, embedder: EmbeddingBackend,
                 config: ExcelProcessorConfig): ...
    def process(self, file_path: str, profile: FileProfile,
                ingest_key: str, ingest_run_id: str) -> ProcessingResult: ...
```

### 10.3 Path C — Hybrid Splitter (`processors/splitter.py`)

**Input:** FileProfile + classification confirming Type C, with per-sheet or per-region type info.

**Steps:**
1. For each sheet, detect distinct regions using multiple heuristics:
   - **Blank separators:** A run of >= 2 blank rows or columns marks a boundary.
   - **Merged cell blocks:** Large merged regions indicate header/title blocks.
   - **Formatting transitions:** Shift from numeric-heavy to text-heavy rows suggests a region boundary.
   - **Header/footer detection:** Repeated merged rows at top/bottom of sheets identified as `HEADER_BLOCK` / `FOOTER_BLOCK`.
   - **Matrix detection:** Regions with both row and column headers identified as `MATRIX_BLOCK`.
2. Each region gets a `SheetRegion` with bounding coordinates and `detection_confidence` (0.0-1.0).
3. Classify each region as Type A or Type B (using Tier 1 signals on the region's data).
4. Route each region to `StructuredDBProcessor` or `TextSerializer`.
5. All chunks/tables share the same `ingest_key` and link via `region_id` in metadata.

**Public interface:**
```python
class HybridSplitter:
    def __init__(self, structured_processor: StructuredDBProcessor,
                 text_serializer: TextSerializer, config: ExcelProcessorConfig): ...
    def process(self, file_path: str, profile: FileProfile,
                classification: ClassificationResult,
                ingest_key: str, ingest_run_id: str) -> ProcessingResult: ...
```

---

## 11. Backend Protocols (`protocols.py`)

All protocols use `typing.Protocol` with `runtime_checkable` for optional isinstance checks.

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class VectorStoreBackend(Protocol):
    def upsert_chunks(self, collection: str, chunks: list[ChunkPayload]) -> int: ...
    def ensure_collection(self, collection: str, vector_size: int) -> None: ...
    def create_payload_index(self, collection: str, field: str, field_type: str) -> None: ...
    def delete_by_ids(self, collection: str, ids: list[str]) -> int: ...

@runtime_checkable
class StructuredDBBackend(Protocol):
    def create_table_from_dataframe(self, table_name: str, df: "pd.DataFrame") -> None: ...
    def drop_table(self, table_name: str) -> None: ...
    def table_exists(self, table_name: str) -> bool: ...
    def get_table_schema(self, table_name: str) -> dict: ...
    def get_connection_uri(self) -> str: ...

@runtime_checkable
class LLMBackend(Protocol):
    def classify(self, prompt: str, model: str, temperature: float = 0.1,
                 timeout: float | None = None) -> dict: ...
    def generate(self, prompt: str, model: str, temperature: float = 0.7,
                 timeout: float | None = None) -> str: ...

@runtime_checkable
class EmbeddingBackend(Protocol):
    def embed(self, texts: list[str], timeout: float | None = None) -> list[list[float]]: ...
    def dimension(self) -> int: ...
```

Note: `delete_by_ids` and `drop_table` added to support caller-side rollback via `WrittenArtifacts`.

---

## 12. Concrete Backends

### 12.1 `backends/qdrant.py` — QdrantVectorStore

- Uses `qdrant-client` library.
- `ensure_collection`: creates collection if not exists with given vector size and cosine distance.
- `upsert_chunks`: batches points with payload metadata.
- `create_payload_index`: creates keyword/integer payload indexes for filtered search.
- `delete_by_ids`: deletes points by ID list.
- All operations respect `config.backend_timeout_seconds` and retry with exponential backoff up to `config.backend_max_retries`.

### 12.2 `backends/sqlite.py` — SQLiteStructuredDB

- Uses `sqlite3` + `pandas.DataFrame.to_sql()`.
- `create_table_from_dataframe`: writes DataFrame to SQLite, replacing if exists.
- `drop_table`: drops table by name.
- `table_exists`: checks `sqlite_master`.
- `get_table_schema`: returns `{column_name: type}` via `PRAGMA table_info`.
- `get_connection_uri`: returns `sqlite:///path/to/db`.

### 12.3 `backends/ollama.py` — OllamaLLM + OllamaEmbedding

**OllamaLLM:**
- Uses Ollama HTTP API (`POST /api/generate` or `POST /api/chat`).
- `classify`: sends prompt, parses JSON from response. Validates against `LLMClassificationResponse` schema. Retries once on malformed JSON with correction hint.
- `generate`: sends prompt, returns raw text response.
- All calls use `timeout` parameter (falls back to `config.backend_timeout_seconds`).
- Connection failures raise with `E_LLM_TIMEOUT` or backend-specific error code.

**OllamaEmbedding:**
- Uses `POST /api/embed` (or `/api/embeddings`).
- `embed`: batches texts, returns list of float vectors. Respects timeout.
- `dimension`: returns configured dimension (default 768 for nomic-embed-text).

### 12.4 Stubs — `backends/milvus.py`, `backends/postgres.py`

Each stub:
- Implements the protocol interface.
- Every method raises `NotImplementedError("XxxBackend not yet implemented. See backends/yyy.py for reference.")`.
- Includes a module-level docstring explaining how to implement it.

---

## 13. Router (`router.py`)

The `ExcelRouter` is the top-level orchestrator.

```python
class ExcelRouter:
    def __init__(
        self,
        vector_store: VectorStoreBackend,
        structured_db: StructuredDBBackend,
        llm: LLMBackend,
        embedder: EmbeddingBackend,
        config: ExcelProcessorConfig | None = None,
    ): ...

    def process(self, file_path: str, source_uri: str | None = None) -> ProcessingResult:
        """Classify and process a single Excel file."""
        ...

    def process_batch(self, file_paths: list[str]) -> list[ProcessingResult]:
        """Process multiple files sequentially."""
        ...
```

### 13.1 `process()` Flow

1. Compute `ingest_key` via `compute_ingest_key(file_path, config.parser_version, config.tenant_id)`.
2. Generate `ingest_run_id` (UUID4, unique per invocation).
3. Parse file via `ParserChain.parse()` — returns `FileProfile` with per-sheet `parser_used` and any fallback warnings.
4. Call `ExcelInspector.classify()` (Tier 1).
5. If inconclusive → call `LLMClassifier.classify()` (Tier 2) with schema-validated output.
6. If still low confidence → call `LLMClassifier.classify()` (Tier 3, if enabled).
7. If all tiers fail → return `ProcessingResult` with `E_CLASSIFY_INCONCLUSIVE` (fail-closed).
8. Based on `ClassificationResult.file_type`:
   - `TABULAR_DATA` → `StructuredDBProcessor.process()`
   - `FORMATTED_DOCUMENT` → `TextSerializer.process()`
   - `HYBRID` → `HybridSplitter.process()`
9. Collect `WrittenArtifacts` from processor.
10. Assemble `ProcessingResult` with all stage artifacts.
11. Log the result (PII-safe).
12. Return `ProcessingResult`.

---

## 14. Error Handling

| Scenario | Error Code | Behavior |
|----------|-----------|----------|
| Password-protected sheet | `W_SHEET_SKIPPED_PASSWORD` | Skip sheet, add warning |
| Hidden sheet | — | Process normally (data is accessible) |
| Chart-only sheet | `W_SHEET_SKIPPED_CHART` | Skip sheet, add warning |
| Corrupt/unreadable file | `E_PARSE_CORRUPT` | All parsers tried and failed, return error result |
| Primary parser fails, fallback succeeds | `W_PARSER_FALLBACK` | Continue with reduced fidelity, record fallback reason |
| Excel date integers (e.g., 45292) | — | Auto-detect and convert to `datetime` |
| LLM returns malformed JSON | `E_LLM_MALFORMED_JSON` | Retry once with correction hint |
| LLM output fails schema validation | `E_LLM_SCHEMA_INVALID` | Retry once, then fall back to Tier 1 or fail |
| LLM confidence out of bounds | `E_LLM_CONFIDENCE_OOB` | Clamp to [0.0, 1.0], add warning |
| Backend connection failure | `E_BACKEND_*_CONNECT` | Retry with backoff up to `max_retries`, then raise |
| Backend timeout | `E_BACKEND_*_TIMEOUT` | Retry with backoff up to `max_retries`, then raise |
| Empty file / no data sheets | `E_PARSE_EMPTY` | Return result with error, zero chunks/tables |
| Sheet exceeds `max_rows_in_memory` | `W_ROWS_TRUNCATED` | Skip sheet, add warning |

---

## 15. Logging

Uses Python `logging` module with logger name `ingestkit_excel`.

### 15.1 PII-Safe by Default

**All log levels** are PII-safe by default. No raw file data, sample rows, chunk text, or LLM prompt/response content appears in logs unless explicitly opted in via config flags.

**INFO level** — every processed file:
```
ingestkit_excel | file=benefits.xlsx | ingest_key=a3f8... | tier=rule_based | type=tabular_data | confidence=0.9 | path=sql_agent | chunks=1 | tables=3 | parser=openpyxl | time=2.4s
```

**WARNING level** — fallbacks and non-fatal issues:
```
ingestkit_excel | file=checklist.xlsx | code=W_PARSER_FALLBACK | sheet=Sheet2 | detail=openpyxl failed, using pandas_fallback
```

**ERROR level** — fatal processing failures:
```
ingestkit_excel | file=corrupt.xlsx | code=E_PARSE_CORRUPT | detail=All parsers failed
```

**DEBUG level** (opt-in only):
- Signal breakdowns (`log_sample_data=False` by default)
- LLM prompts/responses (`log_llm_prompts=False` by default)
- Chunk text previews (`log_chunk_previews=False` by default)

If any `config.redact_patterns` are set, all logged text is scrubbed against those regexes before emission.

---

## 16. Public API

### 16.1 Top-Level Exports (`__init__.py`)

```python
from ingestkit_excel.router import ExcelRouter
from ingestkit_excel.config import ExcelProcessorConfig
from ingestkit_excel.models import (
    FileType, ClassificationTier, ClassificationResult, ProcessingResult,
    ChunkPayload, ChunkMetadata, FileProfile, IngestKey, WrittenArtifacts,
    ParseStageResult, ClassificationStageResult, EmbedStageResult,
)
from ingestkit_excel.errors import ErrorCode, IngestError
from ingestkit_excel.protocols import (
    VectorStoreBackend, StructuredDBBackend, LLMBackend, EmbeddingBackend,
)

def create_default_router(**overrides) -> ExcelRouter:
    """Create a router with default backends (Qdrant, SQLite, Ollama)."""
    ...
```

### 16.2 Usage Example

```python
from ingestkit_excel import create_default_router

router = create_default_router()
result = router.process("path/to/file.xlsx")

print(result.classification.file_type)       # FileType.TABULAR_DATA
print(result.classification_result.tier_used) # ClassificationTier.RULE_BASED
print(result.ingest_key)                      # "a3f8c2..."
print(result.tables_created)                  # 3
print(result.chunks_created)                  # 1
print(result.written.db_table_names)          # ["employee_roster", "departments", "locations"]
print(result.errors)                          # []
```

### 16.3 Custom Backend Example

```python
from ingestkit_excel import ExcelRouter, ExcelProcessorConfig
from ingestkit_excel.backends.qdrant import QdrantVectorStore
from ingestkit_excel.backends.sqlite import SQLiteStructuredDB
from ingestkit_excel.backends.ollama import OllamaLLM, OllamaEmbedding

router = ExcelRouter(
    vector_store=QdrantVectorStore(url="localhost", port=6333),
    structured_db=SQLiteStructuredDB(db_path="./helpdesk.db"),
    llm=OllamaLLM(base_url="http://localhost:11434"),
    embedder=OllamaEmbedding(model="nomic-embed-text", base_url="http://localhost:11434"),
    config=ExcelProcessorConfig(
        classification_model="qwen2.5:7b",
        default_collection="helpdesk",
        tenant_id="client_acme",
    ),
)

result = router.process("quarterly_report.xlsx")
```

### 16.4 Rollback Example (caller-side)

```python
result = router.process("file.xlsx")

if result.errors:
    # Caller decides to roll back based on its own policy
    vector_store.delete_by_ids(result.written.vector_collection, result.written.vector_point_ids)
    for table in result.written.db_table_names:
        structured_db.drop_table(table)
```

---

## 17. Testing Strategy

### 17.1 Mock Backends (`conftest.py`)

Provide in-memory mock implementations of all four protocols for unit testing:
- `MockVectorStore` — stores chunks in a list, supports delete_by_ids.
- `MockStructuredDB` — stores DataFrames in a dict, supports drop_table.
- `MockLLM` — returns configurable canned responses. Supports simulating malformed JSON, schema-invalid responses, and timeouts.
- `MockEmbedding` — returns zero vectors of correct dimension.

### 17.2 Test Fixtures (`fixtures/`)

Create minimal `.xlsx` files programmatically in conftest or as static files:
- `type_a_simple.xlsx` — 3 columns, 20 rows, clean tabular data.
- `type_b_checklist.xlsx` — merged header cells, checklist format, text-heavy.
- `type_c_hybrid.xlsx` — sheet 1 is tabular, sheet 2 is document-formatted.
- `edge_empty.xlsx` — empty workbook.
- `edge_chart_only.xlsx` — sheet with only a chart.
- `edge_large.xlsx` — sheet exceeding `max_rows_in_memory` (generated dynamically).

### 17.3 Test Coverage

| Module | Key test cases |
|--------|---------------|
| `idempotency.py` | Same file → same key; different content → different key; different parser version → different key; tenant_id inclusion |
| `parser_chain.py` | Primary success; primary fail + fallback success; all parsers fail; per-sheet fallback independence; fallback_reason_code correctness |
| `inspector.py` | All 5 signals correctly detected; high/medium/inconclusive thresholds; multi-sheet hybrid detection |
| `llm_classifier.py` | Summary generation (no raw data); schema validation pass/fail; malformed JSON retry; confidence bounds clamping; tier escalation; fail-closed after retries |
| `structured_db.py` | Column cleaning; date parsing; schema description generation; row serialization toggle; ChunkMetadata correctness |
| `serializer.py` | Merged cell handling; section detection; checklist/matrix/free-text serialization; ChunkMetadata correctness |
| `splitter.py` | Region detection (blank rows, merged blocks, formatting transitions); per-region confidence; per-region classification and routing |
| `router.py` | Full flow for each type; tier escalation; parser fallback integration; error code propagation; WrittenArtifacts populated; ingest_key in result; PII-safe logging verification |
| `errors.py` | Error code enum completeness; IngestError construction |
| `backends/` | Protocol compliance; timeout/retry behavior; Qdrant/SQLite/Ollama basic operations (integration, marked) |

### 17.4 Markers

- `@pytest.mark.unit` — runs with mocks only, no external services.
- `@pytest.mark.integration` — requires running Qdrant, Ollama, etc.

---

## 18. Dependencies

```toml
[project]
dependencies = [
    "openpyxl>=3.1",
    "pandas>=2.0",
    "pydantic>=2.0",
]

[project.optional-dependencies]
qdrant = ["qdrant-client>=1.7"]
ollama = ["httpx>=0.27"]
postgres = ["psycopg2-binary>=2.9"]
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "pyyaml>=6.0",
]
```

---

## 19. Future Considerations (not in scope now)

- Async versions of all backend methods (`aembed`, `aupsert_chunks`, etc.).
- Streaming large files in chunks instead of loading entire workbook into memory.
- `.xls` (legacy format) support via `xlrd`.
- Shared `ingestkit-core` package extracting the protocols and common models for reuse across `ingestkit-pdf`, `ingestkit-docx`, etc.
- Plugin discovery via entry points so the main RAG app auto-detects installed ingestkit packages.

See `ROADMAP.md` for the full deferred items list with rationale.
