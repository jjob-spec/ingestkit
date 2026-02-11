# ingestkit-pdf — Technical Specification

**Version:** 1.1
**Status:** DRAFT — Revision incorporating engineering review feedback
**Changelog:**
- v1.1: Address engineering review — flip OCR default to Tesseract baseline, add Platform Support Contract, Execution Model, LLM Outage Contract, Phased Delivery Plan, Release Gates, Security Override Governance, Operational SLOs, Versioning Policy, Requirements Traceability.
- v1.0: Initial draft incorporating OSS benchmark findings from Unstructured, Docling, Marker/Surya, MinerU, and PaperQA.

## 1. Overview

**Package:** `ingestkit-pdf`
**Python package name:** `ingestkit_pdf`
**Parent ecosystem:** `ingestkit` — a plugin-based ingestion framework for the "AI Help Desk in a Box" on-premises RAG system.

This package classifies `.pdf` files into one of three structural types and routes each to the appropriate extraction path, producing chunks in a vector store and/or tables in a structured database.

### 1.1 Problem Statement

PDF files in HR and IT environments are the most common document format (employee handbooks, policy manuals, IT runbooks, compliance documents, benefits guides). They are also the hardest to process because PDFs were designed for printing, not data extraction — two visually identical PDFs can have completely different internal structures.

The three failure modes:
- **Scanned pages** have no text layer; extraction requires OCR.
- **Complex layouts** (multi-column, embedded tables, form fields) break naive text extraction.
- **Mixed documents** contain some digital pages, some scanned pages, and some table-heavy pages within the same file.

A single extraction strategy fails on this diversity. This package solves that with tiered classification and type-specific processing at page granularity.

### 1.2 Scope

**In scope:**
- Classification of PDF files into Type A (text-native), Type B (scanned/image), or Type C (complex/hybrid).
- Three-tier detection: rule-based structural analysis → lightweight LLM → reasoning/vision model.
- Three processing paths: text extraction, OCR pipeline, and complex document processing.
- Page-level routing: a single PDF can have pages routed to different paths.
- Swappable OCR engines: Tesseract (required baseline) and PaddleOCR (optional upgrade), selectable via config.
- Extraction quality scoring with automatic OCR fallback.
- Pre-flight security scanning (decompression bombs, embedded JS, malicious content).
- Resource limits and per-document timeout isolation.
- Header/footer detection and stripping via cross-page similarity.
- Heading hierarchy detection (PDF outline extraction + font-based inference).
- Table extraction from PDF pages with multi-page table stitching.
- Chunking with configurable strategy, size, and overlap.
- Document-level metadata extraction from PDF properties.
- Language detection for OCR routing.
- Deterministic idempotency keying for deduplication (reuses `ingestkit-core` model).
- Abstract backend interfaces (Protocols) imported from shared `ingestkit-core`.
- Normalized error taxonomy.
- Standardized chunk/table metadata schema.
- PII-safe structured logging with configurable redaction.
- Programmatic test PDF generation (no binary fixtures committed to repo).

**Out of scope:**
- Query-time retrieval engine or SQL agent.
- Full RAG pipeline (retrieval, reranking, generation).
- Cloud API backends (GPT-4V, Claude vision, OpenAI, Pinecone, etc.).
- Cloud OCR backends (Google Vision, AWS Textract, Azure Document Intelligence).
- Non-PDF file formats (handled by sibling `ingestkit-*` packages).
- PDF creation, modification, or annotation.
- Docling integration as a default path (optional advanced processor users can enable).
- LLM-based OCR cleanup as default (optional flag, not the default path).
- Durable state machines, dead-letter queues, or job orchestration (caller's responsibility).
- Multi-tenant isolation enforcement (package propagates `tenant_id`; caller enforces).

---

## 2. Architecture

### 2.1 Design Principles

1. **Backend-agnostic core.** All processing logic references Protocol types from `ingestkit-core`, never concrete backends.
2. **Dependency injection.** The `PDFRouter` accepts backend instances; it never creates them.
3. **Structural subtyping.** Backends use `typing.Protocol` (not ABCs) — implement the methods and it works.
4. **Fail gracefully.** Corrupt files, encrypted PDFs, garbled text — return structured errors with normalized codes, never crash.
5. **Page-level granularity.** Unlike Excel (sheet-level), PDF processing works at page granularity. A 50-page document might have 45 text-native pages, 3 scanned pages, and 2 table-heavy pages.
6. **Quality-gated extraction.** Every extraction attempt is scored. Low-quality extraction triggers automatic OCR fallback before producing chunks.
7. **PII-safe observability.** Every file processed emits a structured log entry. No raw text, OCR output, or chunk content in logs without explicit opt-in.
8. **Idempotent by default.** Every ingest produces a deterministic key; re-ingesting the same content with the same parser version produces the same key.
9. **Security-first.** Pre-flight validation rejects decompression bombs, oversized files, and suspicious PDF objects before any extraction begins.
10. **OCR engine agnostic with portable baseline.** Tesseract is the required baseline available on all supported platforms. PaddleOCR is an optional upgrade for higher accuracy where platform support allows. The pipeline does not hardcode either.
11. **Resilient classification.** Tier 1 (rule-based) always produces a result with no external dependencies. LLM outage degrades gracefully to Tier 1, never blocks processing.

### 2.2 High-Level Flow

```
.pdf file
    │
    ▼
┌──────────────────────────┐
│  Pre-flight Security Scan │  magic bytes, file size, page count, JS detection,
│                          │  decompression ratio check
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  Compute ingest_key       │  content_hash + source_uri + parser_version
│  (idempotency)            │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  Extract document profile │  PyMuPDF: per-page text, images, fonts, metadata
│  + quality assessment     │  Detect encrypted, garbled, blank, vector-only pages
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  Tier 1: Inspector        │  Rule-based per-page structural analysis
│  (no LLM, no network)    │  Handles ~80% of files — ALWAYS produces a result
└──────────┬───────────────┘
           │ confidence < threshold or mixed signals
           ▼
┌──────────────────────────┐
│  Tier 2: LLM Classifier  │  Structural summary → JSON classification
│  (requires LLM backend)  │  Handles ~15% of files — SKIPPED if LLM unavailable
└──────────┬───────────────┘
           │ confidence < 0.6
           ▼
┌──────────────────────────┐
│  Tier 3: Reasoning Model  │  Reasoning model + optional page image
│  (requires LLM backend)  │  Handles ~5% of files — SKIPPED if LLM unavailable
└──────────────────────────┘
           │
           ▼
   ClassificationResult
   {type, confidence, tier_used, per_page_types}
           │
           ├─ Type A ──▶ Path A: Text extraction (pymupdf4llm → Markdown)
           ├─ Type B ──▶ Path B: OCR pipeline (render → preprocess → OCR → postprocess)
           └─ Type C ──▶ Path C: Complex processing (per-page routing + table extraction)
           │
           ▼
   ProcessingResult
   {stage_artifacts, written_ids, error_codes, ...}
```

### 2.3 Module Structure

```
ingestkit-pdf/
├── pyproject.toml
├── SPEC.md
├── ROADMAP.md
└── src/
    └── ingestkit_pdf/
        ├── __init__.py              # Exports: PDFRouter, create_default_router
        ├── models.py                # Pydantic models for all data structures
        ├── errors.py                # Normalized error codes and error model
        ├── config.py                # PDFProcessorConfig with all thresholds and defaults
        ├── security.py              # Pre-flight security scanning
        ├── quality.py               # Extraction quality scoring
        ├── inspector.py             # Tier 1: rule-based per-page structural analysis
        ├── llm_classifier.py        # Tier 2 & 3: LLM-based classification
        ├── router.py                # PDFRouter: orchestrates detection → processing
        ├── processors/
        │   ├── __init__.py
        │   ├── text_extractor.py    # Path A: text-native PDF → Markdown → chunks
        │   ├── ocr_processor.py     # Path B: scanned PDF → OCR → chunks
        │   ├── complex_processor.py # Path C: hybrid → per-page routing + tables
        │   └── table_extractor.py   # Shared: pdfplumber table detection/extraction
        ├── utils/
        │   ├── __init__.py
        │   ├── header_footer.py     # Header/footer detection and stripping
        │   ├── heading_detector.py  # Heading hierarchy extraction
        │   ├── ocr_engines.py       # OCR engine abstraction (Tesseract, PaddleOCR)
        │   ├── ocr_postprocess.py   # OCR text cleanup
        │   ├── page_renderer.py     # PDF page → image rendering for OCR
        │   ├── layout_analysis.py   # Multi-column detection, reading order
        │   ├── chunker.py           # Configurable text chunking
        │   └── language.py          # Language detection (FastText)
        └── tests/
            ├── conftest.py          # Shared fixtures, mock backends, PDF generators
            ├── test_security.py
            ├── test_quality.py
            ├── test_inspector.py
            ├── test_llm_classifier.py
            ├── test_text_extractor.py
            ├── test_ocr_processor.py
            ├── test_complex_processor.py
            ├── test_table_extractor.py
            ├── test_header_footer.py
            ├── test_chunker.py
            ├── test_router.py
            └── test_utils.py
```

### 2.4 Shared Package: `ingestkit-core`

Backend protocols and common models are extracted to `packages/ingestkit-core/` for reuse across `ingestkit-excel` and `ingestkit-pdf`. Both packages depend on `ingestkit-core`.

Shared types:
- `VectorStoreBackend`, `StructuredDBBackend`, `LLMBackend`, `EmbeddingBackend` (protocols)
- `ChunkPayload`, `ChunkMetadata` (base schema — each package extends with format-specific fields)
- `IngestKey`, `WrittenArtifacts`
- `IngestError`, `ErrorCode` (base codes — each package extends with format-specific codes)

Until `ingestkit-core` is extracted, this package duplicates `protocols.py` from `ingestkit-excel` with a `# TODO: Extract to ingestkit-core` comment at the top.

---

## 3. Platform Support Contract

### 3.1 Supported Targets

| Target | Architecture | Support Level | Notes |
|--------|-------------|---------------|-------|
| Linux x86_64 | amd64 | **Full support** | Primary CI/CD target |
| Linux ARM64 (DGX-spark-class) | aarch64 | **Full support** | GPU-accelerated OCR available |
| WSL2 (Windows laptop) | amd64 | **Best-effort** | No GPU passthrough assumed; CPU-only OCR |

All targets must pass the full `@pytest.mark.unit` suite and the Tesseract-based `@pytest.mark.ocr` subset. PaddleOCR tests are gated behind `@pytest.mark.ocr_paddle` and only required on targets where PaddleOCR is installable.

### 3.2 OCR Engine Policy

| Engine | Status | Availability | Install |
|--------|--------|-------------|---------|
| **Tesseract** | **Required baseline** | All targets | `apt install tesseract-ocr` + `pip install pytesseract` |
| **PaddleOCR** | Optional upgrade | x86_64 + ARM64 with PaddlePaddle support | `pip install ingestkit-pdf[paddleocr]` |

**Default:** `OCREngine.TESSERACT`

**Rationale:** Tesseract is available via system package manager on every supported platform (Ubuntu, Debian, Alpine, RHEL ARM64, WSL2). PaddleOCR delivers higher accuracy (96.6% vs ~89%) but requires PaddlePaddle, which has platform-specific wheel availability and optional CUDA dependencies that may not be present on all targets.

### 3.3 Engine Unavailability Behavior

When the configured OCR engine is unavailable at runtime:

1. If `ocr_engine=PADDLEOCR` and PaddleOCR is not installed:
   - Attempt automatic fallback to Tesseract.
   - If Tesseract is available: proceed with Tesseract, emit `W_OCR_ENGINE_FALLBACK`.
   - If Tesseract is also unavailable: fail with `E_OCR_ENGINE_UNAVAILABLE`.
2. If `ocr_engine=TESSERACT` and Tesseract is not installed:
   - Fail with `E_OCR_ENGINE_UNAVAILABLE` (no fallback — baseline must be present).
3. Engine availability is checked once at `PDFRouter.__init__()` time, not per-document.

### 3.4 Platform Dependency Matrix

| Dependency | x86_64 Linux | ARM64 Linux | WSL2 |
|-----------|-------------|-------------|------|
| PyMuPDF | pip (wheel) | pip (wheel) | pip (wheel) |
| pymupdf4llm | pip | pip | pip |
| pdfplumber | pip | pip | pip |
| Tesseract binary | `apt install tesseract-ocr` | `apt install tesseract-ocr` | `apt install tesseract-ocr` |
| pytesseract | pip | pip | pip |
| PaddlePaddle | pip (CPU/GPU wheel) | pip (CPU wheel, limited GPU) | pip (CPU only) |
| PaddleOCR | pip | pip (CPU only on most ARM64) | pip (CPU only) |
| OpenCV headless | pip (wheel) | pip (wheel) | pip (wheel) |
| fast-langdetect | pip | pip | pip |

**Known limitations:**
- PaddleOCR GPU acceleration requires CUDA toolkit; not available on WSL2 without GPU passthrough.
- ARM64 PaddlePaddle wheels may lag upstream releases by 1-2 versions.
- Tesseract language packs require separate install (`apt install tesseract-ocr-<lang>`).

### 3.5 CI Matrix

| CI Job | Target | OCR Engine | Markers |
|--------|--------|-----------|---------|
| `test-unit` | x86_64 | None (mocked) | `@pytest.mark.unit` |
| `test-ocr-tesseract` | x86_64 | Tesseract | `@pytest.mark.ocr` |
| `test-ocr-paddle` | x86_64 | PaddleOCR | `@pytest.mark.ocr_paddle` |
| `test-arm64` | ARM64 | Tesseract | `@pytest.mark.unit`, `@pytest.mark.ocr` |
| `lint-typecheck` | x86_64 | N/A | ruff + mypy |

---

## 4. Data Models (`models.py`)

All models use Pydantic v2 (`BaseModel`).

### 4.1 Enumerations

```python
class PDFType(str, Enum):
    TEXT_NATIVE = "text_native"      # Type A
    SCANNED = "scanned"              # Type B
    COMPLEX = "complex"              # Type C

class PageType(str, Enum):
    TEXT = "text"                     # Page has extractable digital text
    SCANNED = "scanned"              # Page is an image, no text layer
    TABLE_HEAVY = "table_heavy"      # Page contains significant table content
    FORM = "form"                    # Page contains interactive form fields
    MIXED = "mixed"                  # Page has both text and scanned regions
    BLANK = "blank"                  # Empty or near-empty page
    VECTOR_ONLY = "vector_only"      # Only vector graphics, no text or images
    TOC = "toc"                      # Table of contents page

class ClassificationTier(str, Enum):
    RULE_BASED = "rule_based"        # Tier 1
    LLM_BASIC = "llm_basic"         # Tier 2
    LLM_REASONING = "llm_reasoning"  # Tier 3

class IngestionMethod(str, Enum):
    TEXT_EXTRACTION = "text_extraction"           # Path A
    OCR_PIPELINE = "ocr_pipeline"                 # Path B
    COMPLEX_PROCESSING = "complex_processing"     # Path C

class OCREngine(str, Enum):
    TESSERACT = "tesseract"
    PADDLEOCR = "paddleocr"

class ExtractionQualityGrade(str, Enum):
    HIGH = "high"       # score >= 0.9
    MEDIUM = "medium"   # score >= 0.6
    LOW = "low"         # score < 0.6

class ContentType(str, Enum):
    """Content type within a page or chunk."""
    NARRATIVE = "narrative"
    TABLE = "table"
    LIST = "list"
    HEADING = "heading"
    FORM_FIELD = "form_field"
    IMAGE_DESCRIPTION = "image_description"
    FOOTER = "footer"
    HEADER = "header"
```

### 4.2 Per-Page Models

```python
class PageProfile(BaseModel):
    """Structural profile of a single PDF page."""
    page_number: int                      # 0-indexed
    text_length: int                      # character count from get_text()
    word_count: int
    image_count: int
    image_coverage_ratio: float           # 0.0-1.0, fraction of page area covered by images
    table_count: int                      # tables detected by pdfplumber
    font_count: int                       # unique fonts on this page
    font_names: list[str]
    has_form_fields: bool
    is_multi_column: bool                 # detected via text block x-coordinate clustering
    page_type: PageType                   # classified page type
    extraction_quality: "ExtractionQuality"

class ExtractionQuality(BaseModel):
    """Quality assessment of text extraction on a page or document."""
    printable_ratio: float                # fraction of printable chars (0.0-1.0)
    avg_words_per_page: float
    pages_with_text: int
    total_pages: int
    extraction_method: str                # "native", "ocr", "ocr_fallback", "repaired"

    @property
    def score(self) -> float:
        """Composite quality score 0.0-1.0."""
        coverage = self.pages_with_text / max(self.total_pages, 1)
        text_quality = min(self.printable_ratio, 1.0)
        density = min(self.avg_words_per_page / 100, 1.0)
        return coverage * 0.4 + text_quality * 0.4 + density * 0.2

    @property
    def grade(self) -> ExtractionQualityGrade:
        s = self.score
        if s >= 0.9:
            return ExtractionQualityGrade.HIGH
        if s >= 0.6:
            return ExtractionQualityGrade.MEDIUM
        return ExtractionQualityGrade.LOW
```

### 4.3 Document-Level Models

```python
class DocumentMetadata(BaseModel):
    """Metadata extracted from PDF document properties."""
    title: str | None = None
    author: str | None = None
    subject: str | None = None
    keywords: str | None = None
    creator: str | None = None             # e.g. "Microsoft Word", "LaTeX"
    producer: str | None = None            # PDF producer software
    creation_date: str | None = None
    modification_date: str | None = None
    pdf_version: str | None = None
    page_count: int = 0
    file_size_bytes: int = 0
    is_encrypted: bool = False
    needs_password: bool = False
    is_signed: bool = False
    has_form_fields: bool = False
    is_linearized: bool = False

class DocumentProfile(BaseModel):
    """Aggregate structural profile of a PDF file."""
    file_path: str
    file_size_bytes: int
    page_count: int
    content_hash: str                      # SHA-256 of file bytes
    metadata: DocumentMetadata
    pages: list[PageProfile]
    page_type_distribution: dict[str, int] # e.g. {"text": 45, "scanned": 3, "table_heavy": 2}
    detected_languages: list[str]          # ISO 639-1 codes
    has_toc: bool                          # PDF has a table of contents / outline
    toc_entries: list[tuple[int, str, int]] | None = None  # [(level, title, page_num), ...]
    overall_quality: ExtractionQuality
    security_warnings: list[str]

class ClassificationResult(BaseModel):
    """Result of the tiered classification."""
    pdf_type: PDFType
    confidence: float                      # 0.0-1.0
    tier_used: ClassificationTier
    reasoning: str
    per_page_types: dict[int, PageType]    # {page_number: PageType}
    signals: dict[str, Any] | None = None
    degraded: bool = False                 # True if LLM tiers were skipped due to outage
```

### 4.4 Processing Models

```python
class OCRResult(BaseModel):
    """Per-page OCR extraction result."""
    page_number: int
    text: str
    confidence: float                      # average word/line confidence 0.0-1.0
    engine_used: OCREngine
    dpi: int
    preprocessing_steps: list[str]         # e.g. ["deskew", "binarize", "denoise"]
    language_detected: str | None = None

class TableResult(BaseModel):
    """Extracted table from a PDF page."""
    page_number: int
    table_index: int                       # 0-based index on the page
    row_count: int
    col_count: int
    headers: list[str] | None = None
    is_continuation: bool = False          # part of a multi-page table
    continuation_group_id: str | None = None

class PDFChunkMetadata(BaseModel):
    """Standardized metadata attached to every chunk. Extends base ChunkMetadata."""
    source_uri: str
    source_format: str = "pdf"
    page_numbers: list[int]                # pages this chunk spans
    ingestion_method: str                  # IngestionMethod value
    parser_version: str
    chunk_index: int
    chunk_hash: str                        # SHA-256 of chunk text
    ingest_key: str
    ingest_run_id: str
    tenant_id: str | None = None
    # Structural metadata
    heading_path: list[str] | None = None  # e.g. ["Chapter 2", "Section 2.1"]
    content_type: str | None = None        # ContentType value
    section_title: str | None = None
    # Document metadata (propagated)
    doc_title: str | None = None
    doc_author: str | None = None
    doc_date: str | None = None
    # OCR-specific (Path B)
    ocr_engine: str | None = None
    ocr_confidence: float | None = None
    ocr_dpi: int | None = None
    ocr_preprocessing: list[str] | None = None
    # Table-specific (Path C)
    table_name: str | None = None
    table_index: int | None = None
    row_count: int | None = None
    columns: list[str] | None = None
    # Language
    language: str | None = None

class ChunkPayload(BaseModel):
    """A single chunk ready for vector store upsert."""
    id: str
    text: str
    vector: list[float]
    metadata: PDFChunkMetadata
```

### 4.5 Stage Artifacts

```python
class ParseStageResult(BaseModel):
    """Typed output of the PDF extraction stage."""
    pages_extracted: int
    pages_skipped: int
    skipped_reasons: dict[int, str]        # {page_number: reason_code}
    extraction_method: str                 # "pymupdf", "ocr_fallback"
    overall_quality: ExtractionQuality
    parse_duration_seconds: float

class ClassificationStageResult(BaseModel):
    """Typed output of the classification stage."""
    tier_used: ClassificationTier
    pdf_type: PDFType
    confidence: float
    signals: dict[str, Any] | None = None
    reasoning: str
    per_page_types: dict[int, PageType]
    classification_duration_seconds: float
    degraded: bool = False                 # True if LLM tiers were unavailable

class OCRStageResult(BaseModel):
    """Typed output of the OCR stage (Path B and C)."""
    pages_ocrd: int
    engine_used: OCREngine
    avg_confidence: float
    low_confidence_pages: list[int]        # pages below ocr_confidence_threshold
    ocr_duration_seconds: float
    engine_fallback_used: bool = False     # True if fell back from PaddleOCR to Tesseract

class EmbedStageResult(BaseModel):
    """Typed output of the embedding stage."""
    texts_embedded: int
    embedding_dimension: int
    embed_duration_seconds: float

class WrittenArtifacts(BaseModel):
    """IDs of everything written to backends, enabling caller-side rollback."""
    vector_point_ids: list[str] = []
    vector_collection: str | None = None
    db_table_names: list[str] = []

class ProcessingResult(BaseModel):
    """Final result returned after processing a file."""
    file_path: str
    ingest_key: str
    ingest_run_id: str
    tenant_id: str | None = None

    # Stage artifacts
    parse_result: ParseStageResult
    classification_result: ClassificationStageResult
    ocr_result: OCRStageResult | None = None
    embed_result: EmbedStageResult | None = None

    # Legacy convenience
    classification: ClassificationResult
    ingestion_method: IngestionMethod

    # Outputs
    chunks_created: int
    tables_created: int
    tables: list[str]
    written: WrittenArtifacts

    # Errors and warnings
    errors: list[str]
    warnings: list[str]
    error_details: list["IngestError"] = []

    processing_time_seconds: float
```

---

## 5. Error Taxonomy (`errors.py`)

### 5.1 Normalized Error Codes

```python
class ErrorCode(str, Enum):
    """Normalized error codes for the ingestkit-pdf pipeline."""

    # Pre-flight / Security errors
    E_SECURITY_INVALID_PDF = "E_SECURITY_INVALID_PDF"          # not a valid PDF (bad magic bytes)
    E_SECURITY_DECOMPRESSION_BOMB = "E_SECURITY_DECOMPRESSION_BOMB"
    E_SECURITY_JAVASCRIPT = "E_SECURITY_JAVASCRIPT"            # embedded JS detected
    E_SECURITY_TOO_LARGE = "E_SECURITY_TOO_LARGE"              # exceeds max_file_size_mb
    E_SECURITY_TOO_MANY_PAGES = "E_SECURITY_TOO_MANY_PAGES"    # exceeds max_page_count

    # Parse / Extraction errors
    E_PARSE_CORRUPT = "E_PARSE_CORRUPT"                        # PyMuPDF cannot open
    E_PARSE_PASSWORD = "E_PARSE_PASSWORD"                      # requires password to open
    E_PARSE_EMPTY = "E_PARSE_EMPTY"                            # zero pages
    E_PARSE_GARBLED = "E_PARSE_GARBLED"                        # CIDFont/encoding failure detected
    E_PARSE_REPAIR_FAILED = "E_PARSE_REPAIR_FAILED"            # repair attempt also failed

    # OCR errors
    E_OCR_ENGINE_UNAVAILABLE = "E_OCR_ENGINE_UNAVAILABLE"      # no OCR engine installed (baseline missing)
    E_OCR_TIMEOUT = "E_OCR_TIMEOUT"                            # OCR exceeded per-page timeout
    E_OCR_FAILED = "E_OCR_FAILED"                              # OCR produced no output

    # Classification errors
    E_CLASSIFY_INCONCLUSIVE = "E_CLASSIFY_INCONCLUSIVE"        # Tier 1 genuinely cannot decide
    E_LLM_TIMEOUT = "E_LLM_TIMEOUT"
    E_LLM_MALFORMED_JSON = "E_LLM_MALFORMED_JSON"
    E_LLM_SCHEMA_INVALID = "E_LLM_SCHEMA_INVALID"
    E_LLM_CONFIDENCE_OOB = "E_LLM_CONFIDENCE_OOB"

    # Backend errors
    E_BACKEND_VECTOR_TIMEOUT = "E_BACKEND_VECTOR_TIMEOUT"
    E_BACKEND_VECTOR_CONNECT = "E_BACKEND_VECTOR_CONNECT"
    E_BACKEND_DB_TIMEOUT = "E_BACKEND_DB_TIMEOUT"
    E_BACKEND_DB_CONNECT = "E_BACKEND_DB_CONNECT"
    E_BACKEND_EMBED_TIMEOUT = "E_BACKEND_EMBED_TIMEOUT"
    E_BACKEND_EMBED_CONNECT = "E_BACKEND_EMBED_CONNECT"

    # Processing errors
    E_PROCESS_TABLE_EXTRACT = "E_PROCESS_TABLE_EXTRACT"
    E_PROCESS_CHUNK = "E_PROCESS_CHUNK"
    E_PROCESS_HEADER_FOOTER = "E_PROCESS_HEADER_FOOTER"

    # Warnings (non-fatal)
    W_PAGE_SKIPPED_BLANK = "W_PAGE_SKIPPED_BLANK"
    W_PAGE_SKIPPED_TOC = "W_PAGE_SKIPPED_TOC"
    W_PAGE_SKIPPED_VECTOR_ONLY = "W_PAGE_SKIPPED_VECTOR_ONLY"
    W_PAGE_LOW_OCR_CONFIDENCE = "W_PAGE_LOW_OCR_CONFIDENCE"
    W_QUALITY_LOW_NATIVE = "W_QUALITY_LOW_NATIVE"              # native extraction quality LOW, falling back to OCR
    W_OCR_FALLBACK = "W_OCR_FALLBACK"                          # garbled text detected, OCR fallback used
    W_OCR_ENGINE_FALLBACK = "W_OCR_ENGINE_FALLBACK"            # configured engine unavailable, fell back to baseline
    W_TABLE_CONTINUATION = "W_TABLE_CONTINUATION"              # multi-page table stitched
    W_ENCRYPTED_OWNER_ONLY = "W_ENCRYPTED_OWNER_ONLY"          # owner-password only, read OK
    W_DOCUMENT_SIGNED = "W_DOCUMENT_SIGNED"                    # digitally signed, read-only extraction
    W_EMBEDDED_FILES = "W_EMBEDDED_FILES"                      # embedded files detected but not processed
    W_LLM_RETRY = "W_LLM_RETRY"
    W_LLM_UNAVAILABLE = "W_LLM_UNAVAILABLE"                   # LLM backend unreachable, Tier 2/3 skipped
    W_CLASSIFICATION_DEGRADED = "W_CLASSIFICATION_DEGRADED"    # classification used Tier 1 only due to LLM outage
    W_SECURITY_OVERRIDE = "W_SECURITY_OVERRIDE"                # a security default was overridden by config

class IngestError(BaseModel):
    """Structured error with code, message, and context."""
    code: ErrorCode
    message: str
    page_number: int | None = None
    stage: str | None = None       # "security", "parse", "ocr", "classify", "process", "embed"
    recoverable: bool = False
```

### 5.2 Fail-Closed Default and LLM Outage Contract

**Fail-closed:**
- If security scan fails → reject the file entirely, no partial processing.
- If native extraction quality is LOW and OCR also fails → flag pages with `W_PAGE_LOW_OCR_CONFIDENCE`, never produce garbage chunks.

**LLM outage behavior (testable contract):**

Tier 1 (rule-based inspector) has **zero external dependencies** — no LLM, no network, no GPU. It runs entirely on the `DocumentProfile` already extracted by PyMuPDF. Therefore:

| Scenario | Behavior | Codes Emitted | Confidence Annotation |
|----------|----------|---------------|----------------------|
| LLM backend healthy | Normal Tier 1 → 2 → 3 escalation | (none) | From highest tier used |
| LLM backend unreachable (connect/timeout) | Skip Tier 2/3, use Tier 1 result | `W_LLM_UNAVAILABLE` + `W_CLASSIFICATION_DEGRADED` | Tier 1 confidence, `degraded=True` |
| LLM returns malformed JSON (2 retries exhausted) | Use Tier 1 result | `E_LLM_MALFORMED_JSON` + `W_CLASSIFICATION_DEGRADED` | Tier 1 confidence, `degraded=True` |
| Tier 1 genuinely ambiguous (< 2 agreeing signals) AND LLM unavailable | Use Tier 1 best-guess with low confidence | `W_LLM_UNAVAILABLE` + `W_CLASSIFICATION_DEGRADED` | Tier 1 confidence (will be low) |
| Tier 1 genuinely ambiguous AND LLM available but all tiers fail | Return `E_CLASSIFY_INCONCLUSIVE`, zero chunks | `E_CLASSIFY_INCONCLUSIVE` | 0.0 |

**Key invariant:** `E_CLASSIFY_INCONCLUSIVE` (zero chunks produced) is ONLY emitted when Tier 1 cannot produce any result AND all LLM tiers also fail. LLM outage alone never causes zero-chunk results.

**Test contract for LLM outage:**
```python
def test_llm_outage_degrades_to_tier1(mock_llm_raising_connection_error):
    """LLM outage → Tier 1 result used with degraded=True."""
    result = router.process("text_native.pdf")
    assert result.classification.tier_used == ClassificationTier.RULE_BASED
    assert result.classification.degraded is True
    assert "W_LLM_UNAVAILABLE" in result.warnings
    assert "W_CLASSIFICATION_DEGRADED" in result.warnings
    assert result.chunks_created > 0  # processing still happens
    assert "E_CLASSIFY_INCONCLUSIVE" not in result.errors
```

---

## 6. Configuration (`config.py`)

```python
class PDFProcessorConfig(BaseModel):
    """All tunable parameters with sensible defaults."""

    # --- Identity ---
    parser_version: str = "ingestkit_pdf:1.0.0"
    tenant_id: str | None = None

    # --- Security / Resource Limits ---
    max_file_size_mb: int = 500
    max_page_count: int = 5000
    per_document_timeout_seconds: int = 300       # hard timeout per PDF
    max_decompression_ratio: int = 100            # reject if ratio exceeds this
    reject_javascript: bool = True                # reject PDFs with embedded JS

    # --- Security Override Governance ---
    # Each security override must include an explicit reason string.
    # Overrides are logged at WARNING level with W_SECURITY_OVERRIDE.
    reject_javascript_override_reason: str | None = None   # if set, allows JS PDFs
    max_file_size_override_reason: str | None = None       # if set, allows larger files
    max_page_count_override_reason: str | None = None      # if set, allows more pages

    # --- Tier 1 Thresholds ---
    min_chars_per_page: int = 200                 # below this → likely scanned
    max_image_coverage_for_text: float = 0.3      # above this → likely scanned
    min_table_count_for_complex: int = 1          # tables on page → leans complex
    min_font_count_for_digital: int = 1           # 0 fonts → scanned
    tier1_high_confidence_signals: int = 4        # out of 5 signals
    tier1_medium_confidence_signals: int = 3

    # --- Tier 2/3 LLM Settings ---
    classification_model: str = "qwen2.5:7b"
    reasoning_model: str = "deepseek-r1:14b"
    tier2_confidence_threshold: float = 0.6
    llm_temperature: float = 0.1
    enable_tier3: bool = True

    # --- OCR Settings ---
    ocr_engine: OCREngine = OCREngine.TESSERACT   # default: portable baseline
    ocr_dpi: int = 300
    ocr_language: str = "en"                      # ISO 639-1 or engine-specific code
    ocr_confidence_threshold: float = 0.7         # pages below this flagged for review
    ocr_preprocessing_steps: list[str] = ["deskew"]  # available: deskew, denoise, binarize, contrast
    ocr_max_workers: int = 4                      # parallel OCR workers (ProcessPoolExecutor)
    ocr_per_page_timeout_seconds: int = 60
    enable_ocr_cleanup: bool = False              # if True, use LLM to clean OCR output
    ocr_cleanup_model: str = "qwen2.5:7b"

    # --- Extraction Quality ---
    quality_min_printable_ratio: float = 0.85     # below this → garbled text detected
    quality_min_words_per_page: int = 10          # below this → likely extraction failure
    auto_ocr_fallback: bool = True                # if True, auto-fallback to OCR on LOW quality

    # --- Header/Footer Detection ---
    header_footer_sample_pages: int = 5           # pages to sample for pattern detection
    header_footer_zone_ratio: float = 0.10        # top/bottom 10% of page
    header_footer_similarity_threshold: float = 0.7  # cross-page similarity threshold

    # --- Heading Detection ---
    heading_min_font_size_ratio: float = 1.2      # font must be 1.2x body size to be a heading

    # --- Table Extraction ---
    table_max_rows_for_serialization: int = 20    # tables <= this: serialize to NL sentences
    table_min_rows_for_db: int = 20               # tables > this: load into StructuredDB
    table_continuation_column_match_threshold: float = 0.8  # for multi-page stitching

    # --- Chunking ---
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 50                # ~10% overlap
    chunk_respect_headings: bool = True           # never split across heading boundaries
    chunk_respect_tables: bool = True             # never split mid-table

    # --- Embedding ---
    embedding_model: str = "nomic-embed-text"
    embedding_dimension: int = 768
    embedding_batch_size: int = 64

    # --- Vector Store ---
    default_collection: str = "helpdesk"

    # --- Language Detection ---
    enable_language_detection: bool = True         # detect language per page for OCR routing
    default_language: str = "en"

    # --- Deduplication ---
    enable_content_dedup: bool = True              # skip files with matching content_hash

    # --- Backend Resilience ---
    backend_timeout_seconds: float = 30.0
    backend_max_retries: int = 2
    backend_backoff_base: float = 1.0

    # --- Logging / PII Safety ---
    log_sample_text: bool = False
    log_llm_prompts: bool = False
    log_chunk_previews: bool = False
    log_ocr_output: bool = False
    redact_patterns: list[str] = []
```

Supports loading from YAML or JSON:
```python
@classmethod
def from_file(cls, path: str) -> "PDFProcessorConfig":
    ...
```

### 6.1 Security Override Semantics

Security defaults (`reject_javascript`, `max_file_size_mb`, `max_page_count`) are strict by design. Overriding any default requires an explicit reason string in the corresponding `*_override_reason` config field. See §7.5 for full governance rules.

---

## 7. Pre-Flight Security Scan (`security.py`)

### 7.1 Purpose

Reject dangerous or oversized PDFs before any extraction. This is critical for compliance-sensitive deployments.

### 7.2 Checks

| Check | Threshold | Error Code |
|-------|-----------|------------|
| Magic bytes | Must start with `%PDF-` | `E_SECURITY_INVALID_PDF` |
| File size | `max_file_size_mb` (default 500MB) | `E_SECURITY_TOO_LARGE` |
| Page count | `max_page_count` (default 5000) | `E_SECURITY_TOO_MANY_PAGES` |
| Embedded JavaScript | Present and `reject_javascript=True` | `E_SECURITY_JAVASCRIPT` |
| Excessive xref objects | > 100,000 objects | `E_SECURITY_DECOMPRESSION_BOMB` |
| Encryption | `needs_pass=True` and no password provided | `E_PARSE_PASSWORD` |

### 7.3 Additional Metadata Collected

During the security scan, also detect and record:
- `is_signed` — presence of `/Sig` form fields (warning: `W_DOCUMENT_SIGNED`)
- `has_embedded_files` — `doc.embfile_count() > 0` (warning: `W_EMBEDDED_FILES`)
- `is_encrypted` with empty user password — try `doc.authenticate("")` before rejecting

### 7.4 Public Interface

```python
class PDFSecurityScanner:
    def __init__(self, config: PDFProcessorConfig): ...
    def scan(self, file_path: str) -> tuple[DocumentMetadata, list[IngestError]]:
        """Run pre-flight checks. Returns metadata and any fatal/warning errors."""
        ...
```

### 7.5 Security Override Governance

Security defaults exist to prevent processing of dangerous or adversarial content. Overriding them is permitted but governed.

**Override mechanism:**

Each overridable security setting has a corresponding `*_override_reason` field in `PDFProcessorConfig`. Setting a non-None reason string activates the override:

```python
config = PDFProcessorConfig(
    reject_javascript=False,
    reject_javascript_override_reason="TICKET-4521: Legacy HR forms require JS for calculation fields",
    max_file_size_mb=1000,
    max_file_size_override_reason="TICKET-4530: Annual compliance bundle exceeds 500MB",
)
```

**Override rules:**

| Rule | Enforcement |
|------|------------|
| Override requires a reason string | Config validation: if `reject_javascript=False` but `reject_javascript_override_reason is None` → `ValidationError` |
| Overrides are logged | `W_SECURITY_OVERRIDE` emitted at WARNING level with reason string for every file processed under override |
| Reason should reference a ticket or approval | Convention, not enforced programmatically |
| Overrides are config-scoped | Per-deployment config file, not per-document. Apply the same override to all files processed with that config |

**Audit log entry (emitted per file under override):**
```
ingestkit_pdf | SECURITY_OVERRIDE | file=form.pdf | override=reject_javascript | reason="TICKET-4521: Legacy HR forms require JS" | approved_by=config
```

**Environment scoping:**

Use separate config files per environment to control override scope:
```
config/
├── production.yaml      # strict defaults, no overrides
├── staging.yaml         # may have overrides for testing
└── migration.yaml       # temporary overrides for legacy content migration
```

---

## 8. Extraction Quality Scoring (`quality.py`)

### 8.1 Purpose

Assess the quality of text extraction before producing chunks. Low-quality extraction (garbled text from CIDFont issues, empty pages from scanned content) triggers automatic OCR fallback.

### 8.2 Quality Signals

| Signal | Computation | Threshold |
|--------|-------------|-----------|
| Printable character ratio | `printable_chars / total_chars` | < 0.85 → LOW |
| Words per page | `total_words / pages_with_text` | < 10 → LOW |
| Page coverage | `pages_with_text / total_pages` | < 0.5 → LOW |

### 8.3 Composite Score

```
score = (coverage * 0.4) + (printable_ratio * 0.4) + (density * 0.2)
```

Where `density = min(avg_words_per_page / 100, 1.0)`.

### 8.4 Quality Gate

If `auto_ocr_fallback=True` and grade is LOW:
1. Re-extract page using OCR.
2. Re-assess quality of OCR output.
3. If OCR quality is also LOW → keep original with `W_PAGE_LOW_OCR_CONFIDENCE`, do not produce chunk.

### 8.5 Public Interface

```python
class QualityAssessor:
    def __init__(self, config: PDFProcessorConfig): ...
    def assess_page(self, page_text: str, page_number: int) -> ExtractionQuality: ...
    def assess_document(self, page_qualities: list[ExtractionQuality]) -> ExtractionQuality: ...
    def needs_ocr_fallback(self, quality: ExtractionQuality) -> bool: ...
```

---

## 9. Tier 1 — Rule-Based Inspector (`inspector.py`)

### 9.1 Purpose

Classify pages without any LLM call. Fast, deterministic, handles ~80% of files. **This tier has zero external dependencies and always produces a result.** It is the foundation of the LLM outage resilience contract (§5.2).

### 9.2 Signals (Per Page)

| # | Signal | Type A (text-native) | Type B (scanned) | Type C (complex) |
|---|--------|---------------------|-------------------|------------------|
| 1 | Text chars per page | >= `min_chars_per_page` (200) | < 50 chars | Variable |
| 2 | Image coverage ratio | < `max_image_coverage_for_text` (0.3) | > 0.7 | Variable |
| 3 | Font count | >= `min_font_count_for_digital` (1) | 0 or near-0 | Variable |
| 4 | Table count | 0 | 0 | >= `min_table_count_for_complex` (1) |
| 5 | Multi-column layout | No | No | Yes |
| 6 | Form fields | No | No | Yes |
| 7 | Page consistency | All pages same type | All pages same type | Mixed page types |

### 9.3 Decision Logic

Per page:
- chars >= 200, low image coverage, fonts present, no tables → **text-native** (high confidence)
- chars < 50, high image coverage, 0 fonts → **scanned** (high confidence)
- tables detected, multi-column, form fields, or mixed signals → **complex** (high confidence)
- Borderline → escalate to Tier 2

Per document:
- If all pages agree on one type → classify document as that type.
- If pages disagree → classify as **complex**, record per-page types.

### 9.4 Public Interface

```python
class PDFInspector:
    def __init__(self, config: PDFProcessorConfig): ...
    def classify(self, profile: DocumentProfile) -> ClassificationResult: ...
```

---

## 10. Tier 2 & 3 — LLM Classifier (`llm_classifier.py`)

### 10.1 Structural Summary Generation

Generate a summary of the document structure. **Never send raw text content.** Summary includes:

```
File: employee_handbook.pdf
Pages: 87
File size: 4.2 MB
Creator: Microsoft Word
PDF version: 1.7

Page type distribution:
  text: 80, scanned: 3, table_heavy: 4

Sample page profiles:
  Page 1: 3,200 chars, 0 images, 3 fonts [Arial, Arial-Bold, TimesNewRoman], 0 tables
  Page 12: 1,800 chars, 2 images (coverage: 0.45), 3 fonts, 2 tables (8 rows, 3 cols)
  Page 50: 42 chars, 1 image (coverage: 0.95), 0 fonts, 0 tables

Detected languages: [en]
Has TOC: yes (15 entries)
Has form fields: no
```

### 10.2 Classification Prompt

```
You are classifying a PDF file for a document ingestion system.
Based on the structural summary below, classify this file as one of:

- "text_native": Digital PDF with extractable text. Clean paragraphs and headings. Suitable for direct text extraction.
- "scanned": Pages are images with no text layer. Requires OCR to extract text.
- "complex": Mix of text, tables, multi-column layouts, form fields, or mixed scanned/digital pages. Requires page-level routing.

Respond with JSON only:
{
  "type": "text_native" | "scanned" | "complex",
  "confidence": <float 0.0-1.0>,
  "reasoning": "brief explanation",
  "page_types": [{"page": 1, "type": "text"}, {"page": 12, "type": "table_heavy"}, ...]
}

Structural summary:
{summary}
```

### 10.3 Schema Validation

Same pattern as `ingestkit-excel`: validate LLM JSON against a strict Pydantic model. Retry once on malformed JSON with correction hint. After 2 failed attempts, fall back to Tier 1 result or fail with `E_LLM_SCHEMA_INVALID`.

### 10.4 Tier Escalation

- **Tier 2:** Uses `config.classification_model` (default: `qwen2.5:7b`).
- **Tier 3:** Triggered when Tier 2 confidence < 0.6. Uses `config.reasoning_model`. Can optionally render a sample page to an image and send to a vision-capable model. Disabled via `config.enable_tier3 = False`.

### 10.5 Public Interface

```python
class PDFLLMClassifier:
    def __init__(self, llm: LLMBackend, config: PDFProcessorConfig): ...
    def classify(self, profile: DocumentProfile, tier: ClassificationTier) -> ClassificationResult: ...
```

### 10.6 LLM Outage Handling

The `PDFRouter` orchestrates classification tiers and implements the outage contract defined in §5.2:

```python
def _classify(self, profile: DocumentProfile) -> ClassificationResult:
    """Tiered classification with LLM outage resilience."""
    # Tier 1: always runs, always produces a result
    tier1_result = self._inspector.classify(profile)

    if tier1_result.confidence >= self._config.tier1_high_confidence_signals / 5:
        return tier1_result  # high confidence, no LLM needed

    # Tier 2: attempt LLM, absorb failures
    try:
        tier2_result = self._llm_classifier.classify(profile, ClassificationTier.LLM_BASIC)
        if tier2_result.confidence >= self._config.tier2_confidence_threshold:
            return tier2_result
        # Tier 3 (if enabled and Tier 2 was low confidence)
        if self._config.enable_tier3:
            return self._llm_classifier.classify(profile, ClassificationTier.LLM_REASONING)
        return tier2_result
    except (ConnectionError, TimeoutError, Exception) as exc:
        # LLM unavailable: degrade to Tier 1
        logger.warning("LLM unavailable (%s), degrading to Tier 1", exc)
        tier1_result.degraded = True
        return tier1_result  # warnings emitted by caller
```

**Testable assertions:**
1. `mock_llm(raises=ConnectionError)` → result has `degraded=True`, `tier_used=RULE_BASED`, warnings contain `W_LLM_UNAVAILABLE` and `W_CLASSIFICATION_DEGRADED`.
2. `mock_llm(raises=TimeoutError)` → same behavior.
3. `mock_llm(returns=malformed_json, retries_exhausted=True)` → same degraded behavior.
4. `mock_llm(healthy=True)` → normal escalation, `degraded=False`.

---

## 11. Processing Paths

### 11.1 Path A — Text Extractor (`processors/text_extractor.py`)

**Input:** DocumentProfile + classification confirming Type A (text-native).

This is the fast path. Most HR/IT documents land here.

**Steps:**
1. Extract text using `pymupdf4llm.to_markdown()` with `header=False, footer=False` for built-in header/footer suppression.
2. If markdown quality is poor (quality score LOW), fall back to `page.get_text("blocks")` for block-level extraction with position data.
3. Run header/footer stripping (cross-page similarity algorithm from `utils/header_footer.py`) as a second pass.
4. Detect and skip TOC pages (high density of `...\d+` patterns).
5. Detect and skip blank pages.
6. Extract heading hierarchy:
   - First attempt: `doc.get_toc()` for PDF outline/bookmarks.
   - Fallback: font-size-based inference from `page.get_text("dict")` spans.
7. Extract document-level metadata from `doc.metadata` and propagate to all chunks.
8. Chunk extracted markdown using configurable strategy (default: recursive character splitter respecting heading boundaries).
9. Embed chunks via `EmbeddingBackend`.
10. Upsert to `VectorStoreBackend` with `PDFChunkMetadata`.

**Metadata per chunk:**
```python
PDFChunkMetadata(
    source_uri="file:///path/to/handbook.pdf",
    source_format="pdf",
    page_numbers=[3, 4],
    ingestion_method="text_extraction",
    heading_path=["Benefits", "Health Insurance", "Eligibility"],
    content_type="narrative",
    doc_title="Employee Benefits Handbook 2024",
    doc_author="HR Department",
    language="en",
    ...
)
```

**Public interface:**
```python
class TextExtractor:
    def __init__(self, vector_store: VectorStoreBackend, embedder: EmbeddingBackend,
                 config: PDFProcessorConfig): ...
    def process(self, file_path: str, profile: DocumentProfile,
                ingest_key: str, ingest_run_id: str) -> ProcessingResult: ...
```

### 11.2 Path B — OCR Processor (`processors/ocr_processor.py`)

**Input:** DocumentProfile + classification confirming Type B (scanned) or pages flagged for OCR fallback.

**Steps:**
1. Render each page to a high-DPI image (default 300 DPI) using `page.get_pixmap()`.
2. Pre-process images based on `config.ocr_preprocessing_steps`:
   - **deskew** — detect skew angle via Hough transform, rotate to correct.
   - **denoise** — `cv2.fastNlMeansDenoisingColored()`.
   - **binarize** — adaptive thresholding (Otsu's method).
   - **contrast** — CLAHE enhancement.
3. Detect language per page if `enable_language_detection=True` (FastText via `fast-langdetect`). Route to appropriate OCR language model.
4. Run OCR via the configured engine (`config.ocr_engine`):
   - **Tesseract:** `pytesseract.image_to_data()` → word-level text + confidence.
   - **PaddleOCR:** `PaddleOCR(lang=lang)` → line-level text + confidence.
5. Collect per-page `OCRResult` with confidence scores.
6. Post-process OCR text (`utils/ocr_postprocess.py`):
   - Merge hyphenated line breaks (`docu-\nment` → `document`).
   - Normalize whitespace and Unicode.
   - Strip OCR artifacts (random isolated characters, repeated punctuation).
7. Optionally clean up via LLM (if `config.enable_ocr_cleanup=True`): send raw OCR text to `config.ocr_cleanup_model` and ask to fix obvious errors while preserving meaning.
8. Flag low-confidence pages (< `config.ocr_confidence_threshold`) with `W_PAGE_LOW_OCR_CONFIDENCE`.
9. From here, cleaned text follows the same path as Path A: heading detection, chunking, embedding, upsert.

**OCR parallelism:** Pages are OCR'd in parallel using `ProcessPoolExecutor` with `config.ocr_max_workers`. Each worker creates its own OCR engine instance (PaddleOCR has known issues with shared instances across processes).

**Metadata per chunk (additional OCR fields):**
```python
PDFChunkMetadata(
    ...
    ingestion_method="ocr_pipeline",
    ocr_engine="tesseract",
    ocr_confidence=0.87,
    ocr_dpi=300,
    ocr_preprocessing=["deskew", "binarize"],
    ...
)
```

**Public interface:**
```python
class OCRProcessor:
    def __init__(self, vector_store: VectorStoreBackend, embedder: EmbeddingBackend,
                 llm: LLMBackend | None, config: PDFProcessorConfig): ...
    def process(self, file_path: str, profile: DocumentProfile,
                pages: list[int] | None,  # specific pages, or None for all
                ingest_key: str, ingest_run_id: str) -> ProcessingResult: ...
```

### 11.3 Path C — Complex Processor (`processors/complex_processor.py`)

**Input:** DocumentProfile + classification confirming Type C (complex), with per-page type assignments.

**Steps:**
1. **Page-level routing:** For each page, based on its `PageType`:
   - `TEXT` → route to `TextExtractor` logic.
   - `SCANNED` → route to `OCRProcessor` logic.
   - `TABLE_HEAVY` → extract tables via `TableExtractor`, extract surrounding text.
   - `FORM` → extract form field names and values as `"Field: Value"` pairs.
   - `MIXED` → extract text natively, OCR image regions.
   - `BLANK`, `TOC`, `VECTOR_ONLY` → skip with appropriate warning.

2. **Table extraction** (via `processors/table_extractor.py`):
   - Detect tables using `pdfplumber` on each page.
   - For each table:
     - Extract as a pandas DataFrame.
     - If <= `table_max_rows_for_serialization` rows: serialize rows as natural language sentences, embed as chunks.
     - If > `table_min_rows_for_db` rows: load into `StructuredDBBackend` via `create_table_from_dataframe()`, embed a schema description.
   - Tag with `content_type="table"`, `table_index`, `page_number`.

3. **Multi-page table stitching:**
   - Compare column count and header text between last table on page N and first table on page N+1.
   - If column count matches AND header similarity >= `table_continuation_column_match_threshold` → concatenate (skip repeated header on page N+1).
   - Tag with `is_continuation=True` and shared `continuation_group_id`.
   - Emit `W_TABLE_CONTINUATION` warning.

4. **Multi-column handling** (via `utils/layout_analysis.py`):
   - Detect multi-column layouts via text block x-coordinate clustering.
   - Reorder text blocks into correct reading order (left column first, then right column) before chunking.

5. **Form field extraction:**
   - Extract AcroForm field names and values.
   - Serialize as `"Field Name: Value"` pairs.
   - Useful for HR forms, onboarding documents, benefits enrollment.

6. **Header/footer stripping:** Same as Path A but run across all extracted content.

7. **Section detection:** Use font size/style changes to detect section boundaries. Map heading hierarchy.

8. After all content extracted: chunk, embed, upsert with rich metadata.

**Public interface:**
```python
class ComplexProcessor:
    def __init__(self, vector_store: VectorStoreBackend, structured_db: StructuredDBBackend,
                 embedder: EmbeddingBackend, llm: LLMBackend | None,
                 config: PDFProcessorConfig): ...
    def process(self, file_path: str, profile: DocumentProfile,
                classification: ClassificationResult,
                ingest_key: str, ingest_run_id: str) -> ProcessingResult: ...
```

---

## 12. OCR Engine Abstraction (`utils/ocr_engines.py`)

### 12.1 Purpose

Provide a unified interface for swappable OCR engines. Tesseract is the required portable baseline; PaddleOCR is the optional high-accuracy upgrade.

### 12.2 Interface

```python
class OCREngineInterface(Protocol):
    """Structural subtyping interface for OCR engines."""
    def recognize(self, image: "Image.Image", language: str) -> OCRPageResult: ...
    def name(self) -> str: ...

class OCRPageResult(BaseModel):
    """Standardized OCR output from any engine."""
    text: str
    confidence: float                      # 0.0-1.0 average
    word_confidences: list[float] | None = None
    language_detected: str | None = None
```

### 12.3 Tesseract Adapter (Baseline)

```python
class TesseractEngine:
    """Adapter for Tesseract via pytesseract. Required baseline on all platforms."""
    def __init__(self, lang: str = "eng"): ...
    def recognize(self, image: "Image.Image", language: str) -> OCRPageResult: ...
    def name(self) -> str:
        return "tesseract"
```

- Uses `pytesseract.image_to_data(output_type=pytesseract.Output.DICT)` for word-level confidence.
- Language codes differ from PaddleOCR (e.g., `eng` not `en`). Adapter handles mapping.
- Gracefully detects if Tesseract binary is not installed → `E_OCR_ENGINE_UNAVAILABLE`.

### 12.4 PaddleOCR Adapter (Optional Upgrade)

```python
class PaddleOCREngine:
    """Adapter for PaddleOCR. Optional, higher accuracy (~96.6% vs ~89%)."""
    def __init__(self, lang: str = "en"): ...
    def recognize(self, image: "Image.Image", language: str) -> OCRPageResult: ...
    def name(self) -> str:
        return "paddleocr"
```

- Uses `paddleocr.PaddleOCR(lang=language, use_angle_cls=True)`.
- Returns line-level text with confidence.
- Supports 100+ languages.
- Each `ProcessPoolExecutor` worker must create its own instance.

### 12.5 Engine Factory with Fallback Chain

```python
def create_ocr_engine(config: PDFProcessorConfig) -> tuple[OCREngineInterface, list[str]]:
    """Create the configured OCR engine with fallback.

    Returns:
        (engine, warnings): The engine instance and any warning codes emitted.
    """
    warnings: list[str] = []

    if config.ocr_engine == OCREngine.PADDLEOCR:
        try:
            return PaddleOCREngine(lang=config.ocr_language), warnings
        except ImportError:
            # PaddleOCR not installed — fall back to Tesseract baseline
            warnings.append("W_OCR_ENGINE_FALLBACK")
            logger.warning(
                "PaddleOCR requested but not installed, falling back to Tesseract baseline"
            )
            # Fall through to Tesseract

    # Tesseract: required baseline — no fallback from here
    if not _tesseract_available():
        raise EngineUnavailableError(
            "Tesseract is the required OCR baseline but is not installed. "
            "Install with: apt install tesseract-ocr && pip install pytesseract"
        )
    return TesseractEngine(lang=config.ocr_language), warnings
```

**Fallback chain:** PaddleOCR (if configured) → Tesseract (required baseline) → `E_OCR_ENGINE_UNAVAILABLE` (hard error).

---

## 13. Header/Footer Detection (`utils/header_footer.py`)

### 13.1 Algorithm

1. Sample `config.header_footer_sample_pages` pages (default 5, evenly distributed across the document).
2. For each sampled page, extract text from the top `header_footer_zone_ratio` (default 10%) and bottom 10% by y-coordinate.
3. Compare text across sampled pages using `difflib.SequenceMatcher`.
4. Text appearing on >= (sampled_pages - 1) pages at the same spatial position with similarity >= `header_footer_similarity_threshold` (default 0.7) is classified as header/footer.
5. Build a set of header patterns and footer patterns.
6. On each page, strip matching text before chunking.

### 13.2 Fast Path

When using `pymupdf4llm`, the built-in `header=False, footer=False` parameters handle most cases. The cross-page similarity algorithm serves as a second pass for edge cases the built-in detector misses.

### 13.3 Public Interface

```python
class HeaderFooterDetector:
    def __init__(self, config: PDFProcessorConfig): ...
    def detect(self, doc: "fitz.Document") -> tuple[list[str], list[str]]:
        """Returns (header_patterns, footer_patterns)."""
        ...
    def strip(self, text: str, page_number: int, headers: list[str], footers: list[str]) -> str:
        """Remove detected headers/footers from extracted text."""
        ...
```

---

## 14. Heading Hierarchy Detection (`utils/heading_detector.py`)

### 14.1 Strategy 1: PDF Outline

```python
toc = doc.get_toc()
# Returns: [[level, title, page_number], ...]
```

If the PDF has bookmarks, this is the authoritative heading hierarchy.

### 14.2 Strategy 2: Font-Based Inference

When no PDF outline exists:
1. Extract all text spans with font metadata via `page.get_text("dict")`.
2. Compute body text font size as the most common font size across the document.
3. Classify spans where `font_size >= body_size * config.heading_min_font_size_ratio` AND `is_bold` as headings.
4. Map the top 3 distinct heading font sizes to H1/H2/H3.
5. Build a heading tree from the page-ordered heading list.

### 14.3 Strategy 3: PyMuPDF4LLM Markdown

`pymupdf4llm.to_markdown()` automatically maps font sizes to `#`/`##`/`###` markdown headers. Parse these from the markdown output.

### 14.4 Public Interface

```python
class HeadingDetector:
    def __init__(self, config: PDFProcessorConfig): ...
    def detect(self, doc: "fitz.Document") -> list[tuple[int, str, int]]:
        """Returns [(level, title, page_number), ...] ordered by appearance."""
        ...
    def get_heading_path(self, page_number: int, position_y: float) -> list[str]:
        """Returns the heading ancestry at a given position. E.g. ['Chapter 2', 'Section 2.1']."""
        ...
```

---

## 15. Chunking (`utils/chunker.py`)

### 15.1 Default Strategy

Recursive character splitting with heading-aware boundaries:

1. Split on `\n## ` (heading level 2) boundaries first.
2. Within each section, split on `\n### ` (heading level 3).
3. Within each sub-section, split on `\n\n` (paragraph).
4. Within each paragraph, split on `. ` (sentence).
5. Within each sentence, split on ` ` (word).
6. Target chunk size: `config.chunk_size_tokens` (default 512 tokens).
7. Overlap: `config.chunk_overlap_tokens` (default 50 tokens, ~10%).

### 15.2 Table-Aware Chunking

Tables are never split mid-table. A table that exceeds `chunk_size_tokens` is kept as a single oversized chunk rather than broken.

### 15.3 Metadata Attachment

Each chunk carries:
- `page_numbers` — which pages the chunk spans.
- `heading_path` — the heading ancestry at the chunk's position.
- `content_type` — narrative, table, list, form_field, etc.
- `chunk_index` — 0-based position within the document.
- `chunk_hash` — SHA-256 of the chunk text.

### 15.4 Public Interface

```python
class PDFChunker:
    def __init__(self, config: PDFProcessorConfig): ...
    def chunk(self, text: str, headings: list[tuple[int, str, int]],
              page_boundaries: list[int]) -> list[dict]:
        """Chunk text with metadata. Returns [{text, page_numbers, heading_path, chunk_index}, ...]."""
        ...
```

---

## 16. Language Detection (`utils/language.py`)

### 16.1 Purpose

Detect the language of each page to route to the appropriate OCR language model. Critical for multi-language HR documents (bilingual handbooks, translated policies).

### 16.2 Implementation

Uses `fast-langdetect` (FastText wrapper):
- 217 languages supported.
- ~1MB model, 95% accuracy.
- Returns `{lang: "en", score: 0.98}`.

### 16.3 Integration

```python
def detect_language(text: str) -> tuple[str, float]:
    """Returns (iso_639_1_code, confidence)."""
    ...

def map_language_to_ocr(lang: str, engine: OCREngine) -> str:
    """Map ISO 639-1 code to engine-specific language code.
    E.g. 'en' -> 'eng' for Tesseract, 'en' -> 'en' for PaddleOCR.
    """
    ...
```

---

## 17. Router (`router.py`)

The `PDFRouter` is the top-level orchestrator and implements the `FileHandler` protocol for pipeline integration.

```python
class PDFRouter:
    def __init__(
        self,
        vector_store: VectorStoreBackend,
        structured_db: StructuredDBBackend,
        llm: LLMBackend,
        embedder: EmbeddingBackend,
        config: PDFProcessorConfig | None = None,
    ): ...

    def can_handle(self, file_path: str) -> bool:
        """Returns True if file_path ends with .pdf (case-insensitive)."""
        return Path(file_path).suffix.lower() == ".pdf"

    def process(self, file_path: str, source_uri: str | None = None) -> ProcessingResult:
        """Classify and process a single PDF file. Synchronous — blocks until complete."""
        ...

    def process_batch(self, file_paths: list[str]) -> list[ProcessingResult]:
        """Process multiple files. Each file is isolated in a subprocess with timeout."""
        ...
```

### 17.1 `process()` Flow

1. **Security scan** via `PDFSecurityScanner.scan()`. Fatal errors → return immediately.
2. Compute `ingest_key`. If `enable_content_dedup=True`, caller can check for duplicates.
3. Generate `ingest_run_id` (UUID4).
4. **Open document** with PyMuPDF. If corrupt → attempt repair → if repair fails → `E_PARSE_CORRUPT`.
5. **Extract document profile**: per-page `PageProfile` with quality assessment.
6. **Detect language** on pages with extractable text (if enabled).
7. **Tier 1:** `PDFInspector.classify()`.
8. If inconclusive → **Tier 2:** `PDFLLMClassifier.classify()` (skipped if LLM unavailable → §5.2).
9. If still low confidence → **Tier 3** (if enabled, skipped if LLM unavailable).
10. If all tiers fail and Tier 1 produced no usable result → return with `E_CLASSIFY_INCONCLUSIVE`.
11. Route based on `ClassificationResult.pdf_type`:
    - `TEXT_NATIVE` → `TextExtractor.process()`
    - `SCANNED` → `OCRProcessor.process()`
    - `COMPLEX` → `ComplexProcessor.process()`
12. Collect `WrittenArtifacts`.
13. Assemble `ProcessingResult`.
14. Log (PII-safe).
15. Return.

### 17.2 Process Isolation

For `process_batch()`, each PDF is processed in an isolated subprocess via `ProcessPoolExecutor` with `config.per_document_timeout_seconds` as the hard timeout. This prevents:
- Memory leaks from accumulating across documents.
- A single malicious/corrupt PDF from hanging the entire batch.
- OOM from a large PDF affecting other documents.

### 17.3 Pipeline Integration

Both `PDFRouter` and `ExcelRouter` implement the `FileHandler` protocol and register identically with the parent pipeline:

```python
from ingestkit_excel import create_default_router as excel_router
from ingestkit_pdf import create_default_router as pdf_router

pipeline = Pipeline()
pipeline.register(excel_router())    # handles .xlsx, .xls, .xlsm
pipeline.register(pdf_router())      # handles .pdf
```

---

## 18. Execution Model

### 18.1 v1.0: Synchronous, Local Process Backend

The v1.0 API is **synchronous**. `process()` blocks the calling thread until processing completes.

```python
# v1.0 — canonical API
result = router.process("document.pdf")          # blocks
results = router.process_batch(["a.pdf", "b.pdf"])  # blocks, internal parallelism via ProcessPoolExecutor
```

**Lifecycle:**
- Caller owns the thread. `process()` is a plain function call.
- `process_batch()` manages its own `ProcessPoolExecutor` internally — caller does not manage workers.
- Backend connections (Qdrant, Ollama, SQLite) are created per-worker in `process_batch()` to avoid cross-process sharing issues.

**Concurrency within `process()`:**
- OCR pages are parallelized via `ProcessPoolExecutor` with `config.ocr_max_workers`.
- Embedding batches are sequential (most embedding backends are already async-internally).

### 18.2 v1.1: Async Wrapper (Planned)

**Target milestone:** v1.1 (see §24 Phased Delivery Plan)

```python
# v1.1 — async wrapper over sync core
result = await router.aprocess("document.pdf")
```

**Design:**
- `aprocess()` wraps `process()` via `asyncio.to_thread()` (or `loop.run_in_executor()`).
- The sync `process()` remains the canonical implementation — `aprocess()` is a thin wrapper, not a rewrite.
- No async backends required in v1.1; async is for caller integration (FastAPI, async job queues), not for internal I/O.

**Migration:**
- `process()` remains available and supported indefinitely. It is NOT deprecated.
- Callers using sync frameworks (CLI, scripts, Celery workers) continue using `process()`.
- Callers using async frameworks (FastAPI, aiohttp) use `aprocess()`.

### 18.3 v1.1+: Distributed Backend (Planned)

**Target milestone:** v1.1+

```python
class ExecutionBackend(Protocol):
    """Pluggable execution backend for document processing."""
    def submit(self, file_path: str, config: PDFProcessorConfig) -> str:
        """Submit a document for processing. Returns a job_id."""
        ...
    def get_result(self, job_id: str, timeout: float | None = None) -> ProcessingResult:
        """Block until result is available or timeout."""
        ...

class LocalExecutionBackend:
    """v1.0 default: process in-process or via ProcessPoolExecutor."""
    ...

class DistributedExecutionBackend:
    """v1.1+: submit to a queue (Redis, RabbitMQ, etc.), workers process."""
    ...
```

**Compatibility contract:**
- `LocalExecutionBackend` is always available and is the default.
- `DistributedExecutionBackend` is optional — requires queue infrastructure.
- Both backends produce identical `ProcessingResult` for the same input.
- Rollback to `LocalExecutionBackend` is always possible for single-node operations.
- Compatibility tests against `ingestkit-core` interfaces are required before shipping any new backend.

---

## 19. Error Handling

| Scenario | Error Code | Behavior |
|----------|-----------|----------|
| Not a valid PDF | `E_SECURITY_INVALID_PDF` | Reject immediately |
| Decompression bomb suspected | `E_SECURITY_DECOMPRESSION_BOMB` | Reject immediately |
| Embedded JavaScript | `E_SECURITY_JAVASCRIPT` | Reject immediately (configurable, see §7.5) |
| File too large | `E_SECURITY_TOO_LARGE` | Reject immediately |
| Too many pages | `E_SECURITY_TOO_MANY_PAGES` | Reject immediately |
| Password-protected (user password) | `E_PARSE_PASSWORD` | Reject with structured error |
| Owner-password only (readable) | `W_ENCRYPTED_OWNER_ONLY` | Process normally, warn |
| Corrupt/unreadable PDF | `E_PARSE_CORRUPT` | Attempt repair, if fails → error |
| Garbled text (CIDFont) | `W_OCR_FALLBACK` | Auto-fallback to OCR (if enabled) |
| Zero pages | `E_PARSE_EMPTY` | Return error result |
| Blank page | `W_PAGE_SKIPPED_BLANK` | Skip, add warning |
| TOC page | `W_PAGE_SKIPPED_TOC` | Skip, add warning |
| Vector-only page | `W_PAGE_SKIPPED_VECTOR_ONLY` | OCR fallback or skip |
| OCR engine not installed (baseline) | `E_OCR_ENGINE_UNAVAILABLE` | Clear error, no crash |
| Configured OCR engine unavailable (optional) | `W_OCR_ENGINE_FALLBACK` | Fall back to Tesseract baseline |
| OCR timeout | `E_OCR_TIMEOUT` | Skip page, add error |
| OCR low confidence | `W_PAGE_LOW_OCR_CONFIDENCE` | Flag for review |
| LLM malformed JSON | `E_LLM_MALFORMED_JSON` | Retry once |
| LLM schema invalid | `E_LLM_SCHEMA_INVALID` | Retry once, fall back to Tier 1 |
| LLM backend unreachable | `W_LLM_UNAVAILABLE` | Skip Tier 2/3, use Tier 1 result |
| Classification degraded (LLM outage) | `W_CLASSIFICATION_DEGRADED` | Tier 1 result used, processing continues |
| Backend connection failure | `E_BACKEND_*_CONNECT` | Retry with backoff |
| Digitally signed PDF | `W_DOCUMENT_SIGNED` | Read-only extraction, warn |
| Embedded files | `W_EMBEDDED_FILES` | Log, do not process attachments |
| Multi-page table stitched | `W_TABLE_CONTINUATION` | Stitch and warn |
| Security default overridden | `W_SECURITY_OVERRIDE` | Log with reason and config source |

---

## 20. Logging

Uses Python `logging` module with logger name `ingestkit_pdf`.

### 20.1 PII-Safe by Default

**INFO level** — every processed file:
```
ingestkit_pdf | file=handbook.pdf | ingest_key=b7c2... | tier=rule_based | type=text_native | confidence=0.92 | degraded=false | path=text_extraction | pages=87 | chunks=156 | tables=0 | ocr_pages=0 | time=4.2s
```

**WARNING level** — fallbacks and non-fatal issues:
```
ingestkit_pdf | file=legacy_scan.pdf | code=W_OCR_FALLBACK | page=12 | detail=garbled text detected (printable_ratio=0.42), falling back to OCR
ingestkit_pdf | file=mixed.pdf | code=W_LLM_UNAVAILABLE | detail=LLM backend unreachable (ConnectionError), classification degraded to Tier 1
ingestkit_pdf | SECURITY_OVERRIDE | file=form.pdf | override=reject_javascript | reason="TICKET-4521" | config_source=migration.yaml
```

**ERROR level** — fatal failures:
```
ingestkit_pdf | file=corrupt.pdf | code=E_PARSE_CORRUPT | detail=PyMuPDF open failed, repair attempt also failed
```

**DEBUG level** (opt-in only):
- Signal breakdowns per page (`log_sample_text=False` by default)
- LLM prompts/responses (`log_llm_prompts=False`)
- OCR raw output (`log_ocr_output=False`)
- Chunk text previews (`log_chunk_previews=False`)

If `config.redact_patterns` are set, all logged text is scrubbed before emission.

---

## 21. Public API

### 21.1 Top-Level Exports (`__init__.py`)

```python
from ingestkit_pdf.router import PDFRouter
from ingestkit_pdf.config import PDFProcessorConfig
from ingestkit_pdf.models import (
    PDFType, PageType, ClassificationTier, ClassificationResult,
    ProcessingResult, ChunkPayload, PDFChunkMetadata,
    DocumentProfile, DocumentMetadata, PageProfile, ExtractionQuality,
    OCRResult, TableResult, WrittenArtifacts,
    ParseStageResult, ClassificationStageResult, OCRStageResult, EmbedStageResult,
)
from ingestkit_pdf.errors import ErrorCode, IngestError

def create_default_router(**overrides) -> PDFRouter:
    """Create a router with default backends (Qdrant, SQLite, Ollama)."""
    ...
```

### 21.2 Usage Example

```python
from ingestkit_pdf import create_default_router

router = create_default_router()
result = router.process("path/to/handbook.pdf")

print(result.classification.pdf_type)           # PDFType.TEXT_NATIVE
print(result.classification_result.tier_used)    # ClassificationTier.RULE_BASED
print(result.classification.degraded)            # False
print(result.chunks_created)                     # 156
print(result.tables_created)                     # 0
print(result.ocr_result)                         # None (no OCR needed)
```

### 21.3 Custom Backend Example

```python
from ingestkit_pdf import PDFRouter, PDFProcessorConfig
from ingestkit_pdf.models import OCREngine

router = PDFRouter(
    vector_store=my_qdrant,
    structured_db=my_sqlite,
    llm=my_ollama_llm,
    embedder=my_ollama_embedding,
    config=PDFProcessorConfig(
        ocr_engine=OCREngine.PADDLEOCR,    # upgrade to PaddleOCR for accuracy
        ocr_language="en",
        classification_model="qwen2.5:7b",
        tenant_id="client_acme",
        chunk_size_tokens=256,             # smaller chunks for factoid queries
    ),
)

result = router.process("scanned_policy.pdf")
print(result.ocr_result.engine_used)            # OCREngine.PADDLEOCR
print(result.ocr_result.avg_confidence)          # 0.94
```

---

## 22. Testing Strategy & Release Gates

### 22.1 Mock Backends (`conftest.py`)

Same pattern as `ingestkit-excel`: in-memory mock implementations of all four protocols.

### 22.2 Programmatic Test PDF Generation

Generate test PDFs in `conftest.py` using `reportlab` or `fpdf2` — **no binary fixtures committed to repo**.

```python
@pytest.fixture
def text_native_pdf(tmp_path) -> Path:
    """Multi-page PDF with headings, paragraphs, and page numbers."""
    # Use reportlab to create a clean digital PDF
    ...

@pytest.fixture
def scanned_pdf(tmp_path) -> Path:
    """PDF with pages that are images (simulating scanned documents)."""
    # Render text to images, embed as full-page PDF images
    ...

@pytest.fixture
def complex_pdf(tmp_path) -> Path:
    """PDF with tables, multi-column layout, and mixed content."""
    # Include: data table, two-column section, headers/footers
    ...

@pytest.fixture
def encrypted_pdf(tmp_path) -> Path:
    """Password-protected PDF."""
    ...

@pytest.fixture
def garbled_pdf(tmp_path) -> Path:
    """PDF with intentionally garbled text (simulating CIDFont issues)."""
    ...
```

### 22.3 Test Coverage

| Module | Key test cases |
|--------|---------------|
| `security.py` | Valid PDF accepted; invalid magic bytes rejected; oversized file rejected; JS detected; decompression bomb detected; encrypted handling; security override logging |
| `quality.py` | HIGH/MEDIUM/LOW scoring; printable ratio calculation; OCR fallback trigger; garbled text detection |
| `inspector.py` | All signals correctly evaluated per page; text-native/scanned/complex classification; multi-page type distribution; hybrid detection |
| `llm_classifier.py` | Summary generation (no raw text); schema validation; malformed JSON retry; tier escalation; fail-closed; **LLM outage degradation** |
| `text_extractor.py` | Markdown extraction; heading hierarchy; header/footer stripping; TOC skipping; chunk metadata correctness |
| `ocr_processor.py` | Tesseract path; PaddleOCR path; preprocessing steps; confidence scoring; low-confidence flagging; parallel processing; **engine fallback** |
| `complex_processor.py` | Page-level routing; table extraction; multi-page table stitching; form field extraction; multi-column reordering |
| `table_extractor.py` | pdfplumber table detection; DataFrame conversion; serialization vs DB routing; continuation stitching |
| `header_footer.py` | Cross-page similarity detection; pattern stripping; edge cases (single-page PDFs, no headers) |
| `chunker.py` | Heading-aware splitting; table-aware splitting; overlap; chunk size limits |
| `ocr_engines.py` | Tesseract adapter; PaddleOCR adapter; engine factory; unavailable engine error; **fallback chain** |
| `router.py` | Full flow for each type; tier escalation; **LLM outage resilience**; security scan integration; process isolation; WrittenArtifacts; PII-safe logging |

### 22.4 Markers

- `@pytest.mark.unit` — runs with mocks only, no external services, no OCR engines required.
- `@pytest.mark.integration` — requires running Qdrant, Ollama, etc.
- `@pytest.mark.ocr` — requires Tesseract installed (baseline).
- `@pytest.mark.ocr_paddle` — requires PaddleOCR installed (optional upgrade).

### 22.5 Release Gates

Each phase (see §24) has hard release gates that must pass before merging to main.

**Phase 1 Gate (v1.0):**

| Gate | Criteria | Evidence Artifact |
|------|----------|-------------------|
| Unit tests | 100% of `@pytest.mark.unit` pass | `pytest --tb=short -m unit` output |
| OCR baseline tests | 100% of `@pytest.mark.ocr` pass (Tesseract) | `pytest --tb=short -m ocr` output |
| Lint | `ruff check .` clean | CI job log |
| Type check | `mypy src/` clean (strict mode) | CI job log |
| Coverage | >= 80% line coverage | `pytest --cov=ingestkit_pdf --cov-report=term` |
| LLM outage test | `test_llm_outage_degrades_to_tier1` passes | Included in unit suite |
| Security override test | Override-without-reason raises `ValidationError` | Included in unit suite |

**Phase 1.1 Gate:**

| Gate | Criteria | Evidence Artifact |
|------|----------|-------------------|
| All Phase 1 gates | Pass | (same) |
| PaddleOCR tests | 100% of `@pytest.mark.ocr_paddle` pass | `pytest --tb=short -m ocr_paddle` output |
| Engine fallback test | PaddleOCR unavailable → Tesseract used with warning | Included in unit suite |
| Async wrapper test | `aprocess()` produces same result as `process()` | Included in unit suite |
| ARM64 CI | Unit + OCR baseline pass on ARM64 runner | CI job log |

**Phase 2 Gate:**

| Gate | Criteria | Evidence Artifact |
|------|----------|-------------------|
| All Phase 1.1 gates | Pass | (same) |
| Integration tests | 100% of `@pytest.mark.integration` pass | `pytest --tb=short -m integration` output |
| Benchmark report | Throughput targets met (see §25) | `benchmark-report-<date>.json` |
| UAT sign-off | Manual review of 10 representative documents per type (A/B/C) | `uat-signoff-<date>.md` signed by reviewer |
| Compatibility tests | `ingestkit-core` interface compliance verified | Included in integration suite |

---

## 23. Dependencies

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "ingestkit-pdf"
version = "0.1.0"
description = "Tiered PDF file processing for RAG pipelines"
requires-python = ">=3.10"
dependencies = [
    "pymupdf>=1.24",
    "pymupdf4llm>=0.0.10",
    "pdfplumber>=0.10",
    "pandas>=2.0",
    "pydantic>=2.0",
    "Pillow>=10.0",
]

[project.optional-dependencies]
# OCR engines
tesseract = ["pytesseract>=0.3"]       # required baseline — install on all targets
paddleocr = ["paddleocr>=2.7"]         # optional upgrade — higher accuracy

# Language detection
langdetect = ["fast-langdetect>=0.2"]

# Image preprocessing
opencv = ["opencv-python-headless>=4.8"]

# Advanced layout analysis (optional)
docling = ["docling>=2.0"]

# Vector store backends
qdrant = ["qdrant-client>=1.7"]

# Structured DB backends
postgres = ["psycopg2-binary>=2.9"]

# LLM / Embedding backends
ollama = ["httpx>=0.27"]

# Convenience bundles
baseline = ["ingestkit-pdf[tesseract,langdetect,opencv,qdrant,ollama]"]
full = ["ingestkit-pdf[tesseract,paddleocr,langdetect,opencv,docling,qdrant,ollama]"]

# Development
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "pyyaml>=6.0",
    "reportlab>=4.0",         # programmatic test PDF generation
    "mypy>=1.5",
    "ruff>=0.1",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "unit: Unit tests (no external services)",
    "integration: Integration tests (require external services)",
    "ocr: Tests requiring Tesseract OCR engine installed (baseline)",
    "ocr_paddle: Tests requiring PaddleOCR installed (optional upgrade)",
]
```

---

## 24. Phased Delivery Plan

### 24.1 Phase 1 — v1.0: Core Pipeline with Local Backend

**Scope:** Complete PDF processing pipeline with synchronous API, Tesseract baseline OCR, and local execution.

**Deliverables:**
- Security scanning (§7)
- Quality scoring (§8)
- Tier 1 inspector (§9)
- Tier 2/3 LLM classifier with outage resilience (§10)
- Path A text extraction (§11.1)
- Path B OCR with Tesseract baseline (§11.2)
- Path C complex processing (§11.3)
- Header/footer detection, heading detection, chunking (§13-15)
- Router with `process()` and `process_batch()` (§17)
- Full unit test suite + OCR baseline tests (§22.5 Phase 1 Gate)

**Entry criteria:** Spec v1.1 approved.
**Exit criteria:** Phase 1 Gate passes. PR merged to main.
**Rollback:** N/A (first release).

### 24.2 Phase 1.1 — PaddleOCR + Async Wrapper

**Scope:** Optional PaddleOCR upgrade, async API wrapper, ARM64 CI validation.

**Deliverables:**
- PaddleOCR adapter with fallback chain (§12.4-12.5)
- `aprocess()` async wrapper via `asyncio.to_thread()` (§18.2)
- ARM64 CI job (§3.5)
- PaddleOCR-specific test suite (`@pytest.mark.ocr_paddle`)

**Entry criteria:** Phase 1 merged and stable (no regressions in 1 week).
**Exit criteria:** Phase 1.1 Gate passes.
**Rollback:** Remove PaddleOCR adapter + async wrapper. Pipeline continues on Tesseract baseline + sync API.

### 24.3 Phase 2 — v1.1: Distributed Backend + Production Hardening

**Scope:** `ExecutionBackend` interface, distributed backend, benchmark validation, UAT.

**Deliverables:**
- `ExecutionBackend` protocol (§18.3)
- `LocalExecutionBackend` (refactor existing `process_batch()`)
- `DistributedExecutionBackend` (queue/worker pattern)
- Integration test suite (`@pytest.mark.integration`)
- Benchmark report against SLO targets (§25)
- UAT sign-off

**Entry criteria:** Phase 1.1 merged and stable.
**Exit criteria:** Phase 2 Gate passes including UAT sign-off.
**Rollback:** Revert to `LocalExecutionBackend`. All distributed backend code is behind the `ExecutionBackend` interface — switching back is a config change, not a code change.

### 24.4 Compatibility Contract Across Phases

| Interface | Introduced | Compatibility Rule |
|-----------|-----------|-------------------|
| `process()` | Phase 1 | Never removed. Always available. |
| `ProcessingResult` schema | Phase 1 | Additive only — new fields get defaults. Never remove fields. |
| `ErrorCode` enum | Phase 1 | Additive only — new codes may be added. Never remove or rename. |
| `PDFProcessorConfig` | Phase 1 | Additive only — new fields get defaults. Changing a default requires `parser_version` bump. |
| `aprocess()` | Phase 1.1 | Wrapper only — semantics identical to `process()`. |
| `ExecutionBackend` | Phase 2 | New protocol — does not affect `process()` or `aprocess()`. |

---

## 25. Operational SLOs

### 25.1 Purpose

Define initial target thresholds for observability and alerting. These are baseline targets refined after Phase 1 benchmarking data is collected.

### 25.2 Throughput Targets

| Hardware Class | Target (pages/sec) | OCR Engine | Notes |
|---------------|--------------------|-----------|----|
| Laptop (WSL2, CPU-only) | >= 10 | Tesseract | Path A text-native: 50+ pages/sec; Path B OCR: 10 pages/sec |
| DGX-spark-class (GPU) | >= 50 | PaddleOCR | GPU-accelerated OCR + parallel workers |
| DGX-spark-class (CPU fallback) | >= 25 | Tesseract | Degraded mode if GPU unavailable |

### 25.3 Error Budgets

| Metric | SLO Target | Alert Threshold | Measurement Window |
|--------|-----------|----------------|-------------------|
| Per-document timeout breach rate | < 2% (laptop), < 0.5% (DGX) | > 5% (laptop), > 2% (DGX) | Rolling 100 documents |
| OCR fallback rate (pages) | 5-15% expected | > 30% triggers investigation | Per batch |
| `E_PARSE_*` error rate | < 1% of documents | > 3% | Rolling 100 documents |
| `W_*` warning rate | < 20% of documents | > 40% | Rolling 100 documents |
| `E_CLASSIFY_INCONCLUSIVE` rate | < 0.5% | > 2% | Rolling 100 documents |
| LLM degraded classification rate | < 5% (healthy infra) | > 10% | Rolling 100 documents |
| OCR low-confidence page rate | < 10% of OCR pages | > 25% | Per batch |

### 25.4 Per-Stage Latency Budgets

For a typical 50-page document on laptop hardware:

| Stage | Target | Max |
|-------|--------|-----|
| Security scan | < 100ms | 500ms |
| Profile extraction | < 2s | 5s |
| Tier 1 classification | < 200ms | 500ms |
| Tier 2 LLM classification | < 5s | 15s |
| Path A text extraction | < 3s | 10s |
| Path B OCR (per page) | < 3s | 10s |
| Embedding (per batch of 64) | < 2s | 10s |
| Total (Path A, 50 pages) | < 15s | 30s |
| Total (Path B, 50 pages) | < 120s | 300s |

### 25.5 Refinement Process

1. Phase 1 ships with these initial targets.
2. After processing 1,000 documents in staging, collect actual percentile data (p50, p95, p99).
3. Revise targets in spec v1.2 based on observed distributions.
4. Integrate thresholds into monitoring/alerting configuration.

---

## 26. Versioning & Change Policy

### 26.1 `parser_version` Bumps

The `parser_version` field (`"ingestkit_pdf:X.Y.Z"`) determines `ingest_key` — changing it means previously processed documents get new keys and will be re-ingested.

| Change Type | Requires `parser_version` Bump? | Example |
|------------|-------------------------------|---------|
| Config default change (e.g., `chunk_size_tokens`) | **Yes** | Changing default from 512 → 256 |
| New config field with default | No | Adding `ocr_contrast_threshold` with sensible default |
| OCR engine model update | **Yes** | Tesseract 4 → Tesseract 5, or PaddleOCR model version change |
| Extraction library update affecting output | **Yes** | PyMuPDF major version bump that changes text extraction |
| Bug fix that changes chunk content | **Yes** | Fixing a chunking boundary bug |
| Bug fix that doesn't change output | No | Fixing a logging format issue |
| New error/warning code | No | Adding `W_NEW_WARNING` |

### 26.2 Metadata Field Compatibility

- `PDFChunkMetadata` fields are **additive only**. New fields get default values (`None` or empty).
- Existing fields are **never removed or renamed** within a major version.
- Field type changes (e.g., `str` → `int`) are breaking and require a major version bump.

### 26.3 Error Code Compatibility

- New `ErrorCode` values may be added in any release.
- Existing values are **never removed or renamed** within a major version.
- Consumers should handle unknown error codes gracefully (log and continue).

### 26.4 Config Compatibility

- New `PDFProcessorConfig` fields may be added with default values in any release.
- Changing a default value requires a `parser_version` bump (see §26.1).
- Removing a config field is breaking and requires a major version bump.

---

## 27. Requirements Traceability

### 27.1 Traceability Matrix

| Req ID | Requirement | Module | Test Cases | Phase | Gate |
|--------|------------|--------|------------|-------|------|
| R-SEC-1 | Reject invalid PDFs (bad magic bytes) | `security.py` | `test_security::test_invalid_magic_bytes` | 1 | Phase 1 |
| R-SEC-2 | Reject decompression bombs | `security.py` | `test_security::test_decompression_bomb` | 1 | Phase 1 |
| R-SEC-3 | Reject embedded JavaScript | `security.py` | `test_security::test_js_rejection`, `test_security::test_js_override_with_reason` | 1 | Phase 1 |
| R-SEC-4 | Reject oversized files | `security.py` | `test_security::test_too_large` | 1 | Phase 1 |
| R-SEC-5 | Security override requires reason | `config.py`, `security.py` | `test_security::test_override_without_reason_fails` | 1 | Phase 1 |
| R-SEC-6 | Override audit logging | `security.py` | `test_security::test_override_audit_log` | 1 | Phase 1 |
| R-QA-1 | Quality scoring composite formula | `quality.py` | `test_quality::test_score_high`, `test_quality::test_score_low` | 1 | Phase 1 |
| R-QA-2 | Auto OCR fallback on LOW quality | `quality.py`, `router.py` | `test_quality::test_ocr_fallback_trigger`, `test_router::test_auto_ocr_fallback` | 1 | Phase 1 |
| R-CLS-1 | Tier 1 rule-based classification | `inspector.py` | `test_inspector::test_text_native`, `test_inspector::test_scanned`, `test_inspector::test_complex` | 1 | Phase 1 |
| R-CLS-2 | Tier 2 LLM classification | `llm_classifier.py` | `test_llm_classifier::test_tier2_classify` | 1 | Phase 1 |
| R-CLS-3 | Tier 3 reasoning classification | `llm_classifier.py` | `test_llm_classifier::test_tier3_classify` | 1 | Phase 1 |
| R-CLS-4 | LLM outage → Tier 1 degrade | `router.py`, `llm_classifier.py` | `test_router::test_llm_outage_degrades_to_tier1` | 1 | Phase 1 |
| R-CLS-5 | Degraded classification emits warnings | `router.py` | `test_router::test_degraded_warnings` | 1 | Phase 1 |
| R-PA-1 | Path A text extraction | `text_extractor.py` | `test_text_extractor::test_markdown_extraction` | 1 | Phase 1 |
| R-PA-2 | Heading hierarchy detection | `heading_detector.py` | `test_text_extractor::test_heading_hierarchy` | 1 | Phase 1 |
| R-PA-3 | Header/footer stripping | `header_footer.py` | `test_header_footer::test_cross_page_detection` | 1 | Phase 1 |
| R-PB-1 | Path B OCR with Tesseract | `ocr_processor.py` | `test_ocr_processor::test_tesseract_path` | 1 | Phase 1 |
| R-PB-2 | Path B OCR with PaddleOCR | `ocr_processor.py` | `test_ocr_processor::test_paddleocr_path` | 1.1 | Phase 1.1 |
| R-PB-3 | OCR engine fallback chain | `ocr_engines.py` | `test_ocr_engines::test_fallback_paddle_to_tesseract` | 1.1 | Phase 1.1 |
| R-PB-4 | OCR preprocessing steps | `ocr_processor.py` | `test_ocr_processor::test_preprocessing` | 1 | Phase 1 |
| R-PB-5 | Low confidence flagging | `ocr_processor.py` | `test_ocr_processor::test_low_confidence_warning` | 1 | Phase 1 |
| R-PC-1 | Path C page-level routing | `complex_processor.py` | `test_complex_processor::test_page_routing` | 1 | Phase 1 |
| R-PC-2 | Table extraction | `table_extractor.py` | `test_table_extractor::test_pdfplumber_extraction` | 1 | Phase 1 |
| R-PC-3 | Multi-page table stitching | `table_extractor.py` | `test_table_extractor::test_continuation_stitching` | 1 | Phase 1 |
| R-PC-4 | Form field extraction | `complex_processor.py` | `test_complex_processor::test_form_fields` | 1 | Phase 1 |
| R-CHK-1 | Heading-aware chunking | `chunker.py` | `test_chunker::test_heading_boundaries` | 1 | Phase 1 |
| R-CHK-2 | Table-aware chunking | `chunker.py` | `test_chunker::test_table_not_split` | 1 | Phase 1 |
| R-RTR-1 | Router full flow per type | `router.py` | `test_router::test_full_flow_type_a`, `test_router::test_full_flow_type_b`, `test_router::test_full_flow_type_c` | 1 | Phase 1 |
| R-RTR-2 | Process isolation (timeout) | `router.py` | `test_router::test_process_isolation_timeout` | 1 | Phase 1 |
| R-RTR-3 | WrittenArtifacts tracking | `router.py` | `test_router::test_written_artifacts` | 1 | Phase 1 |
| R-EXE-1 | Async wrapper `aprocess()` | `router.py` | `test_router::test_aprocess_matches_process` | 1.1 | Phase 1.1 |
| R-EXE-2 | ExecutionBackend protocol | `router.py` | `test_router::test_execution_backend_local` | 2 | Phase 2 |
| R-PLT-1 | Tesseract available on all targets | CI matrix | CI job `test-arm64` passes | 1.1 | Phase 1.1 |
| R-PLT-2 | PaddleOCR fallback to Tesseract | `ocr_engines.py` | `test_ocr_engines::test_fallback_paddle_to_tesseract` | 1.1 | Phase 1.1 |
| R-OBS-1 | PII-safe logging | `router.py` | `test_router::test_no_pii_in_logs` | 1 | Phase 1 |
| R-OBS-2 | Throughput meets SLO | Benchmark suite | `benchmark-report-<date>.json` | 2 | Phase 2 |

### 27.2 Maintenance

This table is updated each phase:
- New requirements → new rows.
- Completed test cases → verified against actual test names post-implementation.
- Phase changes → updated phase/gate columns.
- Table is the source of truth for "is this requirement covered?" during PROVE verification.

---

## 28. Future Considerations (not in scope for current phases)

- **Contextual retrieval enrichment**: Prepend LLM-generated context to each chunk before embedding (Anthropic pattern, 49-67% retrieval improvement). Deferred due to cost per chunk.
- **Semantic deduplication**: MinHash LSH or embedding similarity for near-duplicate chunk detection across documents.
- **Embedded file extraction**: Recursively process PDF portfolios/attachments.
- **Docling integration as a processing path**: Use Docling's TableFormer for higher-accuracy table extraction.
- **Surya as an OCR option**: Add Surya OCR engine for layout-aware text detection with reading order.
- **Vision model integration for Tier 3**: Render sample pages and send to a vision-capable LLM for layout analysis.
- **GPU batch inference**: Support GPU-accelerated OCR with configurable batch sizes.
- **PDF/A detection**: Use conformance level as a quality signal to skip guard checks.
- **Shared `ingestkit-core` package**: Extract protocols and common models for cross-package reuse.
- **Plugin discovery via entry points**: Auto-detect installed ingestkit packages.

See `ROADMAP.md` for the full deferred items list with rationale.
