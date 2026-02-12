# ingestkit-forms --- Technical Specification

**Version:** 1.0
**Status:** DRAFT --- v1.2 technology stack
**Changelog:**
- v1.2: Technology stack decisions applied. PaddleOCR promoted to primary OCR engine. PyMuPDF licensing governance with MIT-safe alternative path. Added VLM fallback tier (Qwen2.5-VL via Ollama). Dependencies restructured into optional groups. New protocols: VLMBackend, PDFWidgetBackend. New config params for VLM. New error codes for VLM. Fallback chain formalized: native → OCR → (optional) VLM.
- v1.1: Applied engineering review feedback. Resolved P0 dual-write consistency, table versioning, and idempotency. Added P1 rollback protocol, manual override validation, redaction targeting, multi-page windowed matching, and review-band contract. Added P2 acceptance criteria, security test requirements, config schema hardening, and observability contract. Resolved all P3 open questions.
- v1.0: Initial draft covering form template system, multi-source extraction, dual-write output, admin UI contract, and pipeline integration as Path F.

## 1. Overview & Motivation

**Package:** `ingestkit-forms`
**Python package name:** `ingestkit_forms`
**Parent ecosystem:** `ingestkit` --- a plugin-based ingestion framework for the "AI Help Desk in a Box" on-premises RAG system.

This package provides template-driven form extraction for structured documents (fillable PDFs, scanned forms, Excel-based forms, and photographed paper forms), producing both structured database rows and RAG-ready vector chunks.

### 1.1 Problem Statement

Organizations accumulate thousands of filled-out forms: leave requests, W-4s, safety checklists, equipment check-out sheets, incident reports. These forms share a common structure:

- **Predictable layout.** The same form is filled out hundreds or thousands of times. Field positions are stable across instances.
- **Mixed input methods.** The same form type might arrive as a fillable PDF, a scanned printout, a photograph taken on a phone, or an Excel worksheet with designated input cells.
- **Dual query needs.** Admins need SQL-style queries ("how many leave requests in Q1?") AND semantic search ("find incident reports mentioning chemical spill").

Existing Path A/B/C processing treats each form as an unknown document, classifying from scratch. This wastes compute and produces inconsistent field extraction because the pipeline does not know which fields to look for. Template-driven extraction solves this: define the form layout once, then extract fields reliably from every instance.

### 1.2 Scope

**In scope:**
- Form template definition with field-level bounding box mappings.
- Template versioning (forms change over time; old templates remain accessible).
- Layout fingerprinting for automatic template matching.
- Three extraction backends with fallback chain: native PDF form fields (PyMuPDF or MIT-safe alternative via `PDFWidgetBackend` protocol), OCR overlay for scanned/image forms (PaddleOCR primary, Tesseract fallback), and Excel cell mapping (openpyxl). Optional fourth tier: local VLM fallback for low-confidence OCR fields.
- Per-field confidence scoring.
- Dual-write output: structured DB rows + RAG chunks with rich metadata.
- Plugin API surface for admin UI integration (template CRUD, document preview, extraction preview, matching).
- Pipeline integration as Path F, invoked before the standard Inspector/Classifier pipeline.
- Optional local VLM fallback tier (Qwen2.5-VL or equivalent via Ollama) for fields where OCR confidence is below threshold. Opt-in; the pipeline operates without VLM by default.
- Deterministic idempotency keying (reuses `IngestKey` pattern).
- PII-safe logging with configurable redaction.
- Normalized error codes extending the existing `ErrorCode` taxonomy.

**Out of scope:**
- Admin UI implementation (this package provides the API; the UI is a separate concern).
- Form generation or PDF creation.
- Handwriting recognition beyond what OCR engines provide.
- Cloud OCR backends (Google Vision, AWS Textract, Azure Document Intelligence).
- Cloud LLM backends for field interpretation.
- Training custom ML models for form detection.
- Query-time retrieval engine or SQL agent.
- Multi-tenant isolation enforcement (package propagates `tenant_id`; caller enforces).

---

## 2. Use Cases

### UC-1: Admin Creates a Form Template
1. Admin uploads a sample leave request PDF via the admin UI.
2. The plugin renders the document as an image for the UI to display.
3. Admin draws bounding boxes on the rendered image, labeling each: "employee_name" (TEXT), "leave_type" (DROPDOWN), "start_date" (DATE), etc.
4. Admin saves the template. The plugin computes a layout fingerprint and stores the template definition.

### UC-2: Automatic Form Matching
1. An employee submits a filled-out leave request PDF.
2. The orchestration layer passes the file to the ingest pipeline.
3. The Form Matcher compares the document's layout to all stored templates.
4. Confidence exceeds threshold (0.8) for the "Leave Request v2" template.
5. The document is routed to Path F, bypassing normal Inspector/Classifier.

### UC-3: Manual Template Assignment
1. A batch of scanned safety checklists arrives with poor scan quality.
2. Auto-detect confidence falls below threshold.
3. Admin manually assigns the "Safety Checklist" template ID in the admin UI.
4. The document is routed to Path F with the specified template.

### UC-4: Field Extraction and Dual Write
1. Path F applies the matched template to the document.
2. For each field: extract value, compute confidence, validate against rules.
3. Results are written as a structured DB row (one row per form, columns per field).
4. Results are serialized as RAG chunks with field name/value pairs and rich metadata.
5. Both outputs include the template_id and version for traceability.

### UC-5: Form Version Migration
1. HR updates the leave request form, adding a "Remote Work" checkbox.
2. Admin creates version 3 of the "Leave Request" template with the new field.
3. Old submissions continue to match version 2 (fingerprint match).
4. New submissions match version 3.
5. Both versions coexist; the structured DB table gains a new nullable column.

### UC-6: Excel-Based Forms
1. IT uses an Excel workbook where cell B2 is "Requester Name", cell B5 is "Equipment Type", etc.
2. Admin creates a template mapping cell coordinates to field names.
3. When an instance arrives, the plugin reads designated cells directly --- no OCR needed.

---

## 3. Supported Form Sources

| Source Type | File Extensions | Extraction Method | Fallback Chain | Notes |
|---|---|---|---|---|
| Fillable PDF | `.pdf` | Native form field extraction (PyMuPDF or MIT-safe alternative) | native → OCR → VLM (optional) | Fastest; highest confidence. Falls back to OCR overlay if fields are flattened. |
| Scanned PDF | `.pdf` | OCR overlay extraction (PaddleOCR primary, Tesseract fallback) | OCR → VLM (optional) | Page rendered at configured DPI; template overlay applied; each field region OCR'd independently. |
| Image (photographed/scanned) | `.jpg`, `.jpeg`, `.png`, `.tiff`, `.tif` | OCR overlay extraction (PaddleOCR primary, Tesseract fallback) | OCR → VLM (optional) | Image loaded directly; same overlay pipeline as scanned PDF pages. |
| Excel form | `.xlsx` | Cell mapping (openpyxl) | N/A | Template maps field names to cell addresses (e.g., `B2`, `D5:D7`). Direct cell value read via openpyxl. No fallback needed. |

### 3.1 Source Detection

The plugin determines extraction method per document:

```
Input file
    |
    v
Is it .xlsx? ------YES-----> ExcelCellExtractor
    |
    NO
    |
    v
Is it .pdf? -------YES-----> Does it have native form fields?
    |                              |
    NO                         YES --> NativePDFExtractor ----+
    |                              |                          |
    v                          NO  --> Are pages scanned?     |
Is it an image? ---YES--->          |                          |
    |                          YES --> OCROverlayExtractor -+  |
    NO                             |                        |  |
    |                          NO  --> OCROverlayExtractor  |  |
    v                                  (flattened form) ----+  |
Reject with                                                 |  |
E_FORM_UNSUPPORTED_FORMAT          +------------------------+--+
                                   |
                                   v
                        Per-field confidence check
                                   |
                        Any field < form_vlm_fallback_threshold
                        AND form_vlm_enabled=True?
                                   |
                            +------+------+
                            |             |
                           YES           NO
                            |             |
                            v             v
                    VLMFieldExtractor   Return results
                    (per-field only,     as-is
                     budget-limited)
                            |
                            v
                      Return results
                      (with VLM-enhanced fields)
```

**3-tier fallback chain:** The extraction pipeline follows a deterministic fallback order: **native → OCR → (optional) VLM**. Each tier is attempted per-field, not per-document. A field that succeeds at the native tier is never sent to OCR. A field that succeeds at OCR with confidence above `form_vlm_fallback_threshold` is never sent to VLM. This minimizes cost and latency while maximizing extraction quality.

---

## 4. Architecture & Pipeline Integration

### 4.1 Design Principles

1. **Backend-agnostic core.** All processing logic references Protocol types, never concrete backends.
2. **Dependency injection.** The `FormRouter` accepts backend instances; it never creates them.
3. **Structural subtyping.** Backends use `typing.Protocol` (not ABCs) --- implement the methods and it works.
4. **Template-driven extraction.** The plugin never guesses at form structure. If no template matches (and none is manually assigned), the document falls through to the standard pipeline.
5. **Fail-closed.** If field extraction confidence is below threshold, the field value is `None` with a warning --- never fabricated. If overall extraction quality is too low, return `ProcessingResult` with `E_FORM_EXTRACTION_LOW_CONFIDENCE` and zero chunks.
6. **PII-safe observability.** No extracted field values in logs unless `log_sample_data=True`. Template names and field names (not values) are always safe to log.
7. **Idempotent by default.** Every ingest produces a deterministic `IngestKey`; re-ingesting the same form with the same template version produces the same key.
8. **Graceful pipeline fallthrough.** If the Form Matcher finds no match and no manual template is assigned, the document continues to the standard Inspector/Classifier pipeline with zero overhead beyond the matching check.

### 4.2 Pipeline Integration

```
Input document
    |
    v
+-----------------------------+
|  Pre-flight Security Scan   |  (existing stage, unchanged)
+-------------+---------------+
              |
              v
+-----------------------------+
|  Compute ingest_key         |  content_hash + source_uri + parser_version
|  (idempotency)              |
+-------------+---------------+
              |
              v
+-----------------------------+
|  Form Matcher               |  Compare layout to stored templates
|  (Path F gate)              |  Only runs if form_match_enabled=True
+-------------+---------------+
              |
     match found? (confidence >= threshold)
     OR manual template_id provided?
              |
      +-------+--------+
      |                 |
     YES               NO
      |                 |
      v                 v
+------------------+  +-----------------------------+
| Path F:          |  | Standard pipeline:          |
| Form Extraction  |  | Inspector -> Classifier ->  |
|                  |  | Path A / B / C              |
+--------+---------+  +-----------------------------+
         |
         v
+------------------+
| Dual Write:      |
| DB row + Chunks  |
+------------------+
         |
         v
  FormProcessingResult
```

**Key integration point:** The Form Matcher runs AFTER security scanning and idempotency keying but BEFORE the standard Inspector/Classifier. This means:
- Security violations are caught before any template matching.
- The ingest key is available for dedup regardless of which path processes the document.
- If form matching declines the document (no match, disabled, or below threshold), the document enters the standard pipeline at the Inspector stage with no state mutation.

### 4.3 Idempotency Keys

Form extraction uses two deterministic keys for deduplication at different scopes:

**Global ingest key** (document-level dedup):
```python
ingest_key_global = sha256(content_hash + source_uri + parser_version)
```
Used for: skipping re-ingestion of identical documents regardless of template. If `ingest_key_global` already exists in the collection, the document has been ingested before.

**Form extraction key** (template-versioned dedup):
```python
form_extraction_key = sha256(ingest_key_global + template_id + template_version)
```
Used for: deterministic chunk IDs and DB row `_form_id`. This key changes when:
- The document content changes (different `content_hash`)
- A different template or template version is applied

**Where each key is used:**

| Key | Used As |
|-----|---------|
| `ingest_key_global` | Dedup gate at pipeline entry. Stored in `ChunkMetadata.ingest_key`. |
| `form_extraction_key` | DB upsert key (`_form_id`). Vector point ID seed (`uuid5(NAMESPACE, form_extraction_key + chunk_index)`). |

**Re-extraction semantics:** If the same document is re-ingested with a newer template version, `ingest_key_global` is unchanged but `form_extraction_key` differs. This produces new DB rows (with the new `_template_version`) and new vector points. The old extraction's artifacts remain unless explicitly cleaned up via `rollback_written_artifacts()`.

### 4.4 Module Structure

```
ingestkit-forms/
+-- pyproject.toml
+-- SPEC.md                        # This document
+-- ROADMAP.md                     # Deferred features
+-- src/
|   +-- ingestkit_forms/
|       +-- __init__.py            # Exports: FormRouter, create_default_router
|       +-- protocols.py           # FormTemplateStore, OCRBackend (form-specific protocols)
|       +-- models.py              # Pydantic models for all data structures
|       +-- errors.py              # Normalized error codes extending base taxonomy
|       +-- config.py              # FormProcessorConfig with all thresholds and defaults
|       +-- idempotency.py         # Reuses IngestKey pattern from ingestkit-core
|       +-- matcher.py             # Form Matcher: layout fingerprinting + template matching
|       +-- router.py              # FormRouter: orchestrates matching -> extraction -> output
|       +-- extractors/
|       |   +-- __init__.py
|       |   +-- native_pdf.py     # NativePDFExtractor: PyMuPDF form field extraction
|       |   +-- ocr_overlay.py    # OCROverlayExtractor: render + template overlay + per-field OCR
|       |   +-- excel_cell.py     # ExcelCellExtractor: openpyxl cell value mapping
|       +-- output/
|       |   +-- __init__.py
|       |   +-- db_writer.py      # Structured DB row writer
|       |   +-- chunk_writer.py   # RAG chunk serializer + embedder
|       +-- api.py                 # Plugin API surface (template CRUD, preview, matching)
+-- tests/
    +-- __init__.py
    +-- conftest.py                # Shared fixtures, mock backends
    +-- test_matcher.py
    +-- test_extractors.py
    +-- test_output.py
    +-- test_api.py
    +-- test_router.py
    +-- test_config.py
    +-- fixtures/                  # Sample form files (PDF, Excel, images)
```

---

## 5. Form Template System

### 5.1 Template Data Model

A `FormTemplate` defines the reusable structure of a form: its fields, their locations, types, and validation rules. Templates are the central concept --- all extraction is template-driven.

```python
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class SourceFormat(str, Enum):
    """File format the template was designed for."""

    PDF = "pdf"
    XLSX = "xlsx"
    IMAGE = "image"


class FieldType(str, Enum):
    """Data type of a form field."""

    TEXT = "text"
    NUMBER = "number"
    DATE = "date"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    SIGNATURE = "signature"
    DROPDOWN = "dropdown"


class BoundingBox(BaseModel):
    """Normalized bounding box for a field region.

    All coordinates are normalized to 0.0-1.0 relative to page dimensions.
    This ensures templates work across different scan resolutions, DPIs,
    and page sizes.
    """

    x: float = Field(ge=0.0, le=1.0, description="Left edge, normalized 0.0-1.0")
    y: float = Field(ge=0.0, le=1.0, description="Top edge, normalized 0.0-1.0")
    width: float = Field(gt=0.0, le=1.0, description="Width, normalized 0.0-1.0")
    height: float = Field(gt=0.0, le=1.0, description="Height, normalized 0.0-1.0")


class CellAddress(BaseModel):
    """Excel cell address or range for cell-mapping extraction.

    Used only when source_format is XLSX. Specifies the exact cell(s) to
    read for a given field.
    """

    cell: str = Field(description="Cell address, e.g. 'B2' or range 'D5:D7'")
    sheet_name: str | None = Field(
        default=None,
        description="Sheet name. None means the active/first sheet.",
    )


class FieldMapping(BaseModel):
    """Mapping of a single form field within a template.

    Defines where the field is located, what type of data it contains,
    and how to validate the extracted value.
    """

    field_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this field within the template.",
    )
    field_name: str = Field(
        description="Machine-readable name, e.g. 'employee_name', 'leave_type'.",
    )
    field_label: str = Field(
        description="Human-readable display label, e.g. 'Employee Name'.",
    )
    field_type: FieldType
    page_number: int = Field(
        ge=0,
        description="0-indexed page number where this field appears.",
    )
    region: BoundingBox | None = Field(
        default=None,
        description="Bounding box for PDF/image extraction. None for Excel.",
    )
    cell_address: CellAddress | None = Field(
        default=None,
        description="Cell address for Excel extraction. None for PDF/image.",
    )
    required: bool = False
    validation_pattern: str | None = Field(
        default=None,
        description="Regex pattern for value validation, e.g. r'^\\d{3}-\\d{2}-\\d{4}$' for SSN.",
    )
    default_value: str | None = None
    options: list[str] | None = Field(
        default=None,
        description="Valid options for CHECKBOX, RADIO, or DROPDOWN fields.",
    )
    extraction_hint: str | None = Field(
        default=None,
        description=(
            "Optional hint to improve OCR accuracy, e.g. 'numeric_only', "
            "'uppercase', 'date_format:MM/DD/YYYY'."
        ),
    )


class FormTemplate(BaseModel):
    """Reusable form template definition.

    Represents the complete structural definition of a form type.
    Created by an admin via the UI; applied to incoming documents
    during extraction.
    """

    template_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique template identifier (UUID).",
    )
    name: str = Field(
        description="Human-readable template name, e.g. 'W-4 2026'.",
    )
    description: str = Field(
        default="",
        description="Optional description of the form type.",
    )
    version: int = Field(
        default=1,
        ge=1,
        description="Template version. Incremented when the form layout changes.",
    )
    source_format: SourceFormat
    page_count: int = Field(
        ge=1,
        description="Number of pages in the form (1 for single-page, >1 for multi-page).",
    )
    fields: list[FieldMapping] = Field(
        min_length=1,
        description="At least one field mapping is required.",
    )
    layout_fingerprint: bytes | None = Field(
        default=None,
        description="Structural fingerprint for auto-matching (see section 5.4).",
    )
    thumbnail: bytes | None = Field(
        default=None,
        description="Optional thumbnail image for display in admin UI.",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = Field(
        default="system",
        description="User or system identifier that created this template.",
    )
    tenant_id: str | None = None

    class Config:
        """Pydantic v2 model configuration."""
        json_encoders = {
            bytes: lambda v: v.hex() if v else None,
        }
```

### 5.2 Field Mapping

Field mappings use a dual-addressing scheme:

- **BoundingBox** for PDF and image forms: normalized coordinates (0.0-1.0) relative to page dimensions. Normalization ensures the same template works across different scan DPIs and page sizes.
- **CellAddress** for Excel forms: direct cell references (e.g., `B2`, `D5:D7`) optionally scoped to a sheet name.

A single `FieldMapping` has exactly one of `region` or `cell_address` populated, determined by the template's `source_format`. Validation enforces this:

```python
from pydantic import model_validator

class FieldMapping(BaseModel):
    # ... fields as above ...

    @model_validator(mode="after")
    def _validate_address_type(self) -> FieldMapping:
        """Ensure exactly one addressing scheme is set based on context.

        PDF/image templates require ``region``; Excel templates require
        ``cell_address``. This is enforced at the template level during
        creation, not on the individual mapping, since the mapping does
        not know its parent template's source_format.
        """
        if self.region is None and self.cell_address is None:
            raise ValueError(
                f"Field '{self.field_name}' must have either 'region' "
                f"(for PDF/image) or 'cell_address' (for Excel)."
            )
        if self.region is not None and self.cell_address is not None:
            raise ValueError(
                f"Field '{self.field_name}' cannot have both 'region' and "
                f"'cell_address'. Use one addressing scheme."
            )
        return self
```

### 5.3 Template Versioning

Templates are versioned to handle form layout changes over time:

- **Version increment:** When an admin updates a template (adds/removes/moves fields), the version number increments. The previous version is retained.
- **Immutable snapshots:** Once a template version is used to extract a form, that version becomes immutable. Edits create a new version.
- **Fingerprint per version:** Each version has its own `layout_fingerprint`. The matcher considers all active versions when auto-detecting.
- **DB schema evolution:** When a new version adds fields, the structured DB table gains new nullable columns. Removed fields remain in the table as nullable (no data loss).
- **Metadata traceability:** Every `FormExtractionResult` records the `template_id` AND `template_version` used, so results are always traceable to the exact template definition.

Version lifecycle:

```
Template "Leave Request" v1  -->  active, fingerprint_v1
                    |
                    | Admin edits: adds "Remote Work" checkbox
                    v
Template "Leave Request" v2  -->  active, fingerprint_v2
                                  v1 remains stored but not used for new matching
```

The `FormTemplateStore` protocol (section 9) defines storage operations. The store must support:
- Retrieval by `(template_id, version)` for exact lookups.
- Retrieval of the latest version by `template_id`.
- Listing all versions of a template.
- Listing all active (latest-version) templates for matching.

### 5.4 Layout Fingerprinting

Layout fingerprinting enables automatic template matching without requiring the admin to manually tag every incoming document.

**Fingerprint computation:**

The fingerprint captures the structural layout of a form --- the positions of text anchors, lines, boxes, and field regions --- without capturing the actual content. Two instances of the same form (filled out differently) should produce similar fingerprints.

```
Fingerprint computation:
    1. Render document page(s) at standardized resolution (150 DPI for fingerprinting).
    2. Convert to grayscale.
    3. Apply adaptive thresholding to isolate structural elements
       (lines, boxes, text regions) from background.
    4. Compute a structural hash:
       a. Divide page into an N x M grid (default: 16 x 20).
       b. For each cell: compute fill ratio (dark pixels / total pixels).
       c. Quantize fill ratios to 4 levels: empty (0), sparse (1), partial (2), dense (3).
       d. The resulting N x M matrix of quantized values IS the fingerprint.
    5. For multi-page forms: concatenate per-page fingerprints.
```

**Fingerprint comparison (similarity score):**

```python
def compute_layout_similarity(fp_a: bytes, fp_b: bytes) -> float:
    """Compare two layout fingerprints.

    Returns a similarity score between 0.0 (no match) and 1.0 (identical layout).

    Algorithm:
        1. Deserialize both fingerprints into NxM matrices.
        2. If page counts differ, score is 0.0 (structural mismatch).
        3. For each grid cell: compute agreement.
           - Exact match: 1.0
           - Off by one quantization level: 0.5
           - Off by two or more: 0.0
        4. Similarity = sum(cell_scores) / total_cells
    """
    ...
```

**Why not perceptual hashing (pHash)?** Perceptual hashes capture visual similarity including content. Two forms with different handwritten entries would have different pHashes. The grid-based structural fingerprint ignores content and captures only layout geometry, which is what we need for template matching.

---

## 6. Form Matching

### 6.1 Auto-Detection Algorithm

When a document enters the pipeline and `form_match_enabled=True`, the Form Matcher runs before the standard Inspector/Classifier:

```
def match_document(file_path: str) -> list[TemplateMatch]:
    """Match an incoming document against all active templates.

    Algorithm:
        1. Determine source format from file extension.
        2. Load all active templates matching that source format.
           (Optimization: skip templates for different formats.)
        3. Compute the incoming document's layout fingerprint.
        4. For each candidate template:
           a. Compute layout similarity between document fingerprint
              and template fingerprint.
           b. If similarity >= form_match_confidence_threshold:
              add to matches.
        5. Sort matches by confidence descending.
        6. Return ranked list of matches.

    Returns:
        List of TemplateMatch objects, sorted by confidence descending.
        Empty list if no template exceeds the confidence threshold.
    """
```

**Performance optimization:** Fingerprint computation is the expensive step. The incoming document's fingerprint is computed once and compared against all candidate templates. Template fingerprints are pre-computed and cached at template creation time.

**Multi-page matching with windowed alignment:** For multi-page forms, the matcher uses a sliding window to handle real-world documents that may have extra pages (cover sheets, appended scans, blank trailing pages):

1. Let `T` = template page count, `D` = document page count.
2. If `D < T`: no match possible (document has fewer pages than template).
3. If `D == T`: compare all pages 1:1.
4. If `D > T`: slide a window of size `T` across all `D` pages. For each window position `i` (where `i` ranges from 0 to `D - T`):
   a. Compare template page `j` against document page `i + j` for all `j` in `0..T-1`.
   b. Compute per-page similarity for this window.
   c. All pages in the window must exceed `form_match_per_page_minimum` (default 0.6).
   d. Apply a configurable penalty for unmatched extra pages: `penalty = unmatched_pages * form_match_extra_page_penalty` (default 0.02 per extra page).
   e. Window confidence = mean(per_page_similarities) - penalty.
5. Select the window with the highest confidence.
6. Report: overall confidence, best window start page, per-page scores.

This handles: cover sheets prepended to forms, blank pages appended by scanners, and multi-form scan batches where the form of interest is embedded within a larger document.

### 6.2 Confidence Scoring

The `TemplateMatch` model captures match quality:

```python
class TemplateMatch(BaseModel):
    """Result of matching a document against a single template."""

    template_id: str
    template_name: str
    template_version: int
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Overall match confidence.",
    )
    per_page_confidence: list[float] = Field(
        description="Per-page similarity scores.",
    )
    matched_features: list[str] = Field(
        description=(
            "Which features contributed to the match: "
            "'layout_grid', 'text_anchors', 'field_positions'."
        ),
    )
```

Confidence thresholds and their actions:

| Confidence Range | Action | Rationale |
|---|---|---|
| >= 0.8 (configurable) | Auto-apply template, proceed to Path F | High structural similarity; safe to extract automatically. |
| 0.5 -- <threshold | Emit `W_FORM_MATCH_BELOW_THRESHOLD` with structured metadata (top-N candidates with confidences). Fall through to standard pipeline. The warning payload includes: `top_matches: list[TemplateMatch]` (up to 3 candidates), enabling the orchestration layer to present review options. The document is NOT held --- it proceeds through Inspector/Classifier immediately. |
| < 0.5 | No match; fall through to standard pipeline | Low similarity; document is not a known form type. |

### 6.3 Manual Override

The admin can bypass auto-detection by providing a `template_id` at ingestion time:

```python
class FormIngestRequest(BaseModel):
    """Request to ingest a form document.

    If template_id is provided, auto-detection is skipped and the
    specified template is applied directly.
    """

    file_path: str
    template_id: str | None = None
    template_version: int | None = Field(
        default=None,
        description="Specific version. None means latest.",
    )
    tenant_id: str | None = None
    source_uri: str | None = None
    metadata: dict[str, str] | None = None
```

When `template_id` is provided:
1. The matcher is skipped entirely.
2. The specified template (and version, if given) is loaded directly.
3. Extraction proceeds with that template regardless of layout similarity.
4. The `FormExtractionResult.match_method` field is set to `"manual_override"`.

This is the escape hatch for documents that auto-detection cannot handle: poor scan quality, unusual paper sizes, or forms not yet fingerprinted.

**Format compatibility check:** When `template_id` is provided via manual override:

1. Resolve the template (and version, if specified; otherwise latest).
2. If the template's `source_format` does not match the input file's detected format, return `E_FORM_TEMPLATE_INVALID` with message "Template source_format '{template_format}' incompatible with input format '{input_format}'" **before extraction starts**. No extraction is attempted.
3. If `template_version` is specified but does not exist, return `E_FORM_TEMPLATE_NOT_FOUND`.
4. If `template_version` is not specified, the latest active version is used.

This prevents silent misrouting where, e.g., a PDF template is applied to an Excel file, which would cause extractor failures downstream.

---

## 7. Field Extraction

### 7.1 Native PDF Form Fields

For fillable PDFs with intact form fields (not flattened):

```
def extract_native_pdf_fields(
    file_path: str,
    template: FormTemplate,
) -> list[ExtractedField]:
    """Extract form field values from a fillable PDF using PyMuPDF.

    Algorithm:
        1. Open PDF with PyMuPDF (fitz).
        2. Iterate over form fields (widgets) on each page.
        3. For each template field mapping:
           a. Find the PDF widget whose bounding box overlaps the
              template field's bounding box (normalized coordinates).
           b. Overlap threshold: >= 50% IoU (Intersection over Union).
           c. If a matching widget is found:
              - Read the widget value.
              - Confidence = 0.95 (native fields are highly reliable).
              - extraction_method = "native_fields".
           d. If no matching widget is found:
              - Fall back to OCR overlay for this specific field.
              - Confidence = OCR confidence (typically lower).
              - extraction_method = "native_fields_with_ocr_fallback".
        4. Validate extracted values against field type and validation_pattern.
        5. Return list of ExtractedField objects.
    """
```

**Widget-to-field matching:** PDF form fields have their own bounding boxes in PDF coordinate space (points from bottom-left origin). These are converted to normalized coordinates (0.0-1.0 from top-left) to match the template's `BoundingBox` format. The IoU (Intersection over Union) threshold of 50% accommodates minor positioning differences between the template and actual PDF fields.

**Flattened form detection:** If a PDF has zero widgets but the template expects native fields, the extractor detects this and automatically falls back to OCR overlay for the entire document, logging a `W_FORM_FIELDS_FLATTENED` warning.

**Backend abstraction:** The `NativePDFExtractor` does not import PyMuPDF directly. It references the `PDFWidgetBackend` protocol (§15.3), and the concrete backend is injected at construction time. This enables swapping implementations without changing extraction logic.

#### 7.1.1 Licensing Governance

PyMuPDF (fitz) is licensed under **AGPL-3.0**. A commercial license is available from Artifex Software.

| Deployment Context | Recommended Backend | License |
|---|---|---|
| Internal/on-premises (no distribution) | PyMuPDF via `PDFWidgetBackend` | AGPL-3.0 (permissible for internal use) |
| Commercial distribution / SaaS | PyMuPDF with Artifex commercial license | Commercial |
| Strict MIT/BSD-only policy | `pdfplumber` (MIT) + `pypdf` (BSD) via `PDFWidgetBackend` | MIT + BSD |

The MIT-safe alternative path uses `pdfplumber` for widget extraction (via `Page.annots()` and coordinate-based text extraction) and `pypdf` for AcroForm field reading. This path has slightly lower accuracy for complex widget types (dropdowns, radio groups) but covers the common cases (text fields, checkboxes) reliably.

Both backends implement the same `PDFWidgetBackend` protocol. The choice is a deployment-time decision configured via the `pdf_widget_backend` config parameter (default: `"pymupdf"`).

### 7.2 OCR Overlay Extraction (Scanned/Image)

**Primary engine: PaddleOCR** (Apache-2.0). PaddleOCR is the default OCR engine for form field extraction. Its KIE module (Semantic Entity Recognition + Relation Extraction) produces higher accuracy on structured form layouts than general-purpose OCR engines. Per-region OCR with layout awareness makes it particularly effective for forms where field labels and values are spatially related.

**Fallback engine: Tesseract** (Apache-2.0). Tesseract serves as a lightweight fallback for environments where PaddlePaddle cannot be installed (macOS ARM without Rosetta, minimal Docker images, resource-constrained deployments). The `form_ocr_engine` config parameter controls engine selection.

For scanned PDFs, photographed forms, and flattened PDFs:

```
def extract_ocr_overlay(
    file_path: str,
    template: FormTemplate,
    config: FormProcessorConfig,
) -> list[ExtractedField]:
    """Extract field values by rendering the document and OCR-ing each field region.

    Algorithm:
        1. Render each page at configured DPI (default: 300).
           - PDF: render with PyMuPDF at target DPI.
           - Image: load directly; resize if resolution exceeds target DPI.
        2. For each template field on the current page:
           a. Convert normalized bounding box to pixel coordinates:
              px_x = region.x * page_width_px
              px_y = region.y * page_height_px
              px_w = region.width * page_width_px
              px_h = region.height * page_height_px
           b. Crop the field region from the rendered page image.
           c. Apply field-specific preprocessing:
              - TEXT: deskew, contrast enhancement
              - NUMBER: deskew, contrast, apply "digits only" OCR config
              - DATE: deskew, contrast, apply date-format hint
              - CHECKBOX: convert to binary, check fill ratio
                (fill_ratio > 0.3 = checked, else unchecked)
              - SIGNATURE: convert to binary, check ink ratio
                (ink_ratio > 0.05 = signed, else blank)
           d. For text-based fields (TEXT, NUMBER, DATE):
              - Run OCR engine on the cropped region.
              - OCR engine returns text + per-character confidence.
              - Field confidence = mean of per-character confidences.
           e. For CHECKBOX/RADIO:
              - Compute fill ratio (dark pixels / total pixels in region).
              - Checked threshold: fill_ratio > 0.3.
              - Confidence based on distance from threshold:
                abs(fill_ratio - 0.3) / 0.3, capped at 1.0.
           f. For SIGNATURE:
              - Compute ink ratio.
              - Value = True (signed) or False (blank).
              - Confidence = abs(ink_ratio - 0.05) / 0.05, capped at 1.0.
        3. Apply validation_pattern regex if defined.
           - If validation fails: set value to None, confidence to 0.0,
             add W_FORM_FIELD_VALIDATION_FAILED warning.
        4. Return list of ExtractedField objects.

    Per-field OCR is preferable to full-page OCR because:
        - Each field region can use optimized OCR settings (digits only,
          date format, etc.).
        - Confidence is per-field, not averaged across the page.
        - Adjacent field text does not bleed into the wrong field.
    """
```

**Image preprocessing pipeline for OCR regions:**

```
Cropped field region
    |
    v
Deskew (correct rotation up to +/- 15 degrees)
    |
    v
Contrast enhancement (CLAHE - Contrast Limited Adaptive Histogram Equalization)
    |
    v
Noise reduction (bilateral filter to preserve edges)
    |
    v
Binarization (adaptive threshold, Otsu's method)
    |
    v
OCR engine (PaddleOCR primary, Tesseract fallback)
    |
    v
Post-processing (strip whitespace, apply extraction_hint if present)
```

### 7.3 Excel Cell Mapping

For Excel-based forms:

```
def extract_excel_cells(
    file_path: str,
    template: FormTemplate,
) -> list[ExtractedField]:
    """Extract field values from designated Excel cells using openpyxl.

    Algorithm:
        1. Open workbook with openpyxl (read_only=False to access merged cells).
        2. For each template field:
           a. Resolve cell_address:
              - If sheet_name is specified: use that sheet.
              - Otherwise: use the active sheet.
              - If cell is a range (e.g., 'D5:D7'): read all cells in range,
                join non-empty values with newline.
           b. Read cell value.
           c. Handle merged cells: if the target cell is part of a merged range,
              read from the top-left cell of the merge.
           d. Type coercion:
              - NUMBER: attempt float(value); if fails, value = None.
              - DATE: attempt datetime parse; if fails, value = None.
              - CHECKBOX: interpret "X", "x", "Yes", "TRUE", 1 as True;
                empty, "No", "FALSE", 0, None as False.
              - TEXT: str(value), strip whitespace.
           e. Confidence:
              - Cell has a value: 0.95 (direct read, very reliable).
              - Cell is empty and field is required: 0.0, add warning.
              - Cell is empty and field is optional: 0.95, value = default_value.
           f. extraction_method = "cell_mapping".
        3. Apply validation_pattern regex if defined.
        4. Return list of ExtractedField objects.
    """
```

**Merged cell handling:** Excel forms often use merged cells for field values. The extractor detects merged ranges and reads from the canonical (top-left) cell. The template author does not need to account for merges --- they map the visible cell address, and the extractor resolves merges transparently.

### 7.4 Per-Field Confidence

Every extracted field carries an independent confidence score:

| Extraction Method | Typical Confidence Range | Factors |
|---|---|---|
| Native PDF fields | 0.90 -- 0.99 | Almost always reliable; minor deductions for unusual encodings. |
| OCR (clean scan, printed text) | 0.80 -- 0.95 | Per-character OCR confidence averaged. |
| OCR (poor scan, handwritten) | 0.40 -- 0.75 | Handwriting and noise reduce OCR confidence. |
| Excel cell mapping | 0.90 -- 0.99 | Direct cell read; deductions only for type coercion ambiguity. |
| Checkbox/radio (fill ratio) | 0.60 -- 0.99 | Distance from threshold determines confidence. |

**Confidence-based actions:**

| Field Confidence | Action |
|---|---|
| >= `form_extraction_min_field_confidence` (default 0.5) | Accept value, include in output. |
| >= `form_vlm_fallback_threshold` (default 0.4) AND < `form_extraction_min_field_confidence` | Accept value with `W_FORM_FIELD_LOW_CONFIDENCE` warning. VLM fallback NOT triggered (confidence is marginal but above VLM threshold). |
| < `form_vlm_fallback_threshold` (default 0.4) AND `form_vlm_enabled=True` | Trigger VLM fallback for this field (see §7.5). If VLM succeeds with confidence >= `form_extraction_min_field_confidence`, accept VLM value. Otherwise, retain OCR value with warning. |
| < `form_vlm_fallback_threshold` AND `form_vlm_enabled=False` | Set value to `None`, emit `W_FORM_FIELD_LOW_CONFIDENCE` warning. |

**Overall form confidence** is the mean confidence of all extracted fields, weighted by `required` status (required fields have 2x weight):

```python
def compute_overall_confidence(fields: list[ExtractedField], template: FormTemplate) -> float:
    """Compute weighted mean confidence across all extracted fields.

    Required fields have 2x weight. Fields with value=None contribute
    their (low) confidence to the average, pulling it down appropriately.
    """
    field_map = {f.field_id: f for f in template.fields}
    total_weight = 0.0
    weighted_sum = 0.0
    for ef in fields:
        weight = 2.0 if field_map.get(ef.field_id, FieldMapping()).required else 1.0
        weighted_sum += ef.confidence * weight
        total_weight += weight
    return weighted_sum / max(total_weight, 1.0)
```

### 7.5 VLM Fallback Extraction (Optional)

When OCR extraction produces low-confidence results for individual fields, an optional local Vision-Language Model (VLM) can be invoked as a third-tier fallback. This is **opt-in** via `form_vlm_enabled=True` (default: `False`).

**Reference model:** Qwen2.5-VL-7B-Instruct (Apache-2.0 license), served locally via Ollama. The model fits on consumer hardware (8GB+ VRAM) and produces structured JSON output with bounding box awareness.

**When triggered:** Per-field, after OCR extraction, when:
1. The field's OCR confidence is below `form_vlm_fallback_threshold` (default 0.4).
2. `form_vlm_enabled=True` in config.
3. The per-document VLM budget has not been exhausted (`form_vlm_max_fields_per_document`, default 10).

**Algorithm:**

```
def extract_field_vlm(
    page_image: bytes,
    field: FieldMapping,
    ocr_result: ExtractedField,
    config: FormProcessorConfig,
) -> ExtractedField:
    """Attempt VLM extraction for a single low-confidence field.

    Algorithm:
        1. Crop the field region from the page image (with 10% padding
           to provide context for the VLM).
        2. Construct a structured prompt:
           - Include field_name, field_type, and extraction_hint.
           - Request JSON output: {"value": "...", "confidence": 0.0-1.0}
           - For CHECKBOX/RADIO: request {"checked": true/false, "confidence": ...}
        3. Send cropped image + prompt to VLM backend.
        4. Parse and validate JSON response.
        5. If VLM confidence >= form_extraction_min_field_confidence:
           - Return VLM result with extraction_method = "vlm_fallback".
        6. If VLM confidence < form_extraction_min_field_confidence:
           - Return original OCR result (VLM did not improve).
           - Log W_FORM_VLM_FALLBACK_USED with both confidences.
        7. On VLM timeout or error:
           - Return original OCR result unchanged.
           - Log E_FORM_VLM_TIMEOUT or E_FORM_VLM_UNAVAILABLE.

    The VLM is never called for:
        - Excel cell mapping fields (always high confidence).
        - Native PDF widget fields (always high confidence).
        - Fields already above form_vlm_fallback_threshold.
    """
```

**Prompt template:**

```
Extract the value of the form field shown in this image.

Field name: {field_name}
Field type: {field_type}
{extraction_hint_line}

Respond with ONLY valid JSON:
{{"value": "<extracted value>", "confidence": <0.0 to 1.0>}}
```

**Budget guard:** To prevent runaway VLM costs on degraded documents (e.g., a heavily damaged scan where every field has low OCR confidence), the VLM is invoked for at most `form_vlm_max_fields_per_document` fields per document. Fields beyond the budget retain their OCR result, and `W_FORM_VLM_BUDGET_EXHAUSTED` is emitted.

**VLM field selection priority:** When more fields need VLM than the budget allows, required fields are prioritized over optional fields, and fields with the lowest OCR confidence are processed first.

**Protocol:** The VLM backend is abstracted via the `VLMBackend` protocol (§15.3). The default implementation calls Ollama's `/api/generate` endpoint with image input. Alternative implementations (vLLM, llama.cpp, etc.) can be injected.

---

## 8. Output: Dual-Write (DB + Chunks)

Every successfully extracted form produces two outputs: a structured database row and one or more RAG chunks. The dual-write behavior is governed by the `dual_write_mode` configuration (see §8.0).

### 8.0 Dual-Write Consistency Model

The dual-write behavior is governed by the `dual_write_mode` configuration:

```python
class DualWriteMode(str, Enum):
    """Controls dual-write failure semantics."""
    BEST_EFFORT = "best_effort"
    STRICT_ATOMIC = "strict_atomic"
```

**Behavior Matrix:**

| DB Write | Vector Write | `best_effort` | `strict_atomic` |
|----------|-------------|---------------|-----------------|
| Success | Success | `FormProcessingResult` with both artifacts. No errors. | Same. |
| Success | Failure | Result with `written.db_row_ids` populated, `written.vector_point_ids` empty. `E_FORM_CHUNK_WRITE_FAILED` in `error_details`. `warnings` includes partial-write notice. **DB row is retained.** | Rollback: delete DB row. Result with zero artifacts, both error codes in `error_details`. |
| Failure | Success | Result with `written.vector_point_ids` populated, `written.db_row_ids` empty. `E_FORM_DB_WRITE_FAILED` in `error_details`. **Vector points are retained.** | Rollback: delete vector points. Result with zero artifacts, both error codes in `error_details`. |
| Failure | Failure | Result with zero artifacts. Both `E_FORM_DB_WRITE_FAILED` and `E_FORM_CHUNK_WRITE_FAILED` in `error_details`. | Same. |

**Retry semantics (both modes):**
- Each backend write is retried up to `backend_max_retries` times with exponential backoff (`backend_backoff_base`).
- Retries are exhausted before declaring failure.
- In `strict_atomic` mode, rollback of the successful write is also retried up to `backend_max_retries` times.
- If rollback itself fails, `W_FORM_ROLLBACK_FAILED` warning is emitted with the orphaned artifact IDs for manual cleanup.

**Default:** `dual_write_mode = "best_effort"` (backward compatible, maximizes data retention).

### 8.1 Structured DB Rows

Each form template maps to a database table. Each form instance becomes a row.

**Table naming convention:**

```
{form_db_table_prefix}{template_name_slug}
```

Example: template "Leave Request" produces table `form_leave_request`.

**One canonical table per template family.** All versions of a template write to the same table. Version is tracked per-row via the `_template_version` column. Schema evolution uses `ALTER TABLE ADD COLUMN` for new fields; columns are never dropped.

This resolves the conflict between version-suffixed table names and schema evolution via ALTER TABLE. A single evolving table is simpler to query, easier to migrate, and avoids table proliferation.

**Schema generation:**

```python
def generate_table_schema(template: FormTemplate) -> dict[str, str]:
    """Generate a DB table schema from a form template.

    Table name is derived from the template name slug (without version suffix).
    All versions of a template share the same table.

    Returns a dict of {column_name: sql_type}.

    Mapping:
        FieldType.TEXT       -> "TEXT"
        FieldType.NUMBER     -> "REAL"
        FieldType.DATE       -> "TEXT"  (ISO 8601 string)
        FieldType.CHECKBOX   -> "INTEGER"  (0 or 1)
        FieldType.RADIO      -> "TEXT"
        FieldType.SIGNATURE  -> "INTEGER"  (0 or 1, signed/unsigned)
        FieldType.DROPDOWN   -> "TEXT"

    Additional columns always present:
        _form_id             -> "TEXT"  (UUID, primary key)
        _template_id         -> "TEXT"
        _template_version    -> "INTEGER"
        _source_uri          -> "TEXT"
        _ingest_key          -> "TEXT"
        _ingest_run_id       -> "TEXT"
        _tenant_id           -> "TEXT"
        _extracted_at        -> "TEXT"  (ISO 8601 timestamp)
        _overall_confidence  -> "REAL"
        _extraction_method   -> "TEXT"
    """
    ...
```

**Schema evolution:** When a new template version adds fields, the table is altered to add new nullable columns. Existing rows retain their original values. Removed fields are NOT dropped from the table (data preservation).

```
def evolve_table_schema(
    db: StructuredDBBackend,
    table_name: str,
    old_template: FormTemplate,
    new_template: FormTemplate,
) -> list[str]:
    """Add columns for new fields in a template version update.

    The table_name is the canonical table for the template family
    (without version suffix), e.g. 'form_leave_request'.

    Returns list of added column names.

    Algorithm:
        1. Compute set difference: new_fields - old_fields (by field_name).
        2. For each new field: ALTER TABLE ADD COLUMN (nullable).
        3. Return list of added column names.
        4. Never drop columns (removed fields become unused but preserved).
    """
```

### 8.2 RAG Chunk Serialization

Form data is serialized into text chunks optimized for embedding and semantic retrieval.

**Chunk format:**

```
Form: Leave Request (v2)
Date Extracted: 2026-03-15

Employee Name: John Smith
Department: Engineering
Leave Type: PTO
Start Date: 2026-03-01
End Date: 2026-03-05
Remote Work: Yes
Supervisor: Jane Doe
Reason: Family vacation
```

**Serialization rules:**
1. One chunk per form instance (for single-page forms with few fields).
2. Multi-page forms with many fields (> `chunk_max_fields`, default 20): split into multiple chunks, one per page, with a header chunk containing form-level metadata.
3. Field name/value pairs are serialized as `{field_label}: {value}`.
4. Fields with `value=None` are serialized as `{field_label}: [not extracted]`.
5. Checkbox/radio fields: serialized as `{field_label}: Yes/No` or `{field_label}: {selected_option}`.
6. The chunk header always includes: form template name, version, and extraction date.

### 8.3 Chunk Metadata

Every chunk carries rich metadata for downstream filtering and retrieval:

```python
class FormChunkMetadata(BaseModel):
    """Metadata attached to every form chunk for vector store upsert.

    Extends the standard chunk metadata pattern from ingestkit-excel/pdf
    with form-specific fields.
    """

    # Standard fields (consistent across all ingestkit packages)
    source_uri: str
    source_format: str  # "pdf", "xlsx", "image"
    ingestion_method: str = "form_extraction"
    parser_version: str
    chunk_index: int
    chunk_hash: str
    ingest_key: str
    ingest_run_id: str
    tenant_id: str | None = None

    # Form-specific fields
    template_id: str
    template_name: str
    template_version: int
    form_id: str = Field(description="Unique ID for this form instance (matches DB row _form_id).")
    field_names: list[str] = Field(description="Names of fields included in this chunk.")
    extraction_method: str  # "native_fields", "ocr_overlay", "cell_mapping"
    overall_confidence: float
    per_field_confidence: dict[str, float] = Field(
        description="Map of field_name -> confidence for fields in this chunk.",
    )
    form_date: str | None = Field(
        default=None,
        description="Date extracted from the form content, if a date field exists.",
    )
    page_numbers: list[int] = Field(
        description="Page numbers covered by this chunk.",
    )
    match_method: str = Field(
        description="How the template was matched: 'auto_detect' or 'manual_override'.",
    )
```

### 8.4 Chunk Payload

```python
class FormChunkPayload(BaseModel):
    """A single form chunk ready for vector store upsert."""

    id: str
    text: str
    vector: list[float]
    metadata: FormChunkMetadata
```

### 8.5 Rollback Protocol

The `WrittenArtifacts` model enables caller-side rollback. In `strict_atomic` mode, the plugin performs rollback automatically. The protocol is also available for external callers.

```python
class RollbackResult(BaseModel):
    """Result of a rollback operation."""

    vector_points_deleted: int = 0
    db_rows_deleted: int = 0
    errors: list[str] = []
    fully_rolled_back: bool = True


def rollback_written_artifacts(
    written: WrittenArtifacts,
    vector_backend: VectorStoreBackend | None = None,
    db_backend: StructuredDBBackend | None = None,
) -> RollbackResult:
    """Compensate a failed dual-write by deleting successfully written artifacts.

    Rollback order:
        1. Delete vector points first (less critical, faster).
        2. Delete DB rows second (more critical, may require transaction).

    Each step is retried up to backend_max_retries times.
    Both backends use idempotent delete semantics:
        - Vector: delete_by_ids() is a no-op for non-existent IDs.
        - DB: DELETE WHERE _form_id IN (...) is a no-op for non-existent rows.

    If any step fails after retries, fully_rolled_back=False and
    errors lists the failures. The caller must handle orphaned artifacts.
    """
```

**Idempotent write contract per backend:**
- `VectorStoreBackend.upsert_chunks()`: Upsert semantics --- re-inserting the same point ID overwrites. Safe to retry.
- `StructuredDBBackend.create_table_from_dataframe()`: The DB writer uses upsert (INSERT OR REPLACE on `_form_id`). Safe to retry.
- `VectorStoreBackend.delete_by_ids()`: No-op for IDs that don't exist. Safe to retry.

---

## 9. Plugin API Surface

The plugin exposes a programmatic API that the orchestration layer wraps into REST endpoints for the admin UI. The plugin itself does not serve HTTP --- it provides Python methods.

### 9.1 API Operations

```python
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class FormPluginAPI(Protocol):
    """Public API surface of the form ingestor plugin.

    The orchestration layer wraps these methods into HTTP endpoints.
    Each method is self-contained: it accepts simple types and returns
    Pydantic models.
    """

    # --- Template CRUD ---

    def list_templates(
        self,
        tenant_id: str | None = None,
        source_format: str | None = None,
    ) -> list[FormTemplate]:
        """List all active templates, optionally filtered by tenant and format.

        Returns the latest version of each template.
        """
        ...

    def get_template(
        self,
        template_id: str,
        version: int | None = None,
    ) -> FormTemplate:
        """Get a specific template by ID.

        If version is None, returns the latest version.
        Raises E_FORM_TEMPLATE_NOT_FOUND if not found.
        """
        ...

    def list_template_versions(
        self,
        template_id: str,
    ) -> list[FormTemplate]:
        """List all versions of a template, ordered by version descending."""
        ...

    def create_template(
        self,
        template_def: FormTemplateCreateRequest,
    ) -> FormTemplate:
        """Create a new template.

        Computes layout fingerprint from the sample document.
        Returns the created template with generated template_id.
        """
        ...

    def update_template(
        self,
        template_id: str,
        template_def: FormTemplateUpdateRequest,
    ) -> FormTemplate:
        """Update a template, creating a new version.

        The previous version is retained. The new version gets a
        recomputed layout fingerprint.
        Returns the new version.
        """
        ...

    def delete_template(
        self,
        template_id: str,
        version: int | None = None,
    ) -> None:
        """Delete a template or a specific version.

        If version is None, soft-deletes all versions (marks inactive).
        If version is specified, soft-deletes that version only.
        Hard deletion is not supported (audit trail preservation).
        """
        ...

    # --- Preview / Test ---

    def render_document(
        self,
        file_path: str,
        page: int = 0,
        dpi: int = 150,
    ) -> bytes:
        """Render a document page as a PNG image.

        Used by the admin UI to display the form for visual field mapping.
        Returns raw PNG bytes.
        """
        ...

    def preview_extraction(
        self,
        file_path: str,
        template_id: str,
        template_version: int | None = None,
    ) -> ExtractionPreview:
        """Run extraction without persisting results.

        Used by the admin UI to test a template against a sample document
        before saving. Returns extracted fields with confidence scores.
        """
        ...

    # --- Matching ---

    def match_document(
        self,
        file_path: str,
        tenant_id: str | None = None,
    ) -> list[TemplateMatch]:
        """Match a document against stored templates.

        Returns ranked list of matches above the confidence threshold.
        """
        ...

    # --- Processing ---

    def extract_form(
        self,
        request: FormIngestRequest,
    ) -> FormProcessingResult:
        """Extract form data and produce dual output (DB + chunks).

        If request.template_id is provided, uses that template.
        Otherwise, runs auto-detection first.
        """
        ...
```

### 9.2 Request/Response Models

```python
class FormTemplateCreateRequest(BaseModel):
    """Request to create a new form template."""

    name: str
    description: str = ""
    source_format: SourceFormat
    sample_file_path: str = Field(
        description="Path to a sample form document used to compute fingerprint.",
    )
    page_count: int = Field(ge=1)
    fields: list[FieldMapping] = Field(min_length=1)
    tenant_id: str | None = None
    created_by: str = "system"


class FormTemplateUpdateRequest(BaseModel):
    """Request to update a form template (creates a new version)."""

    name: str | None = None
    description: str | None = None
    sample_file_path: str | None = Field(
        default=None,
        description="New sample document for recomputing fingerprint.",
    )
    page_count: int | None = None
    fields: list[FieldMapping] | None = None


class ExtractionPreview(BaseModel):
    """Preview of extraction results without persistence."""

    template_id: str
    template_name: str
    template_version: int
    fields: list[ExtractedField]
    overall_confidence: float
    extraction_method: str
    warnings: list[str]
```

---

## 10. Data Models (Pydantic v2)

### 10.1 Extraction Result Models

```python
class ExtractedField(BaseModel):
    """A single extracted field value with confidence and provenance."""

    field_id: str
    field_name: str
    field_label: str
    field_type: FieldType
    value: str | bool | float | None = Field(
        description="Extracted value. None if extraction failed or confidence below threshold.",
    )
    raw_value: str | None = Field(
        default=None,
        description="Raw string before type coercion. Useful for debugging.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Extraction confidence for this field.",
    )
    extraction_method: str = Field(
        description=(
            "'native_fields', 'ocr_overlay', 'cell_mapping', "
            "'native_fields_with_ocr_fallback', or 'vlm_fallback'."
        ),
    )
    bounding_box: BoundingBox | None = Field(
        default=None,
        description="Actual bounding box where the value was found (may differ slightly from template).",
    )
    validation_passed: bool | None = Field(
        default=None,
        description="True if validation_pattern matched, False if it failed, None if no pattern defined.",
    )
    warnings: list[str] = []


class FormExtractionResult(BaseModel):
    """Complete extraction result for a single form instance.

    This is the intermediate result before dual-write. It contains all
    extracted fields and metadata needed to produce both DB rows and chunks.
    """

    form_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID for this form instance.",
    )
    template_id: str
    template_name: str
    template_version: int
    source_uri: str
    source_format: str
    fields: list[ExtractedField]
    overall_confidence: float
    extraction_method: str = Field(
        description="Primary extraction method used: 'native_fields', 'ocr_overlay', 'cell_mapping'.",
    )
    match_method: str = Field(
        description="'auto_detect' or 'manual_override'.",
    )
    match_confidence: float | None = Field(
        default=None,
        description="Template match confidence. None if manual override.",
    )
    pages_processed: int
    extraction_duration_seconds: float
    warnings: list[str] = []
```

### 10.2 Processing Result

```python
class FormProcessingResult(BaseModel):
    """Final result returned after processing a form document.

    Follows the same pattern as ProcessingResult in ingestkit-excel/pdf.
    """

    file_path: str
    ingest_key: str
    ingest_run_id: str
    tenant_id: str | None = None

    extraction_result: FormExtractionResult
    embed_result: EmbedStageResult | None = None

    chunks_created: int
    tables_created: int
    tables: list[str]
    written: WrittenArtifacts

    errors: list[str]
    warnings: list[str]
    error_details: list[IngestError] = []

    processing_time_seconds: float


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
    db_row_ids: list[str] = []
```

---

## 11. Configuration

```python
class FormProcessorConfig(BaseModel):
    """All tunable parameters for the form ingestor plugin.

    Override individual values via constructor kwargs or load a complete
    config from a file with ``FormProcessorConfig.from_file(path)``.
    """

    # --- Identity ---
    parser_version: str = "ingestkit_forms:1.0.0"
    tenant_id: str | None = None

    # --- Form Matching ---
    form_match_enabled: bool = True
    form_match_confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum layout similarity for auto-template assignment.",
    )
    form_match_per_page_minimum: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum per-page similarity for multi-page matching.",
    )
    form_match_extra_page_penalty: float = Field(
        default=0.02,
        ge=0.0,
        le=0.5,
        description="Confidence penalty per unmatched extra page in document.",
    )
    page_match_strategy: str = Field(
        default="windowed",
        description="Multi-page matching strategy: 'windowed' (v1 only).",
    )

    # --- Fingerprinting ---
    fingerprint_dpi: int = Field(
        default=150,
        description="DPI for rendering documents during fingerprint computation.",
    )
    fingerprint_grid_rows: int = Field(
        default=20,
        description="Number of rows in the fingerprint grid.",
    )
    fingerprint_grid_cols: int = Field(
        default=16,
        description="Number of columns in the fingerprint grid.",
    )

    # --- OCR Settings ---
    form_ocr_dpi: int = Field(
        default=300,
        description="DPI for rendering pages during OCR extraction.",
    )
    form_ocr_engine: str = Field(
        default="paddleocr",
        description="OCR engine: 'paddleocr' (primary) or 'tesseract' (lightweight fallback).",
    )
    form_ocr_language: str = Field(
        default="en",
        description="OCR language code.",
    )
    form_ocr_per_field_timeout_seconds: int = Field(
        default=10,
        description="Timeout per field OCR operation.",
    )

    # --- Native PDF Backend ---
    pdf_widget_backend: str = Field(
        default="pymupdf",
        description=(
            "PDF widget extraction backend: 'pymupdf' (AGPL, highest accuracy) "
            "or 'pdfplumber' (MIT, licensing-safe alternative). See §7.1.1."
        ),
    )

    # --- VLM Fallback (Optional) ---
    form_vlm_enabled: bool = Field(
        default=False,
        description="Enable VLM fallback for low-confidence OCR fields. Requires VLMBackend.",
    )
    form_vlm_model: str = Field(
        default="qwen2.5-vl:7b",
        description="VLM model identifier for Ollama (or equivalent backend).",
    )
    form_vlm_fallback_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="OCR confidence below this triggers VLM fallback (when enabled).",
    )
    form_vlm_timeout_seconds: int = Field(
        default=15,
        description="Timeout per VLM field extraction call.",
    )
    form_vlm_max_fields_per_document: int = Field(
        default=10,
        ge=1,
        description="Maximum fields per document sent to VLM (cost/latency guard).",
    )

    # --- Field Extraction ---
    form_extraction_min_field_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Below this confidence, field value is set to None.",
    )
    form_extraction_min_overall_confidence: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "Below this overall confidence, extraction is considered failed. "
            "Result is returned with E_FORM_EXTRACTION_LOW_CONFIDENCE."
        ),
    )
    checkbox_fill_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Fill ratio above which a checkbox is considered checked.",
    )
    signature_ink_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Ink ratio above which a signature field is considered signed.",
    )

    # --- Native PDF Field Matching ---
    native_pdf_iou_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="IoU threshold for matching PDF widgets to template fields.",
    )

    # --- Output: Structured DB ---
    form_db_table_prefix: str = Field(
        default="form_",
        description="Prefix for structured DB table names.",
    )

    # --- Output: Chunking ---
    chunk_max_fields: int = Field(
        default=20,
        description="Maximum fields per chunk. Multi-page forms exceeding this are split.",
    )

    # --- Embedding ---
    embedding_model: str = "nomic-embed-text"
    embedding_dimension: int = 768
    embedding_batch_size: int = 64

    # --- Vector Store ---
    default_collection: str = "helpdesk"

    # --- Template Storage ---
    form_template_storage_path: str = Field(
        default="./form_templates",
        description="Directory or connection string for template persistence.",
    )

    # --- Resource Limits ---
    max_file_size_mb: int = Field(
        default=100,
        description="Maximum file size for form documents.",
    )
    per_document_timeout_seconds: int = Field(
        default=120,
        description="Maximum processing time per form document.",
    )

    # --- Backend Resilience ---
    backend_timeout_seconds: float = 30.0
    backend_max_retries: int = 2
    backend_backoff_base: float = 1.0

    # --- Dual-Write ---
    dual_write_mode: str = Field(
        default="best_effort",
        description="Dual-write failure semantics: 'best_effort' or 'strict_atomic'.",
    )

    # --- Logging / PII Safety ---
    log_sample_data: bool = Field(
        default=False,
        description="If True, extracted field values may appear in logs. Default is PII-safe.",
    )
    log_ocr_output: bool = False
    log_extraction_details: bool = False
    redact_patterns: list[str] = []
    redact_target: str = Field(
        default="both",
        description="Where redaction applies: 'both', 'chunks_only', 'db_only'.",
    )

    @model_validator(mode="after")
    def _validate_enum_fields(self) -> FormProcessorConfig:
        allowed_dual_write = {"best_effort", "strict_atomic"}
        if self.dual_write_mode not in allowed_dual_write:
            raise ValueError(f"dual_write_mode must be one of {allowed_dual_write}")
        allowed_redact = {"both", "chunks_only", "db_only"}
        if self.redact_target not in allowed_redact:
            raise ValueError(f"redact_target must be one of {allowed_redact}")
        allowed_page_match = {"windowed"}
        if self.page_match_strategy not in allowed_page_match:
            raise ValueError(f"page_match_strategy must be one of {allowed_page_match}")
        allowed_ocr_engines = {"paddleocr", "tesseract"}
        if self.form_ocr_engine not in allowed_ocr_engines:
            raise ValueError(f"form_ocr_engine must be one of {allowed_ocr_engines}")
        allowed_pdf_backends = {"pymupdf", "pdfplumber"}
        if self.pdf_widget_backend not in allowed_pdf_backends:
            raise ValueError(f"pdf_widget_backend must be one of {allowed_pdf_backends}")
        if self.form_vlm_fallback_threshold >= self.form_extraction_min_field_confidence:
            raise ValueError(
                "form_vlm_fallback_threshold must be less than "
                "form_extraction_min_field_confidence"
            )
        return self

    @classmethod
    def from_file(cls, path: str) -> FormProcessorConfig:
        """Load configuration from a YAML or JSON file.

        File format is detected by extension: ``.yaml`` / ``.yml`` for YAML,
        ``.json`` for JSON.  Any keys present in the file override the
        corresponding defaults; keys not present retain their defaults.
        """
        import json as json_mod
        import pathlib

        file_path = pathlib.Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        suffix = file_path.suffix.lower()

        if suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "pyyaml is required to load YAML config files. "
                    "Install it with: pip install pyyaml"
                ) from exc
            with open(file_path) as fh:
                data = yaml.safe_load(fh)
        elif suffix == ".json":
            with open(file_path) as fh:
                data = json_mod.load(fh)
        else:
            raise ValueError(
                f"Unsupported config file extension '{suffix}'. "
                "Use .yaml, .yml, or .json."
            )

        if data is None:
            data = {}

        return cls(**data)
```

### Configuration Summary Table

| Parameter | Default | Notes |
|---|---|---|
| `form_match_enabled` | `True` | Set to `False` to skip auto-matching (manual template only). |
| `form_match_confidence_threshold` | `0.8` | Above this: auto-apply template. |
| `form_match_per_page_minimum` | `0.6` | Per-page minimum for multi-page matching. |
| `form_match_extra_page_penalty` | `0.02` | Confidence penalty per unmatched extra page. |
| `page_match_strategy` | `windowed` | Multi-page matching strategy (`windowed` in v1). |
| `fingerprint_dpi` | `150` | Lower DPI for fast fingerprinting. |
| `fingerprint_grid_rows` | `20` | Grid resolution for fingerprint. |
| `fingerprint_grid_cols` | `16` | Grid resolution for fingerprint. |
| `form_ocr_dpi` | `300` | Higher DPI for OCR accuracy. |
| `form_ocr_engine` | `paddleocr` | Primary OCR engine. Tesseract as lightweight fallback. |
| `pdf_widget_backend` | `pymupdf` | PDF widget extraction: `pymupdf` (AGPL) or `pdfplumber` (MIT). |
| `form_vlm_enabled` | `False` | Enable VLM fallback for low-confidence OCR fields. |
| `form_vlm_model` | `qwen2.5-vl:7b` | VLM model identifier for Ollama. |
| `form_vlm_fallback_threshold` | `0.4` | OCR confidence below this triggers VLM (when enabled). |
| `form_vlm_timeout_seconds` | `15` | Per-field VLM timeout. |
| `form_vlm_max_fields_per_document` | `10` | Max VLM calls per document (cost guard). |
| `form_extraction_min_field_confidence` | `0.5` | Below this: field value = None. |
| `form_extraction_min_overall_confidence` | `0.3` | Below this: extraction considered failed. |
| `checkbox_fill_threshold` | `0.3` | Fill ratio for checkbox detection. |
| `signature_ink_threshold` | `0.05` | Ink ratio for signature detection. |
| `native_pdf_iou_threshold` | `0.5` | IoU for matching PDF widgets to template fields. |
| `form_db_table_prefix` | `form_` | Prefix for DB table names. |
| `chunk_max_fields` | `20` | Max fields per chunk before splitting. |
| `embedding_model` | `nomic-embed-text` | Embedding model for chunk vectors. |
| `embedding_dimension` | `768` | Vector size. |
| `form_template_storage_path` | `./form_templates` | Template persistence location. |
| `max_file_size_mb` | `100` | Max form document size. |
| `per_document_timeout_seconds` | `120` | Processing timeout per document. |
| `log_sample_data` | `False` | PII-safe by default. |
| `redact_target` | `both` | Where redaction applies: `both`, `chunks_only`, `db_only`. |
| `dual_write_mode` | `best_effort` | Dual-write failure semantics: `best_effort` or `strict_atomic`. |

---

## 12. Error Handling & Error Codes

### 12.1 Error Codes

The form plugin extends the normalized error code taxonomy with form-specific codes:

```python
class FormErrorCode(str, Enum):
    """Normalized error codes for the ingestkit-forms pipeline.

    Extends the base ErrorCode taxonomy with form-specific codes.
    All codes use the same E_ (error) and W_ (warning) prefix convention.
    """

    # Template errors
    E_FORM_TEMPLATE_NOT_FOUND = "E_FORM_TEMPLATE_NOT_FOUND"
    E_FORM_TEMPLATE_INVALID = "E_FORM_TEMPLATE_INVALID"
    E_FORM_TEMPLATE_VERSION_CONFLICT = "E_FORM_TEMPLATE_VERSION_CONFLICT"
    E_FORM_TEMPLATE_STORE_UNAVAILABLE = "E_FORM_TEMPLATE_STORE_UNAVAILABLE"

    # Matching errors
    E_FORM_NO_MATCH = "E_FORM_NO_MATCH"
    E_FORM_FINGERPRINT_FAILED = "E_FORM_FINGERPRINT_FAILED"

    # Extraction errors
    E_FORM_EXTRACTION_FAILED = "E_FORM_EXTRACTION_FAILED"
    E_FORM_EXTRACTION_LOW_CONFIDENCE = "E_FORM_EXTRACTION_LOW_CONFIDENCE"
    E_FORM_EXTRACTION_TIMEOUT = "E_FORM_EXTRACTION_TIMEOUT"
    E_FORM_UNSUPPORTED_FORMAT = "E_FORM_UNSUPPORTED_FORMAT"
    E_FORM_OCR_FAILED = "E_FORM_OCR_FAILED"
    E_FORM_NATIVE_FIELDS_UNAVAILABLE = "E_FORM_NATIVE_FIELDS_UNAVAILABLE"

    # Output errors
    E_FORM_DB_SCHEMA_EVOLUTION_FAILED = "E_FORM_DB_SCHEMA_EVOLUTION_FAILED"
    E_FORM_DB_WRITE_FAILED = "E_FORM_DB_WRITE_FAILED"
    E_FORM_CHUNK_WRITE_FAILED = "E_FORM_CHUNK_WRITE_FAILED"

    # Dual-write errors
    E_FORM_DUAL_WRITE_PARTIAL = "E_FORM_DUAL_WRITE_PARTIAL"  # strict_atomic: one backend succeeded, rollback initiated

    # Manual override errors
    E_FORM_FORMAT_MISMATCH = "E_FORM_FORMAT_MISMATCH"  # template source_format != input format

    # VLM errors
    E_FORM_VLM_UNAVAILABLE = "E_FORM_VLM_UNAVAILABLE"
    E_FORM_VLM_TIMEOUT = "E_FORM_VLM_TIMEOUT"

    # Security errors
    E_FORM_FILE_TOO_LARGE = "E_FORM_FILE_TOO_LARGE"
    E_FORM_FILE_CORRUPT = "E_FORM_FILE_CORRUPT"

    # Backend errors (reuse base codes where possible)
    E_BACKEND_VECTOR_TIMEOUT = "E_BACKEND_VECTOR_TIMEOUT"
    E_BACKEND_VECTOR_CONNECT = "E_BACKEND_VECTOR_CONNECT"
    E_BACKEND_DB_TIMEOUT = "E_BACKEND_DB_TIMEOUT"
    E_BACKEND_DB_CONNECT = "E_BACKEND_DB_CONNECT"
    E_BACKEND_EMBED_TIMEOUT = "E_BACKEND_EMBED_TIMEOUT"
    E_BACKEND_EMBED_CONNECT = "E_BACKEND_EMBED_CONNECT"

    # Warnings (non-fatal)
    W_FORM_FIELD_LOW_CONFIDENCE = "W_FORM_FIELD_LOW_CONFIDENCE"
    W_FORM_FIELD_VALIDATION_FAILED = "W_FORM_FIELD_VALIDATION_FAILED"
    W_FORM_FIELD_MISSING_REQUIRED = "W_FORM_FIELD_MISSING_REQUIRED"
    W_FORM_FIELD_TYPE_COERCION = "W_FORM_FIELD_TYPE_COERCION"
    W_FORM_FIELDS_FLATTENED = "W_FORM_FIELDS_FLATTENED"
    W_FORM_MATCH_BELOW_THRESHOLD = "W_FORM_MATCH_BELOW_THRESHOLD"
    W_FORM_MULTI_MATCH = "W_FORM_MULTI_MATCH"
    W_FORM_OCR_DEGRADED = "W_FORM_OCR_DEGRADED"
    W_FORM_MERGED_CELL_RESOLVED = "W_FORM_MERGED_CELL_RESOLVED"
    W_FORM_SCHEMA_EVOLVED = "W_FORM_SCHEMA_EVOLVED"
    W_FORM_ROLLBACK_FAILED = "W_FORM_ROLLBACK_FAILED"
    W_FORM_PARTIAL_WRITE = "W_FORM_PARTIAL_WRITE"
    W_FORM_VLM_FALLBACK_USED = "W_FORM_VLM_FALLBACK_USED"
    W_FORM_VLM_BUDGET_EXHAUSTED = "W_FORM_VLM_BUDGET_EXHAUSTED"
```

### 12.2 Structured Error Model

```python
class FormIngestError(BaseModel):
    """Structured error with code, message, and form-specific context.

    Follows the same pattern as IngestError in ingestkit-excel/pdf,
    with form-specific context fields.
    """

    code: FormErrorCode
    message: str
    template_id: str | None = None
    template_version: int | None = None
    field_name: str | None = None
    page_number: int | None = None
    stage: str | None = None
    recoverable: bool = False
    # Diagnostic context (P1)
    candidate_matches: list[dict] | None = Field(
        default=None,
        description="Top template match candidates with confidences (for match-related errors).",
    )
    backend_operation_id: str | None = Field(
        default=None,
        description="Backend-specific operation ID for tracing write/rollback failures.",
    )
    fallback_reason: str | None = Field(
        default=None,
        description="Why fallback was triggered (for degraded-path errors).",
    )
```

### 12.3 Fail-Closed Behavior

The form plugin follows the fail-closed principle established across ingestkit:

| Scenario | Behavior | Error Code |
|---|---|---|
| No template matches and no manual override | Fall through to standard pipeline (not an error). | N/A |
| Template specified but not found | Return error, do not process. | `E_FORM_TEMPLATE_NOT_FOUND` |
| Extraction overall confidence < threshold | Return result with error, zero chunks written. | `E_FORM_EXTRACTION_LOW_CONFIDENCE` |
| OCR fails on a field | Field value = None, warning emitted, other fields still extracted. | `W_FORM_FIELD_LOW_CONFIDENCE` |
| OCR engine unavailable | Return error, do not process. | `E_FORM_OCR_FAILED` |
| DB write fails | Chunks still written (if vector backend available), error recorded. | `E_FORM_DB_WRITE_FAILED` |
| Vector write fails | DB row still written (if DB backend available), error recorded. | `E_FORM_CHUNK_WRITE_FAILED` |
| Both backends fail | Return result with errors, zero artifacts written. | Both error codes |
| Manual override with format mismatch | Return error before extraction. | `E_FORM_FORMAT_MISMATCH` |
| Partial dual-write (strict mode) | Rollback successful write, return error. | `E_FORM_DUAL_WRITE_PARTIAL` |
| Partial dual-write (best_effort mode) | Keep successful write, emit warning. | `W_FORM_PARTIAL_WRITE` |
| Rollback fails after partial write | Emit warning with orphaned artifact IDs. | `W_FORM_ROLLBACK_FAILED` |
| Review-band match (0.5 to <threshold) | Fall through to standard pipeline with warning. | `W_FORM_MATCH_BELOW_THRESHOLD` |
| VLM backend unavailable (when enabled) | Field retains OCR result, warning emitted. | `E_FORM_VLM_UNAVAILABLE` |
| VLM call times out | Field retains OCR result, warning emitted. | `E_FORM_VLM_TIMEOUT` |
| VLM budget exhausted | Remaining low-confidence fields retain OCR results. | `W_FORM_VLM_BUDGET_EXHAUSTED` |

---

## 13. Security Considerations

### 13.1 Input Validation

- **File size limit:** Enforced before any processing. Documents exceeding `max_file_size_mb` are rejected with `E_FORM_FILE_TOO_LARGE`.
- **File type validation:** Only accepted extensions (`.pdf`, `.xlsx`, `.jpg`, `.jpeg`, `.png`, `.tiff`, `.tif`) are processed. Others are rejected with `E_FORM_UNSUPPORTED_FORMAT`.
- **Magic byte verification:** File extension is cross-checked against magic bytes. A `.pdf` file that does not start with `%PDF` is rejected with `E_FORM_FILE_CORRUPT`.
- **Template field count limit:** Templates are limited to 200 fields to prevent resource exhaustion during extraction.
- **Regex validation patterns:** Validation patterns in `FieldMapping.validation_pattern` are compiled at template creation time. Invalid regex is rejected. Regex execution has a per-pattern timeout of 1 second to prevent ReDoS.

### 13.2 PII Safety

- **Default: no field values in logs.** Extracted field values are never logged unless `log_sample_data=True`.
- **Template names and field names are always safe to log** (they describe structure, not content).
- **Confidence scores are always safe to log** (they are numeric, contain no PII).
- **Configurable redaction with target scoping:** Redaction is controlled by two config fields:
  - `redact_patterns: list[str]` --- regex patterns; matching substrings are replaced with `[REDACTED]`.
  - `redact_target: RedactTarget` --- controls where redaction is applied:

  ```python
  class RedactTarget(str, Enum):
      BOTH = "both"            # Redact in DB rows AND RAG chunks (default)
      CHUNKS_ONLY = "chunks_only"  # Redact in chunks only; DB retains raw values for analytics
      DB_ONLY = "db_only"      # Redact in DB only; chunks retain raw values for search
  ```

  Redaction is applied **after** extraction and validation but **before** persistence. Raw values are held in memory during processing but are **never persisted** in their un-redacted form when redaction is active. The `FormExtractionResult` (intermediate, pre-persistence) contains raw values; the written artifacts contain redacted values according to `redact_target`.
- **OCR output:** Raw OCR text is never logged unless `log_ocr_output=True`.

### 13.3 Template Security

- **Soft delete only:** Templates are never hard-deleted. This preserves the audit trail: every extraction result references its template, which must remain accessible.
- **Version immutability:** Once a template version is used in an extraction, it cannot be modified. Edits create a new version.
- **Tenant isolation:** Templates carry a `tenant_id`. The `FormTemplateStore` must enforce tenant scoping --- a tenant can only access its own templates. The plugin propagates `tenant_id` but does not enforce isolation (that is the caller's responsibility, consistent with the rest of ingestkit).

### 13.4 Image Processing Security

- **Decompression bomb protection:** Image files are checked for anomalous compression ratios before loading. Images where `decompressed_size / compressed_size > 100` are rejected.
- **Resolution limits:** Images wider or taller than 10000 pixels are rejected to prevent memory exhaustion.
- **No external URL loading:** The plugin only processes local files. It never fetches images from URLs.

### 13.5 Security Test Requirements

The following security controls MUST have unit and/or integration tests:

| Control | Test Requirement |
|---------|-----------------|
| Regex validation timeout | Test that a ReDoS pattern times out within 1s and does not hang the extraction pipeline. |
| Magic byte mismatch | Test each supported extension (.pdf, .xlsx, .jpg, .png, .tiff) with wrong magic bytes -> `E_FORM_FILE_CORRUPT`. |
| Decompression bomb | Test image with compression ratio > 100 -> rejected before loading. |
| Resolution guardrail | Test image > 10000px wide or tall -> rejected with `E_FORM_FILE_CORRUPT`. |
| File size limit | Test file exceeding `max_file_size_mb` -> `E_FORM_FILE_TOO_LARGE`. |
| Template field count limit | Test template with > 200 fields -> `E_FORM_TEMPLATE_INVALID`. |
| Redaction | Test that redacted values never appear in persisted DB rows or chunks (per `redact_target` mode). |

---

## 14. Admin UI Contract (What the UI Needs from the Plugin)

This section defines the contract between the form plugin and the admin UI. The plugin provides data and operations; the UI renders and interacts.

### 14.1 Template Creation Workflow

```
1. UI: Admin uploads a sample form document.
   -> Plugin: render_document(file_path, page=0) -> PNG bytes
   -> UI displays the rendered page image.

2. UI: Admin draws bounding boxes on the image, labeling each field.
   -> UI captures: list of FieldMapping objects with normalized BoundingBox coordinates.
   -> For Excel forms: UI shows cell grid; admin clicks cells to map fields.
   -> UI captures: list of FieldMapping objects with CellAddress.

3. UI: Admin fills in template metadata (name, description, source_format).
   -> UI sends: FormTemplateCreateRequest to plugin.
   -> Plugin: create_template(request) -> FormTemplate
   -> Plugin computes layout_fingerprint from the sample document.

4. UI: Admin optionally tests the template.
   -> UI sends: preview_extraction(file_path, template_id)
   -> Plugin returns ExtractionPreview with extracted values and confidence.
   -> UI displays results for admin review.

5. UI: Admin saves/activates the template.
   -> Template is now available for auto-matching.
```

### 14.2 Data the UI Needs

| UI Feature | Plugin API Call | Data Returned |
|---|---|---|
| Display template list | `list_templates()` | Template name, description, version, source_format, field count, created_at |
| Display template details | `get_template(id)` | Full template with all field mappings |
| Render form for mapping | `render_document(path, page)` | PNG image bytes |
| Test extraction | `preview_extraction(path, template_id)` | Per-field values, confidence scores, warnings |
| View match candidates | `match_document(path)` | Ranked template matches with confidence |
| Display version history | `list_template_versions(id)` | All versions with metadata |

### 14.3 Coordinate System Convention

The UI and plugin share a coordinate system for bounding boxes:

- **Origin:** Top-left corner of the page.
- **Axes:** X increases rightward, Y increases downward.
- **Units:** Normalized 0.0 to 1.0 (proportion of page width/height).
- **Example:** A field at `x=0.1, y=0.2, width=0.3, height=0.05` occupies the region from 10% to 40% of page width and 20% to 25% of page height.

The UI renders the image at whatever display resolution it wants and converts pixel coordinates to normalized coordinates before sending to the plugin:

```
normalized_x = pixel_x / display_width
normalized_y = pixel_y / display_height
normalized_width = pixel_width / display_width
normalized_height = pixel_height / display_height
```

---

## 15. Dependencies & Package Structure

### 15.1 Package Metadata

```toml
[project]
name = "ingestkit-forms"
version = "1.0.0"
description = "Template-driven form extraction plugin for ingestkit"
requires-python = ">=3.10"
dependencies = [
    "pydantic>=2.0",
    "openpyxl>=3.1",
    "Pillow>=10.0",
]

[project.optional-dependencies]
# PDF widget extraction — choose ONE based on licensing needs
pdf = ["PyMuPDF>=1.24"]                          # AGPL-3.0 (or commercial license)
pdf-mit = ["pdfplumber>=0.11", "pypdf>=4.0"]     # MIT + BSD — licensing-safe alternative

# OCR — choose ONE (or both; engine selected at runtime via config)
ocr = [
    "paddleocr>=2.7; platform_system != 'Darwin' or platform_machine != 'arm64'",
]
ocr-lite = ["pytesseract>=0.3.10"]               # Lightweight fallback for constrained envs

# VLM fallback — optional, for low-confidence OCR field re-extraction
vlm = ["httpx>=0.27"]                            # HTTP client for Ollama API

# Full install — all extraction capabilities
all = [
    "ingestkit-forms[pdf,ocr,vlm]",
]

dev = [
    "pytest>=7.0",
    "pytest-cov",
    "pyyaml>=6.0",
]
```

**Note on core dependencies:** PyMuPDF is no longer a hard dependency. The core package requires only `pydantic`, `openpyxl`, and `Pillow`. PDF and OCR backends are optional extras, selected at install time based on deployment licensing and platform constraints. At least one PDF extra (`pdf` or `pdf-mit`) is required for PDF form processing. At least one OCR extra (`ocr` or `ocr-lite`) is required for scanned/image form processing.

### 15.2 Dependency Rationale

| Dependency | Purpose | Extra Group | License | Notes |
|---|---|---|---|---|
| `pydantic>=2.0` | All data models | core | MIT | Consistent with ingestkit-excel, ingestkit-pdf |
| `openpyxl>=3.1` | Excel cell reading | core | MIT | Already used by ingestkit-excel |
| `Pillow>=10.0` | Image loading, preprocessing, region cropping | core | HPND | Required for OCR overlay pipeline |
| `PyMuPDF>=1.24` | PDF rendering, native form field extraction | `[pdf]` | AGPL-3.0 | Highest accuracy for widget extraction. Commercial license available. |
| `pdfplumber>=0.11` | PDF text/widget extraction (MIT alternative) | `[pdf-mit]` | MIT | Licensing-safe. Slightly lower accuracy for complex widget types. |
| `pypdf>=4.0` | PDF AcroForm field reading (BSD alternative) | `[pdf-mit]` | BSD | Complements pdfplumber for fillable form metadata. |
| `paddleocr>=2.7` | PaddleOCR binding — primary OCR engine | `[ocr]` | Apache-2.0 | SER+RE for form layouts. Limited macOS ARM support. |
| `pytesseract>=0.3.10` | Tesseract OCR binding — lightweight fallback | `[ocr-lite]` | Apache-2.0 | Platform-portable. Lower accuracy on form layouts. |
| `httpx>=0.27` | HTTP client for Ollama VLM API | `[vlm]` | BSD | Only needed when `form_vlm_enabled=True`. |

### 15.3 Protocol Definitions

The form plugin defines two new protocols and reuses three from the existing packages:

```python
# --- New protocols specific to ingestkit-forms ---

@runtime_checkable
class FormTemplateStore(Protocol):
    """Interface for form template persistence.

    Concrete implementations might use: filesystem (JSON/YAML files),
    SQLite, PostgreSQL, or any key-value store.
    """

    def save_template(self, template: FormTemplate) -> None:
        """Persist a template (insert or update)."""
        ...

    def get_template(
        self, template_id: str, version: int | None = None
    ) -> FormTemplate | None:
        """Retrieve a template by ID. None if not found.

        If version is None, returns the latest version.
        """
        ...

    def list_templates(
        self,
        tenant_id: str | None = None,
        source_format: str | None = None,
        active_only: bool = True,
    ) -> list[FormTemplate]:
        """List templates matching the filters."""
        ...

    def list_versions(self, template_id: str) -> list[FormTemplate]:
        """List all versions of a template, ordered by version descending."""
        ...

    def delete_template(
        self, template_id: str, version: int | None = None
    ) -> None:
        """Soft-delete a template or specific version."""
        ...

    def get_all_fingerprints(
        self,
        tenant_id: str | None = None,
        source_format: str | None = None,
    ) -> list[tuple[str, str, int, bytes]]:
        """Return (template_id, name, version, fingerprint) for all active templates.

        Used by the matcher for efficient batch comparison.
        """
        ...


@runtime_checkable
class OCRBackend(Protocol):
    """Interface for OCR engines used in form field extraction.

    Abstracts Tesseract vs. PaddleOCR (or any future engine).
    """

    def ocr_region(
        self,
        image_bytes: bytes,
        language: str = "en",
        config: str | None = None,
        timeout: float | None = None,
    ) -> OCRRegionResult:
        """Run OCR on a cropped image region.

        Args:
            image_bytes: PNG-encoded bytes of the cropped region.
            language: OCR language code.
            config: Engine-specific configuration string
                (e.g., Tesseract --psm and --oem flags).
            timeout: Per-field timeout in seconds.

        Returns:
            OCRRegionResult with text, confidence, and character-level details.
        """
        ...

    def engine_name(self) -> str:
        """Return the name of the OCR engine (e.g., 'tesseract', 'paddleocr')."""
        ...


class OCRRegionResult(BaseModel):
    """Result of OCR on a single field region."""

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    char_confidences: list[float] | None = Field(
        default=None,
        description="Per-character confidence values, if available.",
    )
    engine: str
```

```python
# --- PDF Widget Backend (licensing-safe abstraction) ---

@runtime_checkable
class PDFWidgetBackend(Protocol):
    """Interface for extracting form widgets from fillable PDFs.

    Abstracts PyMuPDF (AGPL) vs. pdfplumber+pypdf (MIT/BSD).
    See §7.1.1 for licensing governance.
    """

    def extract_widgets(
        self,
        file_path: str,
        page: int,
    ) -> list[WidgetField]:
        """Extract all form widgets from the specified page.

        Args:
            file_path: Path to the PDF file.
            page: 0-indexed page number.

        Returns:
            List of WidgetField objects with field_name, field_value,
            field_type, and bounding box in normalized coordinates.
        """
        ...

    def has_form_fields(self, file_path: str) -> bool:
        """Check whether the PDF contains any fillable form fields."""
        ...

    def engine_name(self) -> str:
        """Return the backend name (e.g., 'pymupdf', 'pdfplumber')."""
        ...


class WidgetField(BaseModel):
    """A single form widget extracted from a PDF."""

    field_name: str
    field_value: str | None
    field_type: str  # "text", "checkbox", "radio", "dropdown", "listbox"
    bbox: BoundingBox  # Normalized 0.0-1.0 coordinates
    page: int


# --- VLM Backend (optional, for fallback extraction) ---

@runtime_checkable
class VLMBackend(Protocol):
    """Interface for Vision-Language Model field extraction.

    Abstracts Ollama/Qwen2.5-VL vs. vLLM vs. llama.cpp.
    Only used when form_vlm_enabled=True and OCR confidence is
    below form_vlm_fallback_threshold.
    """

    def extract_field(
        self,
        image_bytes: bytes,
        field_type: str,
        field_name: str,
        extraction_hint: str | None = None,
        timeout: float | None = None,
    ) -> VLMFieldResult:
        """Run VLM extraction on a cropped field image.

        Args:
            image_bytes: PNG-encoded bytes of the cropped field region
                (with padding for context).
            field_type: Expected field type ('text', 'number', 'date',
                'checkbox', 'radio', 'signature', 'dropdown').
            field_name: Human-readable field name for prompt context.
            extraction_hint: Optional hint (e.g., 'date_format:MM/DD/YYYY').
            timeout: Per-field timeout in seconds.

        Returns:
            VLMFieldResult with extracted value and confidence.
        """
        ...

    def model_name(self) -> str:
        """Return the VLM model identifier (e.g., 'qwen2.5-vl:7b')."""
        ...

    def is_available(self) -> bool:
        """Check whether the VLM backend is reachable."""
        ...


class VLMFieldResult(BaseModel):
    """Result of VLM extraction on a single field region."""

    value: str | bool | None
    confidence: float = Field(ge=0.0, le=1.0)
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
```

**Reused protocols (imported from ingestkit-core or defined identically):**

- `VectorStoreBackend` --- for chunk upsert.
- `StructuredDBBackend` --- for form row write and schema evolution.
- `EmbeddingBackend` --- for chunk embedding.

---

## 16. Future Roadmap

The following features are explicitly deferred. Do NOT implement unless a future issue explicitly requests them.

| Feature | Rationale for Deferral |
|---|---|
| **Handwriting recognition model** | Requires training data and GPU resources. OCR engines handle printed text; handwriting is a stretch goal. |
| **Multi-language OCR** | Requires language-specific models. English-only in v1. |
| **Template auto-generation from sample** | Using VLM to automatically detect and label form fields from a sample document. Now feasible via the VLM backend (§7.5) but deferred to reduce v1 scope. |
| **Form diff/change detection** | Comparing two instances of the same form to highlight differences. Useful but not MVP. |
| **Batch form processing API** | Processing multiple forms in a single API call. Individual processing is sufficient for v1. |
| **Form analytics dashboard** | Aggregate statistics across extracted forms. Caller's responsibility. |
| **PDF form filling (write-back)** | Writing extracted values back into a fillable PDF. Out of scope; this is an extraction plugin. |
| **Cloud OCR backends** | Google Vision, AWS Textract, Azure Document Intelligence. On-premises only in v1. |
| **Template sharing across tenants** | Global template library. Security implications need design work. |
| **Active learning for OCR** | Using corrected extractions to improve OCR accuracy over time. |

---

## 17. Resolved Design Decisions

All design questions from the initial draft have been resolved during engineering review:

| # | Decision | Context | Resolution |
|---|---|---|---|
| OQ-1 | ~~Should ingestkit-forms depend on sibling packages or use ingestkit-core?~~ **RESOLVED.** | The form plugin needs PDF rendering (PyMuPDF) and Excel reading (openpyxl). Importing from sibling packages creates coupling. | Extract shared primitives (`IngestKey`, `WrittenArtifacts`, `EmbedStageResult`, common protocols `VectorStoreBackend`, `StructuredDBBackend`, `EmbeddingBackend`, base error models) into `ingestkit-core`. All packages (`ingestkit-excel`, `ingestkit-pdf`, `ingestkit-forms`) depend on `ingestkit-core` only. No sibling imports. This is a prerequisite for ingestkit-forms implementation and should be tracked as a separate issue. |
| OQ-2 | ~~Where should form templates be stored by default?~~ **RESOLVED.** | Options: filesystem (JSON files), SQLite, or require the caller to provide a `FormTemplateStore` backend. | Default to filesystem (JSON files). Provide `FileSystemTemplateStore` as included default. SQLite and PostgreSQL stores are optional backends. |
| OQ-3 | ~~Should the Form Matcher run on every document or only when explicitly enabled?~~ **RESOLVED.** | Running on every document adds latency (fingerprint computation). But opt-in means forms go to the wrong path unless someone remembers to enable it. | Default `form_match_enabled=True`. Fast fingerprinting (< 100ms). For high-throughput systems, the admin can disable auto-matching and use manual template assignment. |
| OQ-4 | ~~How should multi-page form matching handle partial matches?~~ **RESOLVED.** | A 3-page form where page 2 is heavily annotated might match pages 1 and 3 but not page 2. | Use windowed alignment with configurable per-page minimum and extra-page penalty (see §6.1 update). All pages in the best window must exceed `form_match_per_page_minimum` (0.6). Extra pages incur a configurable penalty (`form_match_extra_page_penalty`, default 0.02). |
| OQ-5 | ~~Should the DB table schema evolution be automatic or require admin approval?~~ **RESOLVED.** | Automatic is more convenient but could produce surprising schema changes. | Automatic in v1 with `W_FORM_SCHEMA_EVOLVED` warning. Admin can see the warning in the processing result. Future version could add an approval workflow. |
| OQ-6 | ~~What is the maximum number of templates that can be active simultaneously without degrading match performance?~~ **RESOLVED.** | Each template requires a fingerprint comparison. At 100 templates, this is negligible. At 10,000, it could matter. | v1 target: 500 active templates. Performance targets codified in §18. Fingerprint comparison is O(grid_size) per template, and grid_size is small (320 cells). 500 templates = 160,000 cell comparisons, well under 1ms. |
| OQ-7 | ~~Should the plugin support partial extraction (extract only some fields from a template)?~~ **RESOLVED.** | An admin might want to extract only "Employee Name" and "Date" from a complex form. | Defer partial extraction to v2. All template fields extracted in v1. The admin can create a simplified template with only the fields they need. |

---

## 18. Acceptance Criteria & Non-Functional Targets

### 18.1 Performance Targets

| Metric | Target | Condition |
|--------|--------|-----------|
| Template match latency (p95) | < 50ms | 50 active templates, single-page document |
| Template match latency (p95) | < 200ms | 200 active templates, single-page document |
| Template match latency (p95) | < 500ms | 500 active templates, single-page document |
| OCR field extraction (p95) | < 2s per field | 300 DPI, single text field, Tesseract |
| End-to-end form processing (p95) | < 10s | Single-page PDF, 10 fields, native extraction |
| End-to-end form processing (p95) | < 30s | Single-page scanned PDF, 10 fields, OCR extraction |
| End-to-end form processing (p95) | < 60s | 3-page scanned PDF, 30 fields, OCR extraction |
| VLM field extraction (p95) | < 5s per field | Single field region, Qwen2.5-VL-7B via Ollama |
| End-to-end with VLM fallback (p95) | < 90s | 3-page scanned PDF, 30 fields, 10 VLM fallback calls |

### 18.2 Resource Limits

| Resource | Limit | Enforcement |
|----------|-------|-------------|
| Peak memory per document | < 512MB | Image resolution capped at 10000x10000 px |
| OCR timeout per field | `form_ocr_per_field_timeout_seconds` (10s) | Hard timeout, field value = None on timeout |
| Total processing timeout | `per_document_timeout_seconds` (120s) | Hard timeout, return partial result with error |
| Maximum template field count | 200 | Validated at template creation |
| Maximum active templates | 500 (v1 target) | Soft limit; degradation logged above this |

### 18.3 Reliability Targets

| Metric | Target |
|--------|--------|
| Native PDF field extraction accuracy | >= 95% (fields correctly extracted from fillable PDFs) |
| OCR text field accuracy (clean scan) | >= 85% (character-level accuracy) |
| OCR checkbox detection accuracy | >= 90% |
| Auto-match true positive rate | >= 90% (correct template selected when confidence > threshold) |
| Auto-match false positive rate | < 5% (wrong template selected when confidence > threshold) |
| VLM fallback accuracy | >= 80% (on fields where OCR was < 0.4 confidence) |
| VLM budget compliance | 100% (never exceeds `form_vlm_max_fields_per_document`) |

### 18.4 Observability Contract

**Structured logging per stage:**

Every stage emits a structured log entry with these fields:

| Stage | Log Fields |
|-------|-----------|
| `match` | `template_candidates: int`, `top_confidence: float`, `match_duration_ms: float`, `match_result: str` ("auto", "manual", "fallthrough") |
| `extract` | `template_id`, `template_version`, `fields_extracted: int`, `fields_failed: int`, `extraction_method`, `extract_duration_ms: float` |
| `db_write` | `table_name`, `rows_written: int`, `schema_evolved: bool`, `db_write_duration_ms: float` |
| `embed` | `texts_embedded: int`, `embedding_dimension: int`, `embed_duration_ms: float` |
| `vlm_fallback` | `fields_sent_to_vlm: int`, `fields_improved: int`, `vlm_model: str`, `vlm_budget_remaining: int`, `vlm_total_duration_ms: float` |
| `vector_write` | `collection`, `points_upserted: int`, `vector_write_duration_ms: float` |

**Logger name:** `ingestkit_forms` (consistent with `ingestkit_excel`, `ingestkit_pdf`).

**Warning/error payloads:** All `FormIngestError` instances include the diagnostic context fields (`candidate_matches`, `backend_operation_id`, `fallback_reason`) when applicable. Field values are NEVER included in log payloads unless `log_sample_data=True`.

**Per-stage timing:** The `FormProcessingResult` includes `processing_time_seconds` (total). Individual stage timings are available in log entries only (not in the result model) to keep the result model lean.
