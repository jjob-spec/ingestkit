# Multi-Layer Tagging System --- V1 Specification

**Version:** 1.0
**Status:** BLOCKED
**Parent ecosystem:** `ingestkit` --- a plugin-based ingestion framework for the "AI Help Desk in a Box" on-premises RAG system.

> **BLOCKER:** The team must decide on the tagging architecture before implementation proceeds.
> The core question is where tagging logic lives and how pluggable it needs to be:
>
> | Option | Description | Trade-off |
> |--------|-------------|-----------|
> | **Simple** | Fixed 4-layer tagging built directly into ingestkit. Tag schema, validation, and query parsing all ship as core ingestkit code. | Fastest to ship, but locks all customers into one tagging strategy. |
> | **Plugin** | Tagging is a separate `ingestkit-tags` plugin package. Ingestkit core only defines the `TaggingBackend` Protocol and the `tags` field on chunk metadata. Concrete strategies (flat, hierarchical, taxonomy-service-backed) are pluggable. | Most flexible, but more upfront design work. |
> | **Combo** | ingestkit core owns tag data models + chunk metadata + auto-tag derivation (L4). Query-time parsing, alias management, and tag schema CRUD live in VE-RAG-System (orchestration layer). A `TaggingBackend` Protocol bridges the two. | Splits responsibility cleanly, but requires coordinated delivery across repos. |
>
> **Decision needed:** Which option best fits the product roadmap and customer deployment model?
> Assign to: engineering team lead / product owner.
> This spec is written assuming the **Simple** approach. If **Plugin** or **Combo** is chosen, sections 6, 10-13, and 14 will need significant restructuring.

---

## 1. Overview & Motivation

### 1.1 Problem Statement

Today, ingestkit produces chunks with flat, hardcoded metadata fields (`content_type`, `ingestion_method`, `heading_path`). These fields serve as implicit tags but have three critical limitations:

1. **No organizational taxonomy.** There is no way for an admin to label a document as belonging to HR, Legal, or Engineering. Every chunk from every department sits in the same undifferentiated collection.
2. **No query-time filtering.** Users cannot scope a question to a specific department or document type. The retrieval engine must search the entire corpus, producing noisy results in multi-department deployments.
3. **No controlled vocabulary.** Field values are derived from file structure, not organizational meaning. Two documents about the same topic may use completely different metadata values.

The `create_payload_index()` method exists on `VectorStoreBackend` but is never called. The infrastructure for filtered retrieval exists; the semantic layer on top of it does not.

### 1.2 Solution

A four-layer tagging system that:

- Assigns organizational tags at ingestion time (admin-controlled).
- Propagates tags from document level to every chunk.
- Auto-derives structural tags during processing.
- Resolves tags at query time into vector store pre-filters.
- Supports aliases so users do not need to memorize canonical vocabulary.

### 1.3 Scope

This specification covers the V1 implementation only. V1 delivers first-class tags, single-tag scoping, tag aliases, query-time tag resolution, and tag disambiguation. Features deferred to V2+ are listed in section 16 but not specified in detail.

### 1.4 Design Principles

1. **Admin-controlled taxonomy.** Tag layers and vocabularies are defined in configuration, not hardcoded. Different deployments can have different vocabularies.
2. **Fail-closed on required tags.** If a required tag layer is missing at ingestion time and `require_tags=True`, reject the ingestion call with a structured error. Never silently ingest untagged documents into a tagged collection.
3. **Additive metadata.** Tags are a new field on chunk metadata. Existing fields remain unchanged. No breaking changes to `ChunkMetadata` or `PDFChunkMetadata`.
4. **Backend-agnostic.** Tag filtering is expressed as abstract filter expressions. The vector store backend translates them to its native query language.
5. **Cross-package consistency.** The tagging data models and query-time resolution logic live in a shared location or are identically defined in both `ingestkit-excel` and `ingestkit-pdf`.

---

## 2. Use Cases

### UC-1: Department-Scoped Query

An HR employee asks: `HR: What is the PTO policy?`

The system extracts the `hr` tag, converts it to a vector store pre-filter on the `department` payload field, and retrieves only chunks from documents tagged with `department=hr`. The retrieval is faster and more precise than an unscoped similarity search across the entire corpus.

### UC-2: Multi-Layer Scoped Query

A compliance officer asks: `HR:Policy What are the termination procedures?`

The system extracts two tags --- `hr` (resolved to layer `department`) and `policy` (resolved to layer `document_type`) --- and constructs a conjunctive filter: `department=hr AND document_type=policy`.

### UC-3: Alias Resolution

A user asks: `People-Ops: How do I submit an expense report?`

The alias mapping resolves `people-ops` to canonical tag `hr`. The query proceeds as if the user had typed `HR:`.

### UC-4: Ambiguous Tag Disambiguation

Both `department` and `topic` layers contain the value `safety`. A user asks: `Safety: Do I need a harness?`

The system detects the ambiguity (value exists in multiple layers) and logs a warning. The query falls back to matching `safety` in **all** layers where it appears, producing a broad filter: `department=safety OR topic=safety`. If the user wants precision, they can write `topic.safety: Do I need a harness?`.

### UC-5: Auto-Derived Tags

An Excel file is classified as `TABULAR_DATA` with 15,000 rows. During processing, the system auto-derives L4 tags: `["tabular", "large-dataset"]`. These tags are attached to every chunk from the file and indexed for filtering.

### UC-6: Untagged Fallback

A user asks: `What is the wifi password?`

No tag prefix is detected. The system performs a pure similarity search across all chunks with no pre-filter. This is identical to current behavior.

### UC-7: Zero-Match Fallback

A user asks: `Facilities: Where is the server room?`

The `facilities` tag matches zero documents (no documents have been ingested with `department=facilities`). The system logs a warning, drops the filter, and falls back to unscoped similarity search. The response includes a warning: "No documents found for tag 'facilities'; showing unfiltered results."

### UC-8: Multi-Value Tag

A user asks: `HR+Legal: What about termination?`

The `+` syntax produces a disjunctive filter within the layer: `department IN [hr, legal]`. Combined with similarity search, this retrieves chunks from both HR and Legal documents.

---

## 3. V1 Scope Summary

| Feature | Status | Section |
|---------|--------|---------|
| Four tag layers (L1--L4) | V1 | 4 |
| Tag data models (Pydantic v2) | V1 | 5 |
| Ingestion-time tag assignment | V1 | 6 |
| Auto-tag derivation (L4) | V1 | 7 |
| Chunk metadata extension | V1 | 8 |
| Vector store payload indexing | V1 | 9 |
| Query-time tag resolution pipeline | V1 | 10 |
| Query prefix syntax parsing | V1 | 11 |
| Tag aliases | V1 | 12 |
| Filter construction & fallback | V1 | 13 |
| Configuration surface | V1 | 14 |
| Error handling | V1 | 15 |
| Boost mode (`~` prefix) | V2 | 16 |
| Negative tags (`!` prefix) | V2 | 16 |
| Hierarchical topics | V2 | 16 |
| Role-based default tags | V2 | 16 |
| Conversational tag stickiness | V3 | 16 |
| LLM hint inference | V3 | 16 |
| Temporal tags (current/archived) | V2 | 16 |
| Multi-source cross-reference | V3 | 16 |

---

## 4. Tag Layer Definitions

### 4.1 Layer Overview

| Layer | Name | Purpose | Assignment | Required | Vocabulary | Multiple Values |
|-------|------|---------|------------|----------|------------|-----------------|
| L1 | `department` | Organizational owner | Admin | Yes | Controlled | No |
| L2 | `document_type` | Functional category | Admin or auto | No | Controlled | No |
| L3 | `topic` | Subject matter | Admin | No | Free-form or controlled | Yes |
| L4 | `auto` | Structural metadata | System | No (always derived) | System-defined | Yes |

### 4.2 L1: Department

The organizational unit that owns the document. Every document must have exactly one department tag when `require_tags=True`.

**Default vocabulary:**

| Value | Description |
|-------|-------------|
| `hr` | Human Resources |
| `engineering` | Engineering / IT |
| `legal` | Legal / Compliance |
| `finance` | Finance / Accounting |
| `ops` | Operations / Facilities |

Admins may extend this vocabulary via configuration. The vocabulary is a closed set: values not in the list are rejected at ingestion time with a validation error.

### 4.3 L2: Document Type

The functional category of the document. May be assigned by the admin at ingestion time or auto-classified by the processing pipeline (if `auto_tag_document_type=True` in config).

**Default vocabulary:**

| Value | Description |
|-------|-------------|
| `policy` | Official policy document |
| `procedure` | Step-by-step procedure or SOP |
| `form` | Fillable form or template |
| `report` | Data report or analysis |
| `training` | Training material or guide |
| `reference` | Reference document or lookup table |

### 4.4 L3: Topic

The subject matter of the document. Supports multiple values per document (a benefits policy may cover both `benefits` and `compliance`).

**Default vocabulary (suggested, not enforced unless `topic_vocabulary` is set):**

| Value | Description |
|-------|-------------|
| `leave` | Leave of absence, PTO, sick leave |
| `safety` | Workplace safety, EHS |
| `onboarding` | New hire onboarding |
| `benefits` | Employee benefits |
| `compliance` | Regulatory compliance |
| `compensation` | Pay, bonuses, equity |
| `performance` | Performance reviews, PIPs |
| `it-support` | IT helpdesk, equipment, access |

When no controlled vocabulary is configured, L3 accepts any string value. When a controlled vocabulary is configured, unrecognized values are rejected.

### 4.5 L4: Auto-Derived

Structural metadata computed during the processing pipeline. These tags are system-generated and cannot be overridden by the admin.

**Defined values:**

| Value | Derived When |
|-------|-------------|
| `tabular` | File classified as `TABULAR_DATA` or `TEXT_NATIVE` with table extraction |
| `scanned` | PDF classified as `SCANNED` or contains OCR-processed pages |
| `multi-sheet` | Excel file with more than one non-empty sheet |
| `multi-page` | PDF with more than 10 pages |
| `has-tables` | Document contains extracted tables (any format) |
| `has-formulas` | Excel file contains formula cells |
| `large-dataset` | Tabular data exceeds 5,000 rows |
| `multilingual` | Language detection identifies more than one language |
| `has-images` | PDF pages contain embedded images covering >20% of page area |
| `ocr-processed` | One or more pages were processed via OCR engine |

---

## 5. Data Models

All models use Pydantic v2 `BaseModel`. These models are intended for a shared `ingestkit-core` package or identically duplicated in each ingestkit package until core is extracted.

### 5.1 TagLayer

```python
from __future__ import annotations

from pydantic import BaseModel, field_validator


class TagLayer(BaseModel):
    """Definition of a single tag layer in the taxonomy schema.

    Each layer has a name, an optional controlled vocabulary, and flags
    controlling whether it is required and whether multiple values are
    allowed per document.
    """

    name: str
    """Layer identifier. Must be unique across all layers. Lowercase, no spaces.
    Examples: 'department', 'document_type', 'topic', 'auto'."""

    required: bool = False
    """If True, ingestion fails when this layer is missing and
    ``require_tags=True`` in config."""

    vocabulary: list[str] = []
    """Controlled vocabulary. Empty list means free-form (any value accepted).
    Values are stored and compared in lowercase."""

    allow_multiple: bool = False
    """If True, a document may have multiple values for this layer.
    If False, only a single value is accepted."""

    hierarchical: bool = False
    """Reserved for V2. If True, values support parent/child relationships.
    V1 ignores this field during validation."""

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip().lower()
        if not v.isidentifier():
            raise ValueError(
                f"Tag layer name must be a valid Python identifier, got: {v!r}"
            )
        return v

    @field_validator("vocabulary")
    @classmethod
    def normalize_vocabulary(cls, v: list[str]) -> list[str]:
        return [item.strip().lower() for item in v]
```

### 5.2 TagSet

```python
class TagSet(BaseModel):
    """A set of tags assigned to a document or chunk.

    Keys are layer names. Values are lists of tag values within that layer.
    Single-value layers still use a list (with exactly one element) for
    uniform handling.
    """

    tags: dict[str, list[str]] = {}
    """Mapping of layer_name -> list of tag values.
    Example: {"department": ["hr"], "topic": ["leave", "benefits"]}"""

    def get_layer(self, layer_name: str) -> list[str]:
        """Return tag values for a layer, or empty list if unset."""
        return self.tags.get(layer_name, [])

    def has_layer(self, layer_name: str) -> bool:
        """Return True if the layer has at least one value."""
        return bool(self.tags.get(layer_name))

    def merge(self, other: TagSet) -> TagSet:
        """Return a new TagSet combining self and other.

        Values from other are appended (not replacing). Duplicates are
        removed while preserving order.
        """
        merged: dict[str, list[str]] = {}
        all_keys = set(self.tags) | set(other.tags)
        for key in all_keys:
            combined = self.tags.get(key, []) + other.tags.get(key, [])
            # Deduplicate preserving order
            seen: set[str] = set()
            deduped: list[str] = []
            for val in combined:
                if val not in seen:
                    seen.add(val)
                    deduped.append(val)
            merged[key] = deduped
        return TagSet(tags=merged)

    def flat_values(self) -> set[str]:
        """Return all tag values across all layers as a flat set."""
        result: set[str] = set()
        for values in self.tags.values():
            result.update(values)
        return result
```

### 5.3 TagAlias

```python
class TagAlias(BaseModel):
    """Mapping of alternate names to canonical tag values.

    Aliases are resolved before tag validation, so users can type natural
    names that get mapped to the controlled vocabulary.
    """

    aliases: dict[str, str] = {}
    """Mapping of alternate_name -> canonical_tag_value.
    All keys and values are stored and compared in lowercase.
    Example: {"people-ops": "hr", "guidelines": "policy", "ehs": "safety"}"""

    @field_validator("aliases")
    @classmethod
    def normalize_aliases(cls, v: dict[str, str]) -> dict[str, str]:
        return {k.strip().lower(): val.strip().lower() for k, val in v.items()}

    def resolve(self, value: str) -> str:
        """Resolve a value through the alias map. Returns canonical value
        if an alias exists, otherwise returns the input unchanged (lowered)."""
        normalized = value.strip().lower()
        return self.aliases.get(normalized, normalized)
```

### 5.4 TagSchema

```python
class TagSchema(BaseModel):
    """Complete tag taxonomy definition for a deployment.

    Combines all layer definitions and the alias map into a single
    configuration object that drives ingestion-time validation and
    query-time resolution.
    """

    layers: list[TagLayer] = []
    """Ordered list of tag layer definitions."""

    aliases: TagAlias = TagAlias()
    """Global alias map across all layers."""

    def get_layer(self, name: str) -> TagLayer | None:
        """Look up a layer definition by name."""
        for layer in self.layers:
            if layer.name == name:
                return layer
        return None

    def layer_names(self) -> list[str]:
        """Return ordered list of layer names."""
        return [layer.name for layer in self.layers]

    def required_layers(self) -> list[str]:
        """Return names of layers where required=True."""
        return [layer.name for layer in self.layers if layer.required]

    def find_value_layers(self, value: str) -> list[str]:
        """Return names of all layers whose vocabulary contains the given value.

        Used for disambiguation: if the result has length > 1, the value
        is ambiguous and the user should use layer.value syntax.
        """
        normalized = value.strip().lower()
        matches: list[str] = []
        for layer in self.layers:
            if not layer.vocabulary:
                # Free-form layers always match (cannot be excluded)
                matches.append(layer.name)
            elif normalized in layer.vocabulary:
                matches.append(layer.name)
        return matches
```

### 5.5 Query-Time Models

```python
from enum import Enum


class TagFilterOp(str, Enum):
    """Operator for combining tag filter conditions."""
    AND = "and"
    OR = "or"


class TagFilterCondition(BaseModel):
    """A single tag filter condition for vector store pre-filtering."""

    layer: str
    """Tag layer name (used as the payload field name)."""

    values: list[str]
    """One or more tag values. Multiple values within a single condition
    are combined with OR (match any)."""


class TagFilter(BaseModel):
    """Complete tag filter for a query, combining multiple conditions.

    Conditions across different layers are combined with AND.
    Values within a single condition (same layer) are combined with OR.
    """

    conditions: list[TagFilterCondition] = []
    """List of per-layer filter conditions."""

    @property
    def is_empty(self) -> bool:
        return len(self.conditions) == 0


class ParsedQuery(BaseModel):
    """Result of parsing a user query string for tag prefixes."""

    raw_query: str
    """Original query string as submitted by user."""

    clean_query: str
    """Query string with tag prefixes removed."""

    tag_filter: TagFilter
    """Extracted and validated tag filter."""

    warnings: list[str] = []
    """Warnings generated during parsing (unrecognized tags, ambiguities)."""

    used_aliases: dict[str, str] = {}
    """Record of alias resolutions applied: {original -> canonical}."""
```

---

## 6. Ingestion-Time: Tag Assignment & Validation

### 6.1 Tag Input

Tags are provided as a flat dictionary in the ingestion call. The key is the layer name; the value is a string (single value) or list of strings (multiple values).

```python
# Example: ingesting an HR policy document
result = router.process(
    file_path="employee_handbook.xlsx",
    config=config,
    tags={
        "department": "hr",
        "document_type": "policy",
        "topic": ["onboarding", "benefits"],
    },
)
```

### 6.2 Validation Rules

Validation runs before any parsing or processing begins. This ensures invalid tags never enter the pipeline.

| Rule | Condition | Error Code | Behavior |
|------|-----------|------------|----------|
| Required layer missing | `require_tags=True` and a required layer has no value | `E_TAG_REQUIRED_MISSING` | Reject ingestion |
| Unknown layer name | Tag key is not a defined layer name | `W_TAG_UNKNOWN_LAYER` | Warning; drop the tag |
| Value not in vocabulary | Controlled vocab layer receives unrecognized value | `E_TAG_INVALID_VALUE` | Reject ingestion |
| Multiple values on single-value layer | `allow_multiple=False` but multiple values provided | `E_TAG_MULTIPLE_NOT_ALLOWED` | Reject ingestion |
| L4 layer assigned manually | Admin attempts to set `auto` layer tags | `W_TAG_AUTO_OVERRIDE` | Warning; ignore admin value |

### 6.3 Validation Pseudocode

```python
def validate_tags(
    raw_tags: dict[str, str | list[str]],
    schema: TagSchema,
    require_tags: bool,
) -> tuple[TagSet, list[IngestError]]:
    """Validate and normalize admin-provided tags against the schema.

    Returns a validated TagSet and a list of errors/warnings.
    Errors with non-recoverable codes should abort ingestion.
    """
    errors: list[IngestError] = []
    normalized: dict[str, list[str]] = {}

    # Normalize single values to lists
    for layer_name, values in raw_tags.items():
        if isinstance(values, str):
            values = [values]
        normalized[layer_name] = [v.strip().lower() for v in values]

    # Check for unknown layers
    known_layers = set(schema.layer_names())
    for layer_name in list(normalized.keys()):
        if layer_name not in known_layers:
            errors.append(IngestError(
                code="W_TAG_UNKNOWN_LAYER",
                message=f"Unknown tag layer '{layer_name}'; dropping.",
                stage="tag_validation",
                recoverable=True,
            ))
            del normalized[layer_name]

    # Check for manual L4 override
    if "auto" in normalized:
        errors.append(IngestError(
            code="W_TAG_AUTO_OVERRIDE",
            message="Cannot manually assign 'auto' layer tags; ignoring.",
            stage="tag_validation",
            recoverable=True,
        ))
        del normalized["auto"]

    # Check required layers
    if require_tags:
        for layer_name in schema.required_layers():
            if layer_name not in normalized or not normalized[layer_name]:
                errors.append(IngestError(
                    code="E_TAG_REQUIRED_MISSING",
                    message=f"Required tag layer '{layer_name}' is missing.",
                    stage="tag_validation",
                    recoverable=False,
                ))

    # Validate values against vocabulary
    for layer_name, values in normalized.items():
        layer_def = schema.get_layer(layer_name)
        if layer_def is None:
            continue

        # Check allow_multiple
        if not layer_def.allow_multiple and len(values) > 1:
            errors.append(IngestError(
                code="E_TAG_MULTIPLE_NOT_ALLOWED",
                message=(
                    f"Layer '{layer_name}' does not allow multiple values; "
                    f"got {len(values)}."
                ),
                stage="tag_validation",
                recoverable=False,
            ))

        # Check vocabulary
        if layer_def.vocabulary:
            for val in values:
                if val not in layer_def.vocabulary:
                    errors.append(IngestError(
                        code="E_TAG_INVALID_VALUE",
                        message=(
                            f"Value '{val}' is not in the vocabulary for "
                            f"layer '{layer_name}'. "
                            f"Valid values: {layer_def.vocabulary}"
                        ),
                        stage="tag_validation",
                        recoverable=False,
                    ))

    # Build TagSet from validated tags (only if no non-recoverable errors)
    tag_set = TagSet(tags=normalized)
    return tag_set, errors
```

### 6.4 Tag Propagation

Once validated, the `TagSet` is attached to the processing context and propagated to every chunk produced during processing:

1. **Document level:** `TagSet` stored on the processing context object.
2. **L4 auto-tags:** Derived during processing (see section 7) and merged into the `TagSet`.
3. **Chunk level:** When a `ChunkMetadata` or `PDFChunkMetadata` is created, the `tags` field is populated from the merged `TagSet`.

```
Admin tags (L1-L3)  ──┐
                       ├──▶  Merged TagSet  ──▶  ChunkMetadata.tags
Auto-derived (L4)  ───┘
```

---

## 7. Processing-Time: Auto-Tag Derivation (L4)

### 7.1 Derivation Rules

Auto-tags are derived from signals already computed during the classification and processing stages. No additional file parsing is required.

#### Excel (ingestkit-excel)

| Auto-Tag | Derivation Logic |
|----------|-----------------|
| `tabular` | `ClassificationResult.file_type == FileType.TABULAR_DATA` |
| `multi-sheet` | `FileProfile.sheet_count > 1` and more than one sheet has `row_count > 0` |
| `has-tables` | `ClassificationResult.file_type in (TABULAR_DATA, HYBRID)` |
| `has-formulas` | Any `SheetProfile.has_formulas == True` |
| `large-dataset` | Any `SheetProfile.row_count > config.row_serialization_limit` |

#### PDF (ingestkit-pdf)

| Auto-Tag | Derivation Logic |
|----------|-----------------|
| `tabular` | Any page has `PageType.TABLE_HEAVY` or tables were extracted |
| `scanned` | `ClassificationResult.pdf_type == PDFType.SCANNED` or any page is `PageType.SCANNED` |
| `multi-page` | `DocumentProfile.page_count > 10` |
| `has-tables` | `TableResult` list is non-empty |
| `has-images` | Any `PageProfile.image_coverage_ratio > 0.20` |
| `multilingual` | `len(DocumentProfile.detected_languages) > 1` |
| `ocr-processed` | `OCRStageResult` is not None and `pages_ocrd > 0` |

### 7.2 Derivation Pseudocode (Excel)

```python
def derive_auto_tags_excel(
    file_profile: FileProfile,
    classification: ClassificationResult,
    config: ExcelProcessorConfig,
) -> TagSet:
    """Derive L4 auto-tags from Excel processing artifacts."""
    auto_values: list[str] = []

    if classification.file_type == FileType.TABULAR_DATA:
        auto_values.append("tabular")

    non_empty_sheets = [s for s in file_profile.sheets if s.row_count > 0]
    if len(non_empty_sheets) > 1:
        auto_values.append("multi-sheet")

    if classification.file_type in (FileType.TABULAR_DATA, FileType.HYBRID):
        auto_values.append("has-tables")

    if any(s.has_formulas for s in file_profile.sheets):
        auto_values.append("has-formulas")

    if any(s.row_count > config.row_serialization_limit for s in file_profile.sheets):
        auto_values.append("large-dataset")

    return TagSet(tags={"auto": auto_values} if auto_values else {})
```

---

## 8. Chunk Metadata Extension

### 8.1 Field Addition

Both `ChunkMetadata` (Excel) and `PDFChunkMetadata` (PDF) gain a single new field:

```python
class ChunkMetadata(BaseModel):
    # ... all existing fields unchanged ...

    tags: dict[str, list[str]] = {}
    """Tag layer values propagated from document-level TagSet.
    Keys are layer names, values are lists of tag values.
    Example: {"department": ["hr"], "document_type": ["policy"],
              "topic": ["leave", "benefits"], "auto": ["tabular"]}"""
```

```python
class PDFChunkMetadata(BaseModel):
    # ... all existing fields unchanged ...

    tags: dict[str, list[str]] = {}
    """Tag layer values propagated from document-level TagSet.
    Same schema as ChunkMetadata.tags."""
```

### 8.2 Backward Compatibility

- The default value is `{}` (empty dict). Existing chunks without tags are valid.
- No existing field is removed or renamed.
- Deserialization of payloads without a `tags` field will use the default.
- Code that does not reference `tags` is unaffected.

### 8.3 Payload Serialization

When the chunk is upserted to the vector store, the `tags` dict is flattened into the payload as individual fields per layer:

```python
# Chunk metadata tags:
# {"department": ["hr"], "topic": ["leave", "benefits"], "auto": ["tabular"]}

# Flattened into vector store payload:
{
    "tag_department": ["hr"],
    "tag_document_type": [],          # absent layer = empty list
    "tag_topic": ["leave", "benefits"],
    "tag_auto": ["tabular"],
    # ... all other metadata fields ...
}
```

The `tag_` prefix avoids collisions with existing payload fields. Each tag layer becomes a separate payload field containing a list of strings.

---

## 9. Vector Store Indexing

### 9.1 Index Creation

During collection setup (or on first ingestion after tagging is enabled), call `create_payload_index()` for each tag layer:

```python
def setup_tag_indexes(
    vector_store: VectorStoreBackend,
    collection: str,
    schema: TagSchema,
) -> None:
    """Create payload indexes for all tag layers.

    Should be called once during collection initialization. Calling
    on an already-indexed collection is idempotent (Qdrant ignores
    duplicate index creation).
    """
    for layer in schema.layers:
        field_name = f"tag_{layer.name}"
        vector_store.create_payload_index(
            collection=collection,
            field=field_name,
            field_type="keyword",  # Qdrant keyword index supports exact match on strings
        )
```

### 9.2 Qdrant Payload Index Mapping

| Tag Layer | Payload Field | Qdrant Field Type | Index Type |
|-----------|--------------|-------------------|------------|
| `department` | `tag_department` | `keyword` | Keyword index |
| `document_type` | `tag_document_type` | `keyword` | Keyword index |
| `topic` | `tag_topic` | `keyword` | Keyword index |
| `auto` | `tag_auto` | `keyword` | Keyword index |

Qdrant's `keyword` type supports filtering on string values, including `match` (exact) and `match_any` (OR over a list). This aligns with the tag filter model.

### 9.3 Index Lifecycle

- **Creation:** On collection setup or first tagged ingestion.
- **Updates:** Automatic. Qdrant updates indexes on upsert.
- **Deletion:** Handled by collection deletion (out of scope for V1).
- **Migration:** Existing collections without tag indexes gain them when `setup_tag_indexes` is called. Existing points without tag fields are simply not matched by tag filters (correct behavior).

---

## 10. Query-Time: Tag Resolution Pipeline

### 10.1 Pipeline Overview

```
User query string
        |
        v
  +-----------------+
  | Prefix Parser   |  Extract tag prefixes from query string
  +-----------------+
        |
        v
  +-----------------+
  | Alias Resolver  |  Map alternate names to canonical tags
  +-----------------+
        |
        v
  +-----------------+
  | Disambiguator   |  Resolve which layer each tag belongs to
  +-----------------+
        |
        v
  +-----------------+
  | Tag Validator   |  Validate against schema vocabulary
  +-----------------+
        |
        v
  +-----------------+
  | Filter Builder  |  Convert to vector store pre-filter
  +-----------------+
        |
        v
  ParsedQuery {clean_query, tag_filter, warnings}
```

### 10.2 Pipeline Pseudocode

```python
def resolve_query_tags(
    query: str,
    schema: TagSchema,
) -> ParsedQuery:
    """Full tag resolution pipeline: parse, resolve, disambiguate, validate, build."""
    warnings: list[str] = []
    used_aliases: dict[str, str] = {}

    # Step 1: Parse prefixes
    prefix_str, clean_query = parse_prefix(query)
    if prefix_str is None:
        return ParsedQuery(
            raw_query=query,
            clean_query=clean_query,
            tag_filter=TagFilter(),
            warnings=[],
        )

    # Step 2: Split multi-value prefixes (colon-separated layers, + within layer)
    raw_tags = split_prefix(prefix_str)
    # raw_tags: list of (optional_layer, value) tuples

    # Step 3: Resolve aliases
    resolved_tags: list[tuple[str | None, str]] = []
    for layer, value in raw_tags:
        canonical = schema.aliases.resolve(value)
        if canonical != value:
            used_aliases[value] = canonical
        resolved_tags.append((layer, canonical))

    # Step 4: Disambiguate (determine layer for each value)
    conditions: dict[str, list[str]] = {}  # layer_name -> values
    for explicit_layer, value in resolved_tags:
        if explicit_layer is not None:
            # User specified layer explicitly (layer.value syntax)
            layer_def = schema.get_layer(explicit_layer)
            if layer_def is None:
                warnings.append(f"Unknown layer '{explicit_layer}'; dropping tag.")
                continue
            conditions.setdefault(explicit_layer, []).append(value)
        else:
            # Auto-detect layer
            matching_layers = schema.find_value_layers(value)
            if len(matching_layers) == 0:
                warnings.append(
                    f"Tag '{value}' not found in any layer vocabulary; dropping."
                )
            elif len(matching_layers) == 1:
                conditions.setdefault(matching_layers[0], []).append(value)
            else:
                # Ambiguous: value exists in multiple layers
                # Filter to only controlled-vocab layers that explicitly contain the value
                controlled_matches = [
                    ln for ln in matching_layers
                    if value in (schema.get_layer(ln).vocabulary or [])
                ]
                if len(controlled_matches) == 1:
                    conditions.setdefault(controlled_matches[0], []).append(value)
                else:
                    warnings.append(
                        f"Tag '{value}' is ambiguous (found in layers: "
                        f"{matching_layers}). Use 'layer.value' syntax for "
                        f"precision. Matching across all layers."
                    )
                    for ln in matching_layers:
                        conditions.setdefault(ln, []).append(value)

    # Step 5: Validate values against vocabulary
    valid_conditions: list[TagFilterCondition] = []
    for layer_name, values in conditions.items():
        layer_def = schema.get_layer(layer_name)
        if layer_def is None:
            continue
        if layer_def.vocabulary:
            valid_values = []
            for v in values:
                if v in layer_def.vocabulary:
                    valid_values.append(v)
                else:
                    warnings.append(
                        f"Tag '{v}' is not in vocabulary for layer "
                        f"'{layer_name}'; dropping."
                    )
            if valid_values:
                valid_conditions.append(
                    TagFilterCondition(layer=layer_name, values=valid_values)
                )
        else:
            # Free-form layer: accept all values
            valid_conditions.append(
                TagFilterCondition(layer=layer_name, values=values)
            )

    return ParsedQuery(
        raw_query=query,
        clean_query=clean_query,
        tag_filter=TagFilter(conditions=valid_conditions),
        warnings=warnings,
        used_aliases=used_aliases,
    )
```

---

## 11. Query Syntax & Prefix Parsing

### 11.1 Supported Syntax

| Pattern | Example | Interpretation |
|---------|---------|---------------|
| Single tag | `Policy: What is PTO?` | `document_type=policy` (auto-detected layer) |
| Layer-qualified tag | `department.hr: What is PTO?` | `department=hr` (explicit layer) |
| Multi-layer (colon chain) | `HR:Policy What is PTO?` | `department=hr AND document_type=policy` |
| Multi-value (plus) | `HR+Legal: What about termination?` | `department IN [hr, legal]` |
| Combined | `HR+Legal:Policy What about termination?` | `department IN [hr, legal] AND document_type=policy` |
| No prefix | `What is the wifi password?` | No filter (pure similarity search) |

### 11.2 Grammar

```
query         := prefix_chain? free_text
prefix_chain  := prefix_group (":" prefix_group)* ":"
prefix_group  := tag_value ("+" tag_value)*
tag_value     := layer_name "." value_name
              |  value_name
layer_name    := [a-z][a-z0-9_]*
value_name    := [a-z][a-z0-9_-]*
```

The prefix chain is terminated by the **last** colon followed by a space and free text. If no colon is present, the entire string is free text.

### 11.3 Parsing Pseudocode

```python
import re

# Match one or more colon-separated prefix groups at the start of the query.
# Each prefix group contains one or more '+'-separated tag values.
# The prefix chain ends with ':' followed by a space (or end of string).
PREFIX_PATTERN = re.compile(
    r"^((?:[a-zA-Z][a-zA-Z0-9_.*-]*(?:\+[a-zA-Z][a-zA-Z0-9_.*-]*)*)(?::(?=[a-zA-Z])(?:[a-zA-Z][a-zA-Z0-9_.*-]*(?:\+[a-zA-Z][a-zA-Z0-9_.*-]*)*))*):\s*(.*)",
    re.DOTALL,
)


def parse_prefix(query: str) -> tuple[str | None, str]:
    """Extract tag prefix chain from query string.

    Returns (prefix_string, clean_query). If no prefix is found,
    returns (None, original_query).

    Examples:
        "HR: What is PTO?"           -> ("HR", "What is PTO?")
        "HR:Policy What is PTO?"     -> ("HR:Policy", "What is PTO?")
        "HR+Legal: What about X?"    -> ("HR+Legal", "What about X?")
        "What is the wifi password?" -> (None, "What is the wifi password?")
    """
    query = query.strip()
    match = PREFIX_PATTERN.match(query)
    if match:
        return match.group(1), match.group(2).strip()
    return None, query


def split_prefix(prefix_str: str) -> list[tuple[str | None, str]]:
    """Split a prefix string into (optional_layer, value) tuples.

    Examples:
        "HR"                -> [(None, "hr")]
        "HR:Policy"         -> [(None, "hr"), (None, "policy")]
        "HR+Legal"          -> [(None, "hr"), (None, "legal")]
        "department.hr"     -> [("department", "hr")]
        "HR+Legal:Policy"   -> [(None, "hr"), (None, "legal"), (None, "policy")]
    """
    result: list[tuple[str | None, str]] = []
    groups = prefix_str.split(":")
    for group in groups:
        values = group.split("+")
        for val in values:
            val = val.strip().lower()
            if "." in val:
                parts = val.split(".", 1)
                result.append((parts[0], parts[1]))
            else:
                result.append((None, val))
    return result
```

### 11.4 Edge Cases

| Input | Behavior |
|-------|----------|
| `": What is PTO?"` | Empty prefix; treated as no prefix. |
| `"HR:"` (no query text) | Valid prefix, empty query. Warn caller: "Empty query after tag prefix." |
| `"HR:Policy:Procedure: What?"` | Three prefix groups: `hr`, `policy`, `procedure`. |
| `"http://example.com"` | The `://` is not a valid prefix pattern; treated as untagged query. |
| `"Step 3: Mix the ingredients"` | `Step 3` is not a valid tag (contains a space before `:`); treated as untagged query. |
| `"Re: Your request"` | `Re` is a valid pattern syntactically. If `re` is not in any vocabulary, it produces a warning and falls back to unfiltered. |

---

## 12. Tag Aliases

### 12.1 Purpose

Users should not need to memorize the canonical vocabulary. Aliases map common alternate names to their canonical form.

### 12.2 Default Alias Map

```yaml
aliases:
  # Department aliases
  people-ops: hr
  human-resources: hr
  eng: engineering
  it: engineering
  tech: engineering
  law: legal
  compliance: legal
  accounting: finance
  facilities: ops
  operations: ops

  # Document type aliases
  guidelines: policy
  standard: policy
  sop: procedure
  process: procedure
  howto: procedure
  how-to: procedure
  template: form
  worksheet: form
  analysis: report
  dashboard: report
  guide: training
  tutorial: training
  handbook: training
  manual: reference
  catalog: reference
  directory: reference

  # Topic aliases
  pto: leave
  vacation: leave
  time-off: leave
  sick-leave: leave
  ehs: safety
  health-and-safety: safety
  workplace-safety: safety
  new-hire: onboarding
  orientation: onboarding
  insurance: benefits
  health-plan: benefits
  401k: benefits
  retirement: benefits
  pay: compensation
  salary: compensation
  bonus: compensation
  equity: compensation
  review: performance
  pip: performance
  helpdesk: it-support
  equipment: it-support
  access: it-support
```

### 12.3 Alias Resolution Order

1. Lowercase and strip the input value.
2. Look up in the alias map.
3. If found, replace with canonical value.
4. If not found, pass through unchanged.
5. Record the resolution in `ParsedQuery.used_aliases` for observability.

Aliases are resolved **before** disambiguation and vocabulary validation. This means an alias can map a user-friendly name to a canonical value that then gets correctly routed to its layer.

### 12.4 Alias Conflicts

An alias key must not collide with a canonical vocabulary value. If `safety` is both a canonical value and an alias key, the canonical value takes precedence (alias is ignored for that key). The `TagSchema` validator should warn at config load time if any alias key shadows a canonical vocabulary value.

---

## 13. Filter Construction & Fallback Behavior

### 13.1 Filter Construction

After tag resolution produces a `TagFilter`, it must be translated into the vector store's native query language.

#### Qdrant Filter Example

```python
from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue


def build_qdrant_filter(tag_filter: TagFilter) -> Filter | None:
    """Convert a TagFilter into a Qdrant Filter object.

    Returns None if the tag filter is empty (no conditions).
    Conditions across layers are combined with AND (must).
    Values within a layer are combined with OR (match_any).
    """
    if tag_filter.is_empty:
        return None

    must_conditions: list[FieldCondition] = []

    for condition in tag_filter.conditions:
        field_name = f"tag_{condition.layer}"
        if len(condition.values) == 1:
            must_conditions.append(
                FieldCondition(
                    key=field_name,
                    match=MatchValue(value=condition.values[0]),
                )
            )
        else:
            must_conditions.append(
                FieldCondition(
                    key=field_name,
                    match=MatchAny(any=condition.values),
                )
            )

    return Filter(must=must_conditions)
```

#### Example Filters

**Single tag:** `HR: What is PTO?`

```json
{
    "must": [
        {"key": "tag_department", "match": {"value": "hr"}}
    ]
}
```

**Multi-layer:** `HR:Policy What is PTO?`

```json
{
    "must": [
        {"key": "tag_department", "match": {"value": "hr"}},
        {"key": "tag_document_type", "match": {"value": "policy"}}
    ]
}
```

**Multi-value:** `HR+Legal: What about termination?`

```json
{
    "must": [
        {"key": "tag_department", "match": {"any": ["hr", "legal"]}}
    ]
}
```

**Ambiguous tag (cross-layer):** `Safety: Do I need a harness?`

```json
{
    "should": [
        {"key": "tag_department", "match": {"value": "safety"}},
        {"key": "tag_topic", "match": {"value": "safety"}}
    ]
}
```

Note: When a tag is ambiguous and resolves to multiple layers, the conditions for that value are combined with `should` (OR) rather than `must` (AND), because the user intended one layer but we cannot determine which.

### 13.2 Fallback Behavior

The filter pipeline implements a two-stage fallback:

| Scenario | Behavior |
|----------|----------|
| Tag prefix present, tags valid, results found | Return filtered results. |
| Tag prefix present, tags valid, **zero results** | Log warning `W_TAG_ZERO_MATCHES`. Drop filter. Re-execute as unfiltered similarity search. Prepend warning to response: "No documents matched tag filter; showing unfiltered results." |
| Tag prefix present, **all tags invalid** | Log warning `W_TAG_ALL_DROPPED`. Execute as unfiltered search. Prepend warning: "No recognized tags in query; showing unfiltered results." |
| Tag prefix present, **some tags invalid** | Drop invalid tags (with per-tag warnings). Execute with remaining valid tags. |
| No tag prefix | Execute as unfiltered similarity search. No warnings. |

### 13.3 Fallback Pseudocode

```python
def execute_filtered_search(
    query: str,
    parsed: ParsedQuery,
    vector_store: VectorStoreBackend,
    embedding_backend: EmbeddingBackend,
    collection: str,
    top_k: int = 10,
) -> SearchResult:
    """Execute a vector search with tag pre-filtering and fallback."""

    query_vector = embedding_backend.embed([parsed.clean_query])[0]

    if not parsed.tag_filter.is_empty:
        qdrant_filter = build_qdrant_filter(parsed.tag_filter)
        results = vector_store.search(
            collection=collection,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=top_k,
        )

        if len(results) > 0:
            return SearchResult(
                hits=results,
                warnings=parsed.warnings,
                filter_applied=True,
            )

        # Zero-match fallback
        parsed.warnings.append(
            "No documents matched tag filter; showing unfiltered results."
        )
        results = vector_store.search(
            collection=collection,
            query_vector=query_vector,
            query_filter=None,
            limit=top_k,
        )
        return SearchResult(
            hits=results,
            warnings=parsed.warnings,
            filter_applied=False,
        )

    # No tags: unfiltered search
    results = vector_store.search(
        collection=collection,
        query_vector=query_vector,
        query_filter=None,
        limit=top_k,
    )
    return SearchResult(
        hits=results,
        warnings=parsed.warnings,
        filter_applied=False,
    )
```

---

## 14. Configuration Surface

### 14.1 New Config Fields

The tagging configuration is added to a new top-level config model. It can be provided alongside `ExcelProcessorConfig` or `PDFProcessorConfig`, or loaded from a shared YAML file.

```python
class TaggingConfig(BaseModel):
    """Configuration for the multi-layer tagging system.

    Can be loaded from YAML/JSON alongside the processor config, or
    provided programmatically.
    """

    # --- Tag schema ---
    tag_schema: TagSchema = TagSchema(
        layers=[
            TagLayer(
                name="department",
                required=True,
                vocabulary=["hr", "engineering", "legal", "finance", "ops"],
                allow_multiple=False,
            ),
            TagLayer(
                name="document_type",
                required=False,
                vocabulary=[
                    "policy", "procedure", "form",
                    "report", "training", "reference",
                ],
                allow_multiple=False,
            ),
            TagLayer(
                name="topic",
                required=False,
                vocabulary=[],  # Free-form by default
                allow_multiple=True,
            ),
            TagLayer(
                name="auto",
                required=False,
                vocabulary=[
                    "tabular", "scanned", "multi-sheet", "multi-page",
                    "has-tables", "has-formulas", "large-dataset",
                    "multilingual", "has-images", "ocr-processed",
                ],
                allow_multiple=True,
            ),
        ],
    )

    # --- Alias mapping ---
    tag_aliases: TagAlias = TagAlias()
    """See section 12 for default alias map."""

    # --- Behavior flags ---
    require_tags: bool = True
    """If True, ingestion fails when required tag layers are missing.
    If False, missing required layers produce warnings but ingestion proceeds."""

    auto_tag_document_type: bool = False
    """If True, the system attempts to auto-classify document_type (L2)
    using the same LLM classifier used for structural classification.
    V1: not implemented; reserved for future use."""

    auto_tag_content_type: bool = True
    """If True, L4 auto-tags include structural content type tags
    (tabular, scanned, etc.)."""

    auto_tag_language: bool = True
    """If True, L4 auto-tags include 'multilingual' when multiple
    languages are detected."""

    strict_vocabulary: bool = True
    """If True, reject ingestion when a tag value is not in the
    controlled vocabulary. If False, accept any value with a warning."""

    tag_field_prefix: str = "tag_"
    """Prefix for tag payload fields in the vector store.
    Default 'tag_' produces fields like 'tag_department', 'tag_topic'."""
```

### 14.2 YAML Configuration Example

```yaml
# tagging.yaml - Tag configuration for the Help Desk deployment

tag_schema:
  layers:
    - name: department
      required: true
      vocabulary: [hr, engineering, legal, finance, ops, facilities]
      allow_multiple: false
    - name: document_type
      required: false
      vocabulary: [policy, procedure, form, report, training, reference]
      allow_multiple: false
    - name: topic
      required: false
      vocabulary: [leave, safety, onboarding, benefits, compliance, compensation, performance, it-support]
      allow_multiple: true
    - name: auto
      required: false
      vocabulary: [tabular, scanned, multi-sheet, multi-page, has-tables, has-formulas, large-dataset, multilingual, has-images, ocr-processed]
      allow_multiple: true

tag_aliases:
  aliases:
    people-ops: hr
    human-resources: hr
    eng: engineering
    it: engineering
    law: legal
    guidelines: policy
    sop: procedure
    ehs: safety
    pto: leave

require_tags: true
auto_tag_content_type: true
auto_tag_language: true
strict_vocabulary: true
tag_field_prefix: "tag_"
```

### 14.3 Ingestion Call Example

```python
from ingestkit_excel import ExcelRouter
from ingestkit_excel.config import ExcelProcessorConfig

config = ExcelProcessorConfig.from_file("config.yaml")
tagging_config = TaggingConfig.from_file("tagging.yaml")

router = ExcelRouter(
    vector_store=qdrant_backend,
    structured_db=sqlite_backend,
    llm=ollama_backend,
    embeddings=embedding_backend,
)

result = router.process(
    file_path="employee_handbook.xlsx",
    config=config,
    tagging_config=tagging_config,
    tags={
        "department": "hr",
        "document_type": "policy",
        "topic": ["onboarding", "benefits"],
    },
)

# result.chunks_created -> 42
# Each chunk has metadata.tags == {
#     "department": ["hr"],
#     "document_type": ["policy"],
#     "topic": ["onboarding", "benefits"],
#     "auto": ["multi-sheet", "has-tables"],
# }
```

---

## 15. Error Handling

### 15.1 New Error Codes

The following error codes are added to the `ErrorCode` enum in each package's `errors.py`:

| Code | Type | Description |
|------|------|-------------|
| `E_TAG_REQUIRED_MISSING` | Error | Required tag layer has no value and `require_tags=True`. |
| `E_TAG_INVALID_VALUE` | Error | Tag value not in controlled vocabulary (when `strict_vocabulary=True`). |
| `E_TAG_MULTIPLE_NOT_ALLOWED` | Error | Multiple values provided for a single-value layer. |
| `E_TAG_SCHEMA_INVALID` | Error | Tag schema configuration is malformed (e.g., duplicate layer names). |
| `W_TAG_UNKNOWN_LAYER` | Warning | Tag provided for an undefined layer; dropped. |
| `W_TAG_AUTO_OVERRIDE` | Warning | Admin attempted to set L4 auto tags; ignored. |
| `W_TAG_ZERO_MATCHES` | Warning | Tag filter matched zero documents; fell back to unfiltered. |
| `W_TAG_ALL_DROPPED` | Warning | All tag prefixes in query were unrecognized; unfiltered search used. |
| `W_TAG_AMBIGUOUS` | Warning | Tag value exists in multiple layers; broad filter applied. |
| `W_TAG_ALIAS_SHADOW` | Warning | An alias key shadows a canonical vocabulary value (config load time). |

### 15.2 Error Behavior Summary

| Stage | Error Severity | Behavior |
|-------|---------------|----------|
| Config load | `E_TAG_SCHEMA_INVALID` | Raise immediately. Application cannot start with invalid schema. |
| Config load | `W_TAG_ALIAS_SHADOW` | Log warning. Application starts normally. |
| Ingestion-time validation | `E_TAG_REQUIRED_MISSING` | Abort ingestion. Return `ProcessingResult` with error. |
| Ingestion-time validation | `E_TAG_INVALID_VALUE` | Abort ingestion if `strict_vocabulary=True`. Warning if `False`. |
| Ingestion-time validation | `E_TAG_MULTIPLE_NOT_ALLOWED` | Abort ingestion. |
| Ingestion-time validation | `W_TAG_UNKNOWN_LAYER` | Log warning, drop tag, continue. |
| Ingestion-time validation | `W_TAG_AUTO_OVERRIDE` | Log warning, ignore manual L4 value, continue. |
| Query-time resolution | `W_TAG_ZERO_MATCHES` | Fallback to unfiltered search. Include warning in response. |
| Query-time resolution | `W_TAG_ALL_DROPPED` | Fallback to unfiltered search. Include warning in response. |
| Query-time resolution | `W_TAG_AMBIGUOUS` | Apply broad filter (OR across layers). Include warning in response. |

### 15.3 Logging

All tag-related events use the existing logger names (`ingestkit_excel`, `ingestkit_pdf`). Tag operations are logged at the following levels:

| Event | Level | Example Message |
|-------|-------|----------------|
| Tag validation passed | `DEBUG` | `"Tag validation passed: 3 layers, 5 values"` |
| Tag validation warning | `WARNING` | `"Unknown tag layer 'division'; dropping"` |
| Tag validation error | `ERROR` | `"Required tag layer 'department' is missing"` |
| Auto-tag derivation | `DEBUG` | `"Auto-derived L4 tags: ['tabular', 'has-tables']"` |
| Query prefix parsed | `DEBUG` | `"Parsed tag prefix: department=hr, document_type=policy"` |
| Alias resolved | `DEBUG` | `"Alias resolved: 'people-ops' -> 'hr'"` |
| Ambiguous tag | `WARNING` | `"Ambiguous tag 'safety' found in layers: ['department', 'topic']"` |
| Zero-match fallback | `WARNING` | `"Tag filter matched 0 results; falling back to unfiltered search"` |
| Index created | `INFO` | `"Created payload index 'tag_department' (keyword) on collection 'helpdesk'"` |

No tag values are logged at levels above `DEBUG` unless they come from the controlled vocabulary (i.e., never log free-form user-supplied topic values at `INFO` or above, as they could contain PII).

---

## 16. Future Roadmap (V2/V3)

The following features are explicitly deferred. They are mentioned here for architectural awareness so that V1 does not preclude their implementation.

### V2 (Next Release)

- **Boost mode (`~` prefix):** `~HR:Policy What is PTO?` would boost (not filter) chunks tagged with `hr` and `policy`. Requires the retrieval layer to support score boosting in addition to hard filtering.
- **Negative tags (`!` prefix):** `!Archived: What is the leave policy?` would exclude chunks tagged with a specific value. Maps to Qdrant `must_not` filter conditions.
- **Hierarchical topics:** Support parent/child relationships in L3 topics. A query for `benefits` would also match sub-topics `health-plan`, `retirement`, `401k`. The `hierarchical=True` field on `TagLayer` is reserved for this.
- **Temporal tags:** Add a `status` layer with values `current` and `archived`. Default queries exclude `archived` unless explicitly requested. Requires a document lifecycle management protocol.
- **Auto-classify document type (L2):** Use the existing LLM classifier to automatically assign `document_type` based on document content. The `auto_tag_document_type` config flag is reserved for this.
- **Topic controlled vocabulary enforcement:** Ship a curated default vocabulary for L3 topics and enable strict enforcement.

### V3 (Future)

- **Role-based default tags:** Associate default tag filters with user roles. An HR employee's queries automatically include `department=hr` unless overridden.
- **Conversational tag stickiness:** Once a user specifies a tag in a conversation, subsequent queries in the same session inherit that tag unless explicitly cleared.
- **LLM hint inference:** Use the LLM to infer likely tag filters from natural language queries that do not use explicit prefix syntax. For example, "What is our maternity leave policy?" might infer `department=hr, topic=leave, document_type=policy`.
- **Multi-source cross-reference:** Support queries that span documents from multiple ingestkit packages (e.g., an Excel table and a PDF policy), joined by shared tags.

---

## 17. Open Questions

| # | Question | Impact | Notes |
|---|----------|--------|-------|
| OQ-1 | Should `TaggingConfig` be a field inside `ExcelProcessorConfig` / `PDFProcessorConfig`, or a separate top-level config? | Architecture | Separate config keeps packages decoupled but adds a second config file. Embedding it couples tagging to each package. Recommendation: separate config, shared across packages. |
| OQ-2 | Should tag resolution live in a shared `ingestkit-core` package or be duplicated in each package? | Code organization | V1 can duplicate (models are small). Extract to `ingestkit-core` when a third package is added. |
| OQ-3 | Should free-form L3 topics be validated against a warn-only vocabulary (accept any value but warn on unrecognized)? | UX | This would catch typos without blocking ingestion. Controlled by `strict_vocabulary` flag. |
| OQ-4 | How should the query-time resolution integrate with the retrieval layer? Is `resolve_query_tags()` called by the orchestration layer or by each package's router? | Integration | Recommendation: orchestration layer calls `resolve_query_tags()`, passes the resulting `TagFilter` to the retrieval backend. The ingestkit packages are ingestion-only; query-time resolution is a separate concern. |
| OQ-5 | Should the zero-match fallback be configurable (e.g., `fallback_on_zero_results=True`)? Some deployments may prefer to return zero results rather than unfiltered noise. | UX | Add as a config flag in V1 implementation. Default: `True` (fall back). |
| OQ-6 | For ambiguous tags that match both a controlled-vocabulary layer and a free-form layer, should the controlled-vocabulary match take priority? | Disambiguation | Current spec matches across all layers. An alternative is to prefer controlled-vocab layers. Requires user feedback to decide. |
| OQ-7 | Should `tag_field_prefix` be configurable or hardcoded to `tag_`? | Simplicity | Configurable adds flexibility but also a source of misconfiguration. Recommendation: configurable with a strong default. |

---

*End of specification.*
