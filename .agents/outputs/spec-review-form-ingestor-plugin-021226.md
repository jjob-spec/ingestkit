---
title: Spec Review - Form Ingestor Plugin (ingestkit-forms)
spec: docs/specs/form-ingestor-plugin.md
date: 2026-02-12
agent: SPEC-REVIEWER
---

# Spec Review: Form Ingestor Plugin (ingestkit-forms)

## Specification Summary
**Source**: `docs/specs/form-ingestor-plugin.md`
**Version**: 1.0 (v1.2 technology stack)
**Goals**: Template-driven form extraction for structured documents (fillable PDFs, scanned forms, Excel-based forms, photographed paper forms), producing both structured database rows and RAG-ready vector chunks. Integrates as Path F in the ingestkit pipeline, running before Inspector/Classifier.

## Requirements Extracted

### Backend — Data Models (§5.1, §10)
- Model: SourceFormat, FieldType enums
- Model: BoundingBox, CellAddress, FieldMapping (with dual-addressing validator)
- Model: FormTemplate (versioned, with fingerprint)
- Model: TemplateMatch, FormIngestRequest
- Model: ExtractedField, FormExtractionResult
- Model: FormProcessingResult, EmbedStageResult, WrittenArtifacts
- Model: FormChunkMetadata, FormChunkPayload, RollbackResult
- Model: FormTemplateCreateRequest, FormTemplateUpdateRequest, ExtractionPreview
- Model: DualWriteMode enum

### Backend — Error Taxonomy (§12)
- Enum: FormErrorCode (22 error codes + 14 warning codes = 36 total)
- Model: FormIngestError with diagnostic context fields

### Backend — Configuration (§11)
- Model: FormProcessorConfig with 40+ tunable parameters
- Cross-field model_validator
- from_file() classmethod for YAML/JSON

### Backend — Protocols (§15.3)
- Protocol: FormTemplateStore (6 methods)
- Protocol: OCRBackend (2 methods + OCRRegionResult model)
- Protocol: PDFWidgetBackend (3 methods + WidgetField model)
- Protocol: VLMBackend (3 methods + VLMFieldResult model)
- Reused: VectorStoreBackend, StructuredDBBackend, EmbeddingBackend (from ingestkit-core)

### Backend — Template System (§5, §9)
- Template CRUD: create, update, delete, get, list, list_versions
- Versioning: immutable snapshots, version increment
- Layout fingerprinting: grid-based structural fingerprint (§5.4)
- FileSystemTemplateStore: default JSON-based persistence

### Backend — Form Matching (§6)
- Auto-detection via fingerprint comparison
- Windowed multi-page alignment (§6.1)
- Confidence thresholds with review-band
- Manual override with format compatibility check

### Backend — Extractors (§7)
- NativePDFExtractor: IoU-based widget matching (§7.1)
- OCROverlayExtractor: per-field OCR with preprocessing pipeline (§7.2)
- ExcelCellExtractor: cell mapping with merged cell handling (§7.3)
- VLMFieldExtractor: optional third-tier fallback (§7.5)
- Per-field confidence scoring (§7.4)

### Backend — Output (§8)
- DB writer: table per template family, schema evolution
- Chunk writer: text serialization with splitting
- Dual-write consistency: best_effort and strict_atomic modes
- Rollback protocol
- Redaction targeting

### Backend — Router & API (§4.2, §9)
- FormRouter: orchestrates matching -> extraction -> output
- FormPluginAPI: 10 operations (CRUD, preview, match, extract)
- Source detection routing (§3.1)
- Pipeline integration as Path F

### Backend — Security (§13)
- Input validation: file size, type, magic bytes
- Template field count limit (200)
- ReDoS protection (1s regex timeout)
- Image security: decompression bomb, resolution limits
- PII-safe logging
- 7 mandatory security tests

### Backend — Idempotency (§4.3)
- Global ingest key (document-level)
- Form extraction key (template-versioned)
- UUID5 vector point IDs

### Backend — Prerequisites (§17)
- ingestkit-core extraction (shared primitives across packages)

## Codebase Analysis

### Implemented
- None (this is a new package)

### Partial
- None

### Missing
- **Everything.** The `ingestkit-forms` package does not exist. All 18 spec sections require implementation from scratch.

### Differs from Spec
- None (nothing exists to differ)

## Gap Summary
- Total gaps: 18 (all missing)
- Backend: 18
- Frontend: 0

## Spec Quality Issues

### Clarify During Implementation

1. **Pydantic v2 Config class (§5.1)**: The spec uses `class Config` with `json_encoders`, which is a Pydantic v1 pattern. Pydantic v2 uses `model_config = ConfigDict(...)` or `@field_serializer`. The implementer should use v2 patterns.

2. **FormTemplate.thumbnail storage (§5.1)**: The spec defines `thumbnail: bytes | None` on the template model and persists via JSON. Large binary fields in JSON are inefficient. The FileSystemTemplateStore implementation should consider storing thumbnails as separate files.

3. **ingestkit-core scope (§17 OQ-1)**: The spec says to extract shared primitives, but does not define the exact interface for `ingestkit-core`. The implementer will need to reconcile differences between `ingestkit-excel` and `ingestkit-pdf` protocol definitions.

4. **Backend error code overlap (§12.1)**: The `FormErrorCode` includes backend codes (`E_BACKEND_*`) that already exist in `ingestkit-excel`. After `ingestkit-core` extraction, these should be imported, not redefined.

5. **FieldMapping validator context (§5.2)**: The spec notes that the validator cannot know its parent template's `source_format`. The validation should be enforced at the `FormTemplate` level during creation, not on individual `FieldMapping` instances. This needs careful implementation.

## Issues Created

### Issue #55: [Spec] Extract ingestkit-core shared package from sibling packages (Backend)
- Complexity: COMPLEX
- Stack: backend, foundation
- Link: https://github.com/jjob-spec/ingestkit/issues/55

### Issue #56: [Spec] Scaffold ingestkit-forms package structure (Backend)
- Complexity: SIMPLE
- Stack: backend
- Link: https://github.com/jjob-spec/ingestkit/issues/56

### Issue #57: [Spec] Implement form data models -- Pydantic v2 (Backend)
- Complexity: COMPLEX
- Stack: backend
- Link: https://github.com/jjob-spec/ingestkit/issues/57

### Issue #58: [Spec] Implement form error taxonomy and structured error model (Backend)
- Complexity: SIMPLE
- Stack: backend
- Link: https://github.com/jjob-spec/ingestkit/issues/58

### Issue #59: [Spec] Implement FormProcessorConfig with validation (Backend)
- Complexity: SIMPLE
- Stack: backend
- Link: https://github.com/jjob-spec/ingestkit/issues/59

### Issue #60: [Spec] Implement form-specific protocols (Backend)
- Complexity: SIMPLE
- Stack: backend
- Link: https://github.com/jjob-spec/ingestkit/issues/60

### Issue #61: [Spec] Implement template system -- CRUD, versioning, and fingerprinting (Backend)
- Complexity: COMPLEX
- Stack: backend
- Link: https://github.com/jjob-spec/ingestkit/issues/61

### Issue #62: [Spec] Implement form matching with windowed multi-page alignment (Backend)
- Complexity: COMPLEX
- Stack: backend
- Link: https://github.com/jjob-spec/ingestkit/issues/62

### Issue #63: [Spec] Implement native PDF form field extraction (Backend)
- Complexity: COMPLEX
- Stack: backend
- Link: https://github.com/jjob-spec/ingestkit/issues/63

### Issue #64: [Spec] Implement OCR overlay extraction for scanned/image forms (Backend)
- Complexity: COMPLEX
- Stack: backend
- Link: https://github.com/jjob-spec/ingestkit/issues/64

### Issue #65: [Spec] Implement Excel cell mapping extraction (Backend)
- Complexity: SIMPLE
- Stack: backend
- Link: https://github.com/jjob-spec/ingestkit/issues/65

### Issue #66: [Spec] Implement VLM fallback extraction for low-confidence OCR fields (Backend)
- Complexity: COMPLEX
- Stack: backend
- Link: https://github.com/jjob-spec/ingestkit/issues/66

### Issue #67: [Spec] Implement per-field confidence scoring and overall aggregation (Backend)
- Complexity: SIMPLE
- Stack: backend
- Link: https://github.com/jjob-spec/ingestkit/issues/67

### Issue #68: [Spec] Implement dual-write output -- DB rows and RAG chunks (Backend)
- Complexity: COMPLEX
- Stack: backend
- Link: https://github.com/jjob-spec/ingestkit/issues/68

### Issue #69: [Spec] Implement FormRouter and plugin API surface (Backend)
- Complexity: COMPLEX
- Stack: backend
- Link: https://github.com/jjob-spec/ingestkit/issues/69

### Issue #70: [Spec] Implement security controls and input validation (Backend)
- Complexity: COMPLEX
- Stack: backend
- Link: https://github.com/jjob-spec/ingestkit/issues/70

### Issue #71: [Spec] Implement idempotency keying for form extraction (Backend)
- Complexity: SIMPLE
- Stack: backend
- Link: https://github.com/jjob-spec/ingestkit/issues/71

### Issue #72: [Spec] Implement pipeline integration as Path F (Backend)
- Complexity: SIMPLE
- Stack: backend
- Link: https://github.com/jjob-spec/ingestkit/issues/72

## Implementation Order

### Phase 0: Prerequisite
1. **#55** -- Extract ingestkit-core shared package (COMPLEX, foundation)

### Phase 1: Foundation (can be parallelized after scaffold)
2. **#56** -- Scaffold ingestkit-forms package structure (SIMPLE)
3. **#57** -- Data models (COMPLEX) -- depends on #56
4. **#58** -- Error taxonomy (SIMPLE) -- depends on #56
5. **#59** -- Configuration (SIMPLE) -- depends on #56, #58
6. **#60** -- Protocols (SIMPLE) -- depends on #56, #57
7. **#71** -- Idempotency keying (SIMPLE) -- depends on #56

### Phase 2: Core Systems
8. **#61** -- Template system with fingerprinting (COMPLEX) -- depends on #57, #59, #60
9. **#62** -- Form matching (COMPLEX) -- depends on #61
10. **#67** -- Confidence scoring (SIMPLE) -- depends on #57, #59
11. **#70** -- Security controls (COMPLEX) -- depends on #56, #58

### Phase 3: Extractors (can be parallelized)
12. **#63** -- Native PDF extraction (COMPLEX) -- depends on #57, #60
13. **#64** -- OCR overlay extraction (COMPLEX) -- depends on #57, #59, #60
14. **#65** -- Excel cell extraction (SIMPLE) -- depends on #57
15. **#66** -- VLM fallback extraction (COMPLEX) -- depends on #60, #64, #59

### Phase 4: Output
16. **#68** -- Dual-write output (COMPLEX) -- depends on #57, #59, #60, #58

### Phase 5: Integration
17. **#69** -- FormRouter and plugin API (COMPLEX) -- depends on all above
18. **#72** -- Pipeline integration as Path F (SIMPLE) -- depends on #69, #62

## Complexity Distribution
- TRIVIAL: 0
- SIMPLE: 8 issues (#56, #58, #59, #60, #65, #67, #71, #72)
- COMPLEX: 10 issues (#55, #57, #61, #62, #63, #64, #66, #68, #69, #70)

## Risks & Open Questions

### Risk: ingestkit-core extraction (#55) is a large refactor
The prerequisite to extract shared primitives into `ingestkit-core` touches both existing packages (`ingestkit-excel`, `ingestkit-pdf`) and could introduce regressions. This must be done carefully with all existing tests passing afterward.

### Risk: Pydantic v1/v2 compatibility
The spec uses `class Config` with `json_encoders` (Pydantic v1 pattern). The codebase targets Pydantic v2. Implementers must use `model_config = ConfigDict(...)` and `@field_serializer` instead.

### Risk: OCR engine availability
PaddleOCR has limited macOS ARM support. Tests must work without requiring actual OCR engines installed (use mock `OCRBackend`).

### Risk: VLM dependency on Ollama
The VLM fallback requires a running Ollama instance with Qwen2.5-VL loaded. All tests must mock the `VLMBackend` protocol. Integration tests requiring Ollama should be marked `@pytest.mark.integration`.

### Risk: Image processing memory
The spec caps images at 10000x10000 pixels and 512MB peak memory. The OCR overlay extractor must respect these limits to avoid memory exhaustion during processing.

### ENUM_VALUE Pattern
Multiple enums in the spec use lowercase string values: `SourceFormat.PDF = "pdf"`, `FieldType.TEXT = "text"`, `DualWriteMode.BEST_EFFORT = "best_effort"`. Implementers must use the string VALUES, not Python names.

---
AGENT_RETURN: spec-review-form-ingestor-plugin-021226.md
