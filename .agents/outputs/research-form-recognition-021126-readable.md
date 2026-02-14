# Form Recognition Research (Readable Version)

## TL;DR
If your constraints are **on-prem**, **plugin architecture**, and **no cloud lock-in**, the best practical path is:
1. Template-first extraction for deterministic forms.
2. OCR/KIE fallback for messy scans.
3. Optional local VLM fallback for hard edge cases.

## What Matters for ingestkit
- Deterministic extraction for known forms.
- Works across PDF + scanned/image + Excel.
- Clear metadata and confidence scoring.
- Self-hosted only.
- Compatible licenses.

## Best Tool Shortlist

### 1) Template-Driven / Closest Match
- **TWIX**: Best conceptual match for template inference from repeated form layouts.
  - Good: template inference pipeline, no cloud requirement.
  - Risk: research maturity and smaller ecosystem.
- **Parsee**: typed extraction patterns and template-like controls.
  - Good: structured outputs.
  - Risk: smaller project / less proven for production at scale.

### 2) OCR + KIE Engines (Production Workhorses)
- **PaddleOCR (+ PP-Structure / KIE)**
  - Good: mature, Apache-2.0, strong document parsing stack.
  - Risk: tuning complexity.
- **docTR**
  - Good: strong OCR library and clean APIs.
  - Risk: less template-native; mostly OCR building block.

### 3) PDF/Form Building Blocks
- **PyMuPDF**
  - Good: best-in-class for native fillable PDF widget extraction.
  - Risk: AGPL/commercial licensing implications.
- **pdfplumber**
  - Good: excellent coordinate-based cropping/region extraction.
  - Risk: text-layer PDFs only unless paired with OCR.
- **openpyxl**
  - Good: direct deterministic extraction for Excel forms.

### 4) Optional VLM Fallback (Hard Cases)
- **Qwen2.5-VL (local)**
  - Good: local structured extraction fallback, strong capability.
  - Risk: GPU cost and non-deterministic behavior vs templates.

## Recommended Architecture Pattern

```text
Form match -> template extraction -> validation -> dual write (DB + chunks)
                     |
                     +-> OCR/KIE fallback when native extraction is weak
                     |
                     +-> optional VLM fallback for low-confidence edge cases
```

## Architecture Patterns to Reuse
- **Region-based extraction**: template box -> crop -> OCR per field.
- **Per-field confidence**: keep score per field, not only document-level.
- **Fail-closed extraction**: low confidence => `None` + warning, never hallucinate.
- **Separable pipeline stages**: matcher, extractor, validator, writer.
- **Schema-first outputs**: strict typed models for DB/chunk output.

## License and Risk Notes
- **Prefer**: MIT / Apache-2.0 components for clean redistribution.
- **Watch**: AGPL/GPL components (PyMuPDF AGPL, Marker GPL, MinerU AGPL, etc.) depending on your deployment and distribution model.
- **Model licenses vary**: verify weights and usage terms before production.

## Suggested Stack for v1 (Pragmatic)
- Matching + template storage: custom plugin logic.
- Fillable PDFs: `PyMuPDF` widget extraction (or equivalent licensing-safe path).
- Scanned/images: `PaddleOCR` region OCR.
- Excel: `openpyxl` cell mapping.
- Structured outputs: strict Pydantic models + confidence/warnings.
- Fallback order: native -> OCR -> (optional) local VLM.

## Why this version is easier to use
- Removes the transcript noise.
- Focuses on tool choices and tradeoffs.
- Keeps only decisions relevant to ingestkit implementation.

## Original Extracted Report
- Full extracted report from the raw transcript:
  - `.agents/outputs/research-form-recognition-021126-extracted.md`
