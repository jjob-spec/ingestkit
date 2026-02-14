---
issue: 32
agent: PLAN-CHECK
date: 2026-02-14
status: PASS
---

# PLAN-CHECK: Issue #32 — Configurable Text Chunking (`utils/chunker.py`)

## Executive Summary

The MAP-PLAN for issue #32 is well-structured and accurately reflects the spec requirements from SPEC.md sections 15.1 through 15.4. Config parameter names, model field names, and the public interface all match the actual codebase. The plan creates exactly 2 files, appropriate for SIMPLE complexity. No scope creep detected. Test coverage is thorough with 41 test cases across 10 test classes covering all public methods and edge cases.

---

## Validation Checklist

### Requirement Coverage

- [x] Split hierarchy matches spec 15.1: `\n## ` -> `\n### ` -> `\n\n` -> `. ` -> ` `
- [x] Target chunk size uses `config.chunk_size_tokens` (default 512) -- verified config.py line 91
- [x] Overlap uses `config.chunk_overlap_tokens` (default 50) -- verified config.py line 92
- [x] Heading boundaries respected when `config.chunk_respect_headings=True` -- verified config.py line 93
- [x] Table-aware chunking: never split mid-table when `config.chunk_respect_tables=True` -- verified config.py line 94
- [x] Oversized tables kept as single chunk (spec 15.2)
- [x] Metadata attachment matches spec 15.3: `page_numbers`, `heading_path`, `content_type`, `chunk_index`, `chunk_hash`
- [x] Public interface matches spec 15.4: `PDFChunker.__init__(config)`, `chunk(text, headings, page_boundaries) -> list[dict]`
- [x] `chunk_hash` is SHA-256 of chunk text

### Config Parameter Accuracy

- [x] `chunk_size_tokens: int = 512` -- matches config.py line 91
- [x] `chunk_overlap_tokens: int = 50` -- matches config.py line 92
- [x] `chunk_respect_headings: bool = True` -- matches config.py line 93
- [x] `chunk_respect_tables: bool = True` -- matches config.py line 94
- [x] `log_chunk_previews: bool = False` -- matches config.py line 119
- [x] Plan correctly states "No config changes needed"

### Model Field Accuracy

- [x] `ContentType` enum exists at models.py line 84 with values: `NARRATIVE`, `TABLE`, `LIST`, `HEADING`, `FORM_FIELD`, `IMAGE_DESCRIPTION`, `FOOTER`, `HEADER`
- [x] `PDFChunkMetadata` exists at models.py line 230 with `page_numbers`, `heading_path`, `content_type` fields
- [x] Plan correctly notes chunker returns raw dicts (callers wrap in `PDFChunkMetadata`)

### Scope and Complexity

- [x] File count (2 files: 1 source + 1 test) matches SIMPLE complexity
- [x] No modifications to existing files -- clean addition only
- [x] No scope creep: plan stays within chunker utility, does not touch router, processors, or backends
- [x] No ROADMAP.md items implemented
- [x] No concrete backend implementations added inside the package
- [x] `utils/__init__.py` already exists -- no need to create it

### Token Estimation Approach

- [x] Uses `len(text) // 4` heuristic -- reasonable for a system that does not mandate a specific tokenizer
- [x] Spec does not require tiktoken or sentencepiece
- [x] Minimum of 1 token for empty/short strings prevents division-by-zero edge cases

### Test Plan Coverage

- [x] All public methods covered: `PDFChunker.__init__`, `PDFChunker.chunk`
- [x] All helper functions covered: `_estimate_tokens`, `_compute_chunk_hash`, `_detect_content_type`, `_get_heading_path`, `_get_page_numbers`
- [x] Edge cases: empty text, whitespace-only, oversized tables, zero overlap
- [x] Config integration: custom sizes, large sizes, default config
- [x] Heading respect: enabled vs disabled
- [x] Table awareness: enabled vs disabled, mixed content
- [x] Page boundary mapping: single page, spanning pages, empty boundaries
- [x] Overlap behavior: applied, first chunk exempt, zero overlap
- [x] 41 test cases across 10 test classes -- comprehensive coverage
- [x] No external service dependencies in tests

### Architecture Compliance

- [x] No ABC base classes (uses no inheritance)
- [x] Logger name would be `ingestkit_pdf` if logging is added
- [x] PII-safe: no raw text in logs unless `config.log_chunk_previews=True`
- [x] Pure utility with no backend dependencies (only imports `PDFProcessorConfig`)
- [x] `ErrorCode.E_PROCESS_CHUNK` exists in errors.py for chunk processing errors

---

## Issues Found

None.

---

## Minor Observations (Non-Blocking)

1. The spec 15.4 docstring mentions `{text, page_numbers, heading_path, chunk_index}` but omits `chunk_hash` and `content_type`. The plan correctly includes all six fields from spec 15.3 -- this is the right interpretation.

2. The plan's `_detect_content_type` returns string values (`"narrative"`, `"table"`, etc.) rather than `ContentType` enum members. This is acceptable since the chunker returns raw dicts, but the implementer should ensure the string values match the `ContentType` enum values exactly (they do: `ContentType.NARRATIVE.value == "narrative"`, etc.).

---

## Recommendation

**APPROVED** -- The MAP-PLAN is accurate, complete, and ready for PATCH implementation.

AGENT_RETURN: .agents/outputs/plan-check-32-021426.md
