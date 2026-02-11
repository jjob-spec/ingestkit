---
title: Spec Review - ingestkit-excel
spec: packages/ingestkit-excel/SPEC.md
date: 2026-02-10
agent: SPEC-REVIEWER
---

# Spec Review: ingestkit-excel

## Specification Summary
**Source**: `packages/ingestkit-excel/SPEC.md` v2.0 (commit 3f8bf05)
**Goals**: Tiered classification and processing of Excel files for an on-premises RAG system. Classifies .xlsx files into tabular (Type A), document-formatted (Type B), or hybrid (Type C), then routes to appropriate ingestion path.

## Requirements Extracted

### Backend
- **Foundation**: 5 enums, 12+ Pydantic models, 4 Protocol interfaces, config class, error taxonomy (20+ codes)
- **Idempotency**: `compute_ingest_key()` — SHA-256 deterministic key from content_hash + source_uri + parser_version
- **Parser Chain**: 3-stage fallback (openpyxl → pandas → raw text) with per-sheet reason codes
- **Tier 1 Inspector**: 5 binary signals, threshold-based classification, multi-sheet hybrid detection
- **Tier 2/3 LLM Classifier**: Structural summary generation, Pydantic-validated LLM output, retry with correction, tier escalation, fail-closed
- **Path A Processor**: DataFrame → structured DB + schema description embedding
- **Path B Serializer**: Merged cell parsing → section detection → NL serialization (4 sub-types)
- **Path C Splitter**: Region detection (5 heuristics) → per-region classification → routing to A/B
- **Backends**: Qdrant, SQLite, Ollama (concrete) + Milvus, PostgreSQL (stubs)
- **Router**: Orchestrates all tiers + paths, PII-safe logging, public API
- **Tests**: Mock backends, .xlsx fixtures, unit + integration markers

### Frontend
- None (backend-only package)

## Codebase Analysis

### ✅ Implemented
- (none)

### 🟡 Partial
- (none)

### ❌ Missing
- `pyproject.toml` — package does not exist yet
- `src/ingestkit_excel/models.py` — all data models
- `src/ingestkit_excel/errors.py` — error taxonomy
- `src/ingestkit_excel/config.py` — configuration
- `src/ingestkit_excel/protocols.py` — backend Protocols
- `src/ingestkit_excel/idempotency.py` — ingest key computation
- `src/ingestkit_excel/parser_chain.py` — fallback parser chain
- `src/ingestkit_excel/inspector.py` — Tier 1 rule-based classifier
- `src/ingestkit_excel/llm_classifier.py` — Tier 2/3 LLM classifier
- `src/ingestkit_excel/processors/structured_db.py` — Path A
- `src/ingestkit_excel/processors/serializer.py` — Path B
- `src/ingestkit_excel/processors/splitter.py` — Path C
- `src/ingestkit_excel/backends/qdrant.py` — Qdrant vector store
- `src/ingestkit_excel/backends/sqlite.py` — SQLite structured DB
- `src/ingestkit_excel/backends/ollama.py` — Ollama LLM + embedding
- `src/ingestkit_excel/backends/milvus.py` — stub
- `src/ingestkit_excel/backends/postgres.py` — stub
- `src/ingestkit_excel/router.py` — orchestrator
- `src/ingestkit_excel/__init__.py` — public API exports
- `tests/` — all test infrastructure

### ⚠️ Differs from Spec
- (none — greenfield project)

## Gap Summary
- Total gaps: 11 issues (covering ~20 files)
- Backend: 11
- Frontend: 0

## Spec Quality Issues

### 🟡 Clarify During Implementation
- §10.2 section detection heuristics (merged header rows, indentation patterns) are described in prose, not as concrete algorithms. Implementer will need to make judgment calls on thresholds.
- §10.3 "formatting transitions" heuristic for region detection is vague — shift from numeric-heavy to text-heavy needs a concrete threshold.
- §9.2 structural summary generation says "never send raw data values by default" but the LLM may need sample values to classify ambiguous files. Trade-off between PII safety and classification accuracy will surface during testing.

## Issues Created

| Issue | Title | Complexity | Labels |
|-------|-------|-----------|--------|
| #3 | Scaffold project + foundation models, protocols, config, errors | SIMPLE | from-spec, backend, foundation |
| #4 | Implement idempotency key computation | TRIVIAL | from-spec, backend |
| #5 | Implement parser fallback chain | SIMPLE | from-spec, backend |
| #6 | Implement Tier 1 rule-based inspector | SIMPLE | from-spec, backend |
| #7 | Implement Tier 2/3 LLM classifier with schema validation | COMPLEX | from-spec, backend |
| #8 | Implement Path A structured DB processor | COMPLEX | from-spec, backend |
| #9 | Implement Path B text serializer | COMPLEX | from-spec, backend |
| #10 | Implement Path C hybrid splitter with enhanced region detection | COMPLEX | from-spec, backend |
| #11 | Implement concrete backends: Qdrant, SQLite, Ollama + stubs | SIMPLE | from-spec, backend |
| #12 | Implement ExcelRouter orchestrator and public API | COMPLEX | from-spec, backend |
| #13 | Implement test infrastructure, mock backends, and fixtures | SIMPLE | from-spec, backend |

## Implementation Order

```
Phase 1 (Foundation — no dependencies):
  #3  Foundation models, protocols, config, errors
  #13 Test infrastructure (mock backends, fixtures)

Phase 2 (Core modules — depend on #3):
  #4  Idempotency
  #5  Parser fallback chain
  #6  Tier 1 inspector
  #7  LLM classifier
  #11 Concrete backends

Phase 3 (Processors — depend on #3):
  #8  Path A processor
  #9  Path B serializer

Phase 4 (Hybrid — depends on #8, #9):
  #10 Path C splitter

Phase 5 (Integration — depends on everything):
  #12 Router + public API
```

**Parallelizable**: Within each phase, issues can be worked in parallel.

## Dependency Graph

```
#3 (Foundation) ──┬──► #4  (Idempotency)
                  ├──► #5  (Parser chain)
                  ├──► #6  (Inspector)
                  ├──► #7  (LLM classifier)
                  ├──► #8  (Path A)
                  ├──► #9  (Path B)
                  ├──► #11 (Backends)
                  └──► #13 (Test infra)

#8 (Path A) ──────┐
#9 (Path B) ──────┤
                  └──► #10 (Path C splitter)

#4, #5, #6, #7 ──┐
#8, #9, #10 ─────┤
#11 ─────────────┤
                 └──► #12 (Router — final integration)
```

## Risks & Open Questions

### Risk Flags
- **ENUM_VALUE**: 5 enums defined with explicit string values. LLM classification prompt in §9.3 must use VALUES (`"tabular_data"`) not Python names (`"TABULAR_DATA"`). Flagged in #7.
- **VERIFICATION_GAP**: LLM output is untrusted input. Pydantic schema validation is mandatory per §9.4. Flagged in #7.

### Open Questions
1. **Section detection thresholds (Path B)**: How many blank rows constitute a section break? Spec says "blank row separators" but doesn't specify minimum. Suggest: configurable, default 1.
2. **LLM summary fidelity vs PII**: Sending type-only summaries (no values) to LLM may reduce classification accuracy for ambiguous files. May need A/B testing post-v1.
3. **Parser chain fallback quality**: pandas fallback loses merged cell info, which Path B needs. If a Type B file falls back to pandas, serialization quality degrades. Consider: warn prominently when Type B uses pandas fallback.

---
AGENT_RETURN: spec-review-ingestkit-excel-021026.md
