# CLAUDE.md

This file provides project-specific guidance to Claude Code.

## Project Overview

- **Purpose:** Plugin-based Excel ingestion system for an on-premises RAG (AI Help Desk in a Box). Classifies Excel files by structure (tabular / formatted document / hybrid) and routes them through the appropriate processing path.
- **Users:** Internal platform — consumed by an orchestration layer, not end-users directly.
- **Non-goals:** No UI, no direct LLM chat, no cloud-only dependencies. Backends must remain swappable via protocols.

## Monorepo Layout

```
packages/
  ingestkit-excel/          # Only package so far
    src/ingestkit_excel/    # Source code
    tests/                  # pytest suite
    pyproject.toml
    SPEC.md                 # Authoritative spec (1,100+ lines) — read before making design decisions
    ROADMAP.md              # Deferred features — do NOT implement unless explicitly requested
```

## Development Commands

```bash
# Setup (from repo root)
pip install -e "packages/ingestkit-excel[dev]"

# Run tests (unit only, no external services)
pytest packages/ingestkit-excel/tests -m unit

# Run all tests
pytest packages/ingestkit-excel/tests

# Run a specific test file
pytest packages/ingestkit-excel/tests/test_parser_chain.py -v

# Run with coverage
pytest packages/ingestkit-excel/tests --cov=ingestkit_excel --cov-report=term-missing
```

## Architecture Constraints

### Pipeline Stages

```
Excel file → Parser Chain → Inspector (Tier 1) → LLM Classifier (Tier 2/3) → Processor (Path A/B/C)
```

1. **Parser chain** (`parser_chain.py`): openpyxl → pandas → openpyxl data_only. Per-sheet independent fallback.
2. **Tier 1 Inspector** (`inspector.py`): Rule-based, 5 binary signals, no LLM. Handles ~85% of files.
3. **Tier 2/3 LLM Classifier** (`llm_classifier.py`): Schema-validated JSON from LLM. Tier 3 escalation if confidence < 0.6.
4. **Processors**: Path A = `StructuredDBProcessor` (tabular→DB+embeddings), Path B = TextSerializer (not yet implemented), Path C = HybridSplitter (not yet implemented).

### Required Patterns

- **Backend-agnostic core:** All processing references `Protocol` types from `protocols.py`, never concrete backends. Dependency injection only.
- **Structural subtyping:** Backends use `typing.Protocol` with `@runtime_checkable`, not ABCs.
- **Pydantic v2 models:** All data models in `models.py`. Use `BaseModel`, not dataclasses.
- **Normalized error codes:** Use `ErrorCode` enum from `errors.py` (23 codes). Structured `IngestError` with code, message, sheet_name, stage, recoverable flag.
- **Fail-closed:** If classification is inconclusive, return `ProcessingResult` with `E_CLASSIFY_INCONCLUSIVE` and zero chunks. Never guess.
- **Idempotency:** `IngestKey = SHA256(content_hash | source_uri | parser_version | tenant_id)`. Deterministic, reproducible.
- **PII-safe logging:** Logger name `ingestkit_excel`. No raw cell data in logs unless `log_sample_data=True` in config.

### Forbidden Changes

- Do NOT add concrete backend implementations inside `ingestkit_excel/` — backends live outside this package.
- Do NOT bypass the parser fallback chain — every sheet must attempt openpyxl first.
- Do NOT introduce `ABC` base classes — this project uses structural subtyping via `Protocol`.
- Do NOT implement items from `ROADMAP.md` unless the issue explicitly requests it.

### Data / Security Constraints

- `tenant_id` must propagate through: Config → IngestKey → ChunkMetadata → all artifacts.
- PII redaction patterns are configurable via `config.redact_patterns`. Default: no redaction (caller's responsibility).
- Password-protected sheets emit `W_SHEET_SKIPPED_PASSWORD`, never attempt to crack.

## Key Configuration

`ExcelProcessorConfig` in `config.py` — 21 tunable parameters. Key defaults:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `classification_model` | `qwen2.5:7b` | Tier 2 LLM |
| `reasoning_model` | `deepseek-r1:14b` | Tier 3 LLM |
| `tier2_confidence_threshold` | `0.6` | Below this → escalate to Tier 3 |
| `row_serialization_limit` | `5000` | Path A skips row serialization above this |
| `embedding_model` | `nomic-embed-text` | For schema/chunk embeddings |
| `embedding_dimension` | `768` | Vector size |

Config supports loading from YAML/JSON via `ExcelProcessorConfig.from_file(path)`.

## Delivery Rules

- **Definition of done:** Implementation matches SPEC.md section for the feature. All new code has unit tests. No regressions in existing tests.
- **Required test coverage:** Every public function/method must have unit tests. Use mock backends (never require external services for unit tests). Mark integration tests with `@pytest.mark.integration`.
- **Test conventions:** Tests live in `packages/ingestkit-excel/tests/`. Fixtures in `conftest.py`. One test file per module (`test_<module>.py`).
- **Branch strategy:** Feature branches per issue (`feature/issue-N-description`), merged to `main` via PR.
- **Commit style:** `feat(ingestkit-excel): <description>` for features, `fix(ingestkit-excel): <description>` for fixes.

## File Types (Classification)

| Type | Enum | Example | Processing Path |
|------|------|---------|----------------|
| Tabular data | `FileType.TABULAR_DATA` | Employee roster, inventory | Path A (SQL Agent) |
| Formatted document | `FileType.FORMATTED_DOCUMENT` | Checklist, compliance matrix | Path B (Text Serialization) |
| Hybrid | `FileType.HYBRID` | Mixed tables and prose | Path C (Hybrid Split) |

## Dependencies

Core: `openpyxl>=3.1`, `pandas>=2.0`, `pydantic>=2.0`
Optional: `qdrant-client>=1.7`, `httpx>=0.27`, `psycopg2-binary>=2.9`
Dev: `pytest>=7.0`, `pytest-cov`, `pyyaml>=6.0`
Python: `>=3.10`
