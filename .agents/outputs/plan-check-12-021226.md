# PLAN-CHECK -- Issue #12: ExcelRouter Orchestrator and Public API

---
issue: 12
agent: PLAN-CHECK
date: 2026-02-12
plan_artifact: plan-12-021226.md
status: PASS_WITH_NOTES
---

## Executive Summary

Plan validated against actual codebase signatures. All requirement areas are covered. Three issues found: (1) `compute_ingest_key` signature in plan matches actual code, but plan accesses `.key` property correctly; (2) HybridSplitter constructor signature differs from what plan assumes -- plan passes `(self._structured_processor, self._text_serializer, self._config)` which matches actual `__init__(self, structured_processor: object, text_serializer: object, config)` exactly; (3) one minor scope note on `process_batch` -- plan self-corrects its error handling mid-document, final version is correct. Recommend proceeding to PATCH.

---

## 1. Requirement Coverage

| Requirement (from Issue #12) | Plan Section | Status |
|------------------------------|-------------|--------|
| Constructor accepts 4 backends + config | Section 1, lines 82-94 | PASS |
| `process()` 12-step flow | Section 2, full flow | PASS |
| Step 1: compute ingest_key | Section 2, Step 1 | PASS |
| Step 2: generate UUID4 run_id | Section 2, Step 1 (line 195) | PASS |
| Step 3: parse via ParserChain | Section 2, Step 2 | PASS |
| Step 4: classify via Inspector (Tier 1) | Section 2, Step 3 | PASS |
| Step 5: Tier 2 escalation if inconclusive | Section 2, Step 4 | PASS |
| Step 6: Tier 3 escalation if low confidence | Section 2, Step 4 | PASS |
| Step 7: Fail-closed on all tiers fail | Section 2, Step 5 | PASS |
| Step 8: Route to processor (3 paths) | Section 2, Step 6 | PASS |
| Step 9: Collect WrittenArtifacts | Via processor returns | PASS |
| Step 10: Assemble ProcessingResult | Section 2, Step 7 | PASS |
| Step 11: PII-safe log | Section 7 | PASS |
| Step 12: Return result | Section 2, Step 8 | PASS |
| `process_batch()` | Section 9 | PASS |
| `create_default_router()` | Section 8 | PASS |
| Public exports in `__init__.py` | Section 11 | PASS |
| PII-safe logging format | Section 7 | PASS |
| Tier escalation edge cases | Section 3 | PASS |
| Test plan | Section 10 | PASS |

**Coverage: 20/20 requirements mapped.**

---

## 2. Scope Containment

| File | Action | Allowed? |
|------|--------|----------|
| `src/ingestkit_excel/router.py` | CREATE | YES -- new module |
| `tests/test_router.py` | CREATE | YES -- new test file |
| `src/ingestkit_excel/__init__.py` | EDIT | YES -- add 2 exports |

**No out-of-scope files. PASS.**

---

## 3. Pattern Pre-checks (Signature Verification)

### 3a. `ParserChain.parse()` return type

**Actual** (parser_chain.py:58):
```python
def parse(self, file_path: str) -> tuple[FileProfile, list[IngestError]]
```

**Plan usage** (Section 2, Step 2):
```python
profile, parse_errors = self._parser.parse(file_path)
```

**Verdict: PASS** -- Destructure matches `tuple[FileProfile, list[IngestError]]`.

### 3b. `ExcelInspector.classify()` signature

**Actual** (inspector.py:46):
```python
def classify(self, profile: FileProfile) -> ClassificationResult
```

**Plan usage** (Section 2, Step 3):
```python
classification = self._inspector.classify(profile)
```

**Verdict: PASS** -- Single argument `FileProfile`, returns `ClassificationResult`.

### 3c. `LLMClassifier.classify()` signature

**Actual** (llm_classifier.py:100-104):
```python
def classify(
    self,
    profile: FileProfile,
    tier: ClassificationTier,
) -> ClassificationResult
```

**Plan usage** (Section 2, Step 4):
```python
classification = self._llm_classifier.classify(profile, ClassificationTier.LLM_BASIC)
classification = self._llm_classifier.classify(profile, ClassificationTier.LLM_REASONING)
```

**Verdict: PASS** -- Two positional args `(profile, tier)` match.

### 3d. `compute_ingest_key()` signature

**Actual** (ingestkit_core/idempotency.py:21-26):
```python
def compute_ingest_key(
    file_path: str,
    parser_version: str,
    tenant_id: str | None = None,
    source_uri: str | None = None,
) -> IngestKey
```

**Plan usage** (Section 2, Step 1):
```python
ingest_key_obj = compute_ingest_key(
    file_path=file_path,
    parser_version=config.parser_version,
    tenant_id=config.tenant_id,
    source_uri=source_uri,
)
ingest_key = ingest_key_obj.key
```

**Verdict: PASS** -- All 4 kwargs match actual signature. `.key` property exists on `IngestKey` (ingestkit_core/models.py:52-57, returns SHA-256 hex digest).

### 3e. Processor `process()` signatures

All three processors share identical signatures:

**Actual** (structured_db.py:105-114, serializer.py:86-95, splitter.py:88-97):
```python
def process(
    self,
    file_path: str,
    profile: FileProfile,
    ingest_key: str,
    ingest_run_id: str,
    parse_result: ParseStageResult,
    classification_result: ClassificationStageResult,
    classification: ClassificationResult,
) -> ProcessingResult
```

**Plan usage** (Section 2, Step 6):
```python
processor_args = dict(
    file_path=file_path,
    profile=profile,
    ingest_key=ingest_key,
    ingest_run_id=ingest_run_id,
    parse_result=parse_result,
    classification_result=classification_result,
    classification=classification,
)
result = self._structured_processor.process(**processor_args)
```

**Verdict: PASS** -- 7 kwargs match all three processor signatures exactly.

### 3f. Processor constructor signatures

**StructuredDBProcessor** (structured_db.py:89-99):
```python
def __init__(self, structured_db, vector_store, embedder, config)
```

**Plan** (Section 1, lines 102-104):
```python
self._structured_processor = StructuredDBProcessor(
    self._structured_db, self._vector_store, self._embedder, self._config
)
```
**Verdict: PASS**

**TextSerializer** (serializer.py:72-80):
```python
def __init__(self, vector_store, embedder, config)
```

**Plan** (Section 1, lines 105-107):
```python
self._text_serializer = TextSerializer(
    self._vector_store, self._embedder, self._config
)
```
**Verdict: PASS**

**HybridSplitter** (splitter.py:72-83):
```python
def __init__(self, structured_processor: object, text_serializer: object, config)
```

**Plan** (Section 1, lines 108-110):
```python
self._hybrid_splitter = HybridSplitter(
    self._structured_processor, self._text_serializer, self._config
)
```
**Verdict: PASS** -- Takes processor instances, not backends.

---

## 4. Wiring Check: `__init__.py` Exports

**Current exports** (line 42): Already exports `ParserChain`, `ExcelInspector`, `LLMClassifier`, all three processors, `ExcelProcessorConfig`, all models, all protocols, all backends.

**Plan adds** (Section 11):
```python
from ingestkit_excel.router import ExcelRouter, create_default_router
```
Plus `"ExcelRouter"` and `"create_default_router"` in `__all__`.

**Current `__init__.py` line 5-6 comment**: Already says "Higher-level components (ExcelRouter, create_default_router) will be exported here once implemented in subsequent issues."

**Verdict: PASS** -- Clean addition. No conflicts with existing exports. The placeholder comment should be removed when adding the actual imports.

---

## 5. Issues Found

### Issue 1: `process_batch()` self-correction in plan (LOW)

The plan presents three different versions of `process_batch()` in Section 9 (lines 630-692). The first version catches and re-raises, the second adds a try/except discussion, the third simplifies to a plain loop. The **final version** (lines 684-689) is correct: a simple sequential loop with no exception swallowing.

**Risk**: PATCH implementer might use an earlier version from the plan.
**Recommendation**: PATCH should use the FINAL version (plain loop, no try/except wrapping). Exceptions from `process()` propagate to caller.

### Issue 2: Tier escalation condition overlap (LOW)

The plan has two separate `if` blocks for Tier 2 -> Tier 3 escalation (lines 266-279 and 283-289). The first handles low confidence (0 < conf < threshold), the second handles Tier 2 complete failure (conf == 0.0). Both check `config.enable_tier3`.

After the first `if` block runs (Tier 2 low confidence -> Tier 3), the second `if` block's condition `classification.confidence == 0.0 and classification.tier_used == ClassificationTier.LLM_BASIC` would NOT match because Tier 3 would have updated `classification`. This is correct but could be cleaner -- a single escalation chain with `elif` would be more readable. No functional issue.

**Recommendation**: Consider refactoring to a single escalation chain in PATCH for clarity, but the plan's logic is functionally correct.

### Issue 3: `enable_tier3=False` with Tier 2 failure (INFO)

Plan Section 3 (lines 413-414) correctly states: "enable_tier3=False, Tier 2 confidence==0.0 -> fail-closed". The code at line 299 (`if classification.confidence == 0.0`) handles this correctly since Tier 3 won't run, the confidence stays 0.0, and the fail-closed check catches it.

**Verdict: Correct. No action needed.**

### Issue 4: `structured_db.py` logger uses `__name__` not `"ingestkit_excel"` (INFO)

The plan's `router.py` correctly uses `logging.getLogger("ingestkit_excel")`. However, `structured_db.py:35`, `serializer.py:34`, and `splitter.py:40` all use `logging.getLogger(__name__)` instead. This is not a plan issue but is worth noting -- the router's logging will be consistent with CLAUDE.md requirements even though the processors deviate.

**No plan change needed.**

---

## 6. Model Field Verification

**ProcessingResult fields used in fail-closed** (plan Section 6, verified against models.py:190-214):

| Field | Plan Value | Model Type | Valid? |
|-------|-----------|------------|--------|
| `file_path` | from arg | `str` | YES |
| `ingest_key` | SHA-256 hex | `str` | YES |
| `ingest_run_id` | UUID4 str | `str` | YES |
| `tenant_id` | from config | `str \| None` | YES |
| `parse_result` | built | `ParseStageResult` | YES |
| `classification_result` | built | `ClassificationStageResult` | YES |
| `embed_result` | `None` | `EmbedStageResult \| None` | YES |
| `classification` | built | `ClassificationResult` | YES |
| `ingestion_method` | `IngestionMethod.SQL_AGENT` | `IngestionMethod` | YES |
| `chunks_created` | `0` | `int` | YES |
| `tables_created` | `0` | `int` | YES |
| `tables` | `[]` | `list[str]` | YES |
| `written` | `WrittenArtifacts()` | `WrittenArtifacts` | YES |
| `errors` | list of strings | `list[str]` | YES |
| `warnings` | list of strings | `list[str]` | YES |
| `error_details` | list of IngestError | `list[IngestError]` | YES |
| `processing_time_seconds` | float | `float` | YES |

**All 17 fields verified. PASS.**

---

## 7. Verdict

**PASS WITH NOTES**

The plan is comprehensive, all signatures verified against actual code, scope is contained, and all 20 requirements are covered. Three minor notes for PATCH awareness (process_batch final version, escalation chain clarity, logger name convention) -- none are blockers.

**Proceed to PATCH.**

AGENT_RETURN: plan-check-12-021226.md
