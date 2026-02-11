# Plan Check: Issue #7 — Tier 2/3 LLM Classifier

**Issue:** #7
**Plan:** `/home/jjob/projects/ingestkit/.agents/outputs/plan-7-021026.md`
**MAP:** `/home/jjob/projects/ingestkit/.agents/outputs/map-7-021026.md`
**Spec:** `SPEC.md` §9 (Tier 2 & 3 — LLM Classifier)
**Date:** 2026-02-10
**Status:** **PASS** (with 10/10 validation checks)

---

## Validation Results

### 1. ENUM_VALUE Compliance (String Values, Not Python Names)

**Validation:** Checks for correct use of Literal string values in Pydantic schemas and enum conversions.

| Check | Expected | Found | Status |
|-------|----------|-------|--------|
| FileType enum values (models.py) | `"tabular_data"`, `"formatted_document"`, `"hybrid"` | Lines 31-33 ✓ | PASS |
| LLMClassificationResponse.type Literal | `Literal["tabular_data", "formatted_document", "hybrid"]` | Plan §1.2, line 61 ✓ | PASS |
| LLMClassificationResponse.sheet_types Literal | `Literal["tabular_data", "formatted_document"]` (no hybrid) | Plan §1.2, line 64 ✓ | PASS |
| FileType() constructor usage | `FileType(response.type)` not `FileType[response.type]` | Plan §1.4.6, line 454 ✓ | PASS |
| Prompt text enum values | Shows `"tabular_data"`, `"formatted_document"`, `"hybrid"` | Plan §1.4.4, lines 343-346 ✓ | PASS |
| ErrorCode enum members | `ErrorCode.E_LLM_TIMEOUT` not `"E_LLM_TIMEOUT"` string | Plan §2 section, lines 516-530 ✓ | PASS |
| ClassificationTier usage | `ClassificationTier.LLM_BASIC` / `LLM_REASONING` | Plan §1.4.2, lines 144-147 ✓ | PASS |

**Result:** ✅ PASS — All enum values use correct string representations.

---

### 2. LLMBackend Protocol Method Signatures

**Validation:** Verify `LLMBackend.classify()` return type and parameters match protocol definition.

| Aspect | Protocol (protocols.py) | Plan Expects | Match |
|--------|------------------------|--------------|-------|
| Method name | `classify` | `classify` | ✓ |
| Return type | `-> dict` | Backend returns `dict` (parsed JSON) | ✓ |
| Parameter: prompt | `prompt: str` | `prompt=prompt` | ✓ |
| Parameter: model | `model: str` | `model=model` | ✓ |
| Parameter: temperature | `temperature: float = 0.1` | `temperature=self._config.llm_temperature` | ✓ |
| Parameter: timeout | `timeout: float \| None = None` | `timeout=self._config.backend_timeout_seconds` | ✓ |

**Backend call in plan** (line 171-176):
```python
raw_dict = self._llm.classify(
    prompt=prompt,
    model=model,
    temperature=self._config.llm_temperature,
    timeout=self._config.backend_timeout_seconds,
)
```

**Result:** ✅ PASS — Protocol method signature matches exactly.

---

### 3. Error Codes Match ErrorCode Enum

**Validation:** All error codes used in plan exist in `errors.py` enum.

| Error Code (Plan) | ErrorCode Member | Found in errors.py | Status |
|------------------|-----------------|-------------------|--------|
| `E_LLM_TIMEOUT` | Line 28 | ✓ | PASS |
| `E_LLM_MALFORMED_JSON` | Line 29 | ✓ | PASS |
| `E_LLM_SCHEMA_INVALID` | Line 30 | ✓ | PASS |
| `E_LLM_CONFIDENCE_OOB` | Line 31 | ✓ | PASS |
| `E_CLASSIFY_INCONCLUSIVE` | Line 27 | ✓ | PASS |
| `W_LLM_RETRY` | Line 51 | ✓ | PASS |

**Result:** ✅ PASS — All error codes exist in enum.

---

### 4. Config Field Names Match ExcelProcessorConfig

**Validation:** All config field names used in plan match actual config.py definitions.

| Config Field (Plan) | Type Expected | Found in config.py | Status |
|-------------------|----------------|-------------------|--------|
| `classification_model` | `str` | Line 38, default `"qwen2.5:7b"` | ✓ |
| `reasoning_model` | `str` | Line 39, default `"deepseek-r1:14b"` | ✓ |
| `llm_temperature` | `float` | Line 41, default `0.1` | ✓ |
| `backend_timeout_seconds` | `float` | Line 61, default `30.0` | ✓ |
| `max_sample_rows` | `int` | Line 56, default `3` | ✓ |
| `log_sample_data` | `bool` | Line 66, default `False` | ✓ |
| `log_llm_prompts` | `bool` | Line 67, default `False` | ✓ |
| `redact_patterns` | `list[str]` | Line 69, default `[]` | ✓ |

**Result:** ✅ PASS — All config fields match exactly.

---

### 5. ClassificationResult Field Names

**Validation:** All fields referenced for ClassificationResult construction match models.py.

| Field (Plan) | Type Expected | Found in models.py | Status |
|-------------|----------------|-------------------|--------|
| `file_type` | `FileType` | Line 187 | ✓ |
| `confidence` | `float` | Line 188 | ✓ |
| `tier_used` | `ClassificationTier` | Line 189 | ✓ |
| `reasoning` | `str` | Line 190 | ✓ |
| `per_sheet_types` | `dict[str, FileType] \| None` | Line 191 | ✓ |
| `signals` | `dict[str, Any] \| None` | Line 192 | ✓ |

**Construction in plan** (line 463-469):
```python
return ClassificationResult(
    file_type=file_type,
    confidence=response.confidence,
    tier_used=tier,
    reasoning=response.reasoning,
    per_sheet_types=per_sheet_types,
    signals=None,
)
```

**Result:** ✅ PASS — All field names and types match.

---

### 6. Validation Pipeline Matches Spec §9.4 Sequence

**Validation:** Pipeline follows exact sequence from SPEC section 9.4.

**SPEC §9.4 Sequence:**
1. JSON parse → catch `JSONDecodeError` → `E_LLM_MALFORMED_JSON`, retry
2. Schema validate → catch `ValidationError` → `E_LLM_SCHEMA_INVALID`, retry
3. Confidence bounds check → out-of-bounds → `E_LLM_CONFIDENCE_OOB`, clamp, warn
4. After 2 failed attempts → fail with structured error

**Plan Implementation** (lines 155-239):
1. Retry loop with max 2 attempts ✓
2. JSON parse error handling (lines 177-187) ✓
3. Timeout handling (lines 188-195) ✓
4. Generic exception handling (lines 196-204) ✓
5. Response validation call (line 212) ✓
6. On validation failure, append correction hint and retry (lines 215-218) ✓
7. All attempts exhausted, fail-closed (lines 223-239) ✓

**Validation method** (`_validate_and_parse_response`, lines 367-423):
1. Pydantic schema validation → catch `ValidationError` ✓ (lines 389-398)
2. Manual confidence bounds check → clamp + warn ✓ (lines 401-411)
3. Sheet types key validation ✓ (lines 413-422)

**Result:** ✅ PASS — Pipeline matches spec sequence exactly.

---

### 7. Tier Escalation Logic Matches Spec §9.5

**Validation:** Tier model selection and parameter mapping match spec.

**Spec §9.5 States:**
- Tier 2: `config.classification_model` (qwen2.5:7b)
- Tier 3: `config.reasoning_model` (deepseek-r1:14b)
- Triggered by Tier 2 confidence < 0.6 (managed by router, not classifier)

**Plan Implementation** (lines 143-147):
```python
if tier == ClassificationTier.LLM_BASIC:
    model = self._config.classification_model
else:  # LLM_REASONING
    model = self._config.reasoning_model
```

**Key design note:** Plan §1.4 explicitly states:
> "The classifier does NOT do escalation internally -- that is the responsibility of the router. It classifies at the specific tier requested."

This is CORRECT per spec. The classifier receives a `tier` parameter and uses the appropriate model.

**Result:** ✅ PASS — Tier logic matches spec; escalation responsibility correctly placed with router.

---

### 8. Fail-Closed Behavior

**Validation:** Fail-closed returns correct values after all retries exhausted.

**Spec Requirement** (§9.4):
> After 2 failed attempts → fall back or fail with structured error. Never accept unvalidated LLM output.

**Plan Implementation** (lines 223-239):
```python
errors.append(IngestError(
    code=ErrorCode.E_CLASSIFY_INCONCLUSIVE,
    message="LLM classification failed after all retry attempts",
    stage="classify",
    recoverable=False,
))

return ClassificationResult(
    file_type=FileType.TABULAR_DATA,  # arbitrary; confidence=0.0 signals failure
    confidence=0.0,
    tier_used=tier,
    reasoning="LLM classification failed after exhausting retries. Fail-closed.",
    per_sheet_types=None,
    signals=None,
)
```

**Design:** Returns `confidence=0.0` with arbitrary `file_type`. Router must check confidence, not file_type, to determine usability. ✓

**Result:** ✅ PASS — Fail-closed behavior is correct and PII-safe.

---

### 9. Test Coverage Sufficient for All Acceptance Criteria

**Validation:** Test case list (plan §3.6) covers all critical paths.

| Acceptance Criterion | Test Class | Test Cases | Status |
|---------------------|-----------|-----------|--------|
| Valid responses parse correctly | `TestValidResponseParsing` | 7 tests | ✓ |
| Malformed JSON triggers retry | `TestMalformedJsonRetry` | 4 tests | ✓ |
| Schema validation enforced | `TestSchemaValidation` | 6 tests | ✓ |
| Confidence clamping (not rejection) | `TestConfidenceBounds` | 5 tests | ✓ |
| Timeout handling | `TestTimeoutHandling` | 2 tests | ✓ |
| Fail-closed after retries | `TestFailClosed` | 3 tests | ✓ |
| Tier 2/3 model selection | `TestTierModelSelection` | 6 tests | ✓ |
| Structural summary (PII-safe) | `TestStructuralSummary` | 11 tests | ✓ |
| Prompt content correctness | `TestPromptContent` | 5 tests | ✓ |
| Cell type inference | `TestCellTypeInference` | 7 tests | ✓ |
| LLMClassificationResponse validation | `TestLLMClassificationResponseModel` | 5 tests | ✓ |

**Total test cases:** 61 tests covering:
- Happy path (valid responses, tier selection)
- Retry logic (JSON errors, schema errors, timeouts)
- Validation (confidence bounds, schema enforcement)
- PII safety (structural summary without raw values by default)
- Error handling (fail-closed)

**Result:** ✅ PASS — Test coverage is comprehensive.

---

### 10. PII Safety: Structural Summary Doesn't Leak Raw Values by Default

**Validation:** Summary generation respects `log_sample_data` config flag.

**Default (`log_sample_data=False`)** — Plan §1.4.3, lines 303-307:
```python
else:
    # Structure-only: show types, never raw values
    lines.append("- Sample rows (structure only):")
    for i, row in enumerate(sheet.sample_rows[:max_rows]):
        types = [_infer_cell_type(cell) for cell in row]
        lines.append(f"  Row {i + 1}: [{', '.join(types)}]")
```

Shows only data TYPES (`str`, `float`, `int`, `empty`) — NO actual values. ✓

**Opt-in (`log_sample_data=True`)** — Plan §1.4.3, lines 296-301:
```python
if self._config.log_sample_data:
    # Include actual values (with redaction)
    lines.append("- Sample rows (values):")
    for i, row in enumerate(sheet.sample_rows[:max_rows]):
        redacted_row = [self._redact(cell) for cell in row]
        lines.append(f"  Row {i + 1}: [{', '.join(redacted_row)}]")
```

Includes actual values only when opted in, with redaction applied. ✓

**Header values** — Always included (structural, not PII) ✓

**Filename** — Extracted with `os.path.basename()`, not full path ✓

**Redaction helper** — Plan §1.4.7, lines 490-495:
```python
def _redact(self, text):
    result = text
    for pattern in self._config.redact_patterns:
        result = re.sub(pattern, "[REDACTED]", result)
    return result
```

**Test coverage** — TestStructuralSummary includes:
- `test_summary_contains_no_raw_values_by_default` ✓
- `test_summary_contains_values_when_log_sample_data_true` ✓
- `test_summary_redacts_when_log_sample_data_with_patterns` ✓

**Result:** ✅ PASS — PII safety is correctly implemented. Structural summary never leaks raw values by default.

---

## Summary

| Check | Result | Notes |
|-------|--------|-------|
| 1. ENUM_VALUE compliance | ✅ PASS | All string values correct; no Python names in Literals |
| 2. LLMBackend protocol match | ✅ PASS | `classify()` signature matches exactly |
| 3. Error codes exist | ✅ PASS | All 6 error codes found in `errors.py` |
| 4. Config fields exist | ✅ PASS | All 8 config fields match `config.py` |
| 5. ClassificationResult fields | ✅ PASS | All 6 fields match `models.py` |
| 6. Validation pipeline sequence | ✅ PASS | Matches spec §9.4 exactly: parse → validate → bounds check → fail |
| 7. Tier escalation logic | ✅ PASS | Tier model selection correct; escalation responsibility with router |
| 8. Fail-closed behavior | ✅ PASS | Returns `confidence=0.0`, `E_CLASSIFY_INCONCLUSIVE`, no unvalidated output |
| 9. Test coverage | ✅ PASS | 61 tests across 11 test classes; comprehensive happy path + error paths |
| 10. PII safety | ✅ PASS | Structure-only by default; header values always included; opt-in raw values with redaction |

---

## Approval

**Status:** ✅ **PASS**

The plan for Issue #7 (Tier 2/3 LLM Classifier with Schema Validation) is **APPROVED** for implementation.

**Key strengths:**
1. Comprehensive error handling with retry logic for three failure modes (JSON, schema, timeout)
2. Correct PII-safe defaults (structure-only summary, opt-in raw values)
3. Fail-closed design (zero confidence signals failure, not file_type)
4. Clear separation of concerns (classifier validates per-tier, router handles escalation)
5. Extensive test coverage (61 tests across all major paths)
6. Full enum value compliance (no Python names in string fields)

**No blockers identified.** Implementation can proceed immediately.
