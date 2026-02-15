---
issue: 60
agent: PLAN-CHECK
date: 2026-02-15
complexity: SIMPLE
stack: backend
---

# PLAN-CHECK: Issue #60 — Implement Form-Specific Protocols

**Plan artifact**: `.agents/outputs/map-plan-60-021526.md`
**Issue**: #60 — Form-specific protocols
**Date**: 2026-02-15

---

## Executive Summary

The plan is **APPROVED with one minor correction**. All four protocol signatures match the spec exactly. All three result models match. Protocols correctly use `@runtime_checkable` with `typing.Protocol` (no ABCs). Core protocol re-exports are correct. Tests cover `isinstance` checks with mock classes. Acceptance criteria are complete.

**One correction needed**: The spec (line 2142) says "two new protocols" but actually defines four. The plan correctly identifies all four -- this is a spec prose error, not a plan error.

---

## Check 1: Protocol Signatures vs. SPEC 15.3

### 1.1 FormTemplateStore

| Method | Spec (lines 2155-2196) | Plan (line 88) | Match |
|--------|------------------------|----------------|-------|
| `save_template(template: FormTemplate) -> None` | Yes | Yes | PASS |
| `get_template(template_id: str, version: int \| None = None) -> FormTemplate \| None` | Yes | Yes | PASS |
| `list_templates(tenant_id: str \| None = None, source_format: str \| None = None, active_only: bool = True) -> list[FormTemplate]` | Yes | Yes | PASS |
| `list_versions(template_id: str) -> list[FormTemplate]` | Yes | Yes | PASS |
| `delete_template(template_id: str, version: int \| None = None) -> None` | Yes | Yes | PASS |
| `get_all_fingerprints(tenant_id: str \| None = None, source_format: str \| None = None) -> list[tuple[str, str, int, bytes]]` | Yes | Yes | PASS |

PASS -- all 6 methods match exactly.

### 1.2 OCRBackend

| Method | Spec (lines 2206-2229) | Plan (line 89) | Match |
|--------|------------------------|----------------|-------|
| `ocr_region(image_bytes: bytes, language: str = "en", config: str \| None = None, timeout: float \| None = None) -> OCRRegionResult` | Yes | Yes | PASS |
| `engine_name() -> str` | Yes | Yes | PASS |

PASS -- both methods match exactly.

### 1.3 PDFWidgetBackend

| Method | Spec (lines 2255-2278) | Plan (line 90) | Match |
|--------|------------------------|----------------|-------|
| `extract_widgets(file_path: str, page: int) -> list[WidgetField]` | Yes | Yes | PASS |
| `has_form_fields(file_path: str) -> bool` | Yes | Yes | PASS |
| `engine_name() -> str` | Yes | Yes | PASS |

PASS -- all 3 methods match exactly.

### 1.4 VLMBackend

| Method | Spec (lines 2302-2331) | Plan (line 91) | Match |
|--------|------------------------|----------------|-------|
| `extract_field(image_bytes: bytes, field_type: str, field_name: str, extraction_hint: str \| None = None, timeout: float \| None = None) -> VLMFieldResult` | Yes | Yes | PASS |
| `model_name() -> str` | Yes | Yes | PASS |
| `is_available() -> bool` | Yes | Yes | PASS |

PASS -- all 3 methods match exactly.

---

## Check 2: Associated Data Models

### 2.1 OCRRegionResult (Spec lines 2232-2242)

| Field | Spec | Plan (line 97) | Match |
|-------|------|----------------|-------|
| `text: str` | Yes | Yes | PASS |
| `confidence: float = Field(ge=0.0, le=1.0)` | Yes | Yes | PASS |
| `char_confidences: list[float] \| None = Field(default=None)` | Yes | Yes | PASS |
| `engine: str` | Yes | Yes | PASS |

PASS.

### 2.2 WidgetField (Spec lines 2281-2288)

| Field | Spec | Plan (line 98) | Match |
|-------|------|----------------|-------|
| `field_name: str` | Yes | Yes | PASS |
| `field_value: str \| None` | Yes | Yes | PASS |
| `field_type: str` | Yes | Yes | PASS |
| `bbox: BoundingBox` | Yes | Yes | PASS |
| `page: int` | Yes | Yes | PASS |

PASS. Plan correctly notes `BoundingBox` is imported under `TYPE_CHECKING` from `ingestkit_forms.models` (issue #57 dependency).

### 2.3 VLMFieldResult (Spec lines 2335-2342)

| Field | Spec | Plan (line 99) | Match |
|-------|------|----------------|-------|
| `value: str \| bool \| None` | Yes | Yes | PASS |
| `confidence: float = Field(ge=0.0, le=1.0)` | Yes | Yes | PASS |
| `model: str` | Yes | Yes | PASS |
| `prompt_tokens: int \| None = None` | Yes | Yes | PASS |
| `completion_tokens: int \| None = None` | Yes | Yes | PASS |

PASS.

---

## Check 3: @runtime_checkable with typing.Protocol (No ABCs)

**Spec**: All four protocols are decorated with `@runtime_checkable` in the code blocks (lines 2147, 2199, 2247, 2293).

**Plan**: Lines 122-123 state: "`engine_name()`, `model_name()`, `is_available()` are regular methods (not `@property`) in Protocol definitions -- matching the spec code blocks exactly." The plan's code structure (lines 58-78) imports `Protocol, runtime_checkable` from `typing` and uses `@runtime_checkable` on all four protocols.

**ABC check**: No reference to `ABC` or `abc` module anywhere in the plan. Plan follows `typing.Protocol` structural subtyping per project rules.

PASS.

---

## Check 4: Core Protocol Re-exports

**Spec** (lines 2345-2349): Reuses `VectorStoreBackend`, `StructuredDBBackend`, `EmbeddingBackend` from ingestkit-core.

**Plan** (lines 74-78):
```python
from ingestkit_core.protocols import (
    EmbeddingBackend,
    StructuredDBBackend,
    VectorStoreBackend,
)
```

**Verified against actual `ingestkit_core/protocols.py`**: All three protocols exist at lines 19, 40, and 90 respectively. Import paths are correct.

**Plan also exports them in `__all__`** (lines 106-108). PASS.

**Note**: `LLMBackend` (line 65 of `ingestkit_core/protocols.py`) is NOT re-exported, which is correct -- the spec does not list it as a reused protocol for forms.

PASS.

---

## Check 5: Tests Verify isinstance Checks with Mock Classes

**Plan test cases** (lines 154-167):

| ID | Test | isinstance check | Status |
|----|------|-----------------|--------|
| T1 | `test_form_template_store_is_runtime_checkable` | Yes -- conforming mock class | PASS |
| T2 | `test_ocr_backend_is_runtime_checkable` | Yes -- conforming mock class | PASS |
| T3 | `test_pdf_widget_backend_is_runtime_checkable` | Yes -- conforming mock class | PASS |
| T4 | `test_vlm_backend_is_runtime_checkable` | Yes -- conforming mock class | PASS |
| T11 | `test_non_conforming_class_fails_isinstance` | Yes -- missing method fails check | PASS |

All four protocols have positive isinstance tests. T11 provides a negative test. This is a complete coverage pattern.

PASS.

---

## Check 6: Acceptance Criteria Completeness

**Plan acceptance criteria** (lines 172-183):

| # | Criterion | Spec Coverage | Status |
|---|-----------|---------------|--------|
| 1 | FormTemplateStore 6 methods match SPEC 15.3 | Lines 2148-2196 | PASS |
| 2 | OCRBackend with ocr_region + engine_name match | Lines 2199-2229 | PASS |
| 3 | PDFWidgetBackend with 3 methods match | Lines 2247-2278 | PASS |
| 4 | VLMBackend with 3 methods match | Lines 2293-2331 | PASS |
| 5 | 3 Pydantic models match field definitions | Lines 2232-2342 | PASS |
| 6 | All four protocols @runtime_checkable | Spec code blocks | PASS |
| 7 | No ABC base classes | Project rules | PASS |
| 8 | Core protocols re-exported | Lines 2345-2349 | PASS |
| 9 | All protocols importable from ingestkit_forms.protocols | Plan lines 127-143 | PASS |
| 10 | Unit tests pass | Plan lines 148-167 | PASS |
| 11 | No regressions | Standard | PASS |

**Missing from acceptance criteria**: None identified. The criteria cover all spec requirements, the protocol pattern constraints, the re-exports, the model definitions, and test verification.

PASS.

---

## Issues Found

### Minor: `char_confidences` Field Description

The spec (line 2238) includes a `description` parameter on the `char_confidences` field:
```python
char_confidences: list[float] | None = Field(
    default=None,
    description="Per-character confidence values, if available.",
)
```

The plan (line 97) summarizes this as `char_confidences: list[float] | None = Field(default=None)` without the description. This is cosmetic -- the PATCH implementer should include the description to match the spec code block exactly.

**Severity**: Low. Does not affect functionality.

---

## Summary

**Plan status**: APPROVED

| Check | Result |
|-------|--------|
| Protocol signatures match spec | PASS (all 4 protocols, all methods) |
| Data models included and correct | PASS (all 3 models, all fields) |
| @runtime_checkable + typing.Protocol, no ABCs | PASS |
| Core protocol re-exports correct | PASS |
| Tests verify isinstance with mocks | PASS (4 positive + 1 negative) |
| Acceptance criteria complete | PASS |

**Minor note for PATCH**: Include `description=` on `OCRRegionResult.char_confidences` per spec code block.

**Dependencies**: Plan correctly identifies #56 (scaffold) and #57 (models providing `BoundingBox` and `FormTemplate`) as prerequisites.

**Risk level**: LOW -- single-file protocol definition with well-defined spec, no existing code to break.

**Ready for PATCH phase**: YES

---

AGENT_RETURN: .agents/outputs/plan-check-60-021526.md
